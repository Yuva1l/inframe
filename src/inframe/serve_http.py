"""HTTP plumbing for `inframe serve` (docs/T9A_DESIGN.md sections 2.3 and 4): routes, requests, responses, strict body
reading and Range-aware file responses. Imports nothing from the other serve_* modules; route modules import it.

Handlers are `handler(app, req) -> Response`. A handler refuses with `raise HttpError(code, message, **extra)` or returns
`error(code, message, **extra)`; the body is always `{"error": message, ...extra}`.
"""
from __future__ import annotations

import errno
import json
import os
import re
import socket
import threading
from dataclasses import dataclass, field
from typing import Callable

from .pick import CLIENT_GONE, QuietServer

MAX_BODY = 1024 * 1024  # bytes; above this a request is 413 before anything is read
SOCKET_TIMEOUT = 15.0  # seconds a handler socket may stay silent (a slow body ends with 408)
MAX_THREADS = 32  # handler threads at once; more connections get 503
CHUNK = 1024 * 1024

__all__ = ["CLIENT_GONE", "HttpError", "Request", "Response", "Route", "QuietServer", "BoundedServer", "bytes_response", "error",
           "file_response", "json_response", "read_body", "parse_range"]


class HttpError(Exception):
    """Raised by a handler (or a check) to answer with this status and `{"error": message, **extra}`."""

    def __init__(self, code: int, message: str, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


@dataclass
class Request:
    method: str
    path: str
    params: dict[str, str]  # the named groups of the route regex
    query: dict[str, str]  # first value of each query parameter
    headers: dict[str, str]  # lower-cased names
    json: object = None  # the parsed body (an object) for POST/PUT/DELETE; None for GET/HEAD
    body: bytes = b""


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)
    file: tuple[str, int, int] | None = None  # (path, offset, length): the body is copied from the file
    after: Callable[[], None] | None = None  # called once the response is written (e.g. stop the server after /api/quit)


@dataclass
class Route:
    method: str
    regex: str | re.Pattern
    handler: Callable
    media: bool = False  # GET/HEAD route that also accepts the read-only media key `?m=`

    def __post_init__(self):
        self.pattern = re.compile(self.regex) if isinstance(self.regex, str) else self.regex
        self.method = self.method.upper()

    def match(self, path: str) -> dict[str, str] | None:
        m = self.pattern.fullmatch(path)
        return m.groupdict() if m else None


def json_response(obj, code: int = 200) -> Response:
    return Response(code, json.dumps(obj, separators=(",", ":"), allow_nan=False).encode(), "application/json")


def error(code: int, message: str, **extra) -> Response:
    return json_response({"error": message, **extra}, code)


def bytes_response(data: bytes, ctype: str, code: int = 200) -> Response:
    return Response(code, data, ctype)


# ---------------------------------------------------------------- request bodies
_LENGTH = re.compile(r"[0-9]{1,8}")


def read_body(handler, limit: int = MAX_BODY) -> bytes:
    """The request body, read only after the headers pass: no Transfer-Encoding (400), Content-Length of 1 to 8 digits
    (else 400; missing on POST/PUT is 411), over `limit` is 413 without reading, then exactly that many bytes. A silent
    client ends with 408 (the handler socket has a timeout)."""
    h = handler.headers
    if h.get("Transfer-Encoding") is not None:
        raise HttpError(400, "chunked bodies are not accepted")
    raw = h.get("Content-Length")
    if raw is None:
        if handler.command in ("POST", "PUT"):
            raise HttpError(411, "Content-Length is required")
        return b""
    if not _LENGTH.fullmatch(raw):
        raise HttpError(400, "bad Content-Length")
    n = int(raw)
    if n > limit:
        raise HttpError(413, f"body over {limit} bytes")
    buf = bytearray()
    try:
        while len(buf) < n:
            chunk = handler.rfile.read(min(CHUNK, n - len(buf)))
            if not chunk:
                raise HttpError(400, "body ended early")
            buf += chunk
    except (TimeoutError, socket.timeout):
        raise HttpError(408, "request body timed out") from None
    return bytes(buf)


# ---------------------------------------------------------------- Range
def parse_range(header: str | None, size: int):
    """`None` = send the whole file (no Range, another unit, several ranges, a malformed spec); `(start, end)` inclusive for one
    satisfiable range; `"unsatisfiable"` for a range outside the file."""
    if not header:
        return None
    m = re.fullmatch(r"\s*bytes\s*=\s*([0-9]*)\s*-\s*([0-9]*)\s*", header)
    if not m:
        return None  # another unit, several ranges ("a-b,c-d"), junk
    a, b = m.groups()
    if a == "" and b == "":
        return None
    if a == "":  # suffix: the last n bytes
        n = int(b)
        if n == 0 or size == 0:
            return "unsatisfiable"
        return max(0, size - n), size - 1
    start = int(a)
    end = int(b) if b != "" else size - 1
    if b != "" and end < start:
        return None  # a malformed range is ignored (RFC 9110)
    if start >= size:
        return "unsatisfiable"
    return start, min(end, size - 1)


def file_response(path, ctype: str, req: Request) -> Response:
    """The file at `path` (already checked by the caller), Range-aware: 200, 206 or 416. HEAD gets the headers only."""
    path = str(path)
    try:
        st = os.stat(path)
    except OSError:
        raise HttpError(404, "no such file") from None
    size = st.st_size
    r = parse_range(req.headers.get("range"), size)
    hdr = {"Accept-Ranges": "bytes"}
    if r == "unsatisfiable":
        hdr["Content-Range"] = f"bytes */{size}"
        return Response(416, json.dumps({"error": "range not satisfiable"}).encode(), "application/json", hdr)
    if r is None:
        return Response(200, b"", ctype, hdr, file=(path, 0, size))
    start, end = r
    hdr["Content-Range"] = f"bytes {start}-{end}/{size}"
    return Response(206, b"", ctype, hdr, file=(path, start, end - start + 1))


# macOS sendfile reports a client that went away as ENOTCONN (or EPROTOTYPE), not as one of the CLIENT_GONE classes
GONE_ERRNOS = {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.ENOTCONN, errno.ESHUTDOWN, getattr(errno, "EPROTOTYPE", errno.EPIPE)}


def copy_file(sock, f, offset: int, length: int) -> None:
    """Send `length` bytes of the open file `f` from `offset` (socket.sendfile: os.sendfile where available, else chunks).
    A client that goes away mid-file raises ConnectionAbortedError (one of CLIENT_GONE), whatever errno the OS gave."""
    if length > 0:
        try:
            sock.sendfile(f, offset, length)
        except CLIENT_GONE:
            raise
        except OSError as e:
            if e.errno in GONE_ERRNOS:
                raise ConnectionAbortedError(e.errno, e.strerror) from e
            raise


# ---------------------------------------------------------------- the server
class BoundedServer(QuietServer):
    """QuietServer (a client that goes away is not an error) with at most MAX_THREADS handler threads: over that a
    connection gets 503 and is closed."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, *a, max_threads: int | None = None, **kw):
        self._slots = threading.BoundedSemaphore(max_threads or MAX_THREADS)
        super().__init__(*a, **kw)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(2)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()
