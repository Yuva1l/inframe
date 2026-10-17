"""Two static hosts on 127.0.0.1 with different ports, so a browser sees two origins (docs/T9C_DESIGN.md section 3). Nothing here leaves the machine.

PageHost  serves a built `pages-dist` folder and applies its `_headers` file with the rules Cloudflare Pages documents (a `*` splat in the pattern,
          every matching rule adds its headers), so the CSP Chrome enforces is the one the build generated. It answers 404 for a path that is not a file
          (Pages would answer index.html); the privacy check then reports it.
ModelHost serves the model files like Hugging Face does, on two origins: `GET /resolve/<commit>/<name>` on the model host answers 302 to a signed
          `/cdn/<name>?Expires=..&Signature=..` on a THIRD local origin (the cdn host, like cas-bridge.xethub.hf.co: the page's request is then cross-origin
          from the model host, so Chrome sends `Origin: null` on the second hop), both hops carry Access-Control-Allow-Origin, `/cdn/` answers HEAD and a Range
          request with 206. Anything else it does not know (OPTIONS, POST, PUT, DELETE) is answered 405 after it has been logged, so the log shows what
          the page tried. Every Location the model host issued is kept in its log entry.

Every request that reaches either host is logged as an Entry (method, path, query, the headers as received, the body as received, the status given).
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

CHUNK = 1 << 20
BODY_CAP = 8 << 20  # a request body is read up to this much so the check can search it


@dataclass
class Entry:
    host: str  # "page" or "model"
    method: str
    path: str
    query: str
    headers: dict[str, str]  # names lower-cased
    body: bytes
    status: int = 0
    location: str = ""  # the Location header of the answer, absolute


class Log:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[Entry] = []

    def add(self, e: Entry) -> None:
        with self._lock:
            self._items.append(e)

    def snapshot(self) -> list[Entry]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# ---------------------------------------------------------------- Cloudflare Pages _headers
def parse_headers_file(text: str) -> list[tuple[str, dict[str, str]]]:
    """The rules of a `_headers` file: a line that starts in column 0 is a URL pattern, the indented `Name: value` lines under it are its headers.
    Blank lines and `#` comments are skipped."""
    rules: list[tuple[str, dict[str, str]]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] not in " \t":
            rules.append((line.strip(), {}))
        elif rules and ":" in line:
            name, value = line.strip().split(":", 1)
            rules[-1][1][name.strip()] = value.strip()
    return rules


def pattern_matches(pattern: str, path: str) -> bool:
    """A Pages pattern: `*` is a splat that matches anything (also slashes); everything else is literal."""
    return re.fullmatch(".*".join(re.escape(part) for part in pattern.split("*")), path) is not None


def headers_for(rules: list[tuple[str, dict[str, str]]], path: str) -> dict[str, str]:
    """Every header of every rule that matches the path; for a name set twice the values are joined with ", " (as Pages does)."""
    out: dict[str, str] = {}
    for pat, hdrs in rules:
        if pattern_matches(pat, path):
            for k, v in hdrs.items():
                out[k] = f"{out[k]}, {v}" if k in out else v
    return out


# ---------------------------------------------------------------- servers
class _Quiet(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):  # a closed connection is not news
        pass


class _Base(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    kind = ""

    def log_message(self, *a):  # silence the stderr line per request
        pass

    # --- one entry per request, written when the status is known
    def _entry(self) -> Entry:
        u = urlsplit(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(min(n, BODY_CAP)) if n else b""
        if n > BODY_CAP:
            self.rfile.read(n - BODY_CAP)
        e = Entry(self.kind, self.command, u.path, u.query, {k.lower(): v for k, v in self.headers.items()}, body)
        self.server.log.add(e)  # type: ignore[attr-defined]
        return e

    def _send(self, e: Entry, status: int, headers: dict[str, str], body: bytes = b"", head: bool = False) -> None:
        e.status = status
        e.location = headers.get("Location", "")
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head and body:
            self.wfile.write(body)

    def _refuse(self) -> None:
        e = self._entry()
        self._send(e, 405, {"Allow": "GET, HEAD", **self.extra_headers("/")}, b"method not allowed")

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse

    def extra_headers(self, path: str) -> dict[str, str]:
        return {}

    def do_GET(self):
        self.serve(False)

    def do_HEAD(self):
        self.serve(True)

    def serve(self, head: bool) -> None:
        raise NotImplementedError


class _PageHandler(_Base):
    kind = "page"

    def extra_headers(self, path: str) -> dict[str, str]:
        return headers_for(self.server.rules, path)  # type: ignore[attr-defined]

    def serve(self, head: bool) -> None:
        e = self._entry()
        root: Path = self.server.root  # type: ignore[attr-defined]
        path = e.path
        rel = "index.html" if path == "/" else path.lstrip("/")
        f = (root / rel).resolve()
        if not f.is_file() or root.resolve() not in f.parents:
            return self._send(e, 404, {"Content-Type": "text/plain", **self.extra_headers(path)}, b"not found", head)
        mime = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".txt": "text/plain", ".wasm": "application/wasm"}.get(f.suffix, "application/octet-stream")
        self._send(e, 200, {"Content-Type": mime, **self.extra_headers(path)}, f.read_bytes(), head)


class _ModelHandler(_Base):
    """The first hop (kind "model": /resolve/ only) and the signed files (kind "cdn": /cdn/ only) share the code, each on its own origin."""

    kind = "model"
    CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges, ETag"}

    def serve(self, head: bool) -> None:
        e = self._entry()
        srv = self.server
        m = re.fullmatch(r"/resolve/([0-9a-f]{40})/([A-Za-z0-9][A-Za-z0-9._-]*)", e.path) if self.kind == "model" else None
        if m and m.group(1) == srv.commit and (srv.root / m.group(2)).is_file():  # type: ignore[attr-defined]
            name = m.group(2)
            loc = f"{srv.cdn_origin}/cdn/{name}?Expires={int(time.time()) + 3600}&Signature={'ab12' * 8}"  # type: ignore[attr-defined]
            return self._send(e, 302, {**self.CORS, "Location": loc, "Cache-Control": "no-store", "Accept-Ranges": "bytes"}, b"", head)
        c = re.fullmatch(r"/cdn/([A-Za-z0-9][A-Za-z0-9._-]*)", e.path) if self.kind == "cdn" else None
        if c and (srv.root / c.group(1)).is_file():  # type: ignore[attr-defined]
            q = parse_qs(e.query)
            if "Expires" not in q or "Signature" not in q:
                return self._send(e, 403, self.CORS, b"unsigned", head)
            return self._file(e, srv.root / c.group(1), head)  # type: ignore[attr-defined]
        self._send(e, 404, self.CORS, b"not found", head)

    def _file(self, e: Entry, f: Path, head: bool) -> None:
        size = f.stat().st_size
        h = {**self.CORS, "Accept-Ranges": "bytes", "Content-Type": "application/octet-stream", "ETag": f'"{size}-{int(f.stat().st_mtime)}"', "Cache-Control": "public, max-age=31536000"}
        start, end, status = 0, size - 1, 200
        rng = e.headers.get("range")
        if rng:
            r = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not r or (not r.group(1) and not r.group(2)):
                return self._send(e, 416, {**h, "Content-Range": f"bytes */{size}"}, b"", head)
            if r.group(1):
                start = int(r.group(1))
                end = min(int(r.group(2)), size - 1) if r.group(2) else size - 1
            else:
                start = max(0, size - int(r.group(2)))
            if start > end or start >= size:
                return self._send(e, 416, {**h, "Content-Range": f"bytes */{size}"}, b"", head)
            status = 206
            h["Content-Range"] = f"bytes {start}-{end}/{size}"
        n = end - start + 1
        e.status = status
        self.send_response(status)
        for k, v in h.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(n))
        self.end_headers()
        if head:
            return
        try:
            with f.open("rb") as fh:
                fh.seek(start)
                left = n
                while left > 0:
                    buf = fh.read(min(CHUNK, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def extra_headers(self, path: str) -> dict[str, str]:
        return self.CORS


class _CdnHandler(_ModelHandler):
    kind = "cdn"


@dataclass
class Host:
    """A running host: `origin` (http://127.0.0.1:PORT), its request `log`, and `stop()`. The model host also holds `cdn`, the host it redirects to."""

    kind: str
    httpd: _Quiet
    thread: threading.Thread
    log: Log = field(default_factory=Log)
    cdn: "Host | None" = None

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self) -> None:
        if self.cdn is not None:
            self.cdn.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _start(kind: str, handler, **attrs) -> Host:
    httpd = _Quiet(("127.0.0.1", 0), handler)
    log = Log()
    httpd.log = log  # type: ignore[attr-defined]
    for k, v in attrs.items():
        setattr(httpd, k, v)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    return Host(kind, httpd, t, log)


def start_model_host(models_dir: Path, commit: str) -> Host:
    """The first hop (`/resolve/<commit>/<name>` on one origin) and the signed files (`/cdn/<name>?Expires&Signature` on another) of the files of
    `models_dir` (the exported onnx folder, manifest.json included). Returns the first-hop host; its `.cdn` is the other one."""
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    cdn = _start("cdn", _CdnHandler, root=Path(models_dir), commit=commit, cdn_origin="")
    model = _start("model", _ModelHandler, root=Path(models_dir), commit=commit, cdn_origin=cdn.origin)
    model.cdn = cdn
    return model


def start_page_host(dist: Path) -> Host:
    """Serves a built pages-dist and applies its _headers file."""
    dist = Path(dist)
    hf = dist / "_headers"
    rules = parse_headers_file(hf.read_text()) if hf.is_file() else []
    return _start("page", _PageHandler, root=dist, rules=rules)
