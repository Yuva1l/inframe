"""The one place the server's own outbound requests go (docs/T9A_DESIGN.md section 4, "Outbound network"). Standard library only.

`fetch(url, kind=..., max_bytes=...)` returns the body bytes, or raises `NetError`:

- `https` only, no user name or password in the address.
- `kind="api"`: the host must be exactly `api.openverse.org` (the music search and the per-track record).
- `kind="audio"`: the track file named by that record. Any https host, but every address it resolves to must be global
  (`ipaddress.ip_address(a).is_global`: no loopback, private, link-local or shared-address range). The answer must have a
  `Content-Type` of `audio/*` or `application/octet-stream`.
- A redirect is followed only to the same host over https (at most 5 hops); the address check runs again on every hop.
- The body is capped (`max_bytes`, 50 MB for audio); a longer one is refused, by its `Content-Length` before reading or while reading.

Not covered (accepted in the design): the gap between the address check and the connect, i.e. DNS rebinding of a music host.
It reaches only a download that is capped and must still sniff as audio.
"""
from __future__ import annotations

import contextlib
import http.client
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request

API_HOST = "api.openverse.org"
KINDS = ("api", "audio")
MAX_AUDIO_BYTES = 50 * 1024 * 1024
MAX_API_BYTES = 8 * 1024 * 1024
MAX_REDIRECTS = 5
TIMEOUT_S = 60  # the API answers in 5-30 s and sometimes much slower
USER_AGENT = "inframe/0.1 (local video tool; uses Openverse)"
REDIRECTS = (301, 302, 303, 307, 308)
CHUNK = 64 * 1024


class NetError(RuntimeError):
    """A refused or failed request. `status` is the HTTP status when the other side answered with one."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib never follows a redirect by itself: it raises HTTPError and `fetch` decides."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _open(url: str, kind: str):
    """One request, no redirects (the test seam). Raises urllib.error.HTTPError for 3xx and 4xx/5xx."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json" if kind == "api" else "audio/*, application/octet-stream;q=0.5"})
    return _opener.open(req, timeout=TIMEOUT_S)


def _resolve(host: str, port: int) -> list[str]:
    """Every address `host` resolves to (the test seam over socket.getaddrinfo)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as e:
        raise NetError(f"could not resolve {host}: {getattr(e, 'strerror', None) or e}") from e
    return [str(i[4][0]) for i in infos]


def _is_global(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])  # drop an IPv6 zone id
    except ValueError:
        return False
    if ip.version == 6:
        v4 = _embedded_v4(ip)
        if v4 == "refuse":
            return False
        if v4 is not None:
            ip = v4
    return ip.is_global


_NAT64 = ipaddress.ip_network("64:ff9b::/96")  # RFC 6052 well-known prefix: a gateway turns the last 32 bits into an IPv4 destination
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")  # RFC 8215 local-use NAT64: operator-defined, never public
_V4_COMPAT = ipaddress.ip_network("::/96")  # deprecated IPv4-compatible addresses (::a.b.c.d)
_SIXTO4 = ipaddress.ip_network("2002::/16")  # 6to4: bits 16..47 are an IPv4 address
_TEREDO = ipaddress.ip_network("2001::/32")
_SIIT = ipaddress.ip_network("::ffff:0:0:0/96")  # RFC 6145 IPv4-translated (::ffff:0:a.b.c.d): a stateless translator maps it to IPv4


def _embedded_v4(ip):
    """The IPv4 address an IPv6 one carries (mapped, NAT64, 6to4), judged as IPv4; "refuse" for ranges that tunnel or translate
    to an address we cannot vouch for (IPv4-compatible, IPv4-translated SIIT, local-use NAT64, Teredo); None for a plain IPv6 address."""
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip in _SIXTO4:
        return ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
    if ip in _NAT64_LOCAL or ip in _TEREDO or ip in _V4_COMPAT or ip in _SIIT:
        return "refuse"
    return None


def _show(url: str) -> str:
    return url.split("?", 1)[0][:60]


def check_url(url: str, kind: str) -> tuple[str, int]:
    """The (host, port) of `url` when it may be requested as `kind`; NetError otherwise."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if not isinstance(url, str):
        raise NetError("refusing a non-string address")
    try:
        p = urllib.parse.urlparse(url)
        port = p.port or 443
    except ValueError:
        raise NetError(f"refusing a malformed address: {_show(url)}") from None
    if p.scheme != "https":
        raise NetError(f"refusing a non-https address: {_show(url)}")
    host = (p.hostname or "").lower()
    if not host:
        raise NetError(f"refusing an address without a host: {_show(url)}")
    if p.username is not None or p.password is not None:
        raise NetError("refusing an address with a user name or password")
    if kind == "api" and host != API_HOST:
        raise NetError(f"refusing host {host}: only {API_HOST} is allowed here")
    addrs = _resolve(host, port)
    if not addrs:
        raise NetError(f"could not resolve {host}")
    for a in addrs:
        if not _is_global(a):
            raise NetError(f"refusing {host}: it resolves to a non-public address")
    return host, port


def _content_type(resp) -> str:
    return str(resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()


def _read_capped(resp, max_bytes: int) -> bytes:
    try:
        declared = int(resp.headers.get("Content-Length") or 0)
    except ValueError:
        declared = 0
    if declared > max_bytes:
        raise NetError(f"file is larger than {max_bytes // 2**20} MB")
    buf = bytearray()
    while True:
        chunk = resp.read(min(CHUNK, max_bytes + 1 - len(buf)))
        if not chunk:
            return bytes(buf)
        buf += chunk
        if len(buf) > max_bytes:
            raise NetError(f"file is larger than {max_bytes // 2**20} MB")


def fetch(url: str, *, kind: str, max_bytes: int | None = None) -> bytes:
    """GET `url` under the rules in the module doc and return the body."""
    host, port = check_url(url, kind)
    if max_bytes is None:
        max_bytes = MAX_AUDIO_BYTES if kind == "audio" else MAX_API_BYTES
    cur = url
    for _ in range(MAX_REDIRECTS + 1):
        try:
            resp = _open(cur, kind)
        except urllib.error.HTTPError as e:
            with contextlib.suppress(Exception):
                e.close()
            if e.code in REDIRECTS:
                loc = e.headers.get("Location") if e.headers is not None else None
                if not loc:
                    raise NetError(f"HTTP {e.code} redirect without a location for {_show(cur)}", e.code) from e
                nxt = urllib.parse.urljoin(cur, loc)
                try:
                    np = urllib.parse.urlparse(nxt)
                    nport = np.port or 443
                except ValueError:
                    raise NetError(f"refusing a malformed redirect from {host}") from None
                if np.scheme != "https":
                    raise NetError(f"refusing a redirect to a non-https address: {_show(nxt)}")
                if (np.hostname or "").lower() != host or nport != port:
                    raise NetError(f"refusing a redirect from {host} to another host")
                check_url(nxt, kind)
                cur = nxt
                continue
            raise NetError(f"HTTP {e.code} for {_show(cur)}", e.code) from e
        except NetError:
            raise
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as e:
            raise NetError(f"could not reach {host}: {getattr(e, 'reason', e)}") from e
        with contextlib.closing(resp):
            if kind == "audio":
                ct = _content_type(resp)
                if not (ct.startswith("audio/") or ct == "application/octet-stream"):
                    raise NetError(f"{host} sent {ct or 'no content type'}, not audio")
            try:
                return _read_capped(resp, max_bytes)
            except NetError:
                raise
            except (http.client.HTTPException, TimeoutError, OSError) as e:
                raise NetError(f"could not read from {host}: {e}") from e
    raise NetError(f"too many redirects from {host}")
