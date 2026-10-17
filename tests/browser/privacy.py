"""The privacy check of the Pages build (docs/T9C_DESIGN.md section 3), as a pure function over request records, so it is unit-tested without a browser
(tests/test_privacy_checker.py) and then applied to what Chrome and the two hosts really saw (test_pages_privacy.py).

`check(seen, ...)` returns one line per problem; an empty list means the page did what it promises:
  - only the page origin and the model origin are contacted (a third origin is reported with the first URL that went there);
  - every request is a GET, or a HEAD to the model origin: no OPTIONS, POST, PUT and no body at all;
  - on the model origin only the files of the manifest are asked for (the first hop `/resolve/<commit>/<name>` and the signed `/cdn/<name>?Expires&Signature`),
    without cookies, a referer or an authorization header;
  - on the model origin the signed `/cdn/` URL is exactly the Location the model host issued (so Expires and Signature carry nothing of the page's), Range is a
    plain byte range inside the file, and only the headers of ALLOWED_HEADERS are sent, on both origins (a custom header name cannot carry data);
  - on the page origin no query string and only files that are in the built folder;
  - none of the secrets (the canary token, the file name, runs of bytes of the clip) appears, plain, percent-encoded, in hex or in base64, in any URL, header
    name, header value or body;
  - no URL is longer than `max_url` characters.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from urllib.parse import quote_from_bytes, unquote_to_bytes, urlsplit

IGNORED_SCHEMES = ("blob", "data", "about")
MODEL_QUERY_KEYS = {"Expires", "Signature"}
# The request headers Chrome adds by itself to a fetch, a script, a worker or a navigation. A name outside this list (x-clip, x-<hash>, ...) is reported.
ALLOWED_HEADERS = frozenset({
    "accept", "accept-encoding", "accept-language", "cache-control", "connection", "host", "origin", "pragma", "priority", "range", "referer",
    "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform", "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site", "sec-fetch-user", "sec-purpose",
    "upgrade-insecure-requests", "user-agent", "if-none-match", "if-modified-since", "if-range", "service-worker-navigation-preload",
})
RANGE_RE = re.compile(r"bytes=(\d{1,12})-(\d{0,12})")


@dataclass(frozen=True)
class Seen:
    """One request, from the browser's own event or from a host's log."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)  # names lower-cased
    body: bytes | None = None
    source: str = "browser"  # "browser", "page-host" or "model-host"


def origin_of(url: str) -> str:
    u = urlsplit(url)
    return f"{u.scheme}://{u.netloc}"


def _b64_forms(secret: bytes, encode) -> list[bytes]:
    """A secret inside a larger base64 text starts at any of three alignments. For a secret that starts `pad` bytes after a 3-byte boundary the text
    holds the encoding of secret[3 - pad:] (the chars that mix in the neighbours are not in it) and, when that part does not end on a boundary, all but the
    last char of its encoding (the last char mixes in the bytes that follow). So the needles are the encodings of secret[j:] for j = 0, 1, 2 without the "="
    and, when len(secret[j:]) is not a multiple of 3, without the last char."""
    out = []
    for j in range(3):
        part = secret[j:]
        if not part:
            continue
        enc = encode(part).rstrip(b"=")
        if len(part) % 3:
            enc = enc[:-1]
        out.append(enc)
    return out


def variants(secret: bytes) -> list[bytes]:
    """The forms a secret can take in a request: as is, percent-encoded, hex, base64 (standard and URL-safe, on its own or inside a longer base64 text at
    any of the three alignments), each also lower-cased as text."""
    forms = [secret, quote_from_bytes(secret).encode(), secret.hex().encode(), *_b64_forms(secret, base64.b64encode), *_b64_forms(secret, base64.urlsafe_b64encode)]
    out: list[bytes] = []
    for f in forms:
        for g in (f, f.lower()):
            if g and g not in out:
                out.append(g)
    return out


def _haystacks(s: Seen) -> list[tuple[str, bytes]]:
    """The places a secret could be, as bytes: the URL (raw and percent-decoded), every header name and value, the body."""
    parts: list[tuple[str, bytes]] = [("URL", s.url.encode("utf-8", "replace")), ("URL", unquote_to_bytes(s.url))]
    for k, v in s.headers.items():
        parts.append((f"header {k}", f"{k}: {v}".encode("utf-8", "replace")))
    if s.body:
        parts.append(("body", s.body))
    return parts


def _short(url: str, n: int = 100) -> str:
    return url if len(url) <= n else url[:n] + f"... ({len(url)} chars)"


def check(
    seen: list[Seen],
    *,
    page_origin: str,
    model_origin: str,
    model_paths: set[str],
    page_files: set[str],
    secrets: dict[str, bytes],
    max_url: int = 512,
    require_both: bool = True,
    redirect_origin: str | None = None,
    issued: set[str] | None = None,
    sizes: dict[str, int] | None = None,
    allowed_headers: frozenset[str] = ALLOWED_HEADERS,
) -> list[str]:
    """Problems found in `seen`. `model_paths` are the paths the model host may be asked for (the resolve and the cdn path of every manifest name), `page_files`
    the paths that exist in the built folder (with a leading slash, "/" for the index), `secrets` maps a label to the bytes that must never leave.
    `redirect_origin` is the origin the signed /cdn/ files come from when it is not the model origin (Hugging Face redirects to another host); `issued` the
    absolute URLs the model host really put in a Location header (when given, a /cdn/ request for any other URL is reported); `sizes` the byte size of each
    manifest name (when given, a Range must lie inside the file)."""
    problems: list[str] = []
    origins: dict[str, str] = {}
    cdn_origin = redirect_origin or model_origin
    needles = [(label, v) for label, s in secrets.items() for v in variants(s)]
    for s in seen:
        scheme = urlsplit(s.url).scheme
        if scheme in IGNORED_SCHEMES:
            continue
        o = origin_of(s.url)
        origins.setdefault(o, s.url)
        u = urlsplit(s.url)
        where = f"{s.method} {_short(s.url)} [{s.source}]"
        if o not in (page_origin, model_origin, cdn_origin):
            problems.append(f"third origin {o}: {where}")
            continue
        # methods and bodies
        if s.method not in ("GET", "HEAD") or (s.method == "HEAD" and o == page_origin):
            problems.append(f"method {s.method} is not allowed here: {where}")
        if s.body:
            problems.append(f"request body of {len(s.body)} bytes: {where}")
        # what may be asked for
        for h in s.headers:
            if h not in allowed_headers:
                problems.append(f"header '{h}' is not one a browser adds by itself: {where}")
        rng = s.headers.get("range")
        if rng is not None:
            m = RANGE_RE.fullmatch(rng.strip())
            name = u.path.rsplit("/", 1)[-1]
            if o == page_origin or not m or (m.group(2) and int(m.group(2)) < int(m.group(1))):
                problems.append(f"Range '{rng[:60]}' is not a plain byte range for a model file: {where}")
            elif sizes is not None and name in sizes and (int(m.group(1)) >= sizes[name] or (m.group(2) and int(m.group(2)) >= sizes[name])):
                problems.append(f"Range '{rng}' lies outside the {sizes[name]} bytes of {name}: {where}")
        if o != page_origin:
            if u.path not in model_paths:
                problems.append(f"model host asked for {u.path}, which is not a manifest file: {where}")
            elif u.path.startswith("/resolve/") and o != model_origin:
                problems.append(f"/resolve/ asked of {o}, which is not the model host: {where}")
            elif u.path.startswith("/cdn/") and o != cdn_origin:
                problems.append(f"/cdn/ asked of {o}, which is not the host the model host redirects to: {where}")
            elif u.path.startswith("/resolve/") and u.query:
                problems.append(f"query string on the first hop to the model host: {where}")
            elif u.path.startswith("/cdn/"):
                keys = {p.split("=", 1)[0] for p in u.query.split("&") if p}
                if not keys <= MODEL_QUERY_KEYS:
                    problems.append(f"unexpected query keys {sorted(keys - MODEL_QUERY_KEYS)} on the model host: {where}")
                if issued is not None and s.url not in issued:
                    problems.append(f"the signed URL is not one the model host issued in a Location header: {where}")
            for h in ("cookie", "authorization", "referer"):
                if h in s.headers:
                    problems.append(f"model host request carries a {h} header: {where}")
        else:
            if u.query or "?" in s.url:
                problems.append(f"query string on the page host: {where}")
            if u.path not in page_files:
                problems.append(f"page host asked for {u.path}, which is not in the built folder: {where}")
        if len(s.url) > max_url:
            problems.append(f"URL of {len(s.url)} characters (limit {max_url}): {where}")
        # secrets, in every form, everywhere
        for place, hay in _haystacks(s):
            low = hay.lower()
            for label, needle in needles:
                if needle in hay or needle in low:
                    problems.append(f"secret '{label}' found in the {place}: {where}")
                    break
    if require_both:
        for name, o in (("page", page_origin), ("model", model_origin), ("cdn", cdn_origin)):
            if o not in origins:
                problems.append(f"the {name} origin {o} was never contacted")
    return problems
