"""Checks before (and, with --remote, after the model upload but before) the Cloudflare Pages deploy (`wrangler pages deploy`; docs/T9C_DEPLOY.md, docs/T9C_DESIGN.md section 4).

    uv run python tools/pages_preflight.py            # local: reads files only, contacts nobody
    uv run python tools/pages_preflight.py --remote   # maintainer only: asks the model host about each file
    uv run python tools/pages_preflight.py --no-dist --remote   # the same before the page is built (checks model-host.json only)

Local mode checks web/pages-dist (run `npm --prefix web run build:pages` first): it exists, every file is under 25 MiB, there are fewer than
20,000 files, the `_headers` connect-src is exactly 'self' plus web/pages/model-host.json's `connect`, the model base is https, not the placeholder
and has a 40-hex commit, the built manifest is web/pages/manifest.json byte for byte, and the built page carries the base.

--remote sends HEAD and GET Range: bytes=0-0 for each file of the manifest to the model base with an Origin header, follows redirects one hop at a
time, and prints for each hop the status, the host, Access-Control-Allow-Origin (ACAO), and whether connect-src covers it (the first address by its
path, redirect targets by their origin only, as a browser does). It reports the sizes against the manifest. --digest also downloads each file once
and checks its sha256 (about 239 MB). Exit code 1 when anything is reported as a problem. It never uploads and sends no cookies; the only other
data in a request is the Origin header you pass with --origin. Tests run it against a local stub server only.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import ssl
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
PAGES_FILE_LIMIT = 25 * 1024 * 1024  # Cloudflare static assets: bytes per file
PAGES_FILES_MAX = 20_000  # Cloudflare static assets: files per Worker version (free plan)
PLACEHOLDER = re.compile(r"OWNER|COMMIT")
COMMIT_SEGMENT = re.compile(r"/[0-9a-f]{40}(/|$)")
BARE_HF = re.compile(r"^https://(?:\*\.)?(?:huggingface\.co|hf\.co)/?$")
# Hugging Face redirects a model download to a CDN host (us.aws.cdn.hf.co, cas-bridge.xethub.hf.co, ...) that depends on the visitor's region and changes over time,
# so a fixed list breaks for some visitors. The one wildcard accepted is exactly https://*.cdn.hf.co (the download CDN), and only when the model base is a pinned
# huggingface.co resolve URL (owner/repo/resolve/<40-hex commit>/): every file is checked against the sha256 in the page's own manifest, so a CDN host can serve
# bytes but not change them. The whole of hf.co is not opened: it includes the Xet upload host (cas-server.xethub.hf.co [unverified against HF docs]), and the CSP is
# the backstop that footage cannot leave the browser. The Xet download host (cas-bridge.xethub.hf.co) is named exactly. Every other wildcard stays refused.
# Keep in step with HF_CDN_WILDCARD and HF_UPLOAD_HOSTS in web/vite.config.ts.
HF_CDN_WILDCARD = "https://*.cdn.hf.co"
HF_UPLOAD_HOSTS = ("https://cas-server.xethub.hf.co",)
HF_PINNED_BASE = re.compile(r"^https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/")
DEFAULT_ORIGIN = "https://inframe.pages.dev"  # the live page; pass another site origin with --origin
MAX_REDIRECTS = 6


# --- connect-src -------------------------------------------------------------------------------------------------------------------------------

def _origin(url: str) -> tuple[str, str, int | None]:
    u = urlsplit(url)
    port = u.port or {"https": 443, "http": 80}.get(u.scheme)
    return u.scheme, (u.hostname or "").lower(), port


def connect_covers(entry: str, address: str, first_hop: bool = True) -> bool:
    """Does one connect-src entry (an origin, `https://*.host`, or either with a path: a prefix when it ends in a slash, else an exact match) allow this address? A path in the
    entry is matched on the first request only; after a redirect a browser compares the origin alone."""
    e, a = urlsplit(entry), urlsplit(address)
    if first_hop and e.path not in ("", "/"):
        # CSP: a path ending in a slash is a prefix; any other path must match exactly
        if not (a.path.startswith(e.path) if e.path.endswith("/") else a.path == e.path):
            return False
    es, eh, ep = _origin(entry)
    as_, ah, ap = _origin(address)
    if es != as_ or ep != ap:
        return False
    if eh.startswith("*."):
        return ah.endswith(eh[1:])
    return eh == ah


def _covers_safely(entry: str, address: str) -> bool:
    """connect_covers for an entry that may be malformed (a port of `*`): such an entry covers nothing here; broad_connect reports it."""
    try:
        return connect_covers(entry, address, False) or connect_covers(entry, address, True)
    except ValueError:
        return False


def covered(connect: list[str], address: str, first_hop: bool) -> bool:
    return any(connect_covers(c, address, first_hop) for c in connect)


def headers_connect(text: str) -> list[str] | None:
    """The tokens of connect-src in a _headers file, or None when there is none."""
    m = re.search(r"connect-src ([^;\n]*)", text)
    return m.group(1).split() if m else None


# --- local mode --------------------------------------------------------------------------------------------------------------------------------

def load_host(host_file: Path) -> tuple[dict | None, list[str]]:
    try:
        j = json.loads(host_file.read_text())
    except (OSError, ValueError) as e:
        return None, [f"{host_file}: cannot read ({e})"]
    if not isinstance(j, dict) or not isinstance(j.get("base"), str) or not isinstance(j.get("connect"), list):
        return None, [f"{host_file}: needs a string base and a list connect"]
    return {"base": j["base"], "connect": [str(c) for c in j["connect"]]}, []


def broad_connect(entry: str) -> str | None:
    """Why a connect entry lets the page talk to more than one named host, or None. Catches `*`, bare schemes (https:, data:, ...), CSP keywords,
    a port wildcard and a host wildcard with no registrable name behind it (`https://*`, `https://*.com`)."""
    if entry == "*" or entry.endswith(":") or entry.startswith("'"):
        return "allows connecting anywhere (a bare `*`, scheme or keyword)"
    u = urlsplit(entry)
    if not u.scheme or "://" not in entry or not u.netloc:
        return "is not an origin like https://host"
    host = u.netloc.rsplit("@", 1)[-1]
    name, _, port = host.partition(":") if not host.startswith("[") else (host, "", "")
    if port == "*":
        return "allows any port"
    name = name.rstrip(".")  # `*.com.` is the same host as `*.com`
    if "*" in name and not (name.startswith("*.") and "*" not in name[2:] and "." in name[2:]):
        return "has a wildcard host that is not `*.name.tld`"
    if name.startswith("*.") and _public_suffix(name[2:].lower()):
        return "has a wildcard over a public suffix (anyone can register a name under it)"
    return None


# Suffixes under which unrelated parties register names: a wildcard over one lets the page talk to all of them. No public-suffix
# library (a new dependency to license-check): the common multi-label country suffixes and hosting platforms, plus the shape
# `<co|com|org|net|ac|gov|edu|ne|or|go|gob>.<two-letter country>`.
_SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "edu", "ne", "or", "go", "gob", "mil", "sch", "ltd", "plc", "me", "nom"}
_PLATFORM_SUFFIXES = {"github.io", "gitlab.io", "pages.dev", "workers.dev", "netlify.app", "vercel.app", "herokuapp.com", "appspot.com",
                      "blogspot.com", "web.app", "firebaseapp.com", "cloudfront.net", "azurewebsites.net", "amazonaws.com", "s3.amazonaws.com",
                      "ondigitalocean.app", "onrender.com", "fly.dev", "glitch.me", "repl.co", "surge.sh", "now.sh", "r2.dev", "hf.space"}


def _public_suffix(base: str) -> bool:
    """True when `base` (the part after `*.`, lower case, no trailing dot) is a public suffix, as far as the lists above know."""
    labels = base.split(".")
    if base in _PLATFORM_SUFFIXES:
        return True
    return len(labels) == 2 and labels[0] in _SECOND_LEVEL and len(labels[1]) == 2


def check_host(host: dict, allow_local: bool = False) -> list[str]:
    """The model host in web/pages/model-host.json: https, no placeholder, a 40-hex commit, the base covered by connect, no whole-Hugging-Face entry."""
    base, connect, problems = host["base"], host["connect"], []
    u = urlsplit(base)
    local = allow_local and u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost")
    if PLACEHOLDER.search(base):
        problems.append(f"the model base {base} is still the placeholder (OWNER / COMMIT): put the real repo and its commit in web/pages/model-host.json")
    elif not local:
        if u.scheme != "https":
            problems.append(f"the model base {base} is not https")
        if not COMMIT_SEGMENT.search(u.path):
            problems.append(f"the model base {base} has no 40-hex commit in its path, so the files are not pinned")
    if u.query or u.fragment or u.username or u.password:
        problems.append("the model base carries a query, fragment or credentials")
    if not base.endswith("/"):
        problems.append("the model base must end with a slash")
    for c in connect:
        wide = broad_connect(c)
        if c == HF_CDN_WILDCARD:
            if not HF_PINNED_BASE.match(base):
                problems.append(f"connect entry {c} is accepted only when the model base is a pinned https://huggingface.co/OWNER/REPO/resolve/<40-hex commit>/ URL ({base})")
        elif "*" in c and wide is None:
            problems.append(f"connect entry {c} is a wildcard; the only one accepted is {HF_CDN_WILDCARD}, for a pinned Hugging Face base")
        elif BARE_HF.match(c):
            problems.append(f"connect entry {c} opens all of Hugging Face to the page; name the repo path and the exact redirect host")
        elif wide:
            problems.append(f"connect entry {c} {wide}; the public page may only connect to the model host")
        elif urlsplit(c).scheme != "https" and not (local and urlsplit(c).scheme == "http"):
            problems.append(f"connect entry {c} is not https")
    for up in HF_UPLOAD_HOSTS:
        hit = [c for c in connect if _covers_safely(c, up + "/")]
        if hit:
            problems.append(f"connect entry {' '.join(hit)} covers the Hugging Face upload host {up}: the page could send the video there")
    if not covered(connect, base, True):
        problems.append(f"the model base {base} is not covered by connect ({' '.join(connect) or 'empty'})")
    return problems


def check_local(dist: Path | None, host_file: Path, manifest: Path, allow_local: bool = False) -> list[str]:
    """Problems with the model host file and, unless `dist` is None (--no-dist), with the built folder."""
    problems: list[str] = []
    host, errs = load_host(host_file)
    problems += errs
    if host:
        problems += check_host(host, allow_local)
    if dist is None:
        return problems
    if not dist.is_dir() or not (dist / "index.html").is_file():
        problems.append(f"{dist} is missing or has no index.html: run `npm --prefix web run build:pages`")
        return problems
    files = [p for p in dist.rglob("*") if p.is_file()]
    for p in sorted(files):
        if p.stat().st_size > PAGES_FILE_LIMIT:
            problems.append(f"{p.relative_to(dist)}: {p.stat().st_size} bytes is over the 25 MiB static-assets limit")
    if len(files) >= PAGES_FILES_MAX:
        problems.append(f"{len(files)} files; static assets allow fewer than {PAGES_FILES_MAX}")
    hdr = dist / "_headers"
    if not hdr.is_file():
        problems.append("_headers is missing from the build (the site would be served with no CSP)")
    else:
        tokens = headers_connect(hdr.read_text())
        if tokens is None:
            problems.append("_headers has no connect-src")
        elif host:
            want = ["'self'", *host["connect"]]
            if tokens != want:
                extra, missing = [t for t in tokens if t not in want], [t for t in want if t not in tokens]
                problems.append(f"_headers connect-src does not match model-host.json: extra {extra or 'none'}, missing {missing or 'none'}"
                                + ("" if extra or missing else " (order differs)") + "; rebuild with `npm --prefix web run build:pages`")
    built = dist / "models" / "manifest.json"
    if not built.is_file():
        problems.append("models/manifest.json is missing from the build")
    elif manifest.is_file() and built.read_bytes() != manifest.read_bytes():
        problems.append("models/manifest.json differs from web/pages/manifest.json; rebuild")
    if host and not any(host["base"] in p.read_text(errors="replace") for p in (dist / "assets").glob("*.js")):
        problems.append("no built script carries the model base from model-host.json; the build was made for another host, rebuild")
    return problems


# --- remote mode -------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Hop:
    url: str
    status: int
    headers: dict[str, str]
    body: int = 0  # bytes read from the body
    sent_origin: str | None = None  # the Origin header this request carried: the page origin, or "null" once the origin was tainted by a cross-origin redirect
    sha256: str | None = None


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    hosts: dict[str, bool] = field(default_factory=dict)  # redirect host -> covered


def _connect(u, timeout: float) -> http.client.HTTPConnection:
    if u.scheme == "https":
        return http.client.HTTPSConnection(u.hostname, u.port, timeout=timeout, context=ssl.create_default_context())
    return http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)


def fetch_chain(method: str, url: str, headers: dict[str, str], timeout: float = 30.0, read: int = 1 << 20, digest: bool = False) -> list[Hop]:
    """One request, following redirects by hand; returns one Hop per response. The body of the last response is read up to `read` bytes
    (all of it, hashed, with digest=True). No cookies are sent or kept. As a browser does (Fetch, HTTP-redirect fetch), once a redirect leaves the
    origin of a URL that is itself not the page's origin, the request's origin is tainted and every later hop is sent `Origin: null`."""
    hops: list[Hop] = []
    page_origin = headers.get("Origin")
    tainted = False
    for _ in range(MAX_REDIRECTS + 1):
        u = urlsplit(url)
        sent = {**headers, "Origin": "null"} if tainted and page_origin is not None else headers
        conn = _connect(u, timeout)
        try:
            conn.request(method, (u.path or "/") + (f"?{u.query}" if u.query else ""), headers={**sent, "Accept-Encoding": "identity", "User-Agent": "inframe-pages-preflight"})
            r = conn.getresponse()
            h = {k.lower(): v for k, v in r.getheaders()}
            hop = Hop(url, r.status, h, sent_origin=sent.get("Origin"))
            hops.append(hop)
            if r.status in (301, 302, 303, 307, 308) and h.get("location"):
                nxt = urljoin(url, h["location"])
                if page_origin is not None and _origin(nxt) != _origin(url) and _origin(page_origin) != _origin(url):
                    tainted = True
                url = nxt
                continue
            if method != "HEAD":
                sha = hashlib.sha256() if digest else None
                while True:
                    chunk = r.read(1 << 20 if digest else min(1 << 16, max(read - hop.body, 1)))
                    if not chunk:
                        break
                    hop.body += len(chunk)
                    if sha:
                        sha.update(chunk)
                    elif hop.body >= read:
                        break
                hop.sha256 = sha.hexdigest() if sha else None
            return hops
        finally:
            conn.close()
    raise RuntimeError(f"more than {MAX_REDIRECTS} redirects from {hops[0].url}")


def _acao_ok(hop: Hop, origin: str) -> bool:
    """Would the CORS check pass? It compares ACAO with the Origin the request really carried: `null` after a cross-origin redirect, where only `*` or `null` match."""
    sent = hop.sent_origin if hop.sent_origin is not None else origin
    return hop.headers.get("access-control-allow-origin") in ("*", sent)


def _hop_lines(rep: Report, label: str, hops: list[Hop], connect: list[str], origin: str, allow_local: bool, name: str) -> None:
    for i, hop in enumerate(hops):
        host = urlsplit(hop.url).netloc
        cov = covered(connect, hop.url, i == 0)
        acao = hop.headers.get("access-control-allow-origin", "(none)")
        rep.lines.append(f"  {label} hop {i + 1}: {hop.status} {host}  ACAO {acao}  connect-src {'covers' if cov else 'DOES NOT COVER'} it")
        if i > 0:
            rep.hosts[host] = rep.hosts.get(host, True) and cov
        if not cov:
            rep.problems.append(f"{name}: {label} hop {i + 1} {hop.url.split('?')[0]} is not covered by connect-src")
        if not _acao_ok(hop, origin):
            why = ", and the request carries Origin: null after a cross-origin redirect (only * or null passes)" if hop.sent_origin == "null" else ""
            rep.problems.append(f"{name}: {label} hop {i + 1} ({host}) answers ACAO {acao}, a page at {origin} would be blocked{why}")
        if i > 0 and urlsplit(hop.url).scheme != "https" and not allow_local:
            rep.problems.append(f"{name}: {label} redirects to {hop.url.split('?')[0]}, which is not https")


def check_remote(host: dict, manifest_file: Path, origin: str, timeout: float = 30.0, digest: bool = False, allow_local: bool = False) -> Report:
    """Ask the model host about each manifest file (HEAD, then GET Range 0-0, each with an Origin header)."""
    rep = Report()
    base, connect = host["base"], host["connect"]
    files = json.loads(manifest_file.read_text())["files"]
    for name, meta in sorted(files.items()):
        url = base + name
        rep.lines.append(f"{name} ({meta['bytes']:,} bytes)")
        h = {"Origin": origin}
        try:
            head = fetch_chain("HEAD", url, h, timeout)
            rng = fetch_chain("GET", url, {**h, "Range": "bytes=0-0"}, timeout)
        except (OSError, RuntimeError, http.client.HTTPException) as e:
            rep.problems.append(f"{name}: request failed: {type(e).__name__}: {e}")
            rep.lines.append(f"  request failed: {e}")
            continue
        _hop_lines(rep, "HEAD", head, connect, origin, allow_local, name)
        _hop_lines(rep, "RANGE", rng, connect, origin, allow_local, name)
        if head[-1].status != 200:
            rep.problems.append(f"{name}: HEAD ends with status {head[-1].status}")
        size = head[-1].headers.get("content-length")
        rep.lines.append(f"  size: HEAD content-length {size or '(none)'}, manifest {meta['bytes']}")
        if size is None:
            rep.lines.append("  note: HEAD gave no content-length; the Range answer decides")
        elif int(size) != meta["bytes"]:
            rep.problems.append(f"{name}: HEAD content-length {size} differs from the manifest's {meta['bytes']}")
        last = rng[-1]
        m = re.fullmatch(r"bytes 0-0/(\d+)", last.headers.get("content-range", ""))
        if last.status == 206 and m:
            rep.lines.append(f"  range: 206, total {m.group(1)}, accept-ranges {last.headers.get('accept-ranges', '(none)')}")
            if int(m.group(1)) != meta["bytes"]:
                rep.problems.append(f"{name}: Content-Range total {m.group(1)} differs from the manifest's {meta['bytes']}")
        elif last.status == 200:
            rep.lines.append("  note: Range answered 200 (the host ignores Range); not a problem for the page, which fetches whole files")
        elif last.status == 206:
            rep.lines.append("  note: Range answered 206 without a Content-Range of bytes 0-0/total; the size is taken from HEAD")
        else:
            rep.problems.append(f"{name}: GET (with Range) ends with status {last.status}, not 200 or 206: the page's own GET would be refused too")
            rep.lines.append(f"  range: {last.status}")
        if digest:
            try:
                full = fetch_chain("GET", url, h, max(timeout, 120.0), digest=True)
            except (OSError, RuntimeError, http.client.HTTPException) as e:
                rep.problems.append(f"{name}: download failed: {type(e).__name__}: {e}")
                continue
            got = full[-1]
            ok = got.status == 200 and got.body == meta["bytes"] and got.sha256 == meta["sha256"]
            rep.lines.append(f"  digest: {got.body} bytes, sha256 {(got.sha256 or '')[:16]}... {'matches' if ok else 'DIFFERS from'} the manifest")
            if not ok:
                rep.problems.append(f"{name}: downloaded file (status {got.status}, {got.body} bytes, sha256 {(got.sha256 or '')[:16]}...) differs from the manifest")
    if rep.hosts:
        rep.lines.append("hosts seen after the first request: " + ", ".join(f"{h} ({'covered' if c else 'NOT covered'})" for h, c in sorted(rep.hosts.items())))
        rep.lines.append("connect-src entries: " + " ".join(["'self'", *connect]))
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", action="store_true", help="also ask the model host about each file (contacts the host in model-host.json; the maintainer runs this)")
    ap.add_argument("--digest", action="store_true", help="with --remote: download each file once and check its sha256 (about 239 MB)")
    ap.add_argument("--origin", default=DEFAULT_ORIGIN, help=f"the page origin sent as Origin with --remote (default {DEFAULT_ORIGIN})")
    ap.add_argument("--dist", default=str(WEB / "pages-dist"))
    ap.add_argument("--host-file", default=str(WEB / "pages" / "model-host.json"))
    ap.add_argument("--manifest", default=str(WEB / "pages" / "manifest.json"))
    ap.add_argument("--no-dist", action="store_true", help="check model-host.json only, not the built folder (before build:pages has run)")
    ap.add_argument("--allow-local", action="store_true", help="accept an http://127.0.0.1 or localhost model base (tests only)")
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args(argv)
    if a.digest and not a.remote:
        ap.error("--digest needs --remote")
    problems = check_local(None if a.no_dist else Path(a.dist), Path(a.host_file), Path(a.manifest), a.allow_local)
    print(f"local: {'OK' if not problems else str(len(problems)) + ' problem(s)'}")
    for p in problems:
        print(f"  PROBLEM {p}")
    if a.remote:
        host, errs = load_host(Path(a.host_file))
        if errs or PLACEHOLDER.search(host["base"]):
            print("remote: not run, the model host is not set up (see the local problems above)")
            return 1
        print(f"remote: asking {urlsplit(host['base']).netloc} with Origin {a.origin}")
        rep = check_remote(host, Path(a.manifest), a.origin, a.timeout, a.digest, a.allow_local)
        print("\n".join(rep.lines))
        print(f"remote: {'OK' if not rep.problems else str(len(rep.problems)) + ' problem(s)'}")
        for p in rep.problems:
            print(f"  PROBLEM {p}")
        problems += rep.problems
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
