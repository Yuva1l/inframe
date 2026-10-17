"""`inframe serve`: a loopback web server for the review UI (docs/T9A_DESIGN.md sections 2.3, 3.1 and 4).

This module is the core: `App` (shared state), `Media`, the router with the Host / Origin / fetch-metadata / token checks, the
response headers, static files (or the built-in "UI not built" page), the one-use nonce hand-over, the session, quit, the job
routes 46-48 and the loader for the route modules of the other parts (`serve_media`, `serve_pick`, `serve_runs`, `serve_export`,
`serve_music`; each exports `ROUTES` and `attach(app)`, and is skipped while it does not exist).
Route modules import `serve_http`, `serve_paths` and `serve_jobs`, never this module.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib
import importlib.util
import json
import re
import secrets
import shutil
import os
import sys
import tempfile
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__, models, serve_paths
from .probe import VideoInfo
from .serve_http import (CLIENT_GONE, BoundedServer, HttpError, Request, Response, Route, error, file_response, json_response,
                         read_body)
from .serve_jobs import JobRunner

ROUTE_MODULES = ("serve_media", "serve_pick", "serve_runs", "serve_export", "serve_music")
NONCE_TTL_S = 30.0
PUBLIC_API = frozenset({("POST", "/api/session/claim")})  # no session token (row 1a)
BODY_METHODS = ("POST", "PUT", "DELETE")

CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; "
       "connect-src 'self'; worker-src 'self' blob:; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
# The policy goes with pages and with scripts: a module Worker takes its CSP from its own response, not from the page that starts it.
CSP_TYPES = ("text/html", "text/javascript", "application/javascript")
COMMON_HEADERS = {
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
               ".ogv": "video/ogg"}
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8",
                ".json": "application/json", ".woff2": "font/woff2", ".woff": "font/woff", ".wasm": "application/wasm",
                ".ico": "image/x-icon", ".map": "application/json"}

FALLBACK_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>inframe</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0e0f11;color:#e7e8eb;font:16px/1.5 system-ui,sans-serif}
main{max-width:34rem;padding:2rem}code{background:#1e2024;padding:.15rem .4rem;border-radius:4px}</style></head>
<body><main><h1>inframe</h1><p>The web UI is not built: run <code>npm --prefix web ci &amp;&amp; npm --prefix web run build</code>.</p>
<p>The server itself is running; the API answers at <code>/api/session</code> for a client that holds the session token.</p></main></body></html>
"""


def _clock() -> float:
    return time.monotonic()


def media_id(path: Path, size: int, mtime_ns: int) -> str:
    return hashlib.sha1(f"{path}\0{size}\0{mtime_ns}".encode()).hexdigest()[:12]


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------- Media
@dataclass
class Media:
    id: str
    path: Path  # resolved
    size: int
    mtime_ns: int
    info: VideoInfo
    capture_fps: float
    cache_dir: Path
    picked_copy: bool = False

    def info_json(self) -> dict:
        """The MediaInfo of section 3.2 as far as the probe knows it. The media part (A2) fills in the proxy and filmstrip state."""
        from .cut import AUTO_MIN_CAPTURE_FPS

        i, (dw, dh) = self.info, self.info.display_size
        return {
            "id": self.id, "path": str(self.path), "name": self.path.name, "size": self.size, "mtime": self.mtime_ns / 1e9,
            "codec": i.codec, "profile": i.profile, "pix_fmt": i.pix_fmt, "width": i.width, "height": i.height,
            "display_w": dw, "display_h": dh, "rotation": i.rotation, "mirrored": i.mirrored, "fps": i.fps,
            "capture_fps": self.capture_fps, "slowmo_suggested": self.capture_fps >= AUTO_MIN_CAPTURE_FPS,
            "duration": i.duration, "offset": i.offset, "end_time": i.end_time, "has_audio": i.has_audio,
            "keyframes": [round(k, 3) for k in i.keyframes], "video_url": f"api/media/{self.id}/video",
            "proxy": {"state": "none", "fraction": 0.0, "url": None, "first_frame_file_time": None, "error": None},
            "filmstrip_state": "none", "picked_copy": self.picked_copy,
        }


# ---------------------------------------------------------------- App
class App:
    """What the route modules may read or set (docs/T9A_DESIGN.md 2.3): token, media_key, port, offline, cache_root, state_root,
    lock, media, jobs, photos_dirs, pickers, media_tasks, runs, stop(); methods open_media, media_or_404, run_or_404, out_root."""

    def __init__(self, offline: bool = False, cache_root: Path | None = None, state_root: Path | None = None,
                 dist_dir: Path | None = None, keep_picked: bool = False):
        self.token = secrets.token_urlsafe(16)
        self.media_key = secrets.token_urlsafe(16)
        self.port = 0
        self.offline = offline
        self.keep_picked = keep_picked
        self.cache_root = Path(cache_root) if cache_root else serve_paths.cache_root()
        self.state_root = Path(state_root) if state_root else serve_paths.state_root()
        self.dist_dir = Path(dist_dir) if dist_dir is not None else _default_dist()
        self.lock = threading.RLock()
        self.media: dict[str, Media] = {}
        self.jobs = JobRunner()
        self.photos_dirs: dict[str, Path] = {}
        self.pickers: dict[tuple[str, str, str], object] = {}
        self.media_tasks: dict = {}
        self.runs: dict[str, Path] = {}
        self.start_media: str | None = None
        self.routes: list[Route] = list(CORE_ROUTES)
        self.httpd: BoundedServer | None = None
        self.stopped = threading.Event()
        self._nonces: dict[str, float] = {}

    # ------------------------------------------------------------ secrets
    def new_nonce(self) -> str:
        """A one-use hand-over code that is valid for 30 s (the page trades it for the token and media key)."""
        with self.lock:
            now = _clock()
            for n in [n for n, exp in self._nonces.items() if exp <= now]:
                del self._nonces[n]
            n = secrets.token_urlsafe(16)
            self._nonces[n] = now + NONCE_TTL_S
            return n

    def claim(self, nonce) -> dict | None:
        if not isinstance(nonce, str):
            return None
        with self.lock:
            found = next((n for n in self._nonces if hmac.compare_digest(n.encode(), nonce.encode("utf-8", "replace"))), None)
            if found is None:
                return None
            exp = self._nonces.pop(found)
            if exp <= _clock():
                return None
            return {"token": self.token, "media_key": self.media_key}

    def token_ok(self, given: str | None) -> bool:
        return given is not None and hmac.compare_digest(given.encode("utf-8", "replace"), self.token.encode())

    def media_key_ok(self, given: str | None) -> bool:
        return given is not None and hmac.compare_digest(given.encode("utf-8", "replace"), self.media_key.encode())

    # ------------------------------------------------------------ media
    def open_media(self, path) -> Media:
        """Open a video: the path rules and the extension and magic check (serve_paths.check_video), ffprobe, register. The id is
        stable for the same path, size and mtime, so caches are reused."""
        from . import probe as probe_mod

        p = serve_paths.check_video(path)
        st = p.stat()
        mid = media_id(p, st.st_size, st.st_mtime_ns)
        with self.lock:
            if mid in self.media:
                return self.media[mid]
        try:
            info = probe_mod.probe(str(p))
            cap = probe_mod.capture_fps(str(p))
        except FileNotFoundError:
            raise HttpError(500, "ffprobe was not found: install ffmpeg") from None
        except Exception:  # CalledProcessError, ValueError (no video stream), KeyError, bad JSON
            raise HttpError(415, "This file has no video stream ffprobe can read.") from None
        if not info.duration > 0:
            raise HttpError(415, "This file has no video stream ffprobe can read.")
        cache = self.cache_root / mid
        try:
            cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            pass
        tmp = Path(tempfile.gettempdir()).resolve()
        # the helper exports into <tmp>/inframe-picked-X/<n>/IMG.mp4: any depth under a picker folder counts
        picked = any(a.parent == tmp and a.name.startswith(serve_paths.PICKED_PREFIX) for a in p.parents)
        m = Media(mid, p, st.st_size, st.st_mtime_ns, info, cap, cache, picked)
        with self.lock:
            return self.media.setdefault(mid, m)

    def media_or_404(self, mid: str) -> Media:
        with self.lock:
            m = self.media.get(mid)
        if m is None:
            raise HttpError(404, f"no such media: {mid}")
        return m

    def run_or_404(self, rid: str) -> Path:
        """The run folder of a registered run (part B1 fills `runs`)."""
        with self.lock:
            d = self.runs.get(rid)
        if d is None:
            raise HttpError(404, f"no such run: {rid}")
        return d

    def out_root(self, media: Media) -> Path:
        """The `<stem>_inframe` folder for this media: next to the video when that is possible, else under ~/Movies/inframe
        (~/inframe off macOS), which is also where Photos-picked copies go (their temp folder is deleted). An existing
        `<stem>_inframe` is always used."""
        name = f"{media.path.stem}_inframe"
        fallback = serve_paths.default_out_parent() / name
        if media.picked_copy:
            return fallback
        beside = media.path.parent / name
        if beside.is_dir():
            return beside
        return beside if os.access(media.path.parent, os.W_OK | os.X_OK) else fallback

    # ------------------------------------------------------------ session
    def session_json(self) -> dict:
        from . import classes, photosave, pipeline

        darwin = sys.platform == "darwin"
        caps = {"openDialog": darwin, "photosPicker": darwin, "fileBrowser": True, "locate": True, "reveal": True, "music": True,
                "beatSync": _has("librosa"), "savedTargets": True, "people": True, "proxy": True, "output": "server-folder",
                "streamCopyCut": True, "chooseOutputFolder": True, "savePhotos": photosave.available()}
        with self.lock:
            media = [m.info_json() for m in self.media.values()]
        return {"version": __version__, "engine": "local", "platform": sys.platform, "caps": caps,
                "presets": list(pipeline.PRESETS), "default_preset": "balanced", "classes": sorted(classes.COCO_91.values()),
                "defaults": asdict(pipeline.Settings()), "preflight": self.preflight(), "job": self.jobs.current(), "media": media,
                "start_media": self.start_media}

    def preflight(self) -> dict:
        """What the run needs, looked up without loading anything (section 1.5)."""
        from . import detect, pipeline

        have = set(detect.available_sizes())
        return {"ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
                "ffprobe": bool(shutil.which("ffprobe")),
                "models": {name: p.model in have for name, p in pipeline.PRESETS.items()},
                "identity": _has("torch") and _has("transformers"), "tracking": _has("trackers"), "music": _has("librosa")}

    # ------------------------------------------------------------ lifecycle
    def stop(self) -> None:
        """Stop serving (callable from a request handler): the serve loop returns and `run` cleans up."""
        self.stopped.set()
        h = self.httpd
        if h is not None:
            threading.Thread(target=h.shutdown, name="inframe-serve-stop", daemon=True).start()

    def close(self) -> None:
        """Release what the app holds: the running job is cancelled, and the picker folders this process made are removed
        unless --keep-picked."""
        self.jobs.shutdown()
        for p in list(self.pickers.values()):
            close = getattr(p, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
        if not self.keep_picked:
            serve_paths.cleanup_own_picked()


def _default_dist() -> Path | None:
    try:
        from importlib import resources

        return Path(str(resources.files("inframe") / "web" / "dist"))
    except (ModuleNotFoundError, TypeError, OSError):
        return None


# ---------------------------------------------------------------- routes of this module
def session_claim(app: App, req: Request) -> Response:
    got = app.claim((req.json or {}).get("nonce"))
    if got is None:
        return error(403, "This link was used or has expired. Press Enter in the terminal for a new one.")
    return json_response(got)


def session_get(app: App, req: Request) -> Response:
    return json_response(app.session_json())


def quit_(app: App, req: Request) -> Response:
    r = json_response({"ok": True})
    r.after = app.stop
    return r


def jobs_current(app: App, req: Request) -> Response:
    return json_response({"job": app.jobs.current()})


def jobs_get(app: App, req: Request) -> Response:
    return json_response(app.jobs.get(req.params["jid"]))


def jobs_cancel(app: App, req: Request) -> Response:
    return json_response(app.jobs.cancel(req.params["jid"]))


JID = r"(?P<jid>[A-Za-z0-9_-]{1,32})"
CORE_ROUTES = [
    Route("POST", r"/api/session/claim", session_claim),
    Route("GET", r"/api/session", session_get),
    Route("POST", r"/api/quit", quit_),
    Route("GET", r"/api/jobs/current", jobs_current),
    Route("GET", rf"/api/jobs/{JID}", jobs_get),
    Route("POST", rf"/api/jobs/{JID}/cancel", jobs_cancel),
]


def load_route_modules(app: App) -> list[str]:
    """Import the route modules of the other parts in order; one that does not exist is skipped (a ModuleNotFoundError that names
    exactly that module), any other import error is raised. Returns the names loaded."""
    loaded = []
    for name in ROUTE_MODULES:
        full = f"{__package__}.{name}"
        try:
            mod = importlib.import_module(full)
        except ModuleNotFoundError as e:
            if e.name == full:
                continue
            raise
        app.routes.extend(mod.ROUTES)
        mod.attach(app)
        loaded.append(name)
    return loaded


# ---------------------------------------------------------------- static files
def serve_static(app: App, path: str, req: Request) -> Response:
    dist = app.dist_dir
    index = dist / "index.html" if dist is not None else None
    built = bool(index is not None and index.is_file() and not index.is_symlink())
    if path in ("/", "/index.html"):
        if not built:
            return Response(200, FALLBACK_PAGE.encode(), "text/html; charset=utf-8")
        return file_response(index, STATIC_TYPES[".html"], req)
    if not built:
        return error(404, "not found")
    rel = path.lstrip("/")
    names = rel.split("/")
    if any(n in ("", ".", "..") or n[0] in "._" or "\0" in n or "\\" in n for n in names):
        return error(404, "not found")
    root = dist.resolve()
    f = root.joinpath(*names)
    try:
        r = f.resolve(strict=True)
        if not r.is_relative_to(root) or f.is_symlink() or not r.is_file():
            return error(404, "not found")
    except OSError:
        return error(404, "not found")
    return file_response(r, STATIC_TYPES.get(r.suffix.lower(), "application/octet-stream"), req)


# ---------------------------------------------------------------- browser-mode models (docs/T9B_DESIGN.md section 3)
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")
MODEL_TYPES = {".onnx": "application/octet-stream", ".wasm": "application/wasm", ".json": "application/json"}


BENCH_MODELS = Path(__file__).resolve().parents[2] / "bench/detect/out/models"  # as in detect.py


def onnx_dir() -> Path | None:
    """The `onnx/` folder tools/export_onnx.py writes: `$INFRAME_MODELS/onnx` when the variable is set (only there), otherwise the
    first of the models cache and <repo>/bench/detect/out/models that holds onnx/manifest.json (detect.py's look-up order without the
    current folder), so an installed `inframe serve` finds the models the CLI finds. None when there is none."""
    base = os.environ.get("INFRAME_MODELS")
    bases = [Path(base).expanduser()] if base else [models.cache_dir(), BENCH_MODELS]
    for b in bases:
        d = b / "onnx"
        if d.is_dir() and not d.is_symlink() and (base or (d / "manifest.json").is_file()):
            return d
    return None


def model_names(d: Path) -> set[str]:
    """The names GET /models/ may serve: manifest.json and the files its manifest lists. Nothing else in the folder is reachable."""
    try:
        listed = json.loads((d / "manifest.json").read_text())["files"]
        names = {n for n in listed if isinstance(n, str) and MODEL_NAME.fullmatch(n)}
    except (OSError, ValueError, KeyError, TypeError):
        return set()
    return names | {"manifest.json"}


def serve_models(path: str, req: Request) -> Response:
    """GET /models/<manifest name>: a regular file of the models folder, read-only, Range-aware. Any other path is 404."""
    name = path[len("/models/"):]
    d = onnx_dir()
    if d is None or not MODEL_NAME.fullmatch(name) or name not in model_names(d):
        return error(404, "not found")
    f = d / name
    try:
        if f.is_symlink() or not f.is_file():
            return error(404, "not found")
    except OSError:
        return error(404, "not found")
    r = file_response(f, MODEL_TYPES.get(f.suffix.lower(), "application/octet-stream"), req)
    r.headers["Cache-Control"] = "no-cache"  # the page checks the sha256 of what it downloads; a changed file is fetched again
    return r


# ---------------------------------------------------------------- the handler
def make_handler(app: App):
    from . import serve_http

    timeout = serve_http.SOCKET_TIMEOUT

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "inframe"
        sys_version = ""

        def log_message(self, *a):
            pass

        def handle(self):  # a client that goes away (a cancelled fetch, a closed tab) is not an error
            try:
                super().handle()
            except CLIENT_GONE:
                self.close_connection = True

        # every verb goes through the same checks; unknown verbs get the stdlib's 501
        def do_GET(self): self._serve()
        def do_HEAD(self): self._serve()
        def do_POST(self): self._serve()
        def do_PUT(self): self._serve()
        def do_DELETE(self): self._serve()
        def do_OPTIONS(self): self._serve()
        def do_PATCH(self): self._serve()

        def send_error(self, code, message=None, explain=None):  # malformed requests answer in the same JSON shape and headers
            self.close_connection = True
            self._write(error(code, message or "bad request"), "")

        # ------------------------------------------------------------ one request
        def _serve(self):
            self._body_read = False
            path = unquote(urlsplit(self.path).path)
            try:
                resp, path = self._route()
            except HttpError as e:
                resp = error(e.code, e.message, **e.extra)
            except CLIENT_GONE:
                raise
            except Exception as e:
                resp = error(500, f"{type(e).__name__}: {e}")
            self._write(resp, path)

        def _checks(self, method: str) -> None:
            h = self.headers
            port = app.port
            hosts = h.get_all("Host") or []
            if len(hosts) != 1 or hosts[0] != f"127.0.0.1:{port}":
                raise HttpError(403, "forbidden host")
            origins = h.get_all("Origin") or []
            if len(origins) > 1:
                raise HttpError(403, "forbidden origin")
            if origins:
                if origins[0] != f"http://127.0.0.1:{port}":
                    raise HttpError(403, "forbidden origin")
            elif method in BODY_METHODS:
                raise HttpError(403, "missing origin")
            fetch = h.get("Sec-Fetch-Site")
            if fetch is not None and fetch not in ("same-origin", "none"):
                raise HttpError(403, "cross-site request")

        def _route(self) -> tuple[Response, str]:
            method = self.command
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            if method == "OPTIONS":
                raise HttpError(403, "forbidden")
            self._checks(method)
            query = {k: v[0] for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
            hdr = {k.lower(): v for k, v in self.headers.items()}
            chosen, params, path_known = None, {}, False
            is_api = path == "/api" or path.startswith("/api/")
            if is_api:
                lookup = "GET" if method == "HEAD" else method
                for r in app.routes:
                    m = r.match(path)
                    if m is None:
                        continue
                    path_known = True
                    if r.method == lookup:
                        chosen, params = r, m
                        break
                public = (method, path) in PUBLIC_API
                if not public and not app.token_ok(hdr.get("x-inframe-token")):
                    if not (chosen is not None and chosen.media and method in ("GET", "HEAD") and app.media_key_ok(query.get("m"))):
                        raise HttpError(403, "This page belongs to an older inframe serve. Open the address the terminal printed.")
            body = self._body()
            req = Request(method, path, params, query, hdr, None, body)
            if method in BODY_METHODS:
                if body:
                    try:
                        data = json.loads(body.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        raise HttpError(400, "body is not valid JSON") from None
                    if not isinstance(data, dict):
                        raise HttpError(400, "body must be a JSON object")
                    req.json = data
                else:
                    req.json = {}
            if not is_api:
                if method not in ("GET", "HEAD"):
                    raise HttpError(405, "method not allowed")
                if path.startswith("/models/"):
                    return serve_models(path, req), path
                return serve_static(app, path, req), path
            if chosen is None:
                raise HttpError(405 if path_known else 404, "method not allowed" if path_known else "not found")
            resp = chosen.handler(app, req)
            if not isinstance(resp, Response):
                raise TypeError(f"handler for {path} returned {type(resp).__name__}, not a Response")
            return resp, path

        def _body(self) -> bytes:
            try:
                data = read_body(self)
            except HttpError:
                self.close_connection = True  # the body was not (fully) read: this connection cannot carry another request
                raise
            self._body_read = True
            return data

        # ------------------------------------------------------------ the answer
        def _write(self, resp: Response, path: str) -> None:
            headers = getattr(self, "headers", None)
            if headers is not None and not getattr(self, "_body_read", False) and (
                    headers.get("Content-Length", "0") != "0" or headers.get("Transfer-Encoding") is not None):
                self.close_connection = True  # a body that was never read is still in the pipe
            f, offset, length = None, 0, len(resp.body)
            if resp.file is not None:
                try:
                    f = open(resp.file[0], "rb")
                    offset, length = resp.file[1], resp.file[2]
                except OSError:
                    resp = error(404, "no such file")
                    length = len(resp.body)
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.content_type)
            self.send_header("Content-Length", str(length))
            for k, v in resp.headers.items():
                self.send_header(k, v)
            for k, v in COMMON_HEADERS.items():
                self.send_header(k, v)
            if path == "/api" or path.startswith("/api/"):
                self.send_header("Cache-Control", "no-store")
            if resp.content_type.startswith(CSP_TYPES):
                self.send_header("Content-Security-Policy", CSP)
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            try:
                if self.command != "HEAD":
                    if f is not None:
                        serve_http.copy_file(self.connection, f, offset, length)
                    elif resp.body:
                        self.wfile.write(resp.body)
            finally:
                if f is not None:
                    f.close()
            if resp.after is not None:
                resp.after()

    Handler.timeout = timeout
    return Handler


# ---------------------------------------------------------------- start
def create(port: int = 0, extra_routes=(), max_threads: int | None = None, **app_kw) -> tuple[BoundedServer, App]:
    """The server (bound to 127.0.0.1, not yet serving) and its App. `app_kw` goes to `App`."""
    app = App(**app_kw)
    httpd = BoundedServer(("127.0.0.1", port), make_handler(app), max_threads=max_threads)
    app.httpd = httpd
    app.port = httpd.server_address[1]
    load_route_modules(app)
    app.routes.extend(extra_routes)
    return httpd, app


def run(video: str | None = None, port: int = 0, open_browser: bool = True, offline: bool = False, keep_picked: bool = False,
        announce=print) -> int:
    """`inframe serve`: start the server, print (and open) the one-use link, serve until quit or Ctrl-C. Returns the exit code."""
    models.set_offline(offline)
    try:
        serve_paths.sweep_picked()
    except OSError:
        pass
    httpd, app = create(port, offline=offline, keep_picked=keep_picked)
    try:
        if video:
            try:
                app.start_media = app.open_media(video).id
            except HttpError as e:
                announce(f"inframe: {e.message}")
                return 2
        url = f"http://127.0.0.1:{app.port}/#n={app.new_nonce()}"
        announce(f"inframe serve: open {url}  (the link works once, for 30 seconds; Ctrl-C stops the server)")
        if open_browser:
            try:
                if not webbrowser.open(url):
                    announce("inframe serve: could not open a browser; open the address above yourself")
            except Exception:
                announce("inframe serve: could not open a browser; open the address above yourself")
        _watch_stdin(app, announce)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    finally:
        httpd.server_close()
        app.close()


def _watch_stdin(app: App, announce) -> None:
    """Press Enter for a new one-use link (a new browser tab needs one). Without a terminal that needs a restart."""
    try:
        tty = bool(sys.stdin and sys.stdin.isatty())
    except (ValueError, OSError):
        tty = False
    if not tty:
        announce("inframe serve: no terminal here; a new link needs a restart")
        return

    def loop():
        announce("inframe serve: press Enter for a new link")
        try:
            for _ in sys.stdin:
                announce(f"inframe serve: open http://127.0.0.1:{app.port}/#n={app.new_nonce()}")
        except (ValueError, OSError):
            pass

    threading.Thread(target=loop, name="inframe-serve-stdin", daemon=True).start()
