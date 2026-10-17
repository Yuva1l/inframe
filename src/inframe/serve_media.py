"""Media routes of `inframe serve` (docs/T9A_DESIGN.md sections 2.4, 2.5, 2.9 and 3.2, rows 4-18): open a video (path, native
dialog, Photos picker, a dropped file's name/size/mtime), the file browser, the recent list, Range-served original and proxy,
filmstrip sheets and single frames.

Two media tasks run in threads of the server process, one proxy at a time: the filmstrip (PyAV keyframes -> JPEG sheets) and the
proxy (an ffmpeg child, H.264 where the build has an encoder). Their state lives in `app.media_tasks`; `attach(app)` makes sure
`app.close()` stops them. Imports `serve_http`, `serve_paths` and core modules, never `serve.py`.
"""
from __future__ import annotations

import functools
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from . import serve_paths
from .serve_http import HttpError, Request, Response, Route, bytes_response, error, file_response, json_response

# ---------------------------------------------------------------- constants
CELL_H = 108  # filmstrip cell height; the width follows the display aspect, even, at most CELL_MAX_W
CELL_MAX_W = 192
COLS = ROWS = 10  # cells per sheet side
SLOTS_MAX = 600  # one cell per slot of max(1 s, duration / SLOTS_MAX)
SHEET_QUALITY = 70
FRAME_QUALITY = 85
FRAME_LRU = 32
FRAME_W_DEFAULT, FRAME_W_MAX, FRAME_W_MIN = 960, 1280, 32
PROXY_SHORT_SIDE = 720
PROXY_KEEP = 3  # proxies kept in the cache: the newest ones (and any that a player in this server run has fetched)
FILM_KEEP = 10  # filmstrip sheet sets kept in the cache: the newest ones (and those of media open in this server run)
PROXY_ENCODERS = ("h264_videotoolbox", "libx264", "mpeg4")
DIALOG_TIMEOUT_S = 600
RECENT_MAX = 20
LOCATE_DEPTH = 2  # folders below a root that locate looks into
LOCATE_MAX_ENTRIES = 50_000
LOCATE_MTIME_TOL_MS = 2.0  # a browser's lastModified is whole milliseconds
LOCATE_MISS = "Browsers hide where a dropped file lives. Use Open file... instead."
MEDIA_ID = r"(?P<mid>[0-9a-f]{12})"
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
               ".ogv": "video/ogg"}


def _darwin() -> bool:
    return sys.platform == "darwin"


# ---------------------------------------------------------------- task state in app.media_tasks
@dataclass
class Task:
    kind: str  # "filmstrip" | "proxy"
    state: str = "running"  # running | done | failed
    fraction: float = 0.0
    error: str | None = None
    first_frame_file_time: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    proc: subprocess.Popen | None = None
    thread: threading.Thread | None = None


def _tasks(app) -> dict:
    """`app.media_tasks` (a dict A1 creates): ("filmstrip"|"proxy", media id) -> Task, plus the frame LRU and the filmstrip slot."""
    mt = app.media_tasks
    with app.lock:
        mt.setdefault("frames", OrderedDict())
        mt.setdefault("frames_lock", threading.Lock())
        mt.setdefault("film_slot", threading.Semaphore(1))  # one filmstrip decode at a time
    return mt


def attach(app) -> None:
    """Called by serve.py after the routes are added: set up the task registry and make `app.close()` stop what runs here."""
    _tasks(app)
    orig = app.close

    def close():
        stop_tasks(app)
        orig()

    app.close = close


def stop_tasks(app) -> None:
    """Cancel every media task, kill the ffmpeg child and wait briefly for the threads."""
    with app.lock:
        tasks = [t for k, t in app.media_tasks.items() if isinstance(t, Task)]
    for t in tasks:
        t.cancel.set()
        p = t.proc
        if p is not None and p.poll() is None:
            p.kill()
    for t in tasks:
        if t.thread is not None and t.thread is not threading.current_thread():
            t.thread.join(5)


# ---------------------------------------------------------------- recent list (serve state folder)
def _recent_file(app) -> Path:
    return Path(app.state_root) / "recent.json"


def recent_load(app) -> list[dict]:
    try:
        doc = json.loads(_recent_file(app).read_text())
        items = doc["recent"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and isinstance(it.get("path"), str) and isinstance(it.get("opened"), (int, float)):
            out.append({"path": it["path"], "name": os.path.basename(it["path"]), "opened": float(it["opened"])})
    return out


def _recent_save(app, items: list[dict]) -> None:
    f = _recent_file(app)
    try:
        f.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = f.with_name(f.name + ".part")
        tmp.write_text(json.dumps({"version": 1, "recent": items}))
        os.replace(tmp, f)
    except OSError:
        pass  # a recent list that cannot be saved is not worth failing an open for


def recent_add(app, media) -> None:
    if media.picked_copy:  # its temp folder goes away when the server stops
        return
    with app.lock:
        path = str(media.path)
        items = [i for i in recent_load(app) if i["path"] != path]
        items.insert(0, {"path": path, "name": media.path.name, "opened": round(time.time(), 3)})
        _recent_save(app, items[:RECENT_MAX])


def recent_get(app, req: Request) -> Response:
    with app.lock:
        items = [i for i in recent_load(app) if os.path.isfile(i["path"])]
    return json_response({"recent": items})


def recent_clear(app, req: Request) -> Response:
    with app.lock:
        _recent_save(app, [])
    return json_response({"recent": []})


# ---------------------------------------------------------------- MediaInfo (3.2)
def _proxy_files(media) -> tuple[Path, Path]:
    return media.cache_dir / "proxy.mp4", media.cache_dir / "proxy.json"


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def proxy_state(app, media) -> dict:
    """{"state": none|running|done|failed, "fraction", "url", "first_frame_file_time", "error"}: from the running or failed
    task, else from `proxy.json` on disk (a proxy made by an earlier server run)."""
    url = f"api/media/{media.id}/proxy.mp4"
    with app.lock:
        t = _tasks(app).get(("proxy", media.id))
    if t is not None and t.state in ("running", "failed"):
        return {"state": t.state, "fraction": round(t.fraction, 3), "url": None, "first_frame_file_time": None, "error": t.error}
    mp4, js = _proxy_files(media)
    doc = _read_json(js)
    if isinstance(doc, dict) and doc.get("state") == "done" and mp4.is_file():
        return {"state": "done", "fraction": 1.0, "url": url, "first_frame_file_time": doc.get("first_frame_file_time"), "error": None}
    return {"state": "none", "fraction": 0.0, "url": None, "first_frame_file_time": None, "error": None}


def filmstrip_state(app, media) -> str:
    with app.lock:
        t = _tasks(app).get(("filmstrip", media.id))
    if t is not None and t.state in ("running", "failed"):
        return t.state
    doc = _read_json(media.cache_dir / "filmstrip.json")
    return "done" if isinstance(doc, dict) and doc.get("done") else "none"


def media_info(app, media) -> dict:
    """The MediaInfo of 3.2: `Media.info_json()` with the proxy and filmstrip state filled in."""
    d = media.info_json()
    d["proxy"] = proxy_state(app, media)
    d["filmstrip_state"] = filmstrip_state(app, media)
    return d


def _media(app, req: Request):
    return app.media_or_404(req.params["mid"])


# ---------------------------------------------------------------- rows 4-7: opening
def _opened(app, media) -> Response:
    recent_add(app, media)
    ensure_filmstrip(app, media, retry=True)
    return json_response(media_info(app, media))


def _need_str(body: dict, key: str) -> str:
    v = body.get(key)
    if not isinstance(v, str) or not v:
        raise HttpError(400, f"{key} must be a non-empty string")
    return v


def media_open(app, req: Request) -> Response:
    return _opened(app, app.open_media(_need_str(req.json or {}, "path")))


def run_osascript(script: str) -> subprocess.CompletedProcess:
    """The one place a native dialog is started (tests replace it). Argument list, no shell."""
    return subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=DIALOG_TIMEOUT_S)


OPEN_VIDEO_SCRIPT = 'POSIX path of (choose file with prompt "Open a video" of type {"public.movie"})'


def media_open_dialog(app, req: Request) -> Response:
    kind = (req.json or {}).get("kind", "video")
    if kind != "video":
        raise HttpError(400, "kind must be 'video'")
    if not _darwin():
        raise HttpError(501, "The open dialog is only available on macOS. Use Browse... instead.")
    try:
        r = run_osascript(OPEN_VIDEO_SCRIPT)
    except subprocess.TimeoutExpired:
        raise HttpError(500, "The open dialog timed out.") from None
    except OSError as e:
        raise HttpError(500, f"Could not start osascript: {e}") from None
    if r.returncode != 0:
        if "User canceled" in (r.stderr or ""):
            return json_response({"cancelled": True})
        raise HttpError(500, f"osascript failed: {(r.stderr or '').strip()[:300]}")
    path = (r.stdout or "").strip()
    if not path:
        return json_response({"cancelled": True})
    return _opened(app, app.open_media(path))


def media_photos_pick(app, req: Request) -> Response:
    if not _darwin():
        raise HttpError(501, "The Photos picker is only available on macOS. Use Browse... instead.")
    from . import picker

    out = serve_paths.new_picked_dir()
    try:
        res = picker.pick(out, limit=1)
    except picker.PickerError as e:
        raise HttpError(500, str(e)) from None
    if res.cancelled or not res.files:
        shutil.rmtree(out, ignore_errors=True)
        return json_response({"cancelled": True})
    return _opened(app, app.open_media(res.files[0]))


# ---- locate: a dropped file has a name, size and mtime but no path
def locate_roots(app) -> list[Path]:
    """The recent files' folders, then ~/Movies ~/Downloads ~/Desktop ~/Documents (tests replace this)."""
    home = Path.home()
    folders = [Path(i["path"]).parent for i in recent_load(app)]
    return folders + [home / d for d in ("Movies", "Downloads", "Desktop", "Documents")]


def locate_matches(roots, name: str, size: int, mtime_ms: float) -> list[Path]:
    """Files called `name` with that size and mtime in the roots, `LOCATE_DEPTH` folders deep, by `stat` only. Hidden folders are
    skipped, symlinks are not followed, each real path counts once."""
    found: dict[str, Path] = {}
    seen_dirs: set[str] = set()
    budget = LOCATE_MAX_ENTRIES

    def walk(d: Path, depth: int) -> None:
        nonlocal budget
        try:
            real = os.path.realpath(d)
            if real in seen_dirs:
                return
            seen_dirs.add(real)
            it = os.scandir(d)
        except OSError:
            return
        with it:
            for e in it:
                budget -= 1
                if budget < 0:
                    return
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir(follow_symlinks=False):
                        if depth < LOCATE_DEPTH:
                            walk(Path(e.path), depth + 1)
                    elif e.name == name and e.is_file(follow_symlinks=False):
                        st = e.stat(follow_symlinks=False)
                        if st.st_size == size and abs(st.st_mtime_ns / 1e6 - mtime_ms) <= LOCATE_MTIME_TOL_MS:
                            p = Path(e.path)
                            found[os.path.realpath(p)] = p
                except OSError:
                    continue

    for r in roots:
        walk(Path(r), 0)
    return list(found.values())


def _num(body: dict, key: str, integer: bool = False) -> float:
    v = body.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise HttpError(400, f"{key} must be a non-negative number")
    if integer and int(v) != v:
        raise HttpError(400, f"{key} must be a whole number")
    return v


def media_locate(app, req: Request) -> Response:
    body = req.json or {}
    name = _need_str(body, "name")
    if "/" in name or "\\" in name or "\0" in name or name != os.path.basename(name):
        raise HttpError(400, "name must be a file name")
    size = int(_num(body, "size", integer=True))
    mtime_ms = float(_num(body, "last_modified_ms"))
    matches = locate_matches(locate_roots(app), name, size, mtime_ms)
    if len(matches) != 1:
        return error(404, LOCATE_MISS, matches=len(matches))
    return _opened(app, app.open_media(matches[0]))


# ---------------------------------------------------------------- row 8: file browser
def fs_roots() -> list[dict]:
    """The folders the browser may show: the home folder and each mounted volume in /Volumes (tests replace this)."""
    roots = [{"name": "Home", "path": str(Path.home().resolve())}]
    try:
        with os.scandir("/Volumes") as it:
            for e in sorted(it, key=lambda e: e.name.lower()):
                if not e.name.startswith(".") and e.is_dir(follow_symlinks=False):  # not the boot volume's symlink to /
                    roots.append({"name": e.name, "path": os.path.realpath(e.path)})
    except OSError:
        pass
    return roots


FS_MODES = {"video": serve_paths.VIDEO_EXTS, "audio": serve_paths.AUDIO_EXTS, "folder": ()}


def fs_list(app, req: Request) -> Response:
    mode = req.query.get("mode", "video")
    if mode not in FS_MODES:
        raise HttpError(400, "mode must be video, audio or folder")
    roots = fs_roots()
    if not roots:
        raise HttpError(404, "no folders to browse")
    want = req.query.get("dir") or roots[0]["path"]
    d = serve_paths.check_read(want, "dir")  # absolute, resolved, not hidden, not under ~/Library
    root = next((Path(r["path"]).resolve() for r in roots if d.is_relative_to(Path(r["path"]).resolve())), None)
    if root is None:
        raise HttpError(403, "inframe serve does not open files there.")
    library = (Path.home() / "Library").resolve()
    exts = FS_MODES[mode]
    entries = []
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir(follow_symlinks=False):
                        if Path(e.path).resolve() == library:
                            continue
                        st = e.stat(follow_symlinks=False)
                        entries.append({"name": e.name, "kind": "dir", "size": None, "mtime": round(st.st_mtime, 3)})
                    elif exts and e.is_file(follow_symlinks=False) and os.path.splitext(e.name)[1].lower() in exts:
                        st = e.stat(follow_symlinks=False)
                        entries.append({"name": e.name, "kind": mode, "size": st.st_size, "mtime": round(st.st_mtime, 3)})
                except OSError:
                    continue
    except OSError:
        raise HttpError(403, f"Cannot read {d}") from None
    entries.sort(key=lambda x: (x["kind"] != "dir", x["name"].casefold()))
    parent = str(d.parent) if d != root else None
    return json_response({"dir": str(d), "parent": parent, "roots": roots, "entries": entries})


# ---------------------------------------------------------------- rows 11, 12: info and the original
def media_get(app, req: Request) -> Response:
    m = _media(app, req)
    ensure_filmstrip(app, m)
    return json_response(media_info(app, m))


def media_video(app, req: Request) -> Response:
    m = _media(app, req)
    ctype = VIDEO_TYPES.get(m.path.suffix.lower(), "application/octet-stream")
    as_ = req.query.get("as")
    if as_ is not None:
        if as_ != "mp4":
            raise HttpError(400, "as must be mp4")
        if m.path.suffix.lower() == ".mov":  # the same ISO-BMFF family: a fallback for a browser that refuses video/quicktime
            ctype = "video/mp4"
    return file_response(m.path, ctype, req)


# ---------------------------------------------------------------- proxy (2.4)
_ENC_LINE = re.compile(r"^\s*V\S*\s+(\S+)")


@functools.lru_cache(maxsize=1)
def _ffmpeg_encoders() -> frozenset[str]:
    """The video encoders of this ffmpeg build (`ffmpeg -hide_banner -encoders`, read once)."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    return frozenset(m.group(1) for line in r.stdout.splitlines() if (m := _ENC_LINE.match(line)))


def proxy_encoder() -> str:
    """`h264_videotoolbox`, else `libx264`, else `mpeg4` (always in ffmpeg builds); $INFRAME_PROXY_ENCODER overrides it."""
    env = os.environ.get("INFRAME_PROXY_ENCODER")
    if env:
        if env not in PROXY_ENCODERS:
            raise ValueError(f"INFRAME_PROXY_ENCODER must be one of {', '.join(PROXY_ENCODERS)}, not {env!r}")
        return env
    have = _ffmpeg_encoders()
    return next((e for e in PROXY_ENCODERS if e in have), "mpeg4")


def _encoder_args(enc: str) -> list[str]:
    if enc == "h264_videotoolbox":
        return ["-c:v", enc, "-b:v", "4M"]
    if enc == "libx264":
        return ["-c:v", enc, "-preset", "veryfast", "-crf", "26"]
    return ["-c:v", "mpeg4", "-q:v", "5"]


# 720-px short side, never upscaled, even sizes; ffmpeg's autorotate has already made the picture upright when this runs
PROXY_VF = (f"scale=w='if(gte(iw,ih),-2,trunc(min({PROXY_SHORT_SIDE},iw)/2)*2)':h='if(gte(iw,ih),trunc(min({PROXY_SHORT_SIDE},ih)/2)*2,-2)',"
            "format=yuv420p")


def proxy_command(src: str, out: str, enc: str) -> list[str]:
    return (["ffmpeg", "-nostdin", "-nostats", "-loglevel", "error", "-y", "-protocol_whitelist", "file,pipe", "-i", src,
             "-map", "0:v:0", "-map", "0:a:0?", "-vf", PROXY_VF] + _encoder_args(enc) +
            ["-g", "15", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-f", "mp4", "-progress", "pipe:1", out])


class TaskError(RuntimeError):
    """A media task (proxy or filmstrip) that stopped: cancelled, ffmpeg failed, unreadable output."""


def make_proxy(media, task: Task) -> dict:
    """Encode `<cache>/proxy.mp4` (through `proxy.mp4.part`) and write `proxy.json`; progress goes to `task.fraction`. Returns the
    proxy.json document. `first_frame_file_time` is the file time of the proxy's first picture: the encode keeps ffmpeg's timeline
    (the source container's start is subtracted), so it is the first video packet's pts in the proxy, measured with ffprobe."""
    from . import probe

    try:
        enc = proxy_encoder()
    except ValueError as e:
        raise TaskError(str(e)) from None
    mp4, js = _proxy_files(media)
    part = mp4.with_name(mp4.name + ".part")
    mp4.parent.mkdir(parents=True, exist_ok=True)
    for f in (js, part):
        f.unlink(missing_ok=True)
    cmd = proxy_command(str(media.path), str(part), enc)
    span = max(media.info.end_time, media.info.duration, 1e-6)  # ffmpeg's out_time counts from the container start
    err = tempfile.TemporaryFile()
    try:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, stdin=subprocess.DEVNULL)
        except FileNotFoundError:
            raise TaskError("ffmpeg was not found: install ffmpeg") from None
        task.proc = proc
        for line in proc.stdout:
            k, _, v = line.strip().partition("=")
            if k == "out_time_us" and v.lstrip("-").isdigit():
                task.fraction = min(0.99, max(task.fraction, int(v) / 1e6 / span))
        rc = proc.wait()
        if task.cancel.is_set():
            raise TaskError("cancelled")
        if rc != 0:
            err.seek(0)
            msg = err.read().decode("utf-8", "replace").strip().splitlines()
            raise TaskError(f"ffmpeg failed ({enc}): {msg[-1][:300] if msg else 'exit ' + str(rc)}")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        err.close()
    try:
        v_pts, _ = probe.first_packet_pts(str(part), what="the proxy being written")
        out_info = probe.probe(str(part), with_keyframes=False, what="the proxy being written")
    except Exception as e:
        part.unlink(missing_ok=True)
        raise TaskError(f"the proxy could not be read back: {e}") from None
    os.replace(part, mp4)
    doc = {"version": 1, "state": "done", "encoder": enc, "first_frame_file_time": round(v_pts - out_info.format_start_time, 6),
           "source_offset": round(media.info.offset, 6), "duration": round(out_info.duration, 6),
           "width": out_info.width, "height": out_info.height}
    tmp = js.with_name(js.name + ".part")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, js)
    return doc


def prune_proxies(cache_root: Path, keep: int = PROXY_KEEP, protect=()) -> list[Path]:
    """Delete the proxies of the older media beyond the `keep` newest (by mtime), except those of the media ids in `protect`
    (a proxy a player is reading). Returns the removed proxy.mp4 paths."""
    found = []
    protect = set(protect)
    try:
        for d in os.scandir(cache_root):
            p = Path(d.path) / "proxy.mp4"
            try:
                if d.name not in protect and d.is_dir(follow_symlinks=False) and p.is_file():
                    found.append((p.stat().st_mtime, p))
            except OSError:
                continue
    except OSError:
        return []
    found.sort(key=lambda x: x[0], reverse=True)
    removed = []
    for _, p in found[keep:]:
        p.unlink(missing_ok=True)
        p.with_name("proxy.json").unlink(missing_ok=True)
        removed.append(p)
    return removed


def _mark_used(app, media) -> None:
    """Remember that a player fetched this media's proxy: pruning leaves it alone for the rest of this server run."""
    mt = _tasks(app)
    with app.lock:
        mt.setdefault("proxy_used", set()).add(media.id)


def prune_proxies_for(app) -> list[Path]:
    with app.lock:
        used = set(_tasks(app).get("proxy_used", ()))
    return prune_proxies(app.cache_root, protect=used)


def prune_filmstrips(cache_root: Path, keep: int | None = None, protect=()) -> list[Path]:
    """Delete the filmstrip sheets (JPEG thumbnails of the footage) of the older media beyond the `keep` newest by `filmstrip.json`
    mtime, except those of the media ids in `protect`. `filmstrip.json` goes first, so a half-removed set never reads as done and
    is made again on the next open. Returns the removed `filmstrip` folders."""
    keep = FILM_KEEP if keep is None else keep
    found = []
    protect = set(protect)
    try:
        for d in os.scandir(cache_root):
            base = Path(d.path)
            try:
                if d.name not in protect and d.is_dir(follow_symlinks=False) and (base / "filmstrip").is_dir():
                    jp = base / "filmstrip.json"
                    found.append((jp.stat().st_mtime if jp.is_file() else 0.0, base))
            except OSError:
                continue
    except OSError:
        return []
    found.sort(key=lambda x: x[0], reverse=True)
    removed = []
    for _, base in found[keep:]:
        try:
            (base / "filmstrip.json").unlink(missing_ok=True)
        except OSError:
            continue
        shutil.rmtree(base / "filmstrip", ignore_errors=True)
        removed.append(base / "filmstrip")
    return removed


def prune_filmstrips_for(app) -> list[Path]:
    with app.lock:
        open_ids = set(app.media)
    return prune_filmstrips(app.cache_root, protect=open_ids)


def _proxy_thread(app, media, task: Task) -> None:
    try:
        doc = make_proxy(media, task)
        task.first_frame_file_time = doc["first_frame_file_time"]
        task.fraction, task.state = 1.0, "done"
        prune_proxies_for(app)
    except TaskError as e:
        task.error, task.state = str(e), "failed"
    except Exception as e:
        task.error, task.state = f"{type(e).__name__}: {e}", "failed"


def proxy_post(app, req: Request) -> Response:
    m = _media(app, req)
    with app.lock:
        tasks = _tasks(app)
        mine = tasks.get(("proxy", m.id))
        if mine is not None and mine.state == "running":
            return json_response(_proxy_brief(proxy_state(app, m)))
        other = next((k[1] for k, t in tasks.items() if isinstance(t, Task) and k[0] == "proxy" and t.state == "running"), None)
        if other is not None:
            return error(409, "Another preview copy is being made. Wait for it to finish.", media_id=other)
        st = proxy_state(app, m)
        if st["state"] == "done":
            return json_response(_proxy_brief(st))
        task = Task("proxy")
        tasks[("proxy", m.id)] = task
        task.thread = threading.Thread(target=_proxy_thread, args=(app, m, task), name=f"inframe-proxy-{m.id}", daemon=True)
        task.thread.start()
    return json_response({"state": "running", "fraction": 0.0})


def _proxy_brief(st: dict) -> dict:
    return {"state": st["state"], "fraction": st["fraction"]}


def proxy_get(app, req: Request) -> Response:
    return json_response(proxy_state(app, _media(app, req)))


def proxy_file(app, req: Request) -> Response:
    m = _media(app, req)
    if proxy_state(app, m)["state"] != "done":
        raise HttpError(404, "no preview copy yet")
    _mark_used(app, m)
    return file_response(_proxy_files(m)[0], "video/mp4", req)


# ---------------------------------------------------------------- filmstrip (2.5)
def cell_size(media) -> tuple[int, int]:
    dw, dh = media.info.display_size
    w = round(CELL_H * dw / dh) if dh else CELL_MAX_W
    return max(2, min(CELL_MAX_W, w) // 2 * 2), CELL_H


def _atomic_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    os.replace(tmp, path)


def make_filmstrip(media, task: Task | None = None) -> dict:
    """Decode the keyframes once (`decode.iter_samples(fps=None)`), keep the first keyframe of each slot of max(1 s, duration / 600)
    and write 10x10 sheets `<cache>/filmstrip/sheet_NNN.jpg` as each fills, then `<cache>/filmstrip.json` (rewritten after every
    sheet; `done` true at the end). Slots a sparse GOP skips get no cell: the player draws the nearest one. Returns the document."""
    import numpy as np
    from PIL import Image

    from . import decode

    info = media.info
    cw, ch = cell_size(media)
    slot = max(1.0, info.duration / SLOTS_MAX)
    sheet_dir = media.cache_dir / "filmstrip"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    doc_path = media.cache_dir / "filmstrip.json"
    doc_path.unlink(missing_ok=True)
    cells: list[dict] = []  # only cells whose sheet is written
    pending: list[tuple[float, np.ndarray]] = []
    sheets = 0

    def doc(done: bool) -> dict:
        return {"version": 1, "cell_w": cw, "cell_h": ch, "cols": COLS, "rows": ROWS, "cells": cells, "sheets": sheets, "done": done}

    def flush() -> None:
        nonlocal sheets, pending
        if not pending:
            return
        canvas = np.zeros((ROWS * ch, COLS * cw, 3), np.uint8)
        for i, (t, cell) in enumerate(pending):
            r, c = divmod(i, COLS)
            canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
            cells.append({"t": round(t, 3), "sheet": sheets, "i": i})
        f = sheet_dir / f"sheet_{sheets:03d}.jpg"
        part = f.with_name(f.name + ".part")
        Image.fromarray(canvas).save(part, "JPEG", quality=SHEET_QUALITY)
        os.replace(part, f)
        sheets += 1
        pending = []
        _atomic_json(doc_path, doc(False))

    last_slot = -1
    for s in decode.iter_samples(str(media.path), info, fps=None, long_side=2 * max(cw, ch)):
        if task is not None and task.cancel.is_set():
            raise TaskError("cancelled")
        j = max(0, int((s.t - info.offset) / slot + 1e-9))
        if j <= last_slot:
            continue
        last_slot = j
        im = Image.fromarray(s.rgb)
        w = max(2, round(im.width * ch / im.height))
        im = im.resize((w, ch), Image.BILINEAR)
        cell = np.zeros((ch, cw, 3), np.uint8)  # centred, cropped when wider than the cell, black bars when narrower
        a = np.asarray(im)
        x0 = (w - cw) // 2
        if w >= cw:
            cell[:] = a[:, x0:x0 + cw]
        else:
            cell[:, -x0:-x0 + w] = a
        pending.append((s.t, cell))
        if task is not None:
            task.fraction = min(0.99, (s.t - info.offset) / max(info.duration, 1e-6))
        if len(pending) == COLS * ROWS:
            flush()
    flush()
    final = doc(True)
    _atomic_json(doc_path, final)
    return final


def _film_thread(app, media, task: Task) -> None:
    slot = _tasks(app)["film_slot"]
    with slot:
        if task.cancel.is_set():
            task.state, task.error = "failed", "cancelled"
            return
        try:
            make_filmstrip(media, task)
            task.fraction, task.state = 1.0, "done"
            prune_filmstrips_for(app)
        except TaskError as e:
            task.state, task.error = "failed", str(e)
        except Exception as e:
            task.state, task.error = "failed", f"{type(e).__name__}: {e}"


def ensure_filmstrip(app, media, retry: bool = False) -> None:
    """Start the filmstrip task unless it runs, is done (in memory or on disk), or failed (`retry` starts a failed one again)."""
    with app.lock:
        tasks = _tasks(app)
        t = tasks.get(("filmstrip", media.id))
        if t is not None and (t.state == "running" or (t.state == "done" and filmstrip_state(app, media) == "done")
                              or (t.state == "failed" and not retry)):
            return
        if t is None and filmstrip_state(app, media) == "done":
            return
        task = Task("filmstrip")
        tasks[("filmstrip", media.id)] = task
        task.thread = threading.Thread(target=_film_thread, args=(app, media, task), name=f"inframe-filmstrip-{media.id}", daemon=True)
        task.thread.start()


def filmstrip_get(app, req: Request) -> Response:
    m = _media(app, req)
    ensure_filmstrip(app, m)
    doc = _read_json(m.cache_dir / "filmstrip.json")
    if not isinstance(doc, dict):
        cw, ch = cell_size(m)
        doc = {"version": 1, "cell_w": cw, "cell_h": ch, "cols": COLS, "rows": ROWS, "cells": [], "sheets": 0, "done": False}
    doc["state"] = filmstrip_state(app, m)
    with app.lock:
        t = _tasks(app).get(("filmstrip", m.id))
    if t is not None and t.state == "failed":
        doc["error"] = t.error
    return json_response(doc)


def filmstrip_sheet(app, req: Request) -> Response:
    m = _media(app, req)
    f = m.cache_dir / "filmstrip" / f"sheet_{int(req.params['n']):03d}.jpg"
    if not f.is_file():
        raise HttpError(404, "no such sheet")
    return file_response(f, "image/jpeg", req)


# ---------------------------------------------------------------- row 18: single frames
def frame_jpeg(app, req: Request) -> Response:
    from . import pick

    m = _media(app, req)
    try:
        t = float(req.query["t"]) if "t" in req.query else m.info.offset
        w = int(req.query["w"]) if "w" in req.query else FRAME_W_DEFAULT
    except ValueError:
        raise HttpError(400, "t must be a number and w a whole number") from None
    if not math.isfinite(t):
        raise HttpError(400, "t must be a number")
    t = min(max(t, m.info.offset), m.info.end_time)
    w = min(max(w, FRAME_W_MIN), FRAME_W_MAX)
    key = (m.id, round(t, 3), w)
    mt = _tasks(app)
    with mt["frames_lock"]:
        data = mt["frames"].get(key)
        if data is not None:
            mt["frames"].move_to_end(key)
    if data is None:
        try:
            _, rgb = pick.grab_frame(str(m.path), m.info, t, w)
        except ValueError as e:
            raise HttpError(404, str(e)) from None
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=FRAME_QUALITY)
        data = buf.getvalue()
        with mt["frames_lock"]:
            mt["frames"][key] = data
            while len(mt["frames"]) > FRAME_LRU:
                mt["frames"].popitem(last=False)
    return bytes_response(data, "image/jpeg")


# ---------------------------------------------------------------- the routes
ROUTES = [
    Route("POST", r"/api/media/open", media_open),
    Route("POST", r"/api/media/open-dialog", media_open_dialog),
    Route("POST", r"/api/media/photos-pick", media_photos_pick),
    Route("POST", r"/api/media/locate", media_locate),
    Route("GET", r"/api/fs/list", fs_list),
    Route("GET", r"/api/recent", recent_get),
    Route("DELETE", r"/api/recent", recent_clear),
    Route("GET", rf"/api/media/{MEDIA_ID}", media_get),
    Route("GET", rf"/api/media/{MEDIA_ID}/video", media_video, media=True),
    Route("POST", rf"/api/media/{MEDIA_ID}/proxy", proxy_post),
    Route("GET", rf"/api/media/{MEDIA_ID}/proxy", proxy_get),
    Route("GET", rf"/api/media/{MEDIA_ID}/proxy\.mp4", proxy_file, media=True),
    Route("GET", rf"/api/media/{MEDIA_ID}/filmstrip", filmstrip_get),
    Route("GET", rf"/api/media/{MEDIA_ID}/filmstrip/(?P<n>[0-9]{{1,3}})\.jpg", filmstrip_sheet, media=True),
    Route("GET", rf"/api/media/{MEDIA_ID}/frame\.jpg", frame_jpeg, media=True),
]
