"""Pick and targets routes of `inframe serve` (docs/T9A_DESIGN.md sections 2.8 and 3.3, routes 19-33).

`ServePicker` is `pick.Picker` plus a rank that reports progress and cancels, a reset and a lock around its edits. One exists per
`(media_id, class name, target_key)` in `app.pickers`; `target_key` is "" for plain taps, "photos:<photos_id>" or "saved:<name>".
`Picker` fixes class, photos, embedder and preset model at construction, so another class, photos folder or saved target is
another ServePicker, never a mutation. Each writes its own file, `serve_paths.target_file(out_root, cls, target_key)`
(`<stem>_inframe/targets/target-<cls>[-<key>].json`), so `Picker.save` (which unlinks its file when the taps are empty) never
touches another picker's taps. The run (part B1) copies the active picker's file to `<stem>_inframe/target.json`.

The routes for taps, undo, zone, reset and the state work on the plain picker (target_key ""). Rank, candidates and confirm work on
the picker of the photos folder or saved target that was ranked; candidates and confirm find the one ranked last for the media.

This module imports `serve_http`, `serve_paths`, `serve_jobs` and core `inframe` modules, never `serve.py`.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import stat as _stat
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image

from . import classes, picker, serve_paths
from .pick import Picker, Superseded
from .serve_http import HttpError, Request, Response, Route, bytes_response, error, file_response, json_response

MIN_BOX_SCORE = 0.4  # pick.py shows boxes from this score up
DIALOG_TIMEOUT_S = 600
PRESET_NAME = "balanced"  # the detector the pick page uses unless told otherwise (pipeline.PRESETS)
PEOPLE_JSON_MAX = 1024 * 1024
CROP_SUFFIX = ".jpg"

MID = r"(?P<mid>[A-Za-z0-9_-]{1,32})"
_create_lock = threading.Lock()  # one picker is built at a time (it probes the video)


# ---------------------------------------------------------------- ServePicker
class ServePicker(Picker):
    """`pick.Picker` for the server (docs/T9A_DESIGN.md 2.8)."""

    def __init__(self, video: str, out: Path, preset_model: str = "small", photos: str | None = None, embedder: str | None = None,
                 offline: bool = False, cls: int = classes.PERSON, preset_name: str | None = None, target_key: str = ""):
        from .identity import DEFAULT_EMBEDDER

        super().__init__(video, out, preset_model, photos, embedder or DEFAULT_EMBEDDER, offline, cls, preset_name)
        self.target_key = target_key
        self.edit_lock = threading.RLock()  # taps, zone and candidates change from several request threads
        self.ranked_at = 0.0  # monotonic time of the last finished rank (0: never)

    def rank(self, announce=print, progress=None) -> list[dict]:
        """`Picker.rank` with `progress(fraction, message)` passed to `score_detect`, so a rank job reports and cancels."""
        from . import identity, pipeline

        preset = next((p for p in pipeline.PRESETS.values() if p.model == self.model and p.detect_fps), pipeline.PRESETS["balanced"])
        fps = pipeline.effective_fps(pipeline.Settings(), preset)
        target = identity.PhotoTarget(self.photos, embedder=self.embedder, offline=self.offline, rank_only=True,
                                      detect=identity.lazy_detect(self.model), cls=self.cls)
        announce(f"inframe: ranking the candidates in the video against the photos ({self.embedder}); this reads the whole video once")
        sc = pipeline.score_detect(self.video, self.info, preset, fps, target, progress, "ocsort")
        cands = list(sc.info.get("target_info", {}).get("ranking", []))
        with self.edit_lock:
            self.cands = cands
            self.ranked_at = time.monotonic()
        announce(f"inframe: {len(cands)} candidates")
        return cands

    def reset(self) -> None:
        """Forget this picker's taps, zone and candidates; its file is removed (Picker.save unlinks it for empty taps)."""
        with self.edit_lock:
            self.taps = []
            self.zone = None
            self.cands = []
            self.ranked_at = 0.0
            self.save()


# ---------------------------------------------------------------- request helpers
def _guard(fn):
    """Domain errors of the pick, target and photo code become 422 with their own message; a Superseded request is 409."""
    from . import identity, models, tracks
    from .tap import TapError

    domain = (models.ModelsMissing, identity.EmbedderUnavailable, TapError, tracks.TrackerUnavailable)

    def wrapped(app, req):
        try:
            return fn(app, req)
        except HttpError:
            raise
        except Superseded:
            return error(409, "superseded")
        except domain as e:
            return error(422, str(e))

    wrapped.__name__ = fn.__name__
    return wrapped


def _body(req: Request) -> dict:
    return req.json if isinstance(req.json, dict) else {}


def _cls(value) -> tuple[int, str]:
    """(COCO id, canonical name) of a class given by name; absent means person."""
    if value is None or value == "":
        return classes.PERSON, classes.class_name(classes.PERSON)
    if not isinstance(value, str):
        raise HttpError(400, "cls must be a class name")
    try:
        return classes.parse_class_spec(value)
    except classes.UnknownClass as e:
        raise HttpError(400, str(e)) from None


def _num(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HttpError(400, f"{what} must be a finite number")
    return float(value)


def _query_float(req: Request, name: str) -> float:
    raw = req.query.get(name)
    if raw is None or raw == "":
        raise HttpError(400, f"{name} is required")
    try:
        v = float(raw)
    except ValueError:
        raise HttpError(400, f"{name} must be a number") from None
    if not math.isfinite(v):
        raise HttpError(400, f"{name} must be a finite number")
    return v


def get_picker(app, media, cls_name: str, target_key: str = "", photos: str | None = None) -> ServePicker:
    """The ServePicker of `(media.id, cls_name, target_key)`: made on first use (its detector loads in a thread meanwhile), the same
    object afterwards."""
    key = (media.id, cls_name, target_key)
    with app.lock:
        pk = app.pickers.get(key)
    if pk is not None:
        return pk
    with _create_lock:
        with app.lock:
            pk = app.pickers.get(key)
        if pk is not None:
            return pk
        from .pipeline import PRESETS

        out = serve_paths.target_file(app.out_root(media), cls_name, target_key)
        serve_paths.check_write(out.parent)
        pk = ServePicker(str(media.path), out, PRESETS[PRESET_NAME].model, photos, None, app.offline, classes.class_id(cls_name),
                         PRESET_NAME, target_key)
        with app.lock:
            app.pickers[key] = pk
    threading.Thread(target=pk.warm, name="inframe-pick-warm", daemon=True).start()
    return pk


def _candidates(pk: ServePicker) -> list[dict]:
    return [{"rank": c.get("rank"), "t": c.get("t"), "box": c.get("box"), "track_id": c.get("track_id"), "score": c.get("score")}
            for c in pk.cands]


def pick_state(pk: ServePicker) -> dict:
    """The PickState of section 3.3."""
    with pk.edit_lock:
        taps = [{"time": t.get("time"), "box": t.get("box"), "track_id": t.get("track_id"), "index": t.get("index")} for t in pk.taps]
        return {"target_key": pk.target_key, "taps": taps, "zone": list(pk.zone) if pk.zone else None,
                "file": str(pk.out) if pk.taps and pk.out.is_file() else None, "class": classes.class_name(pk.cls),
                "candidates": _candidates(pk)}


def _jpeg(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _regular(path: Path) -> bool:
    try:
        return _stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _ranked_picker(app, media_id: str, cls_name: str | None = None) -> ServePicker:
    """The picker of this media that ranked last (and holds candidates), optionally of one class."""
    with app.lock:
        found = [pk for (mid, c, _), pk in app.pickers.items()
                 if mid == media_id and (cls_name is None or c == cls_name) and getattr(pk, "cands", None)]
    if not found:
        raise HttpError(404, "no candidates yet: use Find who it is first")
    return max(found, key=lambda pk: getattr(pk, "ranked_at", 0.0))


# ---------------------------------------------------------------- routes 19-25: picking
def boxes(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    t = _query_float(req, "t")
    _, cname = _cls(req.query.get("cls"))
    pk = get_picker(app, m, cname)
    ft, _, dets = pk.frame(t)
    shown = [d for d in dets if d.score >= MIN_BOX_SCORE]
    return json_response({"t": round(ft, 3), "boxes": [{"i": i + 1, "box": [round(v, 4) for v in d.box], "score": round(d.score, 3)}
                                                       for i, d in enumerate(shown)]})


def pickframe(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    t = _query_float(req, "t")
    _, cname = _cls(req.query.get("cls"))
    pk = get_picker(app, m, cname)
    try:
        rgb = pk.frame_image(t)
    except (ValueError, OSError) as e:
        raise HttpError(404, f"no frame at {t:.3f} s: {e}") from None
    return bytes_response(_jpeg(rgb), "image/jpeg")


def pick_get(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    _, cname = _cls(req.query.get("cls"))
    return json_response(pick_state(get_picker(app, m, cname)))


def _check_box(value) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise HttpError(400, "box must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_num(v, "box") for v in value)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise HttpError(400, "box must be fractions of the frame with x0 < x1 and y0 < y1")
    return [x0, y0, x1, y1]


def pick_tap(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    d = _body(req)
    _, cname = _cls(d.get("cls"))
    t = _num(d.get("t"), "t")
    box = _check_box(d.get("box"))
    index = d.get("index")
    if index is not None and (isinstance(index, bool) or not isinstance(index, int)):
        raise HttpError(400, "index must be an integer or null")
    info = m.info
    if not (info.offset - 0.01 <= t <= info.end_time + 0.01):
        raise HttpError(400, f"t is outside the video ({info.offset:.3f} to {info.end_time:.3f} s)")
    pk = get_picker(app, m, cname)
    with pk.edit_lock:
        pk.add_tap(t, box, index)
    return json_response(pick_state(pk))


def pick_undo(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    _, cname = _cls(_body(req).get("cls"))
    pk = get_picker(app, m, cname)
    with pk.edit_lock:
        if pk.taps:
            pk.taps.pop()
            pk.save()
    return json_response(pick_state(pk))


def pick_zone(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    d = _body(req)
    _, cname = _cls(d.get("cls"))
    if "zone" not in d:
        raise HttpError(400, "zone is required (use null to clear it)")
    z = d["zone"]
    if z is not None:
        if not isinstance(z, list) or len(z) != 4:
            raise HttpError(400, "zone must be [x, y, w, h] or null")
        z = [_num(v, "zone") for v in z]
    pk = get_picker(app, m, cname)
    try:
        with pk.edit_lock:
            pk.set_zone(z)
    except ValueError as e:
        raise HttpError(400, str(e)) from None
    return json_response(pick_state(pk))


def pick_reset(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    _, cname = _cls(_body(req).get("cls"))
    pk = get_picker(app, m, cname)
    pk.reset()
    return json_response(pick_state(pk))


# ---------------------------------------------------------------- routes 26-27: saved targets
def targets_list(app, req: Request) -> Response:
    from . import targets

    return json_response({"targets": targets.list_targets()})


def target_crop(app, req: Request) -> Response:
    from . import targets

    try:
        name = targets.check_name(req.params["name"])
    except targets.TargetError as e:
        raise HttpError(400, str(e)) from None
    i = int(req.params["i"])
    try:
        paths = targets.Saved(name).crop_paths()
    except targets.TargetError as e:
        raise HttpError(404, str(e)) from None
    if i >= len(paths):
        raise HttpError(404, f"{name} has {len(paths)} crops")
    p = paths[i]
    if p.suffix.lower() != CROP_SUFFIX or not _regular(p):
        raise HttpError(404, "no such crop")
    return file_response(p, "image/jpeg", req)


# ---------------------------------------------------------------- route 28: photos
def run_osascript(script: str) -> str | None:
    """The path a native dialog returned, or None when the person cancelled."""
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=DIALOG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise HttpError(504, "The folder dialog timed out.") from None
    except OSError as e:
        raise HttpError(501, f"osascript is not available: {e}") from None
    if r.returncode != 0:
        if "User canceled" in r.stderr or "(-128)" in r.stderr:
            return None
        raise HttpError(500, f"osascript failed: {r.stderr.strip()[:200]}")
    return r.stdout.strip() or None


def is_mac() -> bool:
    return sys.platform == "darwin"


FOLDER_DIALOG = 'POSIX path of (choose folder with prompt "Choose the folder with photos of the target")'


def _flatten_picked(d: Path, files: list[Path]) -> Path:
    """The PHPicker helper exports each photo into its own numbered subfolder; `identity.photo_files` reads one folder. Move the
    exported files (only those inside `d`) into `d/photos`."""
    dest = d / "photos"
    dest.mkdir(mode=0o700, exist_ok=True)
    root = d.resolve()
    for i, f in enumerate(files, 1):
        try:
            src = Path(f).resolve(strict=True)
        except OSError:
            continue
        if not src.is_relative_to(root) or not _regular(src):
            continue
        os.replace(src, dest / f"{i:02d}_{src.name}")
    return dest


def photos_choose(app, req: Request) -> Response:
    from . import identity

    d = _body(req)
    source = d.get("source")
    if source == "dialog":
        if not is_mac():
            return error(501, "The folder dialog is macOS only: use the file browser.")
        chosen = run_osascript(FOLDER_DIALOG)
        if chosen is None:
            return json_response({"cancelled": True})
        folder = serve_paths.check_read(chosen, "dir")
    elif source == "photos":
        if not is_mac() and not os.environ.get(picker.HELPER_ENV):
            return error(501, "The Photos picker is macOS only: use a folder.")
        exp = serve_paths.new_picked_dir()
        try:
            res = picker.pick(exp, kind="images")
        except picker.PickerError as e:
            return error(422, str(e))
        if res.cancelled or not res.files:
            return json_response({"cancelled": True})
        folder = serve_paths.check_read(_flatten_picked(exp, res.files), "dir")
    elif source == "path":
        path = d.get("path")
        if not isinstance(path, str):
            raise HttpError(400, "path must be a string")
        folder = serve_paths.check_read(path, "dir")
    else:
        raise HttpError(400, 'source must be "dialog", "photos" or "path"')
    try:
        n = len(identity.photo_files(folder))
    except identity.PhotoError as e:
        return error(422, str(e))
    with app.lock:
        pid = next((k for k, v in app.photos_dirs.items() if v == folder), None)
        if pid is None:
            i = 1
            while f"p{i}" in app.photos_dirs:
                i += 1
            pid = f"p{i}"
            app.photos_dirs[pid] = folder
    return json_response({"id": pid, "dir": str(folder), "n_photos": n})


# ---------------------------------------------------------------- routes 29-31: rank, candidates, confirm
def rank(app, req: Request) -> Response:
    from . import targets

    m = app.media_or_404(req.params["mid"])
    d = _body(req)
    _, cname = _cls(d.get("cls"))
    pid, saved = d.get("photos_id"), d.get("saved")
    if (pid is None) == (saved is None):
        raise HttpError(400, 'give exactly one of "photos_id" and "saved"')
    if pid is not None:
        if not isinstance(pid, str):
            raise HttpError(400, "photos_id must be a string")
        with app.lock:
            folder = app.photos_dirs.get(pid)
        if folder is None:
            raise HttpError(404, f"no such photos id: {pid}")
        key, photos = f"photos:{pid}", str(folder)
    else:
        if not isinstance(saved, str):
            raise HttpError(400, "saved must be a name")
        try:
            targets.check_name(saved)
        except targets.TargetError as e:
            raise HttpError(400, str(e)) from None
        if saved not in targets.names():
            raise HttpError(404, f"no saved target named {saved!r}")
        try:
            targets.check_usable(saved)
        except targets.TargetError as e:
            return error(422, str(e))
        key, photos = f"saved:{saved}", f"saved:{saved}"
    pk = get_picker(app, m, cname, key, photos)

    def job(progress):
        pk.rank(announce=lambda msg: None, progress=progress)
        return {"candidates": _candidates(pk)}

    return json_response({"job": app.jobs.start("rank", job, media_id=m.id)})


def cand_image(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    i = int(req.params["i"])
    pk = _ranked_picker(app, m.id)
    if i >= len(pk.cands):
        raise HttpError(404, f"there are {len(pk.cands)} candidates")
    try:
        data = pk.thumb(i)
    except (ValueError, OSError) as e:
        raise HttpError(404, f"no crop: {e}") from None
    return bytes_response(data, "image/jpeg")


def pick_confirm(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    d = _body(req)
    _, cname = _cls(d.get("cls"))
    i = d.get("i")
    if isinstance(i, bool) or not isinstance(i, int):
        raise HttpError(400, "i must be an integer")
    pk = _ranked_picker(app, m.id, cname)
    with pk.edit_lock:
        if not 0 <= i < len(pk.cands):
            raise HttpError(400, f"i must be 0 to {len(pk.cands) - 1}")
        pk.confirm(i)
    return json_response(pick_state(pk))


# ---------------------------------------------------------------- routes 32-33: people
def people_dir(app, media) -> Path:
    return app.out_root(media) / "people"


def people_scan(app, req: Request) -> Response:
    m = app.media_or_404(req.params["mid"])
    out = people_dir(app, m)
    serve_paths.check_write(out)

    def job(progress):
        from . import people

        doc = people.scan_people(str(m.path), out, progress)
        doc = dict(doc)
        doc["people"] = [dict(p, crop_url=f"api/media/{m.id}/people/{p['person']}.jpg") for p in doc.get("people", [])]
        return doc

    return json_response({"job": app.jobs.start("people", job, media_id=m.id)})


def person_image(app, req: Request) -> Response:
    """The thumbnail of person n. Only the scan id is read from people.json; the file name is built here."""
    m = app.media_or_404(req.params["mid"])
    n = int(req.params["n"])
    d = people_dir(app, m)
    pj = d / "people.json"
    try:
        if not _regular(pj) or os.lstat(pj).st_size > PEOPLE_JSON_MAX:
            raise ValueError
        doc = json.loads(pj.read_text())
        scan_id = doc["scan_id"]
        known = {p["person"] for p in doc["people"]}
    except (OSError, ValueError, KeyError, TypeError):
        raise HttpError(404, "no people scan for this video yet") from None
    if not isinstance(scan_id, str) or not re.fullmatch(r"[0-9a-f]{8}", scan_id) or n not in known:
        raise HttpError(404, "no such person")
    f = d / f"person_{n:03d}_{scan_id}.jpg"
    if not _regular(f) or not f.resolve().is_relative_to(d.resolve()):
        raise HttpError(404, "no such person")
    return file_response(f, "image/jpeg", req)


# ---------------------------------------------------------------- the table
ROUTES = [
    Route("GET", rf"/api/media/{MID}/boxes", _guard(boxes)),
    Route("GET", rf"/api/media/{MID}/pickframe\.jpg", _guard(pickframe), media=True),
    Route("GET", rf"/api/media/{MID}/pick", _guard(pick_get)),
    Route("POST", rf"/api/media/{MID}/pick/tap", _guard(pick_tap)),
    Route("POST", rf"/api/media/{MID}/pick/undo", _guard(pick_undo)),
    Route("POST", rf"/api/media/{MID}/pick/zone", _guard(pick_zone)),
    Route("POST", rf"/api/media/{MID}/pick/reset", _guard(pick_reset)),
    Route("GET", r"/api/targets", _guard(targets_list)),
    Route("GET", r"/api/targets/(?P<name>[^/]{1,128})/crops/(?P<i>[0-9]{1,4})\.jpg", _guard(target_crop), media=True),
    Route("POST", r"/api/photos/choose", _guard(photos_choose)),
    Route("POST", rf"/api/media/{MID}/rank", _guard(rank)),
    Route("GET", rf"/api/media/{MID}/cands/(?P<i>[0-9]{{1,4}})\.jpg", _guard(cand_image), media=True),
    Route("POST", rf"/api/media/{MID}/pick/confirm", _guard(pick_confirm)),
    Route("POST", rf"/api/media/{MID}/people", _guard(people_scan)),
    Route("GET", rf"/api/media/{MID}/people/(?P<n>[0-9]{{1,3}})\.jpg", _guard(person_image), media=True),
]


def attach(app) -> None:
    """The pickers and the photos folders live on the App (docs/T9A_DESIGN.md 2.3); make sure the fields are there."""
    if not hasattr(app, "pickers"):
        app.pickers = {}
    if not hasattr(app, "photos_dirs"):
        app.photos_dirs = {}
