"""Runs and edits on disk for `inframe serve` (docs/T9A_DESIGN.md section 2.6, routes 36-41 call into this).

A run folder is `<stem>_inframe/<slug>/`. The detector's `segments.json` is never written here; the user's edits live next to it in
`edits.json` (revision-guarded, atomic). Run folders are untrusted input (a shared or downloaded folder can hold anything): every
file is opened with O_NOFOLLOW and must be a regular file, a run is listed only when its `source.path` is an opened media, and a
path named inside target.json goes through `serve_paths.check_read` before use.

No routes here (they are in `serve_runs`); this module imports `serve_http` and `serve_paths` only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat as _stat
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import serve_paths
from .cut import snap_to_keyframe
from .serve_http import HttpError

EDITS_FORMAT = 1
MAX_CLIPS = 2000
MIN_CLIP_S = 0.2
BOUNDS_SLACK_S = 0.001
ORIGINS = ("detector", "edited", "split", "merge", "added", "resegment")
ID_RE = re.compile(r"^[a-z0-9]{1,16}$")
MAX_JSON_BYTES = 64 * 1024 * 1024  # segments.json, edits.json, target.json, export.json
MAX_CSV_BYTES = 256 * 1024 * 1024  # samples.csv
KEEP_PREV = 5
PREV_RE = re.compile(r"^edits\.prev-(\d{8}T\d{6}Z)(?:-(\d+))?\.json$")
SAME_TOL_S = 0.0015  # samples.csv holds times to 3 decimals, so a re-segment can differ from the run by a rounding step


class RunFolderError(HttpError):
    """The run folder, or a file in it, cannot be used (missing, a symlink, not a regular file, not what inframe writes)."""

    def __init__(self, message: str, code: int = 404):
        super().__init__(code, message)


class ClipError(HttpError):
    """A clip list that breaks the rules of `validate_clips` (422)."""

    def __init__(self, message: str):
        super().__init__(422, message)


class Conflict(HttpError):
    """The edits were saved by someone else since this client read them (409 with the stored document)."""

    def __init__(self, stored: dict):
        super().__init__(409, "conflict", edits=stored)


# ---------------------------------------------------------------- safe reads
def read_regular(path, limit: int = MAX_JSON_BYTES) -> bytes | None:
    """The bytes of `path` when it is a regular file that is not a symlink and at most `limit` bytes, else None."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:  # missing, a symlink (ELOOP), no permission
        return None
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode) or st.st_size > limit:
            return None
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(limit + 1)
        return data if len(data) <= limit else None
    except OSError:
        return None
    finally:
        os.close(fd)


def read_json_file(path) -> dict | None:
    """A JSON object from a regular file, or None when the file is missing, unsafe, too big or not an object."""
    raw = read_regular(path)
    if raw is None:
        return None
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return d if isinstance(d, dict) else None


def is_real_dir(path) -> bool:
    try:
        return _stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def write_atomic(path: Path, data: bytes) -> None:
    """`<path>.tmp` (created exclusively, never through a symlink) then `os.replace`; the tmp file is removed on failure."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.unlink(missing_ok=True)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def run_lock(run_dir) -> threading.RLock:
    """One lock per run folder: edits.json is read, compared and replaced as one step."""
    key = str(run_dir)
    with _locks_guard:
        return _locks.setdefault(key, threading.RLock())


# ---------------------------------------------------------------- the run folder
def run_id_for(run_dir) -> str:
    return hashlib.sha1(str(run_dir).encode()).hexdigest()[:12]


def run_dir_for(out_root, slug: str) -> Path:
    """`<out_root>/<slug>/`, made (mode 0700 the first time; an existing folder keeps its mode). `slug` is a single safe name."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", slug):
        raise ValueError(f"bad run folder name {slug!r}")
    d = Path(out_root) / slug
    d.parent.mkdir(parents=True, exist_ok=True)
    d.mkdir(mode=0o700, exist_ok=True)
    return d


def read_segments(run_dir) -> dict:
    """The run's `segments.json` (regular file inside a real folder), else `RunFolderError`."""
    if not is_real_dir(run_dir):
        raise RunFolderError("the run folder is gone or is not a plain folder")
    doc = read_json_file(Path(run_dir) / "segments.json")
    if doc is None:
        raise RunFolderError("segments.json is missing or is not a plain JSON file")
    if not isinstance(doc.get("source"), dict) or not isinstance(doc.get("clips", []), list):
        raise RunFolderError("segments.json is not one that inframe wrote")
    return doc


def segments_sha256(run_dir) -> str:
    raw = read_regular(Path(run_dir) / "segments.json")
    if raw is None:
        raise RunFolderError("segments.json is missing or is not a plain JSON file")
    return hashlib.sha256(raw).hexdigest()


def source_bounds(doc: dict) -> tuple[float, float]:
    """(offset, end_time) of the video in file time, from segments.json `source`. The file holds `start_time`,
    `format_start_time` and `duration` (probe.VideoInfo.to_dict); `offset` and `end_time` are computed from them, as
    VideoInfo does. Explicit `offset` / `end_time` keys win when present."""
    s = doc.get("source") or {}
    try:
        offset = float(s["offset"]) if "offset" in s else float(s["start_time"]) - float(s.get("format_start_time", 0.0))
        end = float(s["end_time"]) if "end_time" in s else offset + float(s["duration"])
    except (KeyError, TypeError, ValueError):
        raise RunFolderError("segments.json has no usable source times", 422) from None
    if not (math.isfinite(offset) and math.isfinite(end) and end > offset):
        raise RunFolderError("segments.json has no usable source times", 422)
    return offset, end


def source_matches(doc: dict, media_path: Path) -> bool:
    src = (doc.get("source") or {}).get("path")
    if not isinstance(src, str) or not src or "\0" in src:
        return False
    try:
        return Path(src).resolve() == media_path
    except (OSError, RuntimeError):
        return False


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_runs(out_root, media_path: Path) -> list[dict]:
    """The runs of one video: every real folder of `out_root` whose regular `segments.json` names `media_path` as its source.
    Newest first. A symlinked folder or `segments.json`, an unreadable file and a run of another video are not listed.
    Each entry also carries the absolute `dir` (the route registers it)."""
    out = []
    try:
        entries = list(os.scandir(out_root))
    except OSError:
        return out
    for e in entries:
        if e.name.startswith(".") or not e.is_dir(follow_symlinks=False):
            continue
        d = Path(e.path)
        seg = d / "segments.json"
        raw = read_regular(seg)
        if raw is None:
            continue
        try:
            doc = json.loads(raw.decode("utf-8"))
            mtime = os.lstat(seg).st_mtime
        except (UnicodeDecodeError, ValueError, RecursionError, OSError):
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("clips", []), list) or not source_matches(doc, media_path):
            continue
        st = doc.get("settings") if isinstance(doc.get("settings"), dict) else {}
        edits = load_edits(d)
        sha = hashlib.sha256(raw).hexdigest()
        out.append({"run_id": run_id_for(d), "dir": str(d), "slug": e.name, "target": st.get("target"),
                    "preset": st.get("preset_name") or st.get("preset"), "created": _iso(mtime), "_mtime": mtime,
                    "n_clips": len(doc.get("clips") or []), "error": doc.get("error"), "has_edits": edits is not None,
                    "edits_base_ok": edits is None or edits["base"]["segments_sha256"] == sha})
    out.sort(key=lambda r: r["_mtime"], reverse=True)
    for r in out:
        del r["_mtime"]
    return out


# ---------------------------------------------------------------- samples
def load_samples_arrays(run_dir) -> tuple[list[float], list[float], list[list[float] | None], list[int | None] | None]:
    """`(times, scores, boxes, track_ids)` from `samples.csv`: the inverse of `pipeline.write_samples`. A box is `[x0, y0, x1, y1]`
    or None where the csv has none; `track_ids` is None when the file has no `track` column, else an int or None per sample.
    A row whose time or score is not a finite number is skipped."""
    raw = read_regular(Path(run_dir) / "samples.csv", MAX_CSV_BYTES)
    if raw is None:
        raise RunFolderError("samples.csv is missing or is not a plain file")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"), newline=""))
    names = reader.fieldnames or []
    if "t" not in names or "score" not in names:
        raise RunFolderError("samples.csv is not one that inframe wrote", 422)
    has_track = "track" in names
    times, scores, boxes = [], [], []
    tracks: list[int | None] = []
    for r in reader:
        try:
            t, s = float(r["t"]), float(r["score"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (math.isfinite(t) and math.isfinite(s)):
            continue
        box = None
        try:
            vals = [r.get(k) for k in ("x0", "y0", "x1", "y1")]
            if all(v not in (None, "") for v in vals):
                b = [float(v) for v in vals]
                box = b if all(math.isfinite(v) for v in b) else None
        except ValueError:
            box = None
        tr = None
        if has_track and r.get("track") not in (None, ""):
            try:
                tr = int(float(r["track"]))
            except (ValueError, OverflowError):
                tr = None
        times.append(t)
        scores.append(s)
        boxes.append(box)
        tracks.append(tr)
    return times, scores, boxes, (tracks if has_track else None)


def edited_clip_facts(start: float, end: float, times, scores, keyframes) -> dict:
    """What `run_cut` records for a clip, for an edited clip: `peak_score` (the highest score of the samples inside [start, end],
    0.0 when none) and `expected_copy_start` (where a stream-copy cut really starts: the keyframe at or before `start`)."""
    ts = np.asarray(times, float)
    sa = np.asarray(scores, float)
    m = (ts >= start) & (ts <= end) if ts.size else np.zeros(0, bool)
    peak = float(sa[m].max()) if m.any() else 0.0
    return {"peak_score": round(peak, 5), "expected_copy_start": round(snap_to_keyframe(start, list(keyframes)), 3)}


# ---------------------------------------------------------------- edits
def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_clips(clips, offset: float, end_time: float) -> list[dict]:
    """The clip list, checked and cleaned (only `id`, `start`, `end`, `origin`; times rounded to 1 ms), else `ClipError`.
    Rules (2.6): at most 2000 clips; finite numbers; sorted by start; `offset <= start < end <= end_time` (1 ms slack);
    `end - start >= 0.2`; no overlap; unique ids `[a-z0-9]{1,16}`; origin one of `ORIGINS`."""
    if not isinstance(clips, list):
        raise ClipError("clips must be a list")
    if len(clips) > MAX_CLIPS:
        raise ClipError(f"at most {MAX_CLIPS} clips (got {len(clips)})")
    out: list[dict] = []
    seen: set[str] = set()
    prev_start = prev_end = None
    for n, c in enumerate(clips, 1):
        if not isinstance(c, dict):
            raise ClipError(f"clip {n}: not an object")
        cid, origin, s, e = c.get("id"), c.get("origin"), c.get("start"), c.get("end")
        if not isinstance(cid, str) or not ID_RE.match(cid):
            raise ClipError(f"clip {n}: id must be 1 to 16 characters of a-z and 0-9")
        if cid in seen:
            raise ClipError(f"clip {n}: duplicate id {cid}")
        seen.add(cid)
        if origin not in ORIGINS:
            raise ClipError(f"clip {n}: origin must be one of {', '.join(ORIGINS)}")
        if not (_number(s) and _number(e)):
            raise ClipError(f"clip {n}: start and end must be finite numbers")
        s, e = round(float(s), 3), round(float(e), 3)
        if s < offset - BOUNDS_SLACK_S or e > end_time + BOUNDS_SLACK_S:
            raise ClipError(f"clip {n}: {s:g}-{e:g} s is outside the video ({offset:g}-{end_time:g} s)")
        s, e = round(max(s, offset), 3), round(min(e, end_time), 3)
        if e - s < MIN_CLIP_S - 1e-9:
            raise ClipError(f"clip {n}: {s:g}-{e:g} s is shorter than {MIN_CLIP_S:g} s")
        if prev_start is not None and s < prev_start:
            raise ClipError(f"clip {n}: clips must be sorted by start")
        if prev_end is not None and s < prev_end - 1e-9:
            raise ClipError(f"clip {n}: overlaps the clip before it")
        prev_start, prev_end = s, e
        out.append({"id": cid, "start": s, "end": e, "origin": origin})
    return out


def _doc_ok(d) -> bool:
    if not isinstance(d, dict) or d.get("format") != EDITS_FORMAT:
        return False
    rev, base, clips = d.get("revision"), d.get("base"), d.get("clips")
    if not (isinstance(rev, int) and not isinstance(rev, bool) and rev >= 0):
        return False
    if not (isinstance(base, dict) and isinstance(base.get("segments_sha256"), str)):
        return False
    return isinstance(clips, list) and all(
        isinstance(c, dict) and isinstance(c.get("id"), str) and _number(c.get("start")) and _number(c.get("end")) for c in clips)


def load_edits(run_dir) -> dict | None:
    """The run's `edits.json`, or None when there is none or it is not a document that `save_edits` wrote."""
    d = read_json_file(Path(run_dir) / "edits.json")
    return d if _doc_ok(d) else None


def detector_edits_doc(seg_doc: dict, sha: str) -> dict:
    """The edits document of a run nobody edited yet: revision 0 and the detector's clips (`d1`, `d2`, ...)."""
    clips = [{"id": f"d{i}", "start": c["start"], "end": c["end"], "origin": "detector"}
             for i, c in enumerate(seg_doc.get("clips") or [], 1) if isinstance(c, dict) and _number(c.get("start")) and _number(c.get("end"))]
    return {"format": EDITS_FORMAT, "base": {"segments_sha256": sha, "n_clips": len(clips)}, "revision": 0, "updated": None,
            "next_id": 1, "clips": clips}


def _next_id(stored_next, clips) -> int:
    top = 0
    for c in clips:
        m = re.fullmatch(r"u(\d+)", c["id"])
        if m:
            top = max(top, int(m.group(1)))
    return max(int(stored_next) if isinstance(stored_next, int) and not isinstance(stored_next, bool) else 1, top + 1)


def save_edits(run_dir, revision: int, clips: list) -> dict:
    """Validate and store `clips` as revision `revision + 1`. `revision` is the one the client edited: another value is a
    `Conflict` carrying the stored document (or the detector's clips at revision 0 when nothing is stored). Atomic."""
    run_dir = Path(run_dir)
    seg = read_segments(run_dir)
    offset, end_time = source_bounds(seg)
    clips = validate_clips(clips, offset, end_time)
    with run_lock(run_dir):
        raw = read_regular(run_dir / "segments.json")
        if raw is None:
            raise RunFolderError("segments.json is missing or is not a plain JSON file")
        sha = hashlib.sha256(raw).hexdigest()
        stored = load_edits(run_dir)
        current = stored["revision"] if stored else 0
        if revision != current:
            raise Conflict(stored or detector_edits_doc(seg, sha))
        doc = {"format": EDITS_FORMAT, "base": {"segments_sha256": sha, "n_clips": len(seg.get("clips") or [])},
               "revision": current + 1, "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "next_id": _next_id(stored.get("next_id") if stored else None, clips), "clips": clips}
        write_atomic(run_dir / "edits.json", json.dumps(doc, indent=1).encode())
        return doc


def revert_edits(run_dir) -> str | None:
    """Move `edits.json` aside to `edits.prev-<UTC time>.json` (at most 5 such files are kept) so the detector clips show again
    and the edits can still be restored. Returns the new name, or None when there was nothing to move."""
    run_dir = Path(run_dir)
    with run_lock(run_dir):
        src = run_dir / "edits.json"
        try:
            st = os.lstat(src)
        except OSError:
            return None
        if not _stat.S_ISREG(st.st_mode):  # a symlink or a folder named edits.json is not ours: remove the link, never follow it
            if _stat.S_ISLNK(st.st_mode):
                src.unlink()
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        same = [int(m.group(2) or 0) for m in map(PREV_RE.match, os.listdir(run_dir)) if m and m.group(1) == stamp]
        name = f"edits.prev-{stamp}.json" if not same else f"edits.prev-{stamp}-{max(same) + 1}.json"  # later than every one of this second
        os.replace(src, run_dir / name)
        prev = sorted((f for f in os.listdir(run_dir) if PREV_RE.match(f)),
                      key=lambda f: (PREV_RE.match(f).group(1), int(PREV_RE.match(f).group(2) or 0)))  # oldest first
        for old in prev[:-KEEP_PREV] if len(prev) > KEEP_PREV else []:
            p = run_dir / old
            try:
                if _stat.S_ISREG(os.lstat(p).st_mode):
                    p.unlink()
            except OSError:
                pass
        return name


# ---------------------------------------------------------------- target.json
def check_target_json(path) -> bytes:
    """The bytes of a target.json (a regular file) after the rules for paths named inside it: `photos` must be a saved target name
    (`saved:NAME`) or a folder that `serve_paths.check_read` accepts (not hidden, not under ~/Library), and `video` a readable file.
    Nothing named in the file is read here. Raises `HttpError` (403/404/422)."""
    from . import targets

    raw = read_regular(path, 1024 * 1024)
    if raw is None:
        raise HttpError(422, "target.json is missing or is not a plain file")
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise HttpError(422, "target.json is not valid JSON") from None
    if not isinstance(d, dict):
        raise HttpError(422, "target.json is not a JSON object")
    photos, video = d.get("photos"), d.get("video")
    if photos:
        if not isinstance(photos, str):
            raise HttpError(422, "target.json: photos must be a folder path or saved:NAME")
        if photos.startswith("saved:"):
            try:
                targets.check_name(photos[len("saved:"):])
            except ValueError as e:
                raise HttpError(422, str(e)) from None
        else:
            serve_paths.check_read(photos, "dir")
    if video:
        if not isinstance(video, str):
            raise HttpError(422, "target.json: video must be a file path")
        serve_paths.check_read(video, "file")
    return raw
