"""Export routes for `inframe serve` (docs/T9A_DESIGN.md section 2.7 and section 3.4 rows 42-45).

Row 42 starts an export job that cuts the run's CURRENT clip list (the user's edits, else the detector's clips) into `edited/` (or a
folder the user chose): clip files, sidecars, the LosslessCut CSV, a reel and edit lists. The detector's `segments.json` is never
written; the export gets its own `segments.json` and `export.json`. Row 43 reads the export.json back, row 44 serves a clip, reel or
credits file (Range-aware), row 45 shows a file in the file manager.

Files are only ever removed or served by NAME PATTERN from a directory scan (never by a name read from a JSON file), and only when
they are plain files (lstat) whose resolved parent is the folder itself.
Imports `serve_http`, `serve_paths`, `serve_jobs`, `serve_edits` and `serve_runs`, never `serve.py`.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat as _stat
import subprocess
import sys
import threading
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from . import cut, editlist, serve_edits, serve_paths, serve_runs, sidecar
from . import export as export_mod
from .serve_http import HttpError, Request, Response, Route, file_response, json_response
from .names import safe_stem
from .serve_jobs import Busy
from .serve_music import LIBRARY_NOTICE, _download  # one download at a time, shared with /api/music/get

RID = serve_runs.RID
EDITED = "edited"
FIXED_NAMES = ("segments.json", "export.json")  # written into the export folder by every export
MAX_NAME_BYTES = 255
REVEAL_TIMEOUT_S = 15
CONTENT_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
                 ".ogv": "video/ogg", ".txt": "text/plain; charset=utf-8"}
OUTPUT_KINDS = ("clip", "sidecar", "reel", "credits", "csv", "editlist")  # the order of the `outputs` list
SERVED_KINDS = ("clip", "reel", "credits")  # row 44 serves these; row 45 reveals any output kind


class ExportError(RuntimeError):
    """The export cannot go on (the job ends `failed` with this message)."""


# ---------------------------------------------------------------- names this tool writes
def name_patterns(stem: str) -> dict[str, re.Pattern]:
    """The file names an export writes for a video called `stem`, by kind: the same patterns as `pipeline.clear_old_clips`,
    `sidecar.clear_old_sidecars`, `export.clear_old_exports` and the edit list and LosslessCut CSV names."""
    stem = safe_stem(stem)
    e = re.escape(stem)
    clip = rf"{e}_\d{{2,}}_\d{{4,}}-\d{{4,}}"
    lists = "|".join(re.escape(editlist.edit_list_name(stem, f)) for f in editlist.FORMATS)
    return {"clip": re.compile(rf"^{clip}\.\w+$"), "sidecar": re.compile(rf"^{clip}{re.escape(sidecar.SUFFIX)}$"),
            "reel": re.compile(rf"^{e}_reel\.\w+$"), "credits": re.compile(rf"^{re.escape(export_mod.credits_name(stem))}$"),
            "csv": re.compile(rf"^{e}_losslesscut\.csv$"), "editlist": re.compile(rf"^(?:{lists})$")}


def classify(name: str, patterns: dict[str, re.Pattern]) -> str | None:
    for kind in OUTPUT_KINDS:
        if patterns[kind].match(name):
            return kind
    return None


def scan_folder(folder: Path, stem: str) -> tuple[list[str], list[str]]:
    """`(ours, blockers)` of `folder`: the names that match what an export writes (and `segments.json`, `export.json`), split into
    plain files (`ours`) and anything else (a symlink, a folder: `blockers`). Names come from `os.scandir`, nothing else."""
    pats = name_patterns(stem)
    ours: list[str] = []
    blockers: list[str] = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return ours, blockers
    for e in entries:
        if e.name not in FIXED_NAMES and classify(e.name, pats) is None:
            continue
        (ours if e.is_file(follow_symlinks=False) else blockers).append(e.name)
    return sorted(ours), sorted(blockers)


def clear_previous(folder: Path, names: list[str]) -> list[str]:
    """Remove `names` (from `scan_folder`) from `folder`. A name is removed only when it is still a plain file and its resolved
    parent is the folder. Returns what was removed."""
    root = folder.resolve()
    gone = []
    for n in names:
        p = folder / n
        try:
            if _stat.S_ISREG(os.lstat(p).st_mode) and p.resolve().parent == root:
                p.unlink()
                gone.append(n)
        except OSError:
            pass
    return gone


def list_outputs(folder: Path, stem: str, rid: str) -> list[dict]:
    """The plain files an export left in `folder`, as `ExportOutput`: `{name, kind, size, url}`. `url` is the relative path of
    row 44 (the page adds the media key); kinds that row 44 does not serve get an empty url."""
    pats = name_patterns(stem)
    rows = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return rows
    for e in entries:
        kind = classify(e.name, pats)
        if kind is None or not e.is_file(follow_symlinks=False):
            continue
        rows.append({"name": e.name, "kind": kind, "size": e.stat(follow_symlinks=False).st_size,
                     "url": f"api/runs/{rid}/files/{quote(e.name)}" if kind in SERVED_KINDS else ""})
    rows.sort(key=lambda r: (OUTPUT_KINDS.index(r["kind"]), r["name"]))
    return rows


def check_name(name: str) -> str:
    """Row 44 and 45 name rule: a plain base name. No `/`, `\\` or NUL, not `.` or `..`, no leading `.` or `-`, at most 255 bytes."""
    if (not isinstance(name, str) or not name or "\0" in name or "/" in name or "\\" in name or name != os.path.basename(name)
            or name[0] in ".-" or len(name.encode("utf-8", "replace")) > MAX_NAME_BYTES):
        raise HttpError(400, "bad file name")
    return name


# ---------------------------------------------------------------- where each run's last export went
_dirs: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _remember(app, rid: str, folder: Path) -> None:
    with app.lock:
        _dirs.setdefault(app, {})[rid] = folder


LAST_DIR_FILE = "export_dir.json"  # in the run folder: where a chosen out_dir export went, so a restart does not forget it


def _persist_dir(run_dir: Path, folder: Path, default: bool) -> None:
    """Record a chosen export folder next to segments.json (or drop the record after an export to the default folder)."""
    f = run_dir / LAST_DIR_FILE
    try:
        if default:
            f.unlink(missing_ok=True)
        else:
            serve_edits.write_atomic(f, json.dumps({"out_dir": str(folder)}).encode())
    except OSError:
        pass  # the in-memory record still works for this server


def _stored_dir(run_dir: Path) -> Path | None:
    """The folder `_persist_dir` recorded, when the file is a plain file naming a folder that `check_read` accepts."""
    d = serve_edits.read_json_file(run_dir / LAST_DIR_FILE)
    v = d.get("out_dir") if isinstance(d, dict) else None
    if not isinstance(v, str):
        return None
    try:
        return serve_paths.check_read(v, "dir")
    except HttpError:
        return None


def export_folders(app, rid: str, run_dir: Path) -> list[Path]:
    """The folders row 43 to 45 look in, in order: the folder of this run's last export (when it was not the default; remembered in
    memory and in the run folder), `<run>/edited/`, then the run folder itself (row 44 and 45 only)."""
    with app.lock:
        last = _dirs.get(app, {}).get(rid)
    if last is None:
        last = _stored_dir(run_dir)
    out = [last] if last is not None else []
    default = run_dir / EDITED
    if default not in out:
        out.append(default)
    return [f for f in out if serve_edits.is_real_dir(f)]


# ---------------------------------------------------------------- the plan (checked in the handler, run in the job)
@dataclass
class ExportPlan:
    rid: str
    run_dir: Path
    media: object
    doc: dict
    clips: list[dict]  # [{id, start, end, origin}] in time order
    revision: int
    sha: str
    samples: tuple
    keyframes: list[float]
    out_dir: Path
    default_dir: bool
    make_clips: bool
    reel: bool
    formats: list[str]
    music_spec: dict | None  # {track_id, file, auto: {mood, rank, track?} | None, cc0_only}
    beat_sync: bool
    to_photos: bool = False  # after the export, add its clips and reel to the Photos library (add-only; the files stay in out_dir)
    stem: str = field(init=False)
    music_warnings: list[str] = field(default_factory=list)  # what `automusic.choose` said, set by `resolve_music`, reported by the export

    def __post_init__(self):
        self.stem = Path(self.media.path).stem


def _bad(message: str, code: int = 422, **extra):
    raise HttpError(code, message, **extra)


def _flag(body: dict, key: str, default: bool) -> bool:
    v = body.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):
        _bad(f"{key} must be true or false", 400)
    return v


def current_clips(run_dir: Path, doc: dict, sha: str) -> tuple[list[dict], int]:
    """`(clips, revision)`: the user's edits when there are any, else the detector's clips (revision 0). The edits file is
    untrusted, so its clips go through `validate_clips` again."""
    offset, end_time = serve_edits.source_bounds(doc)
    edits = serve_edits.load_edits(run_dir)
    if edits is not None:
        return serve_edits.validate_clips(edits["clips"], offset, end_time), edits["revision"]
    clips = [c for c in serve_edits.detector_edits_doc(doc, sha)["clips"] if math.isfinite(c["start"]) and math.isfinite(c["end"]) and c["end"] > c["start"]]
    return clips, 0


def _select(clips: list[dict], ids) -> list[dict]:
    if ids is None:
        return clips
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        _bad("clip_ids must be null or a list of clip ids", 400)
    known = {c["id"] for c in clips}
    missing = [i for i in ids if i not in known]
    if missing:
        _bad(f"no such clip: {missing[0]}")
    want = set(ids)
    return [c for c in clips if c["id"] in want]


def _auto_spec(raw) -> dict:
    """`{mood, rank}` of an auto music request, and `track` when the caller names the song its preview showed: the mood is at most 40
    letters, digits, space, `_` or `-` ("" = from the reel), the rank a whole number from 0 to 63 and the track a track id (400 otherwise)."""
    from . import automusic

    if not isinstance(raw, dict):
        _bad("music.auto must be an object {mood, rank}", 400)
    mood, rank = raw.get("mood", ""), raw.get("rank", 0)
    if not isinstance(mood, str) or not automusic.MOOD_RE.fullmatch(mood):
        _bad("mood: at most 40 letters, digits, spaces, _ or -", 400)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= 63:
        _bad("rank must be a whole number from 0 to 63", 400)
    out = {"mood": mood, "rank": rank}
    track = raw.get("track")
    if track is not None:
        from . import music

        if not isinstance(track, str) or not music.ID_RE.match(track):
            _bad("track must be the track id the preview showed", 400)
        out["track"] = track
    return out


def auto_shape(doc: dict, media, clips: list[dict]):
    """The reel's shape from the document alone, for the preview and the export alike, so both choose the same song."""
    from . import automusic

    factor = float((doc.get("slowmo") or {}).get("factor") or 1.0)
    return automusic.shape_from_doc(doc, clips, float(media.info.fps), factor)


def plan_export(app, rid: str, body: dict) -> ExportPlan:
    """Check an export request against the run and turn it into a plan. Cheap: nothing is written, nothing is downloaded.
    Errors are `HttpError`: 400 (shape), 403/404/415 (a path), 422 (the domain refuses), 409 `exists`."""
    from . import music as music_mod

    run_dir, doc, media = serve_runs.load_run(app, rid)
    if doc.get("error"):
        _bad(f"This run failed, so there is nothing to export: {str(doc['error'])[:200]}")
    sha = serve_edits.segments_sha256(run_dir)
    all_clips, revision = current_clips(run_dir, doc, sha)
    clips = _select(all_clips, body.get("clip_ids"))
    if not clips:
        _bad("Nothing to export: the clip list is empty")
    make_clips, reel = _flag(body, "clips", True), _flag(body, "reel", False)
    beat_sync, replace = _flag(body, "beat_sync", False), _flag(body, "replace", False)
    lists = body.get("edit_lists")
    if lists is not None and not (isinstance(lists, list) and all(isinstance(x, str) for x in lists)):
        _bad("edit_lists must be a list of format names", 400)
    try:
        formats = editlist.parse_formats(lists)
    except ValueError as e:
        _bad(str(e))
    if not (make_clips or reel or formats):
        _bad("Nothing to export: choose clips, a reel or an edit list")
    spec = body.get("music")
    if spec is not None:
        if not isinstance(spec, dict):
            _bad("music must be null or an object", 400)
        tid, mfile, auto = spec.get("track_id") or None, spec.get("file") or None, spec.get("auto")
        cc0_only = spec.get("cc0_only", False)
        if not isinstance(cc0_only, bool) or not all(v is None or isinstance(v, str) for v in (tid, mfile)):
            _bad("music: track_id and file are text, cc0_only is true or false", 400)
        if auto is not None:
            auto = _auto_spec(auto)
        if [tid, mfile, auto].count(None) != 2:
            _bad("music needs exactly one of track_id, file or auto")
        if tid is not None and not music_mod.ID_RE.match(tid):
            _bad(f"music: {tid!r} is not a track id")
        spec = {"track_id": tid, "file": str(serve_paths.check_audio(mfile)) if mfile else None, "auto": auto, "cc0_only": cc0_only}
        if not reel:
            _bad("music needs the reel")
    if beat_sync and spec is None:
        _bad("beat sync needs music")
    to_photos = _flag(body, "to_photos", False)
    if to_photos:
        from . import photosave

        if not photosave.available():
            _bad("Saving to Photos needs a Mac")
        if not (make_clips or reel):
            _bad("Saving to Photos needs clips or a reel")
    out = body.get("out_dir")
    if out is None:
        out_dir, default_dir = run_dir / EDITED, True
        serve_paths.check_write(out_dir)
        if os.path.lexists(out_dir) and not serve_edits.is_real_dir(out_dir):
            _bad(f"{out_dir} is not a plain folder")
    else:
        if not isinstance(out, str) or not out:
            _bad("out_dir must be null or an absolute folder path", 400)
        out_dir, default_dir = serve_paths.check_write(out), False
        if out_dir.exists() and not out_dir.is_dir():
            _bad(f"{out_dir} is not a folder")
    stem = Path(media.path).stem
    ours, blockers = scan_folder(out_dir, stem)
    if blockers:
        _bad(f"{out_dir} holds {blockers[0]}, which an export would overwrite and which is not a plain file")
    if "segments.json" in ours:  # only an earlier export of ours may be replaced; a run's own segments.json never
        earlier = serve_edits.read_json_file(out_dir / "segments.json")
        if earlier is None or "edited" not in earlier:
            _bad(f"{out_dir} holds a run (segments.json). Choose another folder.")
    running = app.jobs.current()
    if running is not None:
        raise Busy(running)
    if not replace and any(n != "segments.json" for n in ours):
        raise HttpError(409, "exists", export=serve_edits.read_json_file(out_dir / "export.json"))
    return ExportPlan(rid, run_dir, media, doc, clips, revision, sha, serve_edits.load_samples_arrays(run_dir), list(media.info.keyframes),
                      out_dir, default_dir, make_clips, reel, formats, spec, beat_sync, to_photos)


# ---------------------------------------------------------------- the job
def resolve_music(spec: dict | None, plan: ExportPlan | None = None, progress=None):
    """`(audio file, license metadata or None)` for the request's music, or None. A track id is downloaded (CC0 or CC BY unless `cc0_only`)
    into the music cache; a file was checked by `check_audio` already, and its license record is checked here (CC0 or CC BY, CC BY not with `cc0_only`).
    `auto` asks `automusic.choose` with the shape of the plan's clips (the same call and cache as the preview) and keeps its summary as `meta["auto"]`.
    A `MusicError` ends the job before anything is cleared. While another download holds the lock (a cancelled auto job's thread still
    fetching songs), `progress` is called every 0.1 s, so the job shows it is waiting and a cancel takes effect."""
    if spec is None:
        return None
    from . import music

    if spec.get("auto"):
        from . import automusic

        while not _download.acquire(timeout=0.1):
            if progress:
                progress(0.0, "waiting for the earlier song download to finish")  # raises when the job was cancelled
        try:
            choice = automusic.choose(auto_shape(plan.doc, plan.media, plan.clips), spec["auto"]["mood"], spec["cc0_only"], spec["auto"]["rank"])
        finally:
            _download.release()
        got = str(choice.meta.get("id") or choice.audio.stem)
        want = spec["auto"].get("track")
        if want and got != want:  # the list moved since the preview (a song left the cache, Openverse is down): never swap songs silently
            raise music.MusicError(f"the song you previewed ({want}) is not what auto music picks now ({choice.meta.get('title') or got}); "
                                   "choose a song again")
        if plan is not None:
            plan.music_warnings = list(choice.warnings)
        return choice.audio, dict(choice.meta, auto=choice.summary())
    if spec["file"]:
        p = Path(spec["file"])
        meta = music.load_meta(p)
        music.check_file_license(meta, p.name, spec["cc0_only"])
        return p, meta
    audio = music.get(spec["track_id"], cc0_only=spec["cc0_only"])
    return audio, music.load_meta(audio)


def _iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _markers_inside(doc: dict, start: float, end: float) -> list[dict]:
    """The sound markers of the run's own clips that fall inside [start, end] (each once)."""
    seen, out = set(), []
    for c in doc.get("clips") or []:
        for m in (c.get("markers") or []) if isinstance(c, dict) else []:
            t = m.get("time") if isinstance(m, dict) else None
            if isinstance(t, (int, float)) and not isinstance(t, bool) and start <= t <= end and (t, m.get("kind")) not in seen:
                seen.add((t, m.get("kind")))
                out.append(m)
    return sorted(out, key=lambda m: m["time"])


def export_edits(plan: ExportPlan, music=None, progress=None) -> dict:
    """Section 2.7: clear what an earlier export left (by pattern), cut the clips, write the sidecars and the LosslessCut CSV, then the
    reel and edit lists, then `segments.json` and `export.json` of the export folder. Returns the export.json document (with
    `outputs`). A clip that fails to cut gets `error` and the export goes on. `progress(fraction, message)` may raise to cancel."""
    prog = progress or (lambda f, m="": None)
    doc, stem, out = plan.doc, plan.stem, plan.out_dir
    video = str(plan.media.path)
    suffix = Path(video).suffix.lower() or ".mov"
    times, scores, boxes, track_ids = plan.samples
    settings = doc.get("settings") if isinstance(doc.get("settings"), dict) else {}
    factor = float((doc.get("slowmo") or {}).get("factor") or 1.0)
    slowmo_audio = bool(settings.get("slowmo_audio"))
    make_cut = plan.make_clips or plan.reel
    n = len(plan.clips)
    prog(0.0, "preparing")
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    ours, blockers = scan_folder(out, stem)
    if blockers:
        raise ExportError(f"{out} holds {blockers[0]}, which an export would overwrite and which is not a plain file")
    clear_previous(out, ours)

    warnings = list(doc.get("warnings") or []) + list(plan.music_warnings)
    clips: list[dict] = []
    failed = 0
    cut_end = 0.8 if (plan.reel or plan.formats) else 0.92
    for i, c in enumerate(plan.clips, 1):
        s, e = c["start"], c["end"]
        name = cut.clip_name(stem, i, s, e, suffix)
        clip = dict(index=i, start=s, end=e, file=name, **serve_edits.edited_clip_facts(s, e, times, scores, plan.keyframes))
        markers = _markers_inside(doc, s, e)
        if markers:
            clip["markers"] = markers
        if make_cut:
            prog(0.03 + (cut_end - 0.03) * (i - 1) / n, f"cutting clip {i} of {n}")
            try:
                if os.path.lexists(out / name):
                    raise ExportError(f"{name} is in the way and is not a plain file")
                clip["expected_copy_start"] = round(cut.cut_clip(video, out / name, s, e, factor, slowmo_audio), 3)  # measured; was the keyframe guess
                got = cut.written_duration(str(out / name))
                if got is not None and factor <= 1.0:
                    clip["file_duration_s"] = round(got, 3)  # what the sidecar's clip.duration_s says
                warn = cut.duration_warning(str(out / name), got, clip["expected_copy_start"], e, factor)
                if warn:
                    warnings.append(f"clip {i}: {warn}")
                if factor > 1.0:
                    clip["slowmo"] = dict(factor=factor, source_t0=clip["expected_copy_start"], clip_start_s=0.0, **cut.retimed_facts(str(out / name)))
                    off = clip["slowmo"]["av_offset_ms"]
                    if off is not None and abs(off) > 21.0:
                        warnings.append(f"clip {i}: audio starts {off:g} ms from the picture (limit 21 ms)")
            except Exception as exc:  # keep going, as run_cut does; the clip says what went wrong
                clip["error"] = str(exc)
                failed += 1
        clips.append(clip)
    good = [c for c in clips if "error" not in c]
    if failed:
        warnings.append(f"{failed} of {n} clips could not be cut")

    ranges = [(c["start"], c["end"]) for c in clips]
    edited_doc = deepcopy(doc)
    edited_doc.pop("reel", None)
    edited_doc.pop("edit_lists", None)
    edited_doc["source"]["path"] = video
    edited_doc["clips"] = clips
    edited_doc["presence_intervals"] = [[s, e] for s, e in ranges]  # a clip the user keeps is, by definition, where the target is
    edited_doc["edited"] = {"from": "../segments.json" if plan.default_dir else str(plan.run_dir / "segments.json"),
                            "revision": plan.revision, "segments_sha256": plan.sha}
    if plan.make_clips and good:  # a reel-only export keeps no clip files, so no sidecars or CSV that point at them
        prog(cut_end, "sidecars")
        try:
            cut.write_losslesscut_csv(out / f"{safe_stem(stem)}_losslesscut.csv", [(c["start"], c["end"], c["file"]) for c in good])
            sidecar.write_sidecars(out, edited_doc, times=times, scores=scores, boxes=boxes, track_ids=track_ids, presence=ranges,
                                   target=settings.get("target") or "unknown", embedder=(settings.get("target_info") or {}).get("embedder"),
                                   detector=dict(mode=doc.get("mode") or settings.get("mode"), backend=settings.get("detector"),
                                                 model=settings.get("model"), tracker=settings.get("tracker")))
        except Exception as exc:  # the clips are good; say what is missing
            warnings.append(f"sidecars not written ({type(exc).__name__}: {exc})")
    ex: dict = dict(reel=None, edit_lists=[], warnings=[])
    if plan.reel or plan.formats:
        prog(cut_end + 0.02, "reel and edit lists")
        ex = export_mod.export_run(edited_doc, out, out, plan.reel, plan.formats, music, plan.beat_sync)
        if plan.reel:
            edited_doc["reel"] = ex["reel"]
        if plan.formats:
            edited_doc["edit_lists"] = ex["edit_lists"]
        warnings += ex["warnings"]
    if make_cut and not plan.make_clips:  # clips were cut only as the reel's input: the user unticked Clips
        if (ex["reel"] or {}).get("ok"):
            for c in good:  # only files this export cut
                try:
                    (out / c["file"]).unlink(missing_ok=True)
                except OSError as exc:
                    warnings.append(f"{c['file']} could not be removed ({exc})")
                    continue
                # segments.json and export.json say what is really in the folder: this clip's file is gone ("kept": false). It is not
                # an "error": the clip was cut and is still a clip of the run, so edit lists (which point at the source) still use it
                c.pop("file", None)
                c["kept"] = False
        else:  # the reel failed: the clips are all the user has, so they stay
            warnings.append("the reel was not built, so the cut clips were kept")
    edited_doc["warnings"] = warnings
    prog(0.97, "writing segments.json and export.json")
    serve_edits.write_atomic(out / "segments.json", json.dumps(edited_doc, indent=1).encode())
    result = {"exported": _iso(), "revision": plan.revision, "dir": str(out), "n_clips": n,
              "clips": [{k: c[k] for k in ("index", "start", "end", "file", "kept", "error", "sidecar") if k in c} for c in clips],
              "reel": ex["reel"], "edit_lists": ex["edit_lists"], "warnings": warnings, "outputs": list_outputs(out, stem, plan.rid)}
    serve_edits.write_atomic(out / "export.json", json.dumps(result, indent=1).encode())
    return result


def save_to_photos(plan: ExportPlan, result: dict, progress=None) -> dict:
    """Add the export's clip files and reel to the Photos library and record the answer as `result["photos"]` (also in export.json).
    A denied permission or a failed file never fails the export: the files are in the folder, and `photos.message` says what happened."""
    from . import photosave

    if progress:
        progress(0.98, "saving to Photos")
    # A reel-only export must not put the clips the person did not choose into the library, even when the reel failed and
    # export_edits kept the clips as a fallback: then there is nothing to save, and the answer says so.
    doc = result if plan.make_clips else {"reel": result.get("reel")}
    files = photosave.videos_of_export(plan.out_dir, doc)
    if not files:
        photos = {"status": "failed", "authorization": "unknown", "message": "No clip or reel file to save to Photos.", "saved": [], "failed": []}
    else:
        try:
            photos = photosave.save(files).to_json()
        except photosave.PhotosSaveError as e:
            photos = {"status": "failed", "authorization": "unknown", "message": f"Nothing was saved to Photos: {e}", "saved": [],
                      "failed": [{"name": f.name, "error": str(e)} for f in files]}
    result["photos"] = photos
    if photos["status"] not in ("saved", "skipped"):
        result["warnings"] = list(result.get("warnings") or []) + [photos["message"]]
    try:
        serve_edits.write_atomic(plan.out_dir / "export.json", json.dumps(result, indent=1).encode())
    except OSError:
        pass  # the job result still carries it
    return result


def job_fn(plan: ExportPlan):
    def fn(progress):
        music = resolve_music(plan.music_spec, plan, progress)  # before anything is cleared: a music error costs nothing
        result = export_edits(plan, music, progress)
        return save_to_photos(plan, result, progress) if plan.to_photos else result
    return fn


# ---------------------------------------------------------------- routes
def export_post(app, req: Request) -> Response:
    rid = req.params["rid"]
    plan = plan_export(app, rid, req.json if isinstance(req.json, dict) else {})
    job = app.jobs.start("export", job_fn(plan), media_id=plan.media.id, run_id=rid)
    _remember(app, rid, plan.out_dir)
    _persist_dir(plan.run_dir, plan.out_dir, plan.default_dir)
    return json_response({"job": job})


# ---------------------------------------------------------------- auto music (a job, so it shows progress and can be cancelled)
def auto_answer(choice) -> dict:
    """What the preview job returns: the choice and the track card (credit text when the license asks for one)."""
    from . import music

    m = choice.meta
    needs = music.needs_credit(m.get("license"))
    track = {"id": m.get("id") or choice.audio.stem, "title": m.get("title"), "creator": m.get("creator"), "license": m.get("license"),
             "license_version": m.get("license_version"), "license_url": m.get("license_url"), "duration_s": m.get("duration_s"),
             "bpm": choice.bpm, "score": choice.score.get("score"), "needs_credit": needs,
             "credit_text": music.credits_text(m) if needs else None, "builtin": choice.builtin, "bundled": bool(m.get("bundled"))}
    return {"rank": choice.rank, "n": choice.n, "wrapped": choice.wrapped, "mood": choice.mood, "speed": choice.speed,
            "warnings": choice.warnings, "notice": LIBRARY_NOTICE if track["bundled"] else music.OPENVERSE_NOTICE, "track": track}


def music_auto_post(app, req: Request) -> Response:
    """Row: POST /api/runs/{rid}/music/auto starts a `music-auto` job for the run's clips (all, or `clip_ids`); its result is `auto_answer`.
    Busy (409) while another job runs. 400 for a bad body, 422 for a run that failed; a `MusicError` ends the job `failed`."""
    from . import automusic

    rid = req.params["rid"]
    body = req.json if isinstance(req.json, dict) else {}
    cc0_only = _flag(body, "cc0_only", False)
    auto = _auto_spec({"mood": body.get("mood", ""), "rank": body.get("rank", 0)})
    run_dir, doc, media = serve_runs.load_run(app, rid)
    if doc.get("error"):
        _bad(f"This run failed, so there is no reel to choose music for: {str(doc['error'])[:200]}")
    clips = _select(current_clips(run_dir, doc, serve_edits.segments_sha256(run_dir))[0], body.get("clip_ids"))
    if not clips:
        _bad("Nothing to choose music for: the clip list is empty")
    shape = auto_shape(doc, media, clips)

    def fn(progress):
        wait = "choosing a song (the first time downloads up to 4 songs)"
        progress(0.05, wait)
        box: dict = {}

        def work():  # the downloads cannot be interrupted, so they run beside the job, which watches for a cancel
            try:
                with _download:
                    box["choice"] = automusic.choose(shape, auto["mood"], cc0_only, auto["rank"])
            except BaseException as e:  # handed to the job below
                box["error"] = e

        t = threading.Thread(target=work, name="inframe-music-auto", daemon=True)
        t.start()
        while t.is_alive():
            t.join(0.1)
            progress(0.05, wait)  # raises when the job was cancelled; the songs fetched so far stay in the cache
        if "error" in box:
            raise box["error"]
        progress(0.95, "chosen")
        return auto_answer(box["choice"])

    return json_response({"job": app.jobs.start("music-auto", fn, media_id=media.id, run_id=rid)})


def export_get(app, req: Request) -> Response:
    rid = req.params["rid"]
    run_dir, _, _ = serve_runs.load_run(app, rid)
    for folder in export_folders(app, rid, run_dir):
        doc = serve_edits.read_json_file(folder / "export.json")
        if doc is not None:
            return json_response(doc)
    raise HttpError(404, "this run has not been exported yet")


def _find(app, rid: str, run_dir: Path, media, name: str, kinds: tuple[str, ...]) -> Path:
    """The plain file called `name` in the export folder or the run folder, found by scanning them, else 404."""
    pats = name_patterns(Path(media.path).stem)
    folders = export_folders(app, rid, run_dir) + [run_dir]
    for folder in folders:
        try:
            entries = os.scandir(folder)
        except OSError:
            continue
        with entries:
            for e in entries:
                if e.name != name or classify(e.name, pats) not in kinds or not e.is_file(follow_symlinks=False):
                    continue
                p = Path(e.path)
                if p.resolve().parent == folder.resolve():
                    return p
    raise HttpError(404, f"no such file: {name}")


def files_get(app, req: Request) -> Response:
    rid = req.params["rid"]
    run_dir, _, media = serve_runs.load_run(app, rid)
    p = _find(app, rid, run_dir, media, check_name(req.params["name"]), SERVED_KINDS)
    return file_response(p, CONTENT_TYPES.get(p.suffix.lower(), "application/octet-stream"), req)


def _platform() -> str:
    return sys.platform


def reveal_command(path: Path, is_dir: bool) -> list[str]:
    """The file manager command for an absolute `path`. A file is shown selected in its folder; on macOS a folder is shown selected too, never opened."""
    p = str(path)
    plat = _platform()
    if plat == "darwin":
        return ["open", "-R", "--", p]  # always -R: `open <folder>` starts a folder named *.app (a shared run folder can be one)
    if plat.startswith("win"):
        return ["explorer", p] if is_dir else ["explorer", f"/select,{p}"]
    return ["xdg-open", p if is_dir else str(path.parent)]


def reveal_post(app, req: Request) -> Response:
    rid = req.params["rid"]
    run_dir, _, media = serve_runs.load_run(app, rid)
    body = req.json if isinstance(req.json, dict) else {}
    name = body.get("name")
    if name is None:
        target, is_dir = (export_folders(app, rid, run_dir) or [run_dir])[0], True
    else:
        target, is_dir = _find(app, rid, run_dir, media, check_name(name), OUTPUT_KINDS), False
    cmd = reveal_command(target.resolve(), is_dir)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=REVEAL_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        raise HttpError(501, "no file manager could be started here") from None
    if r.returncode != 0 and not _platform().startswith("win"):  # explorer exits 1 even when it worked
        raise HttpError(500, f"the file manager failed: {(r.stderr or '').strip()[:200]}")
    return json_response({"ok": True})


ROUTES = [
    Route("POST", rf"/api/runs/{RID}/export", export_post),
    Route("POST", rf"/api/runs/{RID}/music/auto", music_auto_post),
    Route("GET", rf"/api/runs/{RID}/export", export_get),
    Route("GET", rf"/api/runs/{RID}/files/(?P<name>[\s\S]+)", files_get, media=True),
    Route("POST", rf"/api/runs/{RID}/reveal", reveal_post),
]


def attach(app) -> None:
    """Nothing to set up: the run registry belongs to `serve_runs`, and where each run's last export went is kept here per app."""
