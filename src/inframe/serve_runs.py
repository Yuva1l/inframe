"""Runs and edits routes for `inframe serve` (docs/T9A_DESIGN.md section 3.3 row 34 and section 3.4 rows 35-41).

Row 35 starts a `run_cut` job (one target per run, in `<stem>_inframe/<slug>/`); 36-38 list a video's runs and read one run and its
samples; 39-40 save and revert the user's clip edits; 41 previews a re-segment; 34 adds crops of a run's person (or a photos folder)
to a saved target. The disk rules (untrusted run folders, the edits file) are in `serve_edits`.
Imports `serve_http`, `serve_paths`, `serve_jobs` and `serve_edits`, never `serve.py`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from . import classes, identity, models, pipeline, serve_edits, serve_paths, targets, tracks
from .activity import check_activity
from .cut import parse_slowmo
from .segments import build_segments
from .serve_http import HttpError, Request, Response, Route, json_response
from .serve_jobs import Busy, NeedsConfirm
from .tap import TapError, resolve_preset
from .zone import Zone, check_overlap, zone_from_target_files

MID = r"(?P<id>[0-9a-f]{12})"
RID = r"(?P<rid>[0-9a-f]{12})"
TARGET_KINDS = ("any", "class", "tap", "saved", "photos", "motion")
ADVANCED_KEYS = ("pad", "min_len", "merge_gap", "enter", "exit", "min_box", "tracker", "activity", "fps")
SAME_TOL_S = serve_edits.SAME_TOL_S


# ---------------------------------------------------------------- small checks
def _bad(message: str, code: int = 422, **extra):
    raise HttpError(code, message, **extra)


def _number(v, name: str, lo: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        _bad(f"{name} must be a finite number")
    if lo is not None and v < lo:
        _bad(f"{name} must be at least {lo:g} (got {v:g})")
    return float(v)


def _opt_number(v, name: str, lo: float | None = None) -> float | None:
    return None if v is None else _number(v, name, lo)


def _slug_part(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")[:48] or "x"


def _body(req: Request) -> dict:
    return req.json if isinstance(req.json, dict) else {}


def media_of_run(app, doc: dict):
    """The opened media this run belongs to (its `source.path`), else 404: a run of a video nobody opened is not served."""
    with app.lock:
        opened = list(app.media.values())
    for m in opened:
        if serve_edits.source_matches(doc, m.path):
            return m
    raise serve_edits.RunFolderError("this run's video is not open in inframe serve")


def load_run(app, rid: str):
    """`(run_dir, segments doc, media)` of a registered run, re-checked on every request (the folder is untrusted)."""
    d = app.run_or_404(rid)
    doc = serve_edits.read_segments(d)
    return d, doc, media_of_run(app, doc)


# ---------------------------------------------------------------- row 35: start a run
@dataclass
class RunPlan:
    media: object
    run_dir: Path
    settings: pipeline.Settings
    target: object
    kind: str


def _slowmo_and_numbers(body: dict) -> dict:
    adv = body.get("advanced")
    adv = {} if adv is None else adv
    if not isinstance(adv, dict):
        _bad("advanced must be an object", 400)
    unknown = sorted(set(adv) - set(ADVANCED_KEYS))
    if unknown:
        _bad(f"unknown advanced setting(s): {', '.join(unknown)}")
    d = pipeline.Settings()
    kw: dict = {}
    for k in ("pad", "min_len", "merge_gap"):
        if adv.get(k) is not None:
            kw[k] = _number(adv[k], k, 0.0)
    for k in ("enter", "exit"):
        kw[k] = _opt_number(adv.get(k), k)
    kw["min_box"] = _number(adv["min_box"], "min_box", 0.0) if adv.get("min_box") is not None else d.min_box
    if adv.get("fps") is not None:
        kw["fps"] = _number(adv["fps"], "fps", 0.0)
    if adv.get("tracker") is not None:
        if adv["tracker"] not in tracks.KINDS:
            _bad(f"tracker must be one of {', '.join(tracks.KINDS)}")
        kw["tracker"] = adv["tracker"]
    try:
        if adv.get("activity") is not None:
            kw["activity"] = check_activity(adv["activity"])
        slowmo = parse_slowmo("off" if body.get("slowmo") is None else body["slowmo"])
    except ValueError as e:
        _bad(str(e))
    audio = body.get("slowmo_audio", False)
    if not isinstance(audio, bool):
        _bad("slowmo_audio must be true or false")
    if audio and slowmo == "off":
        _bad("keeping the sound needs slow motion (auto or a factor)")
    msg = pipeline.check_ranges(kw["enter"], kw["exit"], kw["min_box"])
    if msg:
        _bad(msg)
    if kw["exit"] is not None and kw["enter"] is not None and kw["exit"] > kw["enter"]:
        _bad(f"exit ({kw['exit']:g}) must not exceed enter ({kw['enter']:g})")
    kw["slowmo"], kw["slowmo_audio"] = slowmo, audio
    return kw


def plan_run(app, body: dict) -> RunPlan:
    """Check a RunRequest and turn it into a RunPlan. Errors are HttpError: 400 (shape), 404 (unknown media or photos), 403 (a path
    the server refuses), 422 (the domain code refused the settings or the target). Cheap: no model is loaded."""
    media = app.media_or_404(body.get("media_id") if isinstance(body.get("media_id"), str) else "")
    t = body.get("target")
    t = {"kind": "any"} if t is None else t
    if not isinstance(t, dict):
        _bad("target must be an object", 400)
    kind = t.get("kind", "any")
    if kind not in TARGET_KINDS:
        _bad(f"target.kind must be one of {', '.join(TARGET_KINDS)}")
    kw = _slowmo_and_numbers(body)
    preset = body.get("preset")
    if preset is not None and preset not in pipeline.PRESETS:
        _bad(f"preset must be one of {', '.join(pipeline.PRESETS)}")
    if kind == "motion" and kw.get("activity", "any") != "any":
        _bad("activity riding needs a detector run, not motion mode")
    out_root = serve_paths.check_write(app.out_root(media))
    specs: list[str] = []
    slug = {"any": "any-person", "tap": "tap", "photos": "photos", "motion": "motion-ui"}.get(kind, "")  # not "motion": the CLI default <stem>_inframe/motion
    tap_bytes = None
    cls_name = "person"
    if kind == "class":
        cn = t.get("class")
        if not isinstance(cn, str) or not cn.strip():
            _bad("target.class names the class (a COCO class such as dog)")
        try:
            cls_name = classes.class_name(classes.class_id(cn))
        except ValueError as e:
            _bad(str(e))
        specs, slug = [f"class:{cls_name}"], f"class-{_slug_part(cls_name)}"
    elif kind == "saved":
        name = t.get("name")
        try:
            targets.check_name(name)
        except ValueError as e:
            _bad(str(e))
        specs, slug = [f"saved:{name}"], f"saved-{_slug_part(name)}"
    elif kind == "photos":
        pid = t.get("photos_id")
        with app.lock:
            folder = app.photos_dirs.get(pid) if isinstance(pid, str) else None
        if folder is None:
            _bad("no such photos set: choose the photos again", 404)
        specs = [f"photos:{serve_paths.check_read(folder, 'dir')}"]
    elif kind == "tap":
        cn = t.get("class", "person")
        try:
            cls_name = classes.class_name(classes.class_id(cn if isinstance(cn, str) else "person"))
        except ValueError as e:
            _bad(str(e))
        key = f"photos:{t['photos_id']}" if t.get("photos_id") else f"saved:{t['name']}" if t.get("name") else ""
        src = serve_paths.target_file(out_root, cls_name, key)
        if serve_edits.read_regular(src, 1024 * 1024) is None:
            _bad("No taps yet: tap the person in a frame first.")
        tap_bytes = serve_edits.check_target_json(src)
    running = app.jobs.current()
    if running is not None:  # busy: answer before anything is written
        raise Busy(running)
    if tap_bytes is not None:
        serve_edits.write_atomic(out_root / "target.json", tap_bytes)
        specs = [f"tap:{out_root / 'target.json'}"]
    try:
        preset_name = resolve_preset(specs, preset) if kind == "tap" else (preset or "balanced")
        zone = Zone.of(body["zone"]) if body.get("zone") else (zone_from_target_files(specs) if kind == "tap" else None)
        overlap = check_overlap(body.get("zone_overlap"))
        if zone is None and overlap is not None:
            _bad("zone_overlap needs a zone")
        settings = pipeline.Settings(mode="motion" if kind == "motion" else "detect", preset=preset_name, zone=tuple(zone.as_list()) if zone else None,
                                     zone_overlap=overlap, **kw)
        target = None
        if specs:
            target = identity.parse_targets(specs, identity.DEFAULT_EMBEDDER, app.offline,
                                            identity.lazy_detect(pipeline.PRESETS[preset_name].model))
            if hasattr(target, "min_box"):
                target.min_box = settings.min_box
    except targets.StaleTarget as e:
        raise HttpError(422, str(e), stale=True) from None
    except (ValueError, TapError, models.ModelsMissing, identity.EmbedderUnavailable) as e:
        raise HttpError(422, str(e)) from None
    return RunPlan(media, serve_edits.run_dir_for(out_root, slug), settings, target, kind)


def run_job(app, plan: RunPlan, rid: str):
    """The job function: `run_cut` in the run folder. An ambiguous photo run ends `needs_confirm`. Edits made on the earlier run
    are moved aside once `segments.json` was replaced (also by a failed or cancelled-during-cuts run), never before."""
    def fn(progress):
        before = _sha_or_none(plan.run_dir)
        try:
            doc = pipeline.run_cut(str(plan.media.path), str(plan.run_dir), plan.settings, plan.target, progress)
        except identity.Ambiguous as e:
            raise NeedsConfirm(str(e), {"reason": "ambiguous", "run_id": rid, "media_id": plan.media.id}) from None
        finally:
            if _sha_or_none(plan.run_dir) != before:
                serve_edits.revert_edits(plan.run_dir)
        return {"run_id": rid, "media_id": plan.media.id, "n_clips": len(doc["clips"]), "warnings": doc.get("warnings", []),
                "timing_s": doc.get("timing_s")}
    return fn


def _sha_or_none(run_dir) -> tuple | None:
    """What identifies this run's segments.json: its hash and its mtime, so a new run that writes the same bytes (a short
    clip whose rounded timings repeat) still counts as replacing it."""
    try:
        return serve_edits.segments_sha256(run_dir), (Path(run_dir) / "segments.json").stat().st_mtime_ns
    except (HttpError, OSError):
        return None


def runs_start(app, req: Request) -> Response:
    plan = plan_run(app, _body(req))
    rid = serve_edits.run_id_for(plan.run_dir)
    job = app.jobs.start("run", run_job(app, plan, rid), media_id=plan.media.id, run_id=rid)
    with app.lock:
        app.runs[rid] = plan.run_dir
    return json_response({"job": job, "run_id": rid})


# ---------------------------------------------------------------- rows 36-38: list, read, samples
def runs_list(app, req: Request) -> Response:
    media = app.media_or_404(req.params["id"])
    runs = serve_edits.list_runs(app.out_root(media), media.path)
    with app.lock:
        for r in runs:
            app.runs[r["run_id"]] = Path(r["dir"])
    return json_response({"runs": runs})


def run_get(app, req: Request) -> Response:
    d, doc, media = load_run(app, req.params["rid"])
    edits = serve_edits.load_edits(d)
    sha = serve_edits.segments_sha256(d)
    export = None
    if serve_edits.is_real_dir(d / "edited"):
        export = serve_edits.read_json_file(d / "edited" / "export.json")
    seg = {k: doc.get(k) for k in ("settings", "scores", "timing_s", "slowmo", "error")}
    seg.update({k: doc.get(k) or [] for k in ("warnings", "presence_intervals", "clips")})
    return json_response({"run_id": req.params["rid"], "media_id": media.id, "dir": str(d), "segments": seg, "edits": edits,
                          "edits_base_ok": edits is None or edits["base"]["segments_sha256"] == sha, "export": export})


def run_samples(app, req: Request) -> Response:
    d, doc, _ = load_run(app, req.params["rid"])
    times, scores, boxes, track_ids = serve_edits.load_samples_arrays(d)
    st = doc.get("settings") if isinstance(doc.get("settings"), dict) else {}
    return json_response({"t": times, "score": scores, "box": boxes, "track": track_ids,
                          "enter": st.get("enter"), "exit": st.get("exit"), "merge_gap": st.get("merge_gap"),
                          "min_len": st.get("min_len"), "pad": st.get("pad"), "sound": st.get("sound"), "mode": doc.get("mode")})


# ---------------------------------------------------------------- rows 39-41: edits and re-segment
def edits_put(app, req: Request) -> Response:
    d, _, _ = load_run(app, req.params["rid"])
    body = _body(req)
    rev, clips = body.get("revision"), body.get("clips")
    if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
        _bad("revision must be a whole number from 0", 400)
    if not isinstance(clips, list):
        _bad("clips must be a list", 400)
    return json_response(serve_edits.save_edits(d, rev, clips))


def edits_delete(app, req: Request) -> Response:
    d, _, _ = load_run(app, req.params["rid"])
    serve_edits.revert_edits(d)
    return json_response({"ok": True})


def _pairs(intervals) -> list[list[float]]:
    return [[round(s, 3), round(e, 3)] for s, e in intervals]


def _same(a: list[list[float]], b) -> bool:
    try:
        b = [[float(x[0]), float(x[1])] for x in b]
    except (TypeError, ValueError, IndexError, KeyError):
        return False
    return len(a) == len(b) and all(abs(p[0] - q[0]) <= SAME_TOL_S and abs(p[1] - q[1]) <= SAME_TOL_S for p, q in zip(a, b))


def resegment(app, req: Request) -> Response:
    """What `build_segments` makes from the run's own samples. A field that is missing (or null) is the run's `settings` value, so
    with no fields the preview is the run's own clips; nothing is written."""
    d, doc, _ = load_run(app, req.params["rid"])
    body = _body(req)
    st = doc.get("settings") if isinstance(doc.get("settings"), dict) else {}
    vals = {}
    for k in ("enter", "exit", "merge_gap", "min_len", "pad"):
        v = body.get(k)
        v = st.get(k) if v is None else v
        if v is None:
            _bad(f"{k}: the run has no value to start from; send one")
        vals[k] = _number(v, k, 0.0)
    if vals["exit"] > vals["enter"]:
        _bad(f"exit ({vals['exit']:g}) must not exceed enter ({vals['enter']:g})")
    times, scores, _, _ = serve_edits.load_samples_arrays(d)
    offset, end_time = serve_edits.source_bounds(doc)
    presence, clips = build_segments(times, scores, end_time, vals["enter"], vals["exit"], vals["merge_gap"], vals["min_len"], vals["pad"], offset)
    presence, clips = _pairs(presence), _pairs(clips)
    run_clips = [[c["start"], c["end"]] for c in doc.get("clips") or [] if isinstance(c, dict) and "start" in c and "end" in c]
    same = _same(clips, run_clips) and _same(presence, doc.get("presence_intervals") or [])
    if same:  # equal up to the csv's rounding: hand back the run's own numbers so applying the preview changes nothing
        presence, clips = [list(map(float, p)) for p in doc.get("presence_intervals") or []], run_clips
    return json_response({"presence": presence, "clips": clips, "same_as_run": same})


# ---------------------------------------------------------------- row 34: add crops to a saved target
def target_add(app, req: Request) -> Response:
    name = req.params["name"]
    try:
        targets.check_name(name)
    except ValueError as e:
        _bad(str(e))
    body = _body(req)
    source = body.get("from")
    media_id = run_id = None
    if source == "run":
        rid = body.get("run_id")
        d, doc, media = load_run(app, rid if isinstance(rid, str) else "")
        media_id, run_id = media.id, rid
        seg = str(d / "segments.json")

        def get_crops():
            return targets.crops_from_run(seg), "run"
    elif source == "photos":
        pid = body.get("photos_id")
        with app.lock:
            folder = app.photos_dirs.get(pid) if isinstance(pid, str) else None
        if folder is None:
            _bad("no such photos set: choose the photos again", 404)
        folder = serve_paths.check_read(folder, "dir")

        def get_crops():
            return targets.crops_from(f"photos:{folder}")
    else:
        _bad('from must be "run" or "photos"', 400)

    def fn(progress):
        progress(0.05, "taking crops")
        crops, label = get_crops()
        progress(0.5, f"adding {len(crops)} crops")
        return targets.add_crops(name, crops, label, identity.DEFAULT_EMBEDDER, app.offline)  # no progress call after the change: a cancel now would hide it

    return json_response({"job": app.jobs.start("target-add", fn, media_id=media_id, run_id=run_id)})


ROUTES = [
    Route("POST", r"/api/targets/(?P<name>[^/]+)/add", target_add),
    Route("POST", r"/api/runs", runs_start),
    Route("GET", rf"/api/media/{MID}/runs", runs_list),
    Route("GET", rf"/api/runs/{RID}", run_get),
    Route("GET", rf"/api/runs/{RID}/samples", run_samples),
    Route("PUT", rf"/api/runs/{RID}/edits", edits_put),
    Route("DELETE", rf"/api/runs/{RID}/edits", edits_delete),
    Route("POST", rf"/api/runs/{RID}/resegment", resegment),
]


def attach(app) -> None:
    """`App.run_or_404` and the `runs` registry already live in the App; this makes sure the registry exists."""
    if not hasattr(app, "runs"):
        app.runs = {}
