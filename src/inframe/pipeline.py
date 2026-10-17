"""decode -> detect -> target -> segments -> cut -> outputs."""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import __version__
from .names import safe_stem
from .cut import clip_name, cut_clip, duration_warning, parse_slowmo, resolve_slowmo, retimed_facts, snap_to_keyframe, write_losslesscut_csv, written_duration
from .decode import iter_samples
from .editlist import parse_formats
from .export import clear_old_exports, export_run
from .detect import BATCH, PERSON, available_sizes, for_tap, load_detector, load_fallback_detector
from .motion import auto_thresholds, run_motion
from .probe import VideoInfo, capture_fps, probe
from . import audio, classes, tracks
from .activity import THRESHOLD as ACTIVITY_THRESHOLD, ActivityTarget, check_activity
from .camera import check_static
from .segments import build_segments
from .sidecar import clear_old_sidecars, write_sidecars
from .target import AnyPerson, Target
from .zone import Zone, ZoneTarget, check_overlap


@dataclass(frozen=True)
class Preset:
    name: str
    model: str  # RF-DETR size (detect mode)
    detect_fps: float | None  # None = keyframes only
    motion_fps: float


PRESETS = {
    "fast": Preset("fast", "nano", None, 2.0),
    "balanced": Preset("balanced", "small", 5.0, 5.0),
    "accurate": Preset("accurate", "base", 5.0, 5.0),  # RF-DETR base (560 px): finds smaller people; about 1.5x the detector time of balanced
}
MODES = ("motion", "detect")
DETECT_ENTER, DETECT_EXIT = 0.5, 0.3  # person-score hysteresis
DECODE_LONG_SIDE = 640  # decoded long side of the sampled frames, unless the target asks for more (target.decode_long_side)


@dataclass
class Settings:
    mode: str = "detect"
    preset: str = "balanced"
    fps: float | None = None  # None = the preset's rate; 0 = keyframes only
    pad: float = 1.0
    min_len: float = 1.0
    merge_gap: float = 1.5
    enter: float | None = None  # None = default for the mode (motion: from the score distribution)
    exit: float | None = None
    min_box: float = 0.0
    tracker: str = "ocsort"  # "ocsort", or "botsort" when the camera pans; runs only for targets that want tracks, at 5+ samples/s
    reel: bool = False  # also join the clips into <stem>_reel.<ext>
    edit_lists: tuple[str, ...] = ()  # formats from editlist.FORMATS; "all" works too
    zone: tuple[float, float, float, float] | None = None  # x, y, w, h as fractions of the upright frame: only people there count (static camera only)
    zone_overlap: float | None = None  # None: the box centre must be in the zone; else this fraction of the box (0..1)
    activity: str = "any"  # "riding": a person counts only while their box moves fast enough (detect mode; needs the tracker)
    activity_threshold: float | None = None  # box heights per real second; None = activity.THRESHOLD
    slowmo_factor: float = 1.0  # how many times slower than real time the file plays (a baked slow-motion file): speeds are per real second
    slowmo: str = "off"  # "off", "auto" (capture rate over 30 fps: 240 -> 8, 120 -> 4) or a factor above 1; clips are retimed, never re-encoded
    slowmo_audio: bool = False  # with slowmo: keep the audio, stretched with atempo and re-encoded to AAC (default: dropped)
    sound: str = "off"  # "snap": move try edges to a pop before / a landing after (audio.py); "off" leaves the edges alone
    markers: str = ""  # "clap": a marker at every sound impact inside each clip (segments.json, sidecars, edit lists)
    sound_speed: str = "auto"  # "auto", "normal", or a factor: whether the audio is baked slow motion (windows are scaled by it)


SOUND_MODES = ("off", "snap")
MARKER_KINDS = ("", "clap")


def check_sound(sound: str, markers: str, speed: str) -> str | None:
    """One-line message for the first bad sound setting, else None."""
    if sound not in SOUND_MODES:
        return f"--sound must be one of {', '.join(SOUND_MODES)} (got {sound!r})"
    if markers not in MARKER_KINDS:
        return f"--markers must be clap (got {markers!r})"
    if speed not in ("auto", "normal"):
        try:
            ok = float(speed) >= 1.0 and math.isfinite(float(speed))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return f"--sound-speed must be auto, normal or a factor of 1 or more (got {speed!r})"
    return None


def check_timing(pad: float, min_len: float, merge_gap: float, fps: float | None) -> str | None:
    """One-line message for the first of --pad, --min-len, --merge-gap, --fps that is not a finite number of 0 or more, else None.
    A negative pad shrinks clips to nothing and an inverted clip is still reported as success."""
    for flag, v in (("--pad", pad), ("--min-len", min_len), ("--merge-gap", merge_gap), ("--fps", fps)):
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            return f"{flag} must be a number, 0 or more (got {v!r})"
    return None


def check_ranges(enter: float | None, exit: float | None, min_box: float) -> str | None:
    """One-line message for the first of --enter, --exit, --min-box outside 0..1 (or not a number), else None.
    Scores and box heights are fractions, so a value outside 0..1 can never be met or always is."""
    for flag, v in (("--enter", enter), ("--exit", exit), ("--min-box", min_box)):
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0):  # NaN fails the comparison
            return f"{flag} must be a number from 0 to 1 (got {v!r})"
    return None


Progress = Callable[[float, str], None]  # (fraction done 0..1, short message); called from the run's thread


class ScoringError(RuntimeError):
    """Scoring stopped before the end of the video. Carries what was scored so far."""

    def __init__(self, message: str, times=(), scores=(), boxes=(), track_ids=()):
        super().__init__(message)
        self.times, self.scores, self.boxes, self.track_ids = list(times), list(scores), list(boxes), list(track_ids)


@dataclass
class Scored:
    """Per-sample result of a mode: time, score in [0,1], optional box, plus what to record about the run."""
    times: list[float]
    scores: list[float]
    boxes: list
    info: dict
    warnings: list[str]
    wall_s: float
    track_ids: list | None = None  # per sample, None where the target has no track (or no tracker ran)


def effective_fps(settings: Settings, preset: Preset) -> float | None:
    if settings.fps is not None:
        return settings.fps or None
    return preset.motion_fps if settings.mode == "motion" else preset.detect_fps


def require_ffmpeg() -> None:
    """Fail before any decoding if the tools the run needs at the end are missing."""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"{' and '.join(missing)} not found on PATH; install ffmpeg (brew install ffmpeg)")


def clear_old_clips(out: Path, stem: str) -> list[str]:
    """Remove clip files an earlier run left in `out`, so the folder holds only this run's clips.
    Only names this tool writes (<stem>_NN_mmss-mmss.<ext>) are touched."""
    pat = re.compile(rf"^{re.escape(safe_stem(stem))}_\d{{2,}}_\d{{4,}}-\d{{4,}}\.\w+$")
    gone = []
    for f in out.iterdir():
        if f.is_file() and pat.match(f.name):
            f.unlink()
            gone.append(f.name)
    return gone


def write_failed_run(out: Path, stem: str, info: VideoInfo, settings: Settings, msg: str) -> None:
    """Record a run that stopped before any cut. segments.json is replaced, so the clips and the LosslessCut CSV
    of an earlier run in the folder would no longer match any metadata: they go too."""
    clear_old_clips(out, stem)
    clear_old_sidecars(out, stem)
    clear_old_exports(out, stem)
    (out / f"{safe_stem(stem)}_losslesscut.csv").unlink(missing_ok=True)
    (out / "segments.json").write_text(json.dumps(dict(
        inframe_version=__version__, mode=settings.mode, source=info.to_dict(), settings=asdict(settings),
        error=msg, warnings=[msg], presence_intervals=[], clips=[]), indent=1))


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def model_note(cls: int, model: str, preset_model: str, why: str) -> str:
    """The log line for a class target whose model differs from the preset's; when it does not (nano is all there is), no "instead of"."""
    swap = f"RF-DETR {model} instead of {preset_model}" if model != preset_model else f"RF-DETR {model}"
    return f"[detect] class {classes.class_name(cls)}: {swap}: {why}"


def score_detect(video: str, info: VideoInfo, preset: Preset, fps: float | None, target: Target,
                 progress: Progress | None = None, tracker_kind: str = "ocsort") -> Scored:
    """Run decode + detect + target over the whole video.
    If predict() fails mid-run the batch is retried on the torch fallback; if that is not available the
    ScoringError carries the samples scored so far."""
    def with_slices(d):  # tap mode: small people need more detector pixels
        return for_tap(d) if getattr(target, "wants_slices", False) else d

    cls = getattr(target, "cls", PERSON)
    ckw = {} if cls == PERSON else {"cls": cls}  # a person run calls detect(frames) as ever
    hf = target.height_frac() if hasattr(target, "height_frac") else None
    model, why = classes.model_for(preset.model, cls, hf, available_sizes())
    if why:
        log(model_note(cls, model, preset.model, why))
    det = with_slices(load_detector(model))
    log(f"[detect] {det.backend}, RF-DETR {model}, batch {getattr(det, 'batch', BATCH)}" + (f", class {classes.class_name(cls)}" if cls != PERSON else ""))
    times: list[float] = []
    scores: list[float] = []
    boxes: list[tuple | None] = []
    track_ids: list[int | None] = []
    stats: dict = {}
    warnings: list[str] = []
    fallback_warned = False  # a mid-run fallback already said so
    tracker = None
    if getattr(target, "wants_tracks", False):
        if tracks.usable(fps):
            tracker = tracks.Tracker(tracker_kind, fps)
        else:
            warnings.append(f"no tracker: it needs {tracks.MIN_TRACK_FPS:g}+ samples per second (this run: "
                            f"{'keyframes only' if not fps else f'{fps:g}'}); the target is matched per sample")
    t_detect = 0.0
    t0 = time.perf_counter()
    buf = []
    prev_sig = None
    n_cuts = 0

    def fail(why: str):
        raise ScoringError(f"{why}; {len(times)} samples scored so far", times, scores, boxes, track_ids if tracker else ())

    def detect_batch():
        nonlocal det, fallback_warned
        try:
            return det.detect([s.small if s.small is not None else s.rgb for s in buf], **ckw)
        except Exception as e:
            at = f"t={buf[0].t:.1f} s"
            if "FALLBACK" in det.backend:
                fail(f"the torch fallback detector failed at {at} ({type(e).__name__}: {e})")
            log(f"[detect] WARNING: predict() failed at {at} ({type(e).__name__}: {e}); FALLING BACK to torch MPS")
            try:
                det = with_slices(load_fallback_detector(model))
            except Exception as e2:
                fail(f"detector predict() failed at {at} ({type(e).__name__}: {e}) and the torch fallback is unavailable "
                     f"({type(e2).__name__}: {e2})")
            warnings.append(f"detector predict() failed at {at} ({type(e).__name__}: {e}); the rest of the run used torch MPS")
            fallback_warned = True
            try:
                return det.detect([s.small if s.small is not None else s.rgb for s in buf], **ckw)
            except Exception as e3:
                fail(f"the torch fallback detector failed at {at} ({type(e3).__name__}: {e3})")

    def flush():
        nonlocal t_detect, prev_sig, n_cuts
        if not buf:
            return
        t = time.perf_counter()
        dets = detect_batch()
        t_detect += time.perf_counter() - t
        for s, d in zip(buf, dets):
            if tracker:
                sig = tracks.frame_signature(s.rgb)
                if prev_sig is not None and tracks.frame_distance(prev_sig, sig) > tracks.CUT_THRESHOLD:
                    tracker.reset()  # a hard cut: nobody in the old shot is anybody in the new one
                    if hasattr(target, "on_cut"):
                        target.on_cut()
                    n_cuts += 1
                prev_sig = sig
                d = tracker.update(s.t, d, s.rgb)
            h = target.evaluate(s.t, s.rgb, d)
            times.append(s.t); scores.append(h.score); boxes.append(h.box); track_ids.append(h.track_id)
        buf.clear()

    last_log = 0.0
    try:
        # A target that embeds crops (photo mode) wants more source pixels than the detector needs. The frame is decoded at the target's long
        # side and, from the same decoded picture, once more at DECODE_LONG_SIDE for the detector: its scores (and the reference baseline
        # intervals, which ride on borderline ones) must not move with the decode size.
        long_side = getattr(target, "decode_long_side", DECODE_LONG_SIDE)
        for s in iter_samples(video, info, fps=fps, long_side=long_side, decoder=None, stats=stats, small_side=DECODE_LONG_SIDE):
            buf.append(s)
            if len(buf) == BATCH:
                flush()
                if progress:
                    progress(min(1.0, max(0.0, (s.t - info.offset) / max(info.duration, 1e-9))), f"detect: {s.t:.0f} of {info.end_time:.0f} s")
                if s.t - last_log >= 60:
                    last_log = s.t
                    log(f"[run] {s.t:7.1f}/{info.end_time:.0f} s  samples {len(times)}  elapsed {time.perf_counter() - t0:.0f} s")
        flush()
        if hasattr(target, "finalize"):  # a target that decides once it has seen the whole run (tap mode)
            hits = target.finalize()
            scores[:] = [h.score for h in hits]
            boxes[:] = [h.box for h in hits]
            track_ids[:] = [h.track_id for h in hits]
    except tracks.TrackerUnavailable as e:
        fail(f"tracking failed ({e})")
    finally:
        if tracker:
            tracker.close()
    wall = time.perf_counter() - t0
    if "FALLBACK" in det.backend and not fallback_warned:
        warnings.append("detector fell back to torch MPS (CoreML failed)")
    return Scored(times, scores, boxes, dict(decoder=stats.get("decoder"), detector=det.backend, model=model,
                                             model_package=Path(det.path).name, detect_model_s=round(t_detect, 2),
                                             tracker=tracker_kind if tracker else None, shot_cuts=n_cuts if tracker else None,
                                             **({"target_info": target.describe()} if hasattr(target, "describe") else {}),
                                             **({"class": classes.class_name(cls), "class_id": cls, "model_note": why} if cls != PERSON else {})),
                  warnings, wall, track_ids if tracker else None)


def score_motion(video: str, info: VideoInfo, fps: float | None, progress: Progress | None = None, zone: Zone | None = None) -> Scored:
    # decoding is most of the work; the scoring that follows it gets the last 15 %
    hook = (lambda t: progress(0.85 * min(1.0, max(0.0, (t - info.offset) / max(info.duration, 1e-9))), f"motion: decoding {t:.0f} of {info.end_time:.0f} s")) if progress else None
    r = run_motion(video, info, fps, progress=hook, zone=zone)
    if progress:
        progress(1.0, "motion: scored")
    return Scored(r.times, r.scores, [None] * len(r.times), dict(decoder=r.decoder, **r.details), r.warnings, r.wall_s)


def write_samples(out: Path, times, scores, boxes, track_ids=None) -> None:
    """samples.csv: t, score, box; plus a track column when a tracker ran."""
    with open(out / "samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "score", "x0", "y0", "x1", "y1"] + (["track"] if track_ids else []))
        for i, (t, score, b) in enumerate(zip(times, scores, boxes)):
            row = [f"{t:.3f}", f"{score:.5f}", *([f"{v:.4f}" for v in b] if b else ["", "", "", ""])]
            if track_ids:
                row.append("" if track_ids[i] is None else track_ids[i])
            w.writerow(row)


def score_summary(scores: list[float], enter: float) -> dict:
    a = np.asarray(scores, float)
    if a.size == 0:
        return dict(n_samples=0)
    q = np.percentile(a, [50, 90, 99])
    top = float(a.max()) or 1.0
    hist, _ = np.histogram(a, bins=10, range=(0, top * 1.0000001))
    r = lambda v: round(float(v), 6)
    return dict(n_samples=int(a.size), mean=r(a.mean()), p50=r(q[0]), p90=r(q[1]), p99=r(q[2]), max=r(a.max()),
                frac_at_or_above_enter=round(float((a >= enter).mean()), 4),
                hist_0_to_max_in_tenths=[int(x) for x in hist])


def _nested_taps(target) -> list:
    """The taps of `target`. The targets that wrap another one (zone, activity) hand `taps` on from the inner target."""
    return list(getattr(target, "taps", None) or [])


def run_cut(video: str, out_dir: str, settings: Settings, target: Target | None = None,
            progress: Progress | None = None) -> dict:
    """`progress(fraction, message)` is optional: probe 0-2 %, scoring 2-92 %, cutting 92-100 %."""
    def prog(f: float, msg: str) -> None:
        if progress:
            progress(min(1.0, max(0.0, f)), msg)

    scoring = (lambda f, msg: prog(0.02 + 0.90 * f, msg)) if progress else None
    t_start = time.perf_counter()
    if settings.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if settings.tracker not in tracks.KINDS:
        raise ValueError(f"tracker must be one of {tracks.KINDS} (got {settings.tracker!r})")
    preset = PRESETS[settings.preset]
    formats = parse_formats(list(settings.edit_lists))
    settings = replace(settings, slowmo=parse_slowmo(settings.slowmo))  # bad value: ValueError before any work
    if settings.slowmo_audio and settings.slowmo == "off":
        raise ValueError("--slowmo-audio needs --slowmo auto or a factor")
    require_ffmpeg()
    src = Path(video)
    out = Path(out_dir)

    bad = check_ranges(settings.enter, settings.exit, settings.min_box)
    if bad:
        raise ValueError(bad)
    bad = check_sound(settings.sound, settings.markers, str(settings.sound_speed)) or check_timing(settings.pad, settings.min_len, settings.merge_gap, settings.fps)
    if bad:
        raise ValueError(bad)
    if settings.exit is not None and settings.enter is not None and settings.exit > settings.enter:
        raise ValueError(f"--exit ({settings.exit:g}) must not exceed --enter ({settings.enter:g})")
    if settings.mode == "detect" and settings.enter is None and settings.exit is not None and settings.exit > DETECT_ENTER:
        raise ValueError(f"--exit ({settings.exit:g}) must not exceed the detect default enter threshold ({DETECT_ENTER:g})")
    zone = Zone.of(settings.zone) if settings.zone else None
    check_overlap(settings.zone_overlap)
    if zone is None and settings.zone_overlap is not None:
        raise ValueError("--zone-overlap needs --zone")
    if check_activity(settings.activity) != "any" and settings.mode != "detect":
        raise ValueError("--activity riding needs --mode detect (motion mode has no boxes to measure a speed on)")
    if not (math.isfinite(settings.slowmo_factor) and settings.slowmo_factor > 0):
        raise ValueError(f"the slow-motion factor must be above 0 (got {settings.slowmo_factor})")
    if settings.activity_threshold is not None and not (math.isfinite(settings.activity_threshold) and settings.activity_threshold > 0):
        raise ValueError(f"--activity-threshold must be above 0 (got {settings.activity_threshold})")
    info = probe(video)  # every setting is checked and the video probed before the output folder is created
    log(f"[probe] {info.codec} {info.width}x{info.height} rot {info.rotation} {info.fps:g} fps "
        f"{info.duration:.1f} s (starts at {info.offset:.2f} s in the file), {len(info.keyframes)} keyframes, audio={info.has_audio}")
    for tap in _nested_taps(target):  # fail now, not after the whole detect pass
        if not math.isfinite(tap.t) or tap.t > info.end_time + 1e-6:
            raise ValueError(f"the tap at {tap.t:g} s is past the end of the video ({info.end_time:.1f} s); pick a moment inside it")
    prog(0.02, "probed")
    factor = 1.0
    slowmo_info: dict | None = None
    if settings.slowmo != "off":
        cap = capture_fps(video)
        factor = resolve_slowmo(settings.slowmo, cap)
        slowmo_info = dict(requested=settings.slowmo, factor=factor, capture_fps=round(cap, 3),
                           audio=("none" if not info.has_audio else "atempo" if settings.slowmo_audio else "dropped") if factor > 1 else None,
                           rule="clip_s = (source_s - source_t0) * factor; each clip's source_t0 is its slowmo.source_t0")
        if factor > 1:
            log(f"[slowmo] factor {factor:g} (capture {cap:g} fps), audio {slowmo_info['audio']}")
    fps = effective_fps(settings, preset)
    pre_warnings: list[str] = []
    extra_info: dict = {}
    if zone is not None or settings.activity != "any":
        cam = check_static(video, info)
        extra_info["camera_check"] = cam.to_dict()
        if not cam.static and zone is not None:
            pre_warnings.append(f"the camera does not look static ({cam.reason}); the zone needs a static camera, so the zone is ignored")
            zone = None
        elif not cam.static:
            pre_warnings.append(f"the camera does not look static ({cam.reason}); the activity setting measures speed in the picture, "
                                "so a following or shaking camera makes the speeds wrong")
    extra_info["zone_applied"] = zone.as_list() if zone is not None else None
    out.mkdir(parents=True, exist_ok=True)
    if settings.mode == "motion":
        target_name = "any-motion"
        sc = score_motion(video, info, fps, scoring, zone)
    else:
        target = target or AnyPerson(settings.min_box)
        target_name = target.name
        act = None
        if zone is not None:
            target = ZoneTarget(target, zone, settings.zone_overlap)
        if settings.activity != "any":
            thr = settings.activity_threshold or ACTIVITY_THRESHOLD
            target = act = ActivityTarget(target, thr, settings.slowmo_factor)
        try:
            sc = score_detect(video, info, preset, fps, target, scoring, settings.tracker)
        except ScoringError as e:
            write_samples(out, e.times, e.scores, e.boxes, e.track_ids or None)  # keep the partial scores
            msg = f"{e}; saved in {out / 'samples.csv'}"
            write_failed_run(out, src.stem, info, settings, msg)
            raise ScoringError(msg, e.times, e.scores, e.boxes, e.track_ids) from e
    if settings.mode == "detect" and act is not None:
        extra_info["activity_info"] = act.stats
        if act.skipped:
            pre_warnings.append(f"the activity setting was not applied: {act.skipped}")
    sc.warnings = pre_warnings + sc.warnings
    sc.info.update(extra_info)
    for w in sc.warnings:
        log(f"[WARNING] {w}")
    times, scores, boxes = sc.times, sc.scores, sc.boxes
    log(f"[{settings.mode}] {len(times)} samples in {sc.wall_s:.1f} s")

    enter, exit_ = settings.enter, settings.exit
    thresholds = dict(source="flags" if enter is not None or exit_ is not None else "default")
    if settings.mode == "motion":
        a_enter, a_exit, st = auto_thresholds(scores)
        if enter is None and exit_ is None:
            quiet_only = st.get("quiet_class_only")
            thresholds = dict(source="auto: median + 6 * spread (enter), median + 3 * spread (exit)"
                                     + (", from the quiet samples only (the subject is in frame most of the time)" if quiet_only else ""),
                              **{k: (v if isinstance(v, bool) else round(v, 6)) for k, v in st.items()})
        enter = a_enter if enter is None else enter
        exit_ = min(a_exit, enter) if exit_ is None else exit_
    else:
        enter = DETECT_ENTER if enter is None else enter
        exit_ = min(DETECT_EXIT, enter) if exit_ is None else exit_
    if exit_ > enter:
        # only the automatic motion threshold gets here: the explicit cases were refused before the decode
        msg = (f"--exit ({exit_:g}) must not exceed the automatic enter threshold {enter:.5f} found for this video; "
               f"lower --exit or pass --enter; the scores are saved in {out / 'samples.csv'}")
        write_samples(out, times, scores, boxes)
        write_failed_run(out, src.stem, info, settings, msg)
        raise ValueError(msg)
    log(f"[thresholds] enter {enter:.5f} exit {exit_:.5f} ({thresholds['source']})")

    presence, padded = build_segments(times, scores, info.end_time, enter, exit_,
                                      settings.merge_gap, settings.min_len, settings.pad, info.offset)
    log(f"[segments] {len(presence)} presence intervals -> {len(padded)} clips")
    if not presence and scores:
        sc.warnings.append(f"no stretch found: the highest score {max(scores):.5f} never reached the enter threshold {enter:.5f} "
                           f"for {settings.min_len:g} s; check the enter threshold or the mode")
        log(f"[WARNING] {sc.warnings[-1]}")

    snd = None
    snap_info: list[dict] = [{} for _ in padded]
    if settings.sound != "off" or settings.markers:
        snd = audio.analyse(video, info.offset, str(settings.sound_speed)) if info.has_audio else audio.Sound(note="no audio track: sound cues skipped")
        if snd.note:
            sc.warnings.append(snd.note)
            log(f"[sound] {snd.note}")
        else:
            log(f"[sound] {len(snd.impacts)} impacts, audio speed x{snd.speed:g} ({snd.speed_source})")
            if settings.sound == "snap" and padded:
                moved = audio.snap_clips(padded, snd.impacts, info.keyframes, info.offset, info.end_time, snd.speed, presence, 1.0 / info.fps if info.fps > 0 else 1 / 30)
                n_moved = sum(1 for _, _, i in moved if i)
                padded, snap_info = [(a, b) for a, b, _ in moved], [i for _, _, i in moved]
                log(f"[sound] snapped {n_moved} of {len(moved)} clips")

    # samples.csv is written before the cuts, so a failing cut cannot lose the scoring work
    write_samples(out, times, scores, boxes, sc.track_ids)

    removed = clear_old_clips(out, src.stem)
    if removed:
        log(f"[out] removed {len(removed)} clip(s) from an earlier run in {out}")
    clear_old_sidecars(out, src.stem)
    clear_old_exports(out, src.stem)
    t_cut = time.perf_counter()
    if slowmo_info is not None and factor <= 1.0:
        sc.warnings.append(f"slow motion {settings.slowmo}: the video runs at {slowmo_info['capture_fps']:g} fps, not above 45, so the clips are not retimed")
        log(f"[WARNING] {sc.warnings[-1]}")
    clips = []
    failed = []
    suffix = src.suffix.lower() or ".mov"
    ts = np.asarray(times)
    sa = np.asarray(scores)
    try:
        prog(0.92, f"cutting {len(padded)} clips")
        for i, (s, e) in enumerate(padded, 1):
            name = clip_name(src.stem, i, s, e, suffix)
            m = (ts >= s) & (ts <= e)
            peak = float(sa[m].max()) if m.any() else 0.0
            clip = dict(index=i, start=round(s, 3), end=round(e, 3), file=name, peak_score=round(peak, 5),
                        expected_copy_start=round(snap_to_keyframe(s, info.keyframes), 3))
            if snd is not None and not snd.note:
                cues = audio.clip_cues(snd.impacts, s, e, peak)
                clip["impacts"], clip["confidence"] = cues["impacts"], cues["confidence"]
                if settings.markers:
                    clip["markers"] = [dict(kind=settings.markers, **m) for m in cues["markers"]]
                if snap_info[i - 1]:
                    clip["snapped_from"] = {k.removesuffix("_from"): round(v, 3) for k, v in snap_info[i - 1].items()}
            try:
                clip["expected_copy_start"] = round(cut_clip(video, out / name, s, e, factor, settings.slowmo_audio), 3)  # measured; was the keyframe guess
                got = written_duration(str(out / name))
                if got is not None and factor <= 1.0:
                    clip["file_duration_s"] = round(got, 3)  # what the sidecar's clip.duration_s says
                warn = duration_warning(str(out / name), got, clip["expected_copy_start"], e, factor)
                if warn:
                    sc.warnings.append(f"clip {i}: {warn}")
                    log(f"[WARNING] {sc.warnings[-1]}")
                if factor > 1.0:
                    clip["slowmo"] = dict(factor=factor, source_t0=clip["expected_copy_start"], clip_start_s=0.0,
                                          **retimed_facts(str(out / name)))
                    if clip["slowmo"]["av_offset_ms"] is not None and abs(clip["slowmo"]["av_offset_ms"]) > 21.0:
                        sc.warnings.append(f"clip {i}: audio starts {clip['slowmo']['av_offset_ms']:g} ms from the picture (limit 21 ms)")
            except Exception as exc:  # keep going; the metadata of the whole run is still written below
                clip["error"] = str(exc)
                failed.append(clip)
                log(f"[ERROR] clip {i} ({name}): {exc}")
            clips.append(clip)
            prog(0.92 + 0.08 * i / len(padded), f"cut clip {i} of {len(padded)}")
    except BaseException as exc:  # the progress callback raised (the client cancelled) or the process is stopping
        # the earlier run's clips are already gone and segments.json is not rewritten below: leave a record that matches the folder
        write_failed_run(out, src.stem, info, settings, f"stopped during the cuts ({type(exc).__name__}); no clips are kept, "
                                                        f"the scores are saved in {out / 'samples.csv'}")
        raise
    cut_s = time.perf_counter() - t_cut

    write_losslesscut_csv(out / f"{safe_stem(src.stem)}_losslesscut.csv", [(c["start"], c["end"], c["file"]) for c in clips if "error" not in c])

    doc = dict(
        inframe_version=__version__,
        mode=settings.mode,
        source=info.to_dict(),
        settings=dict(asdict(settings), enter=enter, exit=exit_, preset_name=preset.name,
                      fps=fps if fps else "keyframes", target=target_name, thresholds=thresholds, **sc.info),
        warnings=sc.warnings + ([f"{len(failed)} of {len(clips)} clips could not be cut"] if failed else []),
        scores=score_summary(scores, enter),
        **({} if snd is None or snd.note else dict(sound=dict(impacts=len(snd.impacts), speed=snd.speed, speed_source=snd.speed_source,
                                                              snap=settings.sound == "snap", markers=settings.markers or None))),
        timing_s=dict(total=round(time.perf_counter() - t_start, 2), score=round(sc.wall_s, 2), cut=round(cut_s, 2)),
        presence_intervals=[[round(s, 3), round(e, 3)] for s, e in presence],
        clips=clips,
        **({} if slowmo_info is None else dict(slowmo=slowmo_info)),
    )
    try:
        write_sidecars(out, doc, times=times, scores=scores, boxes=boxes, track_ids=sc.track_ids,
                       presence=presence, target=target_name, embedder=getattr(target, "embedder", None),
                       detector=dict(mode=settings.mode, backend=sc.info.get("detector"), model=sc.info.get("model"),
                                     tracker=sc.info.get("tracker")))
    except Exception as e:  # any failure (OSError, or a TypeError from a non-JSON value): the clips and segments.json are still good; say what is missing
        doc["warnings"].append(f"sidecars not written ({type(e).__name__}: {e})")
        log(f"[WARNING] {doc['warnings'][-1]}")
    if settings.reel or formats:
        t_export = time.perf_counter()
        prog(0.99, "reel and edit lists")
        # a cut that found nothing is a result, not a failed reel (only `inframe export --reel` on an empty run fails)
        ex = export_run(doc, out, out, settings.reel and bool(doc["clips"]), formats)
        if settings.reel:
            doc["reel"] = ex["reel"]
            if not doc["clips"]:
                ex["warnings"].append("reel not built: no clips")
        if formats:
            doc["edit_lists"] = ex["edit_lists"]
        doc["warnings"] += ex["warnings"]
        for w in ex["warnings"]:
            log(f"[WARNING] {w}")
        doc["timing_s"]["export"] = round(time.perf_counter() - t_export, 2)
        doc["timing_s"]["total"] = round(time.perf_counter() - t_start, 2)
    (out / "segments.json").write_text(json.dumps(doc, indent=1))
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(clips)} clips failed to cut (first: {failed[0]['error']}); "
                           f"scores and segments are saved in {out}")
    return doc
