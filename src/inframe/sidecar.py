"""One JSON sidecar per clip: `<clip stem>.inframe.json`, next to the clip (docs/sidecar.schema.json).

It tells a downstream reader (an editor script, an analysis tool) which source the clip came from, where it sits in
source time, and where the target was inside it: one entry per scored sample, in source time and clip time, with the
box in the upright display frame, the confidence and the tracker's id. Container tags are never read, so no location
data can get in. Everything here is pure data shaping; the pipeline calls build_sidecars() after the cuts."""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from . import __version__
from .names import safe_stem

SCHEMA_VERSION = "1.2"  # major.minor: a reader refuses another major; minors only add optional fields
SUFFIX = ".inframe.json"

SCORE_KINDS = dict(
    detect="detector_confidence",  # 0..1 detector score of the target's box; presence is score >= enter
    motion="motion_foreground_fraction")  # share of the frame that moved, usually well under 0.1; presence is score >= enter

TIME_BASE = ("source_time is the file's own timeline in seconds, the one `ffmpeg -ss` counts, and slow motion is NOT "
             "divided out (a baked slow-motion clip stays slow). clip_time = source_time - clip.copy_start_s, the "
             "time a player shows in the cut clip; the clip starts on the keyframe at copy_start_s. When the `slowmo` block is "
             "present the clip was retimed: the time a player shows is clip_time * slowmo.factor.")


def sidecar_name(clip_file: str) -> str:
    """`IMG_0001_01_0018-0036.mov` -> `IMG_0001_01_0018-0036.inframe.json`."""
    return Path(clip_file).stem + SUFFIX


def clear_old_sidecars(out: Path, stem: str) -> None:
    """Remove the sidecars an earlier run left in `out` (names this tool writes: <stem>_NN_mmss-mmss.inframe.json,
    and the ".part" file a killed run was writing)."""
    pat = re.compile(rf"^{re.escape(safe_stem(stem))}_\d{{2,}}_\d{{4,}}-\d{{4,}}{re.escape(SUFFIX)}(\.part)?$")
    for f in out.iterdir():
        if f.is_file() and pat.match(f.name):
            f.unlink()


@lru_cache(maxsize=8)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash(path: str) -> dict:
    """SHA-256 of the whole source file, computed once per (path, size, mtime) in a process."""
    st = Path(path).stat()
    return dict(algorithm="sha256", value=_sha256(str(path), st.st_size, st.st_mtime_ns), size_bytes=st.st_size)


def _box(b) -> list[float] | None:
    """Rounded to 4 places and clamped to the frame: the detector's boxes of a person at the picture's edge overshoot
    0..1 by a thousandth (seen on phone clips), which a reader mapping them to pixels must not have to handle."""
    if b is None:
        return None
    x0, y0, x1, y1 = (round(min(1.0, max(0.0, float(v))), 4) for v in b)
    return [x0, y0, x1, y1] if x0 < x1 and y0 < y1 else None  # a box that clamps to nothing (wholly outside the frame) is no box


def build_sidecar(*, info: dict, src_hash: dict, clip: dict, n_clips: int, times, scores, boxes, track_ids,
                  presence: list, settings: dict, target: str, embedder: str | None, detector: dict, score: dict | None = None) -> dict:
    """The sidecar of one clip as a dict. `info` is VideoInfo.to_dict(); `clip` is the clip's entry of segments.json
    (index, start, end, file, expected_copy_start); times/scores/boxes/track_ids are the run's per-sample lists."""
    copy_start = float(clip["expected_copy_start"])
    start, end = float(clip["start"]), float(clip["end"])
    samples = []
    for i, t in enumerate(times):
        if t < copy_start - 1e-9 or t > end + 1e-9:
            continue
        tid = track_ids[i] if track_ids else None
        samples.append(dict(
            source_time=round(float(t), 3), clip_time=round(float(t) - copy_start, 3), box=_box(boxes[i]),
            confidence=round(float(scores[i]), 5), track_id=None if tid is None else int(tid)))
    ids = sorted({s["track_id"] for s in samples if s["track_id"] is not None and s["box"] is not None})
    inside = [[round(max(s, copy_start), 3), round(min(e, end), 3)] for s, e in presence if e > copy_start and s < end]
    sound = None
    if "impacts" in clip:  # the run read the audio (--sound or --markers)
        sound = dict(impacts=int(clip["impacts"]), confidence=float(clip["confidence"]),
                     confidence_note="peak detector score, raised by 0.2 (capped at 1) when the clip holds at least one impact",
                     snapped_from=clip.get("snapped_from"),
                     markers=[dict(kind=m["kind"], source_time=m["time"], clip_time=round(m["time"] - copy_start, 3), strength=m["strength"])
                              for m in clip.get("markers", [])])
    return dict(
        schema_version=SCHEMA_VERSION,
        inframe_version=__version__,
        source=dict(path=str(info["path"]), hash=src_hash, duration_s=round(float(info["duration"]), 3)),
        clip=dict(file=clip["file"], in_source_s=round(start, 3), out_source_s=round(end, 3),
                  copy_start_s=round(copy_start, 3),
                  duration_s=round(float(clip["file_duration_s"]) if clip.get("file_duration_s") else end - copy_start, 3)),
        try_number=int(clip["index"]),
        tries_total=int(n_clips),
        video=dict(fps=round(float(info["fps"]), 4), rotation=int(info["rotation"]), mirrored=bool(info.get("mirrored", False)),
                   stored_width=int(info["width"]), stored_height=int(info["height"]),
                   display_width=int(info["display_width"]), display_height=int(info["display_height"])),
        time_base=TIME_BASE,
        coordinate_frame=dict(
            description="Upright display frame: the stored picture with the display matrix (rotation, then any mirror) "
                        "applied, as a player shows it.",
            box_format="xyxy", units="normalized", origin="top-left", x_axis="right", y_axis="down",
            scale_to_pixels="x * display_width, y * display_height"),
        target=dict(name=target),
        embedder=embedder,
        detector=detector,
        **({} if score is None else dict(score=score)),
        **({} if not clip.get("slowmo") else dict(slowmo=dict(
            factor=clip["slowmo"]["factor"], played_duration_s=clip["slowmo"]["duration_s"],
            rule="seconds shown by a player = clip_time * factor; clip_time and source_time stay in source seconds"))),
        **({} if sound is None else dict(sound=sound)),
        sampling=dict(fps=settings.get("fps"), n_samples=len(samples)),
        presence_in_clip=dict(source_s=inside, clip_s=[[round(a - copy_start, 3), round(b - copy_start, 3)] for a, b in inside]),
        track_ids=ids,
        samples=samples,
    )


def write_sidecars(out: Path, doc: dict, *, times, scores, boxes, track_ids, presence, target: str, embedder: str | None, detector: dict) -> list[str]:
    """Write one sidecar per clip of `doc` that was cut (and set clip["sidecar"]). Returns the file names."""
    good = [c for c in doc["clips"] if "error" not in c]
    if not good:
        return []
    info = doc["source"]
    h = source_hash(info["path"])
    mode = doc.get("mode") or detector.get("mode")
    st = doc["settings"]
    score = dict(kind=SCORE_KINDS.get(mode, "unknown"), enter=st.get("enter"), exit=st.get("exit"),
                 presence="a sample is in frame from the moment confidence >= enter until it stays below exit; "
                          "compare confidence with these thresholds, never with a fixed number such as 0.5")
    names = []
    failed: list[str] = []
    for c in good:
        sc = build_sidecar(info=info, src_hash=h, clip=c, n_clips=len(doc["clips"]), times=times, scores=scores, boxes=boxes,
                           track_ids=track_ids, presence=presence, settings=doc["settings"], target=target, embedder=embedder, detector=detector, score=score)
        name = sidecar_name(c["file"])
        tmp = out / (name + ".part")
        try:
            tmp.write_text(json.dumps(sc, indent=1))
            os.replace(tmp, out / name)  # a reader never sees half a sidecar
        except Exception as e:
            tmp.unlink(missing_ok=True)
            failed.append(f"{name}: {type(e).__name__}: {e}")
            continue
        c["sidecar"] = name
        names.append(name)
    if failed:
        raise OSError(f"{len(failed)} of {len(good)} sidecars not written ({len(names)} written); first: {failed[0]}")
    return names


def schema_path() -> Path:
    """docs/sidecar.schema.json in a source checkout (not shipped in the wheel)."""
    return Path(__file__).resolve().parents[2] / "docs" / "sidecar.schema.json"
