"""Who is in the video: one detect + track pass that lists every person the tracker followed (T1d, behind the MCP tool
`list_people`). Per person it keeps when they are in frame, one thumbnail (the sharpest-looking sample: confident and large)
and a target.json for `inframe cut --target tap:<file>` / `cut_video(target=...)`.

Track ids are only meaningful inside one scan: `cut` finds the person again by the time and box in target.json, as tap mode does."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import tracks
from .detect import Detection
from .pipeline import PRESETS, Progress, ScoringError, Settings, effective_fps, score_detect
from .probe import probe
from .target import Hit

THUMB = (240, 360)  # longest sides of a thumbnail, pixels
PAD_PX = 4
GAP_S = 1.5  # samples of one person closer than this form one stretch
MIN_SEEN_S = 1.0
MAX_PEOPLE = 20
# files a scan writes: person_NNN_<scan id>.jpg / .json (the scan id is part of the name so a target string from an earlier scan
# of the same folder can never point at another person of this scan; it stops working instead)
_OWN_FILE = re.compile(r"person_\d{3}_[0-9a-f]{8}\.(jpg|json)")


@dataclass
class _Track:
    times: list[float] = field(default_factory=list)
    best: float = -1.0
    best_t: float = 0.0
    best_box: tuple = ()
    thumb: Image.Image | None = None


class PeopleScan:
    """A Target (target.py) that scores nothing: it records, per track id, when it was seen and its best crop."""

    name = "people-scan"
    wants_tracks = True

    def __init__(self) -> None:
        self.tracks: dict[int, _Track] = {}

    def evaluate(self, t: float, frame: np.ndarray, candidates: list[Detection]) -> Hit:
        h, w = frame.shape[:2]
        for d in candidates:
            if d.track_id is None:
                continue
            tr = self.tracks.setdefault(d.track_id, _Track())
            tr.times.append(t)
            q = d.score * (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
            if q > tr.best:
                x0, y0, x1, y1 = d.box
                crop = frame[max(int(y0 * h) - PAD_PX, 0):int(y1 * h) + PAD_PX, max(int(x0 * w) - PAD_PX, 0):int(x1 * w) + PAD_PX]
                if crop.size:
                    im = Image.fromarray(np.ascontiguousarray(crop))
                    im.thumbnail(THUMB)
                    tr.best, tr.best_t, tr.best_box, tr.thumb = q, t, tuple(d.box), im
        return Hit(0.0, None)


def stretches(times: list[float], step: float, gap: float = GAP_S) -> list[list[float]]:
    """Sample times -> [start, end] stretches; samples closer than `gap` join, each sample covers `step` seconds."""
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - out[-1][1] + step <= gap + 1e-9:
            out[-1][1] = t + step
        else:
            out.append([t, t + step])
    return [[round(a, 3), round(b, 3)] for a, b in out]


def scan_people(video: str, out: Path, progress: Progress | None = None, min_seen: float = MIN_SEEN_S,
                max_people: int = MAX_PEOPLE) -> dict:
    """Scan `video` and write people.json, person_NNN.jpg and person_NNN.json into `out`. Returns the people.json document.
    Raises ValueError (unusable arguments) or RuntimeError (the tracker or detector failed)."""
    preset = PRESETS["balanced"]
    fps = effective_fps(Settings(), preset)
    info = probe(video)
    out.mkdir(parents=True, exist_ok=True)  # an unwritable folder fails now, not after the whole detect + track pass
    scan = PeopleScan()
    try:
        sc = score_detect(video, info, preset, fps, scan, progress, "ocsort")
    except ScoringError as e:
        raise RuntimeError(str(e)) from None
    except tracks.TrackerUnavailable as e:
        raise RuntimeError(str(e)) from None
    step = 1.0 / fps
    kept = [(tid, tr) for tid, tr in scan.tracks.items() if tr.thumb is not None and len(tr.times) * step >= min_seen]
    kept.sort(key=lambda kv: -len(kv[1].times))
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file() and _OWN_FILE.fullmatch(f.name):
            f.unlink()  # an earlier scan of this folder; any other file there is not ours
    scan_id = uuid.uuid4().hex[:8]
    people = []
    for n, (tid, tr) in enumerate(kept[:max_people], 1):
        stem = f"person_{n:03d}_{scan_id}"
        tr.thumb.save(out / f"{stem}.jpg", "JPEG", quality=85)
        (out / f"{stem}.json").write_text(json.dumps(dict(video=str(Path(video).resolve()), time=tr.best_t, box=[round(v, 4) for v in tr.best_box], track_id=tid), indent=1))
        iv = stretches(tr.times, step)
        people.append(dict(person=n, track_id=tid, first_s=iv[0][0], last_s=iv[-1][1], seen_s=round(len(tr.times) * step, 2),
                           stretches=iv, thumbnail=str(out / f"{stem}.jpg"), thumbnail_time_s=round(tr.best_t, 3),
                           target=f"tap:{out / f'{stem}.json'}"))
    doc = dict(scan_id=scan_id, video=str(Path(video)), duration=info.duration, sample_fps=fps, n_tracks=len(scan.tracks), n_people=len(people),
               people=people, warnings=sc.warnings, tracker=sc.info.get("tracker"), shot_cuts=sc.info.get("shot_cuts"),
               timing_s=round(sc.wall_s, 2))
    (out / "people.json").write_text(json.dumps(doc, indent=1))
    return doc
