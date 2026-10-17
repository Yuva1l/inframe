"""Activity: is the person doing something (riding), or only standing and walking? `--activity riding` keeps a person only
while the box of their track moves fast enough.

Speed = how far the box centre moves, in box heights, per REAL second. Box heights make it independent of how far the
person is from the camera; real seconds divide out a slow-motion file (a file that plays `slowmo_factor` times slower than
real time shows a third of the movement per file second). The steps and the window are measured over REAL time spans, so a
step spans MIN_STEP_REAL_S * factor file seconds and the window WINDOW_S * factor: scaling file-time speeds by the factor
would multiply the detector jitter by it too (a still person would read as riding at factor 8). Per track, per step, then
the median of the steps within +-WINDOW_S / 2 real seconds of the sample (1 s in all), so one jittery box or
one lost sample moves nothing. A track too short for two steps (under three samples, or in slow motion a few samples spanning
less than two step spans) gets the net movement over the whole track; one of fewer than three samples has no speed and is kept.

The tracks are the module's own, not the tracker's: an IoU tracker (OC-SORT at 5 samples/s) cannot follow a rider, who moves
farther than the box is wide between two samples (measured on sliding person clips, where it lost every rider above
1.8 box heights per second). associate() matches boxes of consecutive samples by centre distance (Hungarian, gated at
MAX_SPEED box heights per second and a size ratio), which needs neither the tracker nor its GPL-licensed OpenCV extra.

Static camera: the speed is measured in the picture, so a camera that follows the rider hides the riding and a hand that
shakes adds to the walking. The pipeline warns when the camera check says the camera moves.

The threshold is calibrated in tools/activity.py on synthetic boxes and spliced clips, never on the reference clip of the regression gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Detection
from .target import Hit, Target

ACTIVITIES = ("any", "riding")
THRESHOLD = 1.65  # box heights per real second: the best worst-class threshold on the calibration clips of tools/activity.py (section 18)
WINDOW_S = 1.0  # the median of the steps within +- WINDOW_S / 2 (the spec: at least 1 s)
MAX_STEP_S = 0.6  # two samples further apart than this are not a step (the track was lost between them)
MIN_STEPS = 2
MIN_STEP_REAL_S = 0.2  # with a slow-motion factor above 1 a step spans at least this many REAL seconds (file seconds = this * factor)
MAX_SPEED = 4.5  # box heights per second a box may move between two samples and still be the same person
GATE_SLACK = 0.15  # box heights of detector jitter added to that gate
SIZE_RATIO = 1.7  # the box height may change by at most this factor between two samples


def step_speeds(ts, boxes, aspect: float = 16 / 9, slowmo_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(time of each step, speed) for one track. `ts` ascending file seconds, `boxes` normalised (x0, y0, x1, y1),
    `aspect` = frame width / frame height of the upright frame, `slowmo_factor` >= 1 = how many times slower than real time the file plays.
    Speed is in box heights per real second. With a factor above 1 a step runs from a sample to the first one at least
    MIN_STEP_REAL_S * factor file seconds later, so the detector jitter is divided by a real-time span as at factor 1."""
    ts = np.asarray(ts, float)
    b = np.asarray(boxes, float).reshape(-1, 4)
    if len(ts) < 2:
        return np.empty(0), np.empty(0)
    cx, cy = (b[:, 0] + b[:, 2]) / 2 * aspect, (b[:, 1] + b[:, 3]) / 2  # in frame heights
    h = np.maximum(b[:, 3] - b[:, 1], 1e-6)
    f = max(float(slowmo_factor), 1.0)
    span = MIN_STEP_REAL_S * f * (f > 1)
    j = np.maximum(np.searchsorted(ts, ts + span - 1e-9), np.arange(len(ts)) + 1)
    # a short track (or the tail of one) has no sample a full span later: its last sample serves when that is at least half a span away
    j = np.where((j >= len(ts)) & (ts[-1] - ts >= span / 2 - 1e-9) & (f > 1), len(ts) - 1, j)
    i = np.flatnonzero(j < len(ts))
    j = j[i]
    dt = ts[j] - ts[i]
    ok = (dt > 1e-6) & (dt <= MAX_STEP_S * f)
    dist = np.hypot(cx[j] - cx[i], cy[j] - cy[i]) / np.maximum((h[j] + h[i]) / 2, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = dist / dt * slowmo_factor
    tm = (ts[j] + ts[i]) / 2
    return tm[ok], v[ok]


def speed_at(tm: np.ndarray, v: np.ndarray, t: float, window: float = WINDOW_S) -> float | None:
    """Median step speed within +- window / 2 of t; None when the track has no two steps near t."""
    if len(v) < MIN_STEPS:
        return None
    near = np.abs(tm - t) <= window / 2 + 1e-9
    if near.sum() >= MIN_STEPS:
        return float(np.median(v[near]))
    idx = np.argsort(np.abs(tm - t))[:MIN_STEPS]  # the two nearest steps, if they are still close
    return float(np.median(v[idx])) if np.abs(tm[idx] - t).max() <= window * 1.5 else None


def short_track_speed(ts, boxes, aspect: float = 16 / 9, slowmo_factor: float = 1.0) -> float | None:
    """Speed of a track too short for two steps of a slow-motion span: the net movement from its first to its last box over the
    whole time between them (box heights per real second; the jitter of one box over the full span, not of each step). None
    for a track of fewer than MIN_STEPS + 1 samples."""
    ts = np.asarray(ts, float)
    if len(ts) < MIN_STEPS + 1 or ts[-1] - ts[0] <= 1e-6:
        return None
    b = np.asarray(boxes, float).reshape(-1, 4)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-6)
    d = np.hypot((b[-1, 0] + b[-1, 2] - b[0, 0] - b[0, 2]) / 2 * aspect, (b[-1, 1] + b[-1, 3] - b[0, 1] - b[0, 3]) / 2) / ((h[0] + h[-1]) / 2)
    return float(d / (ts[-1] - ts[0]) * slowmo_factor)


def smoothed_speeds(ts, boxes, aspect: float = 16 / 9, slowmo_factor: float = 1.0, window: float = WINDOW_S) -> list[float | None]:
    """The speed at every sample of one track (see speed_at)."""
    tm, v = step_speeds(ts, boxes, aspect, slowmo_factor)
    if len(v) < MIN_STEPS:  # a short track: one speed from its net movement
        s = short_track_speed(ts, boxes, aspect, slowmo_factor)
        return [s] * len(ts)
    return [speed_at(tm, v, t, window * max(slowmo_factor, 1.0)) for t in ts]  # the window is in real seconds


def check_activity(v: str | None) -> str:
    if v is None:
        return "any"
    if v not in ACTIVITIES:
        raise ValueError(f"--activity must be one of {ACTIVITIES} (got {v!r})")
    return v


def associate(prev: list[tuple[int, tuple, float]], cands: list[tuple], t: float, aspect: float) -> list[int | None]:
    """For each candidate box, the id of the previous track it continues, or None (a new track).
    `prev` = (id, last box, time of that box) of the tracks still alive. The cost is the centre distance in box heights; a pair
    farther apart than MAX_SPEED * dt + GATE_SLACK, or of very different heights, cannot match."""
    out: list[int | None] = [None] * len(cands)
    if not prev or not cands:
        return out
    cost = np.full((len(cands), len(prev)), 1e6)
    for i, b in enumerate(cands):
        hb = max(b[3] - b[1], 1e-6)
        for j, (_, pb, pt) in enumerate(prev):
            hp = max(pb[3] - pb[1], 1e-6)
            if not hb / hp <= SIZE_RATIO or not hp / hb <= SIZE_RATIO:
                continue
            d = np.hypot(((b[0] + b[2]) - (pb[0] + pb[2])) / 2 * aspect, ((b[1] + b[3]) - (pb[1] + pb[3])) / 2) / ((hb + hp) / 2)
            if d <= MAX_SPEED * max(t - pt, 1e-6) + GATE_SLACK:
                cost[i, j] = d
    for i, j in zip(*linear_sum_assignment(cost)):
        if cost[i, j] < 1e5:
            out[i] = prev[j][0]
    return out


@dataclass
class _Track:
    ts: list[float] = field(default_factory=list)
    boxes: list[tuple] = field(default_factory=list)


class ActivityTarget:
    """Wraps a Target: a person counts only while their track moves at `threshold` box heights per real second or more.

    Speeds need the whole track around a sample, so the verdict comes in finalize(). A stateless inner target (AnyPerson,
    optionally behind a zone) is called then, with only the candidates that move; any other target ran live and its hits
    (also those of its own finalize) are zeroed where their track is too slow."""

    def __init__(self, inner: Target, threshold: float = THRESHOLD, slowmo_factor: float = 1.0, window: float = WINDOW_S):
        if not (threshold > 0 and slowmo_factor > 0 and window > 0):
            raise ValueError("threshold, slowmo_factor and window must be above 0")
        self.inner, self.threshold, self.slowmo_factor, self.window = inner, threshold, slowmo_factor, window
        self.deferred = bool(getattr(inner, "stateless", False))
        self._tracks: dict[int, _Track] = {}
        self._rows: list[tuple[float, list[Detection], Hit | None, list[int | None]]] = []  # time, candidates, live hit, own track ids
        self._alive: dict[int, tuple[tuple, float]] = {}  # own track id -> (last box, time)
        self._next_id = 0
        self._aspect = 16 / 9
        self.skipped: str | None = None
        self.stats: dict = {}

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def wants_tracks(self) -> bool:  # its own tracks: the inner target decides whether the pipeline runs the tracker
        return self.inner.wants_tracks

    def __getattr__(self, item):
        if item in ("inner", "threshold", "slowmo_factor", "window", "deferred", "_tracks", "_rows", "_alive", "_next_id", "finalize") or item.startswith("__"):
            raise AttributeError(item)
        return getattr(self.inner, item)

    def on_cut(self) -> None:
        self._alive.clear()  # a new shot: nobody continues
        if hasattr(self.inner, "on_cut"):
            self.inner.on_cut()

    def evaluate(self, t: float, frame: np.ndarray, candidates: list[Detection]) -> Hit:
        h, w = frame.shape[:2]
        self._aspect = w / max(h, 1)
        f = max(self.slowmo_factor, 1.0)  # the track breaks after MAX_STEP_S REAL seconds without a sample
        self._alive = {i: v for i, v in self._alive.items() if t - v[1] <= MAX_STEP_S * f}
        followed = list(candidates)  # every candidate has a track: one the detector holds at a low score must still meet the gate
        prev = [(i, b, bt) for i, (b, bt) in self._alive.items()]
        matched = associate([(i, b, bt / f) for i, b, bt in prev], [d.box for d in followed], t / f, self._aspect)  # the gate is in real time
        ids: dict[int, int] = {}
        for k, (d, m) in enumerate(zip(followed, matched)):
            if m is None:
                m, self._next_id = self._next_id, self._next_id + 1
            ids[id(d)] = m
            tr = self._tracks.setdefault(m, _Track())
            tr.ts.append(t)
            tr.boxes.append(tuple(d.box))
            self._alive[m] = (tuple(d.box), t)
        live = None if self.deferred else self.inner.evaluate(t, frame, candidates)
        self._rows.append((t, candidates, live, [ids.get(id(d)) for d in candidates]))
        return live if live is not None else Hit(0.0, None)  # deferred: finalize() decides

    def speed(self, track_id: int | None, t: float) -> float | None:
        if track_id is None or track_id not in self._tracks:
            return None
        tm, v = self._steps(track_id)
        if len(v) < MIN_STEPS:  # a track too short for two steps of the span (slow motion): the net movement over the whole track
            tr = self._tracks[track_id]
            return short_track_speed(tr.ts, tr.boxes, self._aspect, self.slowmo_factor)
        return speed_at(tm, v, t, self.window * max(self.slowmo_factor, 1.0))  # the window is in real seconds

    def _steps(self, track_id: int) -> tuple[np.ndarray, np.ndarray]:
        cache = self.__dict__.setdefault("_step_cache", {})
        if track_id not in cache:
            tr = self._tracks[track_id]
            cache[track_id] = step_speeds(tr.ts, tr.boxes, self._aspect, self.slowmo_factor)
        return cache[track_id]

    def _moving(self, track_id: int | None, t: float) -> bool:
        s = self.speed(track_id, t)
        return s is None or s >= self.threshold  # no speed yet = no evidence against

    def samples(self) -> list[tuple[float, list[tuple[tuple, float, int | None, float | None]]]]:
        """Per sample: (time, [(box, score, own track id, smoothed speed)]). For the calibration tool and the tests."""
        return [(t, [(d.box, d.score, i, self.speed(i, t)) for d, i in zip(cands, ids)]) for t, cands, _, ids in self._rows]

    def _hit_track(self, h: Hit, cands: list[Detection], ids: list[int | None]) -> int | None:
        """The own track of the candidate a live hit is about (the best overlap with its box)."""
        best, best_iou = None, 0.5
        for d, i in zip(cands, ids):
            if i is None or h.box is None:
                continue
            ix = max(min(d.box[2], h.box[2]) - max(d.box[0], h.box[0]), 0) * max(min(d.box[3], h.box[3]) - max(d.box[1], h.box[1]), 0)
            u = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) + (h.box[2] - h.box[0]) * (h.box[3] - h.box[1]) - ix
            if u > 0 and ix / u >= best_iou:
                best, best_iou = i, ix / u
        return best

    def finalize(self) -> list[Hit]:
        self.__dict__.pop("_step_cache", None)
        n = len(self._rows)
        n_steps = sum(len(self._steps(i)[1]) or int(self.speed(i, self._tracks[i].ts[0]) is not None) for i in self._tracks)  # a short track counts by its net movement
        if n_steps == 0 and any(c for _, c, _, _ in self._rows):
            self.skipped = "no two samples of one person were close enough in time (the run needs 2+ samples per second), so the speed is unknown"
        out: list[Hit] = []
        gated = 0
        if self.deferred:
            for t, cands, _, ids in self._rows:
                keep = cands if self.skipped else [d for d, i in zip(cands, ids) if i is None or self._moving(i, t)]
                gated += len(cands) - len(keep)
                out.append(self.inner.evaluate(t, np.zeros((1, 1, 3), np.uint8), keep))
        else:
            hits = self.inner.finalize() if hasattr(self.inner, "finalize") else [r[2] for r in self._rows]
            for (t, cands, _, ids), h in zip(self._rows, hits):
                if not self.skipped and h.score > 0 and not self._moving(self._hit_track(h, cands, ids), t):
                    gated += 1
                    h = Hit(0.0, None)
                out.append(h)
        self.stats = dict(samples=n, threshold=self.threshold, slowmo_factor=self.slowmo_factor, window_s=self.window,
                          candidates_dropped=gated, tracks=len(self._tracks))
        return out
