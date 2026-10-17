"""Identity between samples: roboflow/trackers (Apache-2.0), OC-SORT by default, BoT-SORT when the camera pans.

Runs only when the sample rate is at least MIN_TRACK_FPS: below that, objects move too far between samples for
motion-based association. The fast preset (keyframes only) does gallery
matching per keyframe and has no tracker.

`trackers` needs OpenCV, which must not be loaded next to PyAV, so the tracker lives in a child process
(track_worker.py) and this class talks to it. `uv sync --extra tracking` installs it.
"""
from __future__ import annotations

import pickle
import subprocess
import sys
from dataclasses import replace

import numpy as np

from .detect import Detection

MIN_TRACK_FPS = 5.0
KINDS = ("ocsort", "botsort")
LOST_BUFFER_S = 2.0  # a track survives this long without a match (the library counts 30 fps frames, hence * 30 below)
HIGH_CONF = 0.4  # detections below this never start or join a track (the segment logic still sees them)


CUT_THRESHOLD = 0.3  # frame_distance above this between two samples = a hard cut


class TrackerUnavailable(RuntimeError):
    pass


def frame_signature(rgb: np.ndarray, grid: int = 4, bins: int = 8) -> np.ndarray:
    """Colour histograms of a grid x grid layout of the frame (8 levels per channel), each normalised. (grid*grid, bins**3)."""
    h, w = rgb.shape[:2]
    s = rgb[:: max(1, h // 72), :: max(1, w // 128)]
    h, w = s.shape[:2]
    q = (s // (256 // bins)).astype(np.int32)
    idx = q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]
    out = []
    for i in range(grid):
        for j in range(grid):
            c = idx[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid].ravel()
            out.append(np.bincount(c, minlength=bins ** 3) / max(1, c.size))
    return np.asarray(out)


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 (same picture) to 1 (nothing in common): mean over the grid cells of half the L1 distance of their colour histograms.
    Camera motion and fast action stay below CUT_THRESHOLD; a cut to another shot, or another tile layout, rises above it."""
    return float(np.abs(a - b).sum(axis=1).mean() / 2)


def usable(fps: float | None) -> bool:
    return fps is not None and fps >= MIN_TRACK_FPS


def tracker_kwargs(kind: str, fps: float, **overrides) -> dict:
    if kind not in KINDS:
        raise ValueError(f"tracker must be one of {KINDS} (got {kind!r})")
    kw = dict(lost_track_buffer=int(LOST_BUFFER_S * 30), frame_rate=float(fps), minimum_consecutive_frames=2,
              high_conf_det_threshold=HIGH_CONF)
    if kind == "botsort":
        kw["cmc_method"] = "sparseOptFlow"
    return {**kw, **overrides}


class Tracker:
    """tracker.update(t, detections, frame) -> the same detections, each with a track_id (None until the track is confirmed).
    Detections keep their order. Close it (or use it as a context manager) to stop the child process."""

    def __init__(self, kind: str = "ocsort", fps: float = MIN_TRACK_FPS, allow_low_fps: bool = False, **overrides):
        """`allow_low_fps` is for the benchmark only: it measures below MIN_TRACK_FPS to show why the pipeline does not."""
        if not usable(fps) and not allow_low_fps:
            raise ValueError(f"the tracker needs at least {MIN_TRACK_FPS:g} samples per second (got {fps})")
        self.kind, self.fps = kind, fps
        self._base = 0  # added to the library's ids; grows at every reset so a new shot never reuses an id
        self._top = -1
        self.proc = subprocess.Popen([sys.executable, "-m", "inframe.track_worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self._ask(("init", kind, tracker_kwargs(kind, fps, **overrides)))
        except Exception:
            self.close()
            raise

    def _ask(self, msg):
        try:
            pickle.dump(msg, self.proc.stdin, protocol=4)
            self.proc.stdin.flush()
            tag, val = pickle.load(self.proc.stdout)
        except (BrokenPipeError, EOFError, OSError):
            err = self.proc.stderr.read().decode(errors="replace")[-400:] if self.proc.poll() is not None else ""
            raise TrackerUnavailable(f"the tracker process stopped ({err.strip() or 'no message'}); "
                                     "is the tracking extra installed? (uv sync --extra tracking)") from None
        if tag == "error":
            raise TrackerUnavailable(f"tracker {self.kind}: {val}")
        return val

    def update(self, t: float, dets: list[Detection], frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        xyxy = np.asarray([[d.box[0] * w, d.box[1] * h, d.box[2] * w, d.box[3] * h] for d in dets], np.float64).reshape(-1, 4)
        conf = np.asarray([d.score for d in dets], np.float64)
        ids = self._ask(("update", float(t), xyxy, conf, frame if self.kind == "botsort" else None))
        ids = [int(i) + self._base if i >= 0 else None for i in ids]
        self._top = max([self._top, *[i for i in ids if i is not None]])
        return [replace(d, track_id=i) for d, i in zip(dets, ids)]

    def reset(self) -> None:
        """Forget every track (a new shot starts). Ids after a reset continue above the earlier ones, so each is unique in a run."""
        self._ask(("reset",))
        self._base = self._top + 1

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                pickle.dump(("close",), self.proc.stdin, protocol=4)
                self.proc.stdin.flush()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()  # our own child only
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                f.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
