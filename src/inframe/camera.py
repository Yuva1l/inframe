"""Is the camera static? A cheap look before a run that needs it (a zone, a speed).

The keyframes of the video (about one per GOP, so the whole span is covered cheaply) are shrunk to a grey picture of at most
LONG_SIDE pixels; the per-pixel median of up to MAX_REF of them is the background, and every keyframe's global shift
against it is found by phase correlation (the same estimate motion mode uses). A camera that stands still gives shifts near 0
however much the person moves, because the person covers little of the picture and the median ignores them; a pan, a
handheld wobble or a drift shows up as shifts of pixels. A large textured subject can capture the phase correlation in a
few keyframes (shifts of tens of pixels on a tripod). A shift over IMPLAUSIBLE_FRAC of the width is therefore not a
camera shift but an outlier, unless MIN_RUN or more consecutive keyframes hold it (a step change: a tripod re-aimed or bumped,
a quick pan then tripod; a subject captures the correlation for a keyframe or two as it slides by). A keyframe that cannot be
matched at all (a view turned away) counts as far, unless most keyframes are unreadable (a flat picture). The camera counts as
moving when more than MAX_OUTLIER_FRACTION of the keyframes are outliers (a pan), when such a run exists, or when the p95 of the others is over motion.SHIFT_WARN_PX (the limit motion mode warns at), measured at
the same working width.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decode import iter_samples
from .motion import LONG_SIDE, MIN_RESPONSE, SHIFT_WARN_PX, blur5, phase_shift, to_gray
from .probe import VideoInfo

MAX_FRAMES = 240  # at most this many keyframes are compared
MAX_REF = 60
IMPLAUSIBLE_FRAC = 0.15  # a shift over this share of the working width is the subject capturing the correlation, not the camera ...
MAX_OUTLIER_FRACTION = 0.25  # ... unless more than this share of the keyframes shows it (a pan: every keyframe is far from the median)
MIN_RUN = 3  # ... or when this many consecutive keyframes are such outliers (a step change: the camera moved and stayed)
MIN_READABLE = 0.5  # unreadable keyframes count as far only when at least this share of the keyframes is readable (else the picture is flat)
MIN_FRAMES = 4  # fewer keyframes than this say nothing: the camera is taken as static


@dataclass
class CameraCheck:
    static: bool
    p95_px: float
    n_frames: int
    width_px: int
    reason: str = ""
    outliers: int = 0  # keyframes whose shift was implausibly large and left out of the p95

    def to_dict(self) -> dict:
        return dict(static=self.static, p95_shift_px=round(self.p95_px, 3), keyframes=self.n_frames, working_width=self.width_px, reason=self.reason,
                    **({"outlier_keyframes": self.outliers} if self.outliers else {}))


def shifts_from_frames(grays: list[np.ndarray]) -> np.ndarray:
    """Global shift (pixels, working size) of each grey frame against the per-pixel median of a strided subset."""
    stack = np.stack(grays)
    idx = np.linspace(0, len(grays) - 1, min(MAX_REF, len(grays))).astype(int)
    ref = np.median(stack[idx], axis=0).astype(np.float32)
    out = []
    for g in grays:
        (dx, dy), resp = phase_shift(ref, g.astype(np.float32))
        out.append(float(np.hypot(dx, dy)) if resp >= MIN_RESPONSE else np.nan)
    return np.asarray(out)


def check_grays(grays: list[np.ndarray], limit: float = SHIFT_WARN_PX) -> CameraCheck:
    if len(grays) < MIN_FRAMES:
        return CameraCheck(True, 0.0, len(grays), grays[0].shape[1] if grays else 0, "too few keyframes to tell; taken as static")
    sh = shifts_from_frames(grays)
    ok = sh[np.isfinite(sh)]
    w = grays[0].shape[1]
    huge = ok > IMPLAUSIBLE_FRAC * w
    small = ok[~huge]
    p95 = float(np.percentile(small, 95)) if small.size else 0.0
    # A keyframe that cannot be matched to the background at all (a view turned away: no shift has a peak) is as far as a big shift.
    # Only when most keyframes do match: a flat or textureless picture matches nothing and says nothing about the camera.
    lost = ~np.isfinite(sh) & (ok.size >= len(sh) * MIN_READABLE)
    far = lost | (np.isfinite(sh) & (sh > IMPLAUSIBLE_FRAC * w))
    run = best = 0
    for f in far:  # consecutive keyframes
        run = run + 1 if f else 0
        best = max(best, run)
    pan = (bool(ok.size) and float(np.mean(far)) > MAX_OUTLIER_FRACTION) or best >= MIN_RUN
    static = not pan and p95 <= limit
    n_far = int(far.sum())
    return CameraCheck(static, p95, len(grays), w, "" if static else
                       (f"{n_far} of {len(sh)} keyframes ({best} in a row) are over {IMPLAUSIBLE_FRAC:.0%} of the width away from the background or cannot be matched to it, limit {limit:g} px" if pan else
                        f"p95 global shift {p95:.2f} px at {w} px wide, limit {limit:g}"), n_far)


def check_static(video: str, info: VideoInfo, limit: float = SHIFT_WARN_PX) -> CameraCheck:
    """Decode the keyframes (every nth when there are more than MAX_FRAMES) and compare them. See the module doc."""
    n_kf = len(info.keyframes) if info.keyframes else 0
    stride = max(1, -(-n_kf // MAX_FRAMES)) if n_kf else 1
    grays: list[np.ndarray] = []
    for i, s in enumerate(iter_samples(video, info, fps=None, long_side=LONG_SIDE)):
        if i % stride == 0:
            grays.append(blur5(to_gray(s.rgb)))
    return check_grays(grays, limit)
