"""Motion mode: no model. For a static camera.

Samples are downscaled grayscale. The background starts as the per-pixel median of the opening seconds
and then follows slow light changes by a selective update: only pixels that currently look like
background are blended in, so a subject who stands still stays foreground (it is not absorbed the way a
running median would absorb it); a pixel that has been foreground for BG_ABSORB_S seconds becomes background
(a parked object, or a ghost where something stood when the video began). Score = fraction of pixels that
differ from the background after a morphological opening. Thresholds default to median + k * MAD of the
video's own scores; when the subject is in frame most of the time they come from the quiet samples instead.
The camera must be static: global shift against the background is estimated and a warning is
raised when it is not.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

from .decode import iter_samples
from .probe import VideoInfo

LONG_SIDE = 320  # working resolution (long side), grayscale
MIN_DIFF = 12  # grey levels; floor for the per-pixel difference threshold
DIFF_SIGMAS = 6.0  # difference threshold = max(MIN_DIFF, DIFF_SIGMAS * noise sigma)
BG_INIT_S = 10.0  # the first background = per-pixel median of the samples in the opening this many seconds
BG_MAX_FRAMES = 60  # at most this many frames go into one median
BG_RATE_S = 20.0  # time constant of the slow update of background pixels
BG_ABSORB_S = 300.0  # a pixel foreground for this long becomes background
# foreground is grown by a 5x5 ellipse before it is kept out of the background update; speckle is removed by a 3x3 opening
K_ENTER, K_EXIT = 6.0, 3.0  # thresholds = median + k * spread of the scores
SPREAD_FLOOR = 2e-4  # minimum spread (score is a fraction of the frame area)
QUIET_GAP = 10.0  # busy samples this many quiet spreads above the quiet ones count as a separate class
MIN_RESPONSE = 0.05  # phase-correlation peak height below which the shift is not trusted
SHIFT_WARN_PX = 1.5  # p95 global shift at working resolution that triggers the warning


# Image operations in numpy / scipy (BSD). They reproduce the OpenCV calls motion mode used before, which is
# why OpenCV is gone: its wheel and PyAV's both bundle libavdevice, and macOS warns about the duplicate classes.
_BINOMIAL5 = np.array([1, 4, 6, 4, 1])
_CROSS3 = ndi.generate_binary_structure(2, 1)  # OpenCV's 3x3 ellipse
_ELLIPSE5 = np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], bool)  # OpenCV's 5x5 ellipse


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> uint8 grey, with OpenCV's fixed-point weights (0.299, 0.587, 0.114 in 15 bits)."""
    c = rgb.astype(np.uint32)
    return ((c[..., 0] * 9798 + c[..., 1] * 19235 + c[..., 2] * 3735 + 16384) >> 15).astype(np.uint8)


def blur5(g: np.ndarray) -> np.ndarray:
    """5x5 Gaussian (the binomial 1 4 6 4 1 kernel OpenCV uses for ksize 5), borders reflected, rounded."""
    x = ndi.correlate1d(g.astype(np.uint32), _BINOMIAL5, axis=1, mode="mirror")
    x = ndi.correlate1d(x, _BINOMIAL5, axis=0, mode="mirror")
    return ((x + 128) >> 8).astype(np.uint8)


def open3(fg: np.ndarray) -> np.ndarray:
    """Morphological opening of a 0/1 mask with a 3x3 ellipse; the frame edge does not erode (as in OpenCV)."""
    er = ndi.binary_erosion(fg.astype(bool), _CROSS3, border_value=1)
    return ndi.binary_dilation(er, _CROSS3, border_value=0).astype(np.uint8)


def grow5(fg: np.ndarray) -> np.ndarray:
    """Dilation of a 0/1 mask with a 5x5 ellipse."""
    return ndi.binary_dilation(fg.astype(bool), _ELLIPSE5, border_value=0).astype(np.uint8)


@lru_cache(maxsize=4)
def _window(h: int, w: int) -> np.ndarray:
    """OpenCV 5's phase-correlation window: sin(pi * i / (n - 1)) along each axis."""
    return np.outer(np.sin(np.pi * np.arange(h) / (h - 1)), np.sin(np.pi * np.arange(w) / (w - 1))).astype(np.float32)


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[tuple[float, float], float]:
    """Global translation between two float images by phase correlation: ((dx, dy), response).
    Sub-pixel by the weighted centroid of the 5x5 neighbourhood of the correlation peak. A peak with
    little energy (response < MIN_RESPONSE) is not a trustworthy shift."""
    h, w = a.shape
    win = _window(h, w)
    fa, fb = np.fft.rfft2(a * win), np.fft.rfft2(b * win)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    cross = np.where(mag > 0, cross / np.where(mag > 0, mag, 1.0), 0)
    corr = np.fft.fftshift(np.fft.irfft2(cross, s=(h, w)))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    y0, y1, x0, x1 = max(py - 2, 0), min(py + 3, h), max(px - 2, 0), min(px + 3, w)
    box = corr[y0:y1, x0:x1]
    total = float(box.sum())
    if total > 1e-12:
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cy, cx = float((box * yy).sum()) / total, float((box * xx).sum()) / total
    else:
        cy, cx = float(py), float(px)
    return (w / 2 - cx, h / 2 - cy), total


@dataclass
class MotionResult:
    times: list[float]
    scores: list[float]
    details: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    decoder: str | None = None
    wall_s: float = 0.0


def _otsu(a: np.ndarray) -> float:
    """Score that best splits `a` into a low and a high class (maximum between-class variance)."""
    hist, edges = np.histogram(a, bins=256)
    if edges[-1] <= edges[0]:
        return float(edges[0])
    p = hist / hist.sum()
    centres = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(p)
    mu = np.cumsum(p * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where((w > 0) & (w < 1), (mu[-1] * w - mu) ** 2 / (w * (1 - w)), 0.0)
    return float(edges[int(np.argmax(between)) + 1])


def _median_spread(a: np.ndarray, floor: float) -> tuple[float, float, float]:
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return med, mad, max(1.4826 * mad, floor)


def auto_thresholds(scores, k_enter: float = K_ENTER, k_exit: float = K_EXIT, floor: float = SPREAD_FLOOR) -> tuple[float, float, dict]:
    """(enter, exit, stats) from the scores' own distribution: median + k * 1.4826 * MAD.
    The spread never drops below `floor`, so an almost-empty video does not get a threshold at the noise.
    When the subject is in frame most of the time the median is a busy score, so if the middle sample
    falls in the upper of two well-separated classes the quiet (lower) class sets the baseline instead."""
    a = np.asarray(scores, float)
    if a.size == 0:
        return 1.0, 1.0, dict(median=0.0, mad=0.0, spread=floor)
    med, mad, spread = _median_spread(a, floor)
    split = _otsu(a)
    if med > split:
        lo, hi = a[a <= split], a[a > split]
        if lo.size >= 10:
            lo_med, lo_mad, lo_spread = _median_spread(lo, floor)
            if float(np.median(hi)) - lo_med > QUIET_GAP * lo_spread:
                return (lo_med + k_enter * lo_spread, lo_med + k_exit * lo_spread,
                        dict(median=lo_med, mad=lo_mad, spread=lo_spread, quiet_class_only=True))
    return med + k_enter * spread, med + k_exit * spread, dict(median=med, mad=mad, spread=spread)


def background_for(stack: np.ndarray, times: np.ndarray, centre: float, window: float = BG_INIT_S / 2, max_frames: int = BG_MAX_FRAMES) -> np.ndarray:
    """Per-pixel median (uint8) of the samples within +- window seconds of `centre`, strided to max_frames."""
    idx = np.nonzero(np.abs(times - centre) <= window)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(np.abs(times - centre)))])
    if idx.size > max_frames:
        idx = idx[np.linspace(0, idx.size - 1, max_frames).astype(int)]
    return np.median(stack[idx], axis=0).astype(np.uint8)


def _noise_sigma(d: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def run_motion(video: str, info: VideoInfo, fps: float | None, progress=None, zone=None) -> MotionResult:
    """With a `zone` (zone.Zone, upright fractions) the score is the fraction of the ZONE's pixels that are foreground; the background,
    the noise and the camera-shift estimates still use the whole picture. Samples are scored as they arrive. Only the opening BG_INIT_S seconds are held back (the first background is
    their median), so memory stays flat however long the video is."""
    t0 = time.perf_counter()
    stats: dict = {}
    times: list[float] = []
    scores: list[float] = []
    shifts: list[float] = []
    sigmas: list[float] = []
    head: list[tuple[float, np.ndarray]] = []  # opening samples, until the first background exists
    state: dict = {}  # bg_f, fg_for, shape

    def step(ti: float, g: np.ndarray) -> None:
        f = g.astype(np.float32)
        bg_f, fg_for = state["bg_f"], state["fg_for"]
        d = f - bg_f
        d -= np.median(d)  # global brightness drift (auto exposure) is not motion
        sigma = _noise_sigma(d)
        thr = max(MIN_DIFF, DIFF_SIGMAS * sigma)
        fg = open3((np.abs(d) > thr).astype(np.uint8))
        scores.append(float(fg[state["mask"]].mean()) if "mask" in state else float(fg.mean()))
        sigmas.append(sigma)
        (dx, dy), resp = phase_shift(bg_f, f)
        shifts.append(float(np.hypot(dx, dy)) if resp >= MIN_RESPONSE else np.nan)  # a flat scene gives no usable shift
        # selective update: only pixels that look like background follow the light, a still subject stays foreground
        dt = float(min(ti - times[-1], BG_RATE_S)) if times else 0.0
        times.append(ti)
        near_fg = grow5(fg).astype(bool)
        fg_for = np.where(near_fg, fg_for + dt, 0.0).astype(np.float32)
        update = ~near_fg | (fg_for >= BG_ABSORB_S)
        alpha = 1.0 - np.exp(-dt / BG_RATE_S)
        state["bg_f"] = np.where(update, bg_f + alpha * (f - bg_f), bg_f).astype(np.float32)
        state["fg_for"] = fg_for

    def start_background() -> None:
        ts = np.asarray([t for t, _ in head])
        stack = np.stack([g for _, g in head])
        state["shape"] = stack.shape[1:]
        if zone is not None:
            state["mask"] = zone.mask(*stack.shape[1:])
        state["bg_f"] = background_for(stack, ts, ts[0] + BG_INIT_S / 2, window=BG_INIT_S / 2).astype(np.float32)
        state["fg_for"] = np.zeros(state["bg_f"].shape, np.float32)  # seconds each pixel has been foreground in a row
        for ti, g in head:
            step(ti, g)
        head.clear()

    first: float | None = None
    n = 0
    for s in iter_samples(video, info, fps=fps, long_side=LONG_SIDE, stats=stats):
        g = blur5(to_gray(s.rgb))
        n += 1
        if first is None:
            first = s.t
        if "bg_f" not in state:
            if s.t - first <= BG_INIT_S:
                head.append((s.t, g))
                continue
            start_background()
        step(s.t, g)
        if progress and n % 25 == 0:
            progress(s.t)
    res = MotionResult(times=times, scores=scores, decoder=stats.get("decoder"))
    if first is None:
        res.warnings.append("no frames decoded")
        res.wall_s = time.perf_counter() - t0
        return res
    if "bg_f" not in state:  # the whole video is shorter than the opening
        start_background()
    res.times, res.scores = times, scores
    sh = np.asarray(shifts)
    ok = sh[np.isfinite(sh)]
    if ok.size == 0:
        ok = np.zeros(1)
    p95 = float(np.percentile(ok, 95))
    h, w = state["shape"]
    res.details = dict(
        working_size=[int(w), int(h)],
        **({"zone": zone.as_list(), "zone_pixels": int(state["mask"].sum())} if zone is not None else {}),
        background=(f"median of the first {BG_INIT_S:g} s, then a slow update (time constant {BG_RATE_S:g} s) of background pixels only; "
                    f"a pixel foreground for {BG_ABSORB_S:g} s becomes background"),
        diff_threshold_grey=round(max(MIN_DIFF, DIFF_SIGMAS * float(np.median(sigmas))), 2),
        noise_sigma_median=round(float(np.median(sigmas)), 3),
        camera_shift_px=dict(p50=round(float(np.median(ok)), 3), p95=round(p95, 3), max=round(float(ok.max()), 3),
                             frac_over_1px=round(float((ok > 1.0).mean()), 4), frac_reliable=round(float(np.isfinite(sh).mean()), 4)),
    )
    if p95 > SHIFT_WARN_PX:
        res.warnings.append(
            f"camera does not look static (p95 global shift {p95:.2f} px at {w} px wide, limit {SHIFT_WARN_PX}); "
            "motion mode assumes a fixed camera, use --mode detect")
    res.wall_s = time.perf_counter() - t0
    return res
