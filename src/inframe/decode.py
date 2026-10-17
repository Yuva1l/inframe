"""Frame sampling by timestamp, in our own code (never ffmpeg's fps filter).

Decoder is chosen by codec: software for H.264, VideoToolbox for
HEVC / 10-bit, with a software fallback if VideoToolbox is unavailable. PyAV does not rotate, so
the display rotation is applied here.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterator

import av
import numpy as np

from .probe import VideoInfo

EPS = 1e-4


@dataclass
class Sample:
    t: float  # seconds in file time (pts - container start), the timeline ffmpeg -ss uses
    rgb: np.ndarray  # (H, W, 3) uint8, upright (rotation applied)
    small: np.ndarray | None = None  # the same frame at `small_side` (what the detector sees when `rgb` was decoded larger), else None


def choose_decoder(info: VideoInfo) -> str:
    """'videotoolbox' for HEVC and 10-bit, else 'software'."""
    if info.codec in ("hevc", "h265") or "10" in info.pix_fmt:
        return "videotoolbox"
    return "software"


def scaled_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Even-sized (w, h) with the long side at most `long_side`; never upscales."""
    scale = min(1.0, long_side / max(w, h))
    return max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)


def upright(arr: np.ndarray, rotation: int, mirrored: bool = False) -> np.ndarray:
    """Apply the display matrix. `rotation` is ffprobe's displaymatrix value: the counterclockwise
    angle the player rotates the stored frame by (-90 = a quarter turn clockwise). `mirrored` is a
    flip on top of it (a front camera, say). ffprobe folds the flip into the rotation it reports: after
    rotating by an even multiple of 90 the picture is flipped top to bottom (a horizontal flip reads as
    rotation 180, a vertical one as 0), after an odd multiple it is flipped left to right. Checked against
    ffmpeg's own autorotate in tests/test_t1a.py."""
    k = (rotation // 90) % 4  # np.rot90 turns counterclockwise
    out = np.rot90(arr, k) if k else arr
    if mirrored:
        out = np.fliplr(out) if k % 2 else np.flipud(out)
    return np.ascontiguousarray(out) if (k or mirrored) else arr


def _open(path: str, decoder: str):
    if decoder == "videotoolbox":
        from av.codec.hwaccel import HWAccel

        try:
            return av.open(path, hwaccel=HWAccel(device_type="videotoolbox", allow_software_fallback=False)), "videotoolbox"
        except Exception as e:  # no VideoToolbox here: say so and fall back
            print(f"[decode] VideoToolbox unavailable ({e}); using software decode", file=sys.stderr)
    return av.open(path), "software"


def iter_samples(
    path: str,
    info: VideoInfo,
    fps: float | None = 5.0,
    long_side: int = 640,
    decoder: str | None = None,
    stats: dict | None = None,
    small_side: int | None = None,
) -> Iterator[Sample]:
    """Yield upright RGB frames sampled by timestamp.

    fps=None means keyframes only (about one frame per GOP). Otherwise the first decoded frame at or
    after each k/fps slot is yielded; slots a sparse stream skips over are dropped, not duplicated.
    """
    decoder = decoder or choose_decoder(info)
    container, used = _open(path, decoder)
    if stats is not None:
        stats["decoder"] = used
    st = container.streams.video[0]
    st.thread_type = "AUTO"
    if fps is None:
        st.codec_context.skip_frame = "NONKEY"
    tb = float(st.time_base)
    t0 = info.format_start_time
    w, h = scaled_size(info.width, info.height, long_side)
    sm = scaled_size(info.width, info.height, small_side) if small_side and small_side < long_side else None  # a second, smaller copy of every frame
    step = 1.0 / fps if fps else 0.0
    k = 0
    try:
        for fr in container.decode(st):
            if fr.pts is None:
                continue
            t = float(fr.pts) * tb - t0
            if t >= info.end_time + EPS:
                break
            if fps is not None:
                if t + EPS < k * step:
                    continue
                k = max(k + 1, int(math.floor(t / step + EPS)) + 1)
            arr = fr.reformat(width=w, height=h, format="rgb24").to_ndarray()
            small = upright(fr.reformat(width=sm[0], height=sm[1], format="rgb24").to_ndarray(), info.rotation, info.mirrored) if sm else None
            yield Sample(t=t, rgb=upright(arr, info.rotation, info.mirrored), small=small)
    finally:
        container.close()
