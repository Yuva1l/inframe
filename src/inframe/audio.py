"""Sound cues (roadmap T7): impacts (a board pop, a landing) found in the audio, used to place a try's edges.

The onset method follows a standard spectral-flux detector. Impacts are short broadband bursts, found with
spectral flux (Bello et al. 2005, IEEE TSAP 13(5)): log-magnitude STFT, half-wave-rectified frame-to-frame increase summed over
frequency, peaks above a moving median + k * MAD. Imports: numpy and scipy (BSD-3, already dependencies); audio is decoded by
ffmpeg in a child process. Nothing here uses a model.

Three uses, all opt-in (`--sound snap`, `--markers clap`; the pipeline calls `analyse` and `snap_clips`):
- snap a try's start to a pop shortly before it and its end to a landing shortly after it, within +-SNAP_RADIUS_S;
- markers at the impacts inside each clip;
- a clip with at least one impact gets its `confidence` raised (a stretch with a pop and a landing is a try, not a walk-through).

Slow motion: a clip whose audio is baked slow (stretched by a factor f) has impacts that are f times longer, so the search radius, lead
and tail here are multiplied by f. The caps stay absolute (an edge never moves more than MAX_MOVE_S, a clip is never cut below
MIN_CLIP_S), so a misread factor cannot move an edge further than promised. Whether the audio is baked is decided from how fast the impacts rise (`estimate_speed`); `--sound-speed` overrides it.
That estimate is calibrated on synthetic clicks only [unverified on real slow-motion files]."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, stft

from .cut import snap_to_keyframe

SR = 22050
SNAP_RADIUS_S = 1.5  # an edge looks for a pop or landing this far on either side of it
LEAD_S = 0.5  # a clip starts this long before the pop
TAIL_S = 0.5  # and ends this long after the landing
MAX_MOVE_S = 1.0  # an edge never moves further than this
KEYFRAME_EPS_S = 0.002  # a snapped start is requested this far past kf + 2 frames: ffmpeg's mov demuxer goes back one GOP for [kf, kf + 2 frames) in iPhone HEVC
SNAP_MARGIN_S = 0.1  # an edge counts as moved past a pop or landing only when it ends at least this far clear of it
MIN_STRENGTH = 3.0  # an onset must exceed its threshold (median + k * MAD) by this factor to count as an impact
CONFIDENCE_BOOST = 0.2  # added (capped at 1) to the peak score of a clip with an impact
MIN_CLIP_S = 1.0  # a snap that would leave less than this is undone
RISE_REF_S = 0.007  # rise time (10 % to 90 % of the peak) of a normal-speed impact; the baked factor is the median rise over this
BAKED_FROM = 3.0  # estimated factors below this count as normal speed
MAX_AUTO_SPEED = 8.0  # the auto estimate never claims more than this (240 fps over 30 is the slowest capture Photos makes)
CHUNK_S = 20.0  # onset_envelope transforms this much audio at a time, so memory does not grow with the length of the video
MIN_ONSETS_FOR_SPEED = 3


def stream_starts(path: str) -> dict[str, float]:
    """Start time of the first audio and first video stream, and of the container, in seconds ({} when ffprobe fails).
    Only QuickTime/MP4-family files are trusted (their edit list states the offset); others report nothing."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time:format=format_name,start_time",
                              "-of", "json", path], capture_output=True, text=True, timeout=60)
        meta = json.loads(out.stdout or "{}")
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}
    if "mp4" not in meta.get("format", {}).get("format_name", "").split(","):
        return {}
    res: dict[str, float] = {}
    for s in meta.get("streams", []):
        try:
            res.setdefault(s.get("codec_type", ""), float(s["start_time"]))
        except (KeyError, TypeError, ValueError):
            continue
    try:
        res["format"] = float(meta["format"]["start_time"])
    except (KeyError, TypeError, ValueError):
        pass
    return res


def audio_start_in_file_time(path: str, video_offset: float) -> float:
    """File time (the timeline `ffmpeg -ss` counts, which every pipeline time uses) of decoded audio sample 0.
    `video_offset` is where the video starts in file time. For files whose edit lists we do not trust, audio is
    taken to start with the picture."""
    st = stream_starts(path)
    if "audio" in st and "video" in st:
        return video_offset + st["audio"] - st["video"]
    return video_offset


def read_audio(path: str, sr: int = SR) -> np.ndarray | None:
    """Mono float32 audio at `sr`, or None when the file has no audio stream or ffmpeg fails."""
    try:
        out = subprocess.run(["ffmpeg", "-nostdin", "-nostats", "-loglevel", "error", "-i", path, "-vn", "-map", "0:a:0?",
                              "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, timeout=1800)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    return np.frombuffer(out.stdout, dtype=np.float32)


def onset_envelope(y: np.ndarray, sr: int = SR, n_fft: int = 1024, hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Spectral flux per STFT frame and the frame centre times. The STFT is taken in chunks of CHUNK_S seconds (frames are
    independent, flux only needs the previous frame), so the result equals a whole-file STFT at a fraction of the memory."""
    n = len(y)
    if n < n_fft:
        return np.zeros(0), np.zeros(0)
    n_frames = -(-(n - n_fft) // hop) + 1  # scipy's padded STFT: the last frame is zero-filled to fit
    chunk = max(2, int(CHUNK_S * sr) // hop)
    flux = np.zeros(n_frames)
    prev: np.ndarray | None = None
    for f0 in range(0, n_frames, chunk):
        nf = min(chunk, n_frames - f0)
        a, b = f0 * hop, f0 * hop + (nf - 1) * hop + n_fft
        seg = y[a:b]
        if len(seg) < b - a:
            seg = np.concatenate([seg, np.zeros(b - a - len(seg), dtype=seg.dtype)])
        _, _, Z = stft(seg, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
        mag = np.log1p(100 * np.abs(Z))
        del Z
        if prev is not None:
            mag = np.concatenate([prev[:, None], mag], axis=1)
            flux[f0: f0 + nf] = np.maximum(np.diff(mag, axis=1), 0).sum(axis=0)
        elif nf > 1:
            flux[f0 + 1: f0 + nf] = np.maximum(np.diff(mag, axis=1), 0).sum(axis=0)
        prev = mag[:, -1].copy()
        del mag
    return flux, (np.arange(n_frames) * hop + n_fft / 2) / sr


def onsets(flux: np.ndarray, times: np.ndarray, k: float = 4.0, min_gap_s: float = 0.05) -> list[tuple[float, float]]:
    """[(time, strength)]: strength is the flux over its threshold (moving median + k * MAD over 0.5 s); 1.0 is the threshold."""
    if len(flux) < 3:
        return []
    dt = times[1] - times[0]
    win = max(3, int(0.5 / dt) | 1)
    pad = win // 2
    fp = np.pad(flux, pad, mode="edge")
    w = sliding_window_view(fp, win)
    med = np.median(w, axis=1)
    mad = np.median(np.abs(w - med[:, None]), axis=1) + 1e-9
    thr = med + k * mad
    peaks, _ = find_peaks(flux, height=thr, distance=max(1, int(min_gap_s / dt)))
    return [(float(times[p]), float(flux[p] / (thr[p] + 1e-9))) for p in peaks]


def _rise_time(y: np.ndarray, sr: int, t: float, scale: float = 1.0) -> float | None:
    """Seconds the amplitude envelope takes to go from 10 % to 90 % of its peak near an onset at `t`, or None when the burst
    does not stand out of the noise. `scale` widens the search for audio that may be slowed."""
    a = max(0, int((t - 0.06 * scale) * sr))
    b = min(len(y), int((t + 0.10 * scale) * sr))
    if b - a < sr // 100:
        return None
    seg = np.abs(y[max(0, a - sr // 2): b])
    env = uniform_filter1d(maximum_filter1d(seg, max(1, sr // 1000)), max(1, sr // 500))
    off = a - max(0, a - sr // 2)
    win = env[off:]
    p = int(np.argmax(win))
    peak = float(win[p])
    floor = float(np.median(env)) + 1e-9
    if peak < 5 * floor:
        return None
    before = win[: p + 1]
    t10 = np.nonzero(before < 0.1 * peak)[0]
    t90 = np.nonzero(before < 0.9 * peak)[0]
    if not len(t10) or not len(t90):
        return None
    return float(t90[-1] - t10[-1]) / sr


def estimate_speed(y: np.ndarray, found: list[tuple[float, float]], sr: int = SR) -> float:
    """Factor by which the audio is slowed (1.0 = normal speed), from the median rise time of the strongest impacts over
    RISE_REF_S. Under BAKED_FROM, or with fewer than MIN_ONSETS_FOR_SPEED measurable impacts, it says 1.0."""
    strongest = sorted(found, key=lambda x: -x[1])[:12]
    rises = [r for r in (_rise_time(y, sr, t, 4.0) for t, _ in strongest) if r is not None]
    if len(rises) < MIN_ONSETS_FOR_SPEED:
        return 1.0
    f = float(np.median(rises)) / RISE_REF_S
    return float(min(round(f), MAX_AUTO_SPEED)) if f >= BAKED_FROM else 1.0


@dataclass
class Sound:
    """What the audio of one video says. `impacts` are (file time in seconds, strength) pairs, strongest kept only."""
    impacts: list[tuple[float, float]] = field(default_factory=list)
    speed: float = 1.0  # 1.0 normal; f for audio baked slow by f
    speed_source: str = "auto"
    note: str | None = None  # why there is nothing to use (no audio track)


def analyse(path: str, video_offset: float, speed: str | float = "auto") -> Sound:
    """Impacts of `path` in file time. `speed`: "auto", "normal", or a factor (baked slow motion). No audio track: a note, no error."""
    y = read_audio(path)
    if y is None:
        return Sound(note="no audio track, or ffmpeg could not decode it: sound cues skipped")
    if len(y) < SR // 2:
        return Sound(note="audio shorter than half a second: sound cues skipped")
    flux, t = onset_envelope(y)
    found = onsets(flux, t)
    t0 = audio_start_in_file_time(path, video_offset)
    if speed == "normal":
        f, src = 1.0, "flag"
    elif speed == "auto":
        f, src = estimate_speed(y, found), "auto"
    else:
        f, src = max(1.0, float(speed)), "flag"
    return Sound([(t0 + ot, s) for ot, s in found if s >= MIN_STRENGTH], f, src)


def _within(impacts: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    return [(t, s) for t, s in impacts if lo <= t <= hi]


def snap_clips(clips: list[tuple[float, float]], impacts: list[tuple[float, float]], keyframes: list[float], lo: float, hi: float,
               speed: float = 1.0, presence: list[tuple[float, float]] | None = None, frame_s: float = 1 / 30) -> list[tuple[float, float, dict]]:
    """Move each clip's start to just before a pop and its end to just after a landing.

    Start: the earliest impact within SNAP_RADIUS_S of the start; the clip then starts LEAD_S before it, at the keyframe at or
    before that when the keyframe is within MAX_MOVE_S of the old start (a stream copy starts there anyway). End: the latest impact
    within SNAP_RADIUS_S of the end, plus TAIL_S. No edge moves more than MAX_MOVE_S, none crosses a neighbour or [lo, hi], a clip
    never gets shorter than MIN_CLIP_S, and a start already before its pop by LEAD_S is only moved when that tightens it.
    The search radius, lead and tail are multiplied by `speed` (baked slow motion); the 1 s move cap and the minimum length are not.
    With `presence` (the unpadded intervals where the target was seen) no edge moves into one: an edge only tightens as far as
    the presence it encloses, so a small --pad is never trimmed. Returns (start, end, info) per clip; info holds the old edges
    under "start_from" / "end_from" when they moved. A start on a keyframe is requested two frames (`frame_s`) plus KEYFRAME_EPS_S
    after it, so the stream copy starts at that keyframe and not one GOP earlier (`probe.seek_landing`)."""
    R, LEAD, TAIL = (v * speed for v in (SNAP_RADIUS_S, LEAD_S, TAIL_S))
    MOVE, MINLEN = MAX_MOVE_S, MIN_CLIP_S
    seen = sorted(presence or [])
    out: list[tuple[float, float, dict]] = []
    for i, (s, e) in enumerate(clips):
        info: dict = {}
        ns, ne = s, e
        near = _within(impacts, s - R, s + R)
        if near:
            pop = min(t for t, _ in near)
            target = min(max(pop - LEAD, s - MOVE), s + MOVE)
            if target < pop - SNAP_MARGIN_S:  # still ahead of the pop (a fixed margin: the cap may bind at any speed)
                kf = snap_to_keyframe(target, keyframes)
                while keyframes and kf + 2 * frame_s + KEYFRAME_EPS_S > s + MOVE and kf > keyframes[0]:
                    kf = snap_to_keyframe(kf - 1e-3, keyframes)  # the cap would put the request back inside [kf, kf + 2 frames): use the keyframe before
                req = kf + 2 * frame_s + KEYFRAME_EPS_S  # past the demuxer's bad window
                if keyframes and abs(kf - s) <= MOVE and kf >= lo and req <= s + MOVE:
                    ns = req
                else:
                    ns = target
        near = _within(impacts, e - R, e + R)
        if near:
            land = max(t for t, _ in near)
            target = min(max(land + TAIL, e - MOVE), e + MOVE)
            if target > land + SNAP_MARGIN_S:
                ne = target
        for ps, pe in seen:  # never trim into presence: the start stays at or before the presence it reaches, the end at or after
            if ns > s and pe > s and ps < ns:
                ns = max(s, ps)
            if ne < e and ps < e and pe > ne:
                ne = min(e, pe)
        ns = max(ns, lo, out[-1][1] if out else lo)
        ne = min(ne, hi, clips[i + 1][0] if i + 1 < len(clips) else hi)
        if ne - ns < MINLEN:
            ns, ne = s, e
        if abs(ns - s) > 1e-9:
            info["start_from"] = s
        if abs(ne - e) > 1e-9:
            info["end_from"] = e
        out.append((ns, ne, info))
    return out


def clip_cues(impacts: list[tuple[float, float]], start: float, end: float, peak: float) -> dict:
    """What the sound says about one clip: its impact markers (file time), their count, and the raised confidence."""
    here = _within(impacts, start, end)
    conf = min(1.0, peak + CONFIDENCE_BOOST) if here else peak
    return dict(impacts=len(here), markers=[dict(time=round(t, 3), strength=round(s, 2)) for t, s in here], confidence=round(conf, 5))
