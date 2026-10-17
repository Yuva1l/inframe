"""Beat times of a music file, and the plan that puts each clip change of a reel on a beat.

Beat tracking is librosa (ISC; the `music` extra): an onset-strength envelope with a short window (512 samples at a
hop of 64, 22.05 kHz, so about 3 ms) on the low band only (up to 300 Hz, where the kick is; hats and snares sit above and
pull the tracker onto the off-beat) and `beat_track` on that envelope. On synthetic click tracks and kick-and-hat tracks at
100-128 BPM the beats land within 6 ms of the kicks. Downbeats are not detected, so a join lands on any beat, not on a bar line."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

ANALYSIS_RATE = 22050
N_FFT = 512
HOP = 64
MIN_BEATS = 8
LOW_BAND_HZ = 300  # onset strength is computed below this frequency
LOW_BAND_MELS = 8  # few mel bands: at n_fft 512 only about seven FFT bins lie below 300 Hz


class NoBeatError(RuntimeError):
    pass


@dataclass
class BeatGrid:
    tempo: float
    times: list[float]  # seconds in the music file, increasing
    duration: float  # length of the music file


def _load_mono(path: str, max_seconds: float | None = None) -> np.ndarray:
    """The first `max_seconds` of the music (all of it by default) as mono float samples at ANALYSIS_RATE."""
    cut = ["-t", f"{max_seconds:g}"] if max_seconds else []
    r = subprocess.run(["ffmpeg", "-nostdin", "-nostats", "-loglevel", "error", "-i", str(path), *cut, "-vn", "-map", "0:a:0", "-ac", "1",
                        "-ar", str(ANALYSIS_RATE), "-f", "f32le", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode the music: {r.stderr.decode(errors='replace').strip()[-300:]}")
    y = np.frombuffer(r.stdout, dtype="<f4")
    if y.size < ANALYSIS_RATE:
        raise NoBeatError("the music is shorter than one second")
    return y


def _librosa():
    try:
        import librosa
    except ImportError as e:  # pragma: no cover - depends on the install
        raise RuntimeError("beat sync needs librosa: install the music extra (uv sync --extra music, or pip install 'inframe[music]')") from e
    return librosa


def analyse_signal(y: np.ndarray) -> tuple[BeatGrid, np.ndarray]:
    """Beat grid and low-band onset envelope of mono samples `y` at ANALYSIS_RATE. Raises NoBeatError when there is no steady beat."""
    librosa = _librosa()
    oenv = librosa.onset.onset_strength(y=y, sr=ANALYSIS_RATE, n_fft=N_FFT, hop_length=HOP, fmax=LOW_BAND_HZ, n_mels=LOW_BAND_MELS)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=ANALYSIS_RATE, hop_length=HOP, start_bpm=120,
                                           tightness=100, trim=True, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = [float(b) for b in np.atleast_1d(beats)]
    if len(beats) < MIN_BEATS or tempo <= 0:
        raise NoBeatError(f"no steady beat found in the music ({len(beats)} beats)")
    return BeatGrid(tempo=tempo, times=beats, duration=len(y) / ANALYSIS_RATE), oenv


def beat_times(path: str) -> BeatGrid:
    """Beat times of the music at `path` (any file ffmpeg reads). Raises NoBeatError when there is no steady beat."""
    _librosa()
    return analyse_signal(_load_mono(path))[0]


@dataclass
class Plan:
    """Where each clip of a beat-synced reel starts and how many of its frames it shows."""
    fps: float
    frames: list[int]  # pictures shown from each clip, in order
    joins: list[dict] = field(default_factory=list)  # per clip end: {"clip", "frame", "reel_s", "beat_s", "err_ms", "on_beat"}
    music_start: float = 0.0  # where in the music the reel begins (its first beat)
    loop_end: float | None = None  # last beat used; a looped track repeats between music_start and this
    warnings: list[str] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(self.frames)

    def to_dict(self) -> dict:
        return dict(fps=self.fps, frames=self.frames, joins=self.joins, music_start=round(self.music_start, 4),
                    loop_end=None if self.loop_end is None else round(self.loop_end, 4), warnings=self.warnings)


def extend_beats(beats: list[float], upto: float) -> tuple[list[float], float, float]:
    """Beat times relative to the first beat, repeated until they pass `upto` seconds. The music between the first and the
    last beat is the loop body, so each repeat starts exactly on a beat. Returns (times, music_start, loop_end)."""
    first, last = beats[0], beats[-1]
    body = last - first
    rel = [b - first for b in beats[:-1]]  # the last beat is the first beat of the next pass
    out: list[float] = []
    k = 0
    while True:
        out += [r + k * body for r in rel]
        k += 1
        if k * body > upto + body:
            break
    out.append(k * body)
    return out, first, last


def plan_beat_cuts(available: list[int], slack: list[int], beats: list[float], fps: float) -> Plan:
    """Choose how many pictures of each clip the reel shows so that every clip change (and the end of the reel) lands on a beat.

    `available[i]` is how many pictures clip i can show, `slack[i]` how many of its last ones are padding that may be
    dropped. The reel starts on the first beat. A clip change is placed on the latest beat that keeps the clip between
    `available - slack` and `available` pictures; the beat is rounded to the nearest picture (so a join is at most half a
    picture from its beat). If no beat falls in that window the clip is cut at the latest beat before its end (into its
    content) when one exists, and otherwise shown whole with its join off the beat; both are warned about."""
    n = len(available)
    if n == 0 or len(slack) != n:
        raise ValueError("one availability and one slack per clip")
    if len(beats) < 2:
        raise NoBeatError("fewer than two beats")
    horizon = sum(available) / fps + 1.0
    grid, music_start, loop_end = extend_beats(beats, horizon)
    gframes = [round(g * fps) for g in grid]  # beat positions in pictures from the reel start
    plan = Plan(fps=fps, frames=[], music_start=music_start, loop_end=loop_end)
    pos = 0
    for i in range(n):
        lo, hi = pos + max(1, available[i] - slack[i]), pos + available[i]
        window = [k for k, g in enumerate(gframes) if lo <= g <= hi]
        on_beat = True
        if window:
            k = window[-1]
        else:
            before = [k for k, g in enumerate(gframes) if pos < g <= hi]
            if before:
                k = before[-1]
                plan.warnings.append(f"clip {i + 1}: no beat in its padding, cut {(hi - gframes[k]) / fps:.2f} s into its content")
            else:
                k, on_beat = None, False
                plan.warnings.append(f"clip {i + 1}: shorter than the gap between beats, joined off the beat")
        end = gframes[k] if k is not None else hi
        plan.frames.append(end - pos)
        err = (end / fps - grid[k]) * 1000 if k is not None else None
        nearest = min(grid, key=lambda g: abs(g - end / fps))
        plan.joins.append(dict(clip=i + 1, frame=end, reel_s=round(end / fps, 4), beat_s=round(grid[k] if k is not None else nearest, 4),
                               err_ms=round(err if err is not None else (end / fps - nearest) * 1000, 2), on_beat=on_beat))
        pos = end
    return plan
