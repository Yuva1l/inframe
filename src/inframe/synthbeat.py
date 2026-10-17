"""inframe's built-in beat: synthetic music rendered with numpy only (no recording, so no license and no footage involved).
`tools/synth_music.py` is its command line; auto music (`automusic.builtin_beat`) renders the fallback song with it."""
from __future__ import annotations

import wave

import numpy as np

SR = 44100


def beat_grid(bpm: float, seconds: float, offset: float = 0.0) -> np.ndarray:
    """The true beat times `render` puts its beats on."""
    return np.arange(offset, seconds - 0.3, 60.0 / bpm)


def render(bpm: float, seconds: float, offset: float = 0.0, kind: str = "click", seed: int = 0, bed: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(mono float32 samples at 44.1 kHz, true beat times). `bed` adds a steady 1250 Hz tone of that amplitude under the clicks: bare
    clicks are so sparse that -14 LUFS would need peaks past full scale, as with any very peaky recording."""
    n = int(seconds * SR)
    y = np.zeros(n + SR, dtype=np.float64)
    period = 60.0 / bpm
    beats = beat_grid(bpm, seconds, offset)
    rng = np.random.default_rng(seed)
    for k, t in enumerate(beats):
        i = int(round(t * SR))
        if kind == "click":
            m = np.arange(400)
            y[i:i + 400] += 0.8 * np.sin(2 * np.pi * 1000 * m / SR) * np.exp(-m / 80)
            continue
        m = np.arange(int(0.25 * SR))
        f = 55 + 90 * np.exp(-m / (0.03 * SR))  # kick: pitch falls from 145 to 55 Hz
        y[i:i + m.size] += (0.9 if k % 4 == 0 else 0.7) * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-m / (0.08 * SR))
        j = int(round((t + period / 2) * SR))  # hat on the off-beat
        h = np.arange(int(0.05 * SR))
        y[j:j + h.size] += 0.15 * rng.normal(size=h.size) * np.exp(-h / (0.01 * SR))
    if bed:
        y += bed * np.sin(2 * np.pi * 1250 * np.arange(y.size) / SR)
    if kind == "beat":
        t = np.arange(y.size) / SR
        for f in (220.0, 277.18, 329.63):
            y += 0.05 * np.sin(2 * np.pi * f * t)
    y = y[:n]
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32), beats


def write_wav(path: str, y: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
