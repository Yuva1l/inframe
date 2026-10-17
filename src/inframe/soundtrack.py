"""The music track of a reel: decode, trim or loop to the reel's length, fade in and out, set to -14 LUFS.

Loudness is measured with ffmpeg's ebur128 filter (integrated loudness, ITU-R BS.1770 gating) and set with a plain gain
followed by a limiter at -1 dBFS, then measured again; the gain is corrected until the result is within 0.3 LU of the
target, so the number in the report is a measurement of the finished audio, not a promise of a filter."""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

TARGET_LUFS = -14.0
LUFS_TOLERANCE = 1.0  # the reel must land within this many LU of the target
SAMPLE_RATE = 48000
PEAK_LIMIT = 0.891  # -1 dBFS
FADE_IN_S = 0.5
FADE_OUT_S = 2.0
_FF = ["ffmpeg", "-y", "-nostdin", "-nostats", "-loglevel", "error"]


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg {what}: {r.stderr.strip()[-300:]}")
    return r


def measure_lufs(path: str | Path) -> float:
    """Integrated loudness in LUFS of the first audio stream of `path` (-inf for silence)."""
    r = _run(["ffmpeg", "-nostdin", "-nostats", "-loglevel", "error", "-i", str(path), "-vn", "-map", "0:a:0",
              "-af", "ebur128=metadata=1,ametadata=mode=print:key=lavfi.r128.I:file=-", "-f", "null", "-"], "loudness measurement")
    vals = re.findall(r"lavfi\.r128\.I=(-?[\d.]+|-inf)", r.stdout)
    if not vals:
        raise RuntimeError(f"no loudness value for {path} (no audio stream?)")
    v = float(vals[-1])
    return v if v > -69.9 else -math.inf  # ebur128 reports -70 for silence


def audio_duration(path: str | Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise RuntimeError(f"ffprobe found no audio in {path}") from None


def fades(duration: float) -> tuple[float, float]:
    """Fade-in and fade-out lengths for a reel of `duration` seconds (shorter reels get proportionally shorter fades)."""
    return min(FADE_IN_S, duration / 6), min(FADE_OUT_S, duration / 3)


def render_soundtrack(music: str | Path, out: str | Path, duration: float, start: float = 0.0, loop_end: float | None = None) -> dict:
    """Write `out` (AAC in .m4a, 48 kHz stereo, exactly `duration` seconds) from `music`.
    The sound starts `start` seconds into the track. When the track from there is shorter than `duration` it is
    looped; `loop_end` (seconds in the track) ends each pass early, which lets a caller loop between two beats so
    the seam falls on the beat grid. Returns {"lufs", "gain_db", "looped", "fade_in_s", "fade_out_s", "source_s"}."""
    if duration <= 0:
        raise ValueError("duration must be more than 0")
    work = Path(tempfile.mkdtemp(prefix="inframe_music_"))
    try:
        full = work / "full.wav"
        _run([*_FF, "-i", str(music), "-vn", "-map", "0:a:0", "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(full)], "could not decode the music")
        total = audio_duration(full)
        end = min(loop_end, total) if loop_end else total
        if not 0 <= start < end:
            raise ValueError(f"the music is {total:.1f} s long; nothing left from {start:.1f} s")
        body = work / "body.wav"
        _run([*_FF, "-ss", f"{start:.6f}", "-i", str(full), "-t", f"{end - start:.6f}", "-c:a", "pcm_f32le", str(body)], "could not trim the music")
        looped = (end - start) < duration - 1e-6
        fin, fout = fades(duration)
        shaped = work / "shaped.wav"
        loop_in = ["-stream_loop", "-1"] if looped else []
        _run([*_FF, *loop_in, "-i", str(body), "-t", f"{duration:.6f}",
              "-af", f"afade=t=in:st=0:d={fin:.3f},afade=t=out:st={duration - fout:.6f}:d={fout:.3f}",
              "-c:a", "pcm_f32le", str(shaped)], "could not shape the music")
        base = measure_lufs(shaped)
        if not math.isfinite(base):
            raise RuntimeError("the music is silent")
        gain = TARGET_LUFS - base
        got = base
        leveled = work / "leveled.wav"
        for _ in range(5):
            applied = gain  # the gain of the file measured below
            _run([*_FF, "-i", str(shaped), "-af", f"volume={gain:.4f}dB,alimiter=limit={PEAK_LIMIT}:level=false:attack=2:release=50",
                  "-c:a", "pcm_f32le", str(leveled)], "could not set the loudness")
            got = measure_lufs(leveled)
            if abs(got - TARGET_LUFS) <= 0.3:
                break
            gain += TARGET_LUFS - got  # the limiter took some loudness back; ask for the difference
        _run([*_FF, "-i", str(leveled), "-t", f"{duration:.6f}", "-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE), "-ac", "2", str(out)], "could not encode the music")
        return dict(lufs=round(got, 2), gain_db=round(applied, 2), looped=looped, fade_in_s=round(fin, 3), fade_out_s=round(fout, 3), source_s=round(total, 2))
    finally:
        shutil.rmtree(work, ignore_errors=True)
