"""Lossless stream-copy cuts. `-ss` before `-i`, so a clip starts at the keyframe at or before the
requested start (early by up to one GOP, never late) and the video's display rotation is kept.

A second stream-copy pass lines the audio up with the picture. In an open-GOP stream (iPhone HEVC) the
pictures that follow a keyframe in decode order can show before it and cannot be decoded without the
previous GOP, so the first picture a player shows is the keyframe, later than the first packet. The first
pass leaves the audio starting about 0.2 s before that picture. The second pass starts both streams at the
keyframe: the audio before it is dropped, the timeline is shifted to zero, and the undecodable leading
pictures stay in the file at negative times, where the edit list hides them. Nothing is re-encoded.

Slow-motion playback (T3c): a third stream-copy pass multiplies every video
timestamp by the slow-down factor with the `setts` bitstream filter, around the first picture so it stays where the
align pass put it. The picture packets are untouched. Audio is dropped, or with `slowmo_audio` stretched by `atempo` and
re-encoded to AAC (the only re-encode, and only of audio)."""
from __future__ import annotations

import bisect
import csv
import subprocess
from pathlib import Path

from .names import safe_stem
from .photos import slowmo_factor
from .probe import _ffprobe, audio_frame_seconds, capture_fps, container_start_time, first_packet_pts, seek_landing

AAC_FRAME_S = 1024 / 44100  # default audio packet length (44.1 kHz AAC); cut_clip reads the real one from the clip


def mmss(t: float) -> str:
    s = int(t)
    return f"{s // 60:02d}{s % 60:02d}"


def clip_name(stem: str, index: int, start: float, end: float, suffix: str = ".mov") -> str:
    stem = safe_stem(stem)  # a line break or control byte would split a list or a log line
    return f"{stem}_{index:02d}_{mmss(start)}-{mmss(end)}{suffix}"


def snap_to_keyframe(t: float, keyframes: list[float]) -> float:
    """A guess at where a stream-copy cut requested at `t` starts: the keyframe at or before it. It is the guess shown
    before a cut and the fallback for a cut that failed; in an open-GOP file the real start can be one GOP earlier
    (`probe.seek_landing`), so what the sidecar records is what `cut_clip` returns."""
    if not keyframes:
        return t
    i = bisect.bisect_right(keyframes, t + 1e-6) - 1
    return keyframes[max(i, 0)]


def cut_command(src: str, out: str, start: float, end: float) -> list[str]:
    return [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-ss", f"{start:.3f}", "-i", src, "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c", "copy",
        "-map_metadata", "-1",  # container tags can hold GPS; stream metadata and rotation are kept
        "-avoid_negative_ts", "make_zero",
        out,
    ]


def align_command(src: str, out: str, video_start: float, audio_frame_s: float = AAC_FRAME_S) -> list[str]:
    """Second pass: start every stream at `video_start`, the first picture a player can show, in the file's own
    timeline (`-ss` counts from the container start, which the first pass leaves above zero when the audio
    starts late). The audio cut point sits half an AAC frame before it, so the first audio packet is within
    half a frame."""
    return [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-ss", f"{max(video_start - audio_frame_s / 2, 0.0):.6f}", "-i", src,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c", "copy",
        "-copypriorss", "0",  # drop audio packets before the cut point
        "-map_metadata", "-1",
        "-avoid_negative_ts", "disabled",  # the leading pictures keep negative times; the edit list hides them
        out,
    ]


SLOWMO_MAX = 100.0
AUTO_MIN_CAPTURE_FPS = 100.0  # --slowmo auto retimes only high-speed captures (120 fps and up); 60 and 59.94 fps are ordinary phone video
AAC_PRIMING_HALF = 512  # the AAC encoder puts 1024 samples of priming before the first sample; starting the audio half a frame late
# centres the first audio packet half a frame (10.7 ms at 48 kHz) before the first picture instead of a whole frame (21.3 ms, over the limit)
AUDIO_BITRATE = "128k"  # the stretched audio is re-encoded; speech and ambient sound do not need more


def parse_slowmo(value: object) -> str:
    """Normalise a --slowmo value to "off", "auto" or a factor above 1 written as a short number ("4", "8", "2.5")."""
    v = str(value).strip().lower()
    if v in ("off", "auto"):
        return v
    try:
        f = float(v)
    except ValueError:
        f = float("nan")
    if not (f == f and f != float("inf")):
        raise ValueError(f"--slowmo must be auto, off or a factor such as 4 or 8 (got {value!r})")
    if f == 1.0:
        return "off"
    if not 1.0 < f <= SLOWMO_MAX:
        raise ValueError(f"--slowmo factor must be above 1 and at most {SLOWMO_MAX:g} (got {value!r}); clips are only slowed down")
    s = f"{f:g}"
    return "off" if float(s) == 1.0 else s  # 1.0000001 prints as "1": that is no slow motion


def resolve_slowmo(value: str, source_capture_fps: float) -> float:
    """The slow-down factor for a normalised --slowmo value: 1.0 (no retime), the number, or for auto the capture rate over
    30 fps (240 fps gives 8, 120 gives 4, 1.0 for a file under 100 fps, where 60 fps video is an ordinary recording, and never
    above SLOWMO_MAX). The speed ramp inside a Photos slow-motion
    edit cannot be read, so auto is one constant factor."""
    v = parse_slowmo(value)
    if v == "off":
        return 1.0
    if v == "auto":
        if source_capture_fps < AUTO_MIN_CAPTURE_FPS:
            return 1.0
        return min(slowmo_factor(source_capture_fps), SLOWMO_MAX)
    return float(v)


def atempo_chain(factor: float) -> str:
    """ffmpeg audio filter that plays audio `factor` times slower at the same pitch (atempo takes 0.5 to 100 per stage in
    ffmpeg 5 and later; stages of 0.5 keep it valid for old builds too)."""
    r = 1.0 / factor
    parts = []
    while r < 0.5 - 1e-9:
        parts.append("0.5")
        r *= 2
    if abs(r - 1.0) > 1e-9:
        parts.append(f"{r:.8g}")
    return ",".join(f"atempo={p}" for p in parts) or "anull"


def retime_command(src: str, out: str, factor: float, audio: bool) -> list[str]:
    """Third pass: video timestamps times `factor` (setts, stream copy); audio dropped, or stretched and re-encoded.
    `setts` needs pts and dts separately (ts= would overwrite pts with dts and break B-frame order). The scaling is
    anchored at the stream's first timestamps, STARTPTS, so the first picture does not move: the A/V start offset the
    align pass made (at most half an audio frame) is kept instead of growing by the factor. Stretched audio starts
    AAC_PRIMING_HALF samples late, so its first packet is 10.7 ms from the first picture at every start (see above)."""
    f = f"{factor:.8g}"
    setts = f"setts=pts=STARTPTS+(PTS-STARTPTS)*{f}:dts=STARTPTS+(DTS-STARTPTS)*{f}:duration=DURATION*{f}"
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostats", "-nostdin", "-i", src, "-map", "0:v:0"]
    if audio:
        cmd += ["-map", "0:a:0?", "-af", f"{atempo_chain(factor)},asetpts=(NB_CONSUMED_SAMPLES+{AAC_PRIMING_HALF})/(SR*TB)",
                "-c:a", "aac", "-b:a", AUDIO_BITRATE]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "copy", "-bsf:v", setts, "-map_metadata", "-1", "-avoid_negative_ts", "disabled", out]
    return cmd


def retimed_facts(path: str) -> dict:
    """What a retimed clip really is, read with ffprobe: duration, fps, whether it has audio, and the A/V start offset
    (first audio packet minus first picture, in ms; None without audio)."""
    d = _ffprobe(path, "-show_entries", "format=duration", what="a clip")
    v, a = first_packet_pts(path, what="a clip")
    return dict(duration_s=round(float(d["format"]["duration"]), 3), fps=round(capture_fps(path, what="a clip"), 3), has_audio=a is not None,
                av_offset_ms=None if a is None else round((a - v) * 1000, 1))


DURATION_WARN_S = 0.5  # a clip this much longer or shorter than the sidecar says is wrong in a way a reader would notice


def written_duration(path: str) -> float | None:
    """The duration of the clip file's video stream as ffprobe reports it, None when it reports none."""
    d = _ffprobe(path, "-select_streams", "v:0", "-show_entries", "stream=duration", what="a clip")
    try:
        return float((d.get("streams") or [{}])[0]["duration"])
    except (KeyError, TypeError, ValueError):
        return None


def duration_warning(path: str, got: float | None, copy_start: float, end: float, factor: float = 1.0) -> str | None:
    """A sentence when the written clip's video (`got` seconds) differs from (end - copy_start) * factor by more than DURATION_WARN_S."""
    want = (end - copy_start) * factor
    if got is None or abs(got - want) <= DURATION_WARN_S:
        return None
    return f"{Path(path).name}: the file's video lasts {got:.3f} s but the cut should last {want:.3f} s (copy start {copy_start:.3f} s)"


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg {what}: {r.stderr.strip()[-300:]}")


def cut_clip(src: str, out: str | Path, start: float, end: float, slowmo: float = 1.0, slowmo_audio: bool = False) -> float:
    """Cut [start, end] without re-encoding the picture. `slowmo` above 1 then slows the clip down by that factor (the
    clip lasts factor times longer, at 1/factor of the fps); audio is dropped unless `slowmo_audio`.
    Returns the source time of the clip's first picture (clip time 0), measured with the same seek the cut uses."""
    out = Path(out)
    part = out.with_name(out.stem + ".part" + out.suffix)  # does not match the clip names clear_old_clips removes
    aligned = out.with_name(out.stem + ".aligned.part" + out.suffix)
    try:
        _run(cut_command(src, str(part), start, end), f"cut failed ({start:.2f}-{end:.2f})")
        video_start, _ = first_packet_pts(str(part), what="a cut piece")
        frame_s = audio_frame_seconds(str(part), what="a cut piece") or AAC_FRAME_S  # 1024 samples at the clip's own rate (48 kHz is 21 ms)
        target = aligned if slowmo > 1.0 else out
        _run(align_command(str(part), str(target), video_start - container_start_time(str(part), what="a cut piece"), frame_s), f"align failed ({start:.2f}-{end:.2f})")
        if slowmo > 1.0:
            _run(retime_command(str(aligned), str(out), slowmo, slowmo_audio), f"slow-motion retime failed ({start:.2f}-{end:.2f})")
        return round(seek_landing(src, start), 3)  # the same seek, measured, so the sidecar describes this file
    finally:
        part.unlink(missing_ok=True)
        aligned.unlink(missing_ok=True)


def write_losslesscut_csv(path: str | Path, clips: list[tuple[float, float, str]]) -> None:
    """LosslessCut CSV project: start,end,label per row, seconds, no header."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for s, e, label in clips:
            w.writerow([f"{s:.3f}", f"{e:.3f}", label])
