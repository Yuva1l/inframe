"""ffprobe wrapper. Reads only the stream facts we need (never container tags, which may hold GPS)."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class VideoInfo:
    path: str
    codec: str
    profile: str | None
    width: int  # stored size, before rotation
    height: int
    pix_fmt: str
    color_range: str | None
    fps: float
    rotation: int  # degrees as ffprobe reports it (display matrix); 0 if none
    duration: float
    start_time: float
    nb_frames: int | None
    has_audio: bool
    audio_codec: str | None
    keyframes: list[float] = field(default_factory=list)  # seconds, sorted, in file time (see end_time)
    format_start_time: float = 0.0  # container start; ffmpeg -ss and every time we report are relative to it
    mirrored: bool = False  # the display matrix flips the picture (negative determinant): applied after `rotation`
    audio_rate: int | None = None  # Hz of the first audio stream
    sar: float = 1.0  # sample aspect ratio (display width / stored width before rotation); 1.0 when square or unknown

    @property
    def offset(self) -> float:
        """Where the video stream starts in file time (> 0 when the audio, say, starts first)."""
        return self.start_time - self.format_start_time

    @property
    def end_time(self) -> float:
        """Where the video stream ends in file time. All sample, keyframe, segment and cut times use file time,
        the timeline `ffmpeg -ss` uses, so a video that starts late is still cut at the frames we scored."""
        return self.offset + self.duration

    @property
    def display_size(self) -> tuple[int, int]:
        """(width, height) as shown, after the sample aspect ratio (an anamorphic file is wider than its stored pixels) and rotation."""
        w = round(self.width * self.sar) if self.sar > 0 else self.width
        if self.rotation % 180 != 0:
            return self.height, w
        return w, self.height

    def to_dict(self, keyframes: bool = False) -> dict:
        d = asdict(self)
        kf = d.pop("keyframes")
        d["display_width"], d["display_height"] = self.display_size
        d["n_keyframes"] = len(kf)
        if kf:
            gaps = [b - a for a, b in zip(kf, kf[1:])]
            d["keyframe_gap_max_s"] = round(max(gaps), 3) if gaps else None
            d["keyframe_gap_mean_s"] = round(sum(gaps) / len(gaps), 3) if gaps else None
        if keyframes:
            d["keyframes"] = kf
        return d


class ProbeError(ValueError):
    """ffprobe could not read the file (missing, unreadable, truncated, not a media file). The message carries its last error line."""


def _ffprobe(path: str, *args: str, what: str = "a video") -> dict:
    """`what` names the file in the error ("as a video" for the user's input; an intermediate file says what it is)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-of", "json", *args, path], capture_output=True, text=True)
    if r.returncode != 0:
        why = (r.stderr or "").strip().splitlines()
        raise ProbeError(f"ffprobe could not read {path} as {what}: {why[-1] if why else 'unknown error'}")
    return json.loads(r.stdout or "{}")


CAPTURE_VS_AVG_MAX = 2.0  # a slow-motion original with a pre-roll has r_frame_rate / avg_frame_rate of 1.4 (240 fps)


def _rate(s: str | None) -> float:
    if not s or "/" not in s:
        return float(s) if s else 0.0
    n, d = s.split("/")
    return float(n) / float(d) if float(d) else 0.0


def parse_rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"])))
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) if tag is not None else 0


def parse_mirrored(stream: dict) -> bool:
    """True when the display matrix mirrors the picture (determinant of its 2x2 part is negative). ffprobe's
    `rotation` alone cannot say so: a pure horizontal flip reads as rotation 180, a vertical flip as 0. The picture
    is upright after the rotation and then a vertical flip."""
    for sd in stream.get("side_data_list", []) or []:
        rows = [ln.split(":", 1)[1].split() for ln in str(sd.get("displaymatrix", "")).splitlines() if ":" in ln]
        if len(rows) >= 2 and len(rows[0]) >= 2 and len(rows[1]) >= 2:
            try:
                a, b, c, d = (float(x) for x in (*rows[0][:2], *rows[1][:2]))
            except ValueError:
                return False
            return a * d - b * c < 0
    return False


def parse_sar(stream: dict) -> float:
    """The stream's sample_aspect_ratio ("4:3") as a number; 1.0 when it is missing, "0:1" (unknown) or not a ratio."""
    try:
        n, d = str(stream.get("sample_aspect_ratio", "")).split(":")
        n, d = float(n), float(d)
    except ValueError:
        return 1.0
    return n / d if n > 0 and d > 0 else 1.0


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def keyframe_times(path: str, start_time: float = 0.0) -> list[float]:
    d = _ffprobe(path, "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags")
    ts = [
        float(p["pts_time"]) - start_time
        for p in d.get("packets", [])
        if "K" in p.get("flags", "") and p.get("pts_time") not in (None, "N/A")
    ]
    return sorted(ts)


def capture_fps(path: str, what: str = "a video") -> float:
    """The stream's r_frame_rate (the rate of the fastest frames), 0.0 if unknown. avg_frame_rate, which `probe` prefers,
    is pulled down by a short 30 fps pre-roll at the head of a slow-motion file, so r_frame_rate
    wins unless it is more than twice the average: then the timestamps are irregular (a variable-frame-rate file at normal
    speed can report r_frame_rate 450 with avg_frame_rate 30) and the average is the honest rate."""
    d = _ffprobe(path, "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate", what=what)
    st = (d.get("streams") or [{}])[0]
    r, avg = _rate(st.get("r_frame_rate")), _rate(st.get("avg_frame_rate"))
    if avg > 0 and r > CAPTURE_VS_AVG_MAX * avg:
        return avg
    return r


def probe(path: str, with_keyframes: bool = True, what: str = "a video") -> VideoInfo:
    d = _ffprobe(
        path,
        "-show_entries",
        "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,color_range,"
        "avg_frame_rate,r_frame_rate,sample_aspect_ratio,sample_rate,duration,start_time,nb_frames:stream_tags=rotate"
        ":stream_side_data=rotation,displaymatrix:format=duration,start_time",
        what=what,
    )
    streams = d.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError(f"no video stream in {path}")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = d.get("format", {})
    dur = v.get("duration") or fmt.get("duration")
    fps = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))
    fstart = _num(fmt.get("start_time"))
    start = _num(v.get("start_time")) if v.get("start_time") not in (None, "N/A") else fstart
    nb = v.get("nb_frames")
    info = VideoInfo(
        path=path,
        codec=v["codec_name"],
        profile=v.get("profile"),
        width=int(v["width"]),
        height=int(v["height"]),
        pix_fmt=v.get("pix_fmt", ""),
        color_range=v.get("color_range"),
        fps=fps,
        rotation=parse_rotation(v),
        duration=float(dur),
        start_time=start,
        nb_frames=int(nb) if nb not in (None, "N/A") else None,
        has_audio=a is not None,
        audio_codec=a.get("codec_name") if a else None,
        format_start_time=fstart,
        mirrored=parse_mirrored(v),
        sar=parse_sar(v),
        audio_rate=int(a["sample_rate"]) if a and str(a.get("sample_rate", "")).isdigit() else None,
    )
    if with_keyframes:
        info.keyframes = keyframe_times(path, fstart)
    return info


def first_packet_pts(path: str, what: str = "a video") -> tuple[float, float | None]:
    """(video, audio) pts in seconds of each stream's first packet in file order (audio None without audio).
    For a clip cut at a keyframe the first video packet is that keyframe, the first picture a player can show."""
    def one(sel: str) -> float | None:
        d = _ffprobe(path, "-select_streams", sel, "-read_intervals", "%+#1", "-show_entries", "packet=pts_time", what=what)
        pk = d.get("packets", [])
        return float(pk[0]["pts_time"]) if pk and pk[0].get("pts_time") not in (None, "N/A") else None

    v = one("v:0")
    if v is None:
        raise ValueError(f"no video packets in {path}")
    return v, one("a:0")


def seek_landing(path: str, start: float) -> float:
    """Where a stream-copy cut asked to start at `start` (seconds, the `-ss` timeline of `cut_command`) really starts: the
    source time of the first video packet that `-ss {start:.3f} -i` hands to the muxer, which is the first picture a player
    shows (the keyframe). It is measured, not predicted: in an open-GOP file whose video track has a sync sample group
    (iPhone HEVC) ffmpeg's mov demuxer will not seek to a keyframe for a request within two frames after it and goes back
    one GOP instead, so the keyframe at or before `start` is not always where the copy begins."""
    cmd = ["ffmpeg", "-nostats", "-loglevel", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", path, "-map", "0:v:0", "-c", "copy",
           "-frames:v", "1", "-avoid_negative_ts", "disabled", "-f", "framecrc", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg seek check at {start:.3f} s: {r.stderr.strip()[-300:]}")
    num, den = 1, 1
    for line in r.stdout.splitlines():
        if line.startswith("#tb 0:"):
            num, den = (int(x) for x in line.split(":", 1)[1].strip().split("/"))
        elif line and not line.startswith("#"):
            return start + int(line.split(",")[2]) * num / den  # stream, dts, pts, duration, size, crc
    raise ValueError(f"no video packet at {start:.3f} s in {path}")


def container_start_time(path: str, what: str = "a video") -> float:
    """The container's start time. ffmpeg's `-ss` on this file counts from it, not from zero."""
    d = _ffprobe(path, "-show_entries", "format=start_time", what=what)
    return _num(d.get("format", {}).get("start_time"))


def audio_frame_seconds(path: str, what: str = "a video") -> float:
    """Length of one audio packet of the first audio stream (AAC: 1024 samples, 23 ms at 44.1 kHz, 21 ms at 48 kHz).
    Taken from the first packet; 1024 samples at the stream's rate when the packet carries no duration. 0.0 without audio."""
    d = _ffprobe(path, "-select_streams", "a:0", "-read_intervals", "%+#1",
                 "-show_entries", "packet=duration_time:stream=sample_rate", what=what)
    pk = d.get("packets", [])
    if pk and pk[0].get("duration_time") not in (None, "N/A"):
        return float(pk[0]["duration_time"])
    st = d.get("streams", [])
    rate = _num(st[0].get("sample_rate")) if st else 0.0
    return 1024 / rate if rate else 0.0
