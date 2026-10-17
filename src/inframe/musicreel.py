"""A reel with music instead of the clips' own sound.

Two ways, both with music only (the clips' audio is dropped):
- plain: the lossless reel of `reel.build_reel` (stream copy), then its sound replaced by the music;
- beat sync: every clip change lands on a beat. Landing exactly on a beat needs whole pictures cut at chosen frames, so the
  clips are decoded, trimmed (each clip's tail, inside its padding), joined and encoded once (hevc_videotoolbox at high
  quality, libx265 where the Mac encoder does not exist), keeping the display rotation and the colour tags.

The music is trimmed or looped to the reel, faded in and out, and set to -14 LUFS (`soundtrack`)."""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from pathlib import Path

from .beats import BeatGrid, NoBeatError, Plan, beat_times, plan_beat_cuts
from .reel import build_reel, make_work_dir, shown_pictures, work_suffix
from .soundtrack import LUFS_TOLERANCE, TARGET_LUFS, audio_duration, measure_lufs, render_soundtrack

VT_QUALITY = 85  # hevc_videotoolbox -q:v (0-100). On a 17 s clip (8.2 Mb/s source): q75 = 1.7 Mb/s, 46.4 dB PSNR; q85 = 4.5 Mb/s, 49.7 dB; q92 = 13.1 Mb/s, 54.5 dB
X265_CRF = 14
BEAT_TOLERANCE_FRAMES = 1.0
_FF = ["ffmpeg", "-y", "-nostdin", "-nostats", "-loglevel", "error"]
COLOR_FIELDS = ("pix_fmt", "color_range", "color_space", "color_transfer", "color_primaries")


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg {what}: {r.stderr.strip()[-300:]}")


def pick_encoder() -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return "hevc_videotoolbox" if "hevc_videotoolbox" in r.stdout else "libx265"


def video_params(path: str) -> dict:
    """What the re-encode must keep from `path`: frame rate (as a fraction), size, pixel format, colour tags, rotation."""
    from .probe import probe

    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries",
                        "stream=avg_frame_rate,r_frame_rate,width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries,codec_tag_string"
                        ":stream=duration,start_time", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}: {r.stderr.strip()[-200:]}")
    s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    rate = s.get("avg_frame_rate") if s.get("avg_frame_rate") not in (None, "0/0") else s.get("r_frame_rate", "30/1")
    num, _, den = rate.partition("/")
    out = dict(rate=rate, fps=float(num) / float(den or 1), width=s.get("width"), height=s.get("height"),
               rotation=probe(path, with_keyframes=False).rotation)
    for k in COLOR_FIELDS:
        out[k] = s.get(k)
    return out


def encode_args(encoder: str, p: dict) -> list[str]:
    a = ["-c:v", encoder]
    if encoder == "hevc_videotoolbox":
        a += ["-q:v", str(VT_QUALITY)]
    else:
        a += ["-crf", str(X265_CRF), "-preset", "medium", "-x265-params", "log-level=error"]
    if "10" in (p.get("pix_fmt") or ""):
        a += ["-profile:v", "main10"]
    a += ["-tag:v", "hvc1"]
    for flag, key in (("-color_range", "color_range"), ("-colorspace", "color_space"), ("-color_primaries", "color_primaries"), ("-color_trc", "color_transfer")):
        v = p.get(key)
        if v and v != "unknown":
            a += [flag, v]
    return a


def trim_filter(frames: list[int], rate: str) -> str:
    """filter_complex text: take the first frames[i] pictures of input i, join them, and give the joined stream exact picture times."""
    cadence = f"N/(({rate})*TB)"
    parts = [f"[{i}:v]trim=end_frame={n},setpts={cadence}[t{i}]" for i, n in enumerate(frames)]
    joined = "".join(f"[t{i}]" for i in range(len(frames)))
    return ";".join(parts + [f"{joined}concat=n={len(frames)}:v=1:a=0,setpts={cadence}[v]"])


def encode_cuts(clips: list[str], frames: list[int], out: str, p: dict, encoder: str) -> None:
    cmd = [*_FF]
    for c in clips:
        cmd += ["-noautorotate", "-i", c]
    cmd += ["-an", "-filter_complex", trim_filter(frames, p["rate"]), "-map", "[v]", "-fps_mode", "passthrough",
            *encode_args(encoder, p), "-map_metadata", "-1", out]
    _run(cmd, "re-encode of the beat-synced reel failed")


def mux(video: str, audio: str, out: str, rotation: int, credit: str | None = None) -> None:
    """Video (copied, with the display rotation set) plus the music; no other sound. `credit` is stored in the file's comment tag."""
    cmd = [*_FF]
    if rotation:
        cmd += ["-display_rotation", str(rotation)]
    cmd += ["-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-map_metadata", "-1"]
    if credit:
        cmd += ["-metadata", f"comment={credit}"]
    cmd += ["-movflags", "+faststart", out]
    _run(cmd, "joining picture and music failed")


def _packets(path: str, sel: str) -> list[float]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel, "-of", "json", "-show_entries", "packet=pts_time", path], capture_output=True, text=True)
    return sorted(float(p["pts_time"]) for p in json.loads(r.stdout or "{}").get("packets", []) if p.get("pts_time") not in (None, "N/A"))


def stream_info(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_entries",
                        "stream=index,codec_type,codec_name,codec_tag_string,start_time,duration,sample_rate,channels", path], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


MP4_REFUSED_VIDEO = {"vp8", "theora"}  # an .mp4 holds neither (VP9 and AV1 it does)


def video_duration(path: str) -> float:
    """Length of the first picture stream: its own `duration`, or for a Matroska file (no stream duration field) its DURATION tag."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries", "stream=duration:stream_tags=DURATION", path],
                       capture_output=True, text=True)
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    try:
        return float(st["duration"])
    except (KeyError, ValueError):
        pass
    try:
        h, m, sec = str((st.get("tags") or {}).get("DURATION", "")).split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except ValueError:
        pass
    raise RuntimeError(f"cannot tell how long the picture stream of {path} is: it has neither a duration nor a DURATION tag")


def check_music_reel(reel: str, src: dict, plan: Plan | None = None, grid_check: bool = True) -> dict:
    """Numbers about the finished reel: streams (one picture stream, one AAC stream), loudness against -14 LUFS, rotation and
    colour tags against the source `src` (from `video_params`), and with a `plan` the time of every join against its beat.
    Returns {"ok", "problems", "streams", "lufs", "joins", ...}."""
    from .probe import probe

    problems: list[str] = []
    st = stream_info(reel).get("streams", [])
    kinds = [s["codec_type"] for s in st]
    if kinds.count("video") != 1 or kinds.count("audio") != 1 or len(kinds) != 2:
        problems.append(f"the reel has streams {kinds}, expected one video and one audio")
    a = next((s for s in st if s["codec_type"] == "audio"), None)
    v = next((s for s in st if s["codec_type"] == "video"), None)
    out: dict = dict(streams=kinds, audio_codec=a and a["codec_name"], video_tag=v and v["codec_tag_string"])
    if a is None or v is None:
        return dict(out, ok=False, problems=problems or ["no audio or video stream"])
    if a["codec_name"] != "aac":
        problems.append(f"audio is {a['codec_name']}, expected aac")
    lufs = measure_lufs(reel)
    out["lufs"] = round(lufs, 2)
    if not abs(lufs - TARGET_LUFS) <= LUFS_TOLERANCE:
        problems.append(f"loudness is {lufs:.2f} LUFS, the target is {TARGET_LUFS:g} +-{LUFS_TOLERANCE:g}")
    info = probe(reel, with_keyframes=False, what="the finished reel")
    out["rotation"] = info.rotation
    if info.rotation != src["rotation"]:
        problems.append(f"rotation is {info.rotation}, the clips have {src['rotation']}")
    now = video_params(reel)
    diff = {k: (src[k], now[k]) for k in COLOR_FIELDS if src.get(k) and src[k] != "unknown" and now[k] != src[k]}
    out["colour"] = {k: now[k] for k in COLOR_FIELDS}
    if diff:
        problems.append("colour tags differ from the clips: " + ", ".join(f"{k} {a_}->{b_}" for k, (a_, b_) in diff.items()))
    vdur, adur = float(v.get("duration") or 0), float(a.get("duration") or 0)
    out["video_s"], out["audio_s"] = round(vdur, 3), round(adur, 3)
    fps = now["fps"]
    if abs(vdur - adur) > 1.5 / fps:
        problems.append(f"video is {vdur:.3f} s and audio {adur:.3f} s")
    if plan is not None:
        pts = _packets(reel, "v:0")
        t0 = pts[0] if pts else 0.0
        if len(pts) != plan.total_frames:
            problems.append(f"the reel has {len(pts)} pictures, the plan {plan.total_frames}")
        joins = []
        for j in plan.joins:
            i = j["frame"]
            t = (pts[i] - t0) if i < len(pts) else ((pts[-1] - t0) + 1 / fps if pts else 0.0)
            err = (t - j["beat_s"]) * fps  # in pictures
            joins.append(dict(clip=j["clip"], reel_s=round(t, 4), beat_s=j["beat_s"], err_frames=round(err, 3), on_beat=j["on_beat"]))
            if j["on_beat"] and abs(err) > BEAT_TOLERANCE_FRAMES + 1e-6:
                problems.append(f"clip {j['clip']} ends {err:.2f} pictures from its beat")
            if not j["on_beat"]:
                problems.append(f"clip {j['clip']} could not be joined on a beat")
        out["joins"] = joins
        out["max_join_err_frames"] = round(max((abs(j["err_frames"]) for j in joins), default=0.0), 3)
        out["audio_start_offset_ms"] = round((float(a.get("start_time") or 0) - float(v.get("start_time") or 0)) * 1000, 2)
        if grid_check:
            out["audio_beats"] = audio_beat_check(reel, [j["reel_s"] for j in joins], fps)
    return dict(out, ok=not problems, problems=problems)


def audio_beat_check(reel: str, join_times: list[float], fps: float) -> dict:
    """Independent of the plan: find the beats in the reel's own sound (its time starts at the first sample, the join times at the
    first picture) and report how far each join is from the nearest one. Informational: a tracker may lock to another pulse on real music."""
    try:
        grid = beat_times(reel)
    except (NoBeatError, RuntimeError) as e:
        return dict(error=str(e)[:120])
    errs = [min(abs(t - b) for b in grid.times) for t in join_times]
    return dict(tempo=round(grid.tempo, 1), n_beats=len(grid.times), median_err_ms=round(statistics.median(errs) * 1000, 1),
                max_err_ms=round(max(errs) * 1000, 1), median_err_frames=round(statistics.median(errs) * fps, 2))


def build_music_reel(clips: list[str], out: str | Path, music: str, beat_sync: bool, tail_slack_s: list[float], encoder: str | None = None,
                     credit: str | None = None) -> dict:
    """Join `clips` into `out` with `music` as the only sound. Returns the keys of `reel.build_reel`, plus "music" (loudness,
    looping, fades) and, with beat sync, "plan" (where each clip ends) and the join/beat numbers under "check"."""
    if not clips:
        raise ValueError("no clips to join")
    if len(tail_slack_s) != len(clips):
        raise ValueError("one tail slack per clip")
    if not beat_sync:
        bad = next((x["codec_name"] for c in clips for x in stream_info(c).get("streams", [])
                    if x.get("codec_type") == "video" and x.get("codec_name") in MP4_REFUSED_VIDEO), None)
        if bad:
            raise ValueError(f"a music reel without beat sync copies the pictures into an .mp4, which cannot hold {bad.upper()} video; "
                             "turn beat sync on (it re-encodes the pictures as H.264)")
    out = Path(out)
    work = make_work_dir(out.parent, ".inframe_musicreel_")
    part = out.with_name(out.stem + ".part" + out.suffix)
    try:
        src = video_params(clips[0])
        audio = work / "music.m4a"
        if beat_sync:
            grid = beat_times(music)
            fps = src["fps"]
            avail = [shown_pictures(c) for c in clips]
            slack = [max(0, min(a - 1, round(s * fps))) for a, s in zip(avail, tail_slack_s)]
            plan = plan_beat_cuts(avail, slack, grid.times, fps)
            enc = encoder or pick_encoder()
            duration = plan.total_frames / fps
            sound = render_soundtrack(music, audio, duration, start=plan.music_start, loop_end=plan.loop_end)
            silent = work / "picture.mov"
            encode_cuts(clips, plan.frames, str(silent), src, enc)
            mux(str(silent), str(audio), str(part), src["rotation"], credit)
            check = check_music_reel(str(part), src, plan)
            part.replace(out)
            return dict(file=out.name, ok=check["ok"], n_clips=len(clips), dropped_pictures=0, fallback_used=False, check=check,
                        music=dict(sound, beat_sync=True, tempo=round(grid.tempo, 1), n_beats=len(grid.times), encoder=enc),
                        plan=plan.to_dict(), warnings=plan.warnings)
        own = Path(clips[0]).suffix.lower()
        join_suffix = work_suffix(clips[0])
        # the join keeps the clips' codec tag: .mov/.mp4 clips are joined in their own container (a .mkv would drop VP9's 'vp09'),
        # .webm/.mkv clips (no tag) in a .mkv (an .mp4 would add one)
        lossless = work / ("lossless" + (join_suffix if own in (".mov", ".mp4", ".m4v") and join_suffix in (".mov", ".mp4") else ".mkv"))
        r = build_reel(clips, lossless)
        duration = video_duration(str(lossless))
        sound = render_soundtrack(music, audio, duration)
        mux(str(lossless), str(audio), str(part), src["rotation"], credit)
        check = check_music_reel(str(part), src)
        check["lossless_join_check"] = r["check"]
        # the clips' own sound is dropped, so its A/V offset at a join is not a fault of this reel; every other join problem counts
        own_sound = [p for p in r["check"]["problems"] if "audio starts" in p or p.startswith("audio decode time")]
        join_problems = [p for p in r["check"]["problems"] if p not in own_sound]
        check["ignored_clip_audio_problems"] = own_sound
        check["problems"] = check["problems"] + join_problems
        check["ok"] = not check["problems"]
        part.replace(out)
        ok = check["ok"]
        return dict(file=out.name, ok=ok, n_clips=len(clips), dropped_pictures=r["dropped_pictures"], fallback_used=r["fallback_used"], check=check,
                    music=dict(sound, beat_sync=False), warnings=[])
    finally:
        shutil.rmtree(work, ignore_errors=True)
        part.unlink(missing_ok=True)
