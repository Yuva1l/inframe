"""Highlight reel: the clips joined in time order, losslessly (ffmpeg concat demuxer, stream copy), and the check
that the joins are clean.

The check reads the packets of the finished reel and of each clip (ffprobe, no decode):
- decode timestamps strictly increase in each stream, and at every join the first picture of the next clip comes
  after the last picture of the previous one by no more than one and a half frames, so no join jumps back, overlaps
  or leaves a gap (holes inside a clip are the source's own and are not counted);
- at every join (and at the start) the first audio packet of the clip sits within `AV_TOLERANCE_S` of where it sits
  in the clip itself, measured against the clip's first picture: the limit is for what the join adds, the clip's own
  audio lead (which the cut already limits) is not counted twice.

If a join fails, the reel is built once more with `-avoid_negative_ts make_zero` and checked again.

Open-GOP clips (iPhone HEVC) begin with a few leading pictures that need the previous GOP. Inside a clip an edit list
hides them (the demuxer flags them as discard). A joined file has no such edit list at a join, so a player would
decode them against the tail of the previous clip and flash garbage. Each clip is therefore copied without its
discard-flagged packets and any picture past the clip end (PyAV, still no re-encode) before the join."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

AV_TOLERANCE_S = 0.021  # A/V offset allowed at every join
AV_EPS_S = 1e-4  # timestamps are rounded to 1/44100 s and 1/19200 s
GAP_FRAMES = 1.5  # a step across a join bigger than this many frames is a hole


@dataclass
class ReelCheck:
    ok: bool
    problems: list[str] = field(default_factory=list)
    joins: list[dict] = field(default_factory=list)  # per clip: reel time of its first picture, A/V offset in ms
    max_av_offset_ms: float = 0.0
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return dict(ok=self.ok, problems=self.problems, joins=self.joins,
                    max_av_offset_ms=round(self.max_av_offset_ms, 2), duration_s=round(self.duration_s, 3))


def packets(path: str) -> dict:
    """Packet times of the first video and first audio stream: {"v": [(pts, dts)], "a": [...], "fps", "duration", "matroska"}.
    A packet the file gives no decode time (the first few of a Matroska stream with B-frames) gets its picture time as
    the decode time, which is not in order; "v_dts_given" and "a_dts_given" say per packet whether the time is the file's."""
    def one(sel: str) -> tuple[list[tuple[float, float]], list[bool]]:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel, "-of", "json",
                            "-show_entries", "packet=pts_time,dts_time", path], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffprobe could not read {path}: {r.stderr.strip()[-200:]}")
        out, given = [], []
        for p in json.loads(r.stdout or "{}").get("packets", []):
            pts, dts = p.get("pts_time"), p.get("dts_time")
            if pts in (None, "N/A"):
                continue
            out.append((float(pts), float(dts) if dts not in (None, "N/A") else float(pts)))
            given.append(dts not in (None, "N/A"))
        return out, given

    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
                        "-show_entries", "stream=avg_frame_rate:format=duration,format_name", path], capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    n, _, den = (d["streams"][0].get("avg_frame_rate", "0/1") if d.get("streams") else "0/1").partition("/")
    fps = float(n) / float(den) if den and float(den) else 0.0
    (v, v_given), (a, a_given) = one("v:0"), one("a:0")
    return dict(v=v, a=a, v_dts_given=v_given, a_dts_given=a_given, fps=fps, duration=float(d.get("format", {}).get("duration") or 0.0),
                matroska="matroska" in d.get("format", {}).get("format_name", ""))


def codec_tag(path: str) -> str:
    """The four-character code of the first video stream's sample entry ('hvc1', 'hev1', 'avc1', ...)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries", "stream=codec_tag_string", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}: {r.stderr.strip()[-200:]}")
    st = json.loads(r.stdout or "{}").get("streams") or [{}]
    return st[0].get("codec_tag_string", "")


def shown_pictures(path: str) -> int:
    """How many pictures the clip shows: video packets that are not discard-flagged (leading pictures its edit list
    hides) and whose picture starts before the end of the clip's video. Read from the file itself (ffprobe, no
    decode), so it does not depend on how `strip_outside` decides what to keep."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
                        "-show_entries", "stream=start_time,duration:packet=pts_time,flags", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}: {r.stderr.strip()[-200:]}")
    d = json.loads(r.stdout or "{}")
    st = (d.get("streams") or [{}])[0]
    try:
        end = float(st["start_time"]) + float(st["duration"]) - 1e-4
    except (KeyError, ValueError):
        end = None
    n = 0
    for p in d.get("packets", []):
        if "D" in p.get("flags", "") or p.get("pts_time") in (None, "N/A"):
            continue
        if end is None or float(p["pts_time"]) < end:
            n += 1
    return n


def concat_command(list_file: str, out: str, make_zero: bool = False) -> list[str]:
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", list_file,
           "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-map_metadata", "-1"]
    if make_zero:
        cmd += ["-avoid_negative_ts", "make_zero"]
    return cmd + [out]


def _list_line(p: Path, duration: float | None = None, base: Path | None = None) -> str:
    """One `file` line of the concat list. With `base` (the folder holding the list) a file in it is written by its bare name, which the
    demuxer resolves against the list's own folder, so no parent folder name reaches the list. A path with a line break cannot be
    written (the demuxer reads one line per directive and would take the rest as further directives): ValueError."""
    p = Path(p)
    name = p.name if base is not None and p.resolve().parent == Path(base).resolve() else str(p.resolve())
    if "\n" in name or "\r" in name:
        raise ValueError("a line break in a path cannot go into the concat list")
    line = "file '" + name.replace("'", "'\\''") + "'\n"
    return line + (f"duration {duration:.6f}\n" if duration else "")


def _join_duration(path: str) -> float | None:
    """How long the concat demuxer must give `path` when the file's own duration is not where its last picture ends.
    A kept picture past the clip's end (see `strip_outside`) makes the last picture end later than the file's duration,
    and the next clip would then start on top of it; a Matroska file with Vorbis audio goes the other way: its duration
    runs past the last picture (the audio's last packet is a whole block long) and the next clip would start late,
    leaving a hole at the join. Returns the span from the file's start to the end of its last picture, or None when
    the file's own duration already is that."""
    import av

    with av.open(path) as c:
        vs = c.streams.video[0]
        tb = float(vs.time_base)
        fps = float(vs.average_rate or 30)
        end = max((((p.pts + (p.duration or round(1 / fps / tb))) * tb) for p in c.demux(vs) if p.dts is not None and p.pts is not None), default=0.0)
        start = (c.start_time or 0) / av.time_base
        have = (c.duration or 0) / av.time_base
    need = end - start
    return need if abs(need - have) > 1e-6 else None


def check_reel(reel: str, clips: list[str], sources: list[str] | None = None) -> ReelCheck:
    """Check the joins of `reel`, which was built from `clips` in this order. `sources` are the original clips that
    `clips` were made from (the copies `strip_outside` wrote, whose packet counts split the reel): when given, the
    reel must also hold every picture each original shows and carry the same codec tag, which the copies cannot tell
    (anything lost or re-tagged while copying is invisible to a check against the copies)."""
    per_clip = [packets(c) for c in clips]
    r = packets(reel)
    problems: list[str] = []
    if sources is not None:
        if len(sources) != len(clips):
            raise ValueError("sources and clips must pair up")
        tags = {codec_tag(s) for s in sources}
        tag = codec_tag(reel)
        if tag not in tags:
            problems.append(f"the reel's video is tagged '{tag}' but the clips are '{'/'.join(sorted(tags))}'"
                            + (" (Apple players cannot decode hev1)" if tag == "hev1" else ""))
    for kind, label in (("v", "video"), ("a", "audio")):
        dts = [d for _, d in r[kind]]
        # a Matroska file gives no decode time to the first packets of a B-frame stream; the picture time that stands in for it
        # is not in order, so only the packets that have a time of their own are compared (each with the one before it)
        given = r.get(kind + "_dts_given") or [True] * len(dts)
        bad = [i for i in range(1, len(dts)) if given[i] and given[i - 1] and dts[i] <= dts[i - 1]]
        if bad:
            problems.append(f"{label} decode time does not increase at packet {bad[0]} ({dts[bad[0] - 1]:.4f} then {dts[bad[0]]:.4f} s)")
    fps = r["fps"] or per_clip[0]["fps"] or 30.0
    # packet counts split the reel back into its clips
    joins: list[dict] = []
    vi = ai = 0
    worst = 0.0
    prev_hi: float | None = None
    for k, c in enumerate(per_clip):
        nv, na = len(c["v"]), len(c["a"])
        vg, ag = r["v"][vi:vi + nv], r["a"][ai:ai + na]
        vi, ai = vi + nv, ai + na
        if sources is not None:
            shown = shown_pictures(sources[k])
            if len(vg) < shown:
                problems.append(f"clip {k + 1}: the reel holds {len(vg)} pictures of the {shown} the clip shows")
        if len(vg) != nv or len(ag) != na:
            problems.append(f"clip {k + 1}: the reel has fewer packets than the clip ({len(vg)}/{nv} video, {len(ag)}/{na} audio)")
            continue
        if not vg:
            continue
        v0 = vg[0][0]  # first video packet of the clip in decode order: the keyframe, the first picture a player shows
        row = dict(clip=k + 1, reel_start_s=round(v0, 4))
        lo, hi = min(p for p, _ in vg), max(p for p, _ in vg)
        if prev_hi is not None:
            step = lo - prev_hi
            row["step_ms"] = round(step * 1000, 2)
            if step <= 1e-6:
                problems.append(f"clip {k + 1}: its first picture ({lo:.4f} s) is not after the last picture of clip {k} ({prev_hi:.4f} s)")
            elif step > GAP_FRAMES / fps:
                problems.append(f"clip {k + 1}: a hole of {step:.3f} s in the pictures at the join")
        prev_hi = hi
        if ag:
            own = c["a"][0][0] - c["v"][0][0] if c["a"] and c["v"] else 0.0  # the lead the clip has by itself
            off = ag[0][0] - v0 - own  # what the join adds to it
            row["av_offset_ms"] = round(off * 1000, 2)
            row["clip_av_offset_ms"] = round(own * 1000, 2)
            worst = max(worst, abs(off))
            if abs(off) > AV_TOLERANCE_S + AV_EPS_S:
                problems.append(f"clip {k + 1}: audio starts {off * 1000:.1f} ms further from the picture than in the clip, at the join (limit {AV_TOLERANCE_S * 1000:.0f} ms)")
        joins.append(row)
    if vi != len(r["v"]) or ai != len(r["a"]):
        problems.append(f"the reel has {len(r['v'])} video and {len(r['a'])} audio packets, the clips add up to {vi} and {ai}")
    return ReelCheck(ok=not problems, problems=problems, joins=joins, max_av_offset_ms=worst * 1000, duration_s=r["duration"])


HEVC_CRA, HEVC_BLA_W_RADL = 21, 17


def _cra_to_bla(data: bytes, length_size: int) -> bytes:
    """Rewrite every CRA NAL unit in one length-prefixed HEVC access unit as BLA_W_RADL (a change of one header
    byte per NAL unit). Mid-stream a decoder keeps counting picture order from the picture before a CRA, so the
    POC values of the next clip can collide with the pictures still held from the previous one; a BLA restarts
    the count and drops the leading pictures that need earlier GOPs (already removed). The slice data is untouched."""
    out = bytearray(data)
    i = 0
    while i + length_size + 2 <= len(out):
        n = int.from_bytes(out[i:i + length_size], "big")
        h = i + length_size
        if (out[h] >> 1) & 0x3F == HEVC_CRA:
            out[h] = (out[h] & 0x81) | (HEVC_BLA_W_RADL << 1)
        i = h + n
    return bytes(out)


MP4_ONLY_CODECS = {"vp9", "av1", "opus", "flac"}  # a .mov refuses these ("only supported in MP4")
MKV_ONLY_CODECS = {"vp8", "vorbis", "theora"}  # neither a .mov nor an .mp4 holds these


def work_format(path: str) -> str:
    """The container for a work copy of `path` that holds its streams: 'mov' (the default, what iPhone and camera files use),
    'mp4' for VP9, AV1, Opus or FLAC, 'matroska' for VP8, Vorbis or Theora (a .webm source)."""
    import av

    with av.open(path) as c:
        # canonical_name is the codec's own name; codec_context.name is the decoder in use ('libdav1d' for AV1)
        names = {s.codec_context.codec.canonical_name for s in (*c.streams.video[:1], *c.streams.audio[:1])}
    if names & MKV_ONLY_CODECS:
        return "matroska"
    return "mp4" if names & MP4_ONLY_CODECS else "mov"


_FORMAT_SUFFIX = dict(mov=".mov", mp4=".mp4", matroska=".mkv")


def work_suffix(path: str) -> str:
    """File suffix for a work copy or an intermediate join of `path`'s streams (see `work_format`)."""
    return _FORMAT_SUFFIX[work_format(path)]


def _is_flush(p) -> bool:
    """The empty packet a demuxer ends with. A real packet can also lack a decode time (Matroska B-frame streams)."""
    return p.dts is None and p.pts is None


def _fill_dts(dts: list[int | None], step: int) -> list[int]:
    """Decode times for every packet of a stream: the ones the demuxer gave, and for the packets without one (the first few of a
    Matroska stream with B-frames) times one `step` apart that continue the sequence backwards from the first known one."""
    known = next((i for i, d in enumerate(dts) if d is not None), None)
    if known is None:
        return list(range(0, step * len(dts), step))
    out: list[int] = [0] * len(dts)
    for i in range(known - 1, -1, -1):
        out[i] = (dts[known] if i == known - 1 else out[i + 1]) - step
    for i in range(known, len(dts)):
        out[i] = dts[i] if dts[i] is not None else out[i - 1] + step
    return out


def strip_outside(src: str, dst: str, fmt: str | None = None) -> int:
    """Copy `src` to `dst` with only what the clip shows. Dropped: the discard-flagged leading pictures an edit list
    hides; every video packet decoded after the last picture that lies inside the clip's video (pictures past the
    end that are decoded earlier stay: they can be references for the last pictures shown, and the next clip is
    placed after them); and the audio packets that end after the last kept picture ends (they would push the next clip away from it). For HEVC the
    clip's first picture, a CRA, becomes a BLA (see `_cra_to_bla`). Returns how many pictures were dropped.
    Packets are copied as they are; nothing is decoded or re-encoded. `fmt` is the container of `dst` (default: `work_format(src)`)."""
    import av

    dropped = 0
    with av.open(src) as c:
        vs = c.streams.video[0]
        tb = float(vs.time_base)
        fps = float(vs.average_rate or 30)
        end = None
        if vs.duration is not None and vs.start_time is not None:
            end = (vs.start_time + vs.duration) * tb - 1e-4
        # pass 1, video only: where the kept pictures stop and when the last one ends. Packets come in decode order and a
        # B-frame shown inside the clip can be decoded after a reference picture that lies past the clip's end, so the
        # cut goes after the LAST packet whose picture is inside the clip; references before it stay (they are the
        # extra picture or two the cut itself leaves after the edit list's end)
        flags = []  # per video packet in decode order: (discard, pts, pts + duration) in seconds
        dts_in: list[int | None] = []  # the same packets' decode times as the demuxer gave them (Matroska gives none to the first few B-frame packets)
        step = round(1 / fps / tb)
        for p in c.demux(vs):
            if _is_flush(p):
                continue
            dts_in.append(p.dts)
            pts = None if p.pts is None else p.pts * tb
            flags.append((p.is_discard, pts, None if pts is None else (p.pts + (p.duration or round(1 / fps / tb))) * tb))
        last_in = max((i for i, (dis, pts, _) in enumerate(flags) if not dis and pts is not None and (end is None or pts < end)), default=None)
        dts_fill = _fill_dts(dts_in, step)
        cut_at = None if end is None or last_in is None or last_in + 1 >= len(flags) else last_in + 1
        v_end = max((e for dis, pts, e in flags[:cut_at] if not dis and pts is not None), default=0.0)
        length_size = 0
        if vs.codec_context.name == "hevc" and vs.codec_context.extradata and vs.codec_context.extradata[0] == 1:
            length_size = (vs.codec_context.extradata[21] & 3) + 1  # hvcC: lengthSizeMinusOne
    with av.open(src) as c, av.open(dst, "w", format=fmt or work_format(src)) as o:
        vs = c.streams.video[0]
        ins = [vs, *c.streams.audio[:1]]
        # a stream decoded by an external library ('libdav1d' for AV1) has no encoder of that name; copy its parameters as they are
        outs = {s.index: o.add_stream_from_template(s, opaque=True if s.codec_context.codec.name != s.codec_context.codec.canonical_name else None) for s in ins}
        if vs.codec_context.name == "hevc":
            # the template drops the codec tag and the muxer then writes 'hev1' (parameter sets in the stream), which
            # AVFoundation (QuickTime, Photos) refuses; the clips are 'hvc1' (parameter sets in the sample entry)
            outs[vs.index].codec_context.codec_tag = "hvc1"
        first, vi = True, 0
        for p in c.demux(*ins):
            if _is_flush(p):
                continue
            kind, sidx = p.stream.type, p.stream.index
            if kind == "video":
                i, vi = vi, vi + 1
                if p.dts is None:
                    p.dts = dts_fill[i]
                if p.is_discard or (cut_at is not None and i >= cut_at):
                    dropped += 1
                    continue
                if first and length_size:
                    q = av.Packet(_cra_to_bla(bytes(p), length_size))
                    q.pts, q.dts, q.duration, q.time_base, q.is_keyframe = p.pts, p.dts, p.duration, p.time_base, p.is_keyframe
                    p = q
                first = False
            elif p.pts is not None and (p.pts + (p.duration or 0)) * float(p.time_base) > v_end + 1e-6:
                continue  # audio that runs past the last picture would push the next clip's pictures away
            p.stream = outs[sidx]
            o.mux(p)
    return dropped


WORK_PREFIX = ".inframe_reel_"
STALE_WORK_S = 24 * 3600  # a work folder this old belongs to a run that was killed


def make_work_dir(near: Path, prefix: str = WORK_PREFIX) -> Path:
    """A folder for the full copies of the clips, beside the output (not in the system temp folder, where a killed run
    would leave footage copies nobody looks at). Folders of earlier killed runs (older than a day) are removed first."""
    near.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for d in near.glob(prefix + "*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > STALE_WORK_S:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix=prefix, dir=near))


def build_reel(clips: list[str], out: str | Path) -> dict:
    """Join `clips` (paths, in the order given) into `out` with no re-encode. Returns
    {"file", "ok", "n_clips", "dropped_pictures", "fallback_used", "check": ReelCheck dict}.
    A reel that fails the check is kept and reported."""
    if not clips:
        raise ValueError("no clips to join")
    out = Path(out)
    part = out.with_name(out.stem + ".part" + out.suffix)
    work = make_work_dir(out.parent)
    try:
        stripped, dropped = [], 0
        for i, c in enumerate(clips):
            fmt = work_format(str(c))  # once per clip: each call opens the file with PyAV
            d = work / f"{i:04d}{_FORMAT_SUFFIX[fmt]}"
            dropped += strip_outside(str(c), str(d), fmt)
            stripped.append(str(d))
        lst = work / "list.txt"
        lst.write_text("".join(_list_line(Path(c), _join_duration(c), base=work) for c in stripped))
        fallback = False
        for make_zero in (False, True):
            r = subprocess.run(concat_command(str(lst), str(part), make_zero), capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg reel failed: {r.stderr.strip()[-300:]}")
            check = check_reel(str(part), stripped, [str(c) for c in clips])
            fallback = make_zero
            if check.ok:
                break
        part.replace(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        part.unlink(missing_ok=True)
    return dict(file=out.name, ok=check.ok, n_clips=len(clips), dropped_pictures=dropped,
                fallback_used=fallback, check=check.to_dict())
