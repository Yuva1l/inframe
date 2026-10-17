"""Edit lists: the clips of a run as cuts of the original video, for an editor to open.

Formats (all but FCPXML written through OpenTimelineIO, Apache-2.0, and its adapter packages):
- fcpxml: Final Cut Pro X, FilmCraft, DaVinci Resolve. Written by us (see `write_fcpxml`): the OTIO FCPXML
  adapter's output has asset-clip and project elements in places the FCPXML format does not allow.
- fcp7: Final Cut Pro 7 XML, which Premiere imports (otio-fcp-adapter, Apache-2.0).
- kdenlive: Kdenlive project (otio-kdenlive-adapter, MIT; marked deprecated by its authors, because Kdenlive 25.04 and
  later open the `otio` file directly).
- edl: CMX3600 EDL (otio-cmx3600-adapter, Apache-2.0). Only at 60 fps or below: higher rates have no standard timecode.
- otio: OpenTimelineIO JSON.
Sound markers (`--markers clap`, `clips[].markers` in segments.json) are written as FCPXML `marker` elements and as OTIO markers (the
OTIO-based formats keep them as far as their adapter does; the cut read-back check is unchanged).

In and out points are in the original's container time: the clip's start and end in segments.json (seconds from the
container start) plus the container's start_time, rounded to whole frames of the source rate. Every file is read
back with OTIO (the Kdenlive and FCPXML files with their readers too) and compared with segments.json; each cut
has to land within one frame of it."""
from __future__ import annotations

import contextlib
import io
import warnings
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any

from .cut import clip_name
from .names import safe_stem

FORMATS = ("fcpxml", "fcp7", "kdenlive", "edl", "otio")
EDL_MAX_FPS = 60.0
SMPTE_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)  # the rates the FCP7 adapter can write timecode for
_EXT = {"fcpxml": ".fcpxml", "fcp7": "_fcp7.xml", "kdenlive": ".kdenlive", "edl": ".edl", "otio": ".otio"}
_ADAPTER = {"fcp7": "fcp_xml", "kdenlive": "kdenlive", "edl": "cmx_3600", "otio": "otio_json", "fcpxml": "fcpx_xml"}


def edit_list_name(stem: str, fmt: str) -> str:
    return safe_stem(stem) + _EXT[fmt]


def parse_formats(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """'fcpxml,edl', ['all'] or None -> a list of format names in a fixed order. Raises ValueError on an unknown one."""
    if not value:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    items = [i.strip().lower() for i in items if i.strip()]
    if "all" in items:
        return list(FORMATS)
    bad = [i for i in items if i not in FORMATS]
    if bad:
        raise ValueError(f"unknown edit list format {bad[0]!r}; choose from {', '.join(FORMATS)} or all")
    return [f for f in FORMATS if f in items]


def frame_rate(fps: float) -> Fraction:
    """The source rate as an exact fraction: 29.97 is 30000/1001, and so on."""
    for n in (24, 30, 60, 120, 240):
        ntsc = Fraction(n * 1000, 1001)
        if abs(fps - float(ntsc)) < 0.01 * n / 24:
            return ntsc
    r = Fraction(fps).limit_denominator(1000)
    return r


def _fmt_time(frames: int, rate: Fraction) -> str:
    """FCPXML time: frames / rate as a reduced fraction of seconds."""
    t = Fraction(frames) / rate
    if t == 0:
        return "0s"
    return f"{t.numerator}s" if t.denominator == 1 else f"{t.numerator}/{t.denominator}s"


def cuts(doc: dict) -> list[dict]:
    """The clips of a run as frame counts: [{name, start_s, end_s, in_frame, frames}]. `in_frame` counts from the
    container's time zero (container start_time included)."""
    src = doc["source"]
    rate = frame_rate(src["fps"])
    fstart = src.get("format_start_time", 0.0) or 0.0
    out = []
    for c in doc["clips"]:
        if "error" in c:
            continue
        a = round((c["start"] + fstart) * rate)
        b = round((c["end"] + fstart) * rate)
        if b <= a:
            b = a + 1
        marks = [(round((m["time"] + fstart) * rate), m.get("kind", "clap")) for m in c.get("markers", []) if c["start"] <= m["time"] <= c["end"]]
        # a clip whose file was not kept (reel-only export) has no "file"; name it as its file would have been
        name = Path(c["file"]).stem if c.get("file") else Path(clip_name(Path(src["path"]).stem, c["index"], c["start"], c["end"])).stem
        out.append(dict(name=name, start_s=c["start"], end_s=c["end"], in_frame=a, frames=b - a, markers=marks))
    return out


def _source_uri(doc: dict) -> str:
    return Path(doc["source"]["path"]).expanduser().resolve().as_uri()


def build_timeline(doc: dict, with_audio: bool = True):
    import opentimelineio as otio
    from opentimelineio import opentime as ot
    from opentimelineio import schema

    src = doc["source"]
    rate = float(frame_rate(src["fps"]))
    stem = Path(src["path"]).stem
    fstart = src.get("format_start_time", 0.0) or 0.0
    avail = ot.TimeRange(ot.RationalTime(round(src.get("start_time", 0.0) * rate), rate),
                         ot.RationalTime(round(src["duration"] * rate), rate))
    tl = schema.Timeline(name=stem, global_start_time=ot.RationalTime(0, rate))
    tl.metadata["inframe"] = dict(version=doc.get("inframe_version"), mode=doc.get("mode"), container_start_time=fstart)
    kinds = [schema.TrackKind.Video] + ([schema.TrackKind.Audio] if with_audio and src.get("has_audio") else [])
    for kind in kinds:
        tr = schema.Track(name="V1" if kind == schema.TrackKind.Video else "A1", kind=kind)
        for k, c in enumerate(cuts(doc), 1):
            ref = schema.ExternalReference(target_url=_source_uri(doc), available_range=avail)
            clip = schema.Clip(
                name=f"{stem} {k:02d}", media_reference=ref,
                source_range=ot.TimeRange(ot.RationalTime(c["in_frame"], rate), ot.RationalTime(c["frames"], rate)))
            if kind == schema.TrackKind.Video:  # sound markers (--markers clap), in the media's time like the source range
                for mf, mk in c["markers"]:
                    clip.markers.append(schema.Marker(name=mk, marked_range=ot.TimeRange(ot.RationalTime(mf, rate), ot.RationalTime(0, rate)),
                                                      color=schema.MarkerColor.RED))
            tr.append(clip)
        tl.tracks.append(tr)
    return tl


def write_fcpxml(doc: dict, path: str | Path) -> None:
    """FCPXML 1.8: one asset (the original), one event with one project whose spine holds an asset-clip per cut.
    Every time is a whole number of frames over the source rate."""
    src = doc["source"]
    rate = frame_rate(src["fps"])
    frame_dur = 1 / rate
    w, h = src.get("display_width", src["width"]), src.get("display_height", src["height"])
    stem = Path(src["path"]).stem
    fstart = src.get("format_start_time", 0.0) or 0.0
    root = ET.Element("fcpxml", version="1.8")
    res = ET.SubElement(root, "resources")
    ET.SubElement(res, "format", id="r1", name=f"FFVideoFormat{w}x{h}p{float(rate):g}", frameDuration=_fmt_time(1, rate),
                  width=str(w), height=str(h))
    total = round(src["duration"] * rate)
    asset = ET.SubElement(res, "asset", id="r2", name=stem, src=_source_uri(doc), start=_fmt_time(round(src.get("start_time", 0.0) * rate), rate),
                          duration=_fmt_time(total, rate), hasVideo="1", format="r1", hasAudio="1" if src.get("has_audio") else "0")
    if src.get("has_audio"):
        asset.set("audioSources", "1")
        asset.set("audioChannels", "2")
    lib = ET.SubElement(root, "library")
    ev = ET.SubElement(lib, "event", name=f"inframe {stem}")
    proj = ET.SubElement(ev, "project", name=stem)
    cl = cuts(doc)
    seq_frames = sum(c["frames"] for c in cl)
    seq = ET.SubElement(proj, "sequence", format="r1", duration=_fmt_time(seq_frames, rate), tcStart="0s", tcFormat="NDF")
    spine = ET.SubElement(seq, "spine")
    at = 0
    for k, c in enumerate(cl, 1):
        ac = ET.SubElement(spine, "asset-clip", ref="r2", name=f"{stem} {k:02d}", offset=_fmt_time(at, rate),
                           start=_fmt_time(c["in_frame"], rate), duration=_fmt_time(c["frames"], rate), format="r1",
                           tcFormat="NDF")
        for mf, mk in c["markers"]:  # sound markers (--markers clap): one frame long, at the asset's own time
            ET.SubElement(ac, "marker", start=_fmt_time(mf, rate), duration=_fmt_time(1, rate), value=mk)
        at += c["frames"]
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    Path(path).write_text('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n\n' + body + "\n", encoding="utf-8")


def _patch_kdenlive_profile(path: Path, doc: dict) -> None:
    """The adapter writes a 1920x1080 project profile. Set the source's displayed size."""
    src = doc["source"]
    w, h = src.get("display_width", src["width"]), src.get("display_height", src["height"])
    tree = ET.parse(path)
    prof = tree.getroot().find("profile")
    if prof is not None:
        fr = frame_rate(src["fps"])
        a = Fraction(w, h).limit_denominator(100)
        prof.set("width", str(w))
        prof.set("height", str(h))
        prof.set("frame_rate_num", str(fr.numerator))
        prof.set("frame_rate_den", str(fr.denominator))
        prof.set("display_aspect_num", str(a.numerator))
        prof.set("display_aspect_den", str(a.denominator))
        prof.set("description", f"{w}x{h} {float(fr):g} fps")
        tree.write(path, encoding="utf-8", xml_declaration=True)


def write_edit_list(doc: dict, fmt: str, path: str | Path) -> None:
    import opentimelineio as otio

    path = Path(path)
    if fmt == "fcpxml":
        write_fcpxml(doc, path)
        return
    tl = build_timeline(doc, with_audio=fmt != "edl")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):  # the Kdenlive adapter prints a deprecation banner
        otio.adapters.write_to_file(tl, str(path), adapter_name=_ADAPTER[fmt])
    if fmt == "kdenlive":
        _patch_kdenlive_profile(path, doc)


def read_back(path: str | Path, fmt: str, doc: dict) -> list[tuple[float, float]]:
    """Read an edit list with OTIO and return the cuts of its first video track as (start_s, end_s) in the
    original's time (container start_time taken off)."""
    import opentimelineio as otio

    src = doc["source"]
    rate = float(frame_rate(src["fps"]))
    fstart = src.get("format_start_time", 0.0) or 0.0
    if fmt == "fcpxml" and abs(rate - round(rate)) > 1e-6:
        # OTIO's FCPXML reader keeps only the whole part of a fractional rate (29.97 reads as 29), so for those
        # rates the times are read straight from the XML, as exact fractions
        return _read_fcpxml_times(path, fstart)
    kwargs: dict[str, Any] = dict(rate=rate) if fmt == "edl" else {}
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        tl = otio.adapters.read_from_file(str(path), adapter_name=_ADAPTER[fmt], **kwargs)
    if not isinstance(tl, otio.schema.Timeline):  # FCPXML reads as events holding projects
        tls = list(tl.find_children(descended_from_type=otio.schema.Timeline))
        if not tls:
            raise ValueError("no timeline in the file")
        tl = tls[0]
    video = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
    if not video:
        raise ValueError("no video track")
    out = []
    for c in video[0].find_clips():
        sr = c.source_range or c.available_range()
        s = sr.start_time.to_seconds() - fstart
        out.append((s, s + sr.duration.to_seconds()))
    return out


def _read_fcpxml_times(path: str | Path, fstart: float) -> list[tuple[float, float]]:
    def secs(v: str) -> Fraction:
        return Fraction(v.rstrip("s"))

    root = ET.parse(path).getroot()
    out = []
    for ac in root.iterfind(".//spine/asset-clip"):
        s = secs(ac.get("start", "0s"))
        out.append((float(s) - fstart, float(s + secs(ac.get("duration"))) - fstart))
    return out


def check_edit_list(path: str | Path, fmt: str, doc: dict) -> dict:
    """Compare what OTIO reads back from `path` with segments.json. ok = same number of cuts, each start and end
    within one frame."""
    expect = [(c["start"], c["end"]) for c in doc["clips"] if "error" not in c]
    fps = float(frame_rate(doc["source"]["fps"]))
    try:
        got = read_back(path, fmt, doc)
    except Exception as e:  # a reader that cannot read our own file is a failed check, not a crash
        return dict(ok=False, max_err_frames=None, problem=f"read-back failed: {type(e).__name__}: {str(e)[:160]}")
    if len(got) != len(expect):
        return dict(ok=False, max_err_frames=None, problem=f"{len(got)} cuts read back, {len(expect)} in segments.json")
    err = max((max(abs(gs - es), abs(ge - ee)) for (gs, ge), (es, ee) in zip(got, expect)), default=0.0) * fps
    return dict(ok=err <= 1.0 + 1e-6, max_err_frames=round(err, 3),
                problem=None if err <= 1.0 + 1e-6 else f"a cut is {err:.2f} frames from segments.json")


def write_edit_lists(doc: dict, out_dir: str | Path, formats: list[str]) -> tuple[list[dict], list[str]]:
    """Write each format next to the clips and check it. Returns (results, warnings); never raises for one
    format's failure."""
    out = Path(out_dir)
    stem = Path(doc["source"]["path"]).stem
    fps = doc["source"]["fps"]
    results, warnings_ = [], []
    if not any("error" not in c for c in doc["clips"]):
        return results, ["no clips: no edit lists written"] if formats else []
    for fmt in formats:
        if fmt == "edl" and fps > EDL_MAX_FPS + 1e-6:
            msg = f"edl skipped: the source is {fps:g} fps and CMX3600 EDL has no standard timecode above {EDL_MAX_FPS:g} fps"
        elif fmt in ("edl", "fcp7") and not any(abs(fps - r) < 0.01 for r in SMPTE_RATES):
            msg = (f"{fmt} skipped: the source is {fps:g} fps and the {fmt} writer has no timecode for that rate "
                   f"(it knows {', '.join(f'{r:g}' for r in SMPTE_RATES)}); use fcpxml, kdenlive or otio")
        else:
            msg = None
        if msg:
            warnings_.append(msg)
            results.append(dict(format=fmt, file=None, ok=None, skipped=msg))
            continue
        path = out / edit_list_name(stem, fmt)
        try:
            write_edit_list(doc, fmt, path)
        except Exception as e:
            msg = f"{fmt} edit list could not be written: {type(e).__name__}: {str(e)[:160]}"
            warnings_.append(msg)
            results.append(dict(format=fmt, file=None, ok=False, problem=msg))
            continue
        chk = check_edit_list(path, fmt, doc)
        results.append(dict(format=fmt, file=path.name, **chk))
        if not chk["ok"]:
            warnings_.append(f"{fmt} edit list failed its read-back check: {chk['problem']}")
    if not Path(doc["source"]["path"]).exists():
        warnings_.append(f"the source video {doc['source']['path']} is not at that path now; the edit lists point to it")
    return results, warnings_


def clear_old_edit_lists(out: Path, stem: str) -> None:
    for fmt in FORMATS:
        (out / edit_list_name(stem, fmt)).unlink(missing_ok=True)
