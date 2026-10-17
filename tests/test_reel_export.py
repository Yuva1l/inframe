import json
import subprocess
from pathlib import Path

import pytest

from inframe import pipeline
from inframe.cli import main
from inframe.cut import cut_clip
from inframe.detect import Detection
from inframe.editlist import (EDL_MAX_FPS, FORMATS, check_edit_list, cuts, frame_rate, parse_formats, read_back,
                              write_edit_lists)
from inframe.pipeline import Settings, run_cut
from inframe.probe import probe
from inframe.reel import _join_duration, _list_line, build_reel, check_reel, codec_tag, packets, shown_pictures, strip_outside


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


@pytest.fixture(scope="module")
def long_src(tmp_path_factory):
    """12 s, 30 fps, 320x180 h264 with B-frames and audio, a keyframe every 15 frames, display rotation -90."""
    d = tmp_path_factory.mktemp("longsrc")
    plain, rot = d / "plain.mp4", d / "long.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=12", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-c:v", "libx264", "-g", "15", "-keyint_min", "15", "-sc_threshold", "0", "-bf", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(plain))
    _ff("-display_rotation", "-90", "-i", str(plain), "-c", "copy", str(rot))
    return str(rot)


@pytest.fixture(scope="module")
def open_gop_src(tmp_path_factory):
    """10 s HEVC with open GOPs (CRA with leading pictures every 30 frames), like an iPhone."""
    d = tmp_path_factory.mktemp("opengop")
    p = d / "og.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=10", "-f", "lavfi", "-i", "sine=frequency=330:duration=10",
        "-c:v", "libx265", "-x265-params", "open-gop=1:keyint=30:min-keyint=30:bframes=4:scenecut=0:log-level=error",
        "-tag:v", "hvc1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p))
    return str(p)


def _cut(src, tmp_path, spans):
    outs = []
    for i, (s, e) in enumerate(spans):
        o = tmp_path / f"c{i}.mov"
        cut_clip(src, o, s, e)
        outs.append(str(o))
    return outs


def _decode_errors(path):
    r = subprocess.run(["ffmpeg", "-v", "warning", "-nostdin", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    return [ln for ln in r.stderr.splitlines() if "monotonically" not in ln]  # the null muxer's own dts remark


# ---- reel ----

def test_reel_joins_are_clean_h264(long_src, tmp_path):
    clips = _cut(long_src, tmp_path, [(0.7, 3.2), (5.1, 7.4), (9.0, 11.5)])
    r = build_reel(clips, tmp_path / "reel.mov")
    chk = r["check"]
    assert r["ok"] and chk["ok"], chk["problems"]
    assert len(chk["joins"]) == 3
    assert chk["max_av_offset_ms"] <= 21.0 + 0.1
    assert all(abs(j["av_offset_ms"]) <= 21.1 for j in chk["joins"])
    assert all(0 < j["step_ms"] <= 50 for j in chk["joins"][1:])  # at most 1.5 frames from the last picture of the previous clip
    info = probe(str(tmp_path / "reel.mov"))
    assert info.rotation == -90 and info.has_audio
    # every picture the clips show is in the reel (test_reel_holds_every_picture_the_clips_show counts them), plus at
    # most the reference picture or two a clip keeps past its end
    total = sum(probe(c).duration for c in clips)
    assert total - 0.06 <= info.duration <= total + len(clips) * 2 / 30 + 0.06
    assert _decode_errors(str(tmp_path / "reel.mov")) == []


def test_reel_open_gop_hevc_drops_leading_pictures_and_decodes_clean(open_gop_src, tmp_path):
    clips = _cut(open_gop_src, tmp_path, [(0.4, 3.1), (4.2, 6.6), (7.3, 9.4)])
    flagged = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=flags", "-of", "csv=p=0", clips[1]],
                             capture_output=True, text=True).stdout
    assert "D" in flagged  # the clip really has hidden leading pictures, or this test proves nothing
    r = build_reel(clips, tmp_path / "reel.mov")
    assert r["ok"], r["check"]["problems"]
    assert r["dropped_pictures"] > 0 and not r["fallback_used"]
    assert _decode_errors(str(tmp_path / "reel.mov")) == []
    p = packets(str(tmp_path / "reel.mov"))
    dts = [d for _, d in p["v"]]
    assert all(b > a for a, b in zip(dts, dts[1:]))


def test_check_reel_catches_a_naive_concat_of_open_gop_clips(open_gop_src, tmp_path):
    clips = _cut(open_gop_src, tmp_path, [(0.4, 3.1), (4.2, 6.6), (7.3, 9.4)])
    lst = tmp_path / "l.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    _ff("-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", str(tmp_path / "naive.mov"))
    naive = check_reel(str(tmp_path / "naive.mov"), clips)
    assert not naive.ok and any("picture" in p for p in naive.problems)  # the hidden leading pictures land on top of the previous clip


def _decoded_pts(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True)
    return [float(x) for x in r.stdout.split() if x not in ("N/A",)]


def _shown_decoded(path):
    """Pictures the clip shows, counted by decoding it: those that start before the end of its video stream."""
    st = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time,duration",
                                    "-of", "json", path], capture_output=True, text=True).stdout)["streams"][0]
    end = float(st["start_time"]) + float(st["duration"]) - 1e-4
    return sum(1 for t in _decoded_pts(path) if t < end)


def test_hevc_reel_keeps_the_hvc1_tag(open_gop_src, tmp_path):
    clips = _cut(open_gop_src, tmp_path, [(0.4, 3.1), (4.2, 6.6)])
    assert {codec_tag(c) for c in clips} == {"hvc1"}
    r = build_reel(clips, tmp_path / "reel.mov")
    assert r["ok"], r["check"]["problems"]
    assert codec_tag(str(tmp_path / "reel.mov")) == "hvc1"  # hev1 is refused by QuickTime, Photos and FilmCraft on macOS


def test_h264_reel_keeps_its_tag(long_src, tmp_path):
    clips = _cut(long_src, tmp_path, [(0.7, 3.2), (5.1, 7.4)])
    build_reel(clips, tmp_path / "reel.mov")
    assert codec_tag(str(tmp_path / "reel.mov")) == codec_tag(clips[0]) == "avc1"


@pytest.mark.parametrize("src_fixture, spans", [("long_src", [(0.7, 3.2), (5.1, 7.4), (9.0, 11.5)]),
                                                ("open_gop_src", [(0.4, 3.1), (4.2, 6.6), (7.3, 9.4)])])
def test_reel_holds_every_picture_the_clips_show(src_fixture, spans, request, tmp_path):
    """B-frames decoded after a reference that lies past the clip's end are still pictures of the clip."""
    clips = _cut(request.getfixturevalue(src_fixture), tmp_path, spans)
    r = build_reel(clips, tmp_path / "reel.mov")
    assert r["ok"] and not r["fallback_used"], r["check"]["problems"]
    expected = sum(_shown_decoded(c) for c in clips)
    assert sum(shown_pictures(c) for c in clips) == expected  # the check's own count agrees with a decode
    assert len(_decoded_pts(str(tmp_path / "reel.mov"))) >= expected
    assert _decode_errors(str(tmp_path / "reel.mov")) == []


def _stripped(clips, tmp_path):
    out = []
    for i, c in enumerate(clips):
        d = tmp_path / f"s{i}.mov"
        strip_outside(c, str(d))
        out.append(str(d))
    return out


def _concat(files, out):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(_list_line(Path(f), _join_duration(f)) for f in files))
    _ff("-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", str(out))
    return str(out)


def test_check_reel_against_the_original_clips_catches_a_wrong_codec_tag(open_gop_src, tmp_path):
    clips = _cut(open_gop_src, tmp_path, [(0.4, 3.1), (4.2, 6.6)])
    stripped = _stripped(clips, tmp_path)
    good = _concat(stripped, tmp_path / "good.mov")
    assert check_reel(good, stripped, sources=clips).ok
    bad = tmp_path / "bad.mov"
    _ff("-i", good, "-map", "0", "-c", "copy", "-tag:v", "hev1", str(bad))
    chk = check_reel(str(bad), stripped, sources=clips)
    assert codec_tag(str(bad)) == "hev1"
    assert not chk.ok and any("hev1" in p for p in chk.problems)
    assert check_reel(str(bad), stripped).ok is True  # against the copies alone the tag could not be seen


def test_check_reel_against_the_original_clips_catches_missing_pictures(long_src, tmp_path):
    clips = _cut(long_src, tmp_path, [(0.7, 3.2), (5.1, 7.4)])
    stripped = _stripped(clips, tmp_path)
    short = []
    for i, f in enumerate(stripped):  # a copy that lost its last two pictures
        n = len(packets(f)["v"]) - 2
        d = tmp_path / f"short{i}.mov"
        _ff("-i", f, "-map", "0", "-c", "copy", "-frames:v", str(n), str(d))
        short.append(str(d))
    reel = _concat(short, tmp_path / "short_reel.mov")
    assert check_reel(reel, short).ok  # the copies are all the check used to look at
    chk = check_reel(reel, short, sources=clips)
    assert not chk.ok and any("pictures of the" in p for p in chk.problems)


def test_build_reel_needs_clips(tmp_path):
    with pytest.raises(ValueError):
        build_reel([], tmp_path / "r.mov")


# ---- edit lists ----

def _doc(fps=30.0, fstart=0.0, n=3, has_audio=True, path="/videos/Take 1.mov"):
    clips = [dict(index=i + 1, start=round(2.0 + i * 10.37, 3), end=round(7.5 + i * 10.37, 3), file=f"Take 1_{i + 1:02d}_x.mov") for i in range(n)]
    return dict(inframe_version="t", mode="detect", clips=clips, presence_intervals=[],
                source=dict(path=path, width=1280, height=720, display_width=720, display_height=1280, fps=fps, rotation=-90,
                            duration=120.0, start_time=fstart, format_start_time=fstart, has_audio=has_audio))


@pytest.mark.parametrize("fps", [24.0, 25.0, 29.97, 30.0, 59.94, 60.0])
def test_all_edit_lists_read_back_within_one_frame(fps, tmp_path):
    doc = _doc(fps)
    res, warn = write_edit_lists(doc, tmp_path, list(FORMATS))
    assert [r["format"] for r in res] == list(FORMATS), (res, warn)
    for r in res:
        assert r["ok"], r
        assert r["max_err_frames"] <= 1.0
        assert (tmp_path / r["file"]).is_file()
    assert warn == [] or all("not at that path" in w for w in warn)


def test_edl_skipped_above_60_fps_others_written(tmp_path):
    doc = _doc(120.0)
    res, warn = write_edit_lists(doc, tmp_path, list(FORMATS))
    by = {r["format"]: r for r in res}
    assert by["edl"]["skipped"] and by["edl"]["file"] is None and not (tmp_path / "Take 1.edl").exists()
    assert any("edl skipped" in w and "60" in w for w in warn)
    assert by["fcp7"]["skipped"] and not (tmp_path / "Take 1_fcp7.xml").exists()  # the FCP7 writer has no timecode at 120 fps
    assert all(by[f]["ok"] for f in ("fcpxml", "kdenlive", "otio"))
    assert EDL_MAX_FPS == 60.0


def test_edl_and_fcp7_skipped_at_rates_without_timecode(tmp_path):
    doc = _doc(10.0)  # not an SMPTE rate
    res, warn = write_edit_lists(doc, tmp_path, list(FORMATS))
    by = {r["format"]: r for r in res}
    assert by["edl"]["skipped"] and by["fcp7"]["skipped"] and len(warn) >= 2
    assert all(by[f]["ok"] for f in ("fcpxml", "kdenlive", "otio"))


def test_edit_list_in_points_include_container_start(tmp_path):
    doc = _doc(30.0, fstart=1.5)
    c0 = cuts(doc)[0]
    assert c0["in_frame"] == round((2.0 + 1.5) * 30)  # source container time plus start_time
    res, _ = write_edit_lists(doc, tmp_path, ["fcpxml", "fcp7", "kdenlive", "edl", "otio"])
    assert all(r["ok"] for r in res), res
    fx = (tmp_path / "Take 1.fcpxml").read_text()
    assert 'start="7/2s"' in fx  # 3.5 s into the container
    assert read_back(tmp_path / "Take 1.otio", "otio", doc)[0] == pytest.approx((2.0, 7.5), abs=1 / 30)


def test_fcpxml_structure(tmp_path):
    import xml.etree.ElementTree as ET

    doc = _doc(29.97)
    write_edit_lists(doc, tmp_path, ["fcpxml"])
    root = ET.parse(tmp_path / "Take 1.fcpxml").getroot()
    assert root.tag == "fcpxml" and root.get("version") == "1.8"
    fmt = root.find("resources/format")
    assert fmt.get("frameDuration") == "1001/30000s" and (fmt.get("width"), fmt.get("height")) == ("720", "1280")
    asset = root.find("resources/asset")
    assert asset.get("src").startswith("file:///") and "Take%201" in asset.get("src") and asset.get("hasAudio") == "1"
    spine = root.find("library/event/project/sequence/spine")
    ac = spine.findall("asset-clip")
    assert len(ac) == 3 and all(a.get("ref") == asset.get("id") for a in ac)
    # back to back on the timeline
    from fractions import Fraction

    def t(v):
        v = v.rstrip("s")
        return Fraction(v)
    at = Fraction(0)
    for a in ac:
        assert t(a.get("offset")) == at
        at += t(a.get("duration"))
    assert t(root.find("library/event/project/sequence").get("duration")) == at


def test_check_edit_list_fails_when_a_cut_is_wrong(tmp_path):
    doc = _doc(30.0)
    write_edit_lists(doc, tmp_path, ["otio"])
    assert check_edit_list(tmp_path / "Take 1.otio", "otio", doc)["ok"]
    doc["clips"][1]["start"] += 0.2  # six frames away from what was written
    r = check_edit_list(tmp_path / "Take 1.otio", "otio", doc)
    assert not r["ok"] and r["max_err_frames"] > 5
    doc["clips"].pop()
    assert "cuts read back" in check_edit_list(tmp_path / "Take 1.otio", "otio", doc)["problem"]


def test_parse_formats():
    assert parse_formats(None) == [] and parse_formats("") == []
    assert parse_formats("edl,fcpxml") == ["fcpxml", "edl"]
    assert parse_formats("all") == list(FORMATS)
    with pytest.raises(ValueError, match="premiere"):
        parse_formats("fcpxml,premiere")


def test_frame_rate_snaps_ntsc():
    from fractions import Fraction
    assert frame_rate(29.97) == Fraction(30000, 1001) and frame_rate(30.0) == 30 and frame_rate(59.94) == Fraction(60000, 1001)
    assert frame_rate(119.88) == Fraction(120000, 1001) and frame_rate(25.0) == 25


def test_failed_clips_are_left_out(tmp_path):
    doc = _doc(30.0)
    doc["clips"][1]["error"] = "boom"
    res, _ = write_edit_lists(doc, tmp_path, ["otio", "fcpxml"])
    assert all(r["ok"] for r in res)
    assert len(read_back(tmp_path / "Take 1.otio", "otio", doc)) == 2


# ---- pipeline and CLI ----

class _Fake:
    backend, path = "fake", "fake.mlpackage"

    def __init__(self):
        self.n = 0

    def detect(self, frames):
        out = []
        for _ in frames:  # 5 samples per second: person at 1-3 s and 6-9 s
            t = self.n / 5
            out.append([Detection(0.9, (0.1, 0.1, 0.3, 0.5))] if 1 <= t < 3 or 6 <= t < 9 else [])
            self.n += 1
        return out


def test_cut_with_reel_and_edit_lists_then_stale_files_go(long_src, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: _Fake())
    out = tmp_path / "o"
    doc = run_cut(long_src, str(out), Settings(pad=0.5, min_len=0.5, reel=True, edit_lists=("all",)))
    assert len(doc["clips"]) == 2
    assert doc["reel"]["ok"] and doc["reel"]["file"] == "long_reel.mp4" and (out / "long_reel.mp4").is_file()
    assert [r["format"] for r in doc["edit_lists"]] == list(FORMATS) and all(r["ok"] for r in doc["edit_lists"])
    saved = json.loads((out / "segments.json").read_text())
    assert saved["reel"]["check"]["ok"] and saved["settings"]["reel"] is True
    for name in ("long.fcpxml", "long_fcp7.xml", "long.kdenlive", "long.edl", "long.otio"):
        assert (out / name).is_file()
    # reel duration is the two clips together
    assert probe(str(out / "long_reel.mp4")).duration == pytest.approx(
        sum(probe(str(out / c["file"])).duration for c in doc["clips"]) - doc["reel"]["dropped_pictures"] / 30, abs=0.06)
    # the next run without the options leaves no reel or edit list from this one behind
    monkeypatch.setattr(pipeline, "load_detector", lambda size: _Fake())
    doc2 = run_cut(long_src, str(out), Settings(pad=0.5, min_len=0.5))
    assert "reel" not in doc2 and sorted(p.name for p in out.glob("long*") if "reel" in p.name or p.suffix in (".fcpxml", ".kdenlive", ".edl", ".otio")) == []
    assert not (out / "long_fcp7.xml").exists()


def test_unknown_edit_list_format_is_refused_before_any_work(long_src, tmp_path, capsys):
    assert main(["cut", long_src, "--edit-list", "premiere", "--out", str(tmp_path / "o")]) == 2
    assert "unknown edit list format" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_export_command_on_a_finished_run(long_src, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: _Fake())
    run = tmp_path / "run"
    run_cut(long_src, str(run), Settings(pad=0.5, min_len=0.5))
    assert main(["export", str(run)]) == 2  # nothing asked for
    other = tmp_path / "ex"
    assert main(["export", str(run / "segments.json"), "--reel", "--edit-list", "fcpxml,edl", "--out", str(other)]) == 0
    text = capsys.readouterr().out
    assert "reel" in text and "fcpxml" in text and "edl" in text and "FAILED" not in text
    assert (other / "long_reel.mp4").is_file() and (other / "long.fcpxml").is_file() and (other / "export.json").is_file()
    assert not (run / "long_reel.mp4").exists()  # --out kept the run folder untouched
    ex = json.loads((other / "export.json").read_text())
    assert ex["reel"]["ok"] and [r["format"] for r in ex["edit_lists"]] == ["fcpxml", "edl"]


def test_export_command_rejects_missing_and_foreign_json(tmp_path, capsys):
    assert main(["export", str(tmp_path / "nope.json"), "--reel"]) == 2
    f = tmp_path / "x.json"
    f.write_text("{}")
    assert main(["export", str(f), "--reel"]) == 2
    assert "not a segments.json" in capsys.readouterr().err


# ---- low findings ----

def _pk(v0, a0, n=3):
    return dict(v=[(v0 + i / 30, v0 + i / 30) for i in range(n)], a=[(a0 + i * 0.023, a0 + i * 0.023) for i in range(n)], fps=30.0, duration=1.0)


def _fake_reel(monkeypatch, own, in_reel):
    """Two clips whose audio leads the picture by `own` s on its own; in the reel the second one's audio leads by `in_reel[1]`."""
    reel = dict(_pk(0.0, 0.0), v=_pk(0.0, 0)["v"] + _pk(0.1, 0)["v"], a=_pk(0.0, in_reel[0])["a"] + _pk(0.1, 0.1 + in_reel[1])["a"], duration=2.0)
    table = dict(c1=_pk(0.0, own[0]), c2=_pk(1.0, 1.0 + own[1]), reel=reel)
    monkeypatch.setattr("inframe.reel.packets", lambda p: table[p])


def test_av_join_limit_does_not_count_the_clips_own_audio_lead(monkeypatch):
    # each clip has its audio 30 ms ahead of its picture by itself (the cut's business); the reel keeps that, the join adds nothing
    _fake_reel(monkeypatch, (0.03, 0.03), (0.03, 0.03))
    ck = check_reel("reel", ["c1", "c2"])
    assert ck.ok, ck.problems
    assert ck.max_av_offset_ms == pytest.approx(0.0, abs=0.1) and ck.joins[1]["clip_av_offset_ms"] == pytest.approx(30.0, abs=0.1)


def test_av_join_limit_still_fails_when_the_join_moves_the_audio(monkeypatch):
    _fake_reel(monkeypatch, (0.01, 0.01), (0.01, 0.04))
    ck = check_reel("reel", ["c1", "c2"])
    assert not ck.ok and any("clip 2" in p and "audio starts" in p for p in ck.problems)


def test_reel_work_folder_is_beside_the_output_and_gone_afterwards(long_src, tmp_path, monkeypatch):
    import tempfile as tf

    seen = []
    real = tf.mkdtemp
    monkeypatch.setattr(tf, "mkdtemp", lambda *a, **k: seen.append(k.get("dir")) or real(*a, **k))
    out = tmp_path / "o"
    build_reel(_cut(long_src, tmp_path, [(1.0, 3.0), (5.0, 7.0)]), out / "r.mov")
    assert [Path(d) for d in seen] == [out] and sorted(p.name for p in out.iterdir()) == ["r.mov"]


def test_stale_work_folders_of_killed_runs_are_swept(tmp_path):
    import os
    import time

    from inframe.reel import make_work_dir

    old, fresh = tmp_path / ".inframe_reel_old", tmp_path / ".inframe_reel_fresh"
    old.mkdir()
    fresh.mkdir()
    (old / "0000.mov").write_bytes(b"x")
    t = time.time() - 3 * 86400
    os.utime(old, (t, t))
    w = make_work_dir(tmp_path)
    assert not old.exists() and fresh.exists() and w.is_dir()


def test_export_reel_with_no_clips_fails(tmp_path):
    from inframe.export import export_run

    ex = export_run(_doc(30.0, n=0), tmp_path, tmp_path / "o", True, [])
    assert not ex["ok"] and ex["reel"]["file"] is None and "no clips" in ex["reel"]["problem"]


def test_export_command_reel_with_no_clips_exits_nonzero(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "segments.json").write_text(json.dumps(_doc(30.0, n=0)))
    assert main(["export", str(run), "--reel"]) == 1
    assert "reel not written" in capsys.readouterr().out


class _Never(_Fake):
    def detect(self, frames):
        self.n += len(frames)
        return [[] for _ in frames]


def test_cut_reel_with_zero_clips_is_not_a_failure(long_src, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: _Never())
    out = tmp_path / "o"
    assert main(["cut", long_src, "--reel", "--out", str(out)]) == 0
    doc = json.loads((out / "segments.json").read_text())
    assert doc["clips"] == [] and not (doc.get("reel") or {}).get("problem")
    assert any("no clips" in w for w in doc["warnings"])
    assert not list(out.glob("*_reel*"))


def test_sidecar_without_a_license_field_writes_no_credits(tmp_path):
    from inframe.export import write_credits

    w: list[str] = []
    assert write_credits(tmp_path, "src", dict(title="Song", creator="Me"), w) is None
    assert not (tmp_path / "src_credits.txt").exists() and any("no license field" in x for x in w)
    assert write_credits(tmp_path, "src", dict(title="Song", creator="Me", license="by"), []) == "src_credits.txt"


# ---------------------------------------------------------------- VP9/Opus .webm clips (a .mov cannot hold them)

@pytest.fixture(scope="module")
def vp9_clips(tmp_path_factory):
    """Two clips cut from a synthetic 8 s VP9 + Opus .webm (keyframe every 15 frames)."""
    d = tmp_path_factory.mktemp("vp9")
    src = d / "v.webm"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8", "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-c:v", "libvpx-vp9", "-b:v", "300k", "-deadline", "realtime", "-cpu-used", "8", "-g", "15", "-c:a", "libopus", "-shortest", str(src))
    outs = []
    for i, (s, e) in enumerate([(0.5, 2.5), (4.0, 6.0)]):
        o = d / f"c{i}.webm"
        cut_clip(str(src), o, s, e)
        outs.append(str(o))
    return outs


def test_reel_of_vp9_webm_clips(vp9_clips, tmp_path):
    """The lossless reel of VP9 + Opus clips: the work copies need a container that holds them (a .mov refuses VP9 and Opus)."""
    r = build_reel(vp9_clips, tmp_path / "v_reel.webm")
    assert r["ok"], r["check"]["problems"]
    assert (tmp_path / "v_reel.webm").is_file() and r["n_clips"] == 2
    assert not list(tmp_path.glob(".inframe_reel_*"))


def test_reel_of_vp9_webm_clips_without_audio(tmp_path):
    src = tmp_path / "v.webm"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=6", "-c:v", "libvpx-vp9", "-b:v", "300k", "-deadline", "realtime",
        "-cpu-used", "8", "-g", "15", "-an", str(src))
    outs = []
    for i, (s, e) in enumerate([(0.5, 2.5), (3.0, 5.0)]):
        o = tmp_path / f"c{i}.webm"
        cut_clip(str(src), o, s, e)
        outs.append(str(o))
    r = build_reel(outs, tmp_path / "n_reel.webm")
    assert r["ok"], r["check"]["problems"]


def test_list_line_refuses_a_line_break_and_uses_the_bare_name_beside_the_list(tmp_path):
    with pytest.raises(ValueError, match="line break"):
        _list_line(tmp_path / "a\nfile '/etc/passwd'\n.mov")
    with pytest.raises(ValueError, match="line break"):
        _list_line(tmp_path / "a\rb.mov")
    d = tmp_path / "dir\nwith break"
    d.mkdir()
    assert _list_line(d / "0000.mov", 1.5, base=d) == "file '0000.mov'\nduration 1.500000\n"


def test_reel_into_a_folder_with_a_line_break_in_its_name(long_src, tmp_path):
    clips = _cut(long_src, tmp_path, [(0.7, 3.2), (5.1, 7.4)])
    out = tmp_path / "folder\nname" / "reel.mov"
    r = build_reel(clips, out)
    assert r["ok"], r["check"]["problems"]
    assert out.is_file() and probe(str(out)).duration > 4
