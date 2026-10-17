"""MCP server: tool functions called directly, plus one client round trip over stdio on a synthetic clip (motion mode, no model)."""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError

from inframe import mcp_server
from inframe.mcp_server import compare_runs, cut_video, export_reel, probe_video

ROOT = Path(__file__).resolve().parents[1]


class FakeCtx:
    def __init__(self):
        self.calls = []

    async def report_progress(self, progress, total=None, message=None):
        self.calls.append((progress, total, message))


def run_cut_tool(path, ctx=None, **kw):
    return asyncio.run(cut_video(path, ctx=ctx, **kw))


def probe_tool(path):
    return asyncio.run(probe_video(path))


# probe_video
def test_probe_video(synth):
    r = probe_tool(synth["rot"])
    assert r["path"] == synth["rot"] and r["codec"] == "h264" and r["rotation"] == -90
    assert (r["display_width"], r["display_height"]) == (180, 320) and r["has_audio"] and abs(r["duration"] - 3.0) < 0.1
    assert "tags" not in r and "keyframes" not in r


@pytest.mark.parametrize("bad, msg", [("", "non-empty"), ("clip.mp4", "absolute"), ("/no/such/video.mov", "no such file")])
def test_probe_video_rejects_bad_paths(bad, msg):
    with pytest.raises(ToolError, match=msg):
        probe_tool(bad)


def test_probe_video_rejects_folder_and_non_video(tmp_path):
    with pytest.raises(ToolError, match="not a file"):
        probe_tool(str(tmp_path))
    junk = tmp_path / "notes.mov"
    junk.write_text("not a video")
    with pytest.raises(ToolError, match="could not read"):
        probe_tool(str(junk))


# cut_video
def test_cut_video_motion_returns_clips_and_progress(static_box, tmp_path):
    ctx = FakeCtx()
    out = tmp_path / "o"
    r = run_cut_tool(static_box, ctx, mode="motion", pad=0.5, min_len=0.5, out=str(out))
    assert r["mode"] == "motion" and r["out_dir"] == str(out) and r["n_clips"] == 1 and len(r["clips"]) == 1
    c = r["clips"][0]
    assert Path(c["file"]).is_file() and Path(c["file"]).parent == out
    assert c["duration"] == pytest.approx(c["end"] - c["start"], abs=1e-3) and r["kept_s"] == pytest.approx(c["duration"], abs=1e-2)
    s, e = r["presence_intervals"][0]
    assert 1.6 < s < 2.4 and 3.6 < e < 4.4
    doc = json.loads(Path(r["segments_json"]).read_text())
    assert doc["clips"][0]["file"] == Path(c["file"]).name and r["segments_json"] == str(out / "segments.json")
    assert Path(r["losslesscut_csv"]).is_file() and Path(r["samples_csv"]).is_file()
    assert r["source"]["duration"] > 7 and "path" not in r["source"]
    assert ctx.calls, "no progress reported"
    fr = [p for p, _, _ in ctx.calls]
    assert fr == sorted(fr) and fr[-1] == 100.0 and all(t == 100 for _, t, _ in ctx.calls)


def test_cut_video_default_output_folder_is_next_to_the_video(static_box, tmp_path):
    video = tmp_path / "my.mp4"
    video.write_bytes(Path(static_box).read_bytes())
    r = run_cut_tool(str(video), mode="motion", pad=0.5, min_len=0.5)
    assert r["out_dir"] == str(tmp_path / "my_inframe" / "motion") and Path(r["segments_json"]).is_file()


@pytest.mark.parametrize("kw, msg", [
    (dict(mode="face"), "mode must be one of"),
    (dict(preset="slow"), "preset must be one of"),
    (dict(pad=-1), "pad must be"),
    (dict(min_len=float("nan")), "min_len must be"),
    (dict(merge_gap=float("inf")), "merge_gap must be"),
    (dict(out="relative/dir"), "absolute"),
])
def test_cut_video_rejects_bad_arguments(static_box, tmp_path, kw, msg):
    kw.setdefault("out", str(tmp_path / "o"))
    with pytest.raises(ToolError, match=msg):
        run_cut_tool(static_box, **kw)
    assert not (tmp_path / "o").exists(), "nothing is written for a refused call"


def test_cut_video_rejects_bad_video_and_out(static_box, tmp_path):
    with pytest.raises(ToolError, match="no such file"):
        run_cut_tool(str(tmp_path / "gone.mov"))
    junk = tmp_path / "x.mov"
    junk.write_text("nope")
    with pytest.raises(ToolError, match="could not read"):
        run_cut_tool(str(junk), mode="motion", out=str(tmp_path / "o"))
    assert not (tmp_path / "o").exists()
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(ToolError, match="not a folder"):
        run_cut_tool(static_box, mode="motion", out=str(f))


def test_cut_video_refuses_a_second_concurrent_run(static_box, tmp_path):
    assert mcp_server._cut_lock.acquire(blocking=False)
    try:
        with pytest.raises(ToolError, match="in progress"):
            run_cut_tool(static_box, mode="motion", out=str(tmp_path / "o"))
    finally:
        mcp_server._cut_lock.release()
    assert not mcp_server._cut_lock.locked()


def test_cut_video_failure_message_and_lock_released(static_box, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("2 of 3 clips failed to cut; scores and segments are saved in X")

    monkeypatch.setattr(mcp_server, "run_cut", boom)
    with pytest.raises(ToolError, match="clips failed to cut"):
        run_cut_tool(static_box, mode="motion", out=str(tmp_path / "o"))
    assert not mcp_server._cut_lock.locked()


# compare_runs
def _two_runs(static_box, tmp_path):
    a = run_cut_tool(static_box, mode="motion", pad=0.5, min_len=0.5, out=str(tmp_path / "x" / "motion"))
    b = run_cut_tool(static_box, mode="motion", pad=1.5, min_len=0.5, out=str(tmp_path / "x" / "wide"))
    return a, b


def test_compare_runs_and_out_folder(static_box, tmp_path):
    a, b = _two_runs(static_box, tmp_path)
    r = compare_runs(a["segments_json"], b["out_dir"])  # a file, and a run folder
    assert r["only_b_s"] > 0 and r["only_a_s"] == 0 and r["n_disagreements"] >= 1
    assert r["compare_json"] == str(tmp_path / "x" / "compare.json") and Path(r["compare_md"]).is_file()
    o = tmp_path / "elsewhere"
    r2 = compare_runs(a["segments_json"], b["segments_json"], min_len=0.1, out=str(o))
    assert Path(r2["compare_json"]) == o / "compare.json" and (o / "compare.md").is_file()


def test_compare_runs_refuses_different_videos_and_bad_input(static_box, synth, tmp_path):
    a = run_cut_tool(static_box, mode="motion", pad=0.5, min_len=0.5, out=str(tmp_path / "a"))
    other = run_cut_tool(synth["plain"], mode="motion", min_len=0.5, out=str(tmp_path / "b"))
    with pytest.raises(ToolError, match="different videos"):
        compare_runs(a["segments_json"], other["segments_json"], out=str(tmp_path / "c"))
    assert not (tmp_path / "c").exists()
    with pytest.raises(ToolError, match="no such file"):
        compare_runs(a["segments_json"], str(tmp_path / "nope.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ToolError, match="not a readable segments.json"):
        compare_runs(a["segments_json"], str(bad))
    with pytest.raises(ToolError, match="min_len must be"):
        compare_runs(a["segments_json"], a["segments_json"], min_len=-1)


# one client round trip over stdio against the real server process
def test_stdio_round_trip(static_box, tmp_path):
    out = tmp_path / "rt"

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "inframe.mcp_server"], cwd=str(ROOT))
        progress = []

        async def on_progress(p, total, message):
            progress.append((p, total, message))

        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            probe = await client.call_tool("probe_video", {"path": static_box})
            cut = await client.call_tool("cut_video", {"path": static_box, "mode": "motion", "pad": 0.5, "min_len": 0.5,
                                                       "out": str(out)}, progress_callback=on_progress)
            bad = await client.call_tool("cut_video", {"path": static_box, "mode": "face"})
            cmp_ = await client.call_tool("compare_runs", {"a": str(out), "b": str(out)})
        return names, probe, cut, bad, cmp_, progress

    names, probe, cut, bad, cmp_, progress = asyncio.run(go())
    assert names == {"probe_video", "cut_video", "compare_runs", "list_people", "list_targets", "add_target_crops", "export_reel"}
    assert not probe.is_error and probe.structured_content["codec"] == "h264"
    assert not cut.is_error, cut.content
    sc = cut.structured_content
    assert sc["n_clips"] == 1 and Path(sc["clips"][0]["file"]).is_file() and sc["segments_json"] == str(out / "segments.json")
    assert progress and progress[-1][0] == 100.0 and [p for p, _, _ in progress] == sorted(p for p, _, _ in progress)
    assert bad.is_error and "mode must be one of" in bad.content[0].text
    assert not cmp_.is_error and cmp_.structured_content["both_s"] == cmp_.structured_content["a"]["kept_s"]


def test_cut_video_reel_and_edit_lists(static_box, tmp_path):
    out = tmp_path / "o"
    r = run_cut_tool(static_box, mode="motion", pad=0.5, min_len=0.5, reel=True, edit_lists="fcpxml,otio", out=str(out))
    assert r["reel"]["ok"] and Path(r["reel"]["file"]).is_file() and Path(r["reel"]["file"]).parent == out
    assert [e["format"] for e in r["edit_lists"]] == ["fcpxml", "otio"] and all(e["ok"] and Path(e["file"]).is_file() for e in r["edit_lists"])
    plain = run_cut_tool(static_box, mode="motion", pad=0.5, min_len=0.5, out=str(tmp_path / "p"))
    assert "reel" not in plain and "edit_lists" not in plain


def test_cut_video_rejects_unknown_edit_list_format(static_box, tmp_path):
    with pytest.raises(ToolError, match="unknown edit list format"):
        run_cut_tool(static_box, mode="motion", edit_lists="premiere", out=str(tmp_path / "o"))
    assert not (tmp_path / "o").exists()


# export_reel
def test_export_reel_with_auto_music_offline_uses_the_built_in_beat(static_box, tmp_path, monkeypatch):
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("INFRAME_MUSIC_OFFLINE", "1")
    run = run_cut_tool(static_box, mode="motion", pad=0.5, min_len=0.5, out=str(tmp_path / "o"))
    r = asyncio.run(export_reel(run["out_dir"], music="auto"))
    assert r["ok"] and Path(r["reel"]["file"]).is_file() and r["reel"]["file"].startswith(run["out_dir"])
    auto = r["reel"]["music"]["auto"]
    assert auto["builtin"] is True and auto["rank"] == 0 and auto["n"] == 1 and r["reel"]["music"]["creator"] == "inframe"
    assert r["another_song"] == f"export_reel(run={run['segments_json']!r}, music='auto', music_rank=1)"
    assert json.loads(Path(r["export_json"]).read_text())["reel"]["music"]["auto"]["builtin"] is True


def test_export_reel_rejects_bad_arguments(static_box, tmp_path):
    run = run_cut_tool(static_box, mode="motion", pad=0.5, min_len=0.5, out=str(tmp_path / "o"))["out_dir"]
    for kw, msg in ((dict(music="auto:a/b"), "mood"), (dict(music="auto", music_rank=64), "0 to 63"), (dict(music_rank=1), "needs music"),
                    (dict(beat_sync=True), "needs music"), (dict(music="/no/such/song.mp3"), "no such file")):
        with pytest.raises(ToolError, match=msg):
            asyncio.run(export_reel(run, **kw))
    with pytest.raises(ToolError, match="absolute"):
        asyncio.run(export_reel("o"))
