"""T1d: list_people and cut_video(target=...) over MCP. Unit tests on synthetic data; one stdio round trip on a spliced composite
(skipped when the composite, the detector models or the tracking extra are missing; the composite is built by tools/splice.py from your own target clips and openly licensed clips)."""
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError

from inframe import mcp_server, people
from inframe.detect import Detection
from inframe.mcp_server import cut_video, list_people

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import splice  # noqa: E402

COMPOSITE = splice.DEFAULT_OUT / "two_side.mp4"
MODELS = Path(os.environ.get("INFRAME_MODELS", ROOT / "bench/detect/out/models"))


def test_stretches_join_samples_closer_than_the_gap():
    assert people.stretches([1.0, 1.2, 1.4, 5.0, 5.2], 0.2) == [[1.0, 1.6], [5.0, 5.4]]
    assert people.stretches([1.0, 2.4], 0.2) == [[1.0, 2.6]]  # 1.2 s between the end of one sample and the next: joined
    assert people.stretches([], 0.2) == []


def test_people_scan_keeps_best_crop_per_track():
    scan = people.PeopleScan()
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, 100:] = 255
    small, big = (0.1, 0.1, 0.2, 0.4), (0.5, 0.1, 0.9, 0.9)
    scan.evaluate(0.0, frame, [Detection(0.9, small, 7), Detection(0.8, (0.0, 0.0, 0.1, 0.1), None)])
    scan.evaluate(0.2, frame, [Detection(0.9, big, 7)])
    scan.evaluate(0.4, frame, [Detection(0.9, small, 7)])
    assert list(scan.tracks) == [7]
    tr = scan.tracks[7]
    assert tr.times == [0.0, 0.2, 0.4] and tr.best_t == 0.2 and tr.best_box == big
    assert np.asarray(tr.thumb)[50, 50].min() == 255  # the crop of the big box lies in the white half


def test_scan_people_writes_files(monkeypatch, tmp_path):
    from inframe import pipeline

    class FakeInfo:
        duration = 10.0

    def fake_score(video, info, preset, fps, target, progress, kind):
        frame = np.full((90, 160, 3), 100, np.uint8)
        for i in range(10):  # person 3 for 2 s, person 4 for 0.4 s
            target.evaluate(i * 0.2, frame, [Detection(0.9, (0.2, 0.1, 0.5, 0.9), 3)] + ([Detection(0.9, (0.6, 0.1, 0.8, 0.9), 4)] if i < 2 else []))
        return pipeline.Scored([], [], [], dict(tracker="ocsort", shot_cuts=0), [], 0.1)

    monkeypatch.setattr(people, "score_detect", fake_score)
    monkeypatch.setattr(people, "probe", lambda v: FakeInfo())
    doc = people.scan_people("/x/v.mp4", tmp_path / "p")
    assert doc["n_tracks"] == 2 and doc["n_people"] == 1  # person 4 is under 1 s
    p = doc["people"][0]
    assert p["track_id"] == 3 and p["seen_s"] == 2.0 and p["stretches"] == [[0.0, 2.0]]
    stem = f"person_001_{doc['scan_id']}"
    assert Path(p["thumbnail"]).is_file() and p["target"] == f"tap:{tmp_path / 'p' / (stem + '.json')}"
    tj = json.loads((tmp_path / "p" / f"{stem}.json").read_text())
    assert tj["box"] == [0.2, 0.1, 0.5, 0.9] and tj["track_id"] == 3 and "time" in tj
    assert json.loads((tmp_path / "p" / "people.json").read_text())["n_people"] == 1


def _fake_scan(monkeypatch, tracks_by_id):
    """scan_people with a faked scorer: tracks_by_id maps track id -> number of 0.2 s samples."""
    from inframe import pipeline

    class FakeInfo:
        duration = 10.0

    def fake_score(video, info, preset, fps, target, progress, kind):
        frame = np.full((90, 160, 3), 100, np.uint8)
        for tid, n in tracks_by_id.items():
            for i in range(n):
                target.evaluate(i * 0.2, frame, [Detection(0.9, (0.1 * tid, 0.1, 0.1 * tid + 0.2, 0.9), tid)])
        return pipeline.Scored([], [], [], dict(tracker="ocsort", shot_cuts=0), [], 0.1)

    monkeypatch.setattr(people, "score_detect", fake_score)
    monkeypatch.setattr(people, "probe", lambda v: FakeInfo())


def test_scan_people_leaves_files_it_did_not_write(monkeypatch, tmp_path):
    out = tmp_path / "people_dir"
    out.mkdir()
    mine = ["person_a.jpg", "person_b.jpg", "person_notes.json", "person_12.json", "person_001_x.jpg", "notes.txt"]
    for n in mine:
        (out / n).write_text("keep")
    _fake_scan(monkeypatch, {3: 10})
    first = people.scan_people("/x/v.mp4", out)
    _fake_scan(monkeypatch, {})  # an empty second scan removes the first scan's files and nothing else
    people.scan_people("/x/v.mp4", out)
    assert sorted(p.name for p in out.iterdir()) == sorted(mine + ["people.json"])
    assert not list(out.glob(f"person_*_{first['scan_id']}.*"))


def test_rescan_gives_every_person_file_a_new_name(monkeypatch, tmp_path):
    out = tmp_path / "p"
    _fake_scan(monkeypatch, {3: 10, 4: 8})
    a = people.scan_people("/x/v.mp4", out)
    _fake_scan(monkeypatch, {5: 10, 3: 8})  # same person numbers, other tracks
    b = people.scan_people("/x/v.mp4", out)
    assert a["scan_id"] != b["scan_id"]
    old, new = {p["target"] for p in a["people"]}, {p["target"] for p in b["people"]}
    assert not old & new  # an earlier target string cannot point at another person
    assert all(not Path(t[4:]).exists() for t in old) and all(Path(t[4:]).is_file() for t in new)
    assert json.loads((out / "people.json").read_text())["scan_id"] == b["scan_id"]


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("kw, msg", [
    (dict(min_seen=-1), "min_seen must be"),
    (dict(max_people=0), "max_people must be"),
    (dict(out="rel/dir"), "absolute"),
])
def test_list_people_rejects_bad_arguments(static_box, tmp_path, kw, msg):
    with pytest.raises(ToolError, match=msg):
        run(list_people(static_box, **kw))


def test_list_people_rejects_bad_paths(tmp_path):
    with pytest.raises(ToolError, match="no such file"):
        run(list_people(str(tmp_path / "gone.mov")))
    with pytest.raises(ToolError, match="absolute"):
        run(list_people("clip.mp4"))


@pytest.mark.parametrize("target, msg", [
    ("tap:relative/target.json", "absolute"),
    ("photos:relative/dir", "absolute"),
    ("face:3", "tap:<target.json>"),
    ("tap:/no/such/target.json", "not a file"),
    ("tap:1,2,3", "fractions"),
    (["tap:3,0.5,0.5", 7], "target must be"),
])
def test_cut_video_rejects_bad_targets(static_box, tmp_path, target, msg):
    with pytest.raises(ToolError, match=msg):
        run(cut_video(static_box, target=target, out=str(tmp_path / "o")))
    assert not (tmp_path / "o").exists()


def test_cut_video_target_needs_detect_mode(static_box, tmp_path):
    with pytest.raises(ToolError, match="detect"):
        run(cut_video(static_box, mode="motion", target="tap:1,0.5,0.5", out=str(tmp_path / "o")))


def test_cut_video_passes_the_target_to_run_cut(static_box, tmp_path, monkeypatch):
    seen = {}

    def fake(video, out, settings, target, progress=None):
        seen["target"] = target
        raise RuntimeError("stop here")

    monkeypatch.setattr(mcp_server, "run_cut", fake)
    with pytest.raises(ToolError, match="stop here"):
        run(cut_video(static_box, target="tap:3,0.5,0.5", out=str(tmp_path / "o")))
    assert seen["target"].name == "tap" and seen["target"].taps[0].t == 3.0
    assert not mcp_server._cut_lock.locked()


def test_list_people_shares_the_cut_lock(static_box, tmp_path):
    assert mcp_server._cut_lock.acquire(blocking=False)
    try:
        with pytest.raises(ToolError, match="in progress"):
            run(list_people(static_box, out=str(tmp_path / "o")))
    finally:
        mcp_server._cut_lock.release()


def _have(mod):
    return importlib.util.find_spec(mod) is not None


needs_composite = pytest.mark.skipif(
    not (COMPOSITE.is_file() and COMPOSITE.with_suffix(".truth.json").is_file() and (MODELS / "small_coreml_b8").exists() and _have("trackers")),
    reason="needs the two_side composite (tools/splice.py build), the detector models and the tracking extra")


@needs_composite
def test_stdio_round_trip_on_a_composite(tmp_path):
    truth = json.loads(COMPOSITE.with_suffix(".truth.json").read_text())
    out_people, out_cut = tmp_path / "people", tmp_path / "cut"

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "inframe.mcp_server"], cwd=str(ROOT),
                                        env={**os.environ, "INFRAME_MODELS": str(MODELS)})  # the child gets no env by default
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            lp = await client.call_tool("list_people", {"path": str(COMPOSITE), "out": str(out_people)})
            assert not lp.is_error, lp.content
            sc = lp.structured_content
            # which listed people are the target: by the composition truth (tile role at the thumbnail time and box centre), never by looking
            mine = []
            for p in sc["people"]:
                tj = json.loads(Path(p["target"][4:]).read_text())
                b = tj["box"]
                _, tile = splice.tile_at(truth, tj["time"], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                if tile is not None and tile["role"] == "target":
                    mine.append(p)
            assert mine, "no listed person stands in the target's tile"
            cut = await client.call_tool("cut_video", {"path": str(COMPOSITE), "target": [p["target"] for p in mine], "out": str(out_cut)})
            assert not cut.is_error, cut.content
        return names, sc, mine, cut.structured_content

    names, sc, mine, cut = run(go())
    assert {"probe_video", "cut_video", "compare_runs", "list_people"} <= names
    assert sc["n_people"] == len(sc["people"]) >= 2 and sc["people_json"] == str(out_people / "people.json")
    for p in sc["people"]:
        assert Path(p["thumbnail"]).is_file() and p["thumbnail"].startswith(str(out_people)) and 0 <= p["first_s"] < p["last_s"] <= truth["duration"] + 1
        assert p["seen_s"] >= 1.0 and p["stretches"] and p["target"].startswith("tap:")
    assert [p["seen_s"] for p in sc["people"]] == sorted((p["seen_s"] for p in sc["people"]), reverse=True)
    score = splice.score_intervals(truth, cut["presence_intervals"])
    print("composite two_side, list_people then cut_video(target):", score)
    assert score["precision"] >= 0.9 and score["recall"] >= 0.8, score
