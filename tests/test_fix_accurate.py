"""Findings of the accurate-preset review: target.json records the detector, cut follows it; bench_units pairs detections by box."""
import importlib.util
import json
from pathlib import Path

import pytest

from inframe import cli, mcp_server, pick, tap

ROOT = Path(__file__).resolve().parents[1]


def _picked(synth, tmp_path, model="base", name="target.json"):
    out = tmp_path / name
    pk = pick.Picker(synth["plain"], out, model)
    # a tap by hand: the tracker is not needed
    pk.taps = [dict(time=1.0, box=[0.1, 0.1, 0.5, 0.9], track_id=None, index=1)]
    pk.save()
    return out


def test_pick_records_the_preset_and_model(synth, tmp_path):
    d = json.loads(_picked(synth, tmp_path, "base").read_text())
    assert d["model"] == "base" and d["preset"] == "accurate"
    d = json.loads(_picked(synth, tmp_path, "small", "small.json").read_text())
    assert d["model"] == "small" and d["preset"] == "balanced"


def test_cut_defaults_to_the_recorded_preset_and_refuses_another_detector(synth, tmp_path, monkeypatch, capsys):
    f = _picked(synth, tmp_path, "base")
    spec = [f"tap:{f}"]
    assert tap.resolve_preset(spec, None) == "accurate"
    assert tap.resolve_preset(spec, "accurate") == "accurate"
    with pytest.raises(tap.TapError, match="picked with the accurate preset"):
        tap.resolve_preset(spec, "balanced")
    assert tap.resolve_preset(["tap:1,0.5,0.5"], None) == "balanced"  # a hand tap says nothing
    seen = []
    monkeypatch.setattr(cli, "run_cut", lambda video, out, s, target=None, **k: seen.append(s.preset) or (_ for _ in ()).throw(RuntimeError("stop")))
    monkeypatch.setattr(cli.identity, "parse_targets", lambda *a, **k: "T")
    assert cli.main(["cut", synth["plain"], "--target", f"tap:{f}", "--out", str(tmp_path / "o")]) == 1
    assert seen == ["accurate"]
    assert cli.main(["cut", synth["plain"], "--target", f"tap:{f}", "--preset", "balanced", "--out", str(tmp_path / "o")]) == 2
    assert "picked with the accurate preset" in capsys.readouterr().err


def test_mcp_cut_video_follows_and_checks_the_recorded_preset(synth, tmp_path, monkeypatch):
    import anyio

    f = _picked(synth, tmp_path, "base")
    seen = []

    monkeypatch.setattr(mcp_server, "_parse_target", lambda t, m, p="balanced": seen.append(p) or "T")
    monkeypatch.setattr(mcp_server, "_run_exclusive", lambda job, ctx: (_ for _ in ()).throw(RuntimeError(seen[-1])))
    with pytest.raises(RuntimeError, match="accurate"):
        anyio.run(lambda: mcp_server.cut_video(synth["plain"], target=f"tap:{f}"))
    with pytest.raises(mcp_server.ToolError, match="picked with the accurate preset"):
        anyio.run(lambda: mcp_server.cut_video(synth["plain"], preset="balanced", target=f"tap:{f}"))


def _bench():
    spec = importlib.util.spec_from_file_location("bench_units", ROOT / "tools/bench_units.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bench_compare_pairs_by_box_not_score_rank():
    b = _bench()
    left, right = (0.1, 0.1, 0.3, 0.9), (0.6, 0.1, 0.8, 0.9)
    a = [[(0.610, left), (0.600, right)]]
    c = [[(0.605, right), (0.604, left)]]  # the order by score swapped; no score moved more than 0.006
    r = b.compare(a, c)
    assert r["identical"] is False and r["unmatched"] == 0
    assert r["max_score_diff"] == pytest.approx(0.006) and r["max_box_diff"] == 0.0


def test_bench_compare_counts_a_vanished_detection():
    b = _bench()
    box = (0.1, 0.1, 0.3, 0.9)
    r = b.compare([[(0.7, box), (0.6, (0.6, 0.1, 0.8, 0.9))]], [[(0.7, box)]])
    assert r["unmatched"] == 1 and r["frames_with_other_count"] == 1 and r["max_score_diff"] == 0.0
    same = b.compare([[(0.7, box)]], [[(0.7, box)]])
    assert same["identical"] is True and same["unmatched"] == 0 and same["max_score_diff"] == same["max_box_diff"] == 0.0
