"""tools/splice.py scoring that does not flatter the run: tile-aware precision, oracle ceilings, median and spread."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import splice  # noqa: E402

from inframe.detect import Detection  # noqa: E402
from inframe.tap import TapTarget  # noqa: E402

LEFT, RIGHT = (0.1, 0.2, 0.3, 0.8), (0.6, 0.2, 0.8, 0.8)  # a person in the left tile (the target), one in the right tile (a stranger)


def truth_two_tiles(seconds: float = 10.0) -> dict:
    tiles = [dict(id=0, src="targetA", role="target", rect=[0, 0, 640, 720]), dict(id=1, src="pair1", role="other", rect=[640, 0, 640, 720])]
    return dict(name="t", duration=seconds, width=1280, height=720, fps=30, cuts=[],
                segments=[dict(t0=0.0, t1=seconds, layout="side", tiles=tiles)], target_intervals=[[0.0, seconds]])


def grid(seconds: float = 10.0):
    return [round(0.2 * k, 1) for k in range(int(seconds / 0.2))]


def test_a_hit_in_the_stranger_tile_scores_precision_zero():
    truth = truth_two_tiles()
    hits = [(t, RIGHT) for t in grid()]
    r = splice.score_tile(truth, [[0.0, 10.0]], hits)
    assert r["precision"] <= 0.01 and r["recall"] <= 0.01 and r["wrong_tile_s"] >= 9.9  # (the last 0.1 s lies outside the last sample's cell)
    # the old time-only scoring gave full marks, because the target is in frame in those seconds
    assert splice.score_intervals(truth, [[0.0, 10.0]])["precision"] == 1.0
    mine = splice.score_tile(truth, [[0.0, 10.0]], [(t, LEFT) for t in grid()])
    assert mine["precision"] == 1.0 and mine["recall"] >= 0.98
    half = [(t, LEFT if t < 5 else RIGHT) for t in grid()]
    r = splice.score_tile(truth, [[0.0, 10.0]], half)
    assert abs(r["precision"] - 0.5) < 0.03 and abs(r["recall"] - 0.5) < 0.03


def test_a_gap_without_a_box_is_judged_by_time_and_spans_restrict():
    truth = truth_two_tiles()
    hits = [(t, LEFT if t < 4 else None) for t in grid()]  # kept 0-10 s, boxes only until 4 s (the rest was bridged)
    assert splice.score_tile(truth, [[0.0, 10.0]], hits)["precision"] == 1.0
    wrong = [(t, RIGHT) for t in grid()]
    assert splice.score_tile(truth, [[0.0, 10.0]], wrong, spans=[[0.0, 5.0]])["kept_s"] == 5.0
    assert splice.score_tile(truth, [], wrong)["precision"] == 0.0 and splice.score_tile(truth, [], wrong)["recall"] == 0.0


def _target(truth: dict, tracks: bool = True) -> TapTarget:
    """A finished target holding 50 samples: track 1 in the target's tile, track 2 (a stranger) in the other tile; with `tracks` False
    nothing has an id and an extra untracked detection in the target's tile shows up in the last second."""
    tg = TapTarget([])
    frame = np.zeros((360, 640, 3), np.uint8)
    for t in grid():
        dets = [Detection(0.9, LEFT, 1 if tracks else None), Detection(0.9, RIGHT, 2 if tracks else None)]
        tg.evaluate(t, frame, dets)
    return tg


def test_oracle_ceilings_count_only_target_tile():
    truth = truth_two_tiles()
    tg = _target(truth)
    for mode in ("dets", "tracks"):
        r = splice.oracle_ceiling(truth, tg, mode)
        assert r["precision"] == 1.0 and r["recall"] >= 0.98, mode
    hits = splice.oracle_hits(truth, tg, "tracks")
    assert all(h.box == LEFT for h in hits)  # never the stranger, who has the same score
    # without a tracker only "dets" can follow the target
    nt = _target(truth, tracks=False)
    assert splice.oracle_ceiling(truth, nt, "dets")["recall"] >= 0.98


def test_oracle_misses_what_the_detector_never_saw():
    truth = truth_two_tiles()
    tg = TapTarget([])
    frame = np.zeros((360, 640, 3), np.uint8)
    for t in grid():
        tg.evaluate(t, frame, [Detection(0.9, LEFT if t < 6 else RIGHT, 1 if t < 6 else 2)])
    r = splice.oracle_ceiling(truth, tg, "tracks")
    assert r["precision"] == 1.0 and 0.5 < r["recall"] < 0.7  # the stranger alone in the last 4 s is not credited


def test_median_and_spread():
    assert splice.median_spread([0.62, 0.9, 0.71]) == (0.71, 0.9 - 0.62)
    assert splice.median_spread([0.5, 0.5, 0.5]) == (0.5, 0.0)
    assert splice.median_spread([0.4, 0.6])[0] == 0.5


def test_temp_targets_store_restores_the_environment(monkeypatch):
    import os

    monkeypatch.setenv("INFRAME_TARGETS", "/before")
    with splice.temp_targets_store() as store:
        assert os.environ["INFRAME_TARGETS"] == str(store) and Path(store).is_dir()
    assert os.environ["INFRAME_TARGETS"] == "/before" and not Path(store).exists()
    monkeypatch.delenv("INFRAME_TARGETS")
    with splice.temp_targets_store():
        pass
    assert "INFRAME_TARGETS" not in os.environ


def test_work_dirs_follow_out_not_the_home_cache(tmp_path):
    a, b = splice.work_dir(tmp_path / "a", "tap"), splice.work_dir(tmp_path / "b", "tap")
    assert a != b and a.parent.parent == tmp_path / "a"
    assert splice.work_dir(tmp_path / "a", "photo", "run1").name == "run1"
    assert splice.work_dir(tmp_path / "a", "photo") != splice.work_dir(tmp_path / "a", "tap")


def test_subsample_step_is_never_zero():
    assert [splice.subsample_step(f) for f in (2, 5, 10)] == [5, 2, 1]
    assert splice.subsample_step(20) == 1 and splice.subsample_step(30) == 1


def test_run_tracker_accepts_fps_above_the_detection_rate():
    import importlib.util

    if importlib.util.find_spec("trackers") is None:  # the tracking extra is not installed; find_spec does not import it (and its OpenCV)
        pytest.skip("needs the tracking extra")
    dets = [(round(0.1 * k, 1), [((0.1, 0.1, 0.3, 0.9), 0.9)]) for k in range(20)]
    assert splice.run_tracker(dets, 20.0, "ocsort") is not None


def test_comment_names_the_real_default_units():
    from inframe.detect import DEFAULT_UNITS

    src = (ROOT / "tools/splice.py").read_text()
    line = next(l for l in src.splitlines() if "default compute units" in l)
    assert DEFAULT_UNITS in line
