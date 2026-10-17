"""Fixes: splice truth leaves out the seconds the target clips do not hold the target (padding, loop-back), and the
torch fallback loads the checkpoint from the path --offline checked."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inframe import detect, models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import splice  # noqa: E402


def _truth(name):
    return splice.truth_for(name, splice.COMPOSITES[name])


def _close(got, want):
    assert len(got) == len(want), (got, want)
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=1e-3), (got, want)


# ---------------------------------------------------------------- splice truth
def test_truth_leaves_out_clip_padding_and_loop_back():
    # the made-up window of targetA (tests/conftest.py): the file is 17.5 s long and the target is in it from 1.25 to 16.5 s
    t = _truth("swap_sides")
    # shot 2 (8-16 s) plays targetA from 10 s: present until 16.5 (6.5 s in), the file ends and loops back at 7.5 s
    # in, where the first 0.5 s of the clip (padding) are shown; shot 4 (24-32 s) plays from 12 s: present 4.5 s,
    # loops at 5.5 s and the target is back after 1.25 s of padding
    _close([x for iv in t["target_intervals"] for x in iv], [0, 14.5, 16, 28.5, 30.75, 32])  # 32: back in the clip 1.25 s after the loop
    _close(t["cuts"], [8 + 7.5, 24 + 5.5])


def test_truth_keeps_whole_shots_that_stay_inside_the_target_window():
    t = _truth("solo_gaps")  # ss 2, 10 s of a 15 s window: all of it holds the target
    _close([x for iv in t["target_intervals"] for x in iv], [0, 10, 18, 28, 36, 46])
    assert t["cuts"] == []


def test_truth_starting_inside_padding_does_not_count_the_padding():
    iv, cuts = splice.present_in_shot("targetC", ss=0.0, secs=4.0)  # the made-up targetC has 2 s of padding in front
    _close([x for p in iv for x in p], [2.0, 4.0])
    assert cuts == []


def test_every_composite_target_second_is_inside_a_window():
    if not splice.sources_available():
        pytest.skip("the sample clips for the composites are not configured (docs/TESTDATA.md, inframe.localcfg)")
    for name, recipe in splice.COMPOSITES.items():
        t = splice.truth_for(name, recipe)
        total = sum(e - s for s, e in t["target_intervals"])
        whole = sum(secs for _, secs, tiles in recipe if any(tl["role"] == "target" for tl in tiles))
        assert total <= whole + 1e-9

    def total(n):
        return sum(e - s for s, e in _truth(n)["target_intervals"])

    # quad: targetB (made-up window 17.0 s long, in frame 1.0-15.9) ss 2 loses 0.1 s past 15.9; ss 6 passes 15.9, loops at 11 s in and shows
    # 1.0 s of padding: 1.1 + 1.0 = 2.1 s
    assert total("quad") == pytest.approx(28 - 0.1 - 2.1, abs=2e-3)
    # pip_small: targetC (16.5 s long, in frame 2.0-15.5) ss 6 for 12 s passes 15.5, loops at 10.5 s in and shows 1.5 s of padding: 2.5 s
    assert total("pip_small") == pytest.approx(30 - 2.5, abs=2e-3)


def test_id_switches_does_not_count_the_loop_back_cut():
    t = _truth("swap_sides")
    cut = t["cuts"][0]
    right = (0.6, 0.3, 0.7, 0.7)  # targetA is the right tile of shot 2
    before, after = cut - 0.5, cut + 0.2
    samples = [(before, [(1, right, 0.9)]), (after, [(9, right, 0.9)])]
    r = splice.id_switches(t, samples)
    assert r["id_switches"] == 0 and r["strongest_changes"] == 0


def test_tracker_restarts_at_shots_and_loop_backs():
    t = _truth("swap_sides")
    starts = splice.shot_starts(t)
    assert starts == sorted(starts) and 0.0 in starts and t["cuts"][0] in starts and 8.0 in starts


# ---------------------------------------------------------------- torch fallback checkpoint
def _fake_torch(monkeypatch):
    seen = {}

    class RFDETRSmall:
        def __init__(self, **kw):
            seen.update(kw)
            self.model = MagicMock()

    rf = MagicMock()
    rf.RFDETRSmall = RFDETRSmall
    monkeypatch.setitem(sys.modules, "rfdetr", rf)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    return seen


def test_fallback_loads_the_checkpoint_offline_checked_from_any_directory(tmp_path, monkeypatch):
    seen = _fake_torch(monkeypatch)
    cache = tmp_path / "cache"
    monkeypatch.setenv("INFRAME_CACHE", str(cache))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    ckpt = cache / "models" / "rf-detr-small.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    spec = models.ModelFile("rf-detr-small.pth", "https://example.invalid/x", "md5:" + __import__("hashlib").md5(b"weights").hexdigest())
    monkeypatch.setitem(models.REGISTRY, "rf-detr-small.pth", spec)
    models.set_offline(True)
    try:
        detect.load_fallback_detector("small")
    finally:
        models.set_offline(False)
    assert seen.get("pretrain_weights") == str(ckpt)  # not rfdetr's default, which is a path relative to the cwd


def test_fallback_offline_ignores_roboflow_folder_and_cwd_copies(tmp_path, monkeypatch):
    _fake_torch(monkeypatch)
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".roboflow" / "models").mkdir(parents=True)
    (tmp_path / "home" / ".roboflow" / "models" / "rf-detr-small.pth").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rf-detr-small.pth").write_bytes(b"x")
    models.set_offline(True)
    try:
        with pytest.raises(models.OfflineError, match="fetch-models"):
            detect.load_fallback_detector("small")
    finally:
        models.set_offline(False)


def test_fallback_online_fetches_into_the_cache_and_uses_that_path(tmp_path, monkeypatch):
    seen = _fake_torch(monkeypatch)
    got = tmp_path / "rf-detr-small.pth"
    got.write_bytes(b"w")
    monkeypatch.setattr(models, "fetch", lambda spec, *a, **k: got if spec.name == "rf-detr-small.pth" else None)
    monkeypatch.chdir(tmp_path)
    detect.load_fallback_detector("small")
    assert seen.get("pretrain_weights") == str(got)
