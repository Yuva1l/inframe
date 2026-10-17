import json

import numpy as np
import pytest

from inframe.compare import compare_files, compare_intervals, intersect, subtract
from inframe.motion import auto_thresholds, background_for, run_motion
from inframe.pipeline import Settings, run_cut
from inframe.probe import probe


def test_auto_thresholds_from_distribution():
    rng = np.random.default_rng(0)
    quiet = np.abs(rng.normal(0, 3e-4, 900))
    busy = np.full(100, 0.02)
    enter, exit_, st = auto_thresholds(np.concatenate([quiet, busy]))
    assert exit_ < enter
    assert quiet.max() < enter or (quiet > enter).mean() < 0.01  # quiet samples stay below enter
    assert enter < 0.02  # busy samples are above it
    assert auto_thresholds([0.0] * 50)[0] > 0  # an empty video does not get a threshold at zero
    assert auto_thresholds([])[0] >= 1.0


def test_background_median_ignores_a_transient():
    stack = np.full((9, 4, 4), 50, np.uint8)
    stack[3:5, 1:3, 1:3] = 250  # a bright blob for 2 of 9 samples
    t = np.arange(9.0)
    assert (background_for(stack, t, 4.0, window=100) == 50).all()
    assert background_for(stack, t, 4.0, window=0.4).max() == 250  # window of one sample: no median help


def test_motion_finds_box_between_2_and_4_s(static_box):
    info = probe(static_box)
    r = run_motion(static_box, info, fps=5.0)
    assert len(r.times) == 40 and not r.warnings
    sc = np.asarray(r.scores)
    t = np.asarray(r.times)
    assert sc[(t > 2.4) & (t < 3.6)].min() > 0.05  # 50x70 box is 10.9% of the frame
    assert sc[(t < 1.6) | (t > 4.4)].max() < 0.005


def test_motion_end_to_end(static_box, tmp_path):
    out = tmp_path / "m"
    doc = run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5))
    assert doc["mode"] == "motion" and len(doc["presence_intervals"]) == 1
    s, e = doc["presence_intervals"][0]
    assert 1.6 < s < 2.4 and 3.6 < e < 4.4
    c = doc["clips"][0]
    assert (out / c["file"]).is_file() and (out / "segments.json").is_file()
    assert c["start"] == pytest.approx(s - 0.5, abs=1e-3) and c["end"] == pytest.approx(e + 0.5, abs=1e-3)
    assert probe(str(out / c["file"])).has_audio
    assert doc["settings"]["thresholds"]["source"].startswith("auto")


def test_motion_flags_override_auto_thresholds(static_box, tmp_path):
    doc = run_cut(static_box, str(tmp_path / "o"), Settings(mode="motion", fps=5.0, enter=0.5, exit=0.4, min_len=0.5))
    assert doc["presence_intervals"] == []  # the box fills 10.9% only
    assert doc["settings"]["enter"] == 0.5


def test_motion_warns_when_camera_moves(shaky):
    r = run_motion(shaky, probe(shaky), fps=5.0)
    assert r.warnings and "not look static" in r.warnings[0]
    assert r.details["camera_shift_px"]["p95"] > 1.5


def test_interval_arithmetic():
    assert intersect([(0, 5), (8, 10)], [(3, 9)]) == [(3, 5), (8, 9)]
    assert subtract([(0, 10)], [(2, 3), (5, 6)]) == [(0, 2), (3, 5), (6, 10)]
    assert subtract([(0, 1)], [(0, 1)]) == []
    assert subtract([(0, 1)], []) == [(0, 1)]
    r = compare_intervals([(0, 10), (20, 22)], [(5, 15), (20.5, 21)], min_len=1.0)
    assert (r["both_s"], r["only_a_s"], r["only_b_s"]) == (5.5, 6.5, 5.0)
    assert [(d["only_in"], d["start"], d["end"]) for d in r["disagreements"]] == [("A", 0, 5), ("B", 10, 15), ("A", 21, 22)]  # the 0.5 s bit is dropped


def test_compare_files_writes_next_to_parent_folder(tmp_path):
    def doc(mode, clips):
        return dict(mode=mode, source=dict(duration=100.0), clips=[dict(start=s, end=e) for s, e in clips])

    for name, mode, clips in (("motion", "motion", [(0, 10), (30, 40)]), ("detect", "detect", [(5, 12), (30, 40)])):
        (tmp_path / name).mkdir()
        (tmp_path / name / "segments.json").write_text(json.dumps(doc(mode, clips)))
    r = compare_files(tmp_path / "motion" / "segments.json", tmp_path / "detect" / "segments.json")
    assert (r["both_s"], r["only_a_s"], r["only_b_s"]) == (15.0, 5.0, 2.0)
    assert r["n_disagreements"] == 2
    assert (tmp_path / "compare.json").is_file()
    md = (tmp_path / "compare.md").read_text()
    assert "00:00.0" in md and "00:05.0" in md and "only in" in md


def test_compare_files_out_folder_and_same_source_check(tmp_path):
    import pytest as _pt

    def write(name, path, duration=100.0, w=320):
        d = tmp_path / name
        d.mkdir()
        (d / "segments.json").write_text(json.dumps(dict(
            mode=name, source=dict(path=path, duration=duration, width=w, height=180, fps=30.0), clips=[dict(start=0, end=10)])))
        return d / "segments.json"

    a, b = write("motion", "/v/IMG_1.mov"), write("detect", "/other/dir/IMG_1.mov")  # same name and facts: same video
    o = tmp_path / "chosen"
    r = compare_files(a, b, out=o)
    assert (o / "compare.json").is_file() and (o / "compare.md").is_file() and not (tmp_path / "compare.json").exists()
    assert r["compare_json"] == str(o / "compare.json")
    c = write("c", "/v/IMG_2.mov")
    d = write("d", "/v/IMG_1.mov", duration=101.0)
    e = write("e", "/v/IMG_1.mov", w=640)
    for other, why in ((c, "file name"), (d, "duration"), (e, "width")):
        with _pt.raises(ValueError, match=f"different videos.*{why}"):
            compare_files(a, other)
    assert not (tmp_path / "compare.json").exists()
    junk = tmp_path / "junk.json"
    junk.write_text(json.dumps([1, 2]))
    with _pt.raises(ValueError, match="not a segments.json"):
        compare_files(a, junk)


def test_cli_compare_out_and_refusal(tmp_path, capsys):
    from inframe import cli

    def write(name, path):
        d = tmp_path / name
        d.mkdir()
        (d / "segments.json").write_text(json.dumps(dict(mode=name, source=dict(path=path, duration=50.0), clips=[dict(start=0, end=5)])))
        return str(d / "segments.json")

    a, b, c = write("motion", "/v/A.mov"), write("detect", "/v/A.mov"), write("other", "/v/B.mov")
    assert cli.main(["compare", a, b, "--out", str(tmp_path / "res")]) == 0
    assert (tmp_path / "res" / "compare.md").is_file() and not (tmp_path / "compare.json").exists()
    assert cli.main(["compare", a, c]) == 2
    assert "different videos" in capsys.readouterr().err
