"""Second review round: first-GOP cuts, early --exit failure in motion mode, stale clips after a failed detect run."""
import json
import subprocess

import pytest

from inframe import pipeline
from inframe.cut import cut_clip
from inframe.pipeline import Settings, run_cut
from inframe.probe import first_packet_pts


def _video_packets(path: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return int(json.loads(out)["streams"][0]["nb_read_packets"])


@pytest.mark.parametrize("start, end", [(0.0, 1.0), (0.0, 0.3)])
def test_cut_in_the_first_gop_keeps_the_video(open_gop, tmp_path, start, end):
    out = tmp_path / "c.mp4"
    cut_clip(open_gop, out, start, end)
    v, a = first_packet_pts(str(out))
    assert _video_packets(str(out)) >= 5 * (end - start) * 6  # about 30 fps; a lost video stream leaves 0-5 packets
    assert abs(v) < 0.05, v
    assert a is not None and abs(v - a) < 0.03


def test_motion_exit_above_auto_enter_fails_with_the_value_and_keeps_samples(static_box, tmp_path):
    out = tmp_path / "o"
    with pytest.raises(ValueError, match=r"--exit .*automatic enter threshold \d") as e:
        run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5, exit=0.5))
    assert "0.5" in str(e.value)
    assert (out / "samples.csv").is_file()
    assert "error" in json.loads((out / "segments.json").read_text())


def test_explicit_exit_above_enter_fails_before_any_work(static_box, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("scoring ran")

    monkeypatch.setattr(pipeline, "score_motion", boom)
    with pytest.raises(ValueError, match="--exit"):
        run_cut(static_box, str(tmp_path / "o"), Settings(mode="motion", enter=0.2, exit=0.5))


def test_failed_detect_run_removes_the_earlier_clips(static_box, tmp_path, monkeypatch):
    out = tmp_path / "o"
    run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5))
    assert list(out.glob("static_box_0*.mp4")) and (out / "static_box_losslesscut.csv").is_file()

    def fail(*a, **k):
        raise pipeline.ScoringError("torch is not installed", [0.0, 0.2], [0.1, 0.1], [None, None])

    monkeypatch.setattr(pipeline, "score_detect", fail)
    with pytest.raises(pipeline.ScoringError):
        run_cut(static_box, str(out), Settings(mode="detect"))
    assert not list(out.glob("static_box_0*.mp4"))
    assert not (out / "static_box_losslesscut.csv").exists()
    doc = json.loads((out / "segments.json").read_text())
    assert doc["clips"] == [] and "error" in doc
