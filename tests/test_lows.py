"""Edge cases of pipeline.py and editlist.py."""
import json

import pytest

from inframe import pipeline
from inframe.detect import Detection
from inframe.editlist import build_timeline, write_edit_lists, FORMATS
from inframe.pipeline import Settings, ScoringError, run_cut, score_detect, PRESETS
from inframe.probe import probe
from inframe.target import AnyPerson


class Det:
    path = "fake.mlpackage"

    def __init__(self, backend="fake", fail_from=None):
        self.backend, self.fail_from, self.n = backend, fail_from, 0

    def detect(self, frames):
        if self.fail_from is not None and self.n >= self.fail_from:
            raise RuntimeError("boom")
        self.n += len(frames)
        return [[Detection(0.9, (0.1, 0.1, 0.3, 0.5))] for _ in frames]


class TrackingPerson(AnyPerson):
    wants_tracks = True


def test_later_cut_run_removes_export_json(static_box, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "export.json").write_text("{}")
    run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5))
    assert not (out / "export.json").exists()


def test_no_tracker_warning_does_not_hide_the_coreml_fallback_warning(synth, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: Det(backend="torch MPS FALLBACK"))
    info = probe(synth["plain"])
    sc = score_detect(synth["plain"], info, PRESETS["balanced"], 1.0, TrackingPerson(0.0))  # 1 sample/s: no tracker
    assert any("no tracker" in w for w in sc.warnings)
    assert any("fell back to torch MPS" in w for w in sc.warnings), sc.warnings


def test_mid_run_failure_without_tracker_writes_no_track_column(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: Det(fail_from=8))
    monkeypatch.setattr(pipeline, "load_fallback_detector", lambda size: (_ for _ in ()).throw(RuntimeError("no torch")))
    out = tmp_path / "o"
    with pytest.raises(ScoringError):
        run_cut(synth["plain"], str(out), Settings(pad=0.5, min_len=0.5, fps=5.0), None)
    rows = (out / "samples.csv").read_text().splitlines()
    assert len(rows) == 9  # header + the 8 samples of the first batch
    assert rows[0] == "t,score,x0,y0,x1,y1"


@pytest.mark.parametrize("flag,kw", [("--enter", dict(enter="high")), ("--exit", dict(exit=2.0)), ("--min-box", dict(min_box=True))])
def test_range_check_runs_before_the_output_folder_is_made_and_survives_non_numbers(synth, tmp_path, flag, kw):
    out = tmp_path / "never"
    with pytest.raises(ValueError, match=f"{flag} must be a number from 0 to 1"):
        run_cut(synth["plain"], str(out), Settings(**kw))
    assert not out.exists()


def test_check_ranges_names_a_non_number_without_crashing():
    msg = pipeline.check_ranges("high", None, 0.0)
    assert msg == "--enter must be a number from 0 to 1 (got 'high')"


def _offset_doc(fstart, offset):
    return dict(inframe_version="t", mode="detect", presence_intervals=[],
                clips=[dict(index=1, start=2.0 + offset, end=7.0 + offset, file="v_01_x.mov")],
                source=dict(path="/videos/v.mov", width=1280, height=720, fps=30.0, rotation=0, duration=60.0,
                            start_time=fstart + offset, format_start_time=fstart, has_audio=True))


@pytest.mark.parametrize("fstart,offset", [(0.0, 0.5), (1.5, 0.0), (1.5, 0.5)])
def test_cuts_lie_inside_the_available_range(fstart, offset, tmp_path):
    doc = _offset_doc(fstart, offset)
    tl = build_timeline(doc)
    for tr in tl.tracks:
        for c in tr:
            av, sr = c.media_reference.available_range, c.source_range
            assert av.start_time.value == round((fstart + offset) * 30)
            assert av.start_time <= sr.start_time and sr.end_time_exclusive() <= av.end_time_exclusive()
    res, _ = write_edit_lists(doc, tmp_path, list(FORMATS))
    assert all(r["ok"] for r in res), res
