"""T9a browser findings: ui-3 the zone warning must read right in the UI, where there is no --zone flag."""
from inframe.pipeline import Settings, run_cut


def test_zone_warning_does_not_name_the_cli_flag(shaky, tmp_path):
    doc = run_cut(shaky, str(tmp_path / "o"), Settings(mode="motion", zone=(0.0, 0.0, 0.5, 0.5)))
    msgs = [w for w in doc["warnings"] if "static" in w and "zone" in w]
    assert msgs, doc["warnings"]
    assert all("--zone" not in w for w in msgs), msgs
    assert any("the zone is ignored" in w for w in msgs), msgs


# rrb-1: the other warnings the UI can trigger (activity, slowmo, enter) must not name CLI flags either
import pytest  # noqa: E402

from inframe import pipeline  # noqa: E402
from inframe.detect import Detection  # noqa: E402


class _OnePerson:
    backend, path = "fake", "fake.mlpackage"

    def __init__(self, score=0.9):
        self.score = score

    def detect(self, frames):
        return [[Detection(self.score, (0.1, 0.1, 0.3, 0.5), None)] for _ in frames]


class _SameTrack:
    def __init__(self, *a, **k):
        pass

    def update(self, t, dets, frame):
        from dataclasses import replace
        return [replace(d, track_id=1) for d in dets]

    def reset(self):
        pass

    def close(self):
        pass


def _warnings(video, tmp_path, monkeypatch, score=0.9, **kw):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: _OnePerson(score))
    monkeypatch.setattr(pipeline.tracks, "Tracker", _SameTrack)
    return run_cut(video, str(tmp_path / "o"), Settings(pad=0.0, min_len=1.0, **kw))["warnings"]


def test_activity_camera_warning_does_not_name_the_cli_flag(shaky, tmp_path, monkeypatch):
    ws = _warnings(shaky, tmp_path, monkeypatch, activity="riding")
    msgs = [w for w in ws if "static" in w and "speed" in w]
    assert msgs, ws
    assert all("--" not in w for w in msgs), msgs


def test_activity_not_applied_warning_does_not_name_the_cli_flag(static_box, tmp_path, monkeypatch):
    ws = _warnings(static_box, tmp_path, monkeypatch, activity="riding", fps=1.0)
    msgs = [w for w in ws if "not applied" in w]
    assert msgs, ws
    assert all("--" not in w for w in msgs), msgs


def test_no_stretch_warning_does_not_name_the_cli_flag(static_box, tmp_path, monkeypatch):
    ws = _warnings(static_box, tmp_path, monkeypatch, score=0.3)
    msgs = [w for w in ws if "no stretch found" in w]
    assert msgs, ws
    assert all("--" not in w for w in msgs), msgs


def test_slowmo_not_retimed_warning_does_not_name_the_cli_flag(synth, tmp_path, monkeypatch):
    ws = _warnings(synth["plain"], tmp_path, monkeypatch, slowmo="auto")
    msgs = [w for w in ws if "not retimed" in w]
    assert msgs, ws
    assert all("--" not in w for w in msgs), msgs
