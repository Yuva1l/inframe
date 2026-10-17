"""Python edge cases, part 2: each test failed before its fix."""
import subprocess
from types import SimpleNamespace

import pytest


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-nostats", *args], check=True)


def test_a_class_run_on_a_nano_only_install_says_it_runs_on_nano():
    """model_for returned (nano, None) with only nano installed: the run used a model that misses small objects and nothing said so."""
    from inframe import classes

    size, why = classes.model_for("nano", 18, have=("nano",))
    assert size == "nano"
    assert why and "nano" in why and "small" in why
    assert classes.model_for("nano", classes.PERSON, have=("nano",)) == ("nano", None)  # a person run is as approved
    assert classes.model_for("small", 18, have=("nano", "small")) == ("small", None)


def test_the_pipeline_logs_the_nano_only_class_run(monkeypatch, capsys):
    from inframe import pipeline
    from inframe.target import AnyPerson

    seen = {}

    def stop(size):
        seen["size"] = size
        raise RuntimeError("stop")

    monkeypatch.setattr(pipeline, "available_sizes", lambda: ("nano",))
    monkeypatch.setattr(pipeline, "load_detector", stop)
    tgt = AnyPerson(0.0)
    tgt.cls = 18
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.score_detect("x.mp4", None, pipeline.PRESETS["fast"], None, tgt)
    assert seen["size"] == "nano"
    assert "[detect] class dog" in capsys.readouterr().err


def test_build_reel_opens_each_clip_once_to_pick_its_container(tmp_path, monkeypatch):
    from inframe import reel

    clip = tmp_path / "a.mp4"
    _ff("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-g", "15", "-c:a", "aac", "-shortest", str(clip))
    calls = []
    real = reel.work_format

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(reel, "work_format", counting)
    reel.build_reel([str(clip), str(clip)], tmp_path / "reel.mp4")
    assert len(calls) == 2


def test_probe_error_names_what_the_file_is(tmp_path):
    from inframe import probe

    bad = tmp_path / "x.part.mp4"
    bad.write_bytes(b"junk" * 100)
    with pytest.raises(probe.ProbeError, match="as a video"):
        probe.probe(str(bad))  # the user's input keeps the old wording
    with pytest.raises(probe.ProbeError) as e:
        probe.probe(str(bad), what="the proxy being written")
    assert "as the proxy being written" in str(e.value) and "as a video" not in str(e.value)
    with pytest.raises(probe.ProbeError) as e:
        probe.first_packet_pts(str(bad), what="a cut clip")
    assert "as a cut clip" in str(e.value)


def test_retimed_facts_and_the_proxy_read_back_do_not_call_an_intermediate_a_video(tmp_path):
    from inframe import cut

    bad = tmp_path / "clip.mp4"
    bad.write_bytes(b"junk" * 100)
    with pytest.raises(ValueError) as e:
        cut.retimed_facts(str(bad))
    assert "as a video" not in str(e.value) and "as a clip" in str(e.value)


def test_taps_in_the_real_wrapper_targets_are_checked_for_the_end_of_the_video(synth, tmp_path, monkeypatch):
    """Zone and activity wrappers (the only targets that hold another target) hand `taps` on from the inner target, so
    the check needs no walk through `.inner`."""
    from inframe import pipeline
    from inframe.activity import ActivityTarget
    from inframe.pipeline import Settings, run_cut
    from inframe.tap import Tap, TapTarget
    from inframe.zone import Zone, ZoneTarget

    def boom(size):
        raise AssertionError("the detector was loaded")

    monkeypatch.setattr(pipeline, "load_detector", boom)
    zone = Zone.of("0,0,1,1")
    for wrap in (lambda t: t, lambda t: ZoneTarget(t, zone), lambda t: ActivityTarget(t),
                 lambda t: ActivityTarget(ZoneTarget(t, zone)), lambda t: ZoneTarget(ActivityTarget(t), zone)):
        target = wrap(TapTarget([Tap(99.0, point=(0.5, 0.5))]))
        assert pipeline._nested_taps(target)
        with pytest.raises(ValueError, match="past the end of the video"):
            run_cut(synth["plain"], str(tmp_path / "o"), Settings(), target)


def test_nano_only_note_needs_nano_to_be_the_only_model():
    """model_for('nano', 18, have=('base',)) and have=('nano','base') returned the 'only the nano model is installed' note: a false note, and
    with ('base',) a load of a model that is not installed."""
    from inframe import classes

    size, why = classes.model_for("nano", 18, have=("nano", "base"))
    assert size == "base" and "only the nano model" not in why
    size, why = classes.model_for("nano", 18, have=("base",))
    assert size == "base" and "only the nano model" not in why
    size, why = classes.model_for("nano", 18, have=("nano",))
    assert size == "nano" and "only the nano model is installed" in why
