"""Edge cases, part 4: each test failed before its fix."""
import copy
import subprocess

import pytest


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


class _Base:
    backend = "fake"
    path = "x.mlpackage"


def test_sliced_detector_copies_and_pickles():
    """__getattr__ forwarded to self.base even while base was not set yet (copy, pickle): infinite recursion."""
    from inframe.detect import SlicedDetector

    d = SlicedDetector(_Base(), (2, 2))
    c = copy.copy(d)
    assert c.base is d.base and c.path == "x.mlpackage"
    with pytest.raises(AttributeError):
        SlicedDetector.__new__(SlicedDetector).base


@pytest.mark.parametrize("v", ["1.0000001", "1.0000004", " 1.0000001 "])
def test_slowmo_just_above_one_is_off(v):
    from inframe.cut import parse_slowmo

    assert parse_slowmo(v) == "off"
    assert parse_slowmo("1.5") == "1.5"


def test_tap_path_with_two_commas_is_a_file(tmp_path):
    """`tap:a,b,c.json` is a target.json whose name has two commas, not T,x,y."""
    import json

    from inframe.tap import parse_spec, recorded_models

    p = tmp_path / "a,b,c.json"
    p.write_text(json.dumps(dict(time=1.0, box=[0.1, 0.1, 0.4, 0.9], model="m", preset="balanced")))
    taps = parse_spec(str(p))
    assert taps[0].t == 1.0 and taps[0].box == (0.1, 0.1, 0.4, 0.9)
    assert parse_spec("1.5,0.5,0.5")[0].point == (0.5, 0.5)
    assert recorded_models([f"tap:{p}"]) == {"m": "balanced"}


def test_ffprobe_failure_is_a_probe_error(tmp_path):
    from inframe.probe import ProbeError, probe

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"junk" * 100)
    with pytest.raises(ProbeError, match="could not read"):
        probe(str(bad))
    with pytest.raises(ValueError, match="could not read"):
        probe(str(tmp_path / "missing.mp4"))


def test_soundtrack_reports_the_gain_it_measured(tmp_path, monkeypatch):
    """When the loudness loop does not converge, gain_db is the gain of the file that was measured (lufs), not the next try."""
    import re

    from inframe import soundtrack

    _ff("-f", "lavfi", "-i", "sine=frequency=330:duration=3", str(tmp_path / "m.wav"))
    seen = []
    real_run = soundtrack._run

    def spy(cmd, what):
        for i, a in enumerate(cmd):
            m = re.match(r"volume=(-?[\d.]+)dB", a)
            if m:
                seen.append(float(m.group(1)))
        return real_run(cmd, what)

    calls = {"n": 0}

    def fake_lufs(path):
        calls["n"] += 1
        return -30.0 if calls["n"] == 1 else -20.0  # never reaches -14

    monkeypatch.setattr(soundtrack, "_run", spy)
    monkeypatch.setattr(soundtrack, "measure_lufs", fake_lufs)
    r = soundtrack.render_soundtrack(tmp_path / "m.wav", tmp_path / "o.m4a", 2.0)
    assert len(seen) == 5 and r["lufs"] == -20.0
    assert r["gain_db"] == round(seen[-1], 2)


# ---------------------------------------------------------------- sidecars
def _sidecar_inputs(tmp_path, n=2):
    src = tmp_path / "v.mov"
    src.write_bytes(b"x" * 10)
    info = dict(path=str(src), duration=20.0, fps=30.0, rotation=0, mirrored=False, width=1280, height=720,
                display_width=1280, display_height=720)
    clips = [dict(index=i + 1, start=5.0 + 10 * i, end=8.0 + 10 * i, file=f"v_{i + 1:02d}_0005-0008.mov", expected_copy_start=4.5 + 10 * i)
             for i in range(n)]
    return info, clips


def _validator():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("validate_sidecars", Path(__file__).resolve().parents[1] / "tools" / "validate_sidecars.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_box_that_clamps_to_nothing_is_no_box(tmp_path):
    from inframe import sidecar

    info, clips = _sidecar_inputs(tmp_path, 1)
    d = sidecar.build_sidecar(info=info, src_hash=dict(algorithm="sha256", value="0" * 64, size_bytes=1), clip=clips[0], n_clips=1,
                              times=[5.0, 6.0], scores=[0.9, 0.9], boxes=[(1.002, 0.4, 1.01, 0.9), (0.2, 0.3, 0.4, 0.8)], track_ids=None,
                              presence=[], settings=dict(fps=5.0), target="any-person", embedder=None, detector={})
    assert [s["box"] for s in d["samples"]] == [None, [0.2, 0.3, 0.4, 0.8]]
    (tmp_path / clips[0]["file"]).write_bytes(b"")
    assert _validator().extra_checks(d, tmp_path) == []


def test_validator_flags_a_sample_outside_the_clip(tmp_path):
    from inframe import sidecar

    info, clips = _sidecar_inputs(tmp_path, 1)
    (tmp_path / clips[0]["file"]).write_bytes(b"")
    d = sidecar.build_sidecar(info=info, src_hash=dict(algorithm="sha256", value="0" * 64, size_bytes=1), clip=clips[0], n_clips=1,
                              times=[5.0, 6.0], scores=[0.9, 0.9], boxes=[None, None], track_ids=None,
                              presence=[], settings=dict(fps=5.0), target="any-person", embedder=None, detector={})
    v = _validator()
    assert v.extra_checks(d, tmp_path) == []
    d["samples"][1].update(source_time=40.0, clip_time=35.5)
    assert any("outside the clip" in b for b in v.extra_checks(d, tmp_path))


def test_sidecar_write_failure_keeps_the_others_and_leaves_no_partial_file(tmp_path, monkeypatch):
    from inframe import sidecar

    info, clips = _sidecar_inputs(tmp_path, 2)
    out = tmp_path / "o"
    out.mkdir()
    doc = dict(source=info, settings=dict(fps=5.0, enter=0.5, exit=0.3), clips=clips, mode="detect")
    real, calls = sidecar.json.dumps, {"n": 0}

    def flaky(obj, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(obj, **kw)

    monkeypatch.setattr(sidecar.json, "dumps", flaky)
    with pytest.raises(OSError, match=r"1 of 2 sidecars not written \(1 written\)"):
        sidecar.write_sidecars(out, doc, times=[5.0], scores=[0.9], boxes=[None], track_ids=None, presence=[], target="any-person",
                               embedder=None, detector=dict(mode="detect"))
    assert sorted(p.name for p in out.iterdir()) == ["v_01_0005-0008.inframe.json"]
    assert clips[0]["sidecar"] == "v_01_0005-0008.inframe.json" and "sidecar" not in clips[1]


# ---------------------------------------------------------------- photos, audio, classes, people
def test_photos_list_sorts_by_instant_not_by_text():
    from types import SimpleNamespace as NS

    from inframe import photos

    def rec(u, d):
        return NS(uuid=u, date=d, slow_mo=False, ismissing=False, path=None)

    class Db:
        def photos(self, **kw):
            return [rec("late", "2026-01-01T09:00:00+00:00"), rec("early", "2026-01-01T10:00:00+02:00")]  # 09:00 UTC, then 08:00 UTC

    assert [r.uuid for r in photos.list_videos(Db())] == ["early", "late"]


def test_compare_tries_with_no_tries_on_either_side_does_not_pass():
    from inframe import photos

    doc = dict(presence_intervals=[], source=dict(fps=240.0))
    assert photos.compare_tries(doc, dict(presence_intervals=[]))["ok"] is False
    one = dict(presence_intervals=[[1.0, 2.0]], source=dict(fps=30.0))
    assert photos.compare_tries(one, dict(presence_intervals=[[1.0, 2.0]]), factor=1.0)["ok"] is True


def test_failed_audio_decode_is_not_reported_as_short_audio(monkeypatch):
    import numpy as np

    from inframe import audio

    monkeypatch.setattr(audio, "read_audio", lambda p: None)
    assert "could not decode" in audio.analyse("x.mp4", 0.0).note
    monkeypatch.setattr(audio, "read_audio", lambda p: np.zeros(100, dtype=np.float32))
    assert "shorter" in audio.analyse("x.mp4", 0.0).note


def test_class_run_stays_on_nano_when_small_is_not_installed():
    from inframe import classes

    assert classes.model_for("nano", 16, have=("nano",))[0] == "nano"  # and says so: tests/test_pylows2.py
    assert classes.model_for("nano", 16, have=("nano", "small"))[0] == "small"


def test_people_scan_fails_on_an_unwritable_folder_before_the_detect_pass(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    from inframe import people

    monkeypatch.setattr(people, "probe", lambda v: NS(duration=1.0))

    def scanned(*a, **k):
        raise AssertionError("the detect pass ran")

    monkeypatch.setattr(people, "score_detect", scanned)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        people.scan_people("v.mp4", blocker / "out")


@pytest.mark.parametrize("kw,flag", [(dict(pad=-1.0), "--pad"), (dict(min_len=float("nan")), "--min-len"), (dict(merge_gap=-0.5), "--merge-gap"),
                                     (dict(fps=-5.0), "--fps"), (dict(pad=float("inf")), "--pad")])
def test_run_cut_refuses_negative_or_non_finite_timing_settings(tmp_path, kw, flag):
    from inframe.pipeline import Settings, run_cut

    with pytest.raises(ValueError, match=flag):
        run_cut(str(tmp_path / "nope.mp4"), str(tmp_path / "o"), Settings(**kw), None)
    assert not (tmp_path / "o").exists()


def test_a_tap_past_the_end_of_the_video_fails_before_the_detect_pass(synth, tmp_path, monkeypatch):
    from inframe import pipeline
    from inframe.pipeline import Settings, run_cut
    from inframe.tap import Tap, TapTarget, parse_spec, TapError

    def boom(size):
        raise AssertionError("the detector was loaded")

    monkeypatch.setattr(pipeline, "load_detector", boom)
    with pytest.raises(ValueError, match="past the end of the video"):
        run_cut(synth["plain"], str(tmp_path / "o"), Settings(), TapTarget([Tap(99.0, point=(0.5, 0.5))]))
    with pytest.raises(ValueError, match="past the end of the video"):
        run_cut(synth["plain"], str(tmp_path / "o"), Settings(), TapTarget([Tap(float("inf"), box=(0.1, 0.1, 0.5, 0.9))]))
    with pytest.raises(TapError):
        parse_spec("inf,0.5,0.5")
    assert not (tmp_path / "o").exists()


def test_ambiguous_messages_do_not_say_track_none(tmp_path):
    import numpy as np
    from PIL import Image

    from inframe.detect import Detection
    from inframe.identity import Ambiguous, PhotoTarget

    def person(h=160, w=60):
        a = np.zeros((h, w, 3), np.uint8)
        a[: h // 2] = (200, 30, 30)
        a[h // 2:] = (25, 25, 30)
        return a

    d = tmp_path / "photos"
    d.mkdir()
    for i in range(3):
        Image.fromarray(person(180, 70 + 4 * i)).save(d / f"p{i}.jpg", quality=95)
    boxes = [(0.05, 0.1, 0.30, 0.9), (0.55, 0.1, 0.80, 0.9)]
    t = PhotoTarget(str(d), embedder="color")
    for i in range(3):
        f = np.full((600, 400, 3), 120, np.uint8)
        for x0, y0, x1, y1 in boxes:
            a, b, c, e = int(x0 * 400), int(y0 * 600), int(x1 * 400), int(y1 * 600)
            f[b:e, a:c] = person(e - b, c - a)
        t.evaluate(round(i * 0.2, 1), f, [Detection(0.9, b, None) for b in boxes])
    with pytest.raises(Ambiguous) as e:
        t.finalize()
    assert "track None" not in str(e.value) and "the detection at" in str(e.value)
