"""T7 sound cues: onset method, speed estimate, edge snapping, markers, the pipeline path, and a video without audio."""
import json
import subprocess
import wave

import numpy as np
import pytest

from inframe import audio, pipeline
from inframe.detect import Detection
from inframe.editlist import cuts
from inframe.pipeline import Settings, run_cut
from inframe.probe import probe

SR = 44100


def clicks(times, dur, factor=1.0, seed=0):
    """Mono float audio: quiet noise plus a decaying noise burst (an impact) at each time. factor > 1 stretches it in time
    (what baking slow motion into the audio does)."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    y = 0.003 * rng.standard_normal(n)
    m = int(0.12 * SR)
    burst = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.015 * SR)) * (1 - np.exp(-np.arange(m) / (0.003 * SR))) * 0.8  # 3 ms attack, 15 ms decay
    if factor != 1.0:
        burst = np.interp(np.arange(int(m * factor)) / factor, np.arange(m), burst)
    for t in times:
        a = int(t * SR)
        y[a:a + len(burst)] += burst[: n - a]
    return y.astype(np.float32)


def write_wav(path, y):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


def make_video(path, audio_y=None, dur=10, gop=5, fps=10):
    wav = path.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", f"testsrc2=size=160x90:rate={fps}:duration={dur}"]
    if audio_y is not None:
        write_wav(wav, audio_y)
        cmd += ["-i", str(wav), "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return str(path)


def impacts_of(y):
    flux, t = audio.onset_envelope(y[:: 2][: len(y) // 2], SR // 2)
    return audio.onsets(flux, t)


def test_onsets_find_synthetic_clicks_within_a_frame():
    truth = [1.0, 2.5, 4.0, 5.2]
    found = [t for t, s in impacts_of(clicks(truth, 7)) if s >= audio.MIN_STRENGTH]
    assert len(found) == len(truth)
    assert max(abs(f - t) for f, t in zip(found, truth)) < 0.06


def test_speed_estimate_separates_normal_from_baked_slow_motion():
    truth = [1.0, 2.5, 4.0, 5.5, 7.0]
    y = clicks(truth, 9)[:: 2]  # 22.05 kHz like read_audio
    assert audio.estimate_speed(y, audio.onsets(*audio.onset_envelope(y, audio.SR))) == 1.0
    for f in (4, 8):
        yb = clicks([t * f / 4 for t in truth], 9 * f / 4 + 1, factor=f)[:: 2]
        est = audio.estimate_speed(yb, audio.onsets(*audio.onset_envelope(yb, audio.SR)))
        assert est >= audio.BAKED_FROM and abs(est - f) <= f * 0.5, (f, est)


def test_snap_start_before_the_pop_and_end_after_the_landing():
    imp = [(5.2, 4.0), (9.1, 3.0)]
    (s, e, info), = audio.snap_clips([(4.0, 10.0)], imp, [], 0.0, 20.0)
    assert s == pytest.approx(4.7) and e == pytest.approx(9.6)
    assert info == {"start_from": 4.0, "end_from": 10.0}


def test_snap_ignores_far_impacts_and_never_moves_an_edge_more_than_one_second():
    (s, e, info), = audio.snap_clips([(4.0, 10.0)], [(1.0, 5.0), (12.5, 5.0)], [], 0.0, 20.0)
    assert (s, e, info) == (4.0, 10.0, {})
    (s, e, _), = audio.snap_clips([(4.0, 10.0)], [(5.5, 5.0), (8.6, 5.0)], [], 0.0, 20.0)  # target 5.0 / 9.1: both inside the cap
    assert (s, e) == (pytest.approx(5.0), pytest.approx(9.1))
    (s, e, _), = audio.snap_clips([(4.0, 12.0)], [(5.45, 5.0), (10.55, 5.0)], [], 0.0, 20.0)
    assert 0 <= s - 4.0 <= 1.0 and 0 <= 12.0 - e <= 1.0
    for t in np.linspace(2.5, 5.5, 31):
        (s, e, _), = audio.snap_clips([(4.0, 12.0)], [(float(t), 5.0), (float(t) + 6, 5.0)], [], 0.0, 20.0)
        assert abs(s - 4.0) <= 1.0 + 1e-9 and abs(e - 12.0) <= 1.0 + 1e-9


def test_snap_start_sits_on_a_keyframe_before_the_pop_even_with_a_long_gop():
    kfs = [0.0, 3.37, 6.74]  # the clip's real start is up to 0.63 s before the requested one in a stream copy
    (s, e, _), = audio.snap_clips([(3.6, 9.0)], [(4.5, 4.0)], kfs, 0.0, 20.0)
    assert s == pytest.approx(3.37 + 2 / 30 + audio.KEYFRAME_EPS_S) and s < 4.5 - audio.LEAD_S + 0.64
    (s, _e, _), = audio.snap_clips([(3.6, 9.0)], [(4.5, 4.0)], kfs, 0.0, 20.0, frame_s=1 / 60)  # a 60 fps file: two frames are shorter
    assert s == pytest.approx(3.37 + 2 / 60 + audio.KEYFRAME_EPS_S)


def test_snap_keeps_clips_apart_and_in_range():
    out = audio.snap_clips([(0.0, 5.0), (5.5, 10.0)], [(4.6, 4.0), (6.2, 4.0)], [], 0.0, 10.0)
    assert out[0][1] <= out[1][0] and out[0][0] >= 0.0 and out[1][1] <= 10.0


def test_baked_slow_motion_scales_the_search_windows_but_not_the_move_cap():
    (s, e, _), = audio.snap_clips([(10.0, 30.0)], [(14.0, 4.0)], [], 0.0, 60.0, speed=4.0)  # 4 x 1.5 s radius reaches 14.0; lead 2 s, cap stays 1 s
    assert s == pytest.approx(11.0)
    (s, e, info), = audio.snap_clips([(10.0, 30.0)], [(14.0, 4.0)], [], 0.0, 60.0, speed=1.0)
    assert info == {}


def test_clip_cues_raise_confidence_only_with_an_impact():
    imp = [(3.0, 4.0), (9.0, 2.0)]
    assert audio.clip_cues(imp, 2.0, 5.0, 0.7)["confidence"] == pytest.approx(0.9)
    assert audio.clip_cues(imp, 2.0, 5.0, 0.95)["confidence"] == 1.0
    c = audio.clip_cues(imp, 4.0, 8.0, 0.7)
    assert c == dict(impacts=0, markers=[], confidence=0.7)


class Fake:
    backend, path = "fake", "fake.mlpackage"

    def __init__(self):
        self.n = 0

    def detect(self, frames):
        out = []
        for _ in frames:
            out.append([Detection(0.9, (0.1, 0.1, 0.3, 0.5))] if 15 <= self.n <= 29 else [])  # 3.0-6.0 s at 5 samples/s
            self.n += 1
        return out


def run(video, tmp_path, monkeypatch, name, **kw):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: Fake())
    return run_cut(video, str(tmp_path / name), Settings(**kw), None)


def test_pipeline_snaps_to_synthetic_clicks_and_writes_markers(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4", clicks([0.2, 3.2, 6.2, 9.4], 10))
    base = run(video, tmp_path, monkeypatch, "base")
    assert "sound" not in base and "impacts" not in base["clips"][0]  # opt-in: nothing changes by default
    (b,) = base["clips"]
    doc = run(video, tmp_path, monkeypatch, "snap", sound="snap", markers="clap")
    (c,) = doc["clips"]
    info = probe(video)
    # padded clip is about 1.9-6.9 s; pop at 3.2 -> asked start 2.5 + two frames, on the 0.5 s keyframe at 2.5; landing 6.2 -> end 6.7
    assert c["start"] == pytest.approx(2.5, abs=0.25) and c["expected_copy_start"] == pytest.approx(2.5, abs=1e-3) and c["end"] == pytest.approx(6.7, abs=0.1)
    assert abs(c["start"] - b["start"]) <= 1.0 and abs(c["end"] - b["end"]) <= 1.0
    assert c["start"] < 3.2 - 0.1 and c["end"] > 6.2 + 0.1  # starts before the pop, ends after the landing
    assert c["expected_copy_start"] <= 3.2
    assert c["impacts"] == 2 and c["confidence"] == pytest.approx(1.0)
    assert [m["kind"] for m in c["markers"]] == ["clap", "clap"]
    assert [round(m["time"], 1) for m in c["markers"]] == [3.2, 6.2] or all(abs(m["time"] - t) < 0.06 for m, t in zip(c["markers"], (3.2, 6.2)))
    assert c["snapped_from"]["start"] == b["start"]
    assert doc["sound"]["speed"] == 1.0 and doc["sound"]["snap"] is True
    cl = json.loads(next((tmp_path / "snap").glob("*.inframe.json")).read_text())
    assert cl["schema_version"] == "1.2" and cl["sound"]["impacts"] == 2 and cl["sound"]["markers"][0]["clip_time"] == pytest.approx(
        c["markers"][0]["time"] - c["expected_copy_start"], abs=1e-3)
    import jsonschema
    jsonschema.validate(cl, json.load(open(pipeline.Path(__file__).resolve().parents[1] / "docs" / "sidecar.schema.json")))


def test_markers_alone_do_not_move_edges(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4", clicks([3.2, 6.2], 10))
    a = run(video, tmp_path, monkeypatch, "a")
    b = run(video, tmp_path, monkeypatch, "b", markers="clap")
    assert [(c["start"], c["end"]) for c in a["clips"]] == [(c["start"], c["end"]) for c in b["clips"]]
    assert len(b["clips"][0]["markers"]) == 2


def test_markers_in_edit_lists(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4", clicks([3.2, 6.2], 10))
    doc = run(video, tmp_path, monkeypatch, "e", markers="clap", edit_lists=("fcpxml", "otio"))
    assert all(r["ok"] for r in doc["edit_lists"])
    assert sum(len(c["markers"]) for c in cuts(doc)) == 2
    assert (tmp_path / "e" / "v.fcpxml").read_text().count("<marker ") == 2
    assert json.loads((tmp_path / "e" / "v.otio").read_text()).__str__().count("clap") >= 2


def test_video_without_audio_is_skipped_with_a_note(tmp_path, monkeypatch):
    video = make_video(tmp_path / "silent.mp4", None)
    assert not probe(video).has_audio
    doc = run(video, tmp_path, monkeypatch, "s", sound="snap", markers="clap")
    assert len(doc["clips"]) == 1 and "error" not in doc["clips"][0]
    assert any("no audio track" in w for w in doc["warnings"]) and "markers" not in doc["clips"][0] and "sound" not in doc


def test_flag_values_are_checked():
    assert pipeline.check_sound("snap", "clap", "4") is None
    assert "--sound" in pipeline.check_sound("x", "", "auto") and "--markers" in pipeline.check_sound("off", "x", "auto")
    for bad in ("0.5", "fast", "nan"):
        assert "--sound-speed" in pipeline.check_sound("off", "", bad)


@pytest.mark.parametrize("speed", [2.0, 3.0, 4.0, 8.0])
def test_a_misread_speed_never_moves_an_edge_more_than_one_second(speed):
    """The help text promises 'never more than 1 s'; the speed factor widens the search window, not that cap."""
    for pop, land in ((11.0, 26.0), (13.0, 25.0), (9.0, 28.0)):
        (s, e, _), = audio.snap_clips([(10.0, 20.0)], [(pop, 5.0), (land, 5.0)], [], 0.0, 60.0, speed=speed)
        assert abs(s - 10.0) <= 1.0 + 1e-9 and abs(e - 20.0) <= 1.0 + 1e-9, (speed, pop, land, s, e)
    (s, e, _), = audio.snap_clips([(10.0, 20.0)], [(25.0, 5.0)], [], 0.0, 60.0, speed=4.0)
    assert e <= 21.0 + 1e-9


def test_auto_speed_is_capped():
    y = clicks([1.0, 5.0, 9.0, 13.0, 17.0], 22, factor=16)[:: 2]
    est = audio.estimate_speed(y, audio.onsets(*audio.onset_envelope(y, audio.SR)))
    assert est <= audio.MAX_AUTO_SPEED


def test_snap_never_trims_frames_where_the_target_was_seen():
    presence = [(10.2, 19.8)]  # --pad 0.2: the padded clip is 10.0-20.0
    (s, e, info), = audio.snap_clips([(10.0, 20.0)], [(18.6, 5.0)], [], 0.0, 60.0, presence=presence)
    assert e >= 19.8 - 1e-9 and s <= 10.2 + 1e-9
    (s, e, info), = audio.snap_clips([(10.0, 20.0)], [(10.9, 5.0)], [], 0.0, 60.0, presence=presence)  # start would move forward to 10.4 (> 10.2)
    assert s <= 10.2 + 1e-9
    # an edge may still tighten inside the padding, and move outward
    (s, e, _), = audio.snap_clips([(10.0, 20.0)], [(10.4, 5.0), (19.0, 5.0)], [], 0.0, 60.0, presence=[(10.6, 19.2)])
    assert s == pytest.approx(10.0) or s <= 10.6
    (s, e, _), = audio.snap_clips([(10.0, 20.0)], [(18.6, 5.0)], [], 0.0, 60.0, presence=[(10.2, 18.0)])
    assert e == pytest.approx(19.1)
    # two presence intervals in one clip: the end cannot cut into the second
    (s, e, _), = audio.snap_clips([(10.0, 20.0)], [(18.6, 5.0)], [], 0.0, 60.0, presence=[(10.2, 12.0), (18.0, 19.8)])
    assert e >= 19.8 - 1e-9


def test_onset_envelope_is_chunked_and_matches_the_whole_file_stft():
    import tracemalloc
    from scipy.signal import stft
    sr, n_fft, hop = audio.SR, 1024, 256
    y = clicks([3.0, 40.0, 77.0], 120)[:: 2].astype(np.float32)
    tracemalloc.start()
    flux, t = audio.onset_envelope(y)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    _, tr, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    ref = np.concatenate([[0.0], np.maximum(np.diff(np.log1p(100 * np.abs(Z)), axis=1), 0).sum(axis=0)])
    assert len(flux) == len(ref) and np.allclose(flux, ref, atol=1e-6) and np.allclose(t, tr)
    full = Z.shape[0] * Z.shape[1] * 16  # bytes of the complex STFT alone
    assert peak < 0.35 * full, (peak, full)


@pytest.mark.parametrize("speed", [1.0, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("gap", [0.2, 0.5, 0.8])
def test_an_impact_just_outside_a_clip_is_still_included_when_the_move_cap_binds(speed, gap):
    """The 1 s cap may stop an edge short of the usual lead, but never so short that the pop or landing stays cut off."""
    (s, e, _), = audio.snap_clips([(10.0, 20.0)], [(10.0 - gap, 5.0), (20.0 + gap, 5.0)], [], 0.0, 60.0, speed=speed)
    assert s < 10.0 - gap and e > 20.0 + gap, (speed, gap, s, e)
    assert abs(s - 10.0) <= 1.0 + 1e-9 and abs(e - 20.0) <= 1.0 + 1e-9
