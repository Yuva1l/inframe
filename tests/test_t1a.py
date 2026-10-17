"""T1a: display flips, audio rate, compare with failed clips, range checks, streaming motion, models, tracks, tools."""
import asyncio
import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from inframe import mcp_server, models, pipeline, tracks
from inframe.cli import main as cli_main
from inframe.cut import cut_clip
from inframe.decode import iter_samples, upright
from inframe.detect import Detection
from inframe.pipeline import Settings, check_ranges, run_cut
from inframe.probe import audio_frame_seconds, probe
from inframe.target import AnyPerson, Hit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import regress  # noqa: E402
import splice  # noqa: E402


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


# ---------------------------------------------------------------- display matrix flips
@pytest.fixture(scope="module")
def asym(tmp_path_factory):
    d = tmp_path_factory.mktemp("asym")
    p = d / "base.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=5:duration=1", "-c:v", "libx264", "-g", "5", "-pix_fmt", "yuv420p", str(p))
    return d, p


@pytest.mark.parametrize("opts", [[], ["-display_rotation", "90"], ["-display_rotation", "-90"], ["-display_rotation", "180"],
                                  ["-display_hflip"], ["-display_vflip"], ["-display_rotation", "90", "-display_hflip"],
                                  ["-display_rotation", "270", "-display_hflip"], ["-display_rotation", "90", "-display_vflip"]])
def test_upright_matches_ffmpeg_for_rotations_and_flips(asym, opts):
    d, base = asym
    v = d / ("v_" + "_".join(o.strip("-") for o in opts) + ".mp4")
    _ff(*opts, "-i", str(base), "-c", "copy", str(v))
    info = probe(str(v))
    ours = next(iter_samples(str(v), info, fps=5.0, long_side=4096, decoder="software")).rgb
    raw = d / "ref.rgb"
    w, h = info.display_size
    # ffmpeg applies the display matrix itself; the first frame of its raw output is the reference
    _ff("-i", str(v), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw))
    ref = np.frombuffer(raw.read_bytes(), np.uint8).reshape(h, w, 3)
    assert ours.shape == ref.shape
    assert np.abs(ours.astype(int) - ref.astype(int)).mean() < 4, (opts, info.rotation, info.mirrored)


def test_mirrored_is_read_from_the_matrix(asym):
    d, base = asym
    for opts, want in (([], False), (["-display_rotation", "90"], False), (["-display_hflip"], True), (["-display_vflip"], True)):
        v = d / f"m_{len(opts)}_{want}_{'_'.join(opts).replace('-', '')}.mp4"
        _ff(*opts, "-i", str(base), "-c", "copy", str(v))
        assert probe(str(v)).mirrored is want
    arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(upright(arr, 0, True), arr[::-1])


# ---------------------------------------------------------------- audio at 48 kHz
def test_cut_at_48khz_keeps_audio_in_sync(tmp_path):
    src = tmp_path / "a48.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30:duration=4", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-shortest", str(src))
    info = probe(str(src))
    assert info.audio_rate == 48000
    assert abs(audio_frame_seconds(str(src)) - 1024 / 48000) < 1e-4
    out = tmp_path / "c.mp4"
    cut_clip(str(src), out, 1.0, 3.0)
    from inframe.probe import first_packet_pts

    v, a = first_packet_pts(str(out))
    assert a is not None and abs(v - a) < 0.025, (v, a)


# ---------------------------------------------------------------- compare, ranges
def test_compare_does_not_count_clips_that_failed_to_cut(tmp_path):
    from inframe.compare import compare_files

    src = dict(path="/v/x.mov", width=10, height=10, duration=100.0, fps=30.0)
    a = dict(mode="detect", source=src, clips=[dict(start=0, end=10, file="a"), dict(start=20, end=30, file="b", error="boom")])
    b = dict(mode="motion", source=src, clips=[dict(start=0, end=10, file="a")])
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    r = compare_files(tmp_path / "a.json", tmp_path / "b.json", out=tmp_path)
    assert r["a"]["kept_s"] == 10 and r["only_a_s"] == 0 and r["a"]["n_clips"] == 1


@pytest.mark.parametrize("flag, val", [("--enter", "1.5"), ("--enter", "-0.1"), ("--exit", "2"), ("--min-box", "1.2"), ("--min-box", "-1"), ("--enter", "nan")])
def test_cli_rejects_out_of_range_numbers(flag, val, tmp_path, capsys):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    assert cli_main(["cut", str(v), flag, val]) == 2
    assert "from 0 to 1" in capsys.readouterr().err


def test_run_cut_rejects_out_of_range_before_any_work(static_box, tmp_path):
    assert check_ranges(0.5, 0.3, 0.0) is None and check_ranges(None, None, 0.0) is None
    with pytest.raises(ValueError, match="--min-box must be a number from 0 to 1"):
        run_cut(static_box, str(tmp_path / "o"), Settings(min_box=3.0))


# ---------------------------------------------------------------- motion keeps memory flat (same scores as the batch version)
def test_motion_streaming_gives_the_scores_the_batch_version_gave(tmp_path):
    p = tmp_path / "m25.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x404040:s=320x180:r=10:d=25,drawgrid=w=40:h=40:t=2:c=white@0.4,noise=alls=4:allf=t,"
        "drawbox=x=120:y=60:w=50:h=70:color=white:t=fill:enable='between(t,12,16)'", "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(p))
    from inframe.motion import run_motion

    r = run_motion(str(p), probe(str(p)), 10.0)
    s = np.asarray(r.scores)
    assert len(s) == 250
    # golden values recorded from the previous implementation, which held every frame in memory
    assert round(float(s.sum()), 4) == pytest.approx(2.691, abs=1e-4)
    assert int(s.argmax()) in range(120, 160) and s[:100].max() < 0.01 < s[125:155].min()


# ---------------------------------------------------------------- models
class _Serve:
    def __init__(self, body: bytes):
        outer = self
        self.hits = 0

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/m.bin"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def test_models_download_checks_the_checksum_and_progress(tmp_path):
    body = b"weights" * 1000
    s = _Serve(body)
    seen = []
    spec = models.ModelFile("m.bin", s.url, "sha256:" + hashlib.sha256(body).hexdigest())
    p = models.fetch(spec, tmp_path, offline=False, progress=lambda d, t, n: seen.append((d, t, n)))
    assert p.read_bytes() == body and seen[-1] == (len(body), len(body), "m.bin")
    n = s.hits
    assert models.fetch(spec, tmp_path, offline=True) == p and s.hits == n  # cached: no network, fine offline
    s.srv.shutdown()


def test_models_bad_checksum_keeps_nothing(tmp_path):
    s = _Serve(b"corrupt")
    spec = models.ModelFile("m.bin", s.url, "md5:" + "0" * 32)
    with pytest.raises(models.ChecksumError, match="discarded"):
        models.fetch(spec, tmp_path, offline=False, progress=None)
    assert list(tmp_path.iterdir()) == []
    s.srv.shutdown()


def test_models_offline_and_unreachable_messages(tmp_path):
    spec = models.ModelFile("m.bin", "http://127.0.0.1:9/m.bin", "md5:" + "0" * 32)
    with pytest.raises(models.OfflineError, match=r"--offline is set.*fetch-models"):
        models.fetch(spec, tmp_path, offline=True)
    with pytest.raises(models.OfflineError, match="could not download"):
        models.fetch(spec, tmp_path, offline=False, progress=None, timeout=2)
    with pytest.raises(ValueError, match="https only"):
        models.fetch(models.ModelFile("x", "file:///etc/passwd", "md5:0"), tmp_path, offline=False)
    assert all(m.url.startswith("https://") and m.checksum.startswith("md5:") for m in models.REGISTRY.values())


def test_cut_offline_without_models_says_so(static_box, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INFRAME_MODELS", str(tmp_path / "none"))
    monkeypatch.setenv("INFRAME_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path / "cache" / "models")
    from inframe import detect

    monkeypatch.setattr(detect, "models_dir", lambda *a: (_ for _ in ()).throw(models.ModelsMissing(models.missing_message("RF-DETR CoreML models"))))
    rc = cli_main(["cut", static_box, "--offline", "--out", str(tmp_path / "o")])
    models.set_offline(False)
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err and "fetch-models" in err


# ---------------------------------------------------------------- tracks
def _boxes(i):
    # two people crossing paths slowly at 5 fps, normalised boxes
    return [Detection(0.9, (0.10 + 0.01 * i, 0.2, 0.25 + 0.01 * i, 0.8)), Detection(0.85, (0.70 - 0.01 * i, 0.2, 0.85 - 0.01 * i, 0.8))]


def test_tracker_keeps_ids_stable_and_never_loads_opencv_here():
    import importlib.util

    if importlib.util.find_spec("trackers") is None:  # the tracking extra is not installed; find_spec does not import it (and its OpenCV)
        pytest.skip("needs the tracking extra")
    frame = np.zeros((90, 160, 3), np.uint8)
    with tracks.Tracker("ocsort", 5.0) as tr:
        outs = [tr.update(i * 0.2, _boxes(i), frame) for i in range(12)]
        ids = [[d.track_id for d in o] for o in outs]
        assert all(None not in x for x in ids[3:])
        assert len({x[0] for x in ids[3:]}) == 1 and len({x[1] for x in ids[3:]}) == 1 and ids[5][0] != ids[5][1]
        assert [d.box for d in outs[4]] == [d.box for d in _boxes(4)]  # boxes and order are untouched
        tr.reset()
        after = [tr.update(i * 0.2, _boxes(i), frame) for i in range(12, 18)]
        assert after[-1][0].track_id is not None  # a fresh tracker confirms the new shot's people again
    assert "cv2" not in sys.modules  # PyAV and OpenCV must not share this process


def test_tracker_needs_5_fps_and_a_known_kind():
    assert tracks.usable(5.0) and tracks.usable(10.0) and not tracks.usable(2.0) and not tracks.usable(None)
    with pytest.raises(ValueError, match="at least 5"):
        tracks.Tracker("ocsort", 2.0)
    with pytest.raises(ValueError, match="tracker must be one of"):
        tracks.tracker_kwargs("deepsort", 5.0)
    with pytest.raises(ValueError, match="tracker must be one of"):
        run_cut("x", "y", Settings(tracker="nope"))


class _Seq:
    """A fake detector: the same two people on every sample."""
    backend, path = "fake", "fake.mlpackage"

    def detect(self, frames):
        return [[Detection(0.9, (0.1, 0.2, 0.3, 0.9)), Detection(0.8, (0.6, 0.2, 0.8, 0.9))] for _ in frames]


class _Follow:
    """Stateful target: locks onto the first track it sees and scores only that track."""
    name, wants_tracks = "follow", True

    def __init__(self):
        self.locked, self.calls = None, []

    def evaluate(self, t, frame, candidates):
        self.calls.append(t)
        if self.locked is None:
            for d in candidates:
                if d.track_id is not None:
                    self.locked = d.track_id
                    break
        for d in candidates:
            if d.track_id is not None and d.track_id == self.locked:
                return Hit(d.score, d.box, d.track_id)
        return Hit(0.0, None)


def test_pipeline_runs_the_tracker_for_a_target_that_wants_it(synth, tmp_path, monkeypatch):
    import importlib.util

    if importlib.util.find_spec("trackers") is None:  # the tracking extra is not installed; find_spec does not import it (and its OpenCV)
        pytest.skip("needs the tracking extra")
    monkeypatch.setattr(pipeline, "load_detector", lambda size: _Seq())
    tg = _Follow()
    doc = run_cut(synth["rot"], str(tmp_path / "o"), Settings(pad=0.5, min_len=0.5), tg)
    assert tg.calls == sorted(tg.calls) and len(tg.calls) == 15
    assert doc["settings"]["tracker"] == "ocsort"
    lines = (tmp_path / "o" / "samples.csv").read_text().splitlines()
    assert lines[0].endswith(",track") and len({l.split(",")[-1] for l in lines[4:]}) == 1  # one id once confirmed
    assert doc["presence_intervals"], "the locked track is in frame for the rest of the video"


def test_no_tracker_below_5_fps_or_for_any_person(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: _Seq())
    started = []
    monkeypatch.setattr(tracks, "Tracker", lambda *a, **k: started.append(a) or (_ for _ in ()).throw(AssertionError("started")))
    doc = run_cut(synth["rot"], str(tmp_path / "a"), Settings(fps=2.0, pad=0.5, min_len=0.5), _Follow())
    assert any("no tracker" in w for w in doc["warnings"])
    run_cut(synth["rot"], str(tmp_path / "b"), Settings(pad=0.5, min_len=0.5), AnyPerson())
    assert not started
    assert not (tmp_path / "b" / "samples.csv").read_text().splitlines()[0].endswith("track")


# ---------------------------------------------------------------- MCP: a cancelled cut frees the lock
def test_cancelled_cut_stops_and_frees_the_lock(static_box, tmp_path, monkeypatch):
    started = threading.Event()

    def slow_run_cut(video, out, settings, target=None, progress=None):
        for i in range(400):
            started.set()
            progress(i / 1000, "working")  # raises once the client has cancelled
            threading.Event().wait(0.02)
        return {}

    monkeypatch.setattr(mcp_server, "run_cut", slow_run_cut)

    class Ctx:
        async def report_progress(self, *a):
            pass

    async def go():
        task = asyncio.create_task(mcp_server.cut_video(static_box, mode="motion", out=str(tmp_path / "o"), ctx=Ctx()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        assert mcp_server._cut_lock.locked()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(100):  # the worker thread stops at its next progress report
            if not mcp_server._cut_lock.locked():
                return True
            await asyncio.sleep(0.05)
        return False

    assert asyncio.run(go())


# ---------------------------------------------------------------- tools
def test_regress_compare():
    base = [(10.0, 25.0), (60.0, 100.0)]
    ok, msg = regress.compare(base, [(10.2, 24.9), (59.8, 100.2)])
    assert ok and "max deviation 0.20" in msg
    ok, msg = regress.compare(base, [(10.4, 25.0), (60.0, 100.0)])
    assert not ok and "beyond 0.3" in msg
    ok, msg = regress.compare(base, [(10.0, 25.0)])
    assert not ok and "baseline has 2" in msg


def test_splice_truth_and_scores():
    if not splice.sources_available():
        pytest.skip("the sample clips for the composites are not configured (docs/TESTDATA.md, inframe.localcfg)")
    for name, recipe in splice.COMPOSITES.items():
        truth = splice.truth_for(name, recipe)
        assert truth["duration"] >= 20 and (truth["target_intervals"] or name == "pan_follow")  # pan_follow has nobody to find: it only counts ID switches
        for seg in truth["segments"]:
            for tl in seg["tiles"]:
                x, y, w, h = tl["rect"]
                assert 0 <= x and 0 <= y and x + w <= splice.W and y + h <= splice.H
    assert len(splice.COMPOSITES) >= 6
    assert any(tl.get("hue") for r in splice.COMPOSITES["outfit_changed"] for tl in r[2])
    truth = splice.truth_for("solo_gaps", splice.COMPOSITES["solo_gaps"])
    assert truth["target_intervals"] == [[0, 10], [18, 28], [36, 46]]
    perfect = splice.score_intervals(truth, [[0, 10], [18, 28], [36, 46]])
    assert perfect["precision"] == 1 and perfect["recall"] == 1
    padded = splice.score_intervals(truth, [[0, 11], [17, 29], [35, 46]])
    assert padded["recall"] == 1 and padded["precision"] == pytest.approx(30 / 34, abs=1e-3)
    assert splice.score_intervals(truth, [])["recall"] == 0


def test_splice_id_switches_counts_changes_per_tile_and_leaks():
    truth = splice.truth_for("two_side", splice.COMPOSITES["two_side"])
    left, right = (0.1, 0.3, 0.2, 0.7), (0.6, 0.3, 0.7, 0.7)  # boxes in the left and right half
    # first segment (0-12 s): him on the left (tile 0), another person on the right
    ok = [(t / 5, [(1, left, 0.9), (2, right, 0.9)]) for t in range(10)]
    assert splice.id_switches(truth, ok)["id_switches"] == 0
    sw = ok[:5] + [(t / 5, [(7, left, 0.9), (2, right, 0.9)]) for t in range(5, 10)]
    r = splice.id_switches(truth, sw)
    assert r["id_switches"] == 1 and r["ids_per_target"] == 2.0 and r["strongest_changes"] == 1
    # a second person tracked steadily next to him costs nothing in the fragmentation count
    both = [(t / 5, [(1, left, 0.9), (5, (0.12, 0.3, 0.19, 0.7), 0.8), (2, right, 0.9)]) for t in range(10)]
    assert splice.id_switches(truth, both)["id_switches"] == 0
    leak = [(0.0, [(1, left, 0.9), (1, right, 0.8)])]
    assert splice.id_switches(truth, leak)["id_leaks"] == 1
    assert splice.id_switches(truth, [(0.0, [(None, left, 0.9)])])["untracked"] == 1


def test_regress_gate_follows_the_default_units(monkeypatch):
    """The approval gate runs the units inframe runs by default (CPU_AND_GPU with the fp32-topk package); an exported
    INFRAME_DETECT_UNITS, --cpu-only and --gpu say otherwise."""
    from inframe import detect

    monkeypatch.delenv("INFRAME_DETECT_UNITS", raising=False)
    assert regress.apply_units(False) == detect.DEFAULT_UNITS == "CPU_AND_GPU"
    assert "INFRAME_DETECT_UNITS" not in os.environ  # nothing is forced: the gate and the tool cannot disagree
    monkeypatch.setenv("INFRAME_DETECT_UNITS", "CPU_AND_NE")
    assert regress.apply_units(False) == "CPU_AND_NE"  # an explicit export wins
    monkeypatch.delenv("INFRAME_DETECT_UNITS", raising=False)
    assert regress.apply_units(True) == "CPU_ONLY"
    monkeypatch.delenv("INFRAME_DETECT_UNITS", raising=False)
    assert regress.apply_units(False, gpu=True) == "CPU_AND_GPU"


def test_regress_accepted_change_moves_only_its_boundary():
    base = [(12.0, 30.0), (70.0, 110.0), (200.0, 215.0)]
    acc = [{"interval": 3, "edge": "start", "approved": 200.0, "accepted": 199.6, "date": "2000-01-01", "reason": "r"}]
    alts, applied, ignored = regress.apply_accepted(base, acc)
    assert alts == {(3, 0): 199.6} and len(applied) == 1 and not ignored
    gpu = [(12.0, 30.0), (70.0, 110.0), (199.6, 215.0)]
    assert regress.compare(base, gpu, 0.3, alts)[0]
    assert not regress.compare(base, gpu)[0]  # without the accepted change the 0.4 s move fails
    assert regress.compare(base, base, 0.3, alts)[0]  # the approved CPU_ONLY run still passes
    ok, msg = regress.compare(base, [(12.0, 30.0), (70.6, 110.0), (199.6, 215.0)], 0.3, alts)  # any other boundary still fails
    assert not ok and "#2 start" in msg
    ok, msg = regress.compare(base, [(12.0, 30.0), (70.0, 110.0), (199.6, 215.6)], 0.3, alts)  # and so does the same interval's end
    assert not ok and "#3 end" in msg
    ok, msg = regress.compare(base, [(12.0, 30.0), (70.0, 110.0), (199.2, 215.0)], 0.3, alts)  # beyond tol of both values
    assert not ok and "#3 start" in msg
    # an entry that names another approved value (the baseline changed) is ignored, not applied
    alts, applied, ignored = regress.apply_accepted(base, [dict(acc[0], approved=210.0)])
    assert alts == {} and not applied and len(ignored) == 1


@pytest.mark.skipif(not regress.ACCEPTED.is_file(), reason="tools/regress_accepted.json is an optional local file")
def test_regress_accepted_file_entries_are_well_formed():
    acc = regress.load_accepted()
    assert acc and all(isinstance(a["interval"], int) and a["interval"] >= 1 and a["edge"] in ("start", "end") for a in acc)
    assert all(a["accepted"] != a["approved"] and a["date"] and a["reason"] for a in acc)
