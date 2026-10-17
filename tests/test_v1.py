"""v1 changes: detect default, open-GOP audio/video start, no duplicate libavdevice, predict() fallback, CLI validation."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from inframe import cli, pipeline
from inframe.cut import cut_clip
from inframe.detect import Detection
from inframe.pipeline import Settings, run_cut
from inframe.probe import first_packet_pts, probe

ROOT = Path(__file__).resolve().parents[1]
AAC_FRAME = 1024 / 44100  # 23.2 ms


# 1. detect is the default mode
def test_detect_is_the_default_mode():
    assert cli.build_parser().parse_args(["cut", "v.mp4"]).mode == "detect"
    assert Settings().mode == "detect"
    readme = (ROOT / "README.md").read_text()
    assert "`--mode detect` (the default)" in readme and "--mode motion" in readme


# 2. open-GOP: video and audio start together
def test_open_gop_fixture_has_leading_pictures(open_gop):
    pk = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                                    "-of", "json", open_gop], capture_output=True, text=True, check=True).stdout)["packets"]
    k = [i for i, p in enumerate(pk) if "K" in p["flags"]][1]
    assert float(pk[k + 1]["pts_time"]) < float(pk[k]["pts_time"])  # a leading picture follows the keyframe


@pytest.mark.parametrize("start", [1.2, 2.5, 3.05])
def test_open_gop_clip_video_and_audio_start_together(open_gop, tmp_path, start):
    out = tmp_path / "c.mp4"
    cut_clip(open_gop, out, start, start + 1.5)
    v, a = first_packet_pts(str(out))
    assert a is not None and abs(v - a) < AAC_FRAME, (v, a)
    assert abs(v) < AAC_FRAME  # and the clip starts at zero, not 0.2 s late
    info = probe(str(out))
    assert info.rotation == -90 and info.has_audio and info.codec == "hevc"
    assert info.duration > 1.4


def test_closed_gop_clip_still_aligned(synth, tmp_path):
    out = tmp_path / "c.mp4"
    cut_clip(synth["rot"], out, 1.1, 2.4)
    v, a = first_packet_pts(str(out))
    assert abs(v - a) < AAC_FRAME and probe(str(out)).rotation == -90
    assert list(tmp_path.iterdir()) == [out]  # no temp file left behind


# 3. one libavdevice only
def test_no_opencv_dependency():
    code = "import sys; import inframe.cli, inframe.motion; print('cv2' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.stdout.strip() == "False", r.stdout
    assert "objc[" not in r.stderr, r.stderr
    assert "opencv" not in (ROOT / "pyproject.toml").read_text().lower()


def test_motion_helpers_match_the_opencv_results():
    from inframe import motion as m

    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0] = (255, 0, 0); rgb[0, 1] = (0, 255, 0); rgb[1, 0] = (0, 0, 255); rgb[1, 1] = (255, 255, 255)
    assert m.to_gray(rgb).tolist() == [[76, 150], [29, 255]]  # OpenCV's RGB2GRAY values
    imp = np.zeros((9, 9), np.uint8); imp[4, 4] = 255
    k = np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1])
    assert (m.blur5(imp)[2:7, 2:7] == (255 * k + 128) // 256).all()  # 5x5 binomial, rounded
    assert m.blur5(np.full((6, 7), 200, np.uint8)).min() == 200  # borders reflected, no darkening
    fg = np.zeros((12, 12), np.uint8); fg[1, 1] = 1; fg[4:9, 4:9] = 1; fg[:, 10:] = 1
    op = m.open3(fg)
    assert op[1, 1] == 0 and op[5:8, 5:8].all() and op[4, 4] == 0 and op[:, 11].all()  # speckle and blob corners go, the border column stays
    d = m.grow5(np.pad(np.ones((1, 1), np.uint8), 4))
    assert d.sum() == 17  # OpenCV's 5x5 ellipse has 17 pixels
    rng = np.random.default_rng(3)
    base = rng.random((90, 160)).astype(np.float32) * 255
    moved = np.roll(base, (3, 5), (0, 1))
    (dx, dy), resp = m.phase_shift(base, moved)
    assert abs(abs(dx) - 5) < 0.2 and abs(abs(dy) - 3) < 0.2 and resp > 0.05
    assert m.phase_shift(np.full((90, 160), 50, np.float32), np.full((90, 160), 50, np.float32))[1] >= 0


# 4. a predict() failure in the middle of the run
class FlakyDetector:
    backend, path = "coreml-fake", "fake.mlpackage"

    def __init__(self, fail_on_call=2):
        self.calls, self.fail_on_call, self.n = 0, fail_on_call, 0

    def detect(self, frames):
        self.calls += 1
        if self.calls >= self.fail_on_call:
            raise RuntimeError("MIL program failed: predict() error")
        return _person(self, frames)


class SteadyDetector:
    backend, path = "torch-mps-fp16 (FALLBACK)", "rfdetr checkpoint"

    def __init__(self):
        self.n = 8  # starts where the flaky one stopped

    def detect(self, frames):
        return _person(self, frames)


def _person(det, frames):
    out = []
    for _ in frames:
        out.append([Detection(0.9, (0.1, 0.1, 0.3, 0.5))] if 5 <= det.n <= 9 else [])
        det.n += 1
    return out


def test_predict_failure_falls_back_and_keeps_scoring(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FlakyDetector())
    monkeypatch.setattr(pipeline, "load_fallback_detector", lambda size: SteadyDetector())
    doc = run_cut(synth["rot"], str(tmp_path / "o"), Settings(pad=0.5, min_len=0.5))
    assert len(doc["presence_intervals"]) == 1  # samples 5..9, including the batch that failed
    assert any("predict" in w and "torch" in w for w in doc["warnings"])
    assert len((tmp_path / "o" / "samples.csv").read_text().splitlines()) == 16


def test_predict_failure_without_fallback_saves_partial_scores(synth, tmp_path, monkeypatch):
    def no_fallback(size):
        raise ImportError("torch is not installed")

    monkeypatch.setattr(pipeline, "load_detector", lambda size: FlakyDetector())
    monkeypatch.setattr(pipeline, "load_fallback_detector", no_fallback)
    out = tmp_path / "o"
    with pytest.raises(pipeline.ScoringError, match=r"8 samples.*samples\.csv"):
        run_cut(synth["rot"], str(out), Settings(pad=0.5, min_len=0.5))
    assert len((out / "samples.csv").read_text().splitlines()) == 9  # header + the 8 scored samples
    doc = json.loads((out / "segments.json").read_text())
    assert "predict" in doc["error"] and doc["clips"] == []


# 5. CLI validation: one line, exit code 2, no traceback
@pytest.mark.parametrize("args, word", [
    (["--pad", "-1"], "--pad"),
    (["--fps", "0"], "--fps"),
    (["--fps", "-5"], "--fps"),
    (["--min-len", "-0.5"], "--min-len"),
    (["--enter", "0.4", "--exit", "0.6"], "--exit"),
    (["--exit", "0.9"], "--exit"),  # above the default enter threshold of detect mode
    (["--pad", "nan"], "--pad"),
])
def test_cli_rejects_bad_numbers(synth, tmp_path, monkeypatch, capsys, args, word):
    def boom(*a, **k):
        raise AssertionError("ran although the arguments are invalid")

    monkeypatch.setattr(cli, "run_cut", boom)
    assert cli.main(["cut", synth["plain"], "--out", str(tmp_path / "o"), *args]) == 2
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and err.startswith("inframe:") and word in err and "Traceback" not in err


def test_cli_accepts_valid_numbers(synth, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "run_cut", lambda video, out, s, **k: seen.append(s) or dict(clips=[]))
    assert cli.main(["cut", synth["plain"], "--out", str(tmp_path / "o"), "--pad", "0", "--fps", "2", "--min-len", "0",
                     "--enter", "0.5", "--exit", "0.5"]) == 0
    assert seen[0].pad == 0 and seen[0].fps == 2 and seen[0].min_len == 0


def test_cli_turns_run_errors_into_one_line(synth, tmp_path, monkeypatch, capsys):
    def fail(*a, **k):
        raise pipeline.ScoringError("detector failed after 8 samples")

    monkeypatch.setattr(cli, "run_cut", fail)
    assert cli.main(["cut", synth["plain"], "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err.strip()
    assert err == "inframe: detector failed after 8 samples"
