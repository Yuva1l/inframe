"""T1e: the accurate preset (RF-DETR base, a batch-1 CoreML package) and the compute-unit timing tool."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from inframe import cli, detect, mcp_server
from inframe.pipeline import PRESETS

ROOT = Path(__file__).resolve().parents[1]


def test_accurate_preset_is_rfdetr_base_and_others_unchanged():
    assert PRESETS["accurate"].model == "base" and PRESETS["accurate"].detect_fps == 5.0
    assert (PRESETS["balanced"].model, PRESETS["balanced"].detect_fps) == ("small", 5.0)
    assert (PRESETS["fast"].model, PRESETS["fast"].detect_fps) == ("nano", None)
    assert detect.MODELS["base"] == (560, "RFDETRBase")
    ap = cli.build_parser()
    assert ap.parse_args(["cut", "v.mov", "--preset", "accurate"]).preset == "accurate"
    assert ap.parse_args(["cut", "v.mov"]).preset is None  # resolved later: balanced, or the preset recorded in a target.json (tap.resolve_preset)
    assert ap.parse_args(["pick", "v.mov", "--preset", "accurate"]).preset == "accurate"


def test_package_lookup_prefers_batch_8_then_1(tmp_path):
    for name in ("small_coreml_b8", "base_coreml_b1", "base_coreml_b4"):
        (tmp_path / name / "x.mlpackage").mkdir(parents=True)
    assert detect.package_batch(str(tmp_path / "base_coreml_b1" / "x.mlpackage")) == 1
    assert detect.package_batch("odd/name/x.mlpackage") == 0
    assert "small_coreml_b8" in detect._packages(tmp_path, "small")[0]
    assert "base_coreml_b1" in detect._packages(tmp_path, "base")[0]  # batch 1 before any other batch


def _fake(batch):
    d = object.__new__(detect.CoreMLDetector)
    d.size, d.res, d.batch, d.calls = "base", 32, batch, []

    def run(x):
        d.calls.append(x.shape[0])
        # frame k of the call carries its mean grey level in query 0's box width: lets the test see the order
        boxes = np.zeros((x.shape[0], 1, 4), np.float32)
        boxes[:, 0, :] = [0.5, 0.5, 0.2, 0.2]
        logits = np.full((x.shape[0], 1, detect.PERSON + 1), -10.0, np.float32)
        logits[:, 0, detect.PERSON] = 10.0
        return boxes, logits

    d._run = run
    return d


def test_detect_chunks_to_the_package_batch_and_returns_one_list_per_frame():
    frames = [np.full((20, 30, 3), i, np.uint8) for i in range(11)]
    one, eight = _fake(1), _fake(8)
    r1, r8 = one.detect(frames), eight.detect(frames)
    assert one.calls == [1] * 11 and eight.calls == [8, 8]  # the last chunk of 3 is padded to 8
    assert len(r1) == len(r8) == 11 and all(len(f) == 1 for f in r1 + r8)
    assert r1 == r8


def test_mcp_and_pick_follow_the_preset(monkeypatch):
    seen = []
    monkeypatch.setattr(mcp_server.identity, "lazy_detect", lambda size="small": seen.append(size) or (lambda imgs: []))
    monkeypatch.setattr(mcp_server.identity, "parse_targets", lambda *a, **k: "T")
    assert mcp_server._parse_target("tap:1,0.5,0.5", "detect", "accurate") == "T"
    assert mcp_server._parse_target("tap:1,0.5,0.5", "detect") == "T"
    assert seen == ["base", "small"]


def test_bench_units_tool_reports_missing_video_as_a_row():
    r = subprocess.run([sys.executable, str(ROOT / "tools/bench_units.py"), "--model", "base", "--units", "CPU_ONLY", "--video", "/nonexistent.mov"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and json.loads(r.stdout)["error"].startswith("no such video")
    assert (ROOT / "tools/bench_units.sh").stat().st_mode & 0o111


class _Spec:
    def __init__(self, shape):
        inp = type("I", (), {})()
        inp.name = "x"
        inp.type = type("T", (), {"multiArrayType": type("M", (), {"shape": shape})()})()
        self.description = type("D", (), {"input": [inp]})()


class _Model:
    def __init__(self, shape):
        self.shape = shape

    def get_spec(self):
        return _Spec(self.shape)


def test_batch_comes_from_the_package_input_shape_not_the_folder_name(monkeypatch):
    monkeypatch.setattr(detect, "find_package", lambda size: "/m/small_coreml_b8/x.mlpackage")
    monkeypatch.setattr(detect.CoreMLDetector, "_load", lambda self: _Model([4, 3, 512, 512]))
    monkeypatch.setattr(detect.CoreMLDetector, "_arm", lambda self: None)
    assert detect.CoreMLDetector("small", "CPU_ONLY").batch == 4
    monkeypatch.setattr(detect, "find_package", lambda size: "/m/odd/x.mlpackage")  # name says nothing
    assert detect.CoreMLDetector("small", "CPU_ONLY").batch == 4


def test_gpu_load_failure_retries_coreml_on_cpu_only_before_torch(monkeypatch, capsys):
    loads = []

    def load(self):
        loads.append(self.units)
        if self.units == "CPU_AND_GPU":
            raise RuntimeError("no gpu")
        return _Model([8, 3, 512, 512])

    monkeypatch.setattr(detect, "find_package", lambda size: "/m/small_coreml_b8/x.mlpackage")
    monkeypatch.setattr(detect.CoreMLDetector, "_load", load)
    monkeypatch.setattr(detect.CoreMLDetector, "_arm", lambda self: None)
    monkeypatch.setattr(detect, "load_fallback_detector", lambda size: (_ for _ in ()).throw(AssertionError("went to torch")))
    d = detect.load_detector("small", "CPU_AND_GPU")
    assert d.units == "CPU_ONLY" and loads == ["CPU_AND_GPU", "CPU_ONLY"]
    assert "using CPU_ONLY" in capsys.readouterr().err
