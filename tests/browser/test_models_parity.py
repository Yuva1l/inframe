"""RF-DETR small and DINOv2-S in the browser (onnxruntime-web, WebGPU) against python onnxruntime on the CPU, on the same ONNX files and
the same inputs: LCG noise, and one synthetic ffmpeg testsrc2 frame turned into the model's tensor by Python. Every value of every output
must agree within 1e-3 relative, and the detector's sigmoid scores (all and the top 20) within 2e-3 (parity_compare.py; docs/T9B_DESIGN.md section 8). Chrome always; Safari with INFRAME_SAFARI=1
(it opens the real Safari and the page POSTs its results back, as in the T9-0 spike)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import lcg
from parity_compare import compare

ort = pytest.importorskip("onnxruntime")  # --group browser; a plain checkout skips these instead of failing collection

SHAPES = {"rfdetr": (1, 3, 512, 512), "dino": (1, 3, 224, 224)}
FILES = {"rfdetr": "rfdetr_small_v1.onnx", "dino": "dinov2_s_v1.onnx"}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1), np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def frame_tensor(rgb: np.ndarray, side: int) -> np.ndarray:
    a = np.asarray(Image.fromarray(rgb).resize((side, side), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray((a - MEAN) / STD, dtype=np.float32)


def reference(models_dir: Path, model: str, x: np.ndarray) -> dict:
    s = ort.InferenceSession(str(models_dir / FILES[model]), providers=["CPUExecutionProvider"])
    outs = s.run(None, {s.get_inputs()[0].name: x})
    return {o.name: dict(values=a.astype(np.float64).ravel(), n=int(a.size)) for o, a in zip(s.get_outputs(), outs)}


def inputs(model: str, frame: np.ndarray, data_dir: Path) -> dict:
    n = int(np.prod(SHAPES[model]))
    lc = lcg(n).reshape(SHAPES[model])
    fr = frame_tensor(frame, SHAPES[model][2])
    fr.tofile(data_dir / f"{model}_frame.f32")
    return {"lcg": lc, "frame": fr}


def report(capsys, who: str, got: dict, worst: float):
    placed = {m.group(1): int(m.group(2)) for line in got["cpuLines"] if (m := re.search(r"placed on \[(\w+)\]\.\s+Number of nodes: (\d+)", line))}
    with capsys.disabled():
        print(f"\n[parity {who}] {got['model']}/{got['input']}: provider={got['provider']} worst rel err {worst:.2e}; create {got['createMs']:.0f} ms, run {got['runMs']:.0f} ms; "
              f"nodes per execution provider (ort log) {placed or 'not logged'}; ops with no WebGPU kernel {sorted(set(got['kernelMissing']))[:6]}")


@pytest.mark.parametrize("model", ["rfdetr", "dino"])
def test_chrome_matches_python_onnxruntime(chrome, site, models_dir, models_cached, testsrc2_frame, model, capsys):
    from conftest import open_page

    ins = inputs(model, testsrc2_frame, site.data_dir)
    page = open_page(chrome, site)
    try:
        for kind in ("lcg", "frame"):
            ref = reference(models_dir, model, ins[kind])
            got = page.evaluate("([m, i]) => window.__t.parity(m, i)", [model, kind])
            assert got["provider"] == "webgpu" and got["hasGpu"], got
            report(capsys, "chrome", got, compare(ref, got))
    finally:
        page.close()


def test_models_downloaded_once_and_checked(models_cached, models_dir):
    man = json.loads((models_dir / "manifest.json").read_text())["files"]
    assert models_cached["cachedBytes"] == sum(f["bytes"] for f in man.values())
    assert {f[0] for f in models_cached["files"]} == set(man)


@pytest.mark.skipif(os.environ.get("INFRAME_SAFARI") != "1" or sys.platform != "darwin", reason="Safari parity: set INFRAME_SAFARI=1 (macOS; opens the real Safari)")
def test_safari_matches_python_onnxruntime(site, models_dir, testsrc2_frame, capsys):
    ins = {m: inputs(m, testsrc2_frame, site.data_dir) for m in SHAPES}
    site.posted.clear()
    site.posted_event.clear()
    subprocess.run(["open", "-a", "Safari", site.safari_url()], check=True)
    try:
        assert site.posted_event.wait(900), "Safari did not post a result in 15 minutes"
        res = site.posted[-1]
        assert res.get("ok"), res.get("error") or res
        assert len(res["results"]) == 4
        for got in res["results"]:
            assert got["provider"] == "webgpu"
            report(capsys, "safari", got, compare(reference(models_dir, got["model"], ins[got["model"]][got["input"]]), got))
        with capsys.disabled():
            print(f"\n[parity safari] {res['results'][0]['ua']}")
    finally:
        subprocess.run(["osascript", "-e", f'tell application "Safari" to close (every tab of every window whose URL contains "127.0.0.1:{site.app.port}")'], capture_output=True)
