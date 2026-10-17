"""Tap targets in Chrome (docs/T9B_DESIGN.md sections 5 and 8). Skipped unless INFRAME_BROWSER_TESTS=1 (see conftest.py).

1. DINOv2-S embeddings made in the browser (the page's own person_crop, pad_square, bicubic resize, centre crop, normalisation and onnxruntime-web on
   WebGPU) of 5 boxes of one synthetic ffmpeg testsrc2 frame (tall and wide) against the Python path on the same ONNX file: person_crop and
   pad_square from the product code, the Hugging Face image processor of facebook/dinov2-small, python onnxruntime on the CPU, pooler_output
   L2-normalised. Each cosine must be 0.99 or more.
2. A tap run on synthetic HEVC video with no person in it ends with the TapError text, not a crash, and leaves the engine ready for the next run.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import open_page
from inframe.embed_worker import pad_square
from inframe.identity import person_crop

ort = pytest.importorskip("onnxruntime")  # --group browser; a plain checkout skips these instead of failing collection

# tall, tall and narrow, wide, wide and short, near-square: all large enough to be embedded at 640 x 360
BOXES = [
    [0.10, 0.05, 0.30, 0.95],
    [0.55, 0.10, 0.65, 0.90],
    [0.05, 0.40, 0.70, 0.80],
    [0.30, 0.55, 0.95, 0.75],
    [0.40, 0.20, 0.80, 0.85],
]


@pytest.fixture(scope="module")
def processor():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
    except OSError:
        pytest.skip("facebook/dinov2-small is not in the local Hugging Face cache")


def python_embeddings(models_dir, processor, frame, boxes):
    """embed_worker's path on CPU: pad_square, then the processor, then the exported model's pooler_output, L2-normalised."""
    s = ort.InferenceSession(str(models_dir / "dinov2_s_v1.onnx"), providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    out = []
    for b in boxes:
        crop = person_crop(frame, tuple(b))
        assert crop is not None, b
        px = processor(images=[Image.fromarray(pad_square(crop))], return_tensors="np")["pixel_values"].astype(np.float32)
        f = s.run(None, {s.get_inputs()[0].name: px})[names.index("pooler_output")][0]
        out.append(f / (np.linalg.norm(f) + 1e-9))
    return np.array(out, np.float32)


def test_browser_embeddings_match_the_python_processor_path(chrome, site, models_dir, models_cached, processor, testsrc2_frame, capsys):
    h, w = testsrc2_frame.shape[:2]
    (site.data_dir / "frame.rgb").write_bytes(np.ascontiguousarray(testsrc2_frame).tobytes())
    page = open_page(chrome, site)
    try:
        got = page.evaluate("([w, h, b]) => window.__t.embedCrops(w, h, b)", [w, h, BOXES])
    finally:
        page.close()
    assert got["provider"] == "webgpu"
    web = got["embeddings"]
    assert len(web) == 5 and all(e is not None for e in web)
    ref = python_embeddings(models_dir, processor, testsrc2_frame, BOXES)
    cos = [float(np.dot(np.array(e, np.float32), r)) for e, r in zip(web, ref)]
    with capsys.disabled():
        print(f"\n[embed parity chrome] cosines {', '.join(f'{c:.5f}' for c in cos)}; {got['ms']:.0f} ms for 5 crops (crop, resize and model)")
    assert min(cos) >= 0.99, cos
    assert np.allclose([np.linalg.norm(e) for e in web], 1.0, atol=1e-4)


def test_tap_run_without_a_person_ends_with_the_message(chrome, site, models_cached, hevc20, capsys):
    page = open_page(chrome, site)
    chrome.problems.clear()
    try:
        page.set_input_files("#file", str(hevc20))
        r = page.evaluate("(o) => window.__t.run(o)", {"kind": "tap", "tap": {"t": 7.5, "box": [0.2, 0.1, 0.5, 0.9]}})
        with capsys.disabled():
            print(f"\n[tap run] {r['state']}: {r['error']}")
        assert r["state"] == "failed", r
        assert r["error"].startswith("no person detected within 1 s of the tap at 7.5 s (nothing at that box)"), r["error"]
        assert r["stillRunning"] is False
        assert chrome.problems == [], chrome.problems
        again = page.evaluate("(o) => window.__t.run(o)", {})  # the engine and the Worker are fine afterwards
        assert again["state"] == "done", again
    finally:
        page.close()
