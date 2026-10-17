"""Fixes after the T1c review: DINOv2 sees the whole person, backend fallback, growth-loop rivals and cost."""
import sys
import types

import numpy as np
import pytest

from inframe import identity
from inframe.detect import Detection
from inframe.identity import PhotoTarget
from inframe.tap import Tap

from test_t1c import BLUE, GREEN, RED, frame_with, photos_dir


def test_dinov2_embeds_the_whole_crop_not_its_centre():
    """The DINOv2 processor resizes the short edge to 256 and centre-crops 224: a tall person crop lost head and legs. The
    worker pads the crop to a square first, so every row of the person reaches the processor."""
    torch = pytest.importorskip("torch")
    from inframe.embed_worker import TorchEmbedder

    seen = []

    def proc(images, return_tensors):
        seen.extend(np.asarray(i) for i in images)
        return {"pixel_values": torch.zeros(len(images), 3, 8, 8)}

    e = object.__new__(TorchEmbedder)
    e.np, e.torch, e.kind, e.backend, e.proc = np, torch, "dinov2", "coreml", proc
    e.ml = types.SimpleNamespace(predict=lambda d: {"o": np.ones((d["x"].shape[0], 4), np.float32)})
    crop = np.zeros((200, 80, 3), np.uint8)
    crop[:, :, 0] = np.arange(200)[:, None]
    crop[:, :, 1] = 7  # marks the person's pixels (the padding is another value)
    e([crop])
    img = seen[0]
    assert img.shape[0] == img.shape[1]  # square: nothing is left for the processor to cut away
    person_rows = img[(img[:, :, 1] == 7).any(1), img.shape[1] // 2, 0]
    assert sorted(person_rows.tolist()) == list(range(200))  # all 200 rows, none lost, none duplicated


def test_auto_backend_falls_back_to_cpu_when_coreml_parity_fails_and_no_mps(monkeypatch):
    torch = pytest.importorskip("torch")
    from inframe import embed_worker

    class Model:
        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self

    fake = types.SimpleNamespace(AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
                                 AutoModel=types.SimpleNamespace(from_pretrained=lambda *a, **k: Model()))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(embed_worker.TorchEmbedder, "_coreml", lambda self: object())

    def parity(self):
        raise RuntimeError("the CoreML package gives other features than torch")

    monkeypatch.setattr(embed_worker.TorchEmbedder, "_check_parity", parity)
    e = embed_worker.TorchEmbedder("dinov2", "auto", offline=True)
    assert e.backend == "cpu" and e.ml is None  # not "coreml" with no model behind it


def test_a_track_that_can_never_be_the_target_does_not_veto_the_next_fragment():
    """Track 1 (confirmed) continues as track 2; a twin (track 3) stood next to the target over the whole run. Track 3 shares frames
    with the accepted track 1, so it is never a candidate, and must not make track 2 'ambiguous'."""
    d = None
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        d = photos_dir(Path(tmp), RED)
        him, twin = (0.05, 0.1, 0.30, 0.9), (0.55, 0.1, 0.80, 0.9)
        t = PhotoTarget(str(d), [Tap(0.6, box=him)], embedder="color")
        for i in range(20):
            tr = 1 if i < 10 else 2
            t.evaluate(round(i * 0.2, 1), frame_with([(RED, him), (RED, twin)]), [Detection(0.9, him, tr), Detection(0.9, twin, 3)])
        hits = t.finalize()
    assert all(h.box == him and h.track_id == 1 for h in hits[:10])
    assert all(h.box == him and h.track_id == 2 for h in hits[10:]), t.info["ambiguous"]
    assert t.info["ambiguous"] == []


def test_growth_loop_without_a_tracker_is_not_quadratic(tmp_path, monkeypatch):
    """Under 5 samples/s every detection is its own unit. The loop used to rescore every remaining unit on every iteration."""
    d = photos_dir(tmp_path, RED)
    boxes = [(0.05, 0.1, 0.30, 0.9), (0.40, 0.1, 0.62, 0.9), (0.70, 0.1, 0.95, 0.9)]
    frame = frame_with(list(zip([RED, BLUE, GREEN], boxes)))
    t = PhotoTarget(str(d), [Tap(0.0, box=boxes[0])], embedder="color")
    n = 300
    for i in range(n):
        t.evaluate(round(i * 0.2, 1), frame, [Detection(0.9, b) for b in boxes])
    calls = []
    real = identity.unit_score
    monkeypatch.setattr(identity, "unit_score", lambda *a, **k: calls.append(1) or real(*a, **k))
    hits = t.finalize()
    assert all(h.box == boxes[0] for h in hits)  # still only the target
    assert len(calls) <= 5 * n * len(boxes), len(calls)  # was about n * n * len(boxes) / 2
