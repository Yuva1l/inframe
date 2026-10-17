"""Child process for identity.Embedder: runs DINOv2-S or SigLIP 2 base (torch + transformers, `uv sync --extra identity`)
and talks pickle over stdin/stdout, like track_worker.py. The parent never imports torch.

Backends: "coreml" (the model traced and converted once into the model cache, then run by CoreML on the Neural Engine / GPU),
"mps" (torch on the Apple GPU) and "cpu". "auto" takes coreml when the package exists or can be built, else mps, else cpu.
tools/bench_embed.sh re-runs the spike that chose the default.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

REPOS = {"dinov2": "facebook/dinov2-small", "siglip2": "google/siglip2-base-patch16-224"}
BACKENDS = ("auto", "coreml", "mps", "cpu")
COREML_BATCH = 8  # the exported package takes exactly 8 crops per call (a flexible batch fails to convert); the last call is padded


def cache_dir() -> Path:
    base = os.environ.get("INFRAME_CACHE")
    root = Path(base) if base else (Path.home() / "Library/Caches/inframe" if sys.platform == "darwin" else Path.home() / ".cache/inframe")
    return root / "models"


PAD_RGB = (124, 116, 104)  # the ImageNet mean the processor subtracts: padding that carries no signal


def pad_square(crop):
    """The (H, W, 3) crop centred on a square of PAD_RGB, so a centre-cropping processor keeps all of the person."""
    import numpy as np

    h, w = crop.shape[:2]
    if h == w:
        return crop
    s = max(h, w)
    out = np.empty((s, s, 3), crop.dtype)
    out[:] = PAD_RGB
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = crop
    return out


class TorchEmbedder:
    """embedder(list of (H, W, 3) uint8 arrays) -> (n, dim) float32, L2-normalised rows."""

    def __init__(self, kind: str, backend: str = "auto", offline: bool = False):
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if kind not in REPOS:
            raise ValueError(f"unknown embedder {kind!r}; known: {', '.join(REPOS)}")
        self.kind, self.np, self.torch = kind, np, torch
        kw = dict(local_files_only=True) if offline else {}
        try:
            self.proc = AutoImageProcessor.from_pretrained(REPOS[kind], **kw)
            self.model = AutoModel.from_pretrained(REPOS[kind], **kw).eval().requires_grad_(False)
        except OSError as e:
            raise RuntimeError(f"the {kind} model ({REPOS[kind]}) is not in the cache and cannot be downloaded"
                               f"{' (--offline)' if offline else ''}: {str(e)[:200]}") from None
        self.backend, self.ml = "cpu", None
        want = ["coreml", "mps"] if backend == "auto" else [backend]
        for b in want:
            try:
                if b == "coreml":
                    self.ml = self._coreml()
                    self.backend = "coreml"
                    self._check_parity()
                elif b == "mps" and torch.backends.mps.is_available():
                    self.model = self.model.to("mps")
                    self.backend = "mps"
                elif b == "cpu":
                    self.backend = "cpu"
                else:
                    continue
                break
            except Exception as e:  # export or device failure: the next backend takes over
                print(f"[embed_worker] {b} unavailable for {kind}: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
                self.ml, self.backend = None, "cpu"  # a failed export or parity check must not leave the name of a backend with no model
        else:
            if backend != "auto":
                raise RuntimeError(f"backend {backend} is not available for {kind}")

    # -- the model's own forward: DINOv2 pooled CLS token, SigLIP 2 image-tower embedding
    def _features(self, px):
        torch = self.torch
        if self.kind == "dinov2":
            return self.model(pixel_values=px).pooler_output
        f = self.model.get_image_features(pixel_values=px)
        return f if torch.is_tensor(f) else f.pooler_output

    def _coreml(self):
        import coremltools as ct
        from coremltools.converters.mil.frontend.torch import ops as mil_ops

        torch = self.torch
        if not getattr(mil_ops, "_inframe_cast_patched", False):  # coremltools 9 + numpy 2: int(np.array([n])) raises; take the scalar first
            orig_cast = mil_ops._cast

            def _cast(context, node, dtype, dtype_name):
                class Meta(type):  # isinstance(x, dtype) stays true for the real type; calling it takes the scalar of a 1-element array
                    def __call__(cls, v):
                        return dtype(v.reshape(-1)[0] if hasattr(v, "reshape") else v)

                    def __instancecheck__(cls, obj):
                        return isinstance(obj, dtype)

                return orig_cast(context, node, Meta("Scalar", (), {}), dtype_name)

            mil_ops._cast = _cast
            mil_ops._inframe_cast_patched = True
        path = cache_dir() / f"{self.kind}-b{COREML_BATCH}.mlpackage"
        if not path.exists():
            kind, feats = self.kind, self._features

            class Wrap(torch.nn.Module):
                def forward(self, x):
                    return feats(x)

            size = self.proc.size.get("height") or self.proc.crop_size["height"]
            ex = torch.zeros(COREML_BATCH, 3, size, size)
            emb = getattr(self.model, "embeddings", None)
            if kind == "dinov2" and hasattr(emb, "interpolate_pos_encoding"):  # CoreML has no bicubic upsample: the input size is fixed, so do it once
                with torch.no_grad():
                    pe = emb.interpolate_pos_encoding(emb.patch_embeddings(ex), size, size).clone()
                emb.interpolate_pos_encoding = lambda embeddings, height, width: pe
            with torch.no_grad():
                traced = torch.jit.trace(Wrap(), ex, strict=False)
            ml = ct.convert(traced, inputs=[ct.TensorType("x", shape=(COREML_BATCH, 3, size, size))],
                            convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14,
                            compute_precision=ct.precision.FLOAT32 if kind == "siglip2" else ct.precision.FLOAT16)  # SigLIP 2 loses its features in fp16
            path.parent.mkdir(parents=True, exist_ok=True)
            ml.save(str(path))
        return ct.models.MLModel(str(path))

    def _check_parity(self) -> None:
        """A converted package must give the torch features (cosine 0.99 or more) on a probe batch, or the backend is refused."""
        np = self.np
        probe = [np.random.default_rng(1).integers(0, 256, (256, 128, 3), dtype=np.uint8) for _ in range(2)]
        got = self(probe)
        self.backend = "cpu"
        ref = self(probe)
        self.backend = "coreml"
        cos = float((got * ref).sum(1).min())
        if cos < 0.99:
            raise RuntimeError(f"the CoreML package gives other features than torch (cosine {cos:.3f})")

    def __call__(self, crops):
        np, torch = self.np, self.torch
        from PIL import Image

        if self.kind == "dinov2":  # its processor resizes the short edge to 256 and centre-crops 224: a tall crop would lose head and legs
            crops = [pad_square(c) for c in crops]
        imgs = [Image.fromarray(c) for c in crops]
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"]
        if self.backend == "coreml":
            out = []
            for i in range(0, len(px), COREML_BATCH):
                chunk = px[i:i + COREML_BATCH].numpy()
                n = len(chunk)
                if n < COREML_BATCH:
                    chunk = np.concatenate([chunk, np.zeros((COREML_BATCH - n, *chunk.shape[1:]), chunk.dtype)])
                r = self.ml.predict({"x": chunk})
                out.append(np.asarray(next(iter(r.values())), np.float32)[:n])
            f = np.concatenate(out)
        else:
            dev = "mps" if self.backend == "mps" else "cpu"
            with torch.no_grad():
                f = self._features(px.to(dev)).float().cpu().numpy()
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)


def main() -> int:
    proto_in, proto_out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # anything the libraries print must not corrupt the protocol

    def send(obj) -> None:
        pickle.dump(obj, proto_out, protocol=4)
        proto_out.flush()

    emb = None
    while True:
        try:
            msg = pickle.load(proto_in)
        except EOFError:
            return 0
        try:
            if msg[0] == "init":
                _, kind, backend, offline = msg
                emb = TorchEmbedder(kind, backend, offline)
                send(("ready", dict(backend=emb.backend)))
            elif msg[0] == "embed":
                send(("vec", emb(msg[1])))
            elif msg[0] == "close":
                send(("bye", None))
                return 0
        except Exception as e:
            send(("error", f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
    raise SystemExit(main())
