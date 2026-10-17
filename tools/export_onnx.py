"""Export the two browser-mode models to ONNX and write $INFRAME_MODELS/onnx/manifest.json (docs/T9B_DESIGN.md section 3).

Run it in a bench-only venv, never in the project's (rfdetr 1.5.2, torch, transformers, onnx and onnxruntime are only needed here):

    uv venv --python 3.12 ~/.cache/inframe/export-venv
    uv pip install -p ~/.cache/inframe/export-venv/bin/python "rfdetr==1.5.2" onnx onnxruntime onnxsim transformers torch numpy
    INFRAME_MODELS=... ~/.cache/inframe/export-venv/bin/python tools/export_onnx.py [--wasm-from web/node_modules/onnxruntime-web/dist]

It writes rfdetr_small_v1.onnx (RF-DETR small, COCO-91 logits, 512 x 512 input), dinov2_s_v1.onnx (DINOv2-S, 224 x 224 input,
pooler_output) and a copy of the onnxruntime-web asyncify .wasm, then manifest.json with the bytes, sha256, license and source of each file.
The manifest is the allowlist `inframe serve` serves under /models/. The RF-DETR checkpoint is read from ~/.roboflow/models
(bench/detect/fetch_weights.py downloads it, md5-checked); DINOv2 comes from the Hugging Face cache (facebook/dinov2-small).
Weights are fp32 (fp16 was not evaluated in T9b part 3; it would halve the download if tests/browser/test_models_parity.py still held).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

RFDETR_CKPT = Path("~/.roboflow/models/rf-detr-small.pth").expanduser()
WASM = "ort-wasm-simd-threaded.asyncify.wasm"
FILES = {
    "rfdetr_small_v1.onnx": dict(
        role="detector", input="input", shape=[1, 3, 512, 512], license="Apache-2.0",
        source="https://github.com/roboflow/rf-detr (RF-DETR small, COCO checkpoint, rfdetr 1.5.2 export)"),
    "dinov2_s_v1.onnx": dict(
        role="embedder", input="pixel_values", shape=[1, 3, 224, 224], license="Apache-2.0",
        source="https://huggingface.co/facebook/dinov2-small (pooler_output, opset 17)"),
    WASM: dict(
        role="runtime", license="MIT", source="https://www.npmjs.com/package/onnxruntime-web (1.30.0, WebGPU build, ort.webgpu.bundle)"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def export_rfdetr(dst: Path) -> None:
    from rfdetr import RFDETRSmall

    tmp = dst.parent / "_rfdetr_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    RFDETRSmall(pretrain_weights=str(RFDETR_CKPT)).export(output_dir=str(tmp), simplify=False)
    shutil.move(str(tmp / "inference_model.onnx"), dst)
    shutil.rmtree(tmp, ignore_errors=True)


def export_dino(dst: Path) -> None:
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(m, (x,), str(dst), input_names=["pixel_values"], output_names=["last_hidden_state", "pooler_output"],
                      opset_version=17, dynamo=False)


def describe(p: Path) -> dict:
    import onnxruntime as ort

    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    return dict(inputs=[dict(name=i.name, shape=i.shape) for i in s.get_inputs()], outputs=[o.name for o in s.get_outputs()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="models folder (default $INFRAME_MODELS); files go to <out>/onnx")
    ap.add_argument("--wasm-from", default="web/node_modules/onnxruntime-web/dist", help="folder holding " + WASM)
    a = ap.parse_args()
    base = a.out or os.environ.get("INFRAME_MODELS")
    if not base:
        sys.exit("set INFRAME_MODELS or pass --out")
    out = Path(base).expanduser() / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "rfdetr_small_v1.onnx").is_file():
        export_rfdetr(out / "rfdetr_small_v1.onnx")
    if not (out / "dinov2_s_v1.onnx").is_file():
        export_dino(out / "dinov2_s_v1.onnx")
    wasm = Path(a.wasm_from) / WASM
    if not wasm.is_file():
        sys.exit(f"{wasm} not found (npm --prefix web ci)")
    if not (out / WASM).is_file() or sha256(out / WASM) != sha256(wasm):
        shutil.copyfile(wasm, out / WASM)
    files = dict(FILES)
    man = {}
    for name, meta in files.items():
        p = out / name
        man[name] = dict(meta, bytes=p.stat().st_size, sha256=sha256(p), **(describe(p) if name.endswith(".onnx") else {}))
    (out / "manifest.json").write_text(json.dumps(dict(format=1, files=man), indent=1, sort_keys=True) + "\n")
    for n, m in man.items():
        print(f"{n}  {m['bytes']:>11}  {m['sha256'][:16]}  {m['license']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
