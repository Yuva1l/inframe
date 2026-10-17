#!/usr/bin/env bash
# Builds the CoreML detector packages the pipeline runs (nano = fast, small = balanced, base = accurate) into bench/detect/out/models/,
# or into $DETECT_MODELS. Idempotent: venv, deps, weights (resumable, md5-checked), then one export per size with its topk in fp32.
# No frames and no benchmark: that is setup.sh. Point INFRAME_MODELS at the output folder to use the packages.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out/models
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
nice -n 10 uv pip install -q -p .venv/bin/python "rfdetr[coreml,onnx]" onnxruntime onnx pillow numpy
.venv/bin/python fetch_weights.py rf-detr-nano.pth rf-detr-small.pth rf-detr-base.pth
for size in nano small base; do
  nice -n 10 .venv/bin/python export_gpu.py "$size"
done
echo MODELS_DONE
