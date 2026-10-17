#!/usr/bin/env bash
# One command: ms per crop for DINOv2-S and SigLIP 2 base on CoreML, MPS and CPU (synthetic crops; see tools/bench_embed.py).
# The first run downloads the two models (Apache-2.0, a few hundred MB) and exports the CoreML packages into the model cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1
exec uv run --extra tracking --extra identity python tools/bench_embed.py "$@"
