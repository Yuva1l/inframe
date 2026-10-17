#!/usr/bin/env python3
"""Embedder timing (T1c spike): ms per crop for DINOv2-S and SigLIP 2 base on CoreML, MPS and CPU. Synthetic crops only.

    tools/bench_embed.sh [--batch 16] [--iters 8] [--kinds dinov2 siglip2] [--backends coreml mps cpu]

Prints one line per (embedder, backend): ms per crop, including the image processor, for a batch of crops of 256x128
(a person box). A backend that fails prints why and the run goes on. Run it on a quiet machine: the number moves with load.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--kinds", nargs="*", default=["dinov2", "siglip2"])
    ap.add_argument("--backends", nargs="*", default=["coreml", "mps", "cpu"])
    a = ap.parse_args()
    from inframe.embed_worker import TorchEmbedder

    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (256, 128, 3), dtype=np.uint8) for _ in range(a.batch)]
    print(f"load average {os.getloadavg()[0]:.2f} {os.getloadavg()[1]:.2f} {os.getloadavg()[2]:.2f}; batch {a.batch}, {a.iters} timed calls after 2 warm-up calls")
    ref_cache: dict = {}
    for kind in a.kinds:
        for be in a.backends:
            try:
                t0 = time.perf_counter()
                emb = TorchEmbedder(kind, be)
                if emb.backend != be:
                    raise RuntimeError(f"fell back to {emb.backend}")
                load_s = time.perf_counter() - t0
                for _ in range(2):
                    emb(crops)
                ts = []
                for _ in range(a.iters):
                    t0 = time.perf_counter()
                    emb(crops)
                    ts.append((time.perf_counter() - t0) * 1000 / a.batch)
                ref = ref_cache.setdefault(kind, TorchEmbedder(kind, "cpu")(crops[:4]))
                cos = float((emb(crops[:4]) * ref).sum(1).min()) if be != "cpu" else 1.0  # worst cosine to the CPU features
                print(f"{kind:8s} {be:7s} {statistics.median(ts):7.2f} ms per crop (min {min(ts):.2f}, max {max(ts):.2f}; load {load_s:.1f} s; "
                      f"cosine to cpu {cos:.4f})", flush=True)
            except Exception as e:
                print(f"{kind:8s} {be:7s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
