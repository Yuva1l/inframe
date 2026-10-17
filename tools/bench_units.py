#!/usr/bin/env python3
"""Time the detector on the first N seconds of a video for one model and one set of CoreML compute units (one process, one JSON line).

    uv run python tools/bench_units.py --model base --units CPU_ONLY [--video V] [--seconds 60] [--fps 5]

Use tools/bench_units.sh to run the whole grid {small, base} x {CPU_ONLY, CPU_AND_GPU, CPU_AND_NE} and print the table.
The video is read in place and decoded into memory; no frame is written or shown. Reported: model load time, ms per frame (decode excluded,
preprocessing and the model call included, as the pipeline pays it) of two identical passes over the same frames, and whether the two
passes gave bit-identical scores and boxes (the repeatability that decides CPU_ONLY against the faster units), plus the load average
before and after (a busy machine makes the ms per frame meaningless).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("INFRAME_MODELS", str(ROOT / "bench/detect/out/models"))
from inframe import localcfg  # noqa: E402

VIDEO = localcfg.get("regress_video")  # the configured regression video; --video overrides


def passes(det, frames: list, batch: int) -> tuple[float, list]:
    """(seconds, detections per frame) of one pass over `frames` in batches, the way the pipeline calls the detector."""
    t = time.perf_counter()
    out = []
    for i in range(0, len(frames), batch):
        out += det.detect(frames[i:i + batch])
    return time.perf_counter() - t, [[(d.score, d.box) for d in f] for f in out]


MATCH_IOU = 0.5  # two detections of the two passes are the same detection when their boxes overlap at least this much


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def compare(a: list, b: list) -> dict:
    """Repeatability of two passes: identical iff every frame has the same detections with the same scores and boxes.
    Detections are paired across the passes by box overlap (greedy, best IoU first, at least MATCH_IOU), never by score rank, so a swap of
    the order of two people is not a difference. Reported separately: the largest score change and the largest box-coordinate change of
    paired detections, and `unmatched`, the detections found in one pass only (a detection that vanished is counted there, not hidden)."""
    same_count = sum(1 for x, y in zip(a, b) if len(x) == len(y))
    score_diff = box_diff = 0.0
    unmatched = 0
    for x, y in zip(a, b):
        pairs = sorted(((_iou(b1, b2), i, j) for i, (_, b1) in enumerate(x) for j, (_, b2) in enumerate(y)), reverse=True)
        used_x, used_y = set(), set()
        for v, i, j in pairs:
            if v < MATCH_IOU:
                break
            if i in used_x or j in used_y:
                continue
            used_x.add(i)
            used_y.add(j)
            score_diff = max(score_diff, abs(x[i][0] - y[j][0]))
            box_diff = max(box_diff, *(abs(p - q) for p, q in zip(x[i][1], y[j][1])))
        unmatched += (len(x) - len(used_x)) + (len(y) - len(used_y))
    unmatched += sum(len(f) for f in a[len(b):]) + sum(len(f) for f in b[len(a):])
    return dict(identical=a == b, frames_with_other_count=len(a) - same_count, unmatched=unmatched,
                max_score_diff=round(score_diff, 6), max_box_diff=round(box_diff, 6))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["nano", "small", "base"], required=True)
    ap.add_argument("--units", choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"], required=True)
    ap.add_argument("--video", default=str(VIDEO) if VIDEO else None, required=VIDEO is None)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=5.0)
    a = ap.parse_args(argv)
    os.environ["INFRAME_DETECT_UNITS"] = a.units
    if not Path(a.video).is_file():
        print(json.dumps(dict(model=a.model, units=a.units, error=f"no such video: {a.video}")))
        return 2
    from inframe import detect
    from inframe.decode import iter_samples
    from inframe.pipeline import DECODE_LONG_SIDE
    from inframe.probe import probe

    info = probe(a.video)
    frames = []
    for s in iter_samples(a.video, info, fps=a.fps, long_side=DECODE_LONG_SIDE, decoder=None, stats={}, small_side=DECODE_LONG_SIDE):
        if s.t - info.offset >= a.seconds:
            break
        frames.append(s.small if s.small is not None else s.rgb)
    load0 = os.getloadavg()[0]
    row = dict(model=a.model, units=a.units, frames=len(frames), seconds=a.seconds, fps=a.fps, load_avg_before=round(load0, 2))
    try:
        t = time.perf_counter()
        det = detect.CoreMLDetector(a.model)
        det.detect(frames[:det.batch])  # warm-up: the first call compiles for the compute units
        row.update(load_s=round(time.perf_counter() - t, 2), package=Path(det.path).parent.name, batch=det.batch, units_in_effect=det.units)
        s1, d1 = passes(det, frames, detect.BATCH)
        s2, d2 = passes(det, frames, detect.BATCH)
    except Exception as e:  # a unit that cannot run the model is a result, not a crash
        row.update(error=f"{type(e).__name__}: {e}"[:200])
        print(json.dumps(row))
        return 1
    n = max(len(frames), 1)
    row.update(ms_per_frame=round(1000 * min(s1, s2) / n, 2), ms_per_frame_runs=[round(1000 * s1 / n, 2), round(1000 * s2 / n, 2)],
               persons_over_0_5=sum(1 for f in d1 for sc, _ in f if sc >= 0.5), load_avg_after=round(os.getloadavg()[0], 2), **compare(d1, d2))
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
