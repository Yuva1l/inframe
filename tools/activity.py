#!/usr/bin/env python3
"""Calibrate and measure the riding/walking threshold of `--activity riding` (T6). No labelling, no looking at footage.

    uv run python tools/activity.py synthetic                 moving, walking and still boxes with detector jitter: separation numbers
    uv run python tools/activity.py slide build [--out DIR]   composites in which a person clip slides at a known speed (the truth is the
                                                              construction): still, walking speeds, riding speeds
    uv run python tools/activity.py slide score [--out DIR]   detector + tracker + activity.smoothed_speeds on them: separation, the threshold
                                                              chosen on the calibration clips, and its score on the held-out clips
    uv run python tools/activity.py real [--out DIR]          the speed of the tracks in the spliced composites (splice.py) by what the clip is:
                                                              street pedestrians (static camera) and the skate clips; numbers only

The sliding composites are written outside the repo (default ~/.cache/inframe/activity) and never committed. The regression clip is not used
for calibration at all; it only has to stay unchanged (tools/regress.py).

Speeds are in box heights per real second (inframe.activity). Classes of the sliding clips: still 0, walking 0.5 to 1.1 (a walking
person covers about 1.4 m/s = 0.8 body heights per second), riding 1.8 to 3.5 (3 to 6 m/s). The person inside the clip moves too
(squats, tricks), which is what makes the numbers realistic: the box speed is the slide plus the person.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("INFRAME_MODELS", str(ROOT / "bench/detect/out/models"))

from inframe import activity  # noqa: E402

DEFAULT_OUT = Path.home() / ".cache/inframe/activity"
FPS = 5.0
STILL, WALK, RIDE = "still", "walking", "riding"
SLIDE_SPEEDS = [(STILL, 0.0), (WALK, 0.5), (WALK, 0.8), (WALK, 1.1), (RIDE, 1.8), (RIDE, 2.5), (RIDE, 3.5)]  # (class, designed box heights per second)
CALIBRATION_SRC = ["pair1", "lone1"]
HELD_OUT_SRC = ["pair2", "lone2"]
SEG_S = 6.0
TILE_H = 300  # pixels; the person is about 70 % of it
GUESS_BOX_H = 0.7 * TILE_H  # first guess of the box height in pixels, to set the slide speed; the truth uses the measured height
CANVAS_W, CANVAS_H = 1280, 720
TRAVEL = 700  # pixels the tile travels before turning round


# ---------------------------------------------------------------- numbers
def sweep(walk: np.ndarray, ride: np.ndarray, lo: float = 0.6, hi: float = 3.0, step: float = 0.05) -> list[dict]:
    """For every candidate threshold: the fraction of walking samples below it and of riding samples at or above it."""
    rows = []
    for thr in np.arange(lo, hi + 1e-9, step):
        rows.append(dict(threshold=round(float(thr), 2), walking_ok=float((walk < thr).mean()), riding_ok=float((ride >= thr).mean())))
    return rows


def best_threshold(walk: np.ndarray, ride: np.ndarray) -> dict:
    """The threshold with the best worst-class recall; among ties the middle of the tied range."""
    rows = sweep(walk, ride)
    for r in rows:
        r["worst"] = min(r["walking_ok"], r["riding_ok"])
    top = max(r["worst"] for r in rows)
    tied = [r for r in rows if r["worst"] >= top - 1e-9]
    pick = tied[len(tied) // 2]
    return dict(threshold=pick["threshold"], walking_ok=round(pick["walking_ok"], 4), riding_ok=round(pick["riding_ok"], 4),
                tied_range=[tied[0]["threshold"], tied[-1]["threshold"]])


def auc(walk: np.ndarray, ride: np.ndarray) -> float:
    """Probability that a random riding sample is faster than a random walking one (1 = perfect separation)."""
    if not len(walk) or not len(ride):
        return float("nan")
    w = np.sort(walk)
    return float(np.mean([(np.searchsorted(w, r, side="left") + 0.5 * (np.searchsorted(w, r, side="right") - np.searchsorted(w, r, side="left"))) / len(w) for r in ride]))


def summarize(name: str, v: np.ndarray) -> dict:
    if not len(v):
        return dict(group=name, n=0)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return dict(group=name, n=int(len(v)), p5=round(q[0], 2), p25=round(q[1], 2), median=round(q[2], 2), p75=round(q[3], 2), p95=round(q[4], 2))


def report(title: str, groups: dict[str, np.ndarray], thr: float) -> dict:
    walk, ride = groups.get(WALK, np.empty(0)), groups.get(RIDE, np.empty(0))
    out = dict(title=title, groups=[summarize(k, v) for k, v in groups.items()], auc=round(auc(walk, ride), 4),
               at_threshold=dict(threshold=thr, walking_below=round(float((walk < thr).mean()), 4) if len(walk) else None,
                                 riding_at_or_above=round(float((ride >= thr).mean()), 4) if len(ride) else None,
                                 still_below=round(float((groups.get(STILL, np.empty(0)) < thr).mean()), 4) if len(groups.get(STILL, [])) else None),
               best=best_threshold(walk, ride) if len(walk) and len(ride) else None)
    print(f"\n{title}")
    for g in out["groups"]:
        print("  " + "  ".join(f"{k}={v}" for k, v in g.items()))
    a = out["at_threshold"]
    print(f"  AUC {out['auc']}; at threshold {thr:g}: walking below {a['walking_below']}, riding at or above {a['riding_at_or_above']}"
          + (f", still below {a['still_below']}" if a["still_below"] is not None else ""))
    if out["best"]:
        print(f"  best threshold here: {out['best']}")
    return out


# ---------------------------------------------------------------- synthetic boxes
def synthetic(seed: int = 0, n_tracks: int = 300, seconds: float = 8.0, aspect: float = 16 / 9) -> dict[str, np.ndarray]:
    """Per class, the smoothed speeds of random box trajectories at 5 samples per second with detector jitter.
    walking: 0.5 to 1.2 box heights per second with stop-and-go; riding: 1.8 to 3.6 with gentle speed changes; still: 0 to 0.1."""
    rng = np.random.default_rng(seed)
    ts = np.arange(0, seconds, 1 / FPS)
    out: dict[str, list[float]] = {STILL: [], WALK: [], RIDE: []}
    for cls, (lo, hi) in ((STILL, (0.0, 0.1)), (WALK, (0.5, 1.2)), (RIDE, (1.8, 3.6))):
        for _ in range(n_tracks):
            h = rng.uniform(0.15, 0.5)  # box height as a fraction of the frame
            speed = rng.uniform(lo, hi)
            pos = np.array([rng.uniform(0.2, 0.8) * aspect, rng.uniform(0.3, 0.7)])  # in frame heights
            ang = rng.uniform(0, 2 * np.pi)
            boxes = []
            for i, t in enumerate(ts):
                if i % int(FPS) == 0:
                    ang += rng.normal(0, 0.6 if cls == WALK else 0.3)
                gait = 1.0
                if cls == WALK:
                    gait = 0.0 if (int(t) % 4 == 3 and rng.random() < 0.3) else 1.0  # now and then a pause
                elif cls == RIDE:
                    gait = 1.0 + 0.2 * np.sin(0.7 * t + rng.uniform(0, 6))
                v = speed * gait * h
                step = np.array([np.cos(ang), np.sin(ang)]) * v / FPS
                pos = pos + step
                for ax, lim in ((0, aspect), (1, 1.0)):  # bounce off the frame edge
                    if not 0.1 < pos[ax] < lim - 0.1:
                        pos[ax] = min(max(pos[ax], 0.1), lim - 0.1)
                        ang = np.pi - ang if ax == 0 else -ang
                cx = pos[0] + rng.normal(0, 0.02 * h)  # detector jitter
                cy = pos[1] + rng.normal(0, 0.02 * h)
                hh = h * (1 + rng.normal(0, 0.03))
                w = 0.4 * h / aspect
                boxes.append((cx / aspect - w / 2, cy - hh / 2, cx / aspect + w / 2, cy + hh / 2))
            sp = activity.smoothed_speeds(ts, boxes, aspect)
            out[cls] += [s for s in sp if s is not None]
    return {k: np.asarray(v) for k, v in out.items()}


# ---------------------------------------------------------------- sliding composites
def slide_name(src: str) -> str:
    return f"slide_{src}"


def build_slide(out: Path) -> list[Path]:
    import splice

    out.mkdir(parents=True, exist_ok=True)
    made = []
    for src in CALIBRATION_SRC + HELD_OUT_SRC:
        path = splice.source_path(src)
        segs, truth = [], []
        for i, (cls, hs) in enumerate(SLIDE_SPEEDS):
            v_px = hs * GUESS_BOX_H
            x_expr = f"60+abs(mod({v_px:.2f}*t,{2 * TRAVEL})-{TRAVEL})"
            seg = out / f"{slide_name(src)}.seg{i}.mp4"
            cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-nostats", "-f", "lavfi", "-i", f"color=c=0x808080:s={CANVAS_W}x{CANVAS_H}:r={splice.FPS}:d={SEG_S}",
                   "-stream_loop", "-1", "-ss", "2", "-t", f"{SEG_S}", "-i", str(path),
                   "-filter_complex", f"[1:v]fps={splice.FPS},scale=-2:{TILE_H},setsar=1[p];[0:v][p]overlay=x='{x_expr}':y={CANVAS_H - TILE_H - 100}:eval=frame,format=yuv420p[v]",
                   "-map", "[v]", "-t", f"{SEG_S}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-g", str(splice.FPS), str(seg)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"activity: ffmpeg failed for {src} segment {i}: {r.stderr.strip()[-300:]}")
            segs.append(seg)
            truth.append(dict(t0=i * SEG_S, t1=(i + 1) * SEG_S, cls=cls, designed_speed=hs, slide_px_per_s=round(v_px, 2)))
        lst = out / f"{slide_name(src)}.concat.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in segs))
        mp4 = out / f"{slide_name(src)}.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(mp4)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"activity: concat failed: {r.stderr.strip()[-300:]}")
        for p in [*segs, lst]:
            p.unlink()
        (out / f"{slide_name(src)}.truth.json").write_text(json.dumps(dict(src=src, tile_h=TILE_H, segments=truth, note="Truth is the slide speed; the box height is measured."), indent=1))
        made.append(mp4)
        print(f"{mp4.name}: {len(truth)} segments of {SEG_S:g} s")
    return made


def track_speed_samples(video: Path, fps: float = FPS, factor: float = 1.0) -> list[dict]:
    """Detect (cached, 10 fps), follow the boxes the way ActivityTarget does, and return per sample: t, track id, box, score, smoothed speed."""
    import splice
    from inframe.detect import Detection
    from inframe.target import AnyPerson

    dets = splice.detect_composite(video, 10.0)
    step = round(10.0 / fps)
    frame = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    cuts = []
    truth_f = video.parent / f"{video.stem}.truth.json"
    if truth_f.is_file():
        tr = json.loads(truth_f.read_text())
        cuts = splice.shot_starts(tr) if "cuts" in tr else [s["t0"] for s in tr.get("segments", [])]
    cut_i = 0
    act = activity.ActivityTarget(AnyPerson(), slowmo_factor=factor)
    for t, d in dets[::step]:
        while cut_i < len(cuts) and cuts[cut_i] <= t + 1e-6:
            if cuts[cut_i] > 0:
                act.on_cut()
            cut_i += 1
        act.evaluate(t, frame, [Detection(s, tuple(b)) for b, s in d])
    act.finalize()
    return [dict(t=t, id=i, box=b, score=sc, speed=sp) for t, rows in act.samples() for b, sc, i, sp in rows if i is not None]


def score_slide(out: Path) -> dict:
    groups = {"calibration": {}, "held_out": {}}
    heights = {}
    for which, srcs in (("calibration", CALIBRATION_SRC), ("held_out", HELD_OUT_SRC)):
        for src in srcs:
            video = out / f"{slide_name(src)}.mp4"
            truth = json.loads((out / f"{slide_name(src)}.truth.json").read_text())
            rows = track_speed_samples(video)
            for seg in truth["segments"]:
                inside = [r for r in rows if seg["t0"] + 1.0 <= r["t"] < seg["t1"] - 0.2 and r["speed"] is not None and r["score"] >= 0.5]
                key = f"{seg['cls']}"
                groups[which].setdefault(key, []).extend(r["speed"] for r in inside)
                hs = [(r["box"][3] - r["box"][1]) * CANVAS_H for r in rows if seg["t0"] <= r["t"] < seg["t1"]]
                heights.setdefault(src, []).append((seg["designed_speed"], round(float(np.median(hs)), 1) if hs else None,
                                                    round(float(np.median([r["speed"] for r in inside])), 2) if inside else None))
    for src, h in heights.items():
        print(f"{src}: (designed speed, measured box height px, median measured speed) per segment: {h}")
    res = {}
    calib = {k: np.asarray(v) for k, v in groups["calibration"].items()}
    chosen = best_threshold(calib.get(WALK, np.empty(0)), calib.get(RIDE, np.empty(0)))
    thr = chosen["threshold"]
    res["calibration"] = report("sliding clips, calibration sources " + "+".join(CALIBRATION_SRC), calib, thr)
    res["held_out"] = report("sliding clips, held-out sources " + "+".join(HELD_OUT_SRC) + f" (threshold {thr:g} from the calibration set)",
                             {k: np.asarray(v) for k, v in groups["held_out"].items()}, thr)
    res["segment_medians"] = heights
    return res


# ---------------------------------------------------------------- real clips inside the spliced composites
REAL_GROUPS = {"crowd1": "street pedestrians (static camera; walking by the clip's description, not checked)",
               "lone1": "skater, one person (clip description)", "lone2": "skater, one person (clip description)",
               "bystanders": "skater with bystanders (clip description)", "target": "the target person's clips (not labelled)"}


def real(thr: float) -> dict:
    import splice

    base = splice.DEFAULT_OUT
    groups: dict[str, list[float]] = {}
    for name, recipe in splice.COMPOSITES.items():
        video = base / f"{name}.mp4"
        if name == "pan_follow" or not video.is_file() or not (base / f"{name}.truth.json").is_file():
            continue
        truth = json.loads((base / f"{name}.truth.json").read_text())
        rows = track_speed_samples(video)
        for r in rows:
            if r["speed"] is None or r["score"] < 0.5:
                continue
            cx, cy = (r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2
            _, tile = splice.tile_at(truth, r["t"], cx, cy)
            if tile is None:
                continue
            key = tile["src"] if tile["src"] in REAL_GROUPS else ("target" if tile.get("role") == "target" else None)
            if key:
                groups.setdefault(key, []).append(r["speed"])
    print("\nreal clips inside the spliced composites (tracks whose box centre is in that tile; speeds of the tile's own picture):")
    out = {}
    for k, v in groups.items():
        a = np.asarray(v)
        s = summarize(k, a)
        s["fraction_at_or_above_threshold"] = round(float((a >= thr).mean()), 4)
        s["what"] = REAL_GROUPS.get(k, "")
        out[k] = s
        print("  " + "  ".join(f"{kk}={vv}" for kk, vv in s.items()))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synthetic")
    s = sub.add_parser("slide")
    s.add_argument("what", choices=["build", "score"])
    s.add_argument("--out", default=str(DEFAULT_OUT))
    r = sub.add_parser("real")
    r.add_argument("--threshold", type=float, default=activity.THRESHOLD)
    a = p.parse_args(argv)
    if a.cmd == "synthetic":
        g = synthetic()
        report("synthetic boxes (300 tracks per class, 8 s, 5 samples/s, detector jitter)", g, activity.THRESHOLD)
    elif a.cmd == "slide":
        out = Path(a.out)
        if a.what == "build":
            build_slide(out)
        else:
            res = score_slide(out)
            (out / "slide_scores.json").write_text(json.dumps(res, indent=1, default=float))
    else:
        real(a.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
