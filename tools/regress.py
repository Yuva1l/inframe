#!/usr/bin/env python3
"""Regression check against a recorded baseline: detect on your own reference video, compare with the baseline.

The video and the baseline are not in the repo: set INFRAME_REGRESS_VIDEO and INFRAME_REGRESS_BASELINE or the keys regress_video and
regress_baseline in ~/.config/inframe/local.json (inframe.localcfg). With neither the gate prints "SKIP regress: no private clip configured" and exits 2.

    uv run python tools/regress.py                    run detect on the configured video (about 2.5 minutes) and compare
    uv run python tools/regress.py --from-run DIR     compare an earlier run's segments.json, no detect

The detector runs the units inframe uses by default (inframe.detect.DEFAULT_UNITS, CPU_AND_GPU); an exported INFRAME_DETECT_UNITS is
respected and --cpu-only forces CPU_ONLY. CPU_AND_GPU is repeatable only with a package whose topk is fp32 (bench/detect/export_gpu.py;
with the plain fp16 export a frame's score moves by up to 0.92 between runs and can flip PASS/FAIL with no code change,
a measured fact; inframe.detect warns about such a package and says how to re-export it).
Accepted changes (tools/regress_accepted.json): boundaries accepted to move, each with date and reason. That boundary passes within --tol
of the accepted value (or of the baseline one); every other boundary still compares with the baseline. The accepted changes in effect are printed in
the PASS/FAIL line. An entry whose "approved" value no longer matches the baseline is ignored (and reported).
Prints one PASS or FAIL line (the last line of output). Compared: the presence intervals (unpadded stretches)
of the baseline segments.json, same count, each start and end within --tol seconds (default 0.3).
Takes a lock (~/.cache/inframe/regress.lock) so parallel runs queue instead of running two detect jobs at once.
Runs only where the private clip is configured: the video is read in place, clips are cut into ~/.cache/inframe/regress/ and deleted
after the comparison. Nothing from the video is printed except interval times.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inframe import localcfg  # noqa: E402

os.environ.setdefault("INFRAME_MODELS", str(ROOT / "bench/detect/out/models"))
# The gate runs inframe's default units (inframe.detect.DEFAULT_UNITS: CPU_AND_GPU, repeatable with the fp32-topk package,
# bench/detect/export_gpu.py) unless INFRAME_DETECT_UNITS is exported or --cpu-only is given.

LOCK = Path.home() / ".cache/inframe/regress.lock"
WORK = Path.home() / ".cache/inframe/regress"
TOL = 0.3
ACCEPTED = ROOT / "tools/regress_accepted.json"


def apply_units(cpu_only: bool, gpu: bool = False) -> str:
    """Set the detector's compute units for this run and return the name in effect: inframe's default (inframe.detect.DEFAULT_UNITS) or
    an exported INFRAME_DETECT_UNITS; --cpu-only forces CPU_ONLY, --gpu forces CPU_AND_GPU."""
    if cpu_only:
        os.environ["INFRAME_DETECT_UNITS"] = "CPU_ONLY"
    elif gpu:
        os.environ["INFRAME_DETECT_UNITS"] = "CPU_AND_GPU"
    from inframe.detect import compute_units_name

    return compute_units_name()


def load_accepted(path: Path = ACCEPTED) -> list[dict]:
    """The accepted changes: [{"interval": 3, "edge": "start", "approved": 200.0, "accepted": 199.6, "date", "reason"}]."""
    p = Path(path)
    return json.loads(p.read_text())["accepted"] if p.is_file() else []


def apply_accepted(base: list[tuple[float, float]], accepted: list[dict]) -> tuple[dict, list[str], list[str]]:
    """The accepted changes as {(interval number, 0 start / 1 end): accepted value} for compare(), plus notes of the ones applied and of
    the ones ignored (the baseline's value is not the approved one the entry names, or the interval does not exist). The boundary keeps
    its approved value too, so the approved CPU_ONLY run still passes."""
    alts, applied, ignored = {}, [], []
    for a in accepted:
        i, k = int(a["interval"]), {"start": 0, "end": 1}[a["edge"]]
        label = f"#{i} {a['edge']} {a['approved']:.1f} -> {a['accepted']:.1f} ({a['date']})"
        if not 1 <= i <= len(base) or abs(base[i - 1][k] - float(a["approved"])) > 0.05:
            ignored.append(label)
        else:
            alts[(i, k)] = float(a["accepted"])
            applied.append(label)
    return alts, applied, ignored


def load_intervals(segments_json: Path) -> list[tuple[float, float]]:
    d = json.loads(Path(segments_json).read_text())
    return [(float(s), float(e)) for s, e in d["presence_intervals"]]


def compare(base: list[tuple[float, float]], new: list[tuple[float, float]], tol: float = TOL, accepted: dict | None = None) -> tuple[bool, str]:
    """(ok, one-line summary). Same number of intervals and every boundary within `tol` seconds of the baseline, or, for a boundary in
    `accepted` ({(interval, 0 start / 1 end): value}, apply_accepted), of the accepted value."""
    if len(base) != len(new):
        extra = [f"{s:.1f}-{e:.1f}" for s, e in new if all(abs(s - bs) > tol for bs, _ in base)]
        gone = [f"{s:.1f}-{e:.1f}" for s, e in base if all(abs(s - ns) > tol for ns, _ in new)]
        return False, f"{len(new)} intervals, baseline has {len(base)} (new only: {extra or 'none'}; baseline only: {gone or 'none'})"
    worst, where = 0.0, ""
    bad = []
    for i, ((bs, be), (ns, ne)) in enumerate(zip(base, new), 1):
        for k, (what, a, b) in enumerate((("start", bs, ns), ("end", be, ne))):
            dev = abs(a - b)
            if (i, k) in (accepted or {}):
                dev = min(dev, abs(accepted[(i, k)] - b))
            if dev > worst:
                worst, where = dev, f"interval {i} {what}"
            if dev > tol:
                bad.append(f"#{i} {what} {a:.1f} -> {b:.1f}")
    if bad:
        return False, f"{len(base)} intervals, {len(bad)} boundaries beyond {tol:g} s: {'; '.join(bad[:6])}"
    return True, f"{len(base)} intervals, max deviation {worst:.2f} s ({where}), limit {tol:g} s"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    video, baseline = localcfg.get("regress_video"), localcfg.get("regress_baseline")  # read now, not when the module was imported
    ap.add_argument("--video", default=str(video) if video else None)
    ap.add_argument("--baseline", default=str(baseline) if baseline else None)
    ap.add_argument("--tol", type=float, default=TOL, help="seconds each interval boundary may move (default 0.3)")
    ap.add_argument("--from-run", help="a run folder (or its segments.json) to compare instead of running detect")
    ap.add_argument("--tap", action="store_true",
                    help="run with a tap on the target instead of every person: the tap is the box of the best detection in the middle of the first "
                         "baseline interval, read from the samples.csv of an earlier plain regress run (numbers only, no frame is looked at)")
    ap.add_argument("--cpu-only", action="store_true",
                    help="run the detector on the CPU only (deterministic scores, slower, the approved run exactly) instead of inframe's default CPU_AND_GPU")
    ap.add_argument("--gpu", action="store_true",
                    help="run CPU_AND_GPU even when INFRAME_DETECT_UNITS is exported (it is the default); repeatable only with a package exported by bench/detect/export_gpu.py (fp32 topk)")
    ap.add_argument("--accepted", default=str(ACCEPTED), help="accepted changes to the baseline (json; 'none' for the approved clips only)")
    ap.add_argument("--keep", action="store_true", help="keep the cut clips of the run")
    a = ap.parse_args(argv)
    if not a.baseline or (not a.video and not a.from_run):
        print("SKIP regress: no private clip configured (INFRAME_REGRESS_VIDEO and INFRAME_REGRESS_BASELINE, or ~/.config/inframe/local.json)")
        return 2
    if not Path(a.baseline).is_file():
        print(f"FAIL regress: baseline not found: {a.baseline} (this check only runs where the private clip is configured)")
        return 2
    base = load_intervals(Path(a.baseline))
    alts, applied, ignored = apply_accepted(base, [] if a.accepted == "none" else load_accepted(Path(a.accepted)))
    note = (f"; accepted: {', '.join(applied)}" if applied else "") + (f"; accepted but IGNORED (baseline differs): {', '.join(ignored)}" if ignored else "")
    if a.from_run:
        p = Path(a.from_run)
        p = p / "segments.json" if p.is_dir() else p
        ok, msg = compare(base, load_intervals(p), a.tol, alts)
        print(f"{'PASS' if ok else 'FAIL'} regress (from {p.parent.name}): {msg}{note}")
        return 0 if ok else 1
    if not Path(a.video).is_file():
        print(f"FAIL regress: video not found: {a.video} (this check only runs where the private clip is configured)")
        return 2
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[regress] another regress run holds the lock; waiting", file=sys.stderr, flush=True)
            fcntl.flock(lf, fcntl.LOCK_EX)
        from inframe.pipeline import Settings, run_cut

        units = apply_units(a.cpu_only, a.gpu)
        print(f"[regress] detector compute units: {units}", file=sys.stderr, flush=True)

        out = WORK / ("tap" if a.tap else "detect")
        target = None
        if a.tap:
            import csv

            from inframe.tap import Tap, TapTarget

            prev = WORK / "detect" / "samples.csv"
            if not prev.is_file():
                print("FAIL regress: --tap needs the samples.csv of an earlier plain run (uv run python tools/regress.py)")
                return 2
            mid = (base[0][0] + base[0][1]) / 2
            rows = [r for r in csv.DictReader(open(prev)) if r["x0"]]
            r = min(rows, key=lambda r: abs(float(r["t"]) - mid))
            target = TapTarget([Tap(float(r["t"]), box=tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1")))])
            print(f"[regress] tap at {float(r['t']):.1f} s (middle of the first baseline interval)", file=sys.stderr)
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.perf_counter()
        try:
            run_cut(a.video, str(out), Settings(), target=target)  # the settings of the approved run: detect, balanced, 5 fps
        except Exception as e:
            print(f"FAIL regress: run failed: {type(e).__name__}: {e}")
            return 1
        wall = time.perf_counter() - t0
        ok, msg = compare(base, load_intervals(out / "segments.json"), a.tol, alts)
        if not a.keep:
            for f in out.glob(f"{Path(a.video).stem}_*_*-*.mov"):
                f.unlink()
        print(f"{'PASS' if ok else 'FAIL'} regress: {msg}{note}; run {wall:.0f} s; units {units}; results in {out}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
