#!/usr/bin/env bash
# Times detect on the first 60 s of the configured regression video (INFRAME_REGRESS_VIDEO, see inframe.localcfg; 5 samples per second = 300 frames) for {small, base} x {CPU_ONLY, CPU_AND_GPU, CPU_AND_NE}:
# ms per frame, and whether two passes give identical scores. Run it on a quiet machine (the load averages are printed per row), alone:
#     bash tools/bench_units.sh [--seconds 60] [--video PATH] [--models "small base"] [--units "CPU_ONLY CPU_AND_GPU CPU_AND_NE"]
# Rows go to stdout as a table and, one JSON object per line, to $INFRAME_BENCH_OUT (default .logs/bench_units.jsonl).
# The video is read in place; nothing from it is written or printed. The default units were chosen from this table (speed vs repeatability).
set -uo pipefail
cd "$(dirname "$0")/.."
export INFRAME_MODELS="${INFRAME_MODELS:-$PWD/bench/detect/out/models}"
SECONDS_ARG=60; VIDEO_ARG=(); MODELS="small base"; UNITS="CPU_ONLY CPU_AND_GPU CPU_AND_NE"
while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) SECONDS_ARG="$2"; shift 2 ;;
    --video) VIDEO_ARG=(--video "$2"); shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --units) UNITS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
OUT="${INFRAME_BENCH_OUT:-.logs/bench_units.jsonl}"
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
echo "load average at start: $(uptime | sed 's/.*load averages*: //')" >&2
for m in $MODELS; do
  for u in $UNITS; do
    echo "[bench_units] $m $u ..." >&2
    uv run python tools/bench_units.py --model "$m" --units "$u" --seconds "$SECONDS_ARG" ${VIDEO_ARG[@]+"${VIDEO_ARG[@]}"} 2> >(grep -v -e scikit-learn -e "Torch version" -e fake_palettize -e n_bits >&2) >> "$OUT"
  done
done
uv run python - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
print(f"{'model':6} {'units':12} {'ms/frame':>9} {'load s':>7} {'identical':>9} {'score diff':>10} {'box diff':>9} {'unmatched':>9} {'frames':>6} {'load avg':>9}")
for r in rows:
    if "error" in r:
        print(f"{r['model']:6} {r['units']:12} ERROR {r['error']}")
        continue
    print(f"{r['model']:6} {r['units']:12} {r['ms_per_frame']:9.1f} {r['load_s']:7.1f} {str(r['identical']):>9} {r['max_score_diff']:10.4f} {r['max_box_diff']:9.4f} {r['unmatched']:9d} {r['frames']:6d} {r['load_avg_before']:4.1f}/{r['load_avg_after']:.1f}")
PY
