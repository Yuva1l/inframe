// A line-by-line port of src/inframe/segments.py (docs/T9B_DESIGN.md section 4): hysteresis, gap merge, minimum length, padding, clamping.
// Same order of operations as the Python, so the doubles match; tests/golden/segments_v1.json (made by tools/gen_golden.py) holds both
// to the same numbers. Pure: no I/O, no DOM.
import type { ResegmentParams } from '../engine/types';

export type Interval = [number, number];

/** Boundaries of each sample's cell: midpoints between neighbours; the first and last cell extend half a gap outward, clamped to
 *  [lo, duration]. Returns times.length + 1 edges ([] for no samples). */
export function sampleEdges(times: readonly number[], duration: number, lo = 0): number[] {
  const n = times.length;
  if (n === 0) return [];
  if (n === 1) return [Math.max(lo, times[0] - 0.5), Math.min(duration, times[0] + 0.5)];
  const mids: number[] = [];
  for (let i = 0; i + 1 < n; i++) mids.push((times[i] + times[i + 1]) / 2);
  const first = Math.max(lo, times[0] - (mids[0] - times[0]));
  const last = Math.min(duration, times[n - 1] + (times[n - 1] - mids[mids.length - 1]));
  return [first, ...mids, last];
}

/** Presence intervals from per-sample scores. Enter when score >= enter, leave when score < exit (exit <= enter keeps one weak sample
 *  from splitting a stretch). Boundaries sit halfway between the last sample on one side and the first on the other. */
export function hysteresis(times: readonly number[], scores: readonly number[], enter: number, exit: number, duration: number, lo = 0): Interval[] {
  if (times.length !== scores.length) throw new Error('times and scores differ in length');
  if (exit > enter) throw new Error('exit threshold must not exceed enter threshold');
  const edges = sampleEdges(times, duration, lo);
  const out: Interval[] = [];
  let start: number | null = null;
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    if (start === null) {
      if (s >= enter) start = edges[i];
    } else if (s < exit) {
      out.push([start, edges[i]]);
      start = null;
    }
  }
  if (start !== null) out.push([start, edges[edges.length - 1]]);
  return out;
}

/** Join neighbours separated by at most `gap` seconds. Sorted like Python's sorted() on tuples: by start, then by end. */
export function mergeGaps(intervals: readonly Interval[], gap: number): Interval[] {
  const out: Interval[] = [];
  const sorted = [...intervals].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
  for (const [s, e] of sorted) {
    if (out.length && s - out[out.length - 1][1] <= gap) out[out.length - 1] = [out[out.length - 1][0], Math.max(out[out.length - 1][1], e)];
    else out.push([s, e]);
  }
  return out;
}

export function dropShort(intervals: readonly Interval[], minLen: number): Interval[] {
  return intervals.filter(([s, e]) => e - s >= minLen).map(([s, e]) => [s, e]);
}

/** Grow each interval, clamp to [lo, duration] (lo is the video's first picture in file time, 0 when it starts at 0), then merge any
 *  that now touch or overlap. */
export function pad(intervals: readonly Interval[], before: number, after: number, duration: number, lo = 0): Interval[] {
  const grown = intervals.map(([s, e]): Interval => [Math.max(lo, s - before), Math.min(duration, e + after)]);
  return mergeGaps(grown, 0);
}

export interface SegmentParams {
  enter?: number;
  exit?: number;
  mergeGap?: number;
  minLen?: number;
  pad?: number;
  lo?: number;
}

/** (presence intervals unpadded, clip intervals padded). Order: hysteresis, merge, minimum length, pad. Defaults are segments.py's. */
export function buildSegments(times: readonly number[], scores: readonly number[], duration: number, p: SegmentParams = {}): { presence: Interval[]; clips: Interval[] } {
  const { enter = 0.5, exit = 0.3, mergeGap = 1.5, minLen = 1.0, pad: padS = 1.0, lo = 0.0 } = p;
  const presence = dropShort(mergeGaps(hysteresis(times, scores, enter, exit, duration, lo), mergeGap), minLen);
  return { presence, clips: pad(presence, padS, padS, duration, lo) };
}

const FIELDS = ['enter', 'exit', 'merge_gap', 'min_len', 'pad'] as const;

/** What buildSegments makes from a run's own samples (serve_runs.resegment): a param that is missing or null takes the run's
 *  `settings` value; a field with no value anywhere, or exit above enter, throws. `bounds` is the video's first picture and end in file time. */
export function resegment(
  samples: { t: readonly number[]; score: readonly number[] },
  settings: Record<string, unknown>,
  params: ResegmentParams,
  bounds: { offset: number; end: number },
): { presence: Interval[]; clips: Interval[] } {
  const v = {} as Record<(typeof FIELDS)[number], number>;
  for (const k of FIELDS) {
    const given = (params as Record<string, unknown>)[k];
    const x = given === undefined || given === null ? settings[k] : given;
    if (x === undefined || x === null) throw new Error(`${k}: the run has no value to start from; send one`);
    // the same checks as serve_runs._number(v, k, 0.0): a finite number (not a bool or string), at least 0
    if (typeof x !== 'number' || !Number.isFinite(x)) throw new Error(`${k} must be a finite number`);
    if (x < 0) throw new Error(`${k} must be at least 0 (got ${x})`);
    v[k] = x;
  }
  if (v.exit > v.enter) throw new Error(`exit (${v.exit}) must not exceed enter (${v.enter})`);
  return buildSegments(samples.t, samples.score, bounds.end, { enter: v.enter, exit: v.exit, mergeGap: v.merge_gap, minLen: v.min_len, pad: v.pad, lo: bounds.offset });
}
