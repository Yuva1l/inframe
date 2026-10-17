// Presence lane: max score per pixel column for the visible range (docs/T9A_DESIGN.md 2.5).

/** `t` is sorted ascending. Samples with t0 <= t < t1 fall into `columns` equal columns (t == t1 joins the last one);
 *  a column with no sample is NaN. */
export function bucketMax(t: ArrayLike<number>, score: ArrayLike<number>, t0: number, t1: number, columns: number): Float32Array {
  const out = new Float32Array(Math.max(0, Math.floor(columns))).fill(NaN);
  const n = Math.min(t.length, score.length);
  if (out.length === 0 || n === 0 || !(t1 > t0)) return out;
  const scale = out.length / (t1 - t0);
  // first sample at or after t0
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (t[mid] < t0) lo = mid + 1;
    else hi = mid;
  }
  for (let i = lo; i < n && t[i] <= t1; i++) {
    let c = Math.floor((t[i] - t0) * scale);
    if (c >= out.length) c = out.length - 1;
    const s = score[i];
    if (!(out[c] >= s)) out[c] = s; // NaN compares false, so an empty column takes the first score
  }
  return out;
}

/** The index of the last sample at or before t whose gap to t is at most `maxGap` seconds, or -1. `ts` is sorted ascending. */
export function sampleAt(ts: ArrayLike<number>, t: number, maxGap = 1): number {
  let lo = 0;
  let hi = ts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  const i = lo - 1;
  return i >= 0 && t - ts[i] <= maxGap ? i : -1;
}

/** The highest score among samples with a <= t <= b, and when it happened; null when no sample falls in the range. `ts` is sorted ascending. */
export function peakIn(ts: ArrayLike<number>, score: ArrayLike<number>, a: number, b: number): { t: number; score: number } | null {
  let lo = 0;
  let hi = ts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] < a) lo = mid + 1;
    else hi = mid;
  }
  let best: { t: number; score: number } | null = null;
  for (let i = lo; i < ts.length && ts[i] <= b && i < score.length; i++) {
    if (!best || score[i] > best.score) best = { t: ts[i], score: score[i] };
  }
  return best;
}

/** `bucketMax`, with each empty column holding the last value for up to `hold` columns (a step plot between samples that are
 *  farther apart than a pixel); columns beyond the hold stay NaN. */
export function holdColumns(cols: Float32Array, hold: number): Float32Array {
  const out = cols.slice();
  let last = NaN;
  let age = 0;
  for (let i = 0; i < out.length; i++) {
    if (!Number.isNaN(out[i])) {
      last = out[i];
      age = 0;
    } else if (!Number.isNaN(last) && ++age <= hold) out[i] = last;
  }
  return out;
}

/** The largest of `xs`, or `floor` when that is larger or `xs` is empty. A loop, because Math.max(...xs) overflows the stack past about 125k values. */
export function maxOf(xs: ArrayLike<number>, floor = -Infinity): number {
  let m = floor;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  return m;
}
