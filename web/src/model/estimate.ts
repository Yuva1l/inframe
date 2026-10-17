// "About 4 min (estimate)" before Run (docs/T9A_DESIGN.md 1.5). After a run in this UI the estimate is the median of the last five
// seconds-per-source-minute values measured for that preset; before any run it is the detector's model-call cost times the number of
// samples, and it says "at least" because everything else the run does (decode, cuts) comes on top.

/** Seconds per sample the detector alone needs on the default CPU_AND_GPU units (measured: small with the
 *  fp32-topk package 11.0 ms, base 16.7 ms per frame). The fast preset's nano model was not measured there: 5 ms is a deliberately low
 *  guess [unverified], so "at least" still holds; with INFRAME_DETECT_UNITS=CPU_ONLY a run takes about 3x longer than this. */
export const MODEL_SECONDS: Record<string, number> = { fast: 0.005, balanced: 0.011, accurate: 0.0167 };
/** Samples per second of video for each preset's detector (pipeline.PRESETS detect_fps); null = keyframes only. */
export const SAMPLE_FPS: Record<string, number | null> = { fast: null, balanced: 5, accurate: 5 };
export const KEEP = 5;

export function median(xs: readonly number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Add one measurement (seconds of wall time per minute of source) and keep the last five. */
export function pushTiming(list: readonly number[], secPerMinute: number): number[] {
  if (!Number.isFinite(secPerMinute) || secPerMinute <= 0) return [...list];
  return [...list, secPerMinute].slice(-KEEP);
}

export interface Estimate {
  seconds: number;
  basis: 'measured' | 'model';
}

export function estimate(opts: {
  preset: string;
  duration: number;
  keyframes: number;
  fps?: number | null;
  motion: boolean;
  timings: readonly number[];
}): Estimate | null {
  const m = median(opts.timings);
  if (m !== null) return { seconds: (m * opts.duration) / 60, basis: 'measured' };
  if (opts.motion) return null; // no model call to count
  const perSample = MODEL_SECONDS[opts.preset];
  if (perSample === undefined) return null;
  const fps = opts.fps !== undefined && opts.fps !== null ? opts.fps : SAMPLE_FPS[opts.preset];
  const samples = fps === null || fps === undefined ? opts.keyframes : fps > 0 ? opts.duration * fps : opts.keyframes;
  return { seconds: samples * perSample, basis: 'model' };
}

export function span(sec: number): string {
  if (sec < 90) return `${Math.max(1, Math.round(sec / 5) * 5)} s`;
  const min = Math.round(sec / 60);
  if (min < 90) return `${min} min`;
  return `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, '0')} min`;
}

/** "about 4 min (estimate)" or "at least 2 min (estimate)". */
export function estimateText(e: Estimate | null): string {
  if (!e) return 'No estimate yet: it appears after the first run.';
  return `${e.basis === 'model' ? 'at least' : 'about'} ${span(e.seconds)} (estimate)`;
}
