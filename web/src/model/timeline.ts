// Timeline geometry (docs/T9A_DESIGN.md section 5, Part C1). All times are file time in seconds.

export interface View {
  /** file time at the left edge */
  t0: number;
  /** pixels per second */
  pps: number;
  /** width in pixels */
  width: number;
}

export interface Span {
  offset: number;
  end: number;
}

export const MAX_PPS = 400;
export const TICK_STEPS = [1 / 30, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800];

export const timeToX = (v: View, t: number): number => (t - v.t0) * v.pps;
export const xToTime = (v: View, x: number): number => v.t0 + x / v.pps;
export const viewEnd = (v: View): number => v.t0 + v.width / v.pps;

/** The whole video in `width` pixels. */
export function fit(duration: number, width: number, offset: number): View {
  const w = Math.max(1, width);
  return { t0: offset, pps: w / Math.max(duration, 1e-6), width: w };
}

/** Zoom by `factor` around `anchorT`; the anchor keeps its x. pps stays within [width/duration, 400] and the view within the span. */
export function zoomAt(view: View, anchorT: number, factor: number, bounds: Span): View {
  const duration = Math.max(bounds.end - bounds.offset, 1e-6);
  const minPps = view.width / duration;
  const pps = Math.min(Math.max(view.pps * factor, minPps), Math.max(MAX_PPS, minPps));
  const x = timeToX(view, anchorT);
  let t0 = anchorT - x / pps;
  t0 = Math.min(t0, bounds.end - view.width / pps);
  t0 = Math.max(t0, bounds.offset);
  return { t0, pps, width: view.width };
}

/** The first ruler step (seconds) that leaves at least 80 px between ticks. */
export function tickStep(pps: number): number {
  return TICK_STEPS.find((s) => s * pps >= 80) ?? TICK_STEPS[TICK_STEPS.length - 1];
}

/** m:ss or h:mm:ss, with tenths when the step is below one second. */
export function formatTime(t0: number, step = 1): string {
  const t = Math.max(0, t0);
  const tenths = step < 1;
  const total = Math.max(0, Math.round(t * 10) / 10);
  const whole = tenths ? Math.floor(total) : Math.floor(Math.round(t * 1000) / 1000);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  const ss = String(s).padStart(2, '0');
  const frac = tenths ? `.${Math.min(9, Math.round((total - whole) * 10))}` : '';
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${ss}${frac}` : `${m}:${ss}${frac}`;
}

/** The nearest candidate within `thresholdPx`, else t on the frame grid (grid starts at `gridOrigin`, the file's first frame). */
export function snap(t: number, candidates: readonly number[], pps: number, thresholdPx = 8, fps?: number, gridOrigin = 0): number {
  let best: number | null = null;
  let bestD = thresholdPx / pps;
  for (const c of candidates) {
    const d = Math.abs(c - t);
    if (d <= bestD) {
      best = c;
      bestD = d;
    }
  }
  if (best !== null) return best;
  if (!fps || !(fps > 0)) return t;
  return gridOrigin + Math.round((t - gridOrigin) * fps) / fps;
}

/** Where a stream-copy cut requested at `t` really starts: the last keyframe at or before t (cut.snap_to_keyframe). */
export function keyframeStart(keyframes: readonly number[], t: number): number {
  if (!keyframes.length) return t;
  let lo = 0;
  let hi = keyframes.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (keyframes[mid] <= t + 1e-6) lo = mid + 1;
    else hi = mid;
  }
  return keyframes[Math.max(lo - 1, 0)];
}
