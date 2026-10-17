// How long a video the browser engine can take, and when to say "use My Mac" (docs/T9B_DESIGN.md section 7). Pure functions over the constants
// below; the constants are what `tools/browser_limits.py` measured: synthetic 1080p30 HEVC at 8 Mbit/s,
// balanced any-person run, in Google Chrome with WebGPU on an Apple-silicon Mac.
import { span } from './estimate';

/** Safari builds the zip in memory, and a Blob near 2 GB is not reliable there; above this the export dialog says "use My Mac". */
export const BLOB_CAP = 2 * 1024 ** 3;
/** An estimate above this many seconds suggests My Mac. */
export const SUGGEST_AFTER_S = 600;
/** The peak resident memory (the whole browser's process tree) a run may reach before the length is called too long. */
export const PEAK_CEILING_MB = 4096;

export type BrowserKind = 'chrome' | 'safari';

/** One measured run of the balanced preset: video minutes, wall seconds, the detector's share per picture, peak resident MB of the browser's processes. */
export interface Measured {
  minutes: number;
  wallS: number;
  /** `detect_model_s` over the run's samples, ms */
  detectMs: number;
  peakMb: number;
}

export interface BrowserLimits {
  /** the balanced runs measured (5 samples per second); the detector slows down on a long run (the GPU is kept busy for minutes), so its per-picture cost
   *  is read from these by length. The longest run that stayed under PEAK_CEILING_MB bounds the length the page calls safe. */
  runs: Measured[];
  /** everything a balanced run costs besides the detector, seconds per video minute (decode, drawing 5 pictures a second, bookkeeping) */
  otherSecPerMin: number;
  /** the fast preset (keyframes only): the detector per picture, ms (the GPU idles between pictures), and the rest per video minute */
  fastDetectMs: number;
  fastOtherSecPerMin: number;
  /** DINOv2-S, ms per crop (tap runs) [T9-0; a whole tap run was not measured] */
  embedMs: number;
  /** true when the numbers are scaled from Chrome's and were not run in that browser */
  unverified: boolean;
}

// MEASURED_BEGIN (tools/browser_limits.py, 2026-10-16, Chrome headless, load average 1.9 to 2.7)
export const CHROME: BrowserLimits = {
  runs: [
    { minutes: 5, wallS: 95.6, detectMs: 56.9, peakMb: 2225 },
    { minutes: 20, wallS: 545.6, detectMs: 83.0, peakMb: 2183 },
    { minutes: 60, wallS: 1830.4, detectMs: 93.6, peakMb: 2258 },
  ],
  otherSecPerMin: 2.4,
  fastDetectMs: 90,
  fastOtherSecPerMin: 0.6,
  embedMs: 19,
  unverified: false,
};
// MEASURED_END

/** Safari 26.5 ran the detector about 1.9 times slower and DINOv2 1.6 times slower in the T9-0 spike (116 and 30 ms against 61 and 19); a whole run in Safari
 *  was not measured [unverified], so its numbers are Chrome's scaled the same way and its memory is taken as Chrome's. */
const SLOWER = 116 / 61;
export const SAFARI: BrowserLimits = {
  runs: CHROME.runs.map((r) => ({ ...r, wallS: r.wallS * SLOWER, detectMs: r.detectMs * SLOWER })),
  otherSecPerMin: CHROME.otherSecPerMin * SLOWER,
  fastDetectMs: CHROME.fastDetectMs * SLOWER,
  fastOtherSecPerMin: CHROME.fastOtherSecPerMin * SLOWER,
  embedMs: 30,
  unverified: true,
};

/** The detector's ms per picture in a balanced run of `minutes`: between the measured runs on a straight line, the nearest one beyond them. */
export function detectMsAt(l: BrowserLimits, minutes: number): number {
  const r = l.runs;
  if (minutes <= r[0].minutes) return r[0].detectMs;
  for (let i = 1; i < r.length; i++) {
    if (minutes <= r[i].minutes) return r[i - 1].detectMs + ((r[i].detectMs - r[i - 1].detectMs) * (minutes - r[i - 1].minutes)) / (r[i].minutes - r[i - 1].minutes);
  }
  return r[r.length - 1].detectMs;
}

/** What My Mac needs, seconds per video minute: a 20-minute video takes about 150 s on the default CPU_AND_GPU units (README, tools/regress.py) (approximate). The fast preset reads keyframes only with the nano model; its figure is a guess [unverified]. */
export const MAC_SEC_PER_MIN: Record<string, number> = { balanced: 7.4, fast: 1.5, accurate: 9 };
/** Samples per second of the balanced preset, and the crops DINOv2 reads per picture in a tap run (a guess for a busy scene [unverified]). */
export const BALANCED_FPS = 5;
export const TAP_CROPS_PER_SAMPLE = 3;

export const limitsOf = (b: BrowserKind): BrowserLimits => (b === 'safari' ? SAFARI : CHROME);

/** Chrome unless the user agent says Safari (Chrome and Edge also say "Safari"). */
export function browserKindOf(userAgent: string): BrowserKind {
  return /Safari\//.test(userAgent) && !/Chrome\/|Chromium\/|Edg\/|OPR\//.test(userAgent) ? 'safari' : 'chrome';
}

/** The longest video, in minutes, a measured run kept under the memory ceiling. */
export function longestMeasuredMin(b: BrowserKind): number {
  return Math.max(0, ...limitsOf(b).runs.filter((r) => r.peakMb < PEAK_CEILING_MB).map((r) => r.minutes));
}

export interface EstimateInput {
  /** seconds of video */
  duration: number;
  preset: string;
  target: 'any' | 'class' | 'tap' | 'motion' | 'saved' | 'photos';
  browser: BrowserKind;
  /** keyframes of the video (the fast preset looks at those only) */
  keyframes: number;
  /** samples per second the user asked for, else the preset's */
  fps?: number | null;
}

export interface BrowserEstimate {
  /** seconds in this browser */
  seconds: number;
  /** seconds on My Mac */
  macSeconds: number;
  samples: number;
}

/** seconds = video minutes x other + pictures x (detector + crops x DINOv2); My Mac is video minutes x its rate. */
export function estimate(i: EstimateInput): BrowserEstimate {
  const l = limitsOf(i.browser);
  const minutes = Math.max(0, i.duration) / 60;
  const keyframesOnly = i.preset === 'fast' && !(i.fps && i.fps > 0);
  const fps = i.fps && i.fps > 0 ? i.fps : BALANCED_FPS;
  const samples = keyframesOnly ? i.keyframes : Math.round(Math.max(0, i.duration) * fps);
  const perSampleMs = (keyframesOnly ? l.fastDetectMs : detectMsAt(l, minutes)) + (i.target === 'tap' ? TAP_CROPS_PER_SAMPLE * l.embedMs : 0);
  return {
    seconds: minutes * (keyframesOnly ? l.fastOtherSecPerMin : l.otherSecPerMin) + (samples * perSampleMs) / 1000,
    macSeconds: minutes * (MAC_SEC_PER_MIN[i.preset] ?? MAC_SEC_PER_MIN.balanced),
    samples,
  };
}

export interface BrowserCaps {
  webcodecs: boolean;
  /** null = not asked (no HEVC check was made) */
  hevc: boolean | null;
  webgpu: boolean;
}

export interface LocalSuggestion {
  suggest: boolean;
  /** why, one clause each, most important first */
  reasons: string[];
  /** the whole sentence for the panel, or '' */
  text: string;
}

export const RUN_SERVE = 'run `inframe serve` on your Mac';

/** When the page should say "use My Mac": the browser cannot decode this video, has no WebGPU, the estimate passes SUGGEST_AFTER_S, or the video is
 *  longer than the longest one measured under the memory ceiling. */
export function suggestLocal(o: { duration: number; codec: string; browser: BrowserKind; est: BrowserEstimate; caps: BrowserCaps }): LocalSuggestion {
  const reasons: string[] = [];
  if (!o.caps.webcodecs) reasons.push('this browser cannot decode video');
  else if (o.caps.hevc === false && o.codec.toLowerCase() === 'hevc') reasons.push('this browser cannot decode HEVC video (iPhone video)');
  if (!o.caps.webgpu) reasons.push('without WebGPU the models run on the CPU, about 19 times slower');
  const longest = longestMeasuredMin(o.browser);
  if (o.est.seconds > SUGGEST_AFTER_S) reasons.push(`about ${span(o.est.seconds)} here, about ${span(o.est.macSeconds)} on My Mac`);
  if (o.duration / 60 > longest) reasons.push(`longer than the ${longest} min this browser was measured on, so it may run out of memory`);
  if (!reasons.length) return { suggest: false, reasons, text: '' };
  const [first, ...rest] = reasons;
  return { suggest: true, reasons, text: `Use My Mac for this one (${RUN_SERVE}): ${first}${rest.length ? '; ' + rest.join('; ') : ''}.` };
}
