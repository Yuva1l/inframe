// The browser engine's limits (src/model/limits.ts, docs/T9B_DESIGN.md section 7): the estimate and when the page says "use My Mac".
import { describe, expect, it } from 'vitest';
import { BALANCED_FPS, BLOB_CAP, CHROME, detectMsAt, MAC_SEC_PER_MIN, PEAK_CEILING_MB, SAFARI, SUGGEST_AFTER_S, browserKindOf, estimate, limitsOf, longestMeasuredMin, suggestLocal, TAP_CROPS_PER_SAMPLE, type BrowserEstimate } from '../src/model/limits';

const ok = { webcodecs: true, hevc: true, webgpu: true };
const est = (seconds: number, macSeconds = 60): BrowserEstimate => ({ seconds, macSeconds, samples: 0 });
const base = { duration: 300, codec: 'hevc', browser: 'chrome' as const, est: est(100), caps: ok };

describe('estimate', () => {
  it('balanced any-person: minutes x the rest + pictures x the detector, and it reproduces the measured runs', () => {
    const e = estimate({ duration: 600, preset: 'balanced', target: 'any', browser: 'chrome', keyframes: 600 });
    expect(e.samples).toBe(600 * BALANCED_FPS);
    expect(e.seconds).toBeCloseTo(10 * CHROME.otherSecPerMin + (3000 * detectMsAt(CHROME, 10)) / 1000, 6);
    for (const r of CHROME.runs) {
      const x = estimate({ duration: r.minutes * 60, preset: 'balanced', target: 'any', browser: 'chrome', keyframes: r.minutes * 60 });
      expect(Math.abs(x.seconds - r.wallS) / r.wallS, `${r.minutes} min: estimated ${x.seconds.toFixed(0)} s, measured ${r.wallS} s`).toBeLessThan(0.1);
    }
  });
  it('the fast preset looks at keyframes only, unless the user asks for a rate', () => {
    const a = estimate({ duration: 600, preset: 'fast', target: 'any', browser: 'chrome', keyframes: 120 });
    expect(a.samples).toBe(120);
    expect(a.seconds).toBeCloseTo(10 * CHROME.fastOtherSecPerMin + (120 * CHROME.fastDetectMs) / 1000, 6);
    expect(estimate({ duration: 600, preset: 'fast', target: 'any', browser: 'chrome', keyframes: 120, fps: 2 }).samples).toBe(1200);
    expect(estimate({ duration: 600, preset: 'balanced', target: 'any', browser: 'chrome', keyframes: 120, fps: 10 }).samples).toBe(6000);
  });
  it('a tap run adds the recognition crops of every picture', () => {
    const any = estimate({ duration: 600, preset: 'balanced', target: 'any', browser: 'chrome', keyframes: 1 });
    const tap = estimate({ duration: 600, preset: 'balanced', target: 'tap', browser: 'chrome', keyframes: 1 });
    expect(tap.seconds - any.seconds).toBeCloseTo((3000 * TAP_CROPS_PER_SAMPLE * CHROME.embedMs) / 1000, 6);
  });
  it('Safari takes longer than Chrome, and My Mac is the preset\'s rate per video minute', () => {
    const c = estimate({ duration: 1200, preset: 'balanced', target: 'any', browser: 'chrome', keyframes: 1 });
    const s = estimate({ duration: 1200, preset: 'balanced', target: 'any', browser: 'safari', keyframes: 1 });
    expect(s.seconds).toBeGreaterThan(1.5 * c.seconds);
    expect(c.macSeconds).toBeCloseTo(20 * MAC_SEC_PER_MIN.balanced, 6);
    expect(estimate({ duration: 1200, preset: 'unknown', target: 'any', browser: 'chrome', keyframes: 1 }).macSeconds).toBeCloseTo(20 * MAC_SEC_PER_MIN.balanced, 6);
    expect(SAFARI.unverified).toBe(true);
    expect(CHROME.unverified).toBe(false);
  });
  it('an empty video costs nothing', () => {
    expect(estimate({ duration: 0, preset: 'balanced', target: 'any', browser: 'chrome', keyframes: 0 })).toEqual({ seconds: 0, macSeconds: 0, samples: 0 });
  });
});

describe('the detector slows down on a long run', () => {
  it('detectMsAt reads the measured runs, a straight line between them, and stays flat outside', () => {
    const [a, b, c] = CHROME.runs;
    expect(detectMsAt(CHROME, 1)).toBe(a.detectMs);
    expect(detectMsAt(CHROME, a.minutes)).toBe(a.detectMs);
    expect(detectMsAt(CHROME, b.minutes)).toBe(b.detectMs);
    expect(detectMsAt(CHROME, (a.minutes + b.minutes) / 2)).toBeCloseTo((a.detectMs + b.detectMs) / 2, 9);
    expect(detectMsAt(CHROME, c.minutes)).toBe(c.detectMs);
    expect(detectMsAt(CHROME, 600)).toBe(c.detectMs);
  });
  it('so a long video costs more per minute than a short one', () => {
    const short = estimate({ duration: 300, preset: 'balanced', target: 'any', browser: 'chrome', keyframes: 1 });
    const long = estimate({ duration: 3600, preset: 'balanced', target: 'any', browser: 'chrome', keyframes: 1 });
    expect(long.seconds / 60).toBeGreaterThan(1.3 * (short.seconds / 5));
  });
});

describe('browser kind', () => {
  it('Safari says Safari/ and no Chrome; Chrome, Edge and Opera also say Safari/', () => {
    expect(browserKindOf('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Safari/605.1.15')).toBe('safari');
    expect(browserKindOf('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36')).toBe('chrome');
    expect(browserKindOf('Mozilla/5.0 AppleWebKit/537.36 Chrome/141.0 Safari/537.36 Edg/141.0')).toBe('chrome');
    expect(browserKindOf('')).toBe('chrome');
    expect(limitsOf('safari')).toBe(SAFARI);
  });
});

describe('longest measured length', () => {
  it('is the longest run that stayed under the memory ceiling', () => {
    const under = CHROME.runs.filter((r) => r.peakMb < PEAK_CEILING_MB).map((r) => r.minutes);
    expect(longestMeasuredMin('chrome')).toBe(Math.max(...under));
    expect(longestMeasuredMin('chrome')).toBeGreaterThan(0);
  });
});

describe('suggestLocal thresholds', () => {
  it('says nothing for a short, supported video', () => {
    expect(suggestLocal(base)).toEqual({ suggest: false, reasons: [], text: '' });
  });
  it(`the estimate passes ${SUGGEST_AFTER_S} s: at the line it stays quiet, one second over it suggests My Mac with both times`, () => {
    expect(suggestLocal({ ...base, est: est(SUGGEST_AFTER_S) }).suggest).toBe(false);
    const s = suggestLocal({ ...base, est: est(SUGGEST_AFTER_S + 1, 150) });
    expect(s.suggest).toBe(true);
    expect(s.text).toContain('about 10 min here, about 3 min on My Mac');
    expect(s.text).toContain('inframe serve');
  });
  it('a video longer than the longest measured one suggests My Mac, one at that length does not', () => {
    const longest = longestMeasuredMin('chrome');
    expect(suggestLocal({ ...base, duration: longest * 60 }).suggest).toBe(false);
    const s = suggestLocal({ ...base, duration: longest * 60 + 1 });
    expect(s.suggest).toBe(true);
    expect(s.reasons.join(' ')).toContain(`${longest} min`);
  });
  it('no video decoding, no HEVC for an HEVC file (H.264 is fine), no WebGPU: each says why', () => {
    expect(suggestLocal({ ...base, caps: { ...ok, webcodecs: false, hevc: null } }).reasons[0]).toMatch(/cannot decode video/);
    expect(suggestLocal({ ...base, caps: { ...ok, hevc: false } }).reasons[0]).toMatch(/HEVC/);
    expect(suggestLocal({ ...base, codec: 'h264', caps: { ...ok, hevc: false } }).suggest).toBe(false);
    expect(suggestLocal({ ...base, caps: { ...ok, hevc: null } }).suggest).toBe(false);
    const g = suggestLocal({ ...base, caps: { ...ok, webgpu: false } });
    expect(g.suggest).toBe(true);
    expect(g.reasons[0]).toMatch(/WebGPU.*19 times/);
  });
  it('several reasons come in order, the decoder first', () => {
    const s = suggestLocal({ ...base, duration: 99999, est: est(5000), caps: { webcodecs: true, hevc: false, webgpu: false } });
    expect(s.reasons).toHaveLength(4);
    expect(s.reasons[0]).toMatch(/HEVC/);
    expect(s.text.startsWith('Use My Mac for this one')).toBe(true);
  });
});

describe('constants', () => {
  it('the Safari export cap is 2 GB', () => {
    expect(BLOB_CAP).toBe(2 * 1024 ** 3);
  });
});
