import { describe, expect, it } from 'vitest';
import { estimate, estimateText, median, pushTiming } from '../src/model/estimate';
import { editListIssue, plan } from '../src/model/exportplan';
import { blocking, preflightRows } from '../src/model/preflight';
import { fromDrag, MIN_SIDE, move, resize, roundZone } from '../src/model/zone';
import { cleanMood } from '../src/ui/MusicPicker';
import { replaceText, outputRows } from '../src/ui/ExportDialog';
import { slugOf } from '../src/workflow';

describe('zone maths', () => {
  it('draws between two points, clamped to the frame, and refuses a side under 0.02', () => {
    expect(fromDrag([0.1, 0.2], [0.5, 0.6])).toEqual([0.1, 0.2, 0.4, 0.4]);
    expect(fromDrag([0.5, 0.6], [0.1, 0.2])).toEqual([0.1, 0.2, 0.4, 0.4]);
    expect(fromDrag([0.9, 0.9], [1.4, 1.4])).toEqual([0.9, 0.9, 0.1, 0.1]);
    expect(fromDrag([0.1, 0.1], [0.11, 0.5])).toBeNull();
    expect(fromDrag([0.1, 0.1], [0.5, 0.119])).toBeNull();
  });

  it('rounds like pick._round_zone: x and y to 4 decimals, w and h from the rounded far edge, never past 1', () => {
    expect(roundZone([0.123456, 0.2, 0.3, 0.4])).toEqual([0.1235, 0.2, 0.3, 0.4]);
    expect(roundZone([0.9, 0.9, 0.2, 0.2])).toEqual([0.9, 0.9, 0.1, 0.1]);
  });

  it('resizes by a handle with the opposite side fixed, the minimum side and the frame edge holding', () => {
    const z: [number, number, number, number] = [0.2, 0.2, 0.4, 0.4];
    expect(resize(z, 'se', 0.1, 0.2)).toEqual([0.2, 0.2, 0.5, 0.6]);
    expect(resize(z, 'nw', 0.1, 0.1)).toEqual([0.3, 0.3, 0.3, 0.3]);
    expect(resize(z, 'e', -1, 0)).toEqual([0.2, 0.2, MIN_SIDE, 0.4]); // cannot cross the west edge
    expect(resize(z, 'w', -1, 0)).toEqual([0, 0.2, 0.6, 0.4]); // stops at the frame
    expect(resize(z, 'n', 0, 5)).toEqual([0.2, 0.58, 0.4, MIN_SIDE]);
  });

  it('moves the body and keeps it inside the frame', () => {
    expect(move([0.2, 0.2, 0.4, 0.4], 0.1, -0.1)).toEqual([0.3, 0.1, 0.4, 0.4]);
    expect(move([0.2, 0.2, 0.4, 0.4], 5, 5)).toEqual([0.6, 0.6, 0.4, 0.4]);
    expect(move([0.2, 0.2, 0.4, 0.4], -5, -5)).toEqual([0, 0, 0.4, 0.4]);
  });
});

describe('the estimate', () => {
  const base = { preset: 'balanced', duration: 1200, keyframes: 100, motion: false };
  it('says at least, from the model-call cost, before any run', () => {
    const e = estimate({ ...base, timings: [] })!;
    expect(e.basis).toBe('model');
    expect(e.seconds).toBeCloseTo(1200 * 5 * 0.011, 6); // 6000 samples at 11.0 ms (GPU default)
    expect(estimateText(e)).toBe("at least 65 s (estimate)"); // 66 s, rounded down so "at least" holds
  });
  it('counts keyframes for the fast preset and has none for motion mode', () => {
    expect(estimate({ ...base, preset: 'fast', timings: [] })!.seconds).toBeCloseTo(100 * 0.005, 6);
    expect(estimate({ ...base, motion: true, timings: [] })).toBeNull();
    expect(estimateText(null)).toContain('after the first run');
  });
  it('says about, from the median of the last five measured runs', () => {
    let t: number[] = [];
    for (const v of [10, 50, 20, 30, 40, 60]) t = pushTiming(t, v);
    expect(t).toEqual([50, 20, 30, 40, 60]);
    expect(median(t)).toBe(40);
    const e = estimate({ ...base, timings: t })!;
    expect(e).toEqual({ seconds: 800, basis: 'measured' });
    expect(estimateText(e)).toBe('about 13 min (estimate)');
    expect(pushTiming(t, NaN)).toEqual(t);
  });
});

describe('preflight', () => {
  const pf = { ffmpeg: true, models: { fast: true, balanced: true, accurate: false }, identity: false, tracking: false, music: false };
  const none = { identity: false, tracking: false };
  it('checks only the chosen preset and only what the target needs', () => {
    expect(preflightRows(pf, { preset: 'balanced', ...none }).map((r) => r.key)).toEqual(['ffmpeg', 'model-balanced']);
    expect(blocking(preflightRows(pf, { preset: 'accurate', ...none })).map((r) => r.fix)).toEqual(['Run `inframe fetch-models`']);
    expect(blocking(preflightRows(pf, { preset: 'balanced', identity: true, tracking: true })).map((r) => r.fix)).toEqual(['Run `uv sync --extra identity`', 'Run `uv sync --extra tracking`']);
    expect(blocking(preflightRows(pf, { preset: 'balanced', identity: false, tracking: true })).map((r) => r.key)).toEqual(['tracking']); // a tap target
    expect(preflightRows(pf, { preset: 'accurate', motion: true, ...none }).map((r) => r.key)).toEqual(['ffmpeg']);
    expect(preflightRows(pf, { preset: null, identity: null, tracking: null }).length).toBe(6);
  });
});

describe('export plan', () => {
  const clips = [{ start: 5, end: 12 }, { start: 20, end: 28.5 }, { start: 40, end: 47 }];
  const kf = (t: number) => Math.floor(t / 4) * 4;
  it('counts all or the selected clips, the output length (slow motion included), the size and the early starts', () => {
    const all = plan({ clips, selected: [], selectedOnly: false, factor: 1, keyframeOf: kf, fileSize: 6000, duration: 60 });
    expect(all.text).toBe('3 clips - 22.5 s output - about 2.4 KB - 1 clip starts more than 0.5 s early at a keyframe');
    const one = plan({ clips, selected: [clips[0]], selectedOnly: true, factor: 8, keyframeOf: kf, fileSize: 6000, duration: 60 });
    expect(one.text).toMatch(/^1 clip \(selected\) of 3 - 56 s output - about /);
    expect(one.early).toBe(1);
  });
  it('gives the reasons of the edit-list rules of editlist.write_edit_lists', () => {
    expect(editListIssue('edl', 120)).toMatch(/60 fps or less/);
    expect(editListIssue('fcp7', 25.5)).toMatch(/standard frame rate/);
    expect(editListIssue('edl', 29.97)).toBeNull();
    expect(editListIssue('fcpxml', 240)).toBeNull();
    expect(editListIssue('otio', 25.5)).toBeNull();
  });
});

describe('small helpers', () => {
  it('keeps the mood to what the server accepts', () => {
    expect(cleanMood('chill/../x; rm')).toBe('chillx rm');
    expect(cleanMood('a'.repeat(60))).toHaveLength(40);
  });
  it('names the run folder of each target as the server does', () => {
    expect(slugOf({ kind: 'any' })).toBe('any-person');
    expect(slugOf({ kind: 'class', class: 'dog' })).toBe('class-dog');
    expect(slugOf({ kind: 'saved', name: 'alice' })).toBe('saved-alice');
    expect(slugOf({ kind: 'tap' })).toBe('tap');
  });
  it('words the replace question and the result checks', () => {
    expect(replaceText({ clips: [1, 2, 3], reel: { file: 'r.mp4' } })).toBe('Replace the export (3 clips, a reel)?');
    const rows = outputRows({
      outputs: [{ name: 'a.mp4', kind: 'clip', size: 1, url: '' }, { name: 'r.mp4', kind: 'reel', size: 2, url: '' }, { name: 'x.edl', kind: 'editlist', size: 3, url: '' }],
      clips: [{ file: 'a.mp4', error: 'ffmpeg failed' }], reel: { ok: true, check: { ok: true, lufs: -14.1 } }, edit_lists: [{ file: 'x.edl', ok: true, max_err_frames: 0.5 }],
    });
    expect(rows.map((r) => [r.ok, r.check])).toEqual([[false, 'ffmpeg failed'], [true, 'joins ok · -14.1 LUFS'], [true, 'largest cut error 0.5 frames']]);
  });
});
