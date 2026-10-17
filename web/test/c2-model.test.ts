import { describe, expect, it } from 'vitest';
import { runLabel, slowmoLine } from '../src/model/format';
import { bucketMax, holdColumns, peakIn } from '../src/model/presence';
import { keyframeStart } from '../src/model/timeline';

describe('keyframeStart (where a stream copy really starts)', () => {
  const kf = [0, 2.5, 5, 7.5];
  it('is the last keyframe at or before the start', () => {
    expect(keyframeStart(kf, 6)).toBe(5);
    expect(keyframeStart(kf, 5)).toBe(5);
    expect(keyframeStart(kf, 5 - 1e-7)).toBe(5); // the same 1 us slack as cut.snap_to_keyframe
    expect(keyframeStart(kf, 100)).toBe(7.5);
  });
  it('keeps the start when there is nothing before it, or no list', () => {
    expect(keyframeStart(kf, -1)).toBe(0);
    expect(keyframeStart([], 3.3)).toBe(3.3);
  });
});

describe('peakIn', () => {
  const t = [0, 1, 2, 3, 4, 5];
  const s = [0.1, 0.9, 0.3, 0.95, 0.2, 0.1];
  it('finds the highest score inside the range, inclusive', () => {
    expect(peakIn(t, s, 1, 3)).toEqual({ t: 3, score: 0.95 });
    expect(peakIn(t, s, 1, 2)).toEqual({ t: 1, score: 0.9 });
  });
  it('is null when no sample falls inside', () => {
    expect(peakIn(t, s, 1.2, 1.8)).toBeNull();
  });
});

describe('holdColumns', () => {
  it('holds the last value across empty columns, for a limited run', () => {
    const cols = Float32Array.from([0.5, NaN, NaN, NaN, 0.2, NaN]);
    expect(Array.from(holdColumns(cols, 2))).toEqual([0.5, 0.5, 0.5, NaN, 0.2, 0.2].map((v) => (Number.isNaN(v) ? NaN : Math.fround(v))));
  });
  it('turns sparse samples into a step plot with bucketMax', () => {
    const cols = bucketMax([0, 5], [1, 0.4], 0, 10, 10);
    const held = holdColumns(cols, 4);
    expect(held[3]).toBe(1);
    expect(Number.isNaN(held[4])).toBe(false);
    expect(held[9]).toBeCloseTo(0.4, 5);
  });
});

describe('texts', () => {
  it('writes the slow-motion line with source and output seconds', () => {
    expect(slowmoLine(2, 8)).toBe('slow-mo x8: source 2.0 s, output 16.0 s');
  });
  it('labels a run by target and time', () => {
    const when = new Date(2026, 9, 16, 10, 12).getTime() / 1000;
    expect(runLabel({ target: 'saved:alice', slug: 'saved-alice', created: when })).toBe('saved: alice · 10:12');
    expect(runLabel({ target: 'tap', slug: 'tap', created: new Date(2026, 9, 16, 9, 5).toISOString() })).toBe('tap · 09:05');
    expect(runLabel({ target: '', slug: 'motion', created: 'not a date' })).toBe('motion');
  });
});
