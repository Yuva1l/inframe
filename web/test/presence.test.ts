import { describe, expect, it } from 'vitest';
import { bucketMax, sampleAt } from '../src/model/presence';

describe('bucketMax', () => {
  const n = 6000;
  const t = Array.from({ length: n }, (_, i) => i * 0.2);
  const score = t.map((x) => Math.abs(Math.sin(x / 7)));

  it('buckets 6000 samples into 1200 columns by the maximum', () => {
    const out = bucketMax(t, score, 0, 1200, 1200);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out).toHaveLength(1200);
    for (const c of [0, 1, 77, 600, 1199]) {
      const want = Math.max(...score.slice(c * 5, c * 5 + 5));
      expect(out[c]).toBeCloseTo(want, 5);
    }
    expect(out.every((v) => !Number.isNaN(v))).toBe(true);
  });
  it('leaves empty columns NaN and ignores samples outside the range', () => {
    const out = bucketMax([1, 2, 50], [0.4, 0.9, 0.7], 0, 10, 10);
    expect(out[1]).toBeCloseTo(0.4);
    expect(out[2]).toBeCloseTo(0.9);
    expect(Number.isNaN(out[0])).toBe(true);
    expect(Number.isNaN(out[9])).toBe(true);
  });
  it('zooms into a part of the video', () => {
    const out = bucketMax(t, score, 100, 110, 10);
    expect(out[0]).toBeCloseTo(Math.max(...score.slice(500, 505)), 5);
  });
  it('puts a sample at the very end in the last column', () => {
    expect(bucketMax([10], [0.5], 0, 10, 4)[3]).toBeCloseTo(0.5);
  });
  it('copes with no samples and an empty range', () => {
    expect(bucketMax([], [], 0, 10, 5).every(Number.isNaN)).toBe(true);
    expect(bucketMax([1], [1], 5, 5, 5).every(Number.isNaN)).toBe(true);
    expect(bucketMax([1], [1], 0, 5, 0)).toHaveLength(0);
  });
});

describe('sampleAt', () => {
  it('finds the last sample at or before t, within the gap', () => {
    const ts = [0, 0.2, 0.4, 5];
    expect(sampleAt(ts, 0.3)).toBe(1);
    expect(sampleAt(ts, 0.4)).toBe(2);
    expect(sampleAt(ts, 3)).toBe(-1);
    expect(sampleAt(ts, -1)).toBe(-1);
    expect(sampleAt(ts, 5.5)).toBe(3);
  });
});
