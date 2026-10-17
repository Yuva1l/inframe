// resegment() checks its numbers like serve_runs.resegment (_number(v, k, 0.0)): finite numbers, not below 0.
import { describe, expect, it } from 'vitest';
import { resegment } from '../src/model/segments';

const samples = { t: [1, 1.5, 2, 2.5, 3], score: [0, 0.9, 0.9, 0.9, 0] };
const settings = { enter: 0.5, exit: 0.3, merge_gap: 1.5, min_len: 1, pad: 1 };
const bounds = { offset: 1, end: 3.25 };
const run = (params: Record<string, unknown>, s: Record<string, unknown> = settings) => resegment(samples, s, params as never, bounds);

describe('resegment validates its numbers like the server', () => {
  it('still works with sane values', () => {
    const r = run({ pad: 0.5 });
    expect(r.clips.length).toBe(1);
    expect(r.clips[0][1]).toBeGreaterThan(r.clips[0][0]);
  });
  for (const k of ['enter', 'exit', 'merge_gap', 'min_len', 'pad']) {
    it(`${k}: negative is refused`, () => expect(() => run({ [k]: -1 })).toThrow(new RegExp(`${k} must be at least 0`)));
    it(`${k}: NaN, Infinity, strings and booleans are refused`, () => {
      for (const bad of [NaN, Infinity, -Infinity, '0.5', true, false, {}, []]) expect(() => run({ [k]: bad }), String(bad)).toThrow(new RegExp(`${k} must be a finite number`));
    });
    it(`${k}: a bad value in the run's settings is refused too`, () => {
      expect(() => run({}, { ...settings, [k]: '0.5' })).toThrow(/must be a finite number/);
      expect(() => run({}, { ...settings, [k]: -2 })).toThrow(/must be at least 0/);
    });
  }
  it('zero is fine', () => {
    expect(() => run({ pad: 0, exit: 0, min_len: 0, merge_gap: 0 })).not.toThrow();
  });
});
