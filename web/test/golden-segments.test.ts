// The TypeScript segments port against the golden vectors Python made (tests/golden/segments_v1.json, tools/gen_golden.py).
import { describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so the file read is allowed to be untyped
// @ts-ignore
import { readFileSync } from 'node:fs';
import { buildSegments, dropShort, hysteresis, mergeGaps, pad, resegment, sampleEdges, type Interval } from '../src/model/segments';

const TOL = 1e-9;

interface Expect { edges: number[]; hysteresis: Interval[]; merged: Interval[]; kept: Interval[]; presence: Interval[]; clips: Interval[] }
interface Case {
  name: string; kind: 'hand' | 'random'; times: number[]; scores: number[]; duration: number;
  enter: number; exit: number; merge_gap: number; min_len: number; pad: number; lo: number;
  expect?: Expect; error?: 'exit_gt_enter' | 'length_mismatch';
}
interface ReCase {
  name: string; t: number[]; score: number[]; settings: Record<string, unknown>; params: Record<string, number | null>;
  offset: number; end: number; expect?: { presence: Interval[]; clips: Interval[] }; error?: string;
}
const doc = JSON.parse(readFileSync((globalThis as any).process.cwd() + '/../tests/golden/segments_v1.json', 'utf8')) as { cases: Case[]; resegment: ReCase[] };

function near(got: readonly (readonly number[])[] | readonly number[], want: readonly (readonly number[])[] | readonly number[], what: string) {
  const g = (got as any[]).flat();
  const w = (want as any[]).flat();
  expect(g.length, `${what}: count`).toBe(w.length);
  for (let i = 0; i < w.length; i++) expect(Math.abs(g[i] - w[i]), `${what}[${i}]: ${g[i]} vs ${w[i]}`).toBeLessThanOrEqual(TOL);
}

describe('golden segments', () => {
  it('has the cases', () => {
    expect(doc.cases.length).toBeGreaterThanOrEqual(65);
    expect(doc.resegment.length).toBeGreaterThanOrEqual(8);
  });

  for (const c of doc.cases.filter((x) => x.expect)) {
    it(`${c.name}: every stage matches`, () => {
      const e = c.expect!;
      near(sampleEdges(c.times, c.duration, c.lo), e.edges, 'edges');
      const h = hysteresis(c.times, c.scores, c.enter, c.exit, c.duration, c.lo);
      near(h, e.hysteresis, 'hysteresis');
      const m = mergeGaps(h, c.merge_gap);
      near(m, e.merged, 'merged');
      near(dropShort(m, c.min_len), e.kept, 'kept');
      const r = buildSegments(c.times, c.scores, c.duration, { enter: c.enter, exit: c.exit, mergeGap: c.merge_gap, minLen: c.min_len, pad: c.pad, lo: c.lo });
      near(r.presence, e.presence, 'presence');
      near(r.clips, e.clips, 'clips');
      near(pad(e.kept, c.pad, c.pad, c.duration, c.lo), e.clips, 'pad');
    });
  }

  for (const c of doc.cases.filter((x) => x.error)) {
    it(`${c.name}: throws`, () => {
      expect(() => hysteresis(c.times, c.scores, c.enter, c.exit, c.duration, c.lo)).toThrow(c.error === 'length_mismatch' ? /differ in length/ : /must not exceed/);
    });
  }

  for (const c of doc.resegment) {
    it(`resegment ${c.name}`, () => {
      const call = () => resegment({ t: c.t, score: c.score }, c.settings, c.params, { offset: c.offset, end: c.end });
      if (c.error) {
        expect(call).toThrow(c.error === 'no_value' ? /no value to start from/ : /must not exceed/);
        return;
      }
      const r = call();
      near(r.presence, c.expect!.presence, 'presence');
      near(r.clips, c.expect!.clips, 'clips');
    });
  }

  it('mergeGaps sorts by start then end, like Python tuples', () => {
    expect(mergeGaps([[5, 6], [1, 4], [1, 2]], 0)).toEqual([[1, 4], [5, 6]]);
  });

  it('uses the defaults of segments.py', () => {
    const a = buildSegments([0, 1, 2, 3, 4, 5], [0, 0.9, 0.9, 0.1, 0, 0], 5.5);
    const b = buildSegments([0, 1, 2, 3, 4, 5], [0, 0.9, 0.9, 0.1, 0, 0], 5.5, { enter: 0.5, exit: 0.3, mergeGap: 1.5, minLen: 1.0, pad: 1.0, lo: 0 });
    expect(a).toEqual(b);
  });
});
