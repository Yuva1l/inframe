// The TypeScript detector post- and preprocess, the identity-free targets and the COCO class table against the golden vectors Python made
// (tests/golden/detect_v1.json and targets_v1.json, tools/gen_golden.py).
import { describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so the file read is allowed to be untyped
// @ts-ignore
import { readFileSync } from 'node:fs';
import { postprocess, preprocess } from '../src/browser/detect';
import { ALIASES, AnyOfClass, AnyPerson, classId, COCO_91, UnknownClass, Zone, ZoneTarget, type Hit, type Target } from '../src/browser/targets';

const read = (n: string) => JSON.parse(readFileSync((globalThis as any).process.cwd() + `/../tests/golden/${n}`, 'utf8'));
const detect = read('detect_v1.json') as {
  preprocess: { name: string; side: number; rgba: number[]; tensor: number[] }[];
  postprocess: { name: string; queries: number; classes: number; cls: number; min_score: number; logits: number[]; boxes: number[]; expect?: number[][]; error?: string }[];
};
const targets = read('targets_v1.json') as {
  coco_91: Record<string, string>; aliases: Record<string, string>;
  class_ids: { name: string; id?: number; error?: string }[];
  zones: { zone: number[]; valid: boolean }[];
  cases: { name: string; kind: 'any' | 'class' | 'first'; cls: number | null; min_box: number; zone: number[] | null; zone_overlap: number | null; steps: number[][][]; expect: { score: number; box: number[] | null }[]; target_name: string }[];
};

const TOL = 1e-6;

describe('golden preprocess', () => {
  for (const c of detect.preprocess) {
    it(`${c.name}: the tensor matches`, () => {
      const t = preprocess(c.rgba, c.side, c.side);
      expect(t.length).toBe(c.tensor.length);
      let worst = 0;
      for (let i = 0; i < t.length; i++) worst = Math.max(worst, Math.abs(t[i] - c.tensor[i]));
      expect(worst).toBeLessThanOrEqual(TOL);
    });
  }
  it('is channels first with the ImageNet mean and std', () => {
    const t = preprocess([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 124, 116, 104, 255], 2, 2);
    expect(Array.from(t.slice(0, 4)).map((v) => +v.toFixed(3))).toEqual([2.249, -2.118, -2.118, 0.006]); // red channel of each pixel
    expect(+t[4 + 1].toFixed(3)).toBe(2.429); // green channel of pixel 1
    expect(() => preprocess([0, 0, 0, 0], 2, 2)).toThrow();
  });
});

describe('golden postprocess', () => {
  for (const c of detect.postprocess.filter((x) => x.expect)) {
    it(`${c.name}: detections match`, () => {
      const got = postprocess(c.boxes, c.logits, c.queries, c.classes, c.cls, c.min_score);
      expect(got.length).toBe(c.expect!.length);
      got.forEach((d, i) => {
        const want = c.expect![i];
        expect(Math.abs(d.score - want[0]), `score ${i}`).toBeLessThanOrEqual(TOL);
        d.box.forEach((v, k) => expect(Math.abs(v - want[1 + k]), `box ${i}.${k}`).toBeLessThanOrEqual(1e-9));
      });
    });
  }
  it('refuses a class outside the model slots', () => {
    const c = detect.postprocess.find((x) => x.error)!;
    expect(() => postprocess(c.boxes, c.logits, c.queries, c.classes, c.cls, c.min_score)).toThrow(/outside/);
  });
  it('anchors: sigmoid(0) is 0.5, centre and size make the box, best first', () => {
    const boxes = [0.5, 0.5, 0.2, 0.4, 0.25, 0.75, 0.5, 0.5];
    const logits = new Array(2 * 91).fill(-10);
    logits[1] = 0;
    logits[91 + 1] = 2;
    const got = postprocess(boxes, logits, 2, 91);
    expect(got.map((d) => +d.score.toFixed(4))).toEqual([0.8808, 0.5]);
    expect(got[1].box.map((v) => +v.toFixed(4))).toEqual([0.4, 0.3, 0.6, 0.7]);
  });
});

describe('golden targets', () => {
  it('has the same COCO table and aliases', () => {
    expect(Object.fromEntries(Object.entries(COCO_91).map(([k, v]) => [k, v]))).toEqual(targets.coco_91);
    expect(ALIASES).toEqual(targets.aliases);
  });
  for (const c of targets.class_ids) {
    it(`class_id(${JSON.stringify(c.name)})`, () => {
      if (c.id !== undefined) expect(classId(c.name)).toBe(c.id);
      else {
        expect(() => classId(c.name)).toThrow(UnknownClass);
        try {
          classId(c.name);
        } catch (e) {
          expect((e as Error).message).toBe(c.error);
        }
      }
    });
  }
  it('zone validity matches', () => {
    for (const z of targets.zones) {
      const make = () => new Zone(z.zone[0], z.zone[1], z.zone[2], z.zone[3]);
      if (z.valid) expect(make).not.toThrow();
      else expect(make).toThrow();
    }
  });

  class First implements Target {
    name = 'first';
    cls = 1;
    stateless = false;
    evaluate(cands: readonly { score: number; box: [number, number, number, number] }[]): Hit {
      return cands.length ? { score: cands[0].score, box: cands[0].box, trackId: null } : { score: 0, box: null, trackId: null };
    }
  }

  for (const c of targets.cases) {
    it(`${c.name}: every step matches`, () => {
      const inner: Target = c.kind === 'any' ? new AnyPerson(c.min_box) : c.kind === 'class' ? new AnyOfClass(c.cls!, c.min_box) : new First();
      const t: Target = c.zone ? new ZoneTarget(inner, c.zone as [number, number, number, number], c.zone_overlap) : inner;
      expect(t.name).toBe(c.target_name);
      c.steps.forEach((step, i) => {
        const hit = t.evaluate(step.map((x) => ({ score: x[0], box: [x[1], x[2], x[3], x[4]] as [number, number, number, number] })));
        expect(hit.score).toBe(c.expect[i].score);
        expect(hit.box).toEqual(c.expect[i].box);
      });
    });
  }
  it('a stateless target in a zone only sees the people in it (anchor)', () => {
    const tall = { score: 0.9, box: [0, 0, 0.1, 0.5] as [number, number, number, number] };
    const short = { score: 0.8, box: [0.5, 0.5, 0.6, 0.55] as [number, number, number, number] };
    expect(new ZoneTarget(new AnyPerson(), [0.4, 0.4, 0.4, 0.4]).evaluate([tall, short]).score).toBe(0.8);
    expect(new ZoneTarget(new First(), [0.4, 0.4, 0.4, 0.4]).evaluate([tall, short]).box).toBeNull();
  });
});
