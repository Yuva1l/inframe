// The tap gallery, unit_score, the person-crop geometry, pad_square and the DINOv2 input against the golden vectors Python made
// (tests/golden/gallery_v1.json, tools/gen_golden.py): identity.Gallery / unit_score / person_crop, embed_worker.pad_square and Pillow's
// bicubic resize with the processor's centre crop and normalisation.
import { describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so the file read and the hash are allowed to be untyped
// @ts-ignore
import { readFileSync } from 'node:fs';
// @ts-ignore
import { createHash } from 'node:crypto';
import { centerCrop, cropTensor, modelInput, padSquare, personCropRect, resizeBicubic, resizeDims } from '../src/browser/embed';
import { DINOV2, Gallery, unitScore, type Spec } from '../src/browser/gallery';

const doc = JSON.parse(readFileSync((globalThis as any).process.cwd() + '/../tests/golden/gallery_v1.json', 'utf8')) as {
  gallery: { name: string; dim: number; cap: number; spec: string; dedup: number; adds: { vec: number[]; sig: number[] | null }[]; steps: { joined: boolean; tags: number[] }[]; final: { tags: number[]; vectors: number[][] } }[];
  unit_score: { name: string; vecs: number[][]; gallery: number[][]; expect: number }[];
  crop_rect: { w: number; h: number; box: [number, number, number, number]; rect: [number, number, number, number] | null }[];
  pad_square: { name: string; h: number; w: number; seed: number; side: number; sha256: string }[];
  resize: { name: string; w: number; h: number; ow: number; oh: number; seed: number; sha256: string }[];
  pipeline: { name: string; fw: number; fh: number; box: [number, number, number, number]; seed: number; rect: number[] | null; side?: number; square_sha256?: string; crop224_sha256?: string; tensor?: { n: number; mean: number; samples: [number, number][] } }[];
};
const TOL = 1e-6;

const SPECS: Record<string, Spec> = {
  dinov2: DINOV2,
  siglip2: { threshold: 0.83, session: 0.9, margin: 0.02, grow: 0.87, dedup: 0.97 },
  color: { threshold: 0.85, session: 0.92, margin: 0.05, grow: 0.92, dedup: 0.98 },
};

/** The generator's pixel stream (tools/gen_golden.py _lcg_bytes). */
function lcgBytes(seed: number, n: number): Uint8Array {
  const out = new Uint8Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = s >>> 24;
  }
  return out;
}
const sha = (b: Uint8Array | Uint8ClampedArray): string => createHash('sha256').update(b).digest('hex');
const rgbaOf = (rgb: Uint8Array): Uint8ClampedArray => {
  const out = new Uint8ClampedArray((rgb.length / 3) * 4);
  for (let i = 0; i < rgb.length / 3; i++) out.set([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255], i * 4);
  return out;
};

describe('golden Gallery (identity.Gallery)', () => {
  it('has the cases the generator promised', () => {
    expect(doc.gallery.length).toBeGreaterThanOrEqual(10);
    expect(doc.gallery.some((c) => c.steps.some((s) => !s.joined))).toBe(true); // dedup fired somewhere
    expect(doc.gallery.some((c) => c.adds.some((a) => a.sig))).toBe(true); // colour signatures count in some
  });
  for (const c of doc.gallery) {
    it(`${c.name}: the same members join, dedup and eviction leave the same ones`, () => {
      const g = new Gallery<number>(SPECS[c.spec], c.cap);
      expect(g.spec.dedup).toBe(c.dedup);
      c.adds.forEach((a, i) => {
        const joined = g.add(Float32Array.from(a.vec), a.sig, i);
        expect(joined, `add ${i}`).toBe(c.steps[i].joined);
        expect(g.tags, `members after add ${i}`).toEqual(c.steps[i].tags);
      });
      expect(g.length).toBe(c.final.tags.length);
      g.v.forEach((row, k) => row.forEach((x, j) => expect(Math.abs(x - c.final.vectors[k][j])).toBeLessThanOrEqual(TOL)));
    });
  }
  it('refuses a vector of another length than its members', () => {
    const g = new Gallery();
    g.add(Float32Array.from([1, 0, 0]));
    expect(() => g.add(Float32Array.from([1, 0]))).toThrow(/cannot join/);
  });
});

describe('golden unit_score', () => {
  for (const c of doc.unit_score) {
    it(`${c.name}: ${c.expect.toFixed(6)}`, () => {
      const got = unitScore(c.vecs.map((v) => Float32Array.from(v)), c.gallery.map((v) => Float32Array.from(v)));
      expect(Math.abs(got - c.expect)).toBeLessThanOrEqual(TOL);
    });
  }
  it('is the median of the best cosines', () => {
    const e = (x: number, y: number) => Float32Array.from([x, y]);
    // best cosine of each crop to the gallery [ (1,0) ]: 1, 0, 0.6 -> median 0.6
    expect(unitScore([e(1, 0), e(0, 1), e(0.6, 0.8)], [e(1, 0)])).toBeCloseTo(0.6, 6);
    expect(unitScore([], [e(1, 0)])).toBe(0);
    expect(unitScore([e(1, 0)], [])).toBe(0);
  });
});

describe('golden person_crop geometry', () => {
  it('covers boxes that crop and boxes too small', () => {
    expect(doc.crop_rect.length).toBeGreaterThan(100);
    expect(doc.crop_rect.some((r) => r.rect === null)).toBe(true);
  });
  doc.crop_rect.forEach((r, i) => {
    it(`${i}: ${r.w}x${r.h} ${JSON.stringify(r.box.map((v) => +v.toFixed(4)))}`, () => {
      expect(personCropRect(r.w, r.h, r.box)).toEqual(r.rect);
    });
  });
});

describe('golden pad_square', () => {
  for (const c of doc.pad_square) {
    it(`${c.name}: ${c.w}x${c.h} to ${c.side}`, () => {
      const out = padSquare({ data: lcgBytes(c.seed, c.h * c.w * 3), w: c.w, h: c.h });
      expect(out.w).toBe(c.side);
      expect(out.h).toBe(c.side);
      expect(sha(out.data)).toBe(c.sha256);
    });
  }
});

describe("golden resize (Pillow's bicubic)", () => {
  for (const c of doc.resize) {
    it(`${c.name}: ${c.w}x${c.h} to ${c.ow}x${c.oh}, every byte`, () => {
      const out = resizeBicubic({ data: lcgBytes(c.seed, c.w * c.h * 3), w: c.w, h: c.h }, c.ow, c.oh);
      expect([out.w, out.h]).toEqual([c.ow, c.oh]);
      expect(sha(out.data)).toBe(c.sha256);
    });
  }
  it('the processor sizes: the short edge becomes 256', () => {
    expect(resizeDims(300, 300)).toEqual([256, 256]);
    expect(resizeDims(480, 200)).toEqual([614, 256]);
    expect(resizeDims(150, 400)).toEqual([256, 682]);
  });
});

describe('golden DINOv2 input (person_crop, pad_square, resize 256, centre crop 224, normalise)', () => {
  for (const c of doc.pipeline) {
    it(`${c.name}`, () => {
      const rgba = rgbaOf(lcgBytes(c.seed, c.fw * c.fh * 3));
      const rect = personCropRect(c.fw, c.fh, c.box);
      expect(rect).toEqual(c.rect);
      const t = cropTensor(rgba, c.fw, c.fh, c.box);
      if (c.rect === null) {
        expect(t).toBeNull();
        return;
      }
      expect(t).not.toBeNull();
      // the stages on their own, so a failure says where
      const [a, b, x, y] = rect!;
      const pix = new Uint8Array((x - a) * (y - b) * 3);
      for (let r = b; r < y; r++) for (let q = a; q < x; q++) for (let k = 0; k < 3; k++) pix[((r - b) * (x - a) + q - a) * 3 + k] = rgba[(r * c.fw + q) * 4 + k];
      const sq = padSquare({ data: pix, w: x - a, h: y - b });
      expect(sq.w).toBe(c.side);
      expect(sha(sq.data)).toBe(c.square_sha256);
      const [rw, rh] = resizeDims(sq.w, sq.h);
      expect(sha(centerCrop(resizeBicubic(sq, rw, rh)).data)).toBe(c.crop224_sha256);
      const want = c.tensor!;
      expect(t!.length).toBe(want.n);
      for (const [i, v] of want.samples) expect(Math.abs(t![i] - v), `value ${i}`).toBeLessThanOrEqual(TOL);
      let sum = 0;
      for (let i = 0; i < t!.length; i++) sum += t![i];
      expect(Math.abs(sum / t!.length - want.mean)).toBeLessThanOrEqual(TOL);
      expect(modelInput(sq)).toEqual(t);
    });
  }
});
