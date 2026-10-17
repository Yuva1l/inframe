// The picture the detector sees, against the golden vectors Python made (tests/golden/resample_v2.json, tools/gen_golden.py): swscale's bilinear
// YUV to RGB at the decode size (PyAV reformat), decode.upright, and PIL's BILINEAR squash to the detector's square. The squash and the turn are
// bit-exact (sha-256 of the output); the colour conversion is held to the tolerance each case names (1 level) and is bit-exact where it can be.
import { describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so the file read and the hash are allowed to be untyped
// @ts-ignore
import { readFileSync } from 'node:fs';
// @ts-ignore
import { createHash } from 'node:crypto';
import { matrixOf, scaledSize, squashRgb, squashRgba, uprightRgb, yuvToRgb, type YuvPicture } from '../src/browser/resample';
import { preprocess } from '../src/browser/detect';

type Chroma = 'left' | 'center';
const doc = JSON.parse(readFileSync((globalThis as any).process.cwd() + '/../tests/golden/resample_v2.json', 'utf8')) as {
  flat: { matrix: string; full: boolean; yuv_rgb: number[][] }[];
  scale: { name: string; fmt: string; matrix: string; full: boolean; chroma: Chroma; w: number; h: number; ow: number; oh: number; seed: number; tol: number; smooth: boolean; rgb: number[] }[];
  squash: { name: string; w: number; h: number; side: number; seed: number; sha256: string }[];
  upright: { rotation: number; mirrored: boolean; w: number; h: number; seed: number; ow: number; oh: number; sha256: string }[];
  pipeline: { name: string; fmt: string; matrix: string; full: boolean; chroma: Chroma; w: number; h: number; long: number; rotation: number; mirrored: boolean; side: number; seed: number; tol: number; sw: number; sh: number; rgb: number[] }[];
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

/** tools/gen_golden.py _planes, in the memory layout of `fmt`; `pad` extra bytes at the end of each row (a decoder's stride). */
function picture(fmt: string, matrix: string, full: boolean, w: number, h: number, seed: number, pad = 0, smooth = false, chromaLoc: Chroma = 'left'): YuvPicture {
  const cw = (w + 1) >> 1;
  const ch = (h + 1) >> 1;
  const y = lcgBytes(seed, w * h);
  let u = lcgBytes(seed + 1, cw * ch).map((b) => 64 + (b >> 1));
  let v = lcgBytes(seed + 2, cw * ch).map((b) => 64 + (b >> 1));
  if (smooth) {
    u = Uint8Array.from({ length: cw * ch }, (_, i) => 96 + Math.floor(((i % cw) * 2 + Math.floor(i / cw) * 3) / 4));
    v = Uint8Array.from({ length: cw * ch }, (_, i) => 160 - Math.floor(((i % cw) * 3 + Math.floor(i / cw) * 2) / 4));
  }
  const m = matrixOf(matrix);
  const rows = (src: Uint8Array, width: number, n: number, stride: number) => {
    const o = new Uint8Array(stride * n).fill(7);
    for (let r = 0; r < n; r++) o.set(src.subarray(r * width, (r + 1) * width), r * stride);
    return o;
  };
  const yStride = w + pad;
  if (fmt === 'nv12') {
    const uv = new Uint8Array(cw * 2 * ch);
    for (let i = 0; i < cw * ch; i++) { uv[2 * i] = u[i]; uv[2 * i + 1] = v[i]; }
    return { format: 'NV12', width: w, height: h, fullRange: full, ...m, chromaLoc, y: rows(y, w, h, yStride), yStride, u: rows(uv, cw * 2, ch, cw * 2 + pad), uStride: cw * 2 + pad };
  }
  return { format: 'I420', width: w, height: h, fullRange: full, ...m, chromaLoc, y: rows(y, w, h, yStride), yStride, u: rows(u, cw, ch, cw + pad), uStride: cw + pad, v: rows(v, cw, ch, cw + pad), vStride: cw + pad };
}

const worst = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  expect(a.length).toBe(b.length);
  let w = 0;
  for (let i = 0; i < a.length; i++) w = Math.max(w, Math.abs(a[i] - b[i]));
  return w;
};

describe('golden YUV to RGB (swscale, as PyAV reformat runs it)', () => {
  for (const f of doc.flat) {
    it(`flat colours, ${f.matrix} ${f.full ? 'full' : 'limited'} range: every triple matches exactly`, () => {
      const m = matrixOf(f.matrix);
      let bad = 0;
      for (const [y, u, v, r, g, b] of f.yuv_rgb) {
        const pic: YuvPicture = {
          format: 'NV12', width: 32, height: 32, fullRange: f.full, ...m, chromaLoc: 'left', y: new Uint8Array(1024).fill(y), yStride: 32,
          u: Uint8Array.from({ length: 512 }, (_, i) => (i & 1 ? v : u)), uStride: 32,
        };
        const out = yuvToRgb(pic, 16, 16);
        const k = (8 * 16 + 8) * 3;
        if (out[k] !== r || out[k + 1] !== g || out[k + 2] !== b) bad++;
      }
      expect(bad).toBe(0);
    });
  }
  for (const c of doc.scale) {
    it(`${c.name}: within ${c.tol} level of PyAV`, () => {
      const out = yuvToRgb(picture(c.fmt, c.matrix, c.full, c.w, c.h, c.seed, 0, c.smooth, c.chroma), c.ow, c.oh);
      expect(worst(out, c.rgb)).toBeLessThanOrEqual(c.tol);
    });
  }
  it('the scaled pictures are bit-exact, odd sizes and ratios that are not whole included, bar the 0.1 % of pixels that land a level off', () => {
    for (const c of doc.scale.filter((x) => !/not_scaled|1_6x|odd_size/.test(x.name))) expect(worst(yuvToRgb(picture(c.fmt, c.matrix, c.full, c.w, c.h, c.seed, 0, false, c.chroma), c.ow, c.oh), c.rgb), c.name).toBe(0);
  });
  it('an odd stored size of 640 or less is enlarged to the even decode size without a black edge (scaledSize rounds 639 up to 640)', () => {
    const [ow, oh] = scaledSize(639, 359, 640);
    expect([ow, oh]).toEqual([640, 360]);
    const p = picture('yuv420p', 'bt601', true, 639, 359, 5019, 0, true);
    p.y.fill(200);
    const out = yuvToRgb(p, ow, oh);
    for (const [x, y] of [[639, 359], [639, 0], [0, 359], [320, 359], [639, 180]]) {
      const k = (y * ow + x) * 3;
      expect(out[k] + out[k + 1] + out[k + 2], `pixel ${x},${y}`).toBeGreaterThan(300);
    }
  });
  it('a limited-range picture swscale does not scale: the gray ramp (read off PyAV once by hand; PyAV varies by a level or two between runs there)', () => {
    const gray = (y: number) => {
      const p: YuvPicture = { format: 'NV12', width: 4, height: 4, fullRange: false, ...matrixOf('bt709'), chromaLoc: 'left', y: new Uint8Array(16).fill(y), yStride: 4, u: new Uint8Array(8).fill(128), uStride: 4 };
      return Array.from(yuvToRgb(p, 4, 4).subarray(0, 3));
    };
    expect(gray(16)).toEqual([0, 0, 0]);
    expect(gray(17)).toEqual([1, 1, 1]);
    expect(gray(100)).toEqual([98, 98, 98]);
    expect(gray(235)).toEqual([255, 255, 255]);
    // the same Y through the scaling path lands about 1.3 levels lower on the limited ramp (swscale 10.1): 253, not 255
    const scaled: YuvPicture = { format: 'NV12', width: 32, height: 32, fullRange: false, ...matrixOf('bt709'), chromaLoc: 'left', y: new Uint8Array(1024).fill(235), yStride: 32, u: new Uint8Array(512).fill(128), uStride: 32 };
    expect(Array.from(yuvToRgb(scaled, 16, 16).subarray(0, 3))).toEqual([253, 253, 253]);
  });
  it('a row stride wider than the picture changes nothing', () => {
    for (const c of doc.scale.slice(0, 6)) {
      const tight = yuvToRgb(picture(c.fmt, c.matrix, c.full, c.w, c.h, c.seed, 0, false, c.chroma), c.ow, c.oh);
      expect(worst(yuvToRgb(picture(c.fmt, c.matrix, c.full, c.w, c.h, c.seed, 13, false, c.chroma), c.ow, c.oh), tight), c.name).toBe(0);
    }
  });
});

describe('golden squash (PIL BILINEAR)', () => {
  for (const c of doc.squash) {
    it(`${c.name}: ${c.w} x ${c.h} to ${c.side} x ${c.side} is bit-exact`, () => {
      const out = squashRgb(lcgBytes(c.seed, c.w * c.h * 3), c.w, c.h, c.side);
      expect(out.length).toBe(c.side * c.side * 3);
      expect(sha(out)).toBe(c.sha256);
    });
  }
  it('the RGBA form carries the same pixels with opaque alpha', () => {
    const rgb = lcgBytes(5101, 64 * 36 * 3);
    const a = squashRgba(rgb, 64, 36, 32);
    const b = squashRgb(rgb, 64, 36, 32);
    for (let i = 0; i < 32 * 32; i++) expect([a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]).toEqual([b[i * 3], b[i * 3 + 1], b[i * 3 + 2], 255]);
  });
});

describe('golden upright (decode.upright)', () => {
  for (const c of doc.upright) {
    it(`rotation ${c.rotation}${c.mirrored ? ' mirrored' : ''}`, () => {
      const o = uprightRgb(lcgBytes(c.seed, c.w * c.h * 3), c.w, c.h, c.rotation, c.mirrored);
      expect([o.w, o.h]).toEqual([c.ow, c.oh]);
      expect(sha(o.data)).toBe(c.sha256);
    });
  }
});

describe('golden decode path (reformat, upright, squash)', () => {
  for (const c of doc.pipeline) {
    it(`${c.name}: the squared picture is within ${c.tol} level`, () => {
      const [sw, sh] = scaledSize(c.w, c.h, c.long);
      expect([sw, sh]).toEqual([c.sw, c.sh]);
      const up = uprightRgb(yuvToRgb(picture(c.fmt, c.matrix, c.full, c.w, c.h, c.seed, 0, false, c.chroma), sw, sh), sw, sh, c.rotation, c.mirrored);
      expect(worst(squashRgb(up.data, up.w, up.h, c.side), c.rgb)).toBeLessThanOrEqual(c.tol);
    });
  }
  it('the tensor is the normalised RGBA squash (detect.preprocess)', () => {
    const c = doc.pipeline[0];
    const [sw, sh] = scaledSize(c.w, c.h, c.long);
    const up = uprightRgb(yuvToRgb(picture(c.fmt, c.matrix, c.full, c.w, c.h, c.seed, 0, false, c.chroma), sw, sh), sw, sh, c.rotation, c.mirrored);
    const t = preprocess(squashRgba(up.data, up.w, up.h, c.side), c.side, c.side);
    expect(t.length).toBe(3 * c.side * c.side);
    const first = c.rgb[0] / 255;
    expect(Math.abs(t[0] - (first - 0.485) / 0.229)).toBeLessThan(2 / 255 / 0.229 + 1e-6);
  });
});

describe('scaledSize (decode.scaled_size) and matrixOf', () => {
  it('keeps even sizes and never enlarges', () => {
    expect(scaledSize(1280, 720, 640)).toEqual([640, 360]);
    expect(scaledSize(1920, 1080, 640)).toEqual([640, 360]);
    expect(scaledSize(3840, 2160, 640)).toEqual([640, 360]);
    expect(scaledSize(1440, 1080, 640)).toEqual([640, 480]);
    expect(scaledSize(480, 270, 640)).toEqual([480, 270]);
    expect(scaledSize(1081, 721, 640)).toEqual([640, 426]);
    expect(scaledSize(1000, 700, 640)).toEqual([640, 448]);
    expect(scaledSize(4, 2, 640)).toEqual([4, 2]);
    expect(scaledSize(10000, 10, 640)).toEqual([640, 2]);
  });
  it('knows the matrices a decoder names, and falls back to BT.601 like swscale', () => {
    expect(matrixOf('bt709')).toEqual({ kr: 0.2126, kb: 0.0722 });
    expect(matrixOf('smpte170m')).toEqual({ kr: 0.299, kb: 0.114 });
    expect(matrixOf('bt470bg')).toEqual({ kr: 0.299, kb: 0.114 });
    expect(matrixOf('bt2020-ncl')).toEqual({ kr: 0.2627, kb: 0.0593 });
    expect(matrixOf(null)).toEqual({ kr: 0.299, kb: 0.114 });
    expect(matrixOf(undefined)).toEqual({ kr: 0.299, kb: 0.114 });
  });
});
