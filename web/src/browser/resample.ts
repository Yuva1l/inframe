// SPDX-License-Identifier: LGPL-2.1-or-later
// This file is licensed under the GNU Lesser General Public License, version 2.1 or (at your option) any later version (LICENSES/LGPL-2.1.txt), together
// with yuv.ts, the YUV to RGB stage it re-exports, whose makeFilter follows the expressions and tap count of FFmpeg's libswscale (initFilter; Copyright (C)
// Michael Niedermayer and the FFmpeg developers). The rest of the repository is Apache-2.0 (LICENSE); see NOTICE. The squash below is a port of Pillow's Resample.c (HPND, see web/src/notices.txt).
// The picture the detector sees, made the way the Mac makes it (src/inframe/decode.py and detect.preprocess), from the planes of a decoded frame:
//   1. YUV to RGB at the decode size (the long side at most 640 px, even sizes), to match PyAV's reformat on the Mac;
//   2. the upright turn (decode.upright);
//   3. the squash to the detector's square: a fixed-point port of PIL's BILINEAR (Resample.c), bit-exact.
// The canvas the browser path used before (a 2 x 2 box downscale, then the canvas's own smoothing) moved the scores of the clip edges by up to 0.4
// of the enter threshold (the parity notes). Everything here is pure and runs in Node; tests/golden/resample_v2.json (tools/gen_golden.py) holds it
// to PyAV and Pillow.
//
// The squash follows Pillow's libImaging/Resample.c (HPND; the notice ships with the page, see web/src/notices.txt).
// The YUV to RGB stage lives in yuv.ts, written from its functional specification, docs/t9b/YUV_SPEC.md (provenance: web/src/notices.txt; not a verified clean-room).
// The filter in yuv.ts is treated as derived from libswscale's initFilter (LGPL-2.1-or-later), so yuv.ts and this file are LGPL-2.1-or-later (headers above;
// the text is in LICENSES/LGPL-2.1.txt and, with the notice, in web/src/notices.txt and README (Licenses)); both ship as source beside the page.

// ---------------------------------------------------------------- sizes and matrices
/** Python's round(): halves go to the even neighbour. */
function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** decode.scaled_size: even-sized (w, h) with the long side at most `long`; never enlarges. */
export function scaledSize(w: number, h: number, long: number): [number, number] {
  const scale = Math.min(1.0, long / Math.max(w, h));
  return [Math.max(2, pyRound((w * scale) / 2) * 2), Math.max(2, pyRound((h * scale) / 2) * 2)];
}

export interface Matrix {
  kr: number;
  kb: number;
}
const BT709: Matrix = { kr: 0.2126, kb: 0.0722 };
const BT601: Matrix = { kr: 0.299, kb: 0.114 };
const BT2020: Matrix = { kr: 0.2627, kb: 0.0593 };

/** The luma coefficients of a matrix named the way a VideoColorSpace names it. An unnamed one is BT.601: swscale's default. */
export function matrixOf(name: string | null | undefined): Matrix {
  switch (name) {
    case 'bt709': return { ...BT709 };
    case 'bt2020-ncl': case 'bt2020nc': return { ...BT2020 };
    default: return { ...BT601 };
  }
}

// ---------------------------------------------------------------- the planes of a decoded frame
export interface YuvPicture {
  format: 'NV12' | 'I420';
  /** the picture's size (the visible part) */
  width: number;
  height: number;
  /** the range the planes are in (a frame's colorSpace.fullRange); limited range is 16..235 */
  fullRange: boolean;
  kr: number;
  kb: number;
  y: Uint8Array;
  yStride: number;
  /** NV12: the interleaved U V plane; I420: the U plane */
  u: Uint8Array;
  uStride: number;
  /** I420 only */
  v?: Uint8Array;
  vStride?: number;
  /** where the chroma samples of a row sit horizontally: 'left' (on the first luma sample of each pair: what FFmpeg's h264 and hevc decoders tag a 4:2:0
   *  frame with, MPEG-2 siting) or 'center' (between the two: what swscale assumes for a frame that names no siting). Vertically they are always centred. */
  chromaLoc: 'left' | 'center';
}

// ---------------------------------------------------------------- YUV to RGB
export { yuvToRgb } from './yuv';

// ---------------------------------------------------------------- the upright turn
/** decode.upright: the display matrix applied to an interleaved RGB picture. `rotation` is ffprobe's (counter-clockwise degrees), a mirror is a
 *  flip top to bottom after an even number of quarter turns and left to right after an odd one. */
export function uprightRgb(rgb: Uint8Array, w: number, h: number, rotation: number, mirrored: boolean, into?: Uint8Array): { data: Uint8Array; w: number; h: number } {
  const k = ((Math.floor(rotation / 90) % 4) + 4) % 4;
  if (!k && !mirrored) return { data: rgb, w, h };
  const ow = k % 2 ? h : w;
  const oh = k % 2 ? w : h;
  const out = into && into.length >= ow * oh * 3 ? into : new Uint8Array(ow * oh * 3);
  for (let i = 0; i < oh; i++) {
    for (let j = 0; j < ow; j++) {
      const si = mirrored && k % 2 === 0 ? oh - 1 - i : i;
      const sj = mirrored && k % 2 === 1 ? ow - 1 - j : j;
      let r: number;
      let c: number;
      if (k === 0) { r = si; c = sj; } else if (k === 1) { r = sj; c = w - 1 - si; } else if (k === 2) { r = h - 1 - si; c = w - 1 - sj; } else { r = h - 1 - sj; c = si; }
      const s = (r * w + c) * 3;
      const d = (i * ow + j) * 3;
      out[d] = rgb[s];
      out[d + 1] = rgb[s + 1];
      out[d + 2] = rgb[s + 2];
    }
  }
  return { data: out, w: ow, h: oh };
}

// ---------------------------------------------------------------- PIL's BILINEAR squash
const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;
interface Coeffs {
  ksize: number;
  bounds: Int32Array;
  kk: Int32Array;
}
const coeffCache = new Map<string, Coeffs>();

/** Resample.c precompute_coeffs + normalize_coeffs_8bpc for the whole input range, with the bilinear filter (support 1). */
function pilCoeffs(inSize: number, outSize: number): Coeffs {
  const key = `${inSize}>${outSize}`;
  const hit = coeffCache.get(key);
  if (hit) return hit;
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const pre = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    pre.fill(0);
    for (let x = 0; x < xmax; x++) {
      const d = Math.abs((x + xmin - center + 0.5) * ss);
      const w = d < 1 ? 1 - d : 0;
      pre[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const k = ww !== 0 ? pre[x] / ww : pre[x];
      kk[xx * ksize + x] = k < 0 ? Math.trunc(-0.5 + k * ONE) : Math.trunc(0.5 + k * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  const c = { ksize, bounds, kk };
  if (coeffCache.size > 64) coeffCache.clear();
  coeffCache.set(key, c);
  return c;
}

const clip8 = (ss: number): number => {
  const v = ss >> PRECISION_BITS;
  return v < 0 ? 0 : v > 255 ? 255 : v;
};

let scratch: Uint8Array = new Uint8Array(0);

/** PIL's Image.resize((side, side), Image.BILINEAR) of an RGB picture: horizontal pass first, a uint8 picture in between. */
export function squashRgb(rgb: Uint8Array, w: number, h: number, side: number, into?: Uint8Array): Uint8Array {
  const out = into && into.length >= side * side * 3 ? into : new Uint8Array(side * side * 3);
  let cur = rgb;
  if (side !== w) {
    const { ksize, bounds, kk } = pilCoeffs(w, side);
    const horizontalOnly = side === h;
    const mid = horizontalOnly ? out : (scratch.length < side * h * 3 ? (scratch = new Uint8Array(side * h * 3)) : scratch);
    for (let y = 0; y < h; y++) {
      const row = y * w * 3;
      for (let xx = 0; xx < side; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const k0 = xx * ksize;
        let s0 = ONE / 2;
        let s1 = ONE / 2;
        let s2 = ONE / 2;
        for (let x = 0; x < xmax; x++) {
          const kv = kk[k0 + x];
          const p = row + (x + xmin) * 3;
          s0 += cur[p] * kv;
          s1 += cur[p + 1] * kv;
          s2 += cur[p + 2] * kv;
        }
        const o = (y * side + xx) * 3;
        mid[o] = clip8(s0);
        mid[o + 1] = clip8(s1);
        mid[o + 2] = clip8(s2);
      }
    }
    if (horizontalOnly) return out;
    cur = mid;
  }
  if (side !== h) {
    const { ksize, bounds, kk } = pilCoeffs(h, side);
    const n = side * 3;
    for (let yy = 0; yy < side; yy++) {
      const ymin = bounds[yy * 2];
      const ymax = bounds[yy * 2 + 1];
      const k0 = yy * ksize;
      for (let x = 0; x < n; x++) {
        let s = ONE / 2;
        for (let y = 0; y < ymax; y++) s += cur[(y + ymin) * n + x] * kk[k0 + y];
        out[yy * n + x] = clip8(s);
      }
    }
  } else {
    out.set(cur.subarray(0, side * side * 3));
  }
  return out;
}

/** The squash as RGBA with opaque alpha: what `detect.preprocess` takes (a canvas's getImageData gave this shape). */
export function squashRgba(rgb: Uint8Array, w: number, h: number, side: number, into?: Uint8ClampedArray): Uint8ClampedArray {
  const px = squashRgb(rgb, w, h, side);
  const out = into && into.length >= side * side * 4 ? into : new Uint8ClampedArray(side * side * 4);
  for (let i = 0, j = 0, n = side * side; i < n; i++) {
    out[j++] = px[i * 3];
    out[j++] = px[i * 3 + 1];
    out[j++] = px[i * 3 + 2];
    out[j++] = 255;
  }
  return out;
}
