// SPDX-License-Identifier: LGPL-2.1-or-later
// This file is licensed under the GNU Lesser General Public License, version 2.1 or (at your option) any later version (LICENSES/LGPL-2.1.txt): makeFilter
// is treated as derived from FFmpeg's libswscale (initFilter; Copyright (C) Michael Niedermayer and the FFmpeg developers). The rest of the repository is
// Apache-2.0 (LICENSE); see NOTICE. Its source ships beside the page as yuv.ts (LGPL-2.1 section 4).
// The YUV to RGB stage of the picture path: the planes of a decoded 4:2:0 frame to an RGB picture at the decode size.
//
// Written from docs/t9b/YUV_SPEC.md (the section numbers below are that document's), but NOT a verified clean-room: the spec was written by someone
// who worked from the removed swscale-following code, and that code stayed reachable while this file was written (see web/src/notices.txt).
// makeFilter below is that code's filter rewritten term for term: treat it as derived from libswscale (LGPL-2.1-or-later) until it is rewritten by someone who has not seen it.
// Pure, no DOM, runs in Node.
import type { YuvPicture } from './resample';

/** Floor of a / b for non-negative integers below 2^53, exact whatever the rounding of the double quotient. */
function fdiv(a: number, b: number): number {
  let q = Math.floor(a / b);
  while (q * b > a) q--;
  while ((q + 1) * b <= a) q++;
  return q;
}

// ---------------------------------------------------------------- section 5: the filter F(n, m, N, sigma)
interface Filter {
  /** weights per target: `taps` slots each, the first `count[k]` of them used, starting at source index `first[k]` */
  taps: number;
  first: Int32Array;
  count: Int32Array;
  w: Int32Array;
  /** L(F), the length of the filter */
  len: number;
}

const filterCache = new Map<string, Filter>();

function makeFilter(n: number, m: number, N: number, sigma: number): Filter {
  const key = `${n}>${m}:${N}:${sigma}`;
  const hit = filterCache.get(key);
  if (hit) return hit;
  const TICK = 65536;
  const step = fdiv(TICK * n + (m >> 1), m);
  const shrinking = step > TICK;
  const T = shrinking ? 1 + Math.ceil((2 * n) / m) : 3;
  const first = new Int32Array(m);
  const count = new Int32Array(m);
  const w = new Int32Array(m * T);
  const raw = new Float64Array(T);
  const credit = new Float64Array(T);
  let len = 1;
  for (let k = 0; k < m; k++) {
    const c = (step >> 1) - 256 * sigma + k * step;
    const a = Math.trunc((c - 32768 * (T - 2)) / TICK);
    // raw weights of the taps, and the sum over all of them
    let omega = 0;
    for (let t = 0; t < T; t++) {
      let delta = 16384 * Math.abs(TICK * (a + t) - c);
      if (shrinking) delta = fdiv(delta * m, n);
      const o = Math.max(0, 1073741824 - delta);
      raw[t] = o;
      omega += o;
    }
    // edge taps give their weight to the nearest edge sample
    const lo = Math.min(n - 1, Math.max(0, a));
    const hi = Math.min(n - 1, Math.max(0, a + T - 1));
    const cnt = hi - lo + 1;
    credit.fill(0, 0, cnt);
    for (let t = 0; t < T; t++) credit[Math.min(n - 1, Math.max(0, a + t)) - lo] += raw[t];
    // quantise the running sum so the weights add up to N exactly
    let run = 0;
    let prevQ = 0;
    let firstNz = -1;
    let lastNz = -1;
    for (let j = 0; j < cnt; j++) {
      run += credit[j];
      const q = omega > 0 ? fdiv(2 * N * run + omega, 2 * omega) : N;
      const wt = q - prevQ;
      prevQ = q;
      w[k * T + j] = wt;
      if (wt !== 0) {
        if (firstNz < 0) firstNz = j;
        lastNz = j;
      }
    }
    first[k] = lo;
    count[k] = cnt;
    if (firstNz >= 0) len = Math.max(len, lastNz - firstNz + 1);
  }
  const f = { taps: T, first, count, w, len };
  if (filterCache.size > 64) filterCache.clear();
  filterCache.set(key, f);
  return f;
}

// ---------------------------------------------------------------- section 6: one plane, two passes
let midScratch = new Uint16Array(0);

/** Resample a plane (8-bit samples at `stride`, sh rows) to dw x dh: horizontal pass into a 15-bit intermediate, then the vertical pass. */
function resamplePlane(src: Uint8Array, stride: number, sh: number, dw: number, dh: number, fx: Filter, fy: Filter, beta: number, out: Uint8Array): void {
  if (midScratch.length < dw * sh) midScratch = new Uint16Array(dw * sh);
  const mid = midScratch;
  const { taps: tx, first: x0, count: nx, w: wx } = fx;
  for (let r = 0; r < sh; r++) {
    const row = r * stride;
    const mo = r * dw;
    for (let x = 0; x < dw; x++) {
      const s0 = x0[x];
      const cnt = nx[x];
      const wk = x * tx;
      let sum = 0;
      for (let j = 0; j < cnt; j++) sum += wx[wk + j] * src[row + s0 + j];
      const v = sum >> 7;
      mid[mo + x] = v > 32767 ? 32767 : v;
    }
  }
  const { taps: ty, first: y0, count: ny, w: wy } = fy;
  const acc = new Int32Array(dw);
  for (let y = 0; y < dh; y++) {
    acc.fill(beta);
    const s0 = y0[y];
    const cnt = ny[y];
    const wk = y * ty;
    for (let j = 0; j < cnt; j++) {
      const wt = wy[wk + j];
      if (wt === 0) continue;
      const mo = (s0 + j) * dw;
      for (let x = 0; x < dw; x++) acc[x] += wt * mid[mo + x];
    }
    const oo = y * dw;
    for (let x = 0; x < dw; x++) {
      const v = acc[x] >> 19;
      out[oo + x] = v > 255 ? 255 : v;
    }
  }
}

// ---------------------------------------------------------------- section 7: the colour parameters
interface Params {
  /** slope and offset of S_R (of V), S_B (of U), S_GU (of U), S_GV (of V) */
  r: [number, number];
  b: [number, number];
  gu: [number, number];
  gv: [number, number];
}

const BT709_FULL: Params = { r: [1.5748, 0.5746], b: [1.8556, 0.5162], gu: [-0.1873243, 0.0317], gv: [-0.4681243, 0.081] };
const BT709_LIMITED: Params = { r: [1.5396482, 0.0761], b: [1.8141804, 0.2164], gu: [-0.1831429, 0.5593], gv: [-0.4576751, 0.419] };
const BT601_FULL: Params = { r: [1.402, 0.457], b: [1.7719291, 0.8158], gu: [-0.3441363, 0.9537], gv: [-0.7141363, 0.6429] };
const BT601_LIMITED: Params = { r: [1.3707054, 0.453], b: [1.7324464, 0.7524], gu: [-0.3364547, 0.9391], gv: [-0.6981957, 0.6331] };

const LIMITED_SCALE = 255 / 219;

/** The slopes and offsets for a matrix and range; the matrices the reference was measured on use the measured table, any other the formula. */
function paramsOf(kr: number, kb: number, fullRange: boolean): Params {
  if (kr === 0.2126 && kb === 0.0722) return fullRange ? BT709_FULL : BT709_LIMITED;
  if (kr === 0.299 && kb === 0.114) return fullRange ? BT601_FULL : BT601_LIMITED;
  const kg = 1 - kr - kb;
  const kappa = fullRange ? 1 : 219 / 224;
  const aR = 2 * (1 - kr) * kappa;
  const aB = 2 * (1 - kb) * kappa;
  return { r: [aR, 0.5], b: [aB, 0.5], gu: [-(kb / kg) * aB, 0.5], gv: [-(kr / kg) * aR, 0.5] };
}

const shiftTable = (slope: number, offset: number): Int16Array => {
  const t = new Int16Array(256);
  for (let c = 0; c < 256; c++) t[c] = Math.floor(slope * (c - 128) + offset);
  return t;
};

/** Gamma(j) for j from -320 to 575, as a byte table indexed by j + 320. */
const RAMP_LO = 320;
const rampTable = (fullRange: boolean): Uint8Array => {
  const t = new Uint8Array(RAMP_LO + 576);
  for (let i = 0; i < t.length; i++) {
    const j = i - RAMP_LO;
    const v = fullRange ? j : Math.floor(LIMITED_SCALE * (j - 16) - 1.2808);
    t[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return t;
};

interface Tables {
  sr: Int16Array;
  sb: Int16Array;
  sgu: Int16Array;
  sgv: Int16Array;
  ramp: Uint8Array;
}
const tableCache = new Map<string, Tables>();
function tablesOf(kr: number, kb: number, fullRange: boolean): Tables {
  const key = `${kr}/${kb}/${fullRange}`;
  let t = tableCache.get(key);
  if (!t) {
    const p = paramsOf(kr, kb, fullRange);
    t = { sr: shiftTable(p.r[0], p.r[1]), sb: shiftTable(p.b[0], p.b[1]), sgu: shiftTable(p.gu[0], p.gu[1]), sgv: shiftTable(p.gv[0], p.gv[1]), ramp: rampTable(fullRange) };
    if (tableCache.size > 16) tableCache.clear();
    tableCache.set(key, t);
  }
  return t;
}

// ---------------------------------------------------------------- the chroma planes as contiguous planes
function chromaPlanes(p: YuvPicture, cw: number, ch: number): [Uint8Array, Uint8Array] {
  const u = new Uint8Array(cw * ch);
  const v = new Uint8Array(cw * ch);
  if (p.format === 'NV12') {
    for (let r = 0; r < ch; r++) {
      const row = r * p.uStride;
      for (let i = 0; i < cw; i++) {
        u[r * cw + i] = p.u[row + 2 * i];
        v[r * cw + i] = p.u[row + 2 * i + 1];
      }
    }
  } else {
    const pv = p.v as Uint8Array;
    const vs = p.vStride as number;
    for (let r = 0; r < ch; r++) {
      for (let i = 0; i < cw; i++) {
        u[r * cw + i] = p.u[r * p.uStride + i];
        v[r * cw + i] = pv[r * vs + i];
      }
    }
  }
  return [u, v];
}

// ---------------------------------------------------------------- section 8: the unscaled path
function unscaledRgb(p: YuvPicture, cw: number, ch: number, out: Uint8Array): void {
  const { width: w, height: h } = p;
  const [pu, pv] = chromaPlanes(p, cw, ch);
  const q = paramsOf(p.kr, p.kb, p.fullRange);
  const clamp = (x: number): number => (x < 0 ? 0 : x > 255 ? 255 : x);
  let o = 0;
  if (p.fullRange) {
    const sr = shiftTable(q.r[0], 0.5);
    const sb = shiftTable(q.b[0], 0.5);
    const sgu = shiftTable(q.gu[0], 0.5);
    const sgv = shiftTable(q.gv[0], 0.5);
    for (let y = 0; y < h; y++) {
      const ci = (y >> 1) * cw;
      for (let x = 0; x < w; x++) {
        const Y = p.y[y * p.yStride + x];
        const U = pu[ci + (x >> 1)];
        const V = pv[ci + (x >> 1)];
        out[o++] = clamp(Y + sr[V]);
        out[o++] = clamp(Y + sgu[U] + sgv[V]);
        out[o++] = clamp(Y + sb[U]);
      }
    }
    return;
  }
  const aR = q.r[0] * LIMITED_SCALE;
  const aB = q.b[0] * LIMITED_SCALE;
  const aGU = q.gu[0] * LIMITED_SCALE;
  const aGV = q.gv[0] * LIMITED_SCALE;
  for (let y = 0; y < h; y++) {
    const ci = (y >> 1) * cw;
    for (let x = 0; x < w; x++) {
      const g = LIMITED_SCALE * (p.y[y * p.yStride + x] - 16);
      const U = pu[ci + (x >> 1)] - 128;
      const V = pv[ci + (x >> 1)] - 128;
      out[o++] = clamp(Math.floor(g + aR * V + 0.2));
      out[o++] = clamp(Math.floor(g + aGU * U + aGV * V + 0.2));
      out[o++] = clamp(Math.floor(g + aB * U + 0.2));
    }
  }
}

// ---------------------------------------------------------------- the stage
/** A picture's planes to an interleaved RGB picture of outW x outH (both even): the behaviour is specified in docs/t9b/YUV_SPEC.md. */
export function yuvToRgb(p: YuvPicture, outW: number, outH: number, into?: Uint8Array): Uint8Array {
  const out = into && into.length >= outW * outH * 3 ? into : new Uint8Array(outW * outH * 3);
  const { width: w, height: h } = p;
  const cw = (w + 1) >> 1;
  const ch = (h + 1) >> 1;
  if (outW === w && outH === h) {
    unscaledRgb(p, cw, ch, out);
    return out;
  }
  const ocw = (outW + 1) >> 1;
  const sigma = p.chromaLoc === 'left' ? 64 : 128;
  const lx = makeFilter(w, outW, 1 << 14, 128);
  const ly = makeFilter(h, outH, 1 << 12, 128);
  const cx = makeFilter(cw, ocw, 1 << 14, sigma);
  const cy = makeFilter(ch, outH, 1 << 12, 128);
  const beta = ly.len === 2 && cy.len === 2 ? 0 : 1 << 18;
  const [pu, pv] = chromaPlanes(p, cw, ch);
  const oy = new Uint8Array(outW * outH);
  const ou = new Uint8Array(ocw * outH);
  const ov = new Uint8Array(ocw * outH);
  resamplePlane(p.y, p.yStride, h, outW, outH, lx, ly, beta, oy);
  resamplePlane(pu, cw, ch, ocw, outH, cx, cy, beta, ou);
  resamplePlane(pv, cw, ch, ocw, outH, cx, cy, beta, ov);
  const t = tablesOf(p.kr, p.kb, p.fullRange);
  const { sr, sb, sgu, sgv, ramp } = t;
  let o = 0;
  for (let y = 0; y < outH; y++) {
    const yi = y * outW;
    const ci = y * ocw;
    for (let x = 0; x < outW; x++) {
      const Y = oy[yi + x] + RAMP_LO;
      const U = ou[ci + (x >> 1)];
      const V = ov[ci + (x >> 1)];
      out[o++] = ramp[Y + sr[V]];
      out[o++] = ramp[Y + sgu[U] + sgv[V]];
      out[o++] = ramp[Y + sb[U]];
    }
  }
  return out;
}
