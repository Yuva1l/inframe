// The DINOv2-S look of a person for tap targets (docs/T9B_DESIGN.md section 5): the crop of a detection, padded square, resized to 256
// and centre-cropped to 224, ImageNet-normalised, run through the ONNX model, `pooler_output` L2-normalised. It follows
// src/inframe/identity.py (person_crop) and embed_worker.py (pad_square, then the Hugging Face processor of facebook/dinov2-small:
// bicubic resize of the short edge to 256, centre crop 224). Every step before the model is pure, so Node checks the geometry and the
// pixels against the golden vectors of tools/gen_golden.py (tests/golden/gallery_v1.json); the session needs a browser [unverified in Node].
//
// The resize is a port of Pillow's two-pass 8-bit bicubic (Resample.c: a = -0.5, antialiased support, 22-bit fixed-point weights, a uint8
// picture after the horizontal pass), so the tensor matches the Python path made from PIL images. The processor of transformers 5.19 is
// within one grey level of it (tests/browser/test_embed_parity.py checks the cosines of the embeddings, not the pixels).
//
// Licence of the port: the resize below follows Pillow's libImaging/Resample.c, Copyright (c) 1997-2011 by Secret Labs AB and 1995-2011 by
// Fredrik Lundh and contributors, Pillow Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors, under the HPND licence (Pillow's own
// LICENSE file calls it MIT-CMU). Its permission and disclaimer notice is reproduced in the shipped THIRD_PARTY_LICENSES.txt
// (source: web/src/notices.txt) and must stay there while this port ships.
import type { Box4, Detection, Session } from './detect';
import { iou, TAP_MIN_IOU } from './targets';

export const CROP_PAD = 0.04;
export const MIN_CROP_H = 48;
export const MIN_CROP_W = 16;
/** embed_worker.PAD_RGB: the ImageNet mean, padding that carries no signal */
export const PAD_RGB = [124, 116, 104] as const;
export const RESIZE_SHORT = 256;
export const EMBED_SIDE = 224;
export const MEAN = [0.485, 0.456, 0.406] as const;
export const STD = [0.229, 0.224, 0.225] as const;
export const EMBED_DIM = 384;
/** Detections below this score are not embedded, and no more than MAX_EMBED per picture (best first): the embedder runs one crop at a time. */
export const EMBED_MIN_SCORE = 0.3;
export const MAX_EMBED = 8;

/** An interleaved RGB picture. */
export interface Rgb {
  data: Uint8Array | Uint8ClampedArray;
  w: number;
  h: number;
}
export type Rect = [number, number, number, number];

/** identity.person_crop's geometry: the box (fractions of the picture) padded by 4 %, in pixels as [x0, y0, x1, y1]; null when it is too small to say anything. */
export function personCropRect(w: number, h: number, box: Box4): Rect | null {
  const [x0, y0, x1, y1] = box;
  const px = (x1 - x0) * CROP_PAD;
  const py = (y1 - y0) * CROP_PAD;
  const a = Math.max(Math.trunc((x0 - px) * w), 0);
  const b = Math.max(Math.trunc((y0 - py) * h), 0);
  const c = Math.min(Math.trunc((x1 + px) * w) + 1, w);
  const d = Math.min(Math.trunc((y1 + py) * h) + 1, h);
  if (d - b < MIN_CROP_H || c - a < MIN_CROP_W) return null;
  return [a, b, c, d];
}

/** The pixels of `rect` from an RGBA picture of w x h (what getImageData gives), as RGB. */
export function cropRgba(rgba: ArrayLike<number>, w: number, rect: Rect): Rgb {
  const [a, b, c, d] = rect;
  const cw = c - a;
  const ch = d - b;
  const out = new Uint8Array(cw * ch * 3);
  let o = 0;
  for (let y = b; y < d; y++) {
    let i = (y * w + a) * 4;
    for (let x = 0; x < cw; x++, i += 4) {
      out[o++] = rgba[i];
      out[o++] = rgba[i + 1];
      out[o++] = rgba[i + 2];
    }
  }
  return { data: out, w: cw, h: ch };
}

/** embed_worker.pad_square: the crop centred on a square of PAD_RGB, so the centre crop of the processor keeps all of the person. */
export function padSquare(im: Rgb): Rgb {
  if (im.h === im.w) return im;
  const s = Math.max(im.h, im.w);
  const out = new Uint8Array(s * s * 3);
  for (let i = 0; i < s * s; i++) {
    out[i * 3] = PAD_RGB[0];
    out[i * 3 + 1] = PAD_RGB[1];
    out[i * 3 + 2] = PAD_RGB[2];
  }
  const y0 = (s - im.h) >> 1;
  const x0 = (s - im.w) >> 1;
  for (let y = 0; y < im.h; y++) out.set(im.data.subarray(y * im.w * 3, (y + 1) * im.w * 3), ((y + y0) * s + x0) * 3);
  return { data: out, w: s, h: s };
}

// ---------------------------------------------------------------- Pillow's bicubic resize
const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;
const A = -0.5;

function bicubic(xIn: number): number {
  const x = Math.abs(xIn);
  if (x < 1) return ((A + 2) * x - (A + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * A;
  return 0;
}

interface Coeffs {
  ksize: number;
  bounds: Int32Array;
  kk: Int32Array;
}
const coeffCache = new Map<string, Coeffs>();

/** Resample.c precompute_coeffs + normalize_coeffs_8bpc for the whole input range. */
function coeffs(inSize: number, outSize: number): Coeffs {
  const key = `${inSize}>${outSize}`;
  const hit = coeffCache.get(key);
  if (hit) return hit;
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 2 * filterscale;
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
      const w = bicubic((x + xmin - center + 0.5) * ss);
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
  const v = Math.floor(ss / ONE);
  return v < 0 ? 0 : v > 255 ? 255 : v;
};

/** PIL's Image.resize((ow, oh), Image.BICUBIC) of an RGB picture: horizontal pass first, a uint8 picture in between. */
export function resizeBicubic(im: Rgb, ow: number, oh: number): Rgb {
  let cur = im;
  if (ow !== cur.w) {
    const { ksize, bounds, kk } = coeffs(cur.w, ow);
    const out = new Uint8Array(ow * cur.h * 3);
    for (let y = 0; y < cur.h; y++) {
      for (let xx = 0; xx < ow; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        for (let c = 0; c < 3; c++) {
          let ss = ONE / 2;
          for (let x = 0; x < xmax; x++) ss += cur.data[(y * cur.w + x + xmin) * 3 + c] * kk[xx * ksize + x];
          out[(y * ow + xx) * 3 + c] = clip8(ss);
        }
      }
    }
    cur = { data: out, w: ow, h: cur.h };
  }
  if (oh !== cur.h) {
    const { ksize, bounds, kk } = coeffs(cur.h, oh);
    const out = new Uint8Array(cur.w * oh * 3);
    for (let yy = 0; yy < oh; yy++) {
      const ymin = bounds[yy * 2];
      const ymax = bounds[yy * 2 + 1];
      for (let x = 0; x < cur.w * 3; x++) {
        let ss = ONE / 2;
        for (let y = 0; y < ymax; y++) ss += cur.data[(y + ymin) * cur.w * 3 + x] * kk[yy * ksize + y];
        out[yy * cur.w * 3 + x] = clip8(ss);
      }
    }
    cur = { data: out, w: cur.w, h: oh };
  }
  return cur;
}

/** The size the processor gives a picture whose short edge is resized to 256 (the long edge keeps the aspect, truncated). */
export function resizeDims(w: number, h: number, short = RESIZE_SHORT): [number, number] {
  if (w === h) return [short, short];
  if (w < h) return [short, Math.trunc((short * h) / w)];
  return [Math.trunc((short * w) / h), short];
}

/** The processor's centre crop (top = (h - 224) // 2). */
export function centerCrop(im: Rgb, side = EMBED_SIDE): Rgb {
  const top = Math.floor((im.h - side) / 2);
  const left = Math.floor((im.w - side) / 2);
  const out = new Uint8Array(side * side * 3);
  for (let y = 0; y < side; y++) out.set(im.data.subarray(((y + top) * im.w + left) * 3, ((y + top) * im.w + left + side) * 3), y * side * 3);
  return { data: out, w: side, h: side };
}

/** NCHW float32: rescale by 1/255, subtract the mean, divide by the std, all in float32 as the processor does. */
export function toTensor(im: Rgb): Float32Array {
  const n = im.w * im.h;
  const out = new Float32Array(3 * n);
  const scale = 1 / 255;
  for (let c = 0; c < 3; c++) {
    const mean = Math.fround(MEAN[c]);
    const std = Math.fround(STD[c]);
    for (let i = 0; i < n; i++) out[c * n + i] = (Math.fround(im.data[i * 3 + c] * scale) - mean) / std;
  }
  return out;
}

/** From a padded square crop to the model input: resize to 256, centre-crop 224, normalise. */
export function modelInput(square: Rgb): Float32Array {
  const [w, h] = resizeDims(square.w, square.h);
  return toTensor(centerCrop(resizeBicubic(square, w, h)));
}

/** The model input for a box of an RGBA picture; null when the box is too small to embed. */
export function cropTensor(rgba: ArrayLike<number>, w: number, h: number, box: Box4): Float32Array | null {
  const rect = personCropRect(w, h, box);
  if (!rect) return null;
  return modelInput(padSquare(cropRgba(rgba, w, rect)));
}

/** The L2-normalised row the Python embedder returns: f / (|f| + 1e-9). */
export function l2(f: ArrayLike<number>): Float32Array {
  let s = 0;
  for (let i = 0; i < f.length; i++) s += f[i] * f[i];
  const n = Math.sqrt(s) + 1e-9;
  const out = new Float32Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = f[i] / n;
  return out;
}

/** DINOv2-S on a session (batch 1). */
export class Dino {
  constructor(readonly session: Session) {}

  async embed(tensor: Float32Array): Promise<Float32Array> {
    const out = await this.session.run(tensor, [1, 3, EMBED_SIDE, EMBED_SIDE]);
    const name = out.pooler_output ? 'pooler_output' : Object.keys(out).find((k) => out[k].dims.length === 2);
    if (!name) throw new Error('The recognition model gave outputs this page does not know.');
    return l2(out[name].data);
  }

  release = () => this.session.release();
}

/** A picture with its pixels readable, for the crops (the Worker draws one per sample). */
export interface Pixels {
  rgba: ArrayLike<number>;
  w: number;
  h: number;
}

/** What embedDetections left out: detections that qualified (score, size) but were past the cap. */
export interface EmbedStats {
  /** detections skipped only because MAX_EMBED was reached */
  capped: number;
}

/** Fills `embedding` on the detections worth recognising: score from EMBED_MIN_SCORE, the best MAX_EMBED, boxes big enough. Others get null.
 *  `wanted` are the boxes of taps near this picture: a detection that overlaps one (IoU from TAP_MIN_IOU) is embedded whatever its rank or score,
 *  so the tapped person is never lost in a crowd. `stats.capped` counts what the cap dropped. */
export async function embedDetections(
  dino: Pick<Dino, 'embed'>, pic: Pixels, dets: Detection[], opts: { minScore?: number; max?: number; wanted?: readonly Box4[]; stats?: EmbedStats } = {},
): Promise<number> {
  const minScore = opts.minScore ?? EMBED_MIN_SCORE;
  const max = opts.max ?? MAX_EMBED;
  const wanted = opts.wanted ?? [];
  let n = 0;
  for (const d of dets) {
    d.embedding = null;
    const tapped = wanted.some((b) => iou(b, d.box) >= TAP_MIN_IOU);
    if (!tapped && d.score < minScore) continue;
    if (!tapped && n >= max) {
      if (opts.stats && personCropRect(pic.w, pic.h, d.box)) opts.stats.capped++;
      continue;
    }
    const t = cropTensor(pic.rgba, pic.w, pic.h, d.box);
    if (!t) continue;
    d.embedding = await dino.embed(t);
    n++;
  }
  return n;
}
