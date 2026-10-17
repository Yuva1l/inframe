// Targets that need no identity (docs/T9B_DESIGN.md section 5): every person (AnyPerson), every instance of one COCO class
// (AnyOfClass) and a zone around either (ZoneTarget). Exact ports of src/inframe/target.py, zone.py and classes.py; the golden
// vectors of tools/gen_golden.py (tests/golden/targets_v1.json) hold both sides to the same answers.
import type { Box4, Detection } from './detect';
import { Gallery, DINOV2 } from './gallery';
import { EngineError } from '../engine/types';

export const PERSON = 1;

/** classes.COCO_91: the category ids the RF-DETR COCO checkpoints use (91 slots, 80 named; the ids have gaps). */
export const COCO_91: Record<number, string> = {
  1: 'person',
  2: 'bicycle',
  3: 'car',
  4: 'motorcycle',
  5: 'airplane',
  6: 'bus',
  7: 'train',
  8: 'truck',
  9: 'boat',
  10: 'traffic light',
  11: 'fire hydrant',
  13: 'stop sign',
  14: 'parking meter',
  15: 'bench',
  16: 'bird',
  17: 'cat',
  18: 'dog',
  19: 'horse',
  20: 'sheep',
  21: 'cow',
  22: 'elephant',
  23: 'bear',
  24: 'zebra',
  25: 'giraffe',
  27: 'backpack',
  28: 'umbrella',
  31: 'handbag',
  32: 'tie',
  33: 'suitcase',
  34: 'frisbee',
  35: 'skis',
  36: 'snowboard',
  37: 'sports ball',
  38: 'kite',
  39: 'baseball bat',
  40: 'baseball glove',
  41: 'skateboard',
  42: 'surfboard',
  43: 'tennis racket',
  44: 'bottle',
  46: 'wine glass',
  47: 'cup',
  48: 'fork',
  49: 'knife',
  50: 'spoon',
  51: 'bowl',
  52: 'banana',
  53: 'apple',
  54: 'sandwich',
  55: 'orange',
  56: 'broccoli',
  57: 'carrot',
  58: 'hot dog',
  59: 'pizza',
  60: 'donut',
  61: 'cake',
  62: 'chair',
  63: 'couch',
  64: 'potted plant',
  65: 'bed',
  67: 'dining table',
  70: 'toilet',
  72: 'tv',
  73: 'laptop',
  74: 'mouse',
  75: 'remote',
  76: 'keyboard',
  77: 'cell phone',
  78: 'microwave',
  79: 'oven',
  80: 'toaster',
  81: 'sink',
  82: 'refrigerator',
  84: 'book',
  85: 'clock',
  86: 'vase',
  87: 'scissors',
  88: 'teddy bear',
  89: 'hair drier',
  90: 'toothbrush',
};
export const ALIASES: Record<string, string> = {
  'people': 'person',
  'human': 'person',
  'bike': 'bicycle',
  'motorbike': 'motorcycle',
  'plane': 'airplane',
  'aeroplane': 'airplane',
  'ball': 'sports ball',
  'phone': 'cell phone',
  'mobile phone': 'cell phone',
  'television': 'tv',
  'sofa': 'couch',
  'plant': 'potted plant',
  'table': 'dining table',
  'hairdryer': 'hair drier',
  'hair dryer': 'hair drier',
  'puppy': 'dog',
  'kitten': 'cat',
  'lorry': 'truck',
};
const NAME_TO_ID = new Map<string, number>(Object.entries(COCO_91).map(([i, n]) => [n, Number(i)]));

export class UnknownClass extends Error {}

export const normalise = (name: string): string => name.trim().toLowerCase().replace(/_/g, ' ').replace(/-/g, ' ').split(/\s+/).filter(Boolean).join(' ');

/** Python's repr() of a string, for the same messages as classes.py. */
const pyRepr = (s: string): string => (s.includes("'") && !s.includes('"') ? `"${s}"` : `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`);

/** classes.class_id: case, '_' and '-' do not matter; a few aliases (bike, phone, sofa) work. */
export function classId(name: string): number {
  let n = normalise(name);
  n = Object.prototype.hasOwnProperty.call(ALIASES, n) ? ALIASES[n] : n;
  const id = NAME_TO_ID.get(n);
  if (id !== undefined) return id;
  const near = [...NAME_TO_ID.keys()].filter((c) => n && (c.startsWith(n.slice(0, 3)) || c.includes(n))).sort().slice(0, 6);
  throw new UnknownClass(`unknown class ${pyRepr(name)}` + (near.length ? `; did you mean ${near.join(', ')}?` : '') + '; the 80 COCO classes are listed in src/inframe/classes.py');
}

export const className = (id: number): string => COCO_91[id];
/** The 80 class names, in id order (what the class picker lists). */
export const CLASS_NAMES: string[] = Object.keys(COCO_91).map((i) => COCO_91[Number(i)]);

// ---------------------------------------------------------------- targets
export interface Hit {
  /** 0 when the target is not in the frame */
  score: number;
  box: Box4 | null;
  trackId: number | null;
}

export interface Target {
  name: string;
  /** COCO id of the class the detector looks for */
  cls: number;
  /** looks at neither the pixels nor the past: a wrapper may filter the candidates before it */
  stateless: boolean;
  /** `t` is the file time of the picture; the identity-free targets ignore it */
  evaluate(candidates: readonly Detection[], t?: number): Hit;
  /** A target that decides after the last picture (BrowserTap) answers here for every `evaluate` call, in call order. */
  finalize?(): Hit[];
  /** The candidates need their DINOv2 `embedding` before `evaluate` (the Worker computes them). */
  needsEmbeddings?: boolean;
}

/** Every person is the target. Score = best person score; boxes shorter than `minBox` (fraction of frame height) are ignored. */
export class AnyPerson implements Target {
  name = 'any-person';
  cls = PERSON;
  readonly stateless = true;
  constructor(readonly minBox = 0) {}

  evaluate(candidates: readonly Detection[]): Hit {
    for (const d of candidates) {
      // best first
      if (d.box[3] - d.box[1] >= this.minBox) return { score: d.score, box: d.box, trackId: d.trackId ?? null };
    }
    return { score: 0, box: null, trackId: null };
  }
}

/** Every instance of one class is the target (`class:dog`): the same rule for another COCO class. */
export class AnyOfClass extends AnyPerson {
  constructor(cls: number, minBox = 0) {
    super(minBox);
    if (!(cls in COCO_91)) throw new UnknownClass(`unknown class id ${cls}`);
    this.cls = cls;
    this.name = `class:${COCO_91[cls]}`;
  }
}

// ---------------------------------------------------------------- zone
const EPS = 1e-9;
export type ZoneRect = readonly [number, number, number, number];

/** zone.Zone: x, y, w, h as fractions of the upright frame. */
export class Zone {
  constructor(readonly x: number, readonly y: number, readonly w: number, readonly h: number) {
    const vals = [x, y, w, h];
    if (!vals.every((v) => typeof v === 'number' && Number.isFinite(v))) throw new Error(`zone needs four numbers x,y,w,h (got ${vals})`);
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > 1 + 1e-6 || y + h > 1 + 1e-6) {
      throw new Error('zone x,y,w,h are fractions of the upright frame: x and y from 0, w and h above 0, x+w and y+h at most 1');
    }
  }

  static of(z: ZoneRect | Zone): Zone {
    return z instanceof Zone ? z : new Zone(z[0], z[1], z[2], z[3]);
  }

  containsPoint(px: number, py: number): boolean {
    return this.x - EPS <= px && px <= this.x + this.w + EPS && this.y - EPS <= py && py <= this.y + this.h + EPS;
  }

  /** Fraction (0..1) of the box (x0, y0, x1, y1) that lies inside the zone. */
  overlap(box: Box4): number {
    const [x0, y0, x1, y1] = box;
    const area = Math.max(x1 - x0, 0) * Math.max(y1 - y0, 0);
    if (area <= 0) return 0;
    const iw = Math.min(x1, this.x + this.w) - Math.max(x0, this.x);
    const ih = Math.min(y1, this.y + this.h) - Math.max(y0, this.y);
    return (Math.max(iw, 0) * Math.max(ih, 0)) / area;
  }

  /** Centre rule, or the overlap rule when `minOverlap` is given. */
  admits(box: Box4, minOverlap: number | null = null): boolean {
    if (minOverlap === null) return this.containsPoint((box[0] + box[2]) / 2, (box[1] + box[3]) / 2);
    return this.overlap(box) >= minOverlap - EPS;
  }
}

/** Wraps a target so only people in the zone count. A stateless inner target gets only the candidates the zone admits, so a person
 *  outside cannot hide one inside; any other target sees every candidate and its answer is dropped when its box is outside the zone. */
export class ZoneTarget implements Target {
  readonly stateless: boolean;
  readonly zone: Zone;
  readonly needsEmbeddings: boolean;
  readonly finalize?: () => Hit[];
  constructor(readonly inner: Target, zone: ZoneRect | Zone, readonly minOverlap: number | null = null) {
    this.zone = Zone.of(zone);
    this.stateless = inner.stateless;
    this.needsEmbeddings = inner.needsEmbeddings === true;
    if (inner.finalize) this.finalize = () => inner.finalize!().map((h) => this.gate(h));
    if (minOverlap !== null && !(minOverlap > 0 && minOverlap <= 1)) throw new Error(`zone overlap must be a fraction above 0 and at most 1 (got ${minOverlap})`);
  }

  get name(): string {
    return this.inner.name;
  }
  get cls(): number {
    return this.inner.cls;
  }

  private gate(h: Hit): Hit {
    if (h.box !== null && !this.zone.admits(h.box, this.minOverlap)) return { score: 0, box: null, trackId: null };
    return h;
  }

  evaluate(candidates: readonly Detection[], t?: number): Hit {
    if (this.stateless) return this.inner.evaluate(candidates.filter((d) => this.zone.admits(d.box, this.minOverlap)), t);
    return this.gate(this.inner.evaluate(candidates, t));
  }
}

// ---------------------------------------------------------------- tap
/** A tap that cannot be turned into a target (tap.TapError); the message says what to do. */
export class TapError extends EngineError {
  constructor(message: string) {
    super(422, message);
    this.name = 'TapError';
  }
}

export const TAP_WINDOW_S = 1.0; // tap.WINDOW_S: a tap needs a detection at a sample this close in time
export const TAP_MIN_IOU = 0.3; // tap.MIN_IOU: a tapped box must overlap a detection at least this much
/** The look of the candidate may be this much short of the threshold when its box continues the last match (box IoU, up to this gap in time). */
export const CONTINUITY_BONUS = 0.05;
export const CONTINUITY_IOU = 0.3;
export const CONTINUITY_GAP_S = 1.0;

/** tap.iou */
export function iou(a: readonly number[], b: readonly number[]): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

export interface TapPoint {
  /** file time of the tapped picture */
  t: number;
  box: Box4;
}

interface TapSample {
  t: number;
  cands: readonly Detection[];
}

/** One person followed by how they look (docs/T9B_DESIGN.md section 5). There is no tracker here (OC-SORT is Python/OpenCV), so:
 *  1. every tap is resolved on the pictures within TAP_WINDOW_S of it: the detection whose box overlaps the tapped box (IoU >= 0.3, the best one
 *     of the nearest picture that has one) is the person, else a TapError;
 *  2. the embeddings of those detections seed a Gallery (dedup 0.95);
 *  3. the pictures are walked forwards and then backwards in time. In a picture, a candidate matches when its cosine to the gallery (plus
 *     CONTINUITY_BONUS if its box continues the last match) reaches 0.66, its own cosine is at least 0.62, and it leads the runner-up by 0.04.
 *     A match joins the gallery (its cosine is at least the grow threshold 0.62), so the look grows with the person (a second outfit, another side);
 *  4. a matched picture scores with the detector's own score for that box, so the thresholds of the segment logic mean the same as for any person.
 *  Every picture must be shown to `evaluate` (with its time); `finalize` answers for all of them, in the order they were shown. */
export class BrowserTap implements Target {
  readonly name = 'tap';
  readonly stateless = false;
  readonly needsEmbeddings = true;
  readonly cls: number;
  private samples: TapSample[] = [];
  /** the gallery as it ended (for tests and the run's notes) */
  gallery: Gallery | null = null;

  constructor(readonly taps: readonly TapPoint[], cls: number = PERSON, readonly spec = DINOV2) {
    if (!taps.length) throw new TapError('Tap the person in the video first.');
    this.cls = cls;
  }

  evaluate(candidates: readonly Detection[], t?: number): Hit {
    if (t === undefined || !Number.isFinite(t)) throw new Error('BrowserTap needs the time of each picture');
    this.samples.push({ t, cands: candidates });
    return { score: 0, box: null, trackId: null }; // provisional; finalize() decides
  }

  /** tap._resolve: (sample, candidate) the tap lands on: the best overlap among the detections that have a look, on the nearest picture that has one. */
  private resolve(tap: TapPoint): [number, number] {
    const near = this.samples
      .map((s, si) => [Math.abs(s.t - tap.t), si] as const)
      .filter(([dt]) => dt <= TAP_WINDOW_S + 1e-6)
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let unembedded: number | null = null; // a picture where the tap landed on a detection that was never embedded
    for (const [, si] of near) {
      let best = -1;
      let bestIou = -1;
      this.samples[si].cands.forEach((d, ci) => {
        const o = iou(tap.box, d.box);
        if (o < TAP_MIN_IOU) return;
        if (!d.embedding) {
          unembedded ??= si; // a faint duplicate or a box too small to crop: another detection or picture may still serve
          return;
        }
        if (o > bestIou) {
          best = ci;
          bestIou = o;
        }
      });
      if (best >= 0) return [si, best];
    }
    if (unembedded !== null) throw new TapError(`the person at the tap at ${tap.t.toFixed(1)} s is too small to recognise; tap them closer to the camera, or at a moment when they are bigger`);
    throw new TapError(`no ${COCO_91[this.cls] ?? 'object'} detected within ${TAP_WINDOW_S} s of the tap at ${tap.t.toFixed(1)} s (nothing at that box); pick another moment or position`);
  }

  finalize(): Hit[] {
    const S = this.samples;
    const accepted = new Map<number, number>(); // sample -> candidate
    const gallery = new Gallery(this.spec);
    this.gallery = gallery;
    for (const tap of this.taps) {
      const [si, ci] = this.resolve(tap);
      const e = S[si].cands[ci].embedding!; // resolve only returns embedded candidates
      accepted.set(si, ci);
      gallery.add(e);
    }
    const byTime = S.map((_, i) => i).sort((a, b) => S[a].t - S[b].t || a - b);

    const sweep = (order: number[]) => {
      let last: { box: Box4; t: number } | null = null;
      for (const si of order) {
        const s = S[si];
        const fixed = accepted.get(si);
        if (fixed !== undefined) {
          last = { box: s.cands[fixed].box, t: s.t };
          const e = s.cands[fixed].embedding;
          if (e) gallery.add(e);
          continue;
        }
        const scored: { ci: number; cos: number; total: number }[] = [];
        s.cands.forEach((d, ci) => {
          if (!d.embedding) return;
          const cos = gallery.nearest(d.embedding).best;
          const cont = last !== null && Math.abs(s.t - last.t) <= CONTINUITY_GAP_S + 1e-6 && iou(last.box, d.box) >= CONTINUITY_IOU;
          scored.push({ ci, cos, total: cos + (cont ? CONTINUITY_BONUS : 0) });
        });
        if (!scored.length) continue;
        scored.sort((a, b) => b.total - a.total || a.ci - b.ci);
        const [best, second] = scored;
        if (best.total < this.spec.session - 1e-12 || best.cos < this.spec.grow - 1e-12) continue;
        if (second && best.total - second.total < this.spec.margin - 1e-12) continue; // two look-alikes: neither is taken
        accepted.set(si, best.ci);
        last = { box: s.cands[best.ci].box, t: s.t };
        gallery.add(s.cands[best.ci].embedding!);
      }
    };
    sweep(byTime);
    sweep([...byTime].reverse());

    return S.map((s, si) => {
      const ci = accepted.get(si);
      if (ci === undefined) return { score: 0, box: null, trackId: null };
      const d = s.cands[ci];
      return { score: d.score, box: d.box, trackId: null };
    });
  }
}

/** The target of a run request: any person, a class or a tap, optionally inside a zone. Throws for saved targets and photos. */
export function targetFor(req: { kind: string; class?: string }, opts: { zone?: ZoneRect | null; zoneOverlap?: number | null; minBox?: number; taps?: readonly TapPoint[] } = {}): Target {
  const minBox = opts.minBox ?? 0;
  let t: Target;
  if (req.kind === 'any') t = new AnyPerson(minBox);
  else if (req.kind === 'class') t = new AnyOfClass(classId(req.class ?? ''), minBox);
  else if (req.kind === 'tap') t = new BrowserTap(opts.taps ?? [], classId(req.class || 'person'));
  else throw new Error(`the target kind ${req.kind} is not offered in the browser. Use My Mac for it.`);
  return opts.zone ? new ZoneTarget(t, opts.zone, opts.zoneOverlap ?? null) : t;
}
