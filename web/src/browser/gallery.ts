// The tap gallery (docs/T9B_DESIGN.md section 5): exact ports of identity.Gallery (add with dedup, evict the member most like another one) and
// identity.unit_score, held to the golden vectors of tools/gen_golden.py (tests/golden/gallery_v1.json). Vectors are float32 rows; dot
// products are summed in doubles here and in float32 in numpy, so a decision is only compared away from its threshold (the generator checks it).

/** identity.Spec, the numbers of one embedder. */
export interface Spec {
  /** cosine a track must reach against the photos */
  threshold: number;
  /** ... or against the session gallery */
  session: number;
  /** the best must beat a rival that shares a frame with it by this much */
  margin: number;
  /** a crop this close to the gallery joins it */
  grow: number;
  /** a crop this close to a gallery member is not added again */
  dedup: number;
}

/** identity.SPECS["dinov2"] */
export const DINOV2: Spec = { threshold: 0.62, session: 0.66, margin: 0.04, grow: 0.62, dedup: 0.95 };
export const GALLERY_CAP = 24;

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

/** tap.similarity: histogram intersection of two colour signatures, 0 (nothing alike) to 1 (same). */
export function similarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.min(a[i], b[i]);
  return s / 2;
}

export class Gallery<T = unknown> {
  v: Float32Array[] = [];
  sig: (ArrayLike<number> | null)[] = [];
  tags: (T | undefined)[] = [];

  constructor(readonly spec: Spec = DINOV2, readonly cap: number = GALLERY_CAP) {}

  get length(): number {
    return this.sig.length;
  }

  /** The best cosine of `vec` to a member, and which member (the first on a tie); -Infinity for an empty gallery. */
  nearest(vec: ArrayLike<number>): { best: number; at: number } {
    let best = -Infinity;
    let at = -1;
    for (let j = 0; j < this.v.length; j++) {
      if (this.v[j].length !== vec.length) continue;
      const d = dot(this.v[j], vec);
      if (d > best) {
        best = d;
        at = j;
      }
    }
    return { best, at };
  }

  /** true when the vector joined; false when it was too close to a member (dedup). */
  add(vec: Float32Array, sig: ArrayLike<number> | null = null, tag?: T): boolean {
    if (this.length && this.v[0].length === vec.length) {
      const { best, at } = this.nearest(vec);
      if (best >= this.spec.dedup) {
        const other = this.sig[at];
        if (sig === null || other === null || similarity(sig, other) >= 0.9) return false;
      }
    } else if (this.length) {
      throw new Error(`gallery: a vector of ${vec.length} numbers cannot join members of ${this.v[0].length}`);
    }
    this.v.push(Float32Array.from(vec));
    this.sig.push(sig);
    this.tags.push(tag);
    if (this.length > this.cap) this.evict();
    return true;
  }

  /** Drops the member most like another one (embedding and colour signature both count). */
  private evict(): void {
    const n = this.length;
    let k = 0;
    let top = -Infinity;
    for (let i = 0; i < n; i++) {
      let rowMax = -Infinity;
      for (let j = 0; j < n; j++) {
        if (i === j) {
          rowMax = Math.max(rowMax, -1);
          continue;
        }
        const e = dot(this.v[i], this.v[j]);
        const si = this.sig[i];
        const sj = this.sig[j];
        const s = si === null || sj === null ? e : 0.5 * e + 0.5 * similarity(si, sj);
        if (s > rowMax) rowMax = s;
      }
      if (rowMax > top) {
        top = rowMax;
        k = i;
      }
    }
    this.v.splice(k, 1);
    this.sig.splice(k, 1);
    this.tags.splice(k, 1);
  }
}

/** identity.unit_score: the median over the track's crops of the best cosine to any gallery member (0 for nothing, or rows of another length). */
export function unitScore(vecs: readonly Float32Array[], gallery: readonly Float32Array[]): number {
  if (!vecs.length || !gallery.length || vecs[0].length !== gallery[0].length) return 0;
  const best = vecs.map((v) => gallery.reduce((m, g) => Math.max(m, dot(v, g)), -Infinity)).sort((a, b) => a - b);
  const mid = best.length >> 1;
  return best.length % 2 ? best[mid] : (best[mid - 1] + best[mid]) / 2;
}
