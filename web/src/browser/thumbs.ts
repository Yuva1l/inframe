// The filmstrip of the browser engine (docs/T9B_DESIGN.md section 2): one picture per keyframe, a cell for each slot of the timeline,
// packed 10 x 10 to a sheet like serve_media's filmstrip. Only keyframes are decoded (each one on its own), so a sheet costs 100
// single-sample decodes, not a pass over the video. Sheets are made one after the other; `cells` lists those finished so far.
import { EngineError, type Filmstrip } from '../engine/types';
import { canvasBlob, drawUpright, FrameDecoder, makeCanvas, syncBefore, uprightOfVideo, type OpenVideo } from './frames';

export const CELL_H = 108;
export const CELL_MAX_W = 192;
export const COLS = 10;
export const ROWS = 10;
export const SLOTS_MAX = 600; // one cell per slot of max(1 s, duration / SLOTS_MAX)

export interface Plan {
  cell_w: number;
  cell_h: number;
  cols: number;
  rows: number;
  /** the chosen keyframes: file time, and the index of the sample */
  picks: { t: number; sample: number }[];
  sheets: number;
}

/** Which keyframes make the cells: for each slot of the timeline, the keyframe nearest its middle (a keyframe is used once). */
export function planFilmstrip(v: OpenVideo): Plan {
  const d = v.demuxed;
  const s = d.samples;
  const kf: { t: number; sample: number }[] = [];
  for (let i = 0; i < s.n; i++) if (s.sync[i] && s.pts[i] >= d.offset - 1e-6) kf.push({ t: s.pts[i], sample: i });
  kf.sort((a, b) => a.t - b.t);
  const dispW = d.rotation % 180 !== 0 ? d.video.height : d.video.width * d.sar;
  const dispH = d.rotation % 180 !== 0 ? d.video.width * d.sar : d.video.height;
  const cell_w = Math.max(16, Math.min(CELL_MAX_W, Math.round((CELL_H * dispW) / (dispH || 1))));
  const picks: Plan['picks'] = [];
  if (kf.length) {
    const slot = Math.max(1, d.duration / SLOTS_MAX);
    const n = Math.max(1, Math.ceil(d.duration / slot));
    let j = 0;
    let last = -1;
    for (let k = 0; k < n; k++) {
      const mid = d.offset + (k + 0.5) * slot;
      while (j + 1 < kf.length && Math.abs(kf[j + 1].t - mid) < Math.abs(kf[j].t - mid)) j++;
      if (j !== last) picks.push(kf[j]);
      last = j;
    }
  }
  return { cell_w, cell_h: CELL_H, cols: COLS, rows: ROWS, picks, sheets: Math.ceil(picks.length / (COLS * ROWS)) };
}

/** The canvas parts of the build; a test replaces them (there is no canvas in Node). */
export interface FilmOps {
  makeCanvas: typeof makeCanvas;
  draw: typeof drawUpright;
  blob: typeof canvasBlob;
}
const browserOps: FilmOps = { makeCanvas, draw: drawUpright, blob: canvasBlob };

export type FilmState = 'running' | 'done' | 'failed';

/** Builds the sheets in the background. `filmstrip()` is the Filmstrip of the sheets done so far; `sheet(n)` waits for sheet n. */
export class FilmstripBuilder {
  readonly plan: Plan;
  state: FilmState = 'running';
  error: string | null = null;
  private blobs: Blob[] = [];
  private waiters = new Map<number, ((b: Blob | null) => void)[]>();
  private cancelled = false;

  constructor(private v: OpenVideo, private dec: Pick<FrameDecoder, 'decode'>, private ops: FilmOps = browserOps) {
    this.plan = planFilmstrip(v);
  }

  start(): void {
    void this.run();
  }

  /** Stop after the batch in flight (the build queues one batch at a time, so nothing else is left on the decoder) and free what waits for a sheet. */
  cancel(): void {
    this.cancelled = true;
    if (this.state === 'running') {
      this.state = 'failed';
      this.error = 'The filmstrip was cancelled.';
    }
    this.settle();
  }

  filmstrip(): Filmstrip {
    const p = this.plan;
    const per = p.cols * p.rows;
    const done = this.blobs.length;
    const cells = p.picks.slice(0, Math.min(p.picks.length, done * per)).map((x, k) => ({ t: x.t, sheet: Math.floor(k / per), i: k % per }));
    return { version: 1, cell_w: p.cell_w, cell_h: p.cell_h, cols: p.cols, rows: p.rows, cells, sheets: p.sheets, done: this.state === 'done', state: this.state };
  }

  /** Sheet n as a JPEG; null once the build has failed or been cancelled before reaching it. */
  sheet(n: number): Promise<Blob | null> {
    if (n < this.blobs.length) return Promise.resolve(this.blobs[n]);
    if (this.state !== 'running') return Promise.resolve(null);
    return new Promise((resolve) => this.waiters.set(n, [...(this.waiters.get(n) ?? []), resolve]));
  }

  private settle(): void {
    for (const [n, list] of this.waiters) {
      if (n < this.blobs.length) for (const w of list) w(this.blobs[n]);
      else if (this.state !== 'running') for (const w of list) w(null);
      else continue;
      this.waiters.delete(n);
    }
  }

  private async run(): Promise<void> {
    const p = this.plan;
    const per = p.cols * p.rows;
    let anyDrawn = false; // once a cell has been drawn, a failing keyframe is just that keyframe; before, a whole first batch failing one by one fails the filmstrip
    try {
      const u = uprightOfVideo(this.v.demuxed, 1 << 20); // the geometry only; each cell is drawn to fit
      for (let n = 0; n < p.sheets && !this.cancelled; n++) {
        const cells = p.picks.slice(n * per, (n + 1) * per);
        // always the whole grid, like serve_media's sheets: a card positions its cell as row * cell_h inside rows * cell_h
        const canvas = this.ops.makeCanvas(p.cols * p.cell_w, p.rows * p.cell_h);
        const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // each keyframe is a run of its own (the sample alone): 100 small decodes, flushed together in batches of 20
        for (let b = 0; b < cells.length && !this.cancelled; b += 20) {
          const batch = cells.slice(b, b + 20);
          const drawn = new Set<number>();
          const draw = (sample: number, f: VideoFrame, i: number) => {
            try {
              this.ops.draw(ctx, f, u, (i % p.cols) * p.cell_w, Math.floor(i / p.cols) * p.cell_h, p.cell_w, p.cell_h, true);
              drawn.add(sample);
              anyDrawn = true;
            } finally {
              f.close();
            }
          };
          // each frame is drawn and closed as soon as its run is flushed
          try {
            const idx = new Map(batch.map((c, k) => [c.sample, b + k]));
            await this.dec.decode(this.v, batch.map((c) => [syncBefore(this.v.demuxed.samples, c.sample), c.sample] as [number, number]), new Set(batch.map((c) => c.sample)), {
              onFrame: (sample, f) => draw(sample, f, idx.get(sample) ?? 0),
            });
          } catch (e) {
            // a decoder error (one bad keyframe) is not the codec being unsupported: try the cells that are still dark one by one and leave
            // the ones that fail dark; anything else (no WebCodecs, an unsupported codec) fails the filmstrip
            if (!(e instanceof EngineError) || e.status !== 500) throw e;
            let lastMiss: unknown = null;
            for (let k = 0; k < batch.length && !this.cancelled; k++) {
              const c = batch[k];
              if (drawn.has(c.sample)) continue;
              try {
                await this.dec.decode(this.v, [[syncBefore(this.v.demuxed.samples, c.sample), c.sample]], new Set([c.sample]), { onFrame: (sample, f) => draw(sample, f, b + k) });
              } catch (e2) {
                if (!(e2 instanceof EngineError) || e2.status !== 500) throw e2; // this cell stays dark
                lastMiss = e2;
              }
            }
            // nothing has ever decoded, not even one cell of this whole batch: the video cannot be decoded here (hardware decode failing), not a few
            // bad keyframes (a damaged start leaves dark cells and the rest is drawn); show the error
            if (!anyDrawn && lastMiss && !this.cancelled) throw lastMiss;
          }
        }
        if (this.cancelled) break;
        this.blobs.push(await this.ops.blob(canvas, 'image/jpeg', 0.8));
        this.settle();
      }
      if (!this.cancelled) this.state = 'done';
    } catch (e) {
      this.state = 'failed';
      this.error = e instanceof Error ? e.message : String(e);
    }
    this.settle();
  }
}
