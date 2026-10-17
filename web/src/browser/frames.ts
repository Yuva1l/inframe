// Frames out of a video, in the tab (docs/T9B_DESIGN.md section 2): a VideoDecoder fed from the sample table, and an upright picture
// drawn on a canvas whose long side is 640 px (the size the detector gets in part 3). The decode plan and the upright geometry are
// pure functions (tested in Node); the VideoDecoder and canvas parts need a browser [unverified here: no WebCodecs in Node].
import { EngineError } from '../engine/types';
import type { ByteSource, Demuxed, SampleTable } from './demux';

export const FRAME_LONG_SIDE = 640;
/** the biggest span of file bytes read in one slice when decoding a run of samples; a wider one is read sample by sample */
const MAX_SPAN = 48 * 1024 * 1024;

/** Everything needed to decode and draw one video: the demux result and where the bytes live. */
export interface OpenVideo {
  demuxed: Demuxed;
  src: ByteSource;
}

// ---------------------------------------------------------------- the plan (pure)
/** The sample shown at time t (file time): the last one in presentation order with pts <= t, else the first. */
export function sampleAt(s: SampleTable, t: number): number {
  let lo = 0;
  let hi = s.n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (s.pts[s.order[mid]] <= t + 1e-6) lo = mid + 1;
    else hi = mid;
  }
  return s.order[Math.max(0, lo - 1)];
}

/** The sync sample at or before decode index i (0 when none). */
export function syncBefore(s: SampleTable, i: number): number {
  for (let k = i; k >= 0; k--) if (s.sync[k]) return k;
  return 0;
}

/** The sample indices to feed the decoder to show the picture at time t: from the sync sample before it up to it. A flush then gives
 *  every picture decoded so far, the wanted one among them. A picture shown before the keyframe of its group (a leading picture of an open
 *  GOP) is dropped by a decoder started at that keyframe, so the run then starts one keyframe earlier (same rule as pipeline.planRuns). */
export function decodePlan(s: SampleTable, t: number): { target: number; first: number; last: number } {
  const target = sampleAt(s, t);
  const sync = syncBefore(s, target);
  const first = sync > 0 && s.pts[target] < s.pts[sync] - 1e-4 ? syncBefore(s, sync - 1) : sync;
  return { target, first, last: target };
}

export interface Upright {
  /** the picture as shown: the stored size scaled by the sample aspect ratio, then turned */
  w: number;
  h: number;
  /** canvas size so that the long side is `long` px (never enlarged) */
  outW: number;
  outH: number;
  /** clockwise radians the canvas turns the stored picture by (ffprobe's rotation is counter-clockwise) */
  theta: number;
  flipY: boolean;
  sar: number;
}

/** The upright geometry for a stored picture (`width` x `height`): sample aspect, then rotation, then the vertical flip of a mirrored
 *  matrix (probe.py: upright after the rotation and then a vertical flip). */
export function uprightOf(width: number, height: number, rotation: number, mirrored: boolean, sar = 1, long = FRAME_LONG_SIDE): Upright {
  const s = sar > 0 ? sar : 1;
  const quarter = ((rotation % 180) + 180) % 180 === 90;
  const w = quarter ? height : width * s;
  const h = quarter ? width * s : height;
  const k = Math.min(1, long / Math.max(w, h));
  return { w, h, outW: Math.max(1, Math.round(w * k)), outH: Math.max(1, Math.round(h * k)), theta: (-rotation * Math.PI) / 180, flipY: mirrored, sar: s };
}

export const uprightOfVideo = (d: Demuxed, long = FRAME_LONG_SIDE): Upright => uprightOf(d.video.width, d.video.height, d.rotation, d.mirrored, d.sar, long);

// ---------------------------------------------------------------- drawing
type Ctx = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
export type AnyCanvas = OffscreenCanvas | HTMLCanvasElement;

export function makeCanvas(w: number, h: number): AnyCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** Draw a decoded frame upright into the rectangle (x, y, w, h) of `ctx`: scaled to fit inside it, or to cover it (centre crop). */
export function drawUpright(ctx: Ctx, frame: VideoFrame, u: Upright, x: number, y: number, w: number, h: number, cover = false): void {
  const fit = (cover ? Math.max : Math.min)(w / u.w, h / u.h);
  const dw = (frame.displayWidth || frame.codedWidth) * u.sar * fit;
  const dh = (frame.displayHeight || frame.codedHeight) * fit;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.translate(x + w / 2, y + h / 2);
  if (u.flipY) ctx.scale(1, -1); // applied after the rotation below: canvas transforms act on the drawing last-called first
  ctx.rotate(u.theta);
  ctx.drawImage(frame, -dw / 2, -dh / 2, dw, dh);
  ctx.restore();
}

export async function canvasBlob(c: AnyCanvas, type = 'image/jpeg', quality = 0.85): Promise<Blob> {
  if ('convertToBlob' in c) return c.convertToBlob({ type, quality });
  return new Promise((resolve, reject) => (c as HTMLCanvasElement).toBlob((b) => (b ? resolve(b) : reject(new Error('could not encode the picture'))), type, quality));
}

// ---------------------------------------------------------------- decoding
/** Read the bytes of samples first..last: one slice when they sit close together, else one read each. */
async function readRange(v: OpenVideo, first: number, last: number): Promise<Uint8Array[]> {
  const s = v.demuxed.samples;
  let lo = Infinity;
  let hi = 0;
  for (let i = first; i <= last; i++) {
    lo = Math.min(lo, s.offset[i]);
    hi = Math.max(hi, s.offset[i] + s.size[i]);
  }
  const out: Uint8Array[] = [];
  if (hi - lo <= MAX_SPAN) {
    const buf = new Uint8Array(await v.src.read(lo, hi));
    for (let i = first; i <= last; i++) out.push(buf.subarray(s.offset[i] - lo, s.offset[i] - lo + s.size[i]));
    return out;
  }
  for (let i = first; i <= last; i++) out.push(new Uint8Array(await v.src.read(s.offset[i], s.offset[i] + s.size[i])));
  return out;
}

export interface DecoderDeps {
  VideoDecoder: typeof VideoDecoder;
  EncodedVideoChunk: typeof EncodedVideoChunk;
}

const browserDeps = (): DecoderDeps | null =>
  typeof VideoDecoder === 'undefined' || typeof EncodedVideoChunk === 'undefined' ? null : { VideoDecoder, EncodedVideoChunk };

/** One decoder, one job at a time (a hardware decoder is a scarce thing). Frames come back keyed by sample index; the caller closes them. */
export interface DecodeOpts {
  /** a job still waiting in the queue (or between two runs) when this fires is dropped; a requester that went away passes it */
  signal?: AbortSignal;
  /** Hand each wanted picture over as soon as the decoder gives it (checked after every chunk fed, and after the flush of each run), instead of returning them all at the end. The callee closes it, and may be async: the next picture waits for it.
   *  A hardware decoder has a small pool of output buffers: pictures held across several flushes can stall it for good. */
  onFrame?: (sample: number, frame: VideoFrame) => void | Promise<void>;
}

/** The most chunks left waiting in the decoder's queue while a run is fed: the feed waits for it to take them, so the pictures it gives are handed over (and
 *  closed) while the run is still going, not all at its flush. A decoder needs a few chunks ahead of it to reorder and to keep its hardware busy. */
const MAX_QUEUED = 4;
/** the longest the feed waits for one 'dequeue' event before it looks again (a decoder that never fires one must not stall the run) */
const DEQUEUE_WAIT_MS = 25;

/** Let the tasks already posted run: a decoder's output callbacks arrive as tasks, and microtasks do not wait for them. */
function nextTask(): Promise<void> {
  if (typeof MessageChannel !== 'undefined') {
    return new Promise((resolve) => {
      const c = new MessageChannel();
      c.port1.onmessage = () => {
        c.port1.close();
        resolve();
      };
      c.port2.postMessage(0);
    });
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** Resolves when the decoder takes a chunk off its queue (its 'dequeue' event), or after `DEQUEUE_WAIT_MS`. */
function dequeued(dec: VideoDecoder): Promise<void> {
  return new Promise((resolve) => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const done = () => {
      clearTimeout(timer);
      dec.removeEventListener?.('dequeue', done);
      resolve();
    };
    dec.addEventListener?.('dequeue', done);
    timer = setTimeout(done, DEQUEUE_WAIT_MS);
  });
}

const cancelled = (): EngineError => new EngineError(499, 'Cancelled.');

export class FrameDecoder {
  private chain: Promise<unknown> = Promise.resolve();

  constructor(private deps: DecoderDeps | null = browserDeps()) {}

  /** Decode each run (a [first, last] range of sample indices that starts at a sync sample, flushed at its end) and return the pictures
   *  of the samples in `want`, keyed by sample index. The caller closes them. With `onFrame` they go to it as the decoder gives them, and the map is empty. */
  decode(v: OpenVideo, runs: [number, number][], want: Set<number>, opts: DecodeOpts = {}): Promise<Map<number, VideoFrame>> {
    const job = this.chain.then(() => this.run(v, runs, want, opts));
    this.chain = job.catch(() => undefined);
    return job;
  }

  private async run(v: OpenVideo, runs: [number, number][], want: Set<number>, opts: DecodeOpts): Promise<Map<number, VideoFrame>> {
    const d = this.deps;
    if (opts.signal?.aborted) throw cancelled();
    if (!d) throw new EngineError(415, 'This browser cannot decode video (no WebCodecs). Use My Mac.');
    const vt = v.demuxed.video;
    const config: VideoDecoderConfig = { codec: vt.codecString, codedWidth: vt.width, codedHeight: vt.height, ...(vt.description ? { description: vt.description } : {}) };
    const sup = await d.VideoDecoder.isConfigSupported(config).catch(() => ({ supported: false }));
    if (!sup.supported) throw new EngineError(415, `This browser cannot decode ${v.demuxed.codec.toUpperCase()} video (${vt.codecString}). Use My Mac.`);
    const s = v.demuxed.samples;
    const byTs = new Map<number, number>();
    for (const [a, b] of runs) for (let i = a; i <= b; i++) byTs.set(s.tsUs[i], i);
    const got = new Map<number, VideoFrame>();
    const seen = new Set<number>(); // runs may overlap by a group (a leading picture's run starts a keyframe earlier): each picture is handed over once
    let failure: Error | null = null;
    const dec = new d.VideoDecoder({
      output: (f) => {
        const i = byTs.get(f.timestamp);
        if (i !== undefined && want.has(i) && !seen.has(i)) {
          seen.add(i);
          got.set(i, f);
        } else f.close();
      },
      error: (e) => {
        failure = e;
      },
    });
    try {
      dec.configure(config);
      // Hand over whatever wanted pictures the decoder has given so far, in presentation order, and wait for the callee to finish with each. The decoder's
      // output callbacks are tasks, so the feed below yields a task and then waits for the decoder's queue to shrink before each drain: a run holds a few
      // pictures open (what the decoder gave since the last drain), not all of its wanted ones until the flush.
      const drain = async () => {
        while (opts.onFrame && got.size) {
          const ready = [...got].sort((a, b) => s.pts[a[0]] - s.pts[b[0]]); // in presentation order, whatever order the decoder gave them
          got.clear();
          for (let k = 0; k < ready.length; k++) {
            try {
              await opts.onFrame(ready[k][0], ready[k][1]); // awaited: a slow consumer (the detector) holds the decoder back
            } catch (e) {
              for (const [, f] of ready.slice(k + 1)) f.close();
              throw e;
            }
          }
        }
      };
      for (const [a, b] of runs) {
        if (opts.signal?.aborted) throw cancelled();
        const bytes = await readRange(v, a, b);
        for (let i = a; i <= b; i++) {
          if (failure) break;
          dec.decode(new d.EncodedVideoChunk({ type: s.sync[i] ? 'key' : 'delta', timestamp: s.tsUs[i], data: bytes[i - a] }));
          await nextTask();
          await drain();
          while (!failure && typeof dec.decodeQueueSize === 'number' && dec.decodeQueueSize > MAX_QUEUED) {
            if (opts.signal?.aborted) throw cancelled();
            await dequeued(dec);
            await drain();
          }
        }
        await dec.flush();
        await drain();
      }
    } catch (e) {
      for (const f of got.values()) f.close();
      if (e instanceof EngineError) throw e;
      throw new EngineError(500, `Decoding failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      if (dec.state !== 'closed') dec.close();
    }
    if (failure) {
      for (const f of got.values()) f.close();
      throw new EngineError(500, `Decoding failed: ${(failure as Error).message}`);
    }
    return got;
  }
}

/** The JPEG of the picture at time t, upright, the long side at most `long` px. */
export async function frameJpeg(dec: Pick<FrameDecoder, 'decode'>, v: OpenVideo, t: number, long = FRAME_LONG_SIDE, signal?: AbortSignal): Promise<Blob> {
  const plan = decodePlan(v.demuxed.samples, t);
  const frames = await dec.decode(v, [[plan.first, plan.last]], new Set([plan.target]), { signal });
  const f = frames.get(plan.target);
  if (!f) throw new EngineError(500, 'The picture at that time could not be decoded.');
  try {
    const u = uprightOfVideo(v.demuxed, long);
    const c = makeCanvas(u.outW, u.outH);
    const ctx = c.getContext('2d') as Ctx;
    drawUpright(ctx, f, u, 0, 0, u.outW, u.outH);
    return await canvasBlob(c);
  } finally {
    f.close();
  }
}
