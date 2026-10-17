// One run in the tab (docs/T9B_DESIGN.md section 2): pick the samples the way src/inframe/decode.py does, decode them with WebCodecs,
// make each picture the detector's 512 x 512 input (resample.ts), detect, let the target decide, and turn the scores into clips with segments.ts.
// Everything that decides something is pure or takes its parts as arguments, so it runs in Node with stand-ins; FrameRenderer needs a
// canvas and the real decoder and the detector need a browser [unverified in Node].
import type { Box4, Detection } from './detect';
import { INPUT_SIDE, preprocess } from './detect';
import { drawUpright, makeCanvas, uprightOfVideo, type AnyCanvas, type FrameDecoder, type OpenVideo } from './frames';
import { pictureColour, type StreamColour } from './colour';
import { matrixOf, scaledSize, squashRgba, uprightRgb, yuvToRgb, type YuvPicture } from './resample';
import type { SampleTable } from './demux';
import type { Target } from './targets';
import { EngineError, type Box, type DetectorClip, type RunData, type RunRequest, type Samples } from '../engine/types';
import { buildSegments } from '../model/segments';

export const EPS = 1e-4;
export const DETECT_ENTER = 0.5;
export const DETECT_EXIT = 0.3;
/** What the browser offers of pipeline.PRESETS: the small model at 5 samples a second, or at the keyframes only (the Mac's fast preset uses nano). */
export const BROWSER_PRESETS: Record<string, { model: 'small'; fps: number | null }> = { fast: { model: 'small', fps: null }, balanced: { model: 'small', fps: 5 } };

// ---------------------------------------------------------------- the plan (pure)
/** The sample indices (decode order numbers) the detector looks at, in presentation order: with a rate, the first picture at or after
 *  each k / fps slot (slots a sparse stream skips are dropped, not repeated); with null, every sync sample. Same loop as decode.iter_samples. */
export function pickSamples(s: SampleTable, fps: number | null, endTime: number): number[] {
  const out: number[] = [];
  const step = fps ? 1 / fps : 0;
  let k = 0;
  for (let j = 0; j < s.n; j++) {
    const i = s.order[j];
    const t = s.pts[i];
    if (t >= endTime + EPS) break;
    if (fps === null) {
      if (s.sync[i]) out.push(i);
      continue;
    }
    if (t + EPS < k * step) continue;
    k = Math.max(k + 1, Math.floor(t / step + EPS) + 1);
    out.push(i);
  }
  return out;
}

/** The most wanted pictures one merged run may hold. A run's wanted pictures go to the consumer as the decoder gives them (frames.ts), so this no longer
 *  bounds the open frames; it bounds how long one flush-to-flush stretch of decoding is. */
export const MAX_RUN_PICKS = 64;

/** The decoder runs for the picked samples: each starts at the sync sample before its pictures and ends at the last one it needs.
 *  A picture shown before the keyframe of its group (a leading picture of an open GOP) is dropped by a decoder that starts at that keyframe,
 *  so its run starts one keyframe earlier; runs that then overlap are merged (unless the merged run would hold more than MAX_RUN_PICKS wanted
 *  pictures: the pieces then overlap by a group, and the decoder hands each picture over once). */
export function planRuns(s: SampleTable, picks: readonly number[]): { runs: [number, number][]; want: Set<number> } {
  const prevSync = new Int32Array(s.n);
  let last = 0;
  for (let i = 0; i < s.n; i++) {
    if (s.sync[i]) last = i;
    prevSync[i] = last;
  }
  const end = new Map<number, number>();
  const start = new Map<number, number>();
  const wanted = new Uint32Array(s.n + 1); // wanted[i + 1] - wanted[a]: how many picks sit in a..i
  for (const i of picks) {
    const g = prevSync[i];
    end.set(g, Math.max(end.get(g) ?? g, i));
    if (s.pts[i] < s.pts[g] - EPS && g > 0) start.set(g, prevSync[g - 1]);
    wanted[i + 1] = 1;
  }
  for (let i = 0; i < s.n; i++) wanted[i + 1] += wanted[i];
  const raw = [...end].map(([g, b]): [number, number] => [start.get(g) ?? g, b]).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  const runs: [number, number][] = [];
  for (const r of raw) {
    const cur = runs[runs.length - 1];
    if (cur && r[0] <= cur[1] && wanted[Math.max(cur[1], r[1]) + 1] - wanted[cur[0]] <= MAX_RUN_PICKS) cur[1] = Math.max(cur[1], r[1]);
    else runs.push([r[0], r[1]]);
  }
  return { runs, want: new Set(picks) };
}

// ---------------------------------------------------------------- scoring
export interface Scored {
  times: number[];
  scores: number[];
  boxes: (Box4 | null)[];
  /** seconds spent inside the detector */
  detectS: number;
  /** seconds spent inside the embedder (tap targets only) */
  embedS?: number;
  wallS: number;
  planned: number;
}

export interface ScoreDeps {
  video: OpenVideo;
  decoder: Pick<FrameDecoder, 'decode'>;
  /** the NCHW tensor of one decoded picture (converts it, turns it upright, squashes it); the caller of this closes the frame */
  prepare(frame: VideoFrame): Float32Array | Promise<Float32Array>;
  /** detections of the class the target looks for, best first */
  detect(tensor: Float32Array): Promise<Detection[]>;
  /** Fills `embedding` on the detections (tap targets: the Worker crops the picture `prepare` just drew). Called only when the target asks for it. */
  embed?(dets: Detection[], t: number): Promise<void>;
  target: Target;
  now?: () => number;
}

const cancelled = (): EngineError => new EngineError(499, 'Cancelled.');

export async function scoreVideo(d: ScoreDeps, fps: number | null, endTime: number, signal?: AbortSignal, onProgress?: (fraction: number, message: string) => void): Promise<Scored> {
  const now = d.now ?? (() => performance.now());
  const s = d.video.demuxed.samples;
  const picks = pickSamples(s, fps, endTime);
  const { runs, want } = planRuns(s, picks);
  const rows: { t: number; score: number; box: Box4 | null }[] = [];
  let detectMs = 0;
  let embedMs = 0;
  const t0 = now();
  await d.decoder.decode(d.video, runs, want, {
    signal,
    onFrame: async (i, frame) => {
      try {
        if (signal?.aborted) throw cancelled();
        const tensor = await d.prepare(frame);
        const a = now();
        const dets = await d.detect(tensor);
        detectMs += now() - a;
        if (d.target.needsEmbeddings && d.embed) {
          const e = now();
          await d.embed(dets, s.pts[i]);
          embedMs += now() - e;
        }
        const h = d.target.evaluate(dets, s.pts[i]);
        rows.push({ t: s.pts[i], score: h.score, box: h.box });
        onProgress?.(Math.min(0.98, rows.length / Math.max(1, picks.length)), `detect: ${s.pts[i].toFixed(0)} of ${endTime.toFixed(0)} s`);
      } finally {
        frame.close();
      }
    },
  });
  if (signal?.aborted) throw cancelled();
  // a target that decides after the last picture (tap) answers for every picture it was shown, in the order it saw them
  const final = d.target.finalize?.();
  if (final) final.forEach((h, k) => { rows[k] = { t: rows[k].t, score: h.score, box: h.box }; });
  rows.sort((a, b) => a.t - b.t);
  return {
    times: rows.map((r) => r.t), scores: rows.map((r) => r.score), boxes: rows.map((r) => r.box), detectS: detectMs / 1000, ...(d.target.needsEmbeddings ? { embedS: embedMs / 1000 } : {}),
    wallS: (now() - t0) / 1000, planned: picks.length,
  };
}

// ---------------------------------------------------------------- the picture the detector sees
/** The planes of a decoded frame, copied out with `VideoFrame.copyTo`; null for a format this path does not convert (10-bit, 4:2:2, 4:4:4, RGB). */
async function planesOf(frame: VideoFrame, buf: { bytes: Uint8Array }, stream?: StreamColour): Promise<YuvPicture | null> {
  const fmt = frame.format;
  if (fmt !== 'NV12' && fmt !== 'I420' && fmt !== 'I420A') return null;
  const size = frame.allocationSize();
  if (buf.bytes.length < size) buf.bytes = new Uint8Array(size);
  const layout = await frame.copyTo(buf.bytes);
  const w = frame.visibleRect?.width ?? frame.codedWidth;
  const h = frame.visibleRect?.height ?? frame.codedHeight;
  // The Mac converts with what the stream says (an unnamed matrix is BT.601, limited range) and FFmpeg's siting for the codec. Chrome's colorSpace reports bt709 for a
  // stream that names nothing, so the stream's own tags (colour.ts) win and the frame's stand only where the stream's are not read.
  const colour = pictureColour(frame.colorSpace, stream);
  const m = matrixOf(colour.matrix);
  const at = (i: number, rows: number) => buf.bytes.subarray(layout[i].offset, layout[i].offset + layout[i].stride * rows);
  const ch = (h + 1) >> 1;
  const base = { width: w, height: h, fullRange: colour.fullRange, ...m, chromaLoc: colour.chromaLoc, y: at(0, h), yStride: layout[0].stride, u: at(1, ch), uStride: layout[1].stride };
  return fmt === 'NV12' ? { format: 'NV12', ...base } : { format: 'I420', ...base, v: at(2, ch), vStride: layout[2].stride };
}

/** Makes the tensor the detector gets from a decoded frame the way the Mac makes it (resample.ts): the planes are copied out, converted to RGB at the
 *  decode size with swscale's filter, turned upright, and squashed to the detector's square with PIL's BILINEAR. A format whose planes this does not
 *  convert falls back to the canvas (drawImage and its own smoothing), which is close but not the Mac's picture. The buffers are reused. */
export class FrameRenderer {
  private big: AnyCanvas | null = null;
  private small: AnyCanvas | null = null;
  private u;
  private copy = { bytes: new Uint8Array(0) };
  private rgb: Uint8Array = new Uint8Array(0);
  private turned: Uint8Array = new Uint8Array(0);
  private last: { data: Uint8Array; w: number; h: number } | null = null;
  private rgba: Uint8ClampedArray = new Uint8ClampedArray(0);
  /** `readable`: the upright 640 px picture is read back for crops (tap targets) */
  constructor(private video: OpenVideo, private side: number = INPUT_SIDE.small, private long = 640, private readable = false) {
    this.u = uprightOfVideo(video.demuxed, long);
  }

  /** The pixels of the last picture prepared (RGBA), for the crops of a tap target. */
  pixels(): { rgba: Uint8ClampedArray; w: number; h: number } {
    if (this.last) {
      const { data, w, h } = this.last;
      if (this.rgba.length < w * h * 4) this.rgba = new Uint8ClampedArray(w * h * 4);
      for (let i = 0, j = 0, n = w * h; i < n; i++) {
        this.rgba[j++] = data[i * 3];
        this.rgba[j++] = data[i * 3 + 1];
        this.rgba[j++] = data[i * 3 + 2];
        this.rgba[j++] = 255;
      }
      return { rgba: this.rgba.subarray(0, w * h * 4), w, h };
    }
    if (!this.big) throw new Error('no picture has been prepared yet');
    const ctx = this.big.getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D;
    return { rgba: ctx.getImageData(0, 0, this.u.outW, this.u.outH).data, w: this.u.outW, h: this.u.outH };
  }

  prepare = async (frame: VideoFrame): Promise<Float32Array> => {
    const pic = await planesOf(frame, this.copy, this.video.demuxed.colour);
    if (!pic) {
      this.last = null;
      return this.prepareCanvas(frame);
    }
    const dm = this.video.demuxed;
    // the Mac sizes the picture from the stored size (no sample aspect ratio), then turns it
    const [sw, sh] = scaledSize(pic.width, pic.height, this.long);
    if (this.rgb.length < sw * sh * 3) this.rgb = new Uint8Array(sw * sh * 3);
    const flat = yuvToRgb(pic, sw, sh, this.rgb);
    if (this.turned.length < sw * sh * 3) this.turned = new Uint8Array(sw * sh * 3);
    const up = uprightRgb(flat, sw, sh, dm.rotation, dm.mirrored, this.turned);
    this.last = this.readable ? up : null;
    return preprocess(squashRgba(up.data, up.w, up.h, this.side), this.side, this.side);
  };

  /** The canvas path of before, for formats resample.ts does not convert. */
  private prepareCanvas(frame: VideoFrame): Float32Array {
    const u = this.u;
    this.big ??= makeCanvas(u.outW, u.outH);
    this.small ??= makeCanvas(this.side, this.side);
    const bctx = this.big.getContext('2d', { willReadFrequently: this.readable }) as OffscreenCanvasRenderingContext2D;
    bctx.clearRect(0, 0, u.outW, u.outH);
    drawUpright(bctx, frame, u, 0, 0, u.outW, u.outH);
    const sctx = this.small.getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D;
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(this.big as CanvasImageSource, 0, 0, this.side, this.side);
    return preprocess(sctx.getImageData(0, 0, this.side, this.side).data, this.side, this.side);
  }
}

// ---------------------------------------------------------------- the run document
export interface Resolved {
  preset: string;
  /** null = keyframes only */
  fps: number | null;
  pad: number;
  min_len: number;
  merge_gap: number;
  enter: number;
  exit: number;
  min_box: number;
}

const num = (v: unknown, fallback: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);

/** The settings of a request, with the defaults of pipeline.Settings (and the checks of pipeline.run_cut that apply here). */
export function resolveSettings(req: Pick<RunRequest, 'preset' | 'advanced'>): Resolved {
  const preset = BROWSER_PRESETS[req.preset];
  if (!preset) throw new EngineError(400, `The preset ${req.preset} is not offered in the browser. Use fast or balanced.`);
  const a = req.advanced ?? {};
  const fpsAdv = a.fps;
  const fps = typeof fpsAdv === 'number' ? (fpsAdv > 0 ? fpsAdv : null) : preset.fps;
  const enter = a.enter === null || a.enter === undefined ? DETECT_ENTER : num(a.enter, DETECT_ENTER);
  const exit = a.exit === null || a.exit === undefined ? Math.min(DETECT_EXIT, enter) : num(a.exit, DETECT_EXIT);
  for (const [k, v] of [['enter', enter], ['exit', exit], ['min_box', num(a.min_box, 0)]] as const) {
    if (!(v >= 0 && v <= 1)) throw new EngineError(400, `${k} must be between 0 and 1 (got ${v}).`);
  }
  if (exit > enter) throw new EngineError(400, `exit (${exit}) must not exceed enter (${enter}).`);
  return { preset: req.preset, fps, pad: num(a.pad, 1), min_len: num(a.min_len, 1), merge_gap: num(a.merge_gap, 1.5), enter, exit, min_box: num(a.min_box, 0) };
}

/** cut.snap_to_keyframe: where a stream-copy cut requested at t really starts. */
export function snapToKeyframe(t: number, keyframes: readonly number[]): number {
  if (!keyframes.length) return t;
  let lo = 0;
  let hi = keyframes.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (keyframes[mid] <= t + 1e-6) lo = mid + 1;
    else hi = mid;
  }
  return keyframes[Math.max(lo - 1, 0)];
}

const r = (v: number, n: number) => Math.round(v * 10 ** n) / 10 ** n;

export interface SourceFacts {
  offset: number;
  end_time: number;
  keyframes: readonly number[];
}

/** The run's segments.json content and samples.csv content, as pipeline.run_cut writes them (without the file names of cut clips). */
export function buildRunDoc(
  scored: Scored, st: Resolved, src: SourceFacts, info: { target: string; cls: number; zone: Box | null; zoneOverlap: number | null; detector: string; model: string; warnings: string[]; totalS: number },
): { segments: RunData['segments']; samples: Samples } {
  const { presence, clips: padded } = buildSegments(scored.times, scored.scores, src.end_time, { enter: st.enter, exit: st.exit, mergeGap: st.merge_gap, minLen: st.min_len, pad: st.pad, lo: src.offset });
  const warnings = [...info.warnings];
  // a picture the decoder never gave back is missing from the scores: say so instead of letting the gap pass as "nobody there"
  const dropped = Math.max(0, scored.planned - scored.times.length);
  if (dropped) warnings.push(`${dropped} of the ${scored.planned} planned pictures did not come back from the decoder, so the scores have gaps there`);
  if (!presence.length && scored.scores.length) {
    warnings.push(`no stretch found: the highest score ${scored.scores.reduce((a, b) => Math.max(a, b), 0).toFixed(5)} never reached the enter threshold ${st.enter.toFixed(5)} for ${st.min_len} s; check the enter threshold or the target`);
  }
  const clips: DetectorClip[] = padded.map(([s, e], i) => {
    let peak = 0;
    scored.times.forEach((t, k) => {
      if (t >= s && t <= e && scored.scores[k] > peak) peak = scored.scores[k];
    });
    return { index: i + 1, start: r(s, 3), end: r(e, 3), peak_score: r(peak, 5), expected_copy_start: r(snapToKeyframe(s, src.keyframes), 3) };
  });
  const sc = scored.scores;
  const summary = sc.length
    ? { n_samples: sc.length, n_planned: scored.planned, n_dropped: dropped, mean: r(sc.reduce((a, b) => a + b, 0) / sc.length, 6), max: r(sc.reduce((a, b) => Math.max(a, b), 0), 6), frac_at_or_above_enter: r(sc.filter((x) => x >= st.enter).length / sc.length, 4) }
    : { n_samples: 0, n_planned: scored.planned, n_dropped: dropped };
  const segments: RunData['segments'] = {
    settings: {
      mode: 'detect', preset: st.preset, preset_name: st.preset, fps: st.fps ?? 'keyframes', pad: st.pad, min_len: st.min_len, merge_gap: st.merge_gap, enter: st.enter, exit: st.exit,
      min_box: st.min_box, zone: info.zone, zone_overlap: info.zoneOverlap, target: info.target, thresholds: { source: 'default' }, detector: info.detector, model: info.model,
      class_id: info.cls, engine: 'browser',
    },
    warnings,
    scores: summary,
    timing_s: { total: r(info.totalS, 2), score: r(scored.wallS, 2), detect_model_s: r(scored.detectS, 2), ...(scored.embedS !== undefined ? { embed_model_s: r(scored.embedS, 2) } : {}), cut: 0 },
    presence_intervals: presence.map(([s, e]) => [r(s, 3), r(e, 3)] as [number, number]),
    clips,
    error: null,
  };
  const samples: Samples = {
    t: scored.times.map((t) => r(t, 3)), score: scored.scores.map((x) => r(x, 5)), box: scored.boxes.map((b) => (b ? (b.map((v) => r(v, 4)) as Box) : null)), track: null,
    enter: st.enter, exit: st.exit, merge_gap: st.merge_gap, min_len: st.min_len, pad: st.pad, sound: null, mode: 'detect',
  };
  return { segments, samples };
}
