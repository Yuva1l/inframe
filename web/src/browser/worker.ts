// The run, off the page's thread (docs/T9B_DESIGN.md section 2). The page sends one message per job (`run`, `boxes`, `probe`) and gets
// `progress` messages and one `done` or `error` back; `cancel` stops a job between two pictures. The handler is a plain function so the
// page can also run it on its own thread when a browser gives Workers no WebGPU (Safari, to be checked).
import { EngineError } from '../engine/types';
import { fileSource, type Demuxed } from './demux';
import { MIN_PERSON_SCORE, RfDetr, createSession, type Detection } from './detect';
import { FrameDecoder, decodePlan, type OpenVideo } from './frames';
import { FrameRenderer, scoreVideo, type Scored } from './pipeline';
import { DETECTOR_FILE, EMBEDDER_FILE, ModelStore, WASM_FILE, type Manifest } from './models';
import { Dino, embedDetections, MAX_EMBED, type EmbedStats } from './embed';
import { classId, targetFor, TAP_WINDOW_S, type TapPoint, type ZoneRect } from './targets';

interface Common {
  id: number;
  /** absolute base URL of the model files, and the page's copy of their manifest */
  base: string;
  manifest: Manifest;
}
export interface RunMsg extends Common {
  op: 'run';
  file: File;
  demuxed: Demuxed;
  target: { kind: string; class?: string };
  fps: number | null;
  zone: ZoneRect | null;
  zoneOverlap: number | null;
  minBox: number;
  /** the taps of a tap target (file time and box of each) */
  taps?: TapPoint[];
}
export interface BoxesMsg extends Common {
  op: 'boxes';
  file: File;
  demuxed: Demuxed;
  t: number;
  cls: string;
}
export interface ProbeMsg {
  op: 'probe';
  id: number;
}
export interface CancelMsg {
  op: 'cancel';
  id: number;
  /** the job to stop */
  job: number;
}
export type WorkerRequest = RunMsg | BoxesMsg | ProbeMsg | CancelMsg;

export interface RunResult {
  scored: Scored;
  provider: 'webgpu' | 'wasm';
  warnings: string[];
  cls: number;
  targetName: string;
}
export interface BoxesResult {
  /** the file time of the picture that was used */
  t: number;
  detections: Detection[];
}
export interface ProbeResult {
  webgpu: boolean;
  inWorker: boolean;
}
export type WorkerReply =
  | { id: number; type: 'progress'; fraction: number; message: string }
  | { id: number; type: 'done'; result: RunResult | BoxesResult | ProbeResult }
  | { id: number; type: 'error'; status: number; message: string; code?: string };

declare const WorkerGlobalScope: { new (): object } | undefined;
const inWorker = typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined' && self instanceof WorkerGlobalScope;

/** Where the page tells a Worker apart from "no WebGPU here": the error code the page reacts to by running on its own thread. */
export const NO_WEBGPU_HERE = 'no-webgpu-here';

export function createHandler(post: (r: WorkerReply) => void) {
  let detector: Promise<{ det: RfDetr; warnings: string[] }> | null = null;
  let embedder: Promise<Dino> | null = null;
  const jobs = new Map<number, AbortController>();
  const decoder = new FrameDecoder();
  /** A run keeps `decoder` busy for minutes (its jobs go one at a time); a tap picture is one short decode, so it has its own and does not wait for the run. */
  const boxDecoder = new FrameDecoder();

  /** The models run one picture at a time: ONNX Runtime refuses a second run on a busy session, and on WebGPU the detector and DINOv2 sessions share one device
   *  (a run's embedding beside a tap picture's detection fails with a buffer download error). A run and a tap picture overlap, so they take turns. */
  let turn: Promise<unknown> = Promise.resolve();
  const oneAtATime = <T>(fn: () => Promise<T>): Promise<T> => {
    const r = turn.then(fn);
    turn = r.catch(() => undefined);
    return r;
  };

  function load(m: Common) {
    return (detector ??= (async () => {
      const store = new ModelStore(m.base);
      store.adopt(m.manifest);
      const warnings: string[] = [];
      const [model, wasm] = await Promise.all([store.bytes(DETECTOR_FILE), store.bytes(WASM_FILE)]);
      const session = await createSession(model, wasm, { onWarn: (w) => warnings.push(w) });
      return { det: new RfDetr(session), warnings };
    })().catch((e) => {
      detector = null; // the next job tries again
      throw e;
    }));
  }

  /** DINOv2-S, for the targets that recognise a person by look (tap); a second session beside the detector's. */
  function loadEmbedder(m: Common) {
    return (embedder ??= (async () => {
      const store = new ModelStore(m.base);
      store.adopt(m.manifest);
      const [model, wasm] = await Promise.all([store.bytes(EMBEDDER_FILE), store.bytes(WASM_FILE)]);
      return new Dino(await createSession(model, wasm));
    })().catch((e) => {
      embedder = null;
      throw e;
    }));
  }

  async function run(m: RunMsg, signal: AbortSignal): Promise<RunResult> {
    const video: OpenVideo = { demuxed: m.demuxed, src: fileSource(m.file) };
    const target = targetFor(m.target, { zone: m.zone, zoneOverlap: m.zoneOverlap, minBox: m.minBox, taps: m.taps });
    const { det, warnings } = await load(m);
    const dino = target.needsEmbeddings ? await loadEmbedder(m) : null;
    const renderer = new FrameRenderer(video, undefined, 640, dino !== null);
    post({ id: m.id, type: 'progress', fraction: 0, message: 'models ready' });
    const stats: EmbedStats = { capped: 0 };
    const taps = m.taps ?? [];
    const embed = dino
      ? async (dets: Detection[], t: number) => {
          const wanted = taps.filter((p) => Math.abs(p.t - t) <= TAP_WINDOW_S + 1e-6).map((p) => p.box);
          await oneAtATime(() => embedDetections(dino, renderer.pixels(), dets, { wanted, stats }));
        }
      : undefined;
    const scored = await scoreVideo(
      { video, decoder, prepare: renderer.prepare, detect: (t) => oneAtATime(() => det.detect(t, target.cls, MIN_PERSON_SCORE)), embed, target },
      m.fps, m.demuxed.endTime, signal, (fraction, message) => post({ id: m.id, type: 'progress', fraction, message }),
    );
    const out = [...warnings];
    if (stats.capped) out.push(`${stats.capped} people in crowded pictures were not compared by look (only the ${MAX_EMBED} best in a picture are, besides the tapped one), so the followed person may be missed in those pictures`);
    return { scored, provider: det.session.provider, warnings: out, cls: target.cls, targetName: target.name };
  }

  async function boxes(m: BoxesMsg, signal: AbortSignal): Promise<BoxesResult> {
    const video: OpenVideo = { demuxed: m.demuxed, src: fileSource(m.file) };
    const cls = classId(m.cls);
    const { det } = await load(m);
    const plan = decodePlan(video.demuxed.samples, m.t);
    const frames = await boxDecoder.decode(video, [[plan.first, plan.last]], new Set([plan.target]), { signal });
    const f = frames.get(plan.target);
    if (!f) throw new EngineError(500, 'The picture at that time could not be decoded.');
    try {
      const tensor = await new FrameRenderer(video).prepare(f);
      return { t: video.demuxed.samples.pts[plan.target], detections: await oneAtATime(() => det.detect(tensor, cls, MIN_PERSON_SCORE)) };
    } finally {
      f.close();
    }
  }

  return function handle(msg: WorkerRequest): void {
    if (msg.op === 'cancel') {
      jobs.get(msg.job)?.abort();
      return;
    }
    if (msg.op === 'probe') {
      post({ id: msg.id, type: 'done', result: { webgpu: typeof navigator !== 'undefined' && 'gpu' in navigator, inWorker } });
      return;
    }
    const ac = new AbortController();
    jobs.set(msg.id, ac);
    const work =
      inWorker && !(typeof navigator !== 'undefined' && 'gpu' in navigator)
        ? Promise.reject(Object.assign(new EngineError(501, 'This browser gives Workers no WebGPU.'), { code: NO_WEBGPU_HERE }))
        : msg.op === 'run' ? run(msg, ac.signal) : boxes(msg, ac.signal);
    work.then(
      (result) => post({ id: msg.id, type: 'done', result }),
      (e) => post({ id: msg.id, type: 'error', status: e instanceof EngineError ? e.status : 500, message: e instanceof Error ? e.message : String(e), code: (e as { code?: string }).code }),
    ).finally(() => jobs.delete(msg.id));
  };
}

if (inWorker) {
  const handle = createHandler((r) => (self as unknown as { postMessage(m: unknown): void }).postMessage(r));
  self.onmessage = (e: MessageEvent<WorkerRequest>) => handle(e.data);
}
