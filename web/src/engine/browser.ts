// BrowserEngine: the in-browser engine of T9b (WebCodecs, mp4box.js, onnxruntime-web). Part 2 gives it media (open a File, play it, frames
// and the filmstrip) and the runs and edits kept in IndexedDB; detection, tapping and export arrive in parts 3 to 5 and say 501 until then.
import { demux as demuxFile, fileSource, type Demuxed } from '../browser/demux';
import { FrameDecoder, frameJpeg, uprightOf, type OpenVideo } from '../browser/frames';
import { BrowserStore, detectorDoc, openBackend, digest } from '../browser/store';
import { FilmstripBuilder } from '../browser/thumbs';
import { BlobUrlCache, makeKey, parseKey } from '../browser/urls';
import { EMBEDDER_FILE, ModelStore, type Manifest } from '../browser/models';
import { buildRunDoc, resolveSettings } from '../browser/pipeline';
import { chooseSink, exportStream, pageSinkEnv, planExport, stemOf, zipName, type SinkEnv, type ZipSink } from '../browser/exportzip';
import { cutRefusal, readMuxSource, type MuxSource } from '../browser/mux';
import { sha256Of } from '../browser/sha256';
import { CLASS_NAMES, classId, Zone as ZoneRect, type TapPoint } from '../browser/targets';
import { createHandler, NO_WEBGPU_HERE, type BoxesResult, type RunResult, type WorkerReply, type WorkerRequest } from '../browser/worker';
import {
  EngineError, type Box, type BrowserPreflight, type Caps, type EditClip, type EditsDoc, type Engine, type ExportRequest, type ExportResult, type Filmstrip, type FrameBoxes, type Job, type JobRef,
  type MediaInfo, type MediaTask, type ModelControl, type PickState, type RecentEntry, type ResegmentParams, type ResegmentResult, type RunData, type RunRequest, type RunSummary,
  type Samples, type Session, type Tap, type Zone,
} from './types';

/** Browser caps (docs/T9B_DESIGN.md section 1): everything native is off. Output goes to a file or a granted folder where the browser has
 *  showSaveFilePicker or showDirectoryPicker (Chrome), else downloads. Clips are cut by stream copy into a zip; the reel and edit lists are not in T9b. */
export function browserCaps(env: object = globalThis): Caps {
  return {
    openDialog: false, photosPicker: false, fileBrowser: false, locate: false, reveal: false, music: false, beatSync: false,
    savedTargets: false, people: false, proxy: false, output: 'showDirectoryPicker' in env ? 'granted-folder' : 'download',
    streamCopyCut: true, chooseOutputFolder: false, savePhotos: false, reel: false, editLists: false,
  };
}

const nope = (): never => {
  throw new EngineError(501, 'browser mode is T9b');
};
const later = (): never => {
  throw new EngineError(501, 'This comes in a later part of browser mode. Use My Mac for it.');
};

// ---------------------------------------------------------------- the Worker, or this thread
export type Without<T, K extends string> = T extends unknown ? Omit<T, K> : never;
type Job_ = Without<WorkerRequest, 'id'>;

/** Sends one job to wherever the models run and brings back the answer; `signal` cancels it. */
export interface Runner {
  send<R>(msg: Job_, onProgress?: (fraction: number, message: string) => void, signal?: AbortSignal): Promise<R>;
  dispose(): void;
}

const GRACE_MS = 5000;

/** A module Worker that holds the detector session between jobs. A browser that gives Workers no WebGPU makes the same handler run on
 *  this thread instead. A Worker that does not stop within GRACE_MS of a cancel is terminated (its session is made again next time). */
export class WorkerRunner implements Runner {
  private worker: Worker | null = null;
  private inline: ((m: WorkerRequest) => void) | null = null;
  private seq = 0;
  private pending = new Map<number, { ok: (r: unknown) => void; no: (e: unknown) => void; progress?: (f: number, m: string) => void; msg: WorkerRequest; cancelled: boolean; started: boolean }>();
  /** cancelled jobs given up on (answered 499) that the Worker may still be running: they hold its turn until they stop or the Worker is ended */
  private stuck = new Set<number>();

  constructor(private make: () => Worker = () => new Worker(new URL('../browser/worker.ts', import.meta.url), { type: 'module' })) {}

  private deliver = (r: WorkerReply): void => {
    const p = this.pending.get(r.id);
    if (!p) {
      if (r.type !== 'progress') this.stuck.delete(r.id); // the stopped job ended by itself: it holds no turn any more
      return;
    }
    if (r.type === 'progress') {
      p.started = true;
      return p.progress?.(r.fraction, r.message);
    }
    this.pending.delete(r.id);
    if (r.type === 'done') p.ok(r.result);
    else p.no(Object.assign(new EngineError(r.status, r.message), { code: r.code }));
    this.reconsider();
  };

  private target(): (m: WorkerRequest) => void {
    if (this.inline) return this.inline;
    if (!this.worker) {
      const w = this.make();
      w.onmessage = (e: MessageEvent<WorkerReply>) => {
        if (this.worker === w) this.deliver(e.data); // an ended Worker says nothing more
      };
      w.onerror = (e) => this.crash(new EngineError(500, `The worker stopped: ${e.message || 'unknown error'}`));
      this.worker = w;
    }
    const w = this.worker;
    return (m) => w.postMessage(m);
  }

  private crash(e: unknown): void {
    this.worker?.terminate();
    this.worker = null;
    this.stuck.clear();
    for (const p of this.pending.values()) p.no(e);
    this.pending.clear();
  }

  /** A cancelled job did not stop in time: answer for that job alone, and leave it to `reconsider` whether its Worker has to be ended. */
  private restartFor(id: number): void {
    const gone = this.pending.get(id);
    if (!gone) return;
    this.pending.delete(id);
    gone.no(new EngineError(499, 'Cancelled.'));
    if (this.inline) return; // nothing to terminate on this thread
    this.stuck.add(id);
    this.reconsider();
  }

  /** The stopped job may still hold the Worker's turn. A live job that has shown progress is running, so it keeps its Worker (sent again, it would lose
   *  its progress and time); one that has shown none may be queued behind the stopped job, and would wait forever. So whenever a job settles or is
   *  given up on: with a stopped job in the Worker and no live job that has started, the Worker is ended and the live jobs go to a fresh one. */
  private reconsider(): void {
    if (this.inline || !this.stuck.size || !this.worker) return;
    const live = [...this.pending.values()].filter((p) => !p.cancelled);
    if (live.some((p) => p.started)) return;
    this.worker.terminate();
    this.worker = null;
    this.stuck.clear();
    for (const [k, p] of [...this.pending]) {
      if (p.cancelled) {
        this.pending.delete(k);
        p.no(new EngineError(499, 'Cancelled.'));
      }
    }
    if (!this.pending.size) return;
    const post = this.target();
    for (const p of this.pending.values()) post(p.msg);
  }

  async send<R>(msg: Job_, onProgress?: (fraction: number, message: string) => void, signal?: AbortSignal): Promise<R> {
    if (signal?.aborted) throw new EngineError(499, 'Cancelled.');
    const id = ++this.seq;
    const post = this.target();
    const req = { ...msg, id } as WorkerRequest;
    const done = new Promise<unknown>((ok, no) => this.pending.set(id, { ok, no, progress: onProgress, msg: req, cancelled: false, started: false }));
    let timer: ReturnType<typeof setTimeout> | undefined;
    const onAbort = () => {
      const p = this.pending.get(id);
      if (p) p.cancelled = true;
      this.target()({ op: 'cancel', id: 0, job: id });
      timer = setTimeout(() => this.restartFor(id), GRACE_MS);
    };
    signal?.addEventListener('abort', onAbort, { once: true });
    post(req);
    try {
      return (await done) as R;
    } catch (e) {
      if ((e as { code?: string }).code === NO_WEBGPU_HERE) {
        // Workers here have no WebGPU: run the same handler on this thread (slower for the page, but on the GPU). Every job that failed
        // the same way gets here and finds it done by the first; the jobs still waiting on the ended Worker go to this thread too.
        if (!this.inline) {
          this.worker?.terminate();
          this.worker = null;
          this.stuck.clear();
          const inline = (this.inline = createHandler(this.deliver));
          for (const p of this.pending.values()) inline(p.msg);
        }
        return this.send<R>(msg, onProgress, signal);
      }
      throw e;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    }
  }

  dispose(): void {
    this.crash(new EngineError(499, 'Cancelled.'));
  }
}

/** serve_runs._slug_part and the slug of workflow.slugOf (the engine cannot import the UI layer). */
const slugPart = (s: string): string => s.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48) || 'x';
const slugOf = (t: RunRequest['target']): string => (t.kind === 'any' ? 'any-person' : t.kind === 'class' ? `class-${slugPart(t.class ?? '')}` : t.kind);

/** What this browser can do, asked of the browser itself. */
export async function browserEnv(): Promise<Pick<BrowserPreflight, 'webgpu' | 'webcodecs' | 'hevc'>> {
  let webgpu = false;
  try {
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    webgpu = !!gpu && !!(await gpu.requestAdapter());
  } catch { /* no adapter */ }
  const webcodecs = typeof VideoDecoder !== 'undefined' && typeof EncodedVideoChunk !== 'undefined';
  let hevc: boolean | null = null;
  if (webcodecs) {
    hevc = await VideoDecoder.isConfigSupported({ codec: 'hvc1.1.6.L93.B0', codedWidth: 1920, codedHeight: 1080 }).then((r) => !!r.supported, () => false);
  }
  return { webgpu, webcodecs, hevc };
}

/** The filmstrip as the engine sees it (FilmstripBuilder, or a stand-in in the tests). */
export interface FilmLike {
  start(): void;
  filmstrip(): Filmstrip;
  sheet(n: number): Promise<Blob | null>;
  /** stop making sheets (the video was closed or replaced) */
  cancel?(): void;
  readonly state: string;
  readonly error: string | null;
}

/** The parts a test replaces: storage, the demux, the picture renderer, the filmstrip, the blob URL cache and how a File gets its URL. */
export interface BrowserDeps {
  store?: BrowserStore | Promise<BrowserStore>;
  demux?: (file: File) => Promise<Demuxed>;
  renderFrame?: (v: OpenVideo, t: number, long: number, signal?: AbortSignal) => Promise<Blob>;
  decoder?: Pick<FrameDecoder, 'decode'>;
  makeFilmstrip?: (v: OpenVideo) => FilmLike;
  urls?: BlobUrlCache;
  fileUrl?: (file: Blob) => string;
  revokeFileUrl?: (url: string) => void;
  /** where the models are kept and fetched from */
  models?: ModelStore;
  /** runs jobs (the Worker); tests give a stand-in */
  runner?: Runner;
  /** WebGPU, WebCodecs and HEVC support of this browser */
  env?: () => Promise<Pick<BrowserPreflight, 'webgpu' | 'webcodecs' | 'hevc'>>;
  /** the file and folder pickers of the page (export); tests give stand-ins */
  sink?: () => SinkEnv;
}

interface JobRec {
  job: Job;
  ac: AbortController;
  /** the run is over (done, failed or cancelled) and its result is stored */
  settled: boolean;
}

interface Opened {
  file: File;
  video: OpenVideo;
  info: MediaInfo;
  url: string;
  /** the same bytes typed video/mp4, made on demand: a browser that refuses video/quicktime may play an H.264 .mov as mp4 */
  mp4Url?: string;
  film?: FilmLike;
  /** what the export reads back from the file (sample tables, sample entries), made on first use */
  mux?: Promise<MuxSource>;
  /** SHA-256 of the file, made on first use (the sidecars name the source by it) */
  hash?: string;
}

const MIN_BOX_SCORE = 0.4; // pick.py shows boxes from this score up
/** The detector and the runtime are cached (ModelStatus.detectorReady) and so is the DINOv2 file that tap targets need. */
const embedderCached = (m: { detectorReady: boolean; files: { name: string; cached: boolean }[] }): boolean => m.detectorReady && m.files.some((f) => f.name === EMBEDDER_FILE && f.cached);
const num = (v: unknown, what: string): number => {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new EngineError(400, `${what} must be a finite number`);
  return v;
};
const r4 = (v: number): number => Math.round(v * 10000) / 10000;
/** pick._round_zone: x, y, w, h to 4 decimals without the rounded zone leaving the frame. */
const roundZone = (z: ZoneRect): Zone => {
  const x = r4(z.x);
  const y = r4(z.y);
  return [x, y, r4(Math.min(r4(z.x + z.w), 1) - x), r4(Math.min(r4(z.y + z.h), 1) - y)];
};
const emptyPick = (cls: string): PickState => ({ target_key: '', taps: [], zone: null, file: null, class: cls, candidates: [] });
const NOT_OPEN = 'This video is not open in this tab. Choose the file again; its runs come back.';

export class BrowserEngine implements Engine {
  readonly kind = 'browser' as const;
  readonly caps: Caps = browserCaps();
  private opened = new Map<string, Opened>();
  /** The video the UI has open: the last one opened or reopened. Clear recent keeps only this one, not every file this tab ever held. */
  private current: string | null = null;
  private storeP: Promise<BrowserStore> | null = null;
  private decoder: Pick<FrameDecoder, 'decode'> | null = null;
  private urls: BlobUrlCache | null;
  private modelStore: ModelStore | null;
  private runnerInst: Runner | null;
  private jobs = new Map<string, JobRec>();
  private jobSeq = 0;
  private lastExport = new Map<string, { result: ExportResult; zipUrl: string }>();
  /** the downloadable models, for the checklist's buttons */
  readonly models: ModelControl;

  constructor(private deps: BrowserDeps = {}) {
    this.urls = deps.urls ?? null;
    this.modelStore = deps.models ?? null;
    this.runnerInst = deps.runner ?? null;
    this.models = {
      status: async () => this.modelsOf().status(),
      download: (o) => this.modelsOf().download(o),
      remove: () => this.modelsOf().remove(),
    };
  }

  private modelsOf(): ModelStore {
    return (this.modelStore ??= new ModelStore());
  }
  private runner(): Runner {
    return (this.runnerInst ??= new WorkerRunner());
  }

  private store(): Promise<BrowserStore> {
    if (!this.storeP) this.storeP = Promise.resolve(this.deps.store ?? openBackend().then((b) => new BrowserStore(b.backend, b.persistent)));
    return this.storeP;
  }
  private cache(): BlobUrlCache {
    return (this.urls ??= new BlobUrlCache());
  }
  private frames(): Pick<FrameDecoder, 'decode'> {
    return (this.decoder ??= this.deps.decoder ?? new FrameDecoder());
  }
  private need(id: string): Opened {
    const o = this.opened.get(id);
    if (!o) throw new EngineError(409, NOT_OPEN);
    return o;
  }

  // The shell needs a session to draw the open screen.
  async session(): Promise<Session> {
    const env = await (this.deps.env ?? browserEnv)();
    const m = await this.modelsOf().status();
    const models: BrowserPreflight['models'] = {
      available: m.available, error: m.error, detectorReady: m.detectorReady, embedderReady: embedderCached(m), allReady: m.allReady, totalBytes: m.totalBytes, cachedBytes: m.cachedBytes, host: m.host, filesLeft: m.filesLeft,
    };
    return {
      version: '', engine: 'browser', platform: 'browser', caps: this.caps, presets: ['fast', 'balanced'], default_preset: 'balanced',
      classes: CLASS_NAMES, defaults: {},
      preflight: { browser: { ...env, models }, ffmpeg: false, models: { fast: m.detectorReady, balanced: m.detectorReady }, identity: false, tracking: false, music: false },
      job: await this.currentJob(), media: [], start_media: null,
    };
  }

  // ---------------------------------------------------------------- media
  /** Read the file's movie header (a few 8 MB slices at most), remember its facts, and keep the File for playback and frames. */
  async openFile(file: File): Promise<MediaInfo> {
    const d = await (this.deps.demux ? this.deps.demux(file) : demuxFile(fileSource(file)));
    const id = (await digest(`${file.name}|${file.size}|${file.lastModified}`)).slice(0, 12);
    const prev = this.opened.get(id);
    const same = prev?.file === file;
    const url = same ? prev!.url : this.makeUrl(file);
    if (prev && !same) {
      // a new File object for the same video: the old one's filmstrip must stop, or it keeps the decoder busy
      prev.film?.cancel?.();
      this.revoke(prev.url);
      if (prev.mp4Url) this.revoke(prev.mp4Url);
    }
    const u = uprightOf(d.video.width, d.video.height, d.rotation, d.mirrored, d.sar, 1e9);
    const info: MediaInfo = {
      id, path: file.name, name: file.name, size: file.size, mtime: file.lastModified / 1000, codec: d.codec, profile: d.profile, pix_fmt: null,
      width: d.video.width, height: d.video.height, display_w: Math.round(u.w), display_h: Math.round(u.h), rotation: d.rotation, mirrored: d.mirrored,
      fps: d.fps, capture_fps: d.captureFps, slowmo_suggested: d.captureFps >= 100, duration: d.duration, offset: d.offset, end_time: d.endTime,
      has_audio: d.hasAudio, keyframes: d.keyframes.map((k) => Math.round(k * 1000) / 1000), video_url: url,
      proxy: { state: 'none', fraction: 0, url: null, first_frame_file_time: null, error: null }, filmstrip_state: 'none', picked_copy: false,
    };
    this.current = id;
    this.opened.set(id, { file, video: { demuxed: d, src: fileSource(file) }, info, url, film: same ? prev!.film : undefined, mp4Url: same ? prev!.mp4Url : undefined });
    const { video_url: _v, proxy: _p, filmstrip_state: _f, picked_copy: _c, ...facts } = info;
    await (await this.store()).putMedia({ ...facts, opened: Date.now() });
    return info;
  }

  /** The open screen's recent list holds media ids. A file can only be reopened while this tab still has it. */
  async openPath(id: string): Promise<MediaInfo> {
    if (this.opened.has(id)) {
      this.current = id;
      return this.media(id);
    }
    throw new EngineError(404, 'A browser cannot reopen a file by itself. Choose the file again; its runs come back.');
  }
  locate = async () => null;
  recent = async (): Promise<RecentEntry[]> => (await this.store()).recent();
  clearRecent = async (): Promise<RecentEntry[]> => {
    await (await this.store()).clearRecent(this.current && this.opened.has(this.current) ? [this.current] : []);
    return [];
  };

  async media(id: string): Promise<MediaInfo> {
    const o = this.opened.get(id);
    if (o) return { ...o.info, filmstrip_state: o.film?.state ?? 'none' };
    const m = await (await this.store()).getMedia(id);
    if (!m) throw new EngineError(404, `No such video: ${id}`);
    const { opened: _o, ...facts } = m;
    return { ...facts, video_url: '', proxy: { state: 'none', fraction: 0, url: null, first_frame_file_time: null, error: null }, filmstrip_state: 'none', picked_copy: false };
  }
  private makeUrl = (b: Blob): string => (this.deps.fileUrl ?? ((x: Blob) => URL.createObjectURL(x)))(b);
  private revoke = (u: string): void => (this.deps.revokeFileUrl ?? ((x: string) => URL.revokeObjectURL(x)))(u);

  /** The blob URL of the file. `asMp4` is the second try for a .mov (there is no server here to remux): the same bytes, typed video/mp4. */
  videoUrl = (id: string, opts: { asMp4?: boolean } = {}): string => {
    const o = this.opened.get(id);
    if (!o) return '';
    if (!opts.asMp4) return o.url;
    return (o.mp4Url ??= this.makeUrl(o.file.slice(0, o.file.size, 'video/mp4')));
  };

  /** The page is done with this engine: stop every filmstrip, drop the rendered pictures and free the file URLs. */
  close(): void {
    for (const o of this.opened.values()) {
      o.film?.cancel?.();
      this.revoke(o.url);
      if (o.mp4Url) this.revoke(o.mp4Url);
    }
    this.opened.clear();
    this.current = null;
    this.urls?.clear();
    for (const x of this.lastExport.values()) if (x.zipUrl) this.revoke(x.zipUrl);
    this.lastExport.clear();
  }
  makeProxy = async (): Promise<MediaTask> => {
    throw new EngineError(501, 'This browser cannot make a preview copy.');
  };
  async mediaTask(id: string, kind: 'proxy' | 'filmstrip'): Promise<MediaTask> {
    if (kind === 'proxy') return { state: 'none', fraction: 0, url: null, first_frame_file_time: null, error: null };
    const f = this.opened.get(id)?.film;
    if (!f) return { state: 'none', fraction: 0 };
    const fs = f.filmstrip();
    return { state: f.state === 'running' ? 'running' : f.state === 'done' ? 'done' : 'failed', fraction: fs.sheets ? Math.min(1, new Set(fs.cells.map((c) => c.sheet)).size / fs.sheets) : 1, error: f.error };
  }

  /** The filmstrip so far; the first call starts making it. */
  async filmstrip(id: string): Promise<Filmstrip> {
    const o = this.need(id);
    if (!o.film) {
      o.film = this.deps.makeFilmstrip ? this.deps.makeFilmstrip(o.video) : new FilmstripBuilder(o.video, this.frames());
      o.film.start();
    }
    return o.film.filmstrip();
  }
  sheetUrl = (id: string, n: number): string => makeKey('sheet', id, n);
  frameUrl = (id: string, t: number, width = 640): string => makeKey('frame', id, t.toFixed(3), width);
  // the picture the boxes will be drawn over is the picture at t
  pickFrameUrl = (id: string, t: number, _cls: string): string => this.frameUrl(id, t, 640);

  /** A key from sheetUrl / frameUrl as a blob URL (rendered once, then cached); any other URL comes back unchanged. */
  async resolveUrl(url: string, signal?: AbortSignal): Promise<string> {
    const k = parseKey(url);
    if (!k) return url;
    const id = k.parts[0];
    if (k.kind === 'sheet') {
      const n = Number(k.parts[1]);
      return this.cache().get(url, async () => {
        const o = this.need(id);
        if (!o.film) await this.filmstrip(id);
        const b = await o.film!.sheet(n);
        if (!b) throw new EngineError(500, o.film!.error ?? 'The filmstrip could not be made.');
        return b;
      }, signal);
    }
    if (k.kind === 'frame') {
      const t = Number(k.parts[1]);
      const w = Number(k.parts[2]);
      return this.cache().get(url, (sig) => {
        const v = this.need(id).video;
        return this.deps.renderFrame ? this.deps.renderFrame(v, t, w, sig) : frameJpeg(this.frames(), v, t, w, sig);
      }, signal);
    }
    throw new EngineError(501, 'browser mode is T9b');
  }

  // ---------------------------------------------------------------- runs and edits (IndexedDB)
  runs = async (mediaId: string): Promise<RunSummary[]> => (await this.store()).runs(mediaId);
  run = async (runId: string): Promise<RunData> => (await this.store()).run(runId);
  samples = async (runId: string): Promise<Samples> => (await this.store()).samples(runId);
  saveEdits = async (runId: string, revision: number, clips: EditClip[]): Promise<EditsDoc> => (await this.store()).saveEdits(runId, revision, clips);
  revertEdits = async (runId: string): Promise<void> => (await this.store()).revertEdits(runId);
  resegment = async (runId: string, p: ResegmentParams): Promise<ResegmentResult> => (await this.store()).resegment(runId, p);

  // ---------------------------------------------------------------- boxes and runs (part 3)
  /** The people (or the class) the detector sees in the picture at t, from score 0.4 up as `inframe pick` shows them. */
  async boxes(id: string, t: number, cls: string, signal?: AbortSignal): Promise<FrameBoxes> {
    const o = this.need(id);
    const name = cls || 'person';
    classId(name);
    const manifest = await this.readyModels();
    const r = await this.runner().send<BoxesResult>({ op: 'boxes', file: o.file, demuxed: o.video.demuxed, t, cls: name, base: this.modelsOf().base, manifest }, undefined, signal);
    const shown = r.detections.filter((d) => d.score >= MIN_BOX_SCORE);
    return { t: Math.round(r.t * 1000) / 1000, boxes: shown.map((d, i) => ({ i: i + 1, box: d.box.map((v) => Math.round(v * 10000) / 10000) as Box, score: Math.round(d.score * 1000) / 1000 })) };
  }

  /** The manifest, once the detector and the runtime are in the cache. */
  private async readyModels(): Promise<Manifest> {
    const st = await this.modelsOf().status();
    if (!st.detectorReady) throw new EngineError(409, st.available ? 'The models are not downloaded yet. Use "Download models" first.' : (st.error ?? 'No models are available here.'), { error: 'models' });
    return this.modelsOf().manifest();
  }

  private snap(rec: JobRec): Job {
    const j = rec.job;
    return { ...j, elapsed_s: j.state === 'running' ? Date.now() / 1000 - j.started : j.elapsed_s };
  }
  private runningRec(): JobRec | null {
    for (const r of this.jobs.values()) if (!r.settled) return r;
    return null;
  }

  async startRun(req: RunRequest): Promise<JobRef> {
    const busy = this.runningRec();
    if (busy) throw new EngineError(409, 'busy', { error: 'busy', job: this.snap(busy) });
    const o = this.need(req.media_id);
    if (req.target.kind !== 'any' && req.target.kind !== 'class' && req.target.kind !== 'tap') later();
    const st = resolveSettings(req);
    const cls = req.target.kind === 'class' ? req.target.class ?? '' : req.target.kind === 'tap' ? req.target.class || 'person' : 'person';
    classId(cls); // a bad class name stops here, before anything is downloaded or decoded
    const slug = slugOf(req.target);
    const runId = `${o.info.id}-${slug}`;
    const started = Date.now() / 1000;
    // Reserve the slot before the first await: a second startRun while the models are checked must meet 409 busy, not start a second job.
    const rec: JobRec = {
      ac: new AbortController(), settled: false,
      job: { id: `browser-run-${++this.jobSeq}`, kind: 'run', state: 'running', fraction: 0, message: 'starting', started, elapsed_s: 0, media_id: o.info.id, run_id: runId, result: null, error: null },
    };
    this.jobs.set(rec.job.id, rec);
    let manifest: Manifest;
    let taps: TapPoint[] = [];
    try {
      manifest = await this.readyModels();
      if (req.target.kind === 'tap') taps = await this.tapsFor(o.info.id, cls);
    } catch (e) {
      this.jobs.delete(rec.job.id);
      throw e;
    }
    void this.runJob(rec, req, o, st, slug, manifest, taps);
    return { job: this.snap(rec), run_id: runId };
  }

  /** The taps of a tap run, once the recognition model is in the cache too. */
  private async tapsFor(mediaId: string, cls: string): Promise<TapPoint[]> {
    const taps = (await (await this.store()).getPick(mediaId, cls))?.taps ?? [];
    if (!taps.length) throw new EngineError(422, 'Tap the person in the video first.');
    const st = await this.modelsOf().status();
    if (!embedderCached(st)) throw new EngineError(409, 'The recognition model is not downloaded yet. Use "Download models" first.', { error: 'models' });
    return taps.map((t) => ({ t: t.time, box: t.box }));
  }

  private async runJob(rec: JobRec, req: RunRequest, o: Opened, st: ReturnType<typeof resolveSettings>, slug: string, manifest: Manifest, taps: TapPoint[]): Promise<void> {
    const end = (state: Job['state'], error: string | null = null) => {
      rec.job = { ...rec.job, state, error, fraction: state === 'done' ? 1 : rec.job.fraction, elapsed_s: Date.now() / 1000 - rec.job.started };
      rec.settled = true;
    };
    try {
      const r = await this.runner().send<RunResult>(
        {
          op: 'run', file: o.file, demuxed: o.video.demuxed, target: { kind: req.target.kind, class: req.target.class }, fps: st.fps, zone: req.zone,
          zoneOverlap: req.zone_overlap, minBox: typeof req.advanced?.min_box === 'number' ? req.advanced.min_box : 0, base: this.modelsOf().base, manifest,
          ...(req.target.kind === 'tap' ? { taps } : {}),
        },
        (fraction, message) => {
          rec.job = { ...rec.job, fraction, message, elapsed_s: Date.now() / 1000 - rec.job.started };
        },
        rec.ac.signal,
      );
      if (rec.ac.signal.aborted) return end('cancelled');
      const d = o.video.demuxed;
      const zone: Zone | null = req.zone;
      const doc = buildRunDoc(r.scored, st, { offset: d.offset, end_time: d.endTime, keyframes: d.keyframes }, {
        target: r.targetName, cls: r.cls, zone, zoneOverlap: req.zone_overlap, detector: `onnxruntime-web ${r.provider}`, model: 'small', warnings: r.warnings,
        totalS: Date.now() / 1000 - rec.job.started,
      });
      await (await this.store()).putRun({ run_id: rec.job.run_id!, media_id: o.info.id, slug, target: r.targetName, preset: st.preset, segments: doc.segments, samples: doc.samples, source: { offset: d.offset, end_time: d.endTime } });
      end('done');
    } catch (e) {
      if (rec.ac.signal.aborted || (e instanceof EngineError && e.status === 499)) end('cancelled');
      else end('failed', e instanceof Error ? e.message : String(e));
    }
  }

  async job(id: string): Promise<Job> {
    const rec = this.jobs.get(id);
    if (!rec) throw new EngineError(404, `No such job: ${id}`);
    return this.snap(rec);
  }
  async currentJob(): Promise<Job | null> {
    const r = this.runningRec();
    return r ? this.snap(r) : null;
  }
  async cancel(id: string): Promise<Job> {
    const rec = this.jobs.get(id);
    if (!rec) throw new EngineError(404, `No such job: ${id}`);
    if (!rec.settled) rec.ac.abort();
    return this.snap(rec);
  }


  // ---------------------------------------------------------------- export (part 5): the clips cut by stream copy, in one zip
  /** The clips an export would write: the run's edited clips (else the detector's), narrowed to `ids` when given. */
  private async exportClips(runId: string, ids: string[] | null) {
    const st = await this.store();
    const run = await st.getRun(runId);
    if (!run) throw new EngineError(404, `No such run: ${runId}`);
    const all = (run.edits ?? detectorDoc(run)).clips;
    let clips = all;
    if (ids) {
      const known = new Set(all.map((c) => c.id));
      const bad = ids.filter((i) => !known.has(i));
      if (bad.length) throw new EngineError(422, `No such clip: ${bad.slice(0, 3).join(', ')}`);
      clips = all.filter((c) => ids.includes(c.id));
    }
    if (!clips.length) throw new EngineError(422, 'Nothing to export: the clip list is empty.');
    return { run, clips: [...clips].sort((a, b) => a.start - b.start).map((c) => ({ id: c.id, start: c.start, end: c.end })) };
  }

  /** What the export reads back from the file, kept for the next export; a failed read is not kept, so a second try reads again. */
  private muxOf(o: Opened): Promise<MuxSource> {
    const p = (o.mux ??= readMuxSource(o.video.src, o.video.demuxed, o.file.name));
    p.catch(() => {
      if (o.mux === p) o.mux = undefined;
    });
    return p;
  }

  async startExport(runId: string, req: ExportRequest): Promise<JobRef> {
    const busy = this.runningRec();
    if (busy) throw new EngineError(409, 'busy', { error: 'busy', job: this.snap(busy) });
    if (req.reel || req.edit_lists?.length || req.music) throw new EngineError(501, 'The reel and the edit lists are not in browser mode. Use My Mac for them.');
    if (!req.clips) throw new EngineError(422, 'Nothing to export: choose Clips.');
    // The save picker needs the click that started this, and a click's activation lasts seconds: it is asked for once the request is checked
    // and before the slow part (reading the file's sample tables). Every refusal that needs no read of the store or the file comes first.
    const early = [...this.opened.values()].find((x) => runId.startsWith(`${x.info.id}-`));
    if (early) {
      const refusal = cutRefusal(early.video.demuxed.video.fourcc);
      if (refusal) throw refusal;
    }
    // Reserve the slot before the first await: a second click while the picker is open or the file is read must meet 409 busy, not start a second export.
    const rec: JobRec = {
      ac: new AbortController(), settled: false,
      job: { id: `browser-export-${++this.jobSeq}`, kind: 'export', state: 'running', fraction: 0, message: 'choosing where to save', started: Date.now() / 1000, elapsed_s: 0, media_id: early?.info.id ?? null, run_id: runId, result: null, error: null },
    };
    this.jobs.set(rec.job.id, rec);
    const begin = async () => {
      // Check the request against the run first (a read of the stored run, milliseconds), so a refusal comes before the user has picked a file:
      // a picked file that is then never written is left empty.
      const { run, clips } = await this.exportClips(runId, req.clip_ids);
      // a cancel can arrive at any await below (the job is cancellable from the moment its slot is reserved): stop, and give back what was made
      const stopIfCancelled = () => {
        if (rec.ac.signal.aborted) throw new EngineError(499, 'Cancelled.');
      };
      stopIfCancelled();
      const o = this.need(run.media_id);
      const refusal = cutRefusal(o.video.demuxed.video.fourcc);
      if (refusal) throw refusal;
      const env = (this.deps.sink ?? pageSinkEnv)();
      const prepare = async () => {
        const samples = run.samples ?? null;
        const mux = await this.muxOf(o);
        // a placeholder hash for the sizes; the real one is made in the job and goes into the sidecars
        const base = { media: o.info, settings: run.segments.settings, samples, clips, mux, src: o.video.src };
        const plan = planExport({ ...base, hash: '0'.repeat(64) });
        if (!plan.clips.length) throw new EngineError(422, plan.warnings.find((w) => /could not be cut/.test(w)) ?? 'None of the clips could be cut.');
        return { base, plan };
      };
      // The picker is asked first (its call runs synchronously up to the dialog), then the file is read. The zip's name comes from the video;
      // the size it will be is given later (only the in-memory download needs it).
      let giveBytes!: (n: number) => void;
      let failBytes!: (e: unknown) => void;
      const bytes = new Promise<number>((res, rej) => { giveBytes = res; failBytes = rej; });
      bytes.catch(() => undefined); // `work` is awaited below; this copy of its failure is not
      const picked = chooseSink(env, zipName(stemOf(o.info.name)), bytes);
      picked.catch(() => undefined); // if the work fails first, the picker's own outcome is not an unhandled rejection
      const work = prepare();
      work.then((w) => giveBytes(w.plan.bytes), failBytes);
      // a refusal found while the picker is open (what only the file can tell) leaves the empty file the picker made: give it back
      work.catch(() => picked.then((s) => s.discard?.(), () => undefined));
      const [{ base, plan }, sink] = await Promise.all([work, picked]);
      try {
        stopIfCancelled();
      } catch (e) {
        await sink.discard?.();
        throw e;
      }
      return { run, o, base, plan, sink };
    };
    let ready: Awaited<ReturnType<typeof begin>>;
    try {
      ready = await begin();
    } catch (e) {
      // keep the record, finished: a second click that met 409 busy holds this job's id and follows it, and must see a clean end, not a 404
      const cancelled = rec.ac.signal.aborted || (e instanceof EngineError && e.status === 499);
      rec.job = { ...rec.job, state: cancelled ? 'cancelled' : 'failed', error: cancelled ? null : e instanceof Error ? e.message : String(e), elapsed_s: Date.now() / 1000 - rec.job.started };
      rec.settled = true;
      throw e;
    }
    const { run, o, base, plan, sink } = ready;
    rec.job = { ...rec.job, message: 'starting', media_id: o.info.id };
    void this.exportJob(rec, o, runId, run.edits?.revision ?? 0, base, plan, sink);
    return { job: this.snap(rec), run_id: runId };
  }

  private async exportJob(rec: JobRec, o: Opened, runId: string, revision: number, base: Omit<Parameters<typeof planExport>[0], 'hash'>, plan: ReturnType<typeof planExport>, sink: ZipSink): Promise<void> {
    const tick = (fraction: number, message: string) => {
      rec.job = { ...rec.job, fraction, message, elapsed_s: Date.now() / 1000 - rec.job.started };
    };
    try {
      if (o.hash === undefined) {
        tick(0, 'fingerprinting the video for the sidecars');
        o.hash = await sha256Of(o.video.src.size, (a, b) => o.video.src.read(a, b), { onProgress: (f) => tick(f * 0.15, 'fingerprinting the video for the sidecars'), signal: rec.ac.signal });
      }
      const input = { ...base, hash: o.hash };
      const clipBytes = Math.max(1, plan.clips.reduce((s, c) => s + c.muxed.size, 0));
      let done = 0;
      tick(0.15, `cutting ${plan.clips.length} ${plan.clips.length === 1 ? 'clip' : 'clips'}`);
      const stream = exportStream(plan, input, {
        signal: rec.ac.signal,
        onBytes: (n) => {
          done += n;
          tick(0.15 + 0.84 * Math.min(1, done / clipBytes), `writing ${plan.zip}`);
        },
      });
      const written = await sink.write(stream, rec.ac.signal);
      if (rec.ac.signal.aborted) throw new EngineError(499, 'Cancelled.');
      const zipUrl = written.url ?? '';
      const prev = this.lastExport.get(runId);
      if (prev?.zipUrl) this.revoke(prev.zipUrl);
      const outputs: ExportResult['outputs'] = [{ name: plan.zip, kind: 'zip', size: written.size, url: zipUrl }];
      for (const c of plan.clips) {
        outputs.push({ name: c.name, kind: 'clip', size: c.muxed.size, url: '' });
        outputs.push({ name: c.sidecar, kind: 'sidecar', size: c.sidecarBytes, url: '' });
      }
      outputs.push({ name: plan.csvName, kind: 'csv', size: plan.csv.length, url: '' });
      const failed = plan.failed.map((f) => ({ index: f.index, start: f.start, end: f.end, file: f.name, error: f.error }));
      const result: ExportResult = {
        exported: new Date().toISOString(), revision, dir: sink.kind === 'download' ? 'your downloads' : plan.zip, n_clips: plan.clips.length + plan.failed.length,
        clips: [...plan.clips.map((c) => ({ index: c.index, start: c.start, end: c.end, file: c.name, sidecar: c.sidecar })), ...failed].sort((a, b) => a.index - b.index),
        reel: null, edit_lists: [], warnings: plan.warnings, outputs,
      };
      this.lastExport.set(runId, { result, zipUrl });
      rec.job = { ...rec.job, result, state: 'done', fraction: 1, message: 'done', elapsed_s: Date.now() / 1000 - rec.job.started };
    } catch (e) {
      // the picker made the file when the user picked it: a failed or cancelled export gives it back, not an empty or partial zip
      try {
        await sink.discard?.();
      } catch { /* best effort */ }
      const cancelled = rec.ac.signal.aborted || (e instanceof EngineError && e.status === 499) || (e instanceof DOMException && e.name === 'AbortError');
      rec.job = { ...rec.job, state: cancelled ? 'cancelled' : 'failed', error: cancelled ? null : e instanceof Error ? e.message : String(e), elapsed_s: Date.now() / 1000 - rec.job.started };
    } finally {
      rec.settled = true;
    }
  }

  exportResult = async (runId: string): Promise<ExportResult | null> => this.lastExport.get(runId)?.result ?? null;
  /** The zip as a blob URL, while this tab holds one (a browser that downloads it); '' otherwise. */
  runFileUrl = (runId: string, name: string): string => {
    const x = this.lastExport.get(runId);
    return x && x.result.outputs.some((o) => o.name === name && o.url) ? x.result.outputs.find((o) => o.name === name)!.url : '';
  };

  // ---------------------------------------------------------------- picking (part 4): taps and the zone, kept in IndexedDB per video and class
  async pickState(id: string, cls: string): Promise<PickState> {
    const name = cls || 'person';
    classId(name);
    return (await (await this.store()).getPick(id, name)) ?? emptyPick(name);
  }

  /** Remembers the tap: the box the user clicked in the picture at t. The detection it lands on is found when the run reads the pictures around t. */
  async tap(id: string, cls: string, t: number, box: Box, index: number): Promise<PickState> {
    const name = cls || 'person';
    classId(name);
    num(t, 't');
    if (!Array.isArray(box) || box.length !== 4) throw new EngineError(400, 'box must be [x0, y0, x1, y1]');
    const [x0, y0, x1, y1] = box.map((v) => num(v, 'box'));
    if (!(x0 >= 0 && x0 < x1 && x1 <= 1 && y0 >= 0 && y0 < y1 && y1 <= 1)) throw new EngineError(400, 'box must be fractions of the frame with x0 < x1 and y0 < y1');
    if (index !== null && index !== undefined && !Number.isInteger(index)) throw new EngineError(400, 'index must be an integer or null');
    const info = (await this.media(id)) as MediaInfo;
    if (!(info.offset - 0.01 <= t && t <= info.end_time + 0.01)) throw new EngineError(400, `t is outside the video (${info.offset.toFixed(3)} to ${info.end_time.toFixed(3)} s)`);
    const s = await this.pickState(id, name);
    const tap: Tap = { time: Math.round(t * 1000) / 1000, box: [r4(x0), r4(y0), r4(x1), r4(y1)], track_id: null, index };
    return this.putPick(id, { ...s, taps: [...s.taps, tap] });
  }

  async undoTap(id: string, cls: string): Promise<PickState> {
    const s = await this.pickState(id, cls);
    return s.taps.length ? this.putPick(id, { ...s, taps: s.taps.slice(0, -1) }) : s;
  }

  async setZone(id: string, cls: string, zone: Zone | null): Promise<PickState> {
    const s = await this.pickState(id, cls);
    if (zone === null) return this.putPick(id, { ...s, zone: null });
    if (!Array.isArray(zone) || zone.length !== 4) throw new EngineError(400, 'zone must be [x, y, w, h] or null');
    const z = zone.map((v) => num(v, 'zone'));
    try {
      return this.putPick(id, { ...s, zone: roundZone(new ZoneRect(z[0], z[1], z[2], z[3])) });
    } catch (e) {
      throw e instanceof EngineError ? e : new EngineError(400, e instanceof Error ? e.message : String(e));
    }
  }

  async resetPick(id: string, cls: string): Promise<PickState> {
    const name = cls || 'person';
    classId(name);
    await (await this.store()).deletePick(id, name);
    return emptyPick(name);
  }

  /** Stores a pick state; one with neither taps nor a zone is removed (nothing left to remember). */
  private async putPick(id: string, s: PickState): Promise<PickState> {
    const st = await this.store();
    if (!s.taps.length && !s.zone) await st.deletePick(id, s.class);
    else await st.putPick(id, s.class, s);
    return s;
  }

  // ---------------------------------------------------------------- later parts
  openDialog = nope;
  openFromPhotos = nope;
  listDir = nope;
  savedTargets = nope;
  targetCropUrl = nope;
  choosePhotos = nope;
  startRank = nope;
  startPeople = nope;
  confirmCandidate = nope;
  candidateUrl = nope;
  personUrl = nope;
  startTargetAdd = nope;
  reveal = nope;
  musicLibrary = nope;
  musicSearch = nope;
  musicGet = nope;
  musicAuto = nope;
  musicUrl = nope;
  musicChooseFile = nope;
  quit = nope;
}
