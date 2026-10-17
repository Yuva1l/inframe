// The models of browser mode (docs/T9B_DESIGN.md section 3): a manifest next to the files (name, bytes, sha256, license), download with
// progress, the sha256 checked with crypto.subtle before anything is kept, Cache Storage `inframe-models-v1` keyed by the digest,
// navigator.storage.persist(), and removal. The base URL is set at build time (VITE_MODEL_BASE, default ./models/); `inframe serve`
// answers it from $INFRAME_MODELS/onnx. The manifest has its own setting (VITE_MODEL_MANIFEST, default: base + manifest.json): the Pages build
// (docs/T9C_DESIGN.md) serves the manifest from the page's own site and the files from a model host, so the host supplies bytes only and every
// digest comes from the page. Model requests carry no cookies and no referrer. Nothing here runs a model; detect.ts does.
import { EngineError } from '../engine/types';

export const CACHE_NAME = 'inframe-models-v1';
export const MANIFEST_FILE = 'manifest.json';
/** The files a detection run needs, by role (tools/export_onnx.py names them). */
export const DETECTOR_FILE = 'rfdetr_small_v1.onnx';
export const EMBEDDER_FILE = 'dinov2_s_v1.onnx';
export const WASM_FILE = 'ort-wasm-simd-threaded.asyncify.wasm';

export interface ManifestEntry {
  bytes: number;
  sha256: string;
  license: string;
  source: string;
  role: 'detector' | 'embedder' | 'runtime';
  inputs?: { name: string; shape: (number | string | null)[] }[];
  outputs?: string[];
}
export interface Manifest {
  format: 1;
  files: Record<string, ManifestEntry>;
}

export interface FileStatus {
  name: string;
  bytes: number;
  cached: boolean;
  license: string;
  role: ManifestEntry['role'];
}
export interface ModelStatus {
  /** the manifest could be read */
  available: boolean;
  error: string | null;
  files: FileStatus[];
  /** where the model files come from (the host of the base; "this site" when it is the page's own) */
  host?: string;
  /** files not in the cache yet */
  filesLeft: number;
  totalBytes: number;
  cachedBytes: number;
  /** the detector and the runtime are in the cache: a run can start */
  detectorReady: boolean;
  /** everything is in the cache */
  allReady: boolean;
  persisted: boolean | null;
}

export interface Progress {
  /** the file being fetched */
  name: string;
  /** bytes of all files done so far, this one included */
  done: number;
  total: number;
}

/** The base URL of the model files, absolute: `base` (a build-time string such as ./models/) against `root`, with a closing slash. */
export function resolveBase(base: string, root: string): string {
  const u = new URL(base, root);
  if (!u.pathname.endsWith('/')) u.pathname += '/';
  return u.href;
}

export function defaultBase(): string {
  const fromBuild = import.meta.env.VITE_MODEL_BASE as string | undefined;
  const base = fromBuild || './models/';
  const root = typeof document !== 'undefined' ? document.baseURI : typeof location !== 'undefined' ? location.href : null;
  return root ? resolveBase(base, root) : base.endsWith('/') ? base : base + '/';
}

/** Where manifest.json is read: VITE_MODEL_MANIFEST (a build-time URL, against the page) when set, else next to the files. */
export function defaultManifestUrl(base: string = defaultBase()): string {
  const fromBuild = import.meta.env.VITE_MODEL_MANIFEST as string | undefined;
  if (!fromBuild) return base + MANIFEST_FILE;
  const root = typeof document !== 'undefined' ? document.baseURI : typeof location !== 'undefined' ? location.href : null;
  return root ? new URL(fromBuild, root).href : fromBuild;
}

/** Fetch options of every model request: no cookies to the model host, and no page address sent along. */
const MODEL_FETCH = { credentials: 'omit', referrerPolicy: 'no-referrer' } as const;

export function validManifest(m: unknown): m is Manifest {
  const f = (m as Manifest | null)?.files;
  if (!f || typeof f !== 'object' || (m as Manifest).format !== 1) return false;
  return Object.entries(f).every(([n, e]) => /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(n) && Number.isInteger(e?.bytes) && e.bytes > 0 && /^[0-9a-f]{64}$/.test(e.sha256));
}

export const mb = (bytes: number): string => `${Math.round(bytes / 1e6)} MB`;

export async function sha256Hex(buf: ArrayBuffer | Uint8Array, subtle: SubtleCrypto | undefined = globalThis.crypto?.subtle): Promise<string> {
  if (!subtle) throw new EngineError(500, 'This page cannot check downloads (no crypto.subtle). Open it from localhost or https.');
  const d = new Uint8Array(await subtle.digest('SHA-256', buf as BufferSource));
  return [...d].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** The parts a test replaces. */
export interface ModelDeps {
  fetch?: typeof fetch;
  caches?: CacheStorage;
  /** the origin the cache keys live under */
  origin?: string;
  subtle?: SubtleCrypto;
  persist?: () => Promise<boolean>;
  /** whether the browser already keeps this site's storage (navigator.storage.persisted) */
  persisted?: () => Promise<boolean>;
}

export class ModelStore {
  private man: Manifest | null = null;
  private persisted: boolean | null = null;

  readonly manifestUrl: string;

  constructor(readonly base: string = defaultBase(), private deps: ModelDeps = {}, manifestUrl?: string) {
    this.manifestUrl = manifestUrl ?? defaultManifestUrl(base);
  }

  /** The host the model files are fetched from, for the page to name: "this site" when it is the page's own origin. */
  get host(): string {
    try {
      const u = new URL(this.base);
      const own = this.deps.origin ?? (typeof location !== 'undefined' ? location.origin : '');
      return u.origin === own ? 'this site' : u.host;
    } catch {
      return 'this site';
    }
  }

  private get fetchFn(): typeof fetch {
    return this.deps.fetch ?? ((...a) => fetch(...a));
  }
  private cacheStorage(): CacheStorage {
    const c = this.deps.caches ?? (typeof caches === 'undefined' ? undefined : caches);
    if (!c) throw new EngineError(500, 'This browser has no Cache Storage here (a private window?), so the models cannot be kept. Use My Mac.');
    return c;
  }
  private keyFor(sha: string): string {
    const origin = this.deps.origin ?? (typeof location !== 'undefined' ? location.origin : '');
    return `${origin}/__inframe_models__/${sha}`;
  }

  /** manifest.json from the base (read again with `fresh`). */
  async manifest(fresh = false): Promise<Manifest> {
    if (this.man && !fresh) return this.man;
    let r: Response;
    try {
      r = await this.fetchFn(this.manifestUrl, { cache: 'no-cache', ...MODEL_FETCH });
    } catch (e) {
      throw new EngineError(502, `The model list could not be fetched (${e instanceof Error ? e.message : String(e)}).`);
    }
    if (!r.ok) throw new EngineError(r.status === 404 ? 404 : 502, r.status === 404 ? 'No models are published here. Run tools/export_onnx.py, or use My Mac.' : `The model list answered ${r.status}.`);
    const m = (await r.json().catch(() => null)) as unknown;
    if (!validManifest(m)) throw new EngineError(502, 'The model list is not in a format this page knows.');
    return (this.man = m);
  }

  /** Use a manifest that is already known (a Worker is handed the page's copy). */
  adopt(m: Manifest): void {
    this.man = m;
  }

  async status(): Promise<ModelStatus> {
    let m: Manifest;
    try {
      m = await this.manifest();
    } catch (e) {
      return { available: false, error: e instanceof Error ? e.message : String(e), files: [], host: this.host, filesLeft: 0, totalBytes: 0, cachedBytes: 0, detectorReady: false, allReady: false, persisted: this.persisted };
    }
    let cache: Cache | null = null;
    try {
      cache = await this.cacheStorage().open(CACHE_NAME);
    } catch { /* no Cache Storage: nothing is cached */ }
    const files: FileStatus[] = [];
    for (const [name, e] of Object.entries(m.files)) {
      const hit = cache ? await cache.match(this.keyFor(e.sha256)).catch(() => undefined) : undefined;
      files.push({ name, bytes: e.bytes, cached: !!hit, license: e.license, role: e.role });
    }
    const cached = (n: string) => files.some((f) => f.name === n && f.cached);
    if (this.persisted === null && files.some((f) => f.cached)) await this.readPersisted();
    return {
      available: true, error: null, files, host: this.host, filesLeft: files.filter((f) => !f.cached).length, totalBytes: files.reduce((s, f) => s + f.bytes, 0), cachedBytes: files.reduce((s, f) => s + (f.cached ? f.bytes : 0), 0),
      detectorReady: cached(DETECTOR_FILE) && cached(WASM_FILE), allReady: files.every((f) => f.cached), persisted: this.persisted,
    };
  }

  /** Fetch every file that is not cached yet, check each digest and keep it. A file that fails its check is dropped and the error says which. */
  async download(opts: { onProgress?: (p: Progress) => void; signal?: AbortSignal; only?: string[] } = {}): Promise<void> {
    const m = await this.manifest();
    const cache = await this.cacheStorage().open(CACHE_NAME);
    await this.requestPersist();
    const names = Object.keys(m.files).filter((n) => !opts.only || opts.only.includes(n));
    const todo: string[] = [];
    for (const n of names) if (!(await cache.match(this.keyFor(m.files[n].sha256)))) todo.push(n);
    const total = todo.reduce((s, n) => s + m.files[n].bytes, 0);
    let done = 0;
    for (const name of todo) {
      const e = m.files[name];
      const buf = await this.fetchOne(name, e, (n) => opts.onProgress?.({ name, done: done + n, total }), opts.signal);
      const got = await sha256Hex(buf, this.deps.subtle);
      if (got !== e.sha256) throw new EngineError(502, `${name} did not match its checksum, so it was not kept. Try again.`);
      await cache.put(this.keyFor(e.sha256), new Response(buf as BodyInit, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(buf.byteLength) } }));
      done += e.bytes;
      opts.onProgress?.({ name, done, total });
    }
  }

  private async fetchOne(name: string, e: ManifestEntry, onBytes: (n: number) => void, signal?: AbortSignal): Promise<Uint8Array> {
    let r: Response;
    try {
      r = await this.fetchFn(this.base + name, { signal, ...MODEL_FETCH });
    } catch (err) {
      if (signal?.aborted) throw new EngineError(499, 'Cancelled.');
      throw new EngineError(502, `${name} could not be fetched (${err instanceof Error ? err.message : String(err)}).`);
    }
    if (!r.ok) throw new EngineError(502, `${name} answered ${r.status}.`);
    const out = new Uint8Array(e.bytes);
    let at = 0;
    const take = (chunk: Uint8Array) => {
      if (at + chunk.byteLength > e.bytes) throw new EngineError(502, `${name} is larger than the model list says, so it was not kept.`);
      out.set(chunk, at);
      at += chunk.byteLength;
      onBytes(at);
    };
    try {
      if (r.body) {
        const rd = r.body.getReader();
        for (;;) {
          const { done, value } = await rd.read();
          if (done) break;
          take(value);
        }
      } else take(new Uint8Array(await r.arrayBuffer()));
    } catch (err) {
      if (signal?.aborted) throw new EngineError(499, 'Cancelled.');
      if (err instanceof EngineError) throw err;
      throw new EngineError(502, `${name} stopped before the end (${err instanceof Error ? err.message : String(err)}).`);
    }
    if (at !== e.bytes) throw new EngineError(502, `${name} is ${at} bytes, the model list says ${e.bytes}, so it was not kept.`);
    return out;
  }

  /** After a reload `persisted` is unknown: ask the browser what it decided earlier (without asking it to persist), so the note about clearing still shows. */
  private async readPersisted(): Promise<void> {
    try {
      this.persisted = this.deps.persisted ? await this.deps.persisted() : typeof navigator !== 'undefined' && navigator.storage?.persisted ? await navigator.storage.persisted() : null;
    } catch {
      this.persisted = null;
    }
  }

  private async requestPersist(): Promise<void> {
    try {
      this.persisted = this.deps.persist ? await this.deps.persist() : typeof navigator !== 'undefined' && navigator.storage?.persist ? await navigator.storage.persist() : null;
    } catch {
      this.persisted = null;
    }
  }

  /** The bytes of a cached file (a Worker reads its models here). Throws 409 when it was never downloaded. */
  async bytes(name: string): Promise<Uint8Array> {
    const m = await this.manifest();
    const e = m.files[name];
    if (!e) throw new EngineError(404, `${name} is not in the model list.`);
    const hit = await (await this.cacheStorage().open(CACHE_NAME)).match(this.keyFor(e.sha256));
    if (!hit) throw new EngineError(409, 'The models are not downloaded yet. Download them first.');
    return new Uint8Array(await hit.arrayBuffer());
  }

  /** Forget every downloaded file. */
  async remove(): Promise<void> {
    await this.cacheStorage().delete(CACHE_NAME);
  }
}
