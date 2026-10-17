// LocalEngine: the Engine over `inframe serve` (docs/T9A_DESIGN.md 2.10, 3, 4). The session token travels only in the
// X-Inframe-Token header; media URLs (<video>, <img>, <audio>) carry the read-only media key as ?m= and never the session token.
import {
  EditConflict,
  EngineError,
  type Box,
  type Caps,
  type DirListing,
  type EditClip,
  type EditsDoc,
  type Engine,
  type ExportRequest,
  type ExportResult,
  type Filmstrip,
  type FrameBoxes,
  type Job,
  type JobRef,
  type MediaInfo,
  type MediaTask,
  type MusicAutoQuery,
  type MusicAutoResult,
  type MusicFile,
  type MusicLibraryResult,
  type MusicQuery,
  type MusicSearchResult,
  type PhotoSet,
  type PickState,
  type RankRequest,
  type RecentEntry,
  type ResegmentParams,
  type ResegmentResult,
  type RunData,
  type RunRequest,
  type RunSummary,
  type SavedTarget,
  type Samples,
  type Session,
  type TargetAddRequest,
  type Zone,
} from './types';

/** how often a running `music-auto` job is asked for its progress */
const AUTO_POLL_MS = 500;

export const OLDER_SERVE = 'This page belongs to an older inframe serve. Open the address the terminal printed.';
export const NOT_RUNNING = 'inframe serve is not running. Start it again in the terminal.';
const STORE_KEY = 'inframe.session';

const NO_CAPS: Caps = {
  openDialog: false, photosPicker: false, fileBrowser: false, locate: false, reveal: false, music: false, beatSync: false,
  savedTargets: false, people: false, proxy: false, output: 'server-folder', streamCopyCut: false, chooseOutputFolder: false, savePhotos: false,
  reel: true, editLists: true,
};

export interface LocalOptions {
  token: string;
  mediaKey: string;
  base?: string;
  fetch?: typeof fetch;
  onDown?: (down: boolean) => void;
}

export interface ConnectOptions {
  base?: string;
  fetch?: typeof fetch;
  storage?: Pick<Storage, 'getItem' | 'setItem'> | null;
  hash?: string;
  /** Called with the address to show after the fragment is removed. */
  replaceUrl?: (url: string) => void;
  onDown?: (down: boolean) => void;
}

function enc(s: string): string {
  return encodeURIComponent(s);
}

export class LocalEngine implements Engine {
  readonly kind = 'local' as const;
  caps: Caps = NO_CAPS;
  private readonly token: string;
  private readonly mediaKey: string;
  private readonly base: string;
  private readonly fetchFn: typeof fetch;
  private readonly onDown?: (down: boolean) => void;

  constructor(o: LocalOptions) {
    this.token = o.token;
    this.mediaKey = o.mediaKey;
    this.base = o.base ?? './';
    this.fetchFn = o.fetch ?? ((...a) => fetch(...a));
    this.onDown = o.onDown;
  }

  /** Trade the one-use nonce in the URL fragment for the token and media key (kept in sessionStorage), or read them back after a reload.
   *  Returns null when there is neither a nonce nor stored keys. A refused claim throws the server's message. */
  static async connect(o: ConnectOptions = {}): Promise<LocalEngine | null> {
    const base = o.base ?? './';
    const f = o.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
    const storage = o.storage === undefined ? safeSessionStorage() : o.storage;
    const hash = o.hash ?? (typeof location !== 'undefined' ? location.hash : '');
    const nonce = /^#n=([A-Za-z0-9_-]+)/.exec(hash)?.[1];
    let creds: { token: string; media_key: string } | null = null;
    if (nonce) {
      let r: Response;
      try {
        r = await f(`${base}api/session/claim`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nonce }),
        });
      } catch {
        throw new EngineError(0, NOT_RUNNING);
      }
      const body = await readJson(r);
      if (!r.ok) throw new EngineError(r.status, String(body?.error ?? 'This link was used or has expired. Press Enter in the terminal for a new one.'), body ?? {});
      creds = { token: String(body.token), media_key: String(body.media_key) };
      try { storage?.setItem(STORE_KEY, JSON.stringify(creds)); } catch { /* private window: the page works until it is reloaded */ }
      if (o.replaceUrl) o.replaceUrl(base === './' && typeof location !== 'undefined' ? location.pathname + location.search : '');
      else if (typeof history !== 'undefined' && typeof location !== 'undefined') history.replaceState(null, '', location.pathname + location.search);
    } else {
      try {
        const raw = storage?.getItem(STORE_KEY);
        if (raw) {
          const p = JSON.parse(raw);
          if (typeof p?.token === 'string' && typeof p?.media_key === 'string') creds = p;
        }
      } catch { /* unreadable: no credentials */ }
    }
    if (!creds) return null;
    return new LocalEngine({ token: creds.token, mediaKey: creds.media_key, base, fetch: o.fetch, onDown: o.onDown });
  }

  // ---------------------------------------------------------------- plumbing
  private async req<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    const headers: Record<string, string> = { 'X-Inframe-Token': this.token };
    const init: RequestInit = { method, headers, signal };
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    let r: Response;
    try {
      r = await this.fetchFn(`${this.base}${path}`, init);
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      this.onDown?.(true);
      throw new EngineError(0, NOT_RUNNING);
    }
    this.onDown?.(false);
    const data = await readJson(r);
    if (r.ok) return data as T;
    const msg = typeof data?.error === 'string' ? data.error : '';
    if (r.status === 403 && (!msg || msg.includes('older inframe serve') || msg === 'forbidden')) throw new EngineError(403, OLDER_SERVE, data ?? {});
    throw new EngineError(r.status, msg || r.statusText || `request failed (${r.status})`, data ?? {});
  }

  private media_(path: string, query = ''): string {
    return `${this.base}${path}?${query}m=${enc(this.mediaKey)}`;
  }

  private keyed(url: string | null | undefined): string | null {
    if (!url) return null;
    const rel = url.startsWith('/') ? url.slice(1) : url;
    return `${this.base}${rel}${rel.includes('?') ? '&' : '?'}m=${enc(this.mediaKey)}`;
  }

  private task(t: MediaTask): MediaTask {
    return { ...t, url: this.keyed(t.url) };
  }

  // ---------------------------------------------------------------- session, media
  async session(): Promise<Session> {
    const s = await this.req<Session>('GET', 'api/session');
    // an older server does not name reel and editLists: My Mac always has both
    this.caps = { ...s.caps, reel: s.caps.reel ?? true, editLists: s.caps.editLists ?? true };
    return { ...s, caps: this.caps };
  }
  openPath(path: string) { return this.req<MediaInfo>('POST', 'api/media/open', { path }); }
  async openDialog() { return cancelled(await this.req<MediaInfo | { cancelled: true }>('POST', 'api/media/open-dialog', { kind: 'video' })); }
  async openFromPhotos() { return cancelled(await this.req<MediaInfo | { cancelled: true }>('POST', 'api/media/photos-pick', {})); }
  async locate(f: { name: string; size: number; lastModified: number }) {
    try {
      return await this.req<MediaInfo>('POST', 'api/media/locate', { name: f.name, size: f.size, last_modified_ms: f.lastModified });
    } catch (e) {
      if (e instanceof EngineError && e.status === 404) return null;
      throw e;
    }
  }
  listDir(dir?: string, mode: 'video' | 'folder' | 'audio' = 'video') {
    const q = (dir ? `dir=${enc(dir)}&` : '') + `mode=${mode}`;
    return this.req<DirListing>('GET', `api/fs/list?${q}`);
  }
  async recent() { return (await this.req<{ recent: RecentEntry[] }>('GET', 'api/recent')).recent; }
  async clearRecent() { return (await this.req<{ recent: RecentEntry[] }>('DELETE', 'api/recent')).recent; }
  media(id: string) { return this.req<MediaInfo>('GET', `api/media/${enc(id)}`); }
  videoUrl(id: string, opts: { asMp4?: boolean } = {}) {
    return this.media_(`api/media/${enc(id)}/video`, opts.asMp4 ? 'as=mp4&' : '');
  }
  async makeProxy(id: string) { return this.task(await this.req<MediaTask>('POST', `api/media/${enc(id)}/proxy`, {})); }
  async mediaTask(id: string, kind: 'proxy' | 'filmstrip') {
    if (kind === 'proxy') return this.task(await this.req<MediaTask>('GET', `api/media/${enc(id)}/proxy`));
    const f = await this.filmstrip(id);
    const state = f.state === 'running' || f.state === 'done' || f.state === 'failed' || f.state === 'none' ? f.state : f.done ? 'done' : 'running';
    return { state, fraction: f.done ? 1 : 0 } as MediaTask;
  }
  filmstrip(id: string) { return this.req<Filmstrip>('GET', `api/media/${enc(id)}/filmstrip`); }
  sheetUrl(id: string, n: number) { return this.media_(`api/media/${enc(id)}/filmstrip/${n}.jpg`); }
  frameUrl(id: string, t: number, width?: number) {
    return this.media_(`api/media/${enc(id)}/frame.jpg`, `t=${t.toFixed(3)}&${width ? `w=${Math.round(width)}&` : ''}`);
  }
  pickFrameUrl(id: string, t: number, cls: string) {
    return this.media_(`api/media/${enc(id)}/pickframe.jpg`, `t=${t.toFixed(3)}&cls=${enc(cls)}&`);
  }
  boxes(id: string, t: number, cls: string, signal?: AbortSignal) {
    return this.req<FrameBoxes>('GET', `api/media/${enc(id)}/boxes?t=${t.toFixed(3)}&cls=${enc(cls)}`, undefined, signal);
  }

  // ---------------------------------------------------------------- target
  pickState(id: string, cls: string) { return this.req<PickState>('GET', `api/media/${enc(id)}/pick?cls=${enc(cls)}`); }
  tap(id: string, cls: string, t: number, box: Box, index: number) {
    return this.req<PickState>('POST', `api/media/${enc(id)}/pick/tap`, { cls, t, box, index });
  }
  undoTap(id: string, cls: string) { return this.req<PickState>('POST', `api/media/${enc(id)}/pick/undo`, { cls }); }
  setZone(id: string, cls: string, zone: Zone | null) { return this.req<PickState>('POST', `api/media/${enc(id)}/pick/zone`, { cls, zone }); }
  resetPick(id: string, cls: string) { return this.req<PickState>('POST', `api/media/${enc(id)}/pick/reset`, { cls }); }
  async savedTargets() { return (await this.req<{ targets: SavedTarget[] }>('GET', 'api/targets')).targets; }
  targetCropUrl(name: string, i: number) { return this.media_(`api/targets/${enc(name)}/crops/${i}.jpg`); }
  async choosePhotos(source: 'dialog' | 'photos' | 'path', path?: string) {
    return cancelled(await this.req<PhotoSet | { cancelled: true }>('POST', 'api/photos/choose', path === undefined ? { source } : { source, path }));
  }

  // ---------------------------------------------------------------- jobs
  startRun(req: RunRequest) { return this.req<JobRef>('POST', 'api/runs', req); }
  startRank(mediaId: string, req: RankRequest) { return this.jobRef(this.req<{ job: Job }>('POST', `api/media/${enc(mediaId)}/rank`, req)); }
  startPeople(mediaId: string) { return this.jobRef(this.req<{ job: Job }>('POST', `api/media/${enc(mediaId)}/people`, {})); }
  confirmCandidate(mediaId: string, i: number, cls = 'person') {
    return this.req<PickState>('POST', `api/media/${enc(mediaId)}/pick/confirm`, { cls, i });
  }
  candidateUrl(mediaId: string, i: number) { return this.media_(`api/media/${enc(mediaId)}/cands/${i}.jpg`); }
  personUrl(mediaId: string, n: number) { return this.media_(`api/media/${enc(mediaId)}/people/${n}.jpg`); }
  startExport(runId: string, req: ExportRequest) { return this.jobRef(this.req<{ job: Job }>('POST', `api/runs/${enc(runId)}/export`, req)); }
  startTargetAdd(name: string, req: TargetAddRequest) { return this.jobRef(this.req<{ job: Job }>('POST', `api/targets/${enc(name)}/add`, req)); }
  job(id: string) { return this.req<Job>('GET', `api/jobs/${enc(id)}`); }
  async currentJob() { return (await this.req<{ job: Job | null }>('GET', 'api/jobs/current')).job; }
  cancel(id: string) { return this.req<Job>('POST', `api/jobs/${enc(id)}/cancel`, {}); }
  private async jobRef(p: Promise<{ job: Job }>): Promise<JobRef> { return p; }

  // ---------------------------------------------------------------- runs and edits
  async runs(mediaId: string) { return (await this.req<{ runs: RunSummary[] }>('GET', `api/media/${enc(mediaId)}/runs`)).runs; }
  run(runId: string) { return this.req<RunData>('GET', `api/runs/${enc(runId)}`); }
  samples(runId: string) { return this.req<Samples>('GET', `api/runs/${enc(runId)}/samples`); }
  async saveEdits(runId: string, revision: number, clips: EditClip[]) {
    try {
      return await this.req<EditsDoc>('PUT', `api/runs/${enc(runId)}/edits`, { revision, clips });
    } catch (e) {
      if (e instanceof EngineError && e.status === 409) throw new EditConflict((e.extra.edits as EditsDoc | undefined) ?? null);
      throw e;
    }
  }
  async revertEdits(runId: string) { await this.req('DELETE', `api/runs/${enc(runId)}/edits`); }
  resegment(runId: string, p: ResegmentParams) { return this.req<ResegmentResult>('POST', `api/runs/${enc(runId)}/resegment`, p); }
  async exportResult(runId: string) {
    try {
      return await this.req<ExportResult>('GET', `api/runs/${enc(runId)}/export`);
    } catch (e) {
      if (e instanceof EngineError && e.status === 404) return null;
      throw e;
    }
  }
  runFileUrl(runId: string, name: string) { return this.media_(`api/runs/${enc(runId)}/files/${enc(name)}`); }
  async reveal(runId: string, name?: string) { await this.req('POST', `api/runs/${enc(runId)}/reveal`, { name: name ?? null }); }

  // ---------------------------------------------------------------- music
  async musicLibrary(cc0Only: boolean) {
    return this.req<MusicLibraryResult>('GET', `api/music/library?cc0_only=${cc0Only ? '1' : '0'}`);
  }
  async musicSearch(q: MusicQuery) {
    const p = new URLSearchParams({ mood: q.mood, cc0_only: q.cc0_only ? '1' : '0' });
    if (q.min_s !== undefined) p.set('min_s', String(q.min_s));
    if (q.limit !== undefined) p.set('limit', String(q.limit));
    return this.req<MusicSearchResult>('GET', `api/music/search?${p}`);
  }
  musicGet(id: string, cc0Only: boolean) { return this.req<MusicFile>('POST', 'api/music/get', { id, cc0_only: cc0Only }); }
  async musicAuto(runId: string, q: MusicAutoQuery, onJob?: (job: Job) => void, signal?: AbortSignal) {
    let job = (await this.req<{ job: Job }>('POST', `api/runs/${enc(runId)}/music/auto`, q)).job;
    onJob?.(job);
    while (job.state === 'running') {
      if (signal?.aborted) throw new EngineError(499, 'Cancelled.');
      await new Promise((r) => setTimeout(r, AUTO_POLL_MS));
      if (signal?.aborted) throw new EngineError(499, 'Cancelled.');
      job = await this.job(job.id);
      onJob?.(job);
    }
    if (job.state !== 'done') throw new EngineError(job.state === 'cancelled' ? 499 : 502, job.state === 'cancelled' ? 'Cancelled.' : job.error || 'No song could be chosen.');
    return job.result as MusicAutoResult;
  }
  musicUrl(id: string) { return this.media_(`api/music/${enc(id)}/audio`); }
  async musicChooseFile() {
    return cancelled(await this.req<{ file: string; license: string | null } | { cancelled: true }>('POST', 'api/music/choose-file', {}));
  }
  async quit() { await this.req('POST', 'api/quit', {}); }
}

async function readJson(r: Response): Promise<any> {
  try {
    return await r.json();
  } catch {
    return null;
  }
}

function cancelled<T extends object>(v: T | { cancelled: true }): T | null {
  return v && (v as { cancelled?: boolean }).cancelled ? null : (v as T);
}

function safeSessionStorage(): Storage | null {
  try {
    return typeof sessionStorage === 'undefined' ? null : sessionStorage;
  } catch {
    return null;
  }
}
