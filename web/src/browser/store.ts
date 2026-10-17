// What the browser engine remembers between visits (docs/T9B_DESIGN.md section 2): IndexedDB `inframe` with media facts, runs (the
// detector's segments, samples, the edits document) and pick state. It never holds video, frames or embeddings. When IndexedDB is not
// there (a private window, blocked site data) the same interface runs on a Map and forgets at the next reload.
// The edits follow serve_edits.save_edits: a revision guard (EditConflict carries the stored document), atomic per run, and
// revert moves the document aside (at most five kept) instead of deleting it.
import { EditConflict, EngineError, type EditClip, type EditsDoc, type MediaInfo, type PickState, type RecentEntry, type ResegmentParams, type ResegmentResult, type RunData, type RunSummary, type Samples } from '../engine/types';
import { validate, r3 } from '../model/edits';
import { resegment as buildResegment } from '../model/segments';

export const DB_NAME = 'inframe';
const DB_VERSION = 1;
const KEEP_PREV = 5;
const SAME_TOL_S = 0.0015; // serve_edits.SAME_TOL_S: samples are kept to 3 decimals, so a re-segment can differ by a rounding step
const EDITS_FORMAT = 1;

/** MediaInfo as stored: the facts of the file, without the URLs that only mean something while the file is open. */
export type MediaRecord = Omit<MediaInfo, 'video_url' | 'proxy' | 'filmstrip_state' | 'picked_copy'> & { opened: number };

export interface RunRecord {
  run_id: string;
  media_id: string;
  /** ms since the epoch */
  created: number;
  slug: string;
  target: string;
  preset: string;
  segments: RunData['segments'];
  samples: Samples | null;
  /** the video's first picture and end in file time, as serve_edits.source_bounds reads them from segments.json */
  source: { offset: number; end_time: number };
  /** identifies the run's segments (edits.base.segments_sha256) */
  sha: string;
  edits: EditsDoc | null;
  prev_edits: EditsDoc[];
}

type StoreName = 'media' | 'runs' | 'pick';

/** The storage seam: whole values by key, plus an atomic read-modify-write (one IndexedDB transaction). */
export interface Backend {
  get<T>(store: StoreName, key: string): Promise<T | undefined>;
  put(store: StoreName, key: string, value: unknown): Promise<void>;
  delete(store: StoreName, key: string): Promise<void>;
  /** removes every record of a store */
  clear(store: StoreName): Promise<void>;
  all<T>(store: StoreName): Promise<T[]>;
  update<T, R>(store: StoreName, key: string, fn: (cur: T | undefined) => { value?: T; result: R }): Promise<R>;
}

// ---------------------------------------------------------------- backends
const req = <T>(r: IDBRequest<T>): Promise<T> => new Promise((ok, no) => {
  r.onsuccess = () => ok(r.result);
  r.onerror = () => no(r.error ?? new Error('IndexedDB request failed'));
});
const done = (tx: IDBTransaction): Promise<void> => new Promise((ok, no) => {
  tx.oncomplete = () => ok();
  tx.onabort = () => no(tx.error ?? new Error('IndexedDB transaction aborted'));
  tx.onerror = () => no(tx.error ?? new Error('IndexedDB transaction failed'));
});

export class IdbBackend implements Backend {
  private constructor(private db: IDBDatabase) {}

  static async open(factory: IDBFactory = indexedDB, name = DB_NAME): Promise<IdbBackend> {
    const r = factory.open(name, DB_VERSION);
    r.onupgradeneeded = () => {
      const db = r.result;
      for (const s of ['media', 'runs', 'pick'] as const) if (!db.objectStoreNames.contains(s)) db.createObjectStore(s);
    };
    return new IdbBackend(await req(r));
  }

  close(): void {
    this.db.close();
  }

  async get<T>(store: StoreName, key: string): Promise<T | undefined> {
    return (await req(this.db.transaction(store).objectStore(store).get(key))) as T | undefined;
  }
  async put(store: StoreName, key: string, value: unknown): Promise<void> {
    const tx = this.db.transaction(store, 'readwrite');
    tx.objectStore(store).put(value, key);
    await done(tx);
  }
  async delete(store: StoreName, key: string): Promise<void> {
    const tx = this.db.transaction(store, 'readwrite');
    tx.objectStore(store).delete(key);
    await done(tx);
  }
  async clear(store: StoreName): Promise<void> {
    const tx = this.db.transaction(store, 'readwrite');
    tx.objectStore(store).clear();
    await done(tx);
  }
  async all<T>(store: StoreName): Promise<T[]> {
    return (await req(this.db.transaction(store).objectStore(store).getAll())) as T[];
  }
  async update<T, R>(store: StoreName, key: string, fn: (cur: T | undefined) => { value?: T; result: R }): Promise<R> {
    const tx = this.db.transaction(store, 'readwrite');
    const os = tx.objectStore(store);
    const finished = done(tx);
    try {
      const out = fn((await req(os.get(key))) as T | undefined);
      if (out.value !== undefined) os.put(out.value, key);
      await finished;
      return out.result;
    } catch (e) {
      try {
        tx.abort(); // nothing is written when fn throws (EditConflict)
      } catch { /* already finished */ }
      finished.catch(() => {});
      throw e;
    }
  }
}

export class MemoryBackend implements Backend {
  private m: Record<StoreName, Map<string, unknown>> = { media: new Map(), runs: new Map(), pick: new Map() };
  async get<T>(store: StoreName, key: string) {
    return structuredClone(this.m[store].get(key)) as T | undefined;
  }
  async put(store: StoreName, key: string, value: unknown) {
    this.m[store].set(key, structuredClone(value));
  }
  async delete(store: StoreName, key: string) {
    this.m[store].delete(key);
  }
  async clear(store: StoreName) {
    this.m[store].clear();
  }
  async all<T>(store: StoreName) {
    return [...this.m[store].values()].map((v) => structuredClone(v) as T);
  }
  async update<T, R>(store: StoreName, key: string, fn: (cur: T | undefined) => { value?: T; result: R }) {
    const out = fn(structuredClone(this.m[store].get(key)) as T | undefined);
    if (out.value !== undefined) this.m[store].set(key, structuredClone(out.value));
    return out.result;
  }
}

/** IndexedDB when it works, else memory. */
export async function openBackend(factory: IDBFactory | undefined = typeof indexedDB === 'undefined' ? undefined : indexedDB): Promise<{ backend: Backend; persistent: boolean }> {
  if (factory) {
    try {
      return { backend: await IdbBackend.open(factory), persistent: true };
    } catch { /* blocked: fall through */ }
  }
  return { backend: new MemoryBackend(), persistent: false };
}

// ---------------------------------------------------------------- hashing
/** A hex digest naming the run's segments. SHA-256 where crypto.subtle exists; elsewhere (a plain http page, an old test runtime) four
 *  FNV-1a lanes, which only have to tell two documents apart. */
export async function digest(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  try {
    const subtle = globalThis.crypto?.subtle;
    if (subtle) return [...new Uint8Array(await subtle.digest('SHA-256', bytes))].map((b) => b.toString(16).padStart(2, '0')).join('');
  } catch { /* fall through */ }
  const lanes = [0x811c9dc5, 0x01000193, 0xdeadbeef, 0x9e3779b9];
  for (const b of bytes) for (let k = 0; k < 4; k++) lanes[k] = Math.imul(lanes[k] ^ (b + k), 0x01000193) >>> 0;
  return lanes.map((x) => x.toString(16).padStart(8, '0')).join('').repeat(2);
}

// ---------------------------------------------------------------- the store
const pairs = (iv: readonly (readonly [number, number])[]): [number, number][] => iv.map(([s, e]) => [Math.round(s * 1000) / 1000, Math.round(e * 1000) / 1000]);
const sameIntervals = (a: readonly (readonly number[])[], b: unknown): boolean => {
  try {
    const bb = (b as number[][]).map((x) => [Number(x[0]), Number(x[1])]);
    return a.length === bb.length && a.every((p, i) => Math.abs(p[0] - bb[i][0]) <= SAME_TOL_S && Math.abs(p[1] - bb[i][1]) <= SAME_TOL_S);
  } catch {
    return false;
  }
};

/** serve_edits.detector_edits_doc: revision 0 with the detector's clips as d1, d2, ... */
export function detectorDoc(run: Pick<RunRecord, 'segments' | 'sha'>): EditsDoc {
  const clips: EditClip[] = (run.segments.clips ?? []).map((c, i) => ({ id: `d${i + 1}`, start: c.start, end: c.end, origin: 'detector' as const }));
  return { format: EDITS_FORMAT, base: { segments_sha256: run.sha, n_clips: clips.length }, revision: 0, updated: '', next_id: 1, clips };
}

const nextId = (stored: number | undefined, clips: readonly EditClip[]): number => {
  let top = 0;
  for (const c of clips) {
    const m = /^u(\d+)$/.exec(c.id);
    if (m) top = Math.max(top, Number(m[1]));
  }
  return Math.max(stored !== undefined && Number.isInteger(stored) ? stored : 1, top + 1);
};

export class BrowserStore {
  constructor(readonly backend: Backend, readonly persistent = true, private now: () => number = Date.now) {}

  // media ------------------------------------------------------------
  async putMedia(m: MediaRecord): Promise<void> {
    await this.backend.put('media', m.id, m);
  }
  getMedia(id: string): Promise<MediaRecord | undefined> {
    return this.backend.get<MediaRecord>('media', id);
  }
  /** newest first; a record with `opened` 0 is left out */
  async recent(): Promise<RecentEntry[]> {
    const all = await this.backend.all<MediaRecord>('media');
    return all.filter((m) => m.opened > 0).sort((a, b) => b.opened - a.opened).map((m) => ({ path: m.id, name: m.name, opened: m.opened }));
  }
  /** Deletes what the list was built from: every media record (the file names), the runs made on them (their slugs carry the name) and the
   *  pick state. A public page must not keep file names after "Clear recent". `keep` names the media this tab has open: they, their runs and
   *  their picks stay (as records out of the list, `opened` 0), or the open video could no longer save edits or export. */
  async clearRecent(keep: readonly string[] = []): Promise<void> {
    const keepSet = new Set(keep);
    const media = (await this.backend.all<MediaRecord>('media')).filter((m) => keepSet.has(m.id));
    const runs = (await this.backend.all<RunRecord>('runs')).filter((r) => keepSet.has(r.media_id));
    // pick records carry no key, but the key is `${media}|${state.class}` (putPick): a state belongs to the open media when that key holds it
    const picks = new Map<string, PickState>();
    for (const p of await this.backend.all<PickState>('pick')) {
      for (const m of keepSet) {
        const k = `${m}|${p.class}`;
        if (!picks.has(k) && JSON.stringify(await this.backend.get('pick', k)) === JSON.stringify(p)) picks.set(k, p);
      }
    }
    for (const s of ['media', 'runs', 'pick'] as const) await this.backend.clear(s);
    for (const m of media) await this.backend.put('media', m.id, { ...m, opened: 0 });
    for (const r of runs) await this.backend.put('runs', r.run_id, r);
    for (const [k, v] of picks) await this.backend.put('pick', k, v);
  }
  // runs -------------------------------------------------------------
  /** Store a finished run (part 3 calls this when a run ends). `sha` is made from the segments when not given. */
  async putRun(r: Omit<RunRecord, 'sha' | 'edits' | 'prev_edits' | 'created'> & Partial<Pick<RunRecord, 'sha' | 'edits' | 'prev_edits' | 'created'>>): Promise<RunRecord> {
    const rec: RunRecord = { ...r, created: r.created ?? this.now(), sha: r.sha ?? (await digest(JSON.stringify(r.segments))), edits: r.edits ?? null, prev_edits: r.prev_edits ?? [] };
    await this.backend.put('runs', rec.run_id, rec);
    return rec;
  }
  getRun(runId: string): Promise<RunRecord | undefined> {
    return this.backend.get<RunRecord>('runs', runId);
  }
  private async needRun(runId: string): Promise<RunRecord> {
    const r = await this.getRun(runId);
    if (!r) throw new EngineError(404, `No such run: ${runId}`);
    return r;
  }

  async runs(mediaId: string): Promise<RunSummary[]> {
    const all = (await this.backend.all<RunRecord>('runs')).filter((r) => r.media_id === mediaId).sort((a, b) => b.created - a.created);
    return all.map((r) => ({
      run_id: r.run_id, dir: `browser:${r.run_id}`, slug: r.slug, target: r.target, preset: r.preset, created: r.created, n_clips: r.segments.clips?.length ?? 0,
      error: r.segments.error ?? null, has_edits: r.edits !== null, edits_base_ok: r.edits === null || r.edits.base.segments_sha256 === r.sha,
    }));
  }

  async run(runId: string): Promise<RunData> {
    const r = await this.needRun(runId);
    return {
      run_id: r.run_id, media_id: r.media_id, dir: `browser:${r.run_id}`, segments: r.segments, edits: r.edits,
      edits_base_ok: r.edits === null || r.edits.base.segments_sha256 === r.sha, export: null,
    };
  }

  async samples(runId: string): Promise<Samples> {
    const r = await this.needRun(runId);
    if (!r.samples) throw new EngineError(404, 'This run has no samples.');
    return r.samples;
  }

  /** Validate and store `clips` as revision `revision + 1`. A `revision` that is not the stored one (0 when nothing is stored) throws
   *  EditConflict carrying the stored document, or the detector's clips at revision 0. Atomic per run. */
  async saveEdits(runId: string, revision: number, clips: readonly EditClip[]): Promise<EditsDoc> {
    if (!Number.isInteger(revision) || revision < 0) throw new EngineError(400, 'revision must be a whole number from 0');
    if (!Array.isArray(clips)) throw new EngineError(400, 'clips must be a list');
    return this.backend.update<RunRecord, EditsDoc>('runs', runId, (run) => {
      if (!run) throw new EngineError(404, `No such run: ${runId}`);
      const bounds = { offset: run.source.offset, end: run.source.end_time };
      const clean: EditClip[] = clips.map((c) => ({ id: c.id, start: r3(c.start), end: r3(c.end), origin: c.origin }));
      const problems = validate(clean, bounds);
      if (problems.length) throw new EngineError(422, problems[0], { problems });
      for (const c of clean) {
        c.start = r3(Math.max(c.start, bounds.offset));
        c.end = r3(Math.min(c.end, bounds.end));
      }
      const current = run.edits?.revision ?? 0;
      if (revision !== current) throw new EditConflict(run.edits ?? detectorDoc(run));
      const doc: EditsDoc = {
        format: EDITS_FORMAT, base: { segments_sha256: run.sha, n_clips: run.segments.clips?.length ?? 0 }, revision: current + 1,
        updated: new Date(this.now()).toISOString().replace(/\.\d+Z$/, 'Z'), next_id: nextId(run.edits?.next_id, clean), clips: clean,
      };
      return { value: { ...run, edits: doc }, result: doc };
    });
  }

  /** The edits document goes aside (the newest five are kept) so the detector's clips show again. */
  async revertEdits(runId: string): Promise<void> {
    await this.backend.update<RunRecord, void>('runs', runId, (run) => {
      if (!run) throw new EngineError(404, `No such run: ${runId}`);
      if (!run.edits) return { result: undefined };
      return { value: { ...run, edits: null, prev_edits: [...run.prev_edits, run.edits].slice(-KEEP_PREV) }, result: undefined };
    });
  }

  /** What segments.build_segments makes from the run's own samples (serve_runs.resegment). Nothing is written. */
  async resegment(runId: string, p: ResegmentParams): Promise<ResegmentResult> {
    const r = await this.needRun(runId);
    if (!r.samples) throw new EngineError(404, 'This run has no samples.');
    let built;
    try {
      built = buildResegment(r.samples, r.segments.settings ?? {}, p, { offset: r.source.offset, end: r.source.end_time });
    } catch (e) {
      throw new EngineError(422, e instanceof Error ? e.message : String(e));
    }
    let presence: [number, number][] = pairs(built.presence);
    let clips: [number, number][] = pairs(built.clips);
    const runClips = (r.segments.clips ?? []).map((c): [number, number] => [c.start, c.end]);
    const same = sameIntervals(clips, runClips) && sameIntervals(presence, r.segments.presence_intervals ?? []);
    if (same) { // equal up to the rounding: hand back the run's own numbers so applying the preview changes nothing
      presence = (r.segments.presence_intervals ?? []).map(([s, e]) => [Number(s), Number(e)] as [number, number]);
      clips = runClips;
    }
    return { presence, clips, same_as_run: same };
  }

  // pick state -------------------------------------------------------
  getPick(mediaId: string, cls: string): Promise<PickState | undefined> {
    return this.backend.get<PickState>('pick', `${mediaId}|${cls}`);
  }
  putPick(mediaId: string, cls: string, s: PickState): Promise<void> {
    return this.backend.put('pick', `${mediaId}|${cls}`, s);
  }
  deletePick(mediaId: string, cls: string): Promise<void> {
    return this.backend.delete('pick', `${mediaId}|${cls}`);
  }
}

