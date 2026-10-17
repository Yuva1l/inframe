// @vitest-environment node
// The browser engine's IndexedDB store (src/browser/store.ts) on fake-indexeddb: save, conflict, revert and re-segment round trips, and
// that the engine's media and run calls read what the store wrote. Re-segment must equal the golden build_segments from Python.
import { IDBFactory } from 'fake-indexeddb';
import { describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so the file read is allowed to be untyped
// @ts-ignore
import { readFileSync } from 'node:fs';
import { buildSegments } from '../src/model/segments';
import { BrowserEngine } from '../src/engine/browser';
import { EditConflict, EngineError, type EditClip, type MediaInfo } from '../src/engine/types';
import { BrowserStore, IdbBackend, MemoryBackend, openBackend, type Backend, type MediaRecord, type RunRecord } from '../src/browser/store';

interface ReCase {
  name: string; t: number[]; score: number[]; settings: Record<string, number>; params: Record<string, number | null>;
  offset: number; end: number; expect?: { presence: [number, number][]; clips: [number, number][] }; error?: string;
}
const golden = JSON.parse(readFileSync((globalThis as any).process.cwd() + '/../tests/golden/segments_v1.json', 'utf8')) as { resegment: ReCase[] };

const near = (a: number[][], b: number[][], tol = 0.0005) => {
  expect(a.length).toBe(b.length);
  a.forEach((p, i) => p.forEach((x, k) => expect(Math.abs(x - b[i][k])).toBeLessThanOrEqual(tol)));
};

/** a run of 60 s with two detector clips, from a golden-style sample series */
function run(over: Partial<RunRecord> = {}): Parameters<BrowserStore['putRun']>[0] {
  const t: number[] = [];
  const score: number[] = [];
  for (let i = 0; i < 120; i++) {
    t.push(i * 0.5);
    score.push((i >= 20 && i < 40) || (i >= 80 && i < 100) ? 0.9 : 0.05);
  }
  // the detector's own result for these samples: what Python's build_segments gives (the golden test holds the port to it)
  const built = buildSegments(t, score, 60, { enter: 0.5, exit: 0.3, mergeGap: 1.5, minLen: 1, pad: 1 });
  const r3 = (iv: number[][]) => iv.map(([a, b]) => [Math.round(a * 1000) / 1000, Math.round(b * 1000) / 1000]);
  return {
    run_id: 'run1', media_id: 'm1', slug: 'any-person', target: 'any person', preset: 'balanced',
    segments: { settings: { enter: 0.5, exit: 0.3, merge_gap: 1.5, min_len: 1, pad: 1 }, warnings: [], clips: r3(built.clips).map(([start, end]) => ({ start, end, peak_score: 0.9 })), presence_intervals: r3(built.presence) as [number, number][] },
    samples: { t, score, box: t.map(() => null), track: null, enter: 0.5, exit: 0.3, merge_gap: 1.5, min_len: 1, pad: 1, sound: null, mode: 'detect' },
    source: { offset: 0, end_time: 60 }, ...over,
  };
}
const clip = (id: string, start: number, end: number, origin: EditClip['origin'] = 'edited'): EditClip => ({ id, start, end, origin });
const fresh = async () => new BrowserStore(await IdbBackend.open(new IDBFactory()), true, () => Date.UTC(2026, 9, 16, 12, 0, 0));

describe('edits: save, conflict, revert', () => {
  it('saves revision 1 from 0, then 2 from 1, and the run shows the document', async () => {
    const s = await fresh();
    await s.putRun(run());
    expect((await s.run('run1')).edits).toBeNull();
    const d1 = await s.saveEdits('run1', 0, [clip('d1', 8.75, 20.75, 'detector'), clip('u1', 30, 35, 'added')]);
    expect(d1).toMatchObject({ format: 1, revision: 1, next_id: 2, updated: '2026-10-16T12:00:00Z', base: { n_clips: 2 } });
    expect(d1.base.segments_sha256).toMatch(/^[0-9a-f]{64}$/);
    const d2 = await s.saveEdits('run1', 1, [clip('d1', 9, 22)]);
    expect(d2.revision).toBe(2);
    expect(d2.next_id).toBe(2); // never goes back below the stored next_id
    const r = await s.run('run1');
    expect(r.edits).toEqual(d2);
    expect(r.edits_base_ok).toBe(true);
    expect((await s.runs('m1'))[0]).toMatchObject({ run_id: 'run1', has_edits: true, edits_base_ok: true, n_clips: 2, target: 'any person' });
  });

  it('a stale revision throws EditConflict carrying the stored document, and writes nothing', async () => {
    const s = await fresh();
    await s.putRun(run());
    await s.saveEdits('run1', 0, [clip('d1', 8.75, 20.75, 'detector')]);
    const stored = await s.saveEdits('run1', 1, [clip('d1', 10, 21)]);
    const err = await s.saveEdits('run1', 1, [clip('d1', 5, 6)]).then(() => null, (e) => e);
    expect(err).toBeInstanceOf(EditConflict);
    expect(err.status).toBe(409);
    expect(err.stored).toEqual(stored);
    expect((await s.run('run1')).edits).toEqual(stored);
    // revision 0 with nothing stored is fine; with something stored it conflicts
    await s.putRun(run({ run_id: 'run2' }));
    expect(await s.saveEdits('run2', 0, [])).toMatchObject({ revision: 1, clips: [] });
    // and a conflict on a run nobody edited carries the detector's clips at revision 0
    await s.putRun(run({ run_id: 'run3' }));
    const c3 = await s.saveEdits('run3', 4, []).then(() => null, (e) => e);
    expect(c3).toBeInstanceOf(EditConflict);
    expect(c3.stored).toMatchObject({ revision: 0, clips: [{ id: 'd1', start: 8.75, end: 20.75, origin: 'detector' }, { id: 'd2', start: 38.75, end: 50.75, origin: 'detector' }] });
  });

  it('two saves from the same revision at once: exactly one wins', async () => {
    const s = await fresh();
    await s.putRun(run());
    const out = await Promise.allSettled([s.saveEdits('run1', 0, [clip('u1', 1, 3)]), s.saveEdits('run1', 0, [clip('u2', 5, 8)])]);
    expect(out.filter((o) => o.status === 'fulfilled')).toHaveLength(1);
    const lost = out.find((o) => o.status === 'rejected') as PromiseRejectedResult;
    expect(lost.reason).toBeInstanceOf(EditConflict);
    expect((await s.run('run1')).edits?.revision).toBe(1);
  });

  it('refuses clips the server would refuse (422) and rounds to a millisecond', async () => {
    const s = await fresh();
    await s.putRun(run());
    const bad = async (clips: EditClip[], re: RegExp) => {
      const e = await s.saveEdits('run1', 0, clips).then(() => null, (x) => x);
      expect(e).toBeInstanceOf(EngineError);
      expect(e.status).toBe(422);
      expect(e.message).toMatch(re);
    };
    await bad([clip('u1', 5, 5.1)], /shorter/);
    await bad([clip('u1', 5, 70)], /outside/);
    await bad([clip('u1', 10, 20), clip('u2', 15, 25)], /overlap/);
    await bad([clip('u1', 10, 20), clip('u1', 30, 40)], /duplicate/);
    await bad([clip('U1', 10, 20)], /bad id/);
    expect((await s.run('run1')).edits).toBeNull();
    const d = await s.saveEdits('run1', 0, [clip('u1', 1.00049, 3.9996)]);
    expect(d.clips[0]).toMatchObject({ start: 1, end: 4 });
  });

  it('revert moves the document aside: the detector clips show again, edits_base_ok stays true, the next save starts at 0', async () => {
    const s = await fresh();
    await s.putRun(run());
    await s.saveEdits('run1', 0, [clip('u1', 1, 3, 'added')]);
    await s.revertEdits('run1');
    const r = await s.run('run1');
    expect(r.edits).toBeNull();
    expect((await s.getRun('run1'))!.prev_edits).toHaveLength(1);
    expect((await s.runs('m1'))[0].has_edits).toBe(false);
    expect((await s.saveEdits('run1', 0, [clip('u1', 2, 4)])).revision).toBe(1);
    await s.revertEdits('run1');
    await s.revertEdits('run1'); // nothing to move: no error
    for (let i = 0; i < 7; i++) {
      await s.saveEdits('run1', 0, [clip('u1', 2, 4)]);
      await s.revertEdits('run1');
    }
    expect((await s.getRun('run1'))!.prev_edits).toHaveLength(5); // the newest five
  });

  it('edits made against other segments are flagged (edits_base_ok false)', async () => {
    const s = await fresh();
    const rec = await s.putRun(run());
    await s.saveEdits('run1', 0, [clip('u1', 1, 3)]);
    await s.backend.put('runs', 'run1', { ...(await s.getRun('run1')), sha: rec.sha.replace(/^./, (c) => (c === '0' ? '1' : '0')) });
    expect((await s.run('run1')).edits_base_ok).toBe(false);
    expect((await s.runs('m1'))[0].edits_base_ok).toBe(false);
  });

  it('an unknown run is a 404', async () => {
    const s = await fresh();
    for (const f of [() => s.run('nope'), () => s.samples('nope'), () => s.saveEdits('nope', 0, []), () => s.revertEdits('nope'), () => s.resegment('nope', {})]) {
      await expect(f()).rejects.toMatchObject({ status: 404 });
    }
  });
});

describe('resegment equals the golden build_segments', () => {
  const cases = golden.resegment.filter((c) => c.expect);
  it('has golden cases to compare with', () => expect(cases.length).toBeGreaterThanOrEqual(5));
  for (const c of cases) {
    it(`${c.name}`, async () => {
      const s = await fresh();
      await s.putRun(run({
        run_id: 'g', segments: { settings: c.settings, warnings: [], clips: [], presence_intervals: [] },
        samples: { t: c.t, score: c.score, box: c.t.map(() => null), track: null, enter: 0.5, exit: 0.3, merge_gap: 1, min_len: 1, pad: 1, sound: null, mode: 'detect' },
        source: { offset: c.offset, end_time: c.end },
      }));
      const params = Object.fromEntries(Object.entries(c.params).filter(([, v]) => v !== null)) as Record<string, number>;
      const r = await s.resegment('g', params);
      near(r.presence, c.expect!.presence);
      near(r.clips, c.expect!.clips);
      expect(r.same_as_run).toBe(c.expect!.clips.length === 0); // the run holds no clips of its own here
    });
  }
  it('with no fields the preview is the run itself, and same_as_run says so (the run keeps its own numbers)', async () => {
    const s = await fresh();
    const base = run();
    await s.putRun(base);
    const r = await s.resegment('run1', {});
    expect(r.same_as_run).toBe(true);
    expect(r.clips).toEqual([[8.75, 20.75], [38.75, 50.75]]);
    expect(r.presence).toEqual([[9.75, 19.75], [39.75, 49.75]]);
    const wider = await s.resegment('run1', { pad: 2 });
    expect(wider.same_as_run).toBe(false);
    expect(wider.clips[0]).toEqual([7.75, 21.75]);
  });
  it('refuses what the server refuses (422)', async () => {
    const s = await fresh();
    await s.putRun(run());
    await expect(s.resegment('run1', { exit: 0.9 })).rejects.toMatchObject({ status: 422, message: expect.stringMatching(/must not exceed/) });
    await expect(s.resegment('run1', { pad: -1 })).rejects.toMatchObject({ status: 422 });
  });
});

describe('persistence and fallbacks', () => {
  const media: MediaRecord = {
    id: 'm1', path: 'a.mov', name: 'a.mov', size: 10, mtime: 1, codec: 'hevc', profile: 'Main', pix_fmt: null, width: 1280, height: 720, display_w: 720, display_h: 1280,
    rotation: 90, mirrored: false, fps: 30, capture_fps: 30, slowmo_suggested: false, duration: 10, offset: 0, end_time: 10, has_audio: true, keyframes: [0, 1], opened: 5,
  };
  it('data written through one connection is read by a new one on the same database; recent is newest first and clearable', async () => {
    const factory = new IDBFactory();
    const a = new BrowserStore(await IdbBackend.open(factory));
    await a.putMedia(media);
    await a.putMedia({ ...media, id: 'm2', name: 'b.mov', opened: 9 });
    await a.putRun(run());
    await a.putPick('m1', 'person', { target_key: 'k', taps: [], zone: null, file: null, class: 'person', candidates: [] });
    const b = new BrowserStore(await IdbBackend.open(factory));
    expect((await b.recent()).map((r) => r.name)).toEqual(['b.mov', 'a.mov']);
    expect((await b.getMedia('m1'))!.rotation).toBe(90);
    expect((await b.runs('m1')).map((r) => r.run_id)).toEqual(['run1']);
    expect((await b.getPick('m1', 'person'))!.target_key).toBe('k');
    await b.deletePick('m1', 'person');
    expect(await b.getPick('m1', 'person')).toBeUndefined();
    await b.clearRecent();
    expect(await b.recent()).toEqual([]);
    // clearing removes the file names, the media facts, the runs and the pick state from storage, not just from the list
    expect(await b.getMedia('m1')).toBeUndefined();
    expect(await b.getMedia('m2')).toBeUndefined();
    expect(await b.runs('m1')).toEqual([]);
    expect(await b.getPick('m1', 'person')).toBeUndefined();
    expect(await b.backend.all('media')).toEqual([]);
    expect(await b.backend.all('runs')).toEqual([]);
    expect(await b.backend.all('pick')).toEqual([]);
  });
  it('clearRecent leaves no file name anywhere in the database, also from a record already marked cleared', async () => {
    const factory = new IDBFactory();
    const idb = await IdbBackend.open(factory);
    const s = new BrowserStore(idb);
    await s.putMedia({ ...media, name: 'secret-name.mov' });
    await s.putMedia({ ...media, id: 'old', name: 'cleared-before.mov', opened: 0 });
    await s.putRun(run());
    await s.clearRecent();
    const dump = JSON.stringify([await idb.all('media'), await idb.all('runs'), await idb.all('pick')]);
    expect(dump).not.toContain('secret-name');
    expect(dump).not.toContain('cleared-before');
    expect(dump).toBe('[[],[],[]]');
    // and the store keeps working afterwards
    await s.putMedia({ ...media, id: 'm3', name: 'new.mov', opened: 20 });
    expect((await s.recent()).map((r) => r.name)).toEqual(['new.mov']);
  });
  it('falls back to memory when IndexedDB is missing or refuses to open, with the same edit rules', async () => {
    const none = await openBackend(undefined);
    expect(none.persistent).toBe(false);
    const broken = { open: () => { throw new Error('blocked'); } } as unknown as IDBFactory;
    const b = await openBackend(broken);
    expect(b.persistent).toBe(false);
    const s = new BrowserStore(b.backend, false);
    await s.putRun(run());
    await s.saveEdits('run1', 0, [clip('u1', 1, 3)]);
    await expect(s.saveEdits('run1', 0, [clip('u1', 1, 3)])).rejects.toBeInstanceOf(EditConflict);
    const m: Backend = new MemoryBackend();
    await m.put('media', 'x', { a: 1 });
    expect(await m.get('media', 'x')).toEqual({ a: 1 });
  });
});

describe('BrowserEngine over the store', () => {
  it('reads and writes runs and edits through the Engine interface; recent comes from the media it opened', async () => {
    const store = await fresh();
    const e = new BrowserEngine({ store });
    await store.putRun(run());
    expect((await e.runs('m1')).map((r) => r.run_id)).toEqual(['run1']);
    expect((await e.run('run1')).segments.clips).toHaveLength(2);
    expect((await e.samples('run1')).t).toHaveLength(120);
    const d = await e.saveEdits('run1', 0, [clip('u1', 1, 3, 'added')]);
    await expect(e.saveEdits('run1', 0, [])).rejects.toBeInstanceOf(EditConflict);
    expect((await e.run('run1')).edits).toEqual(d);
    await e.revertEdits('run1');
    expect((await e.run('run1')).edits).toBeNull();
    expect((await e.resegment('run1', { pad: 2 })).clips[0]).toEqual([7.75, 21.75]);
    expect(await e.recent()).toEqual([]);
    expect(await e.clearRecent()).toEqual([]);
  });

  it('opens a File through the demux, remembers the facts, and a media id comes back from storage after a reload', async () => {
    const store = await fresh();
    const info: Partial<MediaInfo> = { codec: 'hevc' };
    void info;
    const fake = {
      codec: 'hevc', profile: 'Main', video: { trackId: 1, codecString: 'hvc1.1.6.L93.90', fourcc: 'hvc1', description: null, width: 1280, height: 720, timescale: 15360 },
      samples: { n: 0 }, keyframes: [0, 1.0004], rotation: 90, mirrored: false, sar: 1, fps: 30, captureFps: 30, duration: 10, offset: 0, endTime: 10,
      formatStart: 0, hasAudio: true, audio: null, warnings: [], stats: { chunkBytes: 1, reads: 1, bytesRead: 1, peakHeldBytes: 1 },
    } as never;
    const urls: string[] = [];
    const e = new BrowserEngine({ store, demux: async () => fake, fileUrl: () => { urls.push('blob:one'); return 'blob:one'; }, revokeFileUrl: () => {} });
    const file = new File([new Uint8Array(16)], 'IMG.MOV', { lastModified: 1_700_000_000_000 });
    const m = await e.openFile(file);
    expect(m).toMatchObject({ name: 'IMG.MOV', size: 16, codec: 'hevc', rotation: 90, display_w: 720, display_h: 1280, keyframes: [0, 1], has_audio: true, video_url: 'blob:one', mtime: 1_700_000_000 });
    expect(m.id).toMatch(/^[0-9a-f]{12}$/);
    expect((await e.openFile(file)).id).toBe(m.id); // the same file is the same media
    expect(e.videoUrl(m.id)).toBe('blob:one');
    expect((await e.recent())[0]).toMatchObject({ path: m.id, name: 'IMG.MOV' });
    await expect(e.openPath(m.id)).resolves.toMatchObject({ id: m.id });

    const later = new BrowserEngine({ store }); // a reload: the facts are there, the file is not
    expect(await later.media(m.id)).toMatchObject({ id: m.id, video_url: '', rotation: 90, name: 'IMG.MOV' });
    expect(later.videoUrl(m.id)).toBe('');
    await expect(later.openPath(m.id)).rejects.toMatchObject({ status: 404 });
    await expect(later.filmstrip(m.id)).rejects.toMatchObject({ status: 409 });
    await expect(later.media('nope')).rejects.toMatchObject({ status: 404 });
  });

  it('Clear recent keeps only the video the UI has open: an earlier file of this tab loses its name, runs and picks', async () => {
    const store = await fresh();
    const fake = {
      codec: 'h264', profile: 'Main', video: { trackId: 1, codecString: 'avc1.4d401f', fourcc: 'avc1', description: null, width: 1280, height: 720, timescale: 15360 },
      samples: { n: 0 }, keyframes: [0], rotation: 0, mirrored: false, sar: 1, fps: 30, captureFps: 30, duration: 10, offset: 0, endTime: 10,
      formatStart: 0, hasAudio: false, audio: null, warnings: [], stats: { chunkBytes: 1, reads: 1, bytesRead: 1, peakHeldBytes: 1 },
    } as never;
    const e = new BrowserEngine({ store, demux: async () => fake, fileUrl: () => 'blob:one', revokeFileUrl: () => {} });
    const first = await e.openFile(new File([new Uint8Array(16)], 'first-private-name.mov', { lastModified: 1_700_000_000_000 }));
    const second = await e.openFile(new File([new Uint8Array(17)], 'second-name.mov', { lastModified: 1_700_000_000_000 }));
    await store.putRun(run({ run_id: 'r-first', media_id: first.id, slug: 'first-private-name-any-person' }));
    await store.putRun(run({ run_id: 'r-second', media_id: second.id }));
    const pick = { target_key: 'k', taps: [], zone: [0, 0, 1, 1], file: null, class: 'person', candidates: [] };
    await store.putPick(first.id, 'person', pick as never);
    await store.putPick(second.id, 'person', pick as never);
    expect(await e.clearRecent()).toEqual([]);
    expect(await store.getMedia(first.id)).toBeUndefined();
    expect(await store.getRun('r-first')).toBeUndefined();
    expect(await store.getPick(first.id, 'person')).toBeUndefined();
    expect(await store.getMedia(second.id)).toMatchObject({ name: 'second-name.mov' });
    expect(await store.getRun('r-second')).toBeDefined();
    expect(await store.getPick(second.id, 'person')).toEqual(pick);
    // going back to the first file (still held by this tab) makes it the open one
    await e.openPath(first.id);
    await e.clearRecent();
    expect(await store.getMedia(second.id)).toBeUndefined();
  });

  it('Clear recent while a video is open keeps that video, its runs and its edits (saving and exporting still work), and deletes the rest', async () => {
    const store = await fresh();
    const fake = {
      codec: 'h264', profile: 'Main', video: { trackId: 1, codecString: 'avc1.4d401f', fourcc: 'avc1', description: null, width: 1280, height: 720, timescale: 15360 },
      samples: { n: 0 }, keyframes: [0], rotation: 0, mirrored: false, sar: 1, fps: 30, captureFps: 30, duration: 10, offset: 0, endTime: 10,
      formatStart: 0, hasAudio: false, audio: null, warnings: [], stats: { chunkBytes: 1, reads: 1, bytesRead: 1, peakHeldBytes: 1 },
    } as never;
    const e = new BrowserEngine({ store, demux: async () => fake, fileUrl: () => 'blob:one', revokeFileUrl: () => {} });
    const open = await e.openFile(new File([new Uint8Array(16)], 'open-name.mov', { lastModified: 1_700_000_000_000 }));
    await store.putMedia({ ...(open as object), id: 'other', name: 'other-name.mov', opened: 5 } as never);
    await store.putRun(run({ run_id: 'r-open', media_id: open.id }));
    await store.putRun(run({ run_id: 'r-other', media_id: 'other' }));
    await store.saveEdits('r-open', 0, [clip('u1', 1, 3, 'added')]);
    const pick = { target_key: 'k', taps: [], zone: [0, 0, 1, 1], file: null, class: 'person', candidates: [] };
    await store.putPick(open.id, 'person', pick as never);
    await store.putPick('other', 'person', pick as never);
    expect(await e.clearRecent()).toEqual([]);
    expect(await e.recent()).toEqual([]);
    // the open video keeps working: a second save, and its run and facts are still there
    const d = await e.saveEdits('r-open', 1, [clip('u1', 1, 4, 'added')]);
    expect(d.revision).toBe(2);
    expect((await e.runs(open.id)).map((r) => r.run_id)).toEqual(['r-open']);
    expect(await e.media(open.id)).toMatchObject({ name: 'open-name.mov' });
    // everything else is gone
    expect(await store.getPick(open.id, 'person')).toEqual(pick);
    expect(await store.getPick('other', 'person')).toBeUndefined();
    expect(await store.getMedia('other')).toBeUndefined();
    expect(await e.runs('other')).toEqual([]);
  });
});
