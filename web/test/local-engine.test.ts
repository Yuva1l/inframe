import { describe, expect, it, vi } from 'vitest';
import { LocalEngine, NOT_RUNNING, OLDER_SERVE } from '../src/engine/local';
import { EditConflict, EngineError } from '../src/engine/types';

const TOKEN = 'tok-SESSION-1234567890';
const MKEY = 'key-MEDIA-1234567890';

interface Call { url: string; method: string; headers: Record<string, string>; body?: string }

function fakeFetch(handler: (c: Call) => { status?: number; body?: unknown } | Promise<{ status?: number; body?: unknown }>) {
  const calls: Call[] = [];
  const f = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    const c: Call = { url: String(url), method: init?.method ?? 'GET', headers: { ...(init?.headers as Record<string, string>) }, body: init?.body as string | undefined };
    calls.push(c);
    const r = await handler(c);
    const status = r.status ?? 200;
    return new Response(JSON.stringify(r.body ?? {}), { status, headers: { 'Content-Type': 'application/json' } });
  });
  return { f: f as unknown as typeof fetch, calls };
}

const memory = () => {
  const m = new Map<string, string>();
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v), m };
};

describe('connect (nonce claim)', () => {
  it('trades the fragment nonce for the keys, stores them and removes the fragment', async () => {
    const { f, calls } = fakeFetch(() => ({ body: { token: TOKEN, media_key: MKEY } }));
    const storage = memory();
    const replaced: string[] = [];
    const e = await LocalEngine.connect({ fetch: f, storage, hash: '#n=abc_DEF-123', replaceUrl: (u) => replaced.push(u) });
    expect(e).toBeInstanceOf(LocalEngine);
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe('./api/session/claim');
    expect(calls[0].method).toBe('POST');
    expect(JSON.parse(calls[0].body!)).toEqual({ nonce: 'abc_DEF-123' });
    expect(calls[0].headers['X-Inframe-Token']).toBeUndefined(); // the claim carries no token
    expect(JSON.parse(storage.m.get('inframe.session')!)).toEqual({ token: TOKEN, media_key: MKEY });
    expect(replaced).toHaveLength(1);
    expect(replaced[0]).not.toContain('#');
  });
  it('reads the keys back after a reload (no fragment)', async () => {
    const storage = memory();
    storage.setItem('inframe.session', JSON.stringify({ token: TOKEN, media_key: MKEY }));
    const { f, calls } = fakeFetch(() => ({ body: {} }));
    const e = await LocalEngine.connect({ fetch: f, storage, hash: '' });
    expect(e).toBeInstanceOf(LocalEngine);
    expect(calls).toHaveLength(0);
  });
  it('has nothing to offer with neither a nonce nor stored keys, or with a storage that throws', async () => {
    const { f } = fakeFetch(() => ({}));
    expect(await LocalEngine.connect({ fetch: f, storage: memory(), hash: '' })).toBeNull();
    const broken = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); } };
    expect(await LocalEngine.connect({ fetch: f, storage: broken, hash: '' })).toBeNull();
    const ok = fakeFetch(() => ({ body: { token: TOKEN, media_key: MKEY } }));
    expect(await LocalEngine.connect({ fetch: ok.f, storage: broken, hash: '#n=x', replaceUrl: () => {} })).toBeInstanceOf(LocalEngine);
  });
  it('shows the server’s message when the link was used', async () => {
    const { f } = fakeFetch(() => ({ status: 403, body: { error: 'This link was used or has expired. Press Enter in the terminal for a new one.' } }));
    await expect(LocalEngine.connect({ fetch: f, storage: memory(), hash: '#n=x', replaceUrl: () => {} })).rejects.toMatchObject({
      status: 403, message: expect.stringContaining('used or has expired'),
    });
  });
});

function engine(handler: Parameters<typeof fakeFetch>[0], onDown?: (d: boolean) => void) {
  const { f, calls } = fakeFetch(handler);
  return { e: new LocalEngine({ token: TOKEN, mediaKey: MKEY, fetch: f, onDown }), calls };
}

describe('API calls', () => {
  it('send the session token in the header and nowhere else', async () => {
    const { e, calls } = engine(() => ({ body: { recent: [], runs: [], job: null } }));
    await e.recent();
    await e.runs('abc');
    await e.currentJob();
    await e.openPath('/Users/me/a.mov');
    expect(calls.map((c) => c.url)).toEqual(['./api/recent', './api/media/abc/runs', './api/jobs/current', './api/media/open']);
    for (const c of calls) {
      expect(c.headers['X-Inframe-Token']).toBe(TOKEN);
      expect(c.url).not.toContain(TOKEN);
      expect(c.url).not.toContain('k=');
    }
    expect(JSON.parse(calls[3].body!)).toEqual({ path: '/Users/me/a.mov' });
    expect(calls[3].headers['Content-Type']).toBe('application/json');
  });
  it('answer a cancelled dialog with null', async () => {
    const { e } = engine(() => ({ body: { cancelled: true } }));
    expect(await e.openDialog()).toBeNull();
    expect(await e.openFromPhotos()).toBeNull();
    expect(await e.choosePhotos('dialog')).toBeNull();
  });
  it('turn a 409 on saveEdits into EditConflict with the stored document', async () => {
    const stored = { revision: 8, clips: [], next_id: 1, format: 1, base: {}, updated: '' };
    const { e, calls } = engine(() => ({ status: 409, body: { error: 'conflict', edits: stored } }));
    const err = await e.saveEdits('r1', 7, [{ id: 'd1', start: 1, end: 2, origin: 'detector' }]).catch((x) => x);
    expect(err).toBeInstanceOf(EditConflict);
    expect(err).toBeInstanceOf(EngineError);
    expect(err.stored).toEqual(stored);
    expect(calls[0].method).toBe('PUT');
    expect(JSON.parse(calls[0].body!)).toEqual({ revision: 7, clips: [{ id: 'd1', start: 1, end: 2, origin: 'detector' }] });
  });
  it('turn a 403 into "older inframe serve"', async () => {
    for (const body of [{ error: OLDER_SERVE }, { error: 'forbidden' }, {}]) {
      const { e } = engine(() => ({ status: 403, body }));
      await expect(e.session()).rejects.toMatchObject({ status: 403, message: OLDER_SERVE });
    }
    expect(OLDER_SERVE).toContain('older inframe serve');
  });
  it('keeps the server’s own message for other errors, 403 for a refused path included', async () => {
    const { e } = engine(() => ({ status: 403, body: { error: 'inframe serve does not open files there.' } }));
    await expect(e.openPath('/x')).rejects.toMatchObject({ status: 403, message: 'inframe serve does not open files there.' });
    const n = engine(() => ({ status: 404, body: { error: 'No such file: /x' } }));
    await expect(n.e.openPath('/x')).rejects.toMatchObject({ status: 404, message: 'No such file: /x' });
  });
  it('treats a missing export and an unmatched drop as null', async () => {
    const { e } = engine(() => ({ status: 404, body: { error: 'none', matches: 2 } }));
    expect(await e.exportResult('r')).toBeNull();
    expect(await e.locate({ name: 'a.mov', size: 1, lastModified: 2 })).toBeNull();
  });
  it('reports a stopped server and tells the page', async () => {
    const down: boolean[] = [];
    const e = new LocalEngine({ token: TOKEN, mediaKey: MKEY, fetch: (async () => { throw new TypeError('Failed to fetch'); }) as typeof fetch, onDown: (d) => down.push(d) });
    await expect(e.recent()).rejects.toMatchObject({ status: 0, message: NOT_RUNNING });
    expect(down).toEqual([true]);
  });
  it('passes the abort signal on and lets an abort through', async () => {
    const ctl = new AbortController();
    const f = vi.fn(async (_u: unknown, init?: RequestInit) => {
      expect(init?.signal).toBe(ctl.signal);
      throw new DOMException('aborted', 'AbortError');
    });
    const e = new LocalEngine({ token: TOKEN, mediaKey: MKEY, fetch: f as unknown as typeof fetch });
    await expect(e.boxes('m', 1.5, 'person', ctl.signal)).rejects.toMatchObject({ name: 'AbortError' });
  });
});

describe('media URLs', () => {
  const e = new LocalEngine({ token: TOKEN, mediaKey: MKEY, fetch: (async () => new Response('{}')) as typeof fetch });
  it('carry the media key as ?m= and never the session token', () => {
    const urls = [
      e.videoUrl('abc'), e.videoUrl('abc', { asMp4: true }), e.sheetUrl('abc', 3), e.frameUrl('abc', 12.34567, 320), e.frameUrl('abc', 1),
      e.candidateUrl('abc', 2), e.personUrl('abc', 3), e.pickFrameUrl('abc', 12.3456, 'dog'), e.targetCropUrl('alice', 0), e.runFileUrl('r1', 'clip 1.mp4'), e.musicUrl('trk-1'),
    ];
    for (const u of urls) {
      expect(u).toMatch(/[?&]m=key-MEDIA-1234567890$/);
      expect(u).not.toContain(TOKEN);
      expect(u.startsWith('./api/')).toBe(true);
    }
    expect(e.videoUrl('abc', { asMp4: true })).toBe(`./api/media/abc/video?as=mp4&m=${MKEY}`);
    expect(e.frameUrl('abc', 12.34567, 320)).toBe(`./api/media/abc/frame.jpg?t=12.346&w=320&m=${MKEY}`);
    expect(e.pickFrameUrl('abc', 12.3456, 'dog')).toBe(`./api/media/abc/pickframe.jpg?t=12.346&cls=dog&m=${MKEY}`);
    expect(e.personUrl('abc', 3)).toBe(`./api/media/abc/people/3.jpg?m=${MKEY}`);
    expect(e.runFileUrl('r1', 'clip 1.mp4')).toBe(`./api/runs/r1/files/clip%201.mp4?m=${MKEY}`);
  });
  it('key the preview copy’s URL the server sends', async () => {
    const { e: eng } = engine(() => ({ body: { state: 'done', fraction: 1, url: 'api/media/abc/proxy.mp4', first_frame_file_time: 0.007, error: null } }));
    const t = await eng.mediaTask('abc', 'proxy');
    expect(t.url).toBe(`./api/media/abc/proxy.mp4?m=${MKEY}`);
    expect(t.first_frame_file_time).toBe(0.007);
  });
});

describe('the panels\' calls', () => {
  it('searches music with the mood and the numbers only, and returns the notice with the tracks', async () => {
    const { e, calls } = engine(() => ({ body: { tracks: [{ id: 't1' }], notice: 'Found through Openverse' } }));
    const r = await e.musicSearch({ mood: 'upbeat', min_s: 90, limit: 10, cc0_only: false });
    expect(r.notice).toBe('Found through Openverse');
    expect(r.tracks).toHaveLength(1);
    expect(calls[0].url).toBe('./api/music/search?mood=upbeat&cc0_only=0&min_s=90&limit=10');
    expect(calls[0].headers['X-Inframe-Token']).toBe(TOKEN);
  });
  it('lists the included songs from the server with the CC0 choice and no search text', async () => {
    const { e, calls } = engine(() => ({ body: { tracks: [{ id: 'b1', bundled: true }], notice: 'These songs come with inframe' } }));
    const r = await e.musicLibrary(false);
    expect(r.tracks).toHaveLength(1);
    expect(calls[0]).toMatchObject({ method: 'GET', url: './api/music/library?cc0_only=0' });
    await e.musicLibrary(true);
    expect(calls[1].url).toBe('./api/music/library?cc0_only=1');
  });
  it('sends the CC0 only choice to the server as cc0_only, for the search and for the download', async () => {
    const { e, calls } = engine(() => ({ body: { tracks: [], notice: '' } }));
    await e.musicSearch({ mood: 'calm', cc0_only: true });
    expect(calls[0].url).toBe('./api/music/search?mood=calm&cc0_only=1');
    await e.musicGet('trk1', true);
    expect(JSON.parse(String(calls[1].body))).toEqual({ id: 'trk1', cc0_only: true });
    await e.musicGet('trk1', false);
    expect(JSON.parse(String(calls[2].body))).toEqual({ id: 'trk1', cc0_only: false });
  });
  it('answers a cancelled music dialog with null and a choice with its file', async () => {
    expect(await engine(() => ({ body: { cancelled: true } })).e.musicChooseFile()).toBeNull();
    const { e, calls } = engine(() => ({ body: { file: '/m/a.mp3', license: null } }));
    expect(await e.musicChooseFile()).toEqual({ file: '/m/a.mp3', license: null });
    expect(calls[0]).toMatchObject({ method: 'POST', url: './api/music/choose-file' });
  });
  it('starts a run and an export with the request as JSON, and sees a 409 exists with the stored export', async () => {
    const { e, calls } = engine((c) => (c.url.endsWith('/export') ? { status: 409, body: { error: 'exists', export: { clips: [1] } } } : { body: { job: { id: 'j1' }, run_id: 'r1' } }));
    const ref = await e.startRun({ media_id: 'm', target: { kind: 'any' }, zone: null, zone_overlap: null, preset: 'balanced', slowmo: 'off', slowmo_audio: false, advanced: {} });
    expect(ref.run_id).toBe('r1');
    expect(calls[0].url).toBe('./api/runs');
    await expect(e.startExport('r1', { clip_ids: null, out_dir: null, replace: false, clips: true, reel: false, music: null, beat_sync: false, edit_lists: [] }))
      .rejects.toMatchObject({ status: 409, extra: { error: 'exists', export: { clips: [1] } } });
  });
});
