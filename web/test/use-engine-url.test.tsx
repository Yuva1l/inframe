// useEngineUrl (src/ui/useEngineUrl.ts): a LocalEngine URL passes through unchanged; a BrowserEngine key resolves to a blob URL.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { BlobUrlCache, makeKey, parseKey } from '../src/browser/urls';
import { BrowserEngine } from '../src/engine/browser';
import { LocalEngine } from '../src/engine/local';
import { MockEngine } from '../src/engine/mock';
import type { Engine } from '../src/engine/types';
import { useEngineUrl, useEngineUrls } from '../src/ui/useEngineUrl';

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
let root: HTMLElement;
beforeEach(() => {
  root = document.createElement('div');
  document.body.appendChild(root);
});
afterEach(() => {
  render(null, root);
  root.remove();
});

function Probe({ engine, url }: { engine: Engine; url: string | null }) {
  const src = useEngineUrl(engine, url);
  return <img data-testid="img" src={src} />;
}
function Many({ engine, urls }: { engine: Engine; urls: string[] }) {
  const ready = useEngineUrls(engine, urls);
  return <ul>{urls.map((u) => <li key={u} data-u={u}>{ready(u) ?? 'pending'}</li>)}</ul>;
}
/** wait (up to 2 s) until the img has a src that matches */
async function srcIs(re: RegExp) {
  for (let i = 0; i < 100; i++) {
    await tick(20);
    if (re.test(img().getAttribute('src') ?? '')) return;
  }
}
const img = () => root.querySelector('img') as HTMLImageElement;

/** a BrowserEngine whose pictures are rendered by a stand-in (no WebCodecs here) and whose blob URLs are counted */
function browserEngine() {
  const made: Blob[] = [];
  const revoked: string[] = [];
  let n = 0;
  const urls = new BlobUrlCache(3, { create: (b) => { made.push(b); return `blob:test/${n++}`; }, revoke: (u) => revoked.push(u) });
  const rendered: string[] = [];
  const engine = new BrowserEngine({
    urls,
    demux: async () => ({
      codec: 'h264', profile: null, video: { trackId: 1, codecString: 'avc1', fourcc: 'avc1', description: null, width: 640, height: 360, timescale: 1000 },
      samples: { n: 0 }, keyframes: [], rotation: 0, mirrored: false, sar: 1, fps: 30, captureFps: 30, duration: 5, offset: 0, endTime: 5, formatStart: 0,
      hasAudio: false, audio: null, warnings: [], stats: { chunkBytes: 1, reads: 1, bytesRead: 1, peakHeldBytes: 1 },
    }) as never,
    fileUrl: () => 'blob:video',
    renderFrame: async (_v, t, w) => {
      rendered.push(`${t}@${w}`);
      return new Blob([`frame ${t}`], { type: 'image/jpeg' });
    },
    makeFilmstrip: () => ({
      state: 'done', error: null, start() {}, sheet: async (k: number) => new Blob([`sheet ${k}`]),
      filmstrip: () => ({ version: 1, cell_w: 192, cell_h: 108, cols: 10, rows: 10, cells: [{ t: 0, sheet: 0, i: 0 }], sheets: 1, done: true }),
    }),
  });
  return { engine, made, revoked, rendered };
}

describe('keys', () => {
  it('round-trip, and anything else is not a key', () => {
    const k = makeKey('frame', 'abc123', '12.500', 320);
    expect(k).toBe('inframe-blob:frame/abc123/12.500/320');
    expect(parseKey(k)).toEqual({ kind: 'frame', parts: ['abc123', '12.500', '320'] });
    expect(parseKey(makeKey('sheet', 'a/b', 2))?.parts).toEqual(['a/b', '2']);
    for (const u of ['blob:http://x/1', 'api/media/x/video', 'http://a/b.jpg', 'inframe-blob:', '']) expect(parseKey(u)).toBeNull();
  });
});

describe('useEngineUrl', () => {
  it('a LocalEngine URL passes through unchanged, on the first render', async () => {
    const local = new LocalEngine({ token: 't', mediaKey: 'k', fetch: (async () => new Response('{}')) as typeof fetch });
    expect((local as Engine).resolveUrl).toBeUndefined();
    const url = local.frameUrl('m1', 12.5, 320);
    expect(url).toContain('api/media/m1/frame');
    render(<Probe engine={local} url={url} />, root);
    expect(img().getAttribute('src')).toBe(url); // before any tick: no state, no flicker
    await tick(5);
    expect(img().getAttribute('src')).toBe(url);
  });

  it('the MockEngine (no resolveUrl) passes through too', async () => {
    const mock = new MockEngine();
    const url = mock.sheetUrl('m', 0);
    render(<Probe engine={mock} url={url} />, root);
    expect(img().getAttribute('src')).toBe(url);
  });

  it('a BrowserEngine key resolves to a blob URL, rendered once and cached', async () => {
    const { engine, rendered } = browserEngine();
    const m = await engine.openFile(new File([new Uint8Array(4)], 'a.mp4'));
    const key = engine.frameUrl(m.id, 12.5, 320);
    expect(key.startsWith('inframe-blob:')).toBe(true);
    render(<Probe engine={engine} url={key} />, root);
    expect(img().getAttribute('src')).toBeNull(); // pending: nothing to load yet
    await srcIs(/^blob:test\/0$/);
    expect(img().getAttribute('src')).toBe('blob:test/0');
    render(<Probe engine={engine} url={null} />, root);
    render(<Probe engine={engine} url={key} />, root);
    await srcIs(/^blob:test\/0$/);
    expect(img().getAttribute('src')).toBe('blob:test/0');
    expect(rendered).toEqual(['12.5@320']); // one render for two uses
  });

  it('a sheet key waits for the filmstrip sheet; a URL that is not a key passes through even on the BrowserEngine', async () => {
    const { engine } = browserEngine();
    const m = await engine.openFile(new File([new Uint8Array(4)], 'a.mp4'));
    await engine.filmstrip(m.id);
    render(<Probe engine={engine} url={engine.sheetUrl(m.id, 0)} />, root);
    await srcIs(/^blob:test\//);
    expect(img().getAttribute('src')).toMatch(/^blob:test\//);
    render(<Probe engine={engine} url="blob:video" />, root);
    expect(img().getAttribute('src')).toBe('blob:video');
    expect(await engine.resolveUrl('api/media/x/video')).toBe('api/media/x/video');
  });

  it('a key for a video that is not open ends as nothing, not as a crash', async () => {
    const { engine } = browserEngine();
    render(<Probe engine={engine} url={engine.frameUrl('nosuchvideo', 1, 320)} />, root);
    await tick(5);
    expect(img().getAttribute('src')).toBeNull();
    await expect(engine.resolveUrl(engine.frameUrl('nosuchvideo', 1, 320))).rejects.toMatchObject({ status: 409 });
  });

  it('follows the url when it changes', async () => {
    const { engine } = browserEngine();
    const m = await engine.openFile(new File([new Uint8Array(4)], 'a.mp4'));
    render(<Probe engine={engine} url={engine.frameUrl(m.id, 1, 320)} />, root);
    await srcIs(/^blob:test\//);
    const first = img().getAttribute('src');
    expect(first).toMatch(/^blob:test\//);
    render(<Probe engine={engine} url={engine.frameUrl(m.id, 2, 320)} />, root);
    await srcIs(/^blob:test\/[1-9]/);
    expect(img().getAttribute('src')).not.toBe(first);
    expect(img().getAttribute('src')).toMatch(/^blob:test\//);
  });
});

describe('useEngineUrls', () => {
  it('resolves a list: keys become blob URLs, other URLs stay', async () => {
    const { engine } = browserEngine();
    const m = await engine.openFile(new File([new Uint8Array(4)], 'a.mp4'));
    await engine.filmstrip(m.id);
    const urls = [engine.sheetUrl(m.id, 0), 'blob:other'];
    render(<Many engine={engine} urls={urls} />, root);
    expect(root.querySelector('[data-u="blob:other"]')!.textContent).toBe('blob:other');
    expect(root.querySelector('li')!.textContent).toBe('pending');
    for (let i = 0; i < 100 && root.querySelector('li')!.textContent === 'pending'; i++) await tick(20);
    expect(root.querySelector('li')!.textContent).toMatch(/^blob:test\//);
  });
});

describe('useEngineUrls holds', () => {
  it('asks again for a picture it still shows when the list changes, so the engine can keep its URL', async () => {
    const asked: { url: string; signal: AbortSignal }[] = [];
    const engine = { resolveUrl: async (url: string, signal: AbortSignal) => (asked.push({ url, signal }), `blob:${url}`) } as unknown as Engine;
    const a = 'inframe-blob:sheet/1/0';
    const b = 'inframe-blob:sheet/1/1';
    render(<Many engine={engine} urls={[a]} />, root);
    for (let i = 0; i < 50 && root.querySelector('li')!.textContent === 'pending'; i++) await tick(10);
    expect(root.querySelector('li')!.textContent).toBe(`blob:${a}`); // shown
    render(<Many engine={engine} urls={[a, b]} />, root);
    for (let i = 0; i < 50 && root.querySelectorAll('li')[1]?.textContent !== `blob:${b}`; i++) await tick(10);
    const live = asked.filter((x) => !x.signal.aborted).map((x) => x.url).sort();
    expect(live).toEqual([a, b]); // a's first request ended with the first list; the second is live
    expect(root.querySelector('li')!.textContent).toBe(`blob:${a}`);
  });
});

describe('BlobUrlCache', () => {
  it('keeps the newest URLs and revokes the oldest; two askers share one render; clear revokes all', async () => {
    const revoked: string[] = [];
    let n = 0;
    const c = new BlobUrlCache(2, { create: () => `blob:${n++}`, revoke: (u) => revoked.push(u) });
    let renders = 0;
    const make = (s: string) => async () => (renders++, new Blob([s]));
    const [a1, a2] = await Promise.all([c.get('a', make('a')), c.get('a', make('a'))]);
    expect(a1).toBe(a2);
    expect(renders).toBe(1);
    await c.get('b', make('b'));
    await c.get('a', make('a')); // touch a: b is now the oldest
    await c.get('c', make('c'));
    expect(revoked).toEqual(['blob:1']);
    expect(c.has('b')).toBe(false);
    expect(c.has('a')).toBe(true);
    c.clear();
    expect(revoked).toHaveLength(3);
    expect(c.size).toBe(0);
  });
});
