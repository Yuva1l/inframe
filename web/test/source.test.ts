// Which source the player ends up with: the original, ?as=mp4 for a .mov, then a preview copy; and the clock offset it measured.
import { afterEach, describe, expect, it } from 'vitest';
import type { PlayableFn } from '../src/engine/playability';
import { MockEngine } from '../src/engine/mock';
import { createStore, type Store } from '../src/state';

let store: Store;
afterEach(() => store.dispose());

async function open(engine: MockEngine, playable: PlayableFn) {
  store = createStore(engine, { playable, toastMs: 0 });
  await store.init();
  await store.openMedia(engine.media_);
  store.attachVideo(document.createElement('video'));
  await new Promise((r) => setTimeout(r, 20));
}

describe('prepareSource', () => {
  it('uses the original when the browser plays it, with the measured clock offset', async () => {
    const tried: string[] = [];
    await open(new MockEngine({ offset: 2 }), async (_el, url, first) => { tried.push(`${url} ${first}`); return { ok: true, clockOffset: first - 0 }; });
    expect(tried).toEqual(['mock://video/mock00000001 2']);
    expect(store.playSrc.value?.kind).toBe('original');
    expect(store.clockOffset.value).toBe(2);
    expect(store.banner.value).toBeNull();
  });
  it('retries a .mov as mp4 before asking for a preview copy', async () => {
    const tried: string[] = [];
    await open(new MockEngine({ name: 'a.mov' }), async (_el, url) => { tried.push(url); return { ok: url.includes('as=mp4'), clockOffset: 0 }; });
    expect(tried).toEqual(['mock://video/mock00000001', 'mock://video/mock00000001?as=mp4']);
    expect(store.playSrc.value?.kind).toBe('mp4');
  });
  it('makes a preview copy when nothing plays, and measures its clock from the proxy’s own first frame', async () => {
    const e = new MockEngine({ name: 'a.ogv' });
    e.makeProxy = async () => ({ state: 'done', fraction: 1, url: 'mock://proxy', first_frame_file_time: 0.5, error: null });
    const tried: string[] = [];
    await open(e, async (_el, url, first) => { tried.push(`${url} ${first}`); return { ok: url === 'mock://proxy', clockOffset: 0.25 }; });
    expect(tried).toEqual(['mock://video/mock00000001 0', 'mock://proxy 0.5']);
    expect(store.playSrc.value).toEqual({ url: 'mock://proxy', kind: 'proxy' });
    expect(store.clockOffset.value).toBe(0.25);
  });
  it('says so when the preview copy fails', async () => {
    const e = new MockEngine({ name: 'a.ogv' });
    e.makeProxy = async () => ({ state: 'failed', fraction: 0, url: null, first_frame_file_time: null, error: 'no encoder' });
    await open(e, async () => ({ ok: false, clockOffset: 0 }));
    expect(store.playSrc.value).toBeNull();
    expect(store.banner.value?.text).toContain('No preview: ffmpeg failed (no encoder)');
  });
});
