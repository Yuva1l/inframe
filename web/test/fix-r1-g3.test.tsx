// UI fixes: preview copy that already exists, autosave across runs, clips-only tolerance, click after an edge drag, file browser roots.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import type { RunData } from '../src/engine/types';
import { createStore, type Store } from '../src/state';
import { FileBrowser } from '../src/ui/FileBrowser';

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
const ptr = (type: string, x: number) =>
  new ((window as any).PointerEvent ?? MouseEvent)(type, { clientX: x, button: 0, bubbles: true, cancelable: true });
let root: HTMLElement;
let store: Store | null = null;

beforeAll(() => {
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
  root = document.createElement('div');
  document.body.appendChild(root);
});
afterEach(() => {
  render(null, root);
  store?.dispose();
  store = null;
  root.remove();
});

const RUN_B = 'mockrun00002';
class TwoRuns extends MockEngine {
  async runs() { const [a] = await super.runs(); return [a, { ...a, run_id: RUN_B, slug: 'saved-alice', target: 'saved:alice' }]; }
  async run(id: string): Promise<RunData> { const r = await super.run(id); return id === RUN_B ? { ...r, run_id: RUN_B, edits: null } : r; }
}
async function boot(engine: MockEngine, autosaveMs = 200, withApp = false) {
  store = createStore(engine, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs, toastMs: 0, retryMs: 5 });
  await store.init();
  if (withApp) await act(async () => void render(<App store={store!} />, root));
  await act(async () => void (await store!.openPath(MOCK_PATH)));
  await tick(5);
  return store;
}
const saves = (e: MockEngine) => e.calls.filter((c) => c.startsWith('saveEdits'));

describe('contract-2: a preview copy that already exists', () => {
  it('POST /proxy answers done without a url: the url is read with GET /proxy, not reported as an ffmpeg failure', async () => {
    const e = new MockEngine({ name: 'a.ogv' });
    e.makeProxy = async () => ({ state: 'done', fraction: 1 }); // serve_media._proxy_brief
    e.mediaTask = async () => ({ state: 'done', fraction: 1, url: 'mock://proxy', first_frame_file_time: 0.5, error: null });
    const tried: string[] = [];
    store = createStore(e, { playable: async (_el, url) => { tried.push(url); return { ok: url === 'mock://proxy', clockOffset: 0 }; }, toastMs: 0 });
    await store.init();
    await store.openMedia(e.media_);
    store.attachVideo(document.createElement('video'));
    await new Promise((r) => setTimeout(r, 20));
    expect(store.banner.value).toBeNull();
    expect(store.playSrc.value).toEqual({ url: 'mock://proxy', kind: 'proxy' });
    expect(tried).toContain('mock://proxy');
  });
});

describe('frontend-1: a pending autosave stays with its own run', () => {
  it('opening another run saves the first run edits to the first run, and nothing to the second', async () => {
    const e = new TwoRuns();
    await boot(e, 200);
    await act(() => { store!.seek(store!.edit.value.clips[0].start + 0.5); store!.dispatch('add'); store!.dispatch('split'); });
    const n = store!.edit.value.clips.length;
    await act(async () => void (await store!.loadRun(RUN_B)));
    await tick(300);
    expect(saves(e)).toEqual([`saveEdits mockrun00001 r0 n${n}`]);
    expect(store!.run.value?.run_id).toBe(RUN_B);
    expect(store!.saveState.value).toBe('saved');
  });

  it('a save in flight when the other run loads does not hand its revision to the new run', async () => {
    const e = new TwoRuns();
    await boot(e, 5);
    const orig = e.saveEdits.bind(e);
    e.saveEdits = async (...a: Parameters<MockEngine['saveEdits']>) => { await new Promise((r) => setTimeout(r, 60)); return orig(...a); };
    await act(() => { store!.seek(store!.edit.value.clips[0].start + 0.5); store!.dispatch('add'); store!.dispatch('split'); });
    await tick(20); // the save has started
    await act(async () => void (await store!.loadRun(RUN_B)));
    await tick(150);
    expect(store!.revision.value).toBe(0);
    expect(saves(e).filter((c) => c.includes(RUN_B))).toEqual([]);
  });

  it('opening another video saves the pending edits to the run they belong to', async () => {
    const e = new MockEngine();
    await boot(e, 200);
    await act(() => { store!.seek(store!.edit.value.clips[0].start + 0.5); store!.dispatch('add'); store!.dispatch('split'); });
    const n = store!.edit.value.clips.length;
    await act(async () => void (await store!.openMedia(e.media_)));
    await tick(300);
    expect(saves(e)).toEqual([`saveEdits mockrun00001 r0 n${n}`]);
  });
});

describe('frontend-6: clips-only mode tolerance is a frame', () => {
  it('a frame presented just before a clip start (30 ms at 30 fps) is not a gap to seek over', async () => {
    const e = new MockEngine();
    store = createStore(e, { playable: async () => ({ ok: true, clockOffset: 0 }), toastMs: 0 });
    await store.init();
    await store.openMedia(e.media_);
    const el = document.createElement('video');
    store.attachVideo(el);
    await new Promise((r) => setTimeout(r, 20));
    store.edit.value = { clips: [{ id: 'x', start: 5.03, end: 8, origin: 'detector' }], nextId: 2 };
    store.clipsOnly.value = true;
    store.playing.value = true;
    Object.defineProperty(el, 'currentTime', { value: 5.0, writable: true, configurable: true });
    el.dispatchEvent(new Event('timeupdate'));
    expect(store.time.value).toBeCloseTo(5.0, 6); // not jumped to 5.03
  });
});

describe('frontend-5: the click after an edge drag', () => {
  it('does not select the clip and seek back to its start', async () => {
    await boot(new MockEngine(), 5, true);
    await act(() => store!.setTimelineWidth(1200)); // 20 px a second
    store!.snapOn.value = false;
    const blk = root.querySelector('[role=button][data-clip=d2]') as HTMLElement;
    const h = blk.querySelector('[data-edge=end]') as HTMLElement;
    await act(() => {
      const native: boolean = 'onpointerdown' in (h as object);
      h.dispatchEvent(ptr('pointerdown', 570));
      if (!native) h.dispatchEvent(ptr('PointerDown', 570));
    });
    await act(() => void window.dispatchEvent(ptr('pointermove', 580)));
    await act(() => void window.dispatchEvent(ptr('pointerup', 580)));
    const edge = store!.time.value;
    expect(edge).toBeCloseTo(29, 6);
    await act(() => void h.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(store!.time.value).toBeCloseTo(edge, 6);
    expect(store!.selection.value).toEqual([]);
  });
});

describe('contract-1: the file browser with the server roots', () => {
  it('renders roots sent as {name, path} objects without throwing', async () => {
    const e = new MockEngine();
    const real = { dir: '/Users/x', parent: '/Users', roots: [{ name: 'Home', path: '/Users/x' }, { name: 'Card', path: '/Volumes/Card' }], entries: [] };
    e.listDir = async () => real;
    await act(async () => void render(<FileBrowser engine={e} mode="folder" onPick={() => {}} onClose={() => {}} />, root));
    await tick(5);
    expect([...root.querySelectorAll('.roots button')].map((b) => b.textContent)).toEqual(['Home', 'Card']);
  });
});
