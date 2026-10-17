// UI fixes: a clip that starts before the video's first frame, and a mock engine that answers in the server's shapes.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { createStore, type Store } from '../src/state';
import { fromDetector, validate } from '../src/model/edits';
import { FileBrowser } from '../src/ui/FileBrowser';

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
let root: HTMLElement;
let store: Store | null = null;

beforeAll(() => {
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  root = document.createElement('div');
  document.body.appendChild(root);
});
afterEach(() => {
  render(null, root);
  store?.dispose();
  store = null;
  root.remove();
});

describe('frontend-2: detector clips against the video span', () => {
  it('fromDetector clamps a clip that starts before the first frame and drops one that ends up too short', () => {
    const s = fromDetector(
      [{ index: 1, start: 0.9, end: 10 }, { index: 2, start: 11, end: 11.1 }, { index: 3, start: 30, end: 70 }, { index: 4, start: 0, end: 1.5 }],
      { offset: 2, end: 62 },
    );
    expect(s.clips.map((c) => [c.id, c.start, c.end])).toEqual([['d1', 2, 10], ['d3', 30, 62]]);
    expect(validate(s.clips, { offset: 2, end: 62 })).toEqual([]);
    expect(s.nextId).toBe(4);
  });

  it('a run whose first clip starts at 0.9 s in a video that starts at 2 s still saves', async () => {
    const engine = new MockEngine({ offset: 2, end_time: 62 });
    const run = engine.run.bind(engine);
    engine.run = async (id: string) => {
      const d = await run(id);
      d.segments.clips = [{ index: 1, start: 0.9, end: 10 }, { index: 2, start: 20, end: 28.5 }];
      return d;
    };
    store = createStore(engine, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5 });
    await store.init();
    await act(async () => void render(<App store={store!} />, root));
    await act(async () => void (await store!.openPath(MOCK_PATH)));
    await tick(5);
    expect(store.edit.value.clips[0].start).toBe(2);
    await act(() => store!.seek(5));
    await act(() => void document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true, cancelable: true })));
    await tick(40);
    expect(store.saveState.value).toBe('saved');
    expect(engine.calls.filter((c) => c.startsWith('saveEdits'))).toEqual(['saveEdits mockrun00001 r0 n3']);
  });
});

describe('contract-4: the mock engine answers in the server\'s shapes', () => {
  it('listDir roots are {name, path} objects and a folder has no size (serve_media.fs_list)', async () => {
    const l = await new MockEngine().listDir('/mock');
    expect(l.roots.every((r) => typeof r === 'object' && typeof r.name === 'string' && typeof r.path === 'string')).toBe(true);
    expect(l.entries.find((e) => e.kind === 'dir')?.size).toBeNull();
  });

  it('the file browser shows the roots by name and opens one by path', async () => {
    const engine = new MockEngine();
    const seen: (string | undefined)[] = [];
    const listDir = engine.listDir.bind(engine);
    engine.listDir = async (d?: string) => (seen.push(d), listDir(d));
    await act(async () => void render(<FileBrowser engine={engine} mode="video" onPick={() => {}} onClose={() => {}} />, root));
    await tick(5);
    const roots = [...root.querySelectorAll('.roots button')];
    expect(roots.map((b) => b.textContent)).toEqual(['Home', 'Card']);
    await act(() => void (roots[1] as HTMLElement).click());
    await tick(5);
    expect(seen.at(-1)).toBe('/Volumes/Card');
  });

  it('makeProxy answers like POST /proxy (no url yet) and mediaTask like GET /proxy', async () => {
    const e = new MockEngine();
    const started = await e.makeProxy();
    expect(started.state).toBe('running');
    expect(started.url ?? null).toBeNull();
    const done = await e.mediaTask();
    expect(done.state).toBe('done');
    expect(done.url).toBeTruthy();
  });
});
