// The run panel and the export dialog in browser mode (docs/T9B_DESIGN.md sections 6 and 7): the estimate with its My Mac comparison, the "use My Mac"
// suggestion, and a zip instead of an output folder. Drawn with the mock engine posing as the browser engine.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import type { BrowserPreflight, MediaInfo, Session } from '../src/engine/types';
import { createStore, type Store } from '../src/state';

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
async function until(cond: () => unknown, what = 'condition') {
  for (let i = 0; i < 100; i++) {
    if (cond()) return;
    await tick(10);
  }
  throw new Error(`timed out waiting for ${what}`);
}
let root: HTMLElement;
let store: Store;
const q = <T extends HTMLElement = HTMLElement>(sel: string) => root.querySelector(sel) as T;
const text = (sel: string) => q(sel)?.textContent ?? '';

class BrowserLike extends MockEngine {
  env: Partial<BrowserPreflight> = {};
  constructor(over: Partial<MediaInfo> = {}, output: 'download' | 'granted-folder' = 'download') {
    super();
    Object.defineProperty(this, 'kind', { value: 'browser' });
    Object.assign(this.caps, { openDialog: false, fileBrowser: false, locate: false, savedTargets: false, people: false, proxy: false, streamCopyCut: true, output, reel: false, editLists: false, music: false });
    Object.assign((this as unknown as { media_: MediaInfo }).media_, over);
  }
  async session(): Promise<Session> {
    const s = await super.session();
    const browser: BrowserPreflight = { webgpu: true, webcodecs: true, hevc: true, models: { available: true, error: null, detectorReady: true, embedderReady: true, allReady: true, totalBytes: 1, cachedBytes: 1 }, ...this.env };
    return { ...s, engine: 'browser', preflight: { ...s.preflight, browser } };
  }
}

beforeAll(() => {
  HTMLMediaElement.prototype.pause = () => {};
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
  root = document.createElement('div');
  document.body.appendChild(root);
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(() => ({ left: 0, top: 0, right: 800, bottom: 450, width: 800, height: 450, x: 0, y: 0, toJSON() {} }) as DOMRect);
});
afterEach(() => {
  vi.restoreAllMocks();
  render(null, root);
  store?.dispose();
  root.remove();
});

async function boot(engine: MockEngine) {
  store = createStore(engine, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5, pollMs: 5 });
  await store.init();
  await act(async () => void render(<App store={store} />, root));
  await act(async () => void (await store.openPath(MOCK_PATH)));
  await tick(5);
}

describe('the estimate and the My Mac suggestion', () => {
  it('a short video: an estimate with the My Mac time beside it and no suggestion', async () => {
    await boot(new BrowserLike({ duration: 120, end_time: 120, codec: 'hevc' }));
    expect(text('[data-testid=estimate]')).toMatch(/^about \d+ (s|min) \(estimate\), about \d+ (s|min) on My Mac$/);
    expect(q('[data-testid=use-my-mac]')).toBeNull();
  });

  it('a long video: the estimate passes 10 minutes and the panel suggests My Mac with both times', async () => {
    await boot(new BrowserLike({ duration: 3000, end_time: 3000 }));
    const t = text('[data-testid=use-my-mac]');
    expect(t).toContain('Use My Mac for this one');
    expect(t).toContain('inframe serve');
    expect(t).toMatch(/about \d+ min here, about \d+ min on My Mac/);
  });

  it('no HEVC for an HEVC video, and no WebGPU, are said with the same suggestion', async () => {
    const hevc = new BrowserLike({ duration: 60, end_time: 60, codec: 'hevc' });
    hevc.env = { hevc: false };
    await boot(hevc);
    expect(text('[data-testid=use-my-mac]')).toMatch(/cannot decode HEVC/);
    render(null, root);
    store.dispose();
    const gpu = new BrowserLike({ duration: 60, end_time: 60 });
    gpu.env = { webgpu: false };
    await boot(gpu);
    expect(text('[data-testid=use-my-mac]')).toMatch(/without WebGPU/);
  });

  it('My Mac itself shows the old line and no suggestion', async () => {
    await boot(new MockEngine());
    expect(text('[data-testid=estimate]')).not.toContain('on My Mac');
    expect(q('[data-testid=use-my-mac]')).toBeNull();
  });
});

describe('the export dialog as a zip', () => {
  const open = async () => {
    await act(() => store.openExport());
    await until(() => q('[role=dialog][aria-label=Export]'), 'the dialog');
  };

  it('says it writes one zip instead of naming an output folder', async () => {
    await boot(new BrowserLike({}, 'granted-folder'));
    await open();
    expect(q('[data-testid=export-folder]')).toBeNull();
    expect(text('[data-testid=export-zip-note]')).toContain('you choose where to save it');
    expect(text('[role=dialog][aria-label=Export]')).toContain('One zip with a file per clip');
  });

  it('a browser that downloads says so, and above 2 GB it refuses and points to My Mac', async () => {
    await boot(new BrowserLike({ size: 50 * 1024 ** 3 }));
    await open();
    expect(text('[data-testid=export-zip-note]')).toContain('downloads');
    expect(text('[data-testid=export-too-big]')).toContain('Use My Mac'.replace('Use', 'use'));
    const go = [...root.querySelectorAll('[role=dialog][aria-label=Export] button')].find((b) => b.textContent === 'Export') as HTMLButtonElement;
    expect(go.disabled).toBe(true);
  });

  it('a small export is allowed to go', async () => {
    await boot(new BrowserLike({}));
    await open();
    expect(q('[data-testid=export-too-big]')).toBeNull();
    const go = [...root.querySelectorAll('[role=dialog][aria-label=Export] button')].find((b) => b.textContent === 'Export') as HTMLButtonElement;
    expect(go.disabled).toBe(false);
  });
});
