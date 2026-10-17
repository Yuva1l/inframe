// The Target panel in browser mode (docs/T9B_DESIGN.md section 5): the note about simpler matching, and saved targets, photos and people
// hidden by the engine's caps. The panel is drawn with the mock engine posing as the browser engine.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { createStore, type Store } from '../src/state';

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
let root: HTMLElement;
let store: Store;
const tabs = () => [...root.querySelectorAll('[role=tab]')].map((t) => t.textContent);

class BrowserLike extends MockEngine {
  constructor() {
    super();
    Object.defineProperty(this, 'kind', { value: 'browser' });
    Object.assign(this.caps, { openDialog: false, fileBrowser: false, locate: false, savedTargets: false, people: false, proxy: false, streamCopyCut: false, output: 'download' });
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

describe('target panel in the browser engine', () => {
  it('shows Anyone and Tap only, and the note under the Tap tab', async () => {
    await boot(new BrowserLike());
    expect(tabs()).toEqual(['Anyone', 'Tap']);
    await act(() => void ([...root.querySelectorAll('button[role=tab]')].find((b) => b.textContent === 'Tap') as HTMLElement).click());
    await tick(5);
    const note = root.querySelector('[data-testid=tap-browser-note]');
    expect(note?.textContent).toContain('Browser tap matching is simpler than My Mac');
    expect([...root.querySelectorAll('button')].some((b) => b.textContent === 'Who is in it?')).toBe(false); // people: hidden by caps
  });

  it('My Mac keeps all four tabs and has no such note', async () => {
    await boot(new MockEngine());
    expect(tabs()).toEqual(['Anyone', 'Tap', 'Saved', 'Photos']);
    await act(() => void ([...root.querySelectorAll('button[role=tab]')].find((b) => b.textContent === 'Tap') as HTMLElement).click());
    await tick(5);
    expect(root.querySelector('[data-testid=tap-browser-note]')).toBeNull();
  });
});
