// T9c deploy: the open screen links to the source code, for both engines ("My Mac" and "This browser" both draw this screen).
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { App } from '../src/app';
import { MockEngine } from '../src/engine/mock';
import { SOURCE_URL } from '../src/model/about';
import { createStore, type Store } from '../src/state';

let root: HTMLElement | null = null;
let store: Store | null = null;
afterEach(() => { if (root) render(null, root); store?.dispose(); root?.remove(); root = null; store = null; });
beforeAll(() => { HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext; });

async function draw(kind: 'local' | 'browser') {
  const engine = new MockEngine();
  Object.defineProperty(engine, 'kind', { value: kind });
  store = createStore(engine, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5 });
  await store.init();
  root = document.createElement('div');
  document.body.appendChild(root);
  await act(async () => void render(<App store={store!} />, root!));
  return root;
}

describe('the source code link', () => {
  it('points at the public repository', () => {
    expect(SOURCE_URL).toBe('https://github.com/Yuva1l/inframe');
  });
  for (const kind of ['local', 'browser'] as const) {
    it(`is on the open screen of the ${kind} engine, opens in a new tab and sends no referrer`, async () => {
      const r = await draw(kind);
      const a = r.querySelector('[data-testid="source-link"] a') as HTMLAnchorElement;
      expect(a).not.toBeNull();
      expect(a.getAttribute('href')).toBe(SOURCE_URL);
      expect(a.textContent).toBe('Source code');
      expect(a.getAttribute('target')).toBe('_blank');
      expect(a.getAttribute('rel')).toBe('noopener noreferrer');
    });
  }
});
