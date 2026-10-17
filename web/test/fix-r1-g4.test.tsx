// UI fixes: a very long run must not overflow the call stack; a multi-word class names the server's run folder.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import type { RunData, Samples } from '../src/engine/types';
import { createStore, type Store } from '../src/state';
import { slugOf } from '../src/workflow';

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
let root: HTMLElement;
let store: Store | null = null;

beforeAll(() => {
  // a context that accepts every call, so the lane drawing code really runs
  const ctx = new Proxy({}, { get: (_t, k) => (k === 'canvas' ? {} : () => undefined), set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => ctx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
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

const N = 300_000; // 1.5 h at 30 fps is 162k; spread arguments overflow the stack near 125k
function bigSamples(mode: string): Samples {
  const t = new Array<number>(N);
  const score = new Array<number>(N);
  for (let i = 0; i < N; i++) {
    t[i] = i / 30;
    score[i] = 0.1 + 0.05 * Math.sin(i);
  }
  return { t, score, box: new Array(N).fill(null), track: null, enter: 0.5, exit: 0.35, merge_gap: 1.5, min_len: 1, pad: 1, sound: null, mode };
}

class Long extends MockEngine {
  constructor(private mode: string) { super(); }
  async run(id: string): Promise<RunData> {
    const r = await super.run(id);
    return { ...r, segments: { ...r.segments, clips: [], presence_intervals: [] } };
  }
  async samples() { return bigSamples(this.mode); }
}

for (const mode of ['detect', 'motion']) {
  describe(`frontend-4: ${mode} run with ${N} samples`, () => {
    it('renders the Inspector, the presence lane and the empty clip track', async () => {
      store = createStore(new Long(mode), { playable: async () => ({ ok: true, clockOffset: 0 }), toastMs: 0 });
      await store.init();
      await act(async () => void render(<App store={store!} />, root));
      await act(async () => void (await store!.openPath(MOCK_PATH)));
      await tick(20);
      expect(store.samples.value?.t.length).toBe(N);
      expect(root.querySelector('.resegment')).not.toBeNull();
      expect(root.querySelector('.lane .empty')?.textContent).toMatch(/highest score 0\.\d\d/);
    });
  });
}

describe('contract-3: the client names the folder the server names', () => {
  it('a class with a space becomes class-sports-ball, like serve_runs._slug_part', () => {
    expect(slugOf({ kind: 'class', class: 'sports ball' })).toBe('class-sports-ball');
    expect(slugOf({ kind: 'class', class: 'traffic light' })).toBe('class-traffic-light');
    expect(slugOf({ kind: 'class', class: 'dog' })).toBe('class-dog');
  });
  it('a saved name is cleaned the same way (other characters to -, ends trimmed, 48 long, x when empty)', () => {
    expect(slugOf({ kind: 'saved', name: 'my  dog!' })).toBe('saved-my-dog');
    expect(slugOf({ kind: 'saved', name: 'a'.repeat(60) })).toBe(`saved-${'a'.repeat(48)}`);
    expect(slugOf({ kind: 'saved', name: '!!!' })).toBe('saved-x');
  });
});
