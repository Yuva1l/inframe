// UI edge cases: a superseded boxes request stops the spinner; edits made while loadRun fetches are saved, not dropped.
import { act } from 'preact/test-utils';
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { EngineError, type RunData } from '../src/engine/types';
import { createStore, type Store } from '../src/state';

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
let store: Store | null = null;

beforeAll(() => {
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => sessionStorage.clear());
afterEach(() => {
  store?.dispose();
  store = null;
});

class SupersededEngine extends MockEngine {
  async boxes(): Promise<never> {
    throw new EngineError(409, 'superseded', { error: 'superseded' });
  }
}

describe('a superseded boxes request', () => {
  it('stops the Tap spinner and shows no error', async () => {
    store = createStore(new SupersededEngine(), { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5, pollMs: 5 });
    await store.init();
    await act(async () => void (await store!.openPath(MOCK_PATH)));
    await tick(5);
    await act(() => void (store!.tab.value = 'tap'));
    await tick(400); // the debounce, then the 409
    expect(store.pickInfo.value.loading).toBe(false);
    expect(store.pickInfo.value.error).toBeNull();
  });
});

const RUN_B = 'mockrun00002';
class SlowSecondRun extends MockEngine {
  slow = 0;
  async runs() { const [a] = await super.runs(); return [a, { ...a, run_id: RUN_B, slug: 'saved-alice', target: 'saved:alice' }]; }
  async run(id: string): Promise<RunData> {
    if (this.slow) await new Promise((r) => setTimeout(r, this.slow));
    const r = await super.run(id);
    return id === RUN_B ? { ...r, run_id: RUN_B, edits: null } : r;
  }
}

describe('loadRun while the user keeps editing', () => {
  it('an edit made during the fetch is saved to the run it belongs to', async () => {
    const e = new SlowSecondRun();
    store = createStore(e, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 200, toastMs: 0, retryMs: 5 });
    await store.init();
    await act(async () => void (await store!.openPath(MOCK_PATH)));
    await tick(5);
    e.slow = 60;
    const load = store.loadRun(RUN_B);
    await tick(20); // the fetch is out; the user trims now
    await act(() => { store!.seek(store!.edit.value.clips[0].start + 0.5); store!.dispatch('add'); store!.dispatch('split'); });
    const n = store.edit.value.clips.length;
    await act(async () => void (await load));
    await tick(300);
    expect(e.calls.filter((c) => c.startsWith('saveEdits'))).toEqual([`saveEdits mockrun00001 r0 n${n}`]);
    expect(store.run.value?.run_id).toBe(RUN_B);
    expect(store.saveState.value).toBe('saved');
  });
});
