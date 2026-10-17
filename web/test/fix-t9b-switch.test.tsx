// Switching engine must not lose clip edits that are not saved yet (shell.tsx apply).
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { BrowserEngine } from '../src/engine/browser';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { createShell, type Shell } from '../src/shell';

let root: HTMLElement;
let shell: Shell | undefined;
let mock: MockEngine;
const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
const saves = () => mock.calls.filter((c) => c.startsWith('saveEdits')).length;

function memoryStorage() {
  const m = new Map<string, string>();
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, String(v)), removeItem: (k: string) => void m.delete(k), clear: () => m.clear(), key: () => null, get length() { return m.size; } };
}

async function boot(autosaveMs: number) {
  const sh = (shell = createShell({ local: mock, browser: new BrowserEngine() }, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs, toastMs: 0, retryMs: 5, pollMs: 5 }));
  await sh.start();
  await act(async () => void render(<sh.Root />, root));
  await act(async () => void (await sh.current.value!.openPath(MOCK_PATH)));
  await tick(5);
  return sh.current.value!;
}

beforeAll(() => {
  HTMLMediaElement.prototype.pause = () => {};
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
  mock = new MockEngine();
  root = document.createElement('div');
  document.body.appendChild(root);
  vi.stubGlobal('localStorage', memoryStorage());
});
afterEach(() => {
  render(null, root);
  shell?.current.value?.dispose();
  shell = undefined;
  root.remove();
  vi.unstubAllGlobals();
});

describe('switching engine with edits that are not saved', () => {
  it('stays on the old engine, with the edits and a message, when the pending save fails; then switches once the save works', async () => {
    const store = await boot(10000);
    const n = store.edit.value.clips.length;
    await act(async () => void store.addAt(0.05));
    expect(store.edit.value.clips.length).toBe(n + 1);
    mock.failSave = 'down';
    await act(async () => void (await shell!.switcher.select('browser')));

    expect(shell!.switcher.kind.value).toBe('local');
    expect(shell!.current.value).toBe(store);
    expect(store.engine.kind).toBe('local');
    expect(store.edit.value.clips.length).toBe(n + 1);
    expect(store.saveState.value).toBe('error');
    expect(store.toasts.value.some((t) => t.kind === 'error' && /not saved/i.test(t.text))).toBe(true);
    expect(localStorage.getItem('inframe.engine')).toBeNull();

    // the server is back: the second try saves the edit, then switches
    await act(async () => void (await shell!.switcher.select('browser')));
    expect(mock.edits?.clips.length).toBe(n + 1);
    expect(shell!.switcher.kind.value).toBe('browser');
    expect(shell!.current.value!.engine.kind).toBe('browser');
  });

  it('stays when the save is a conflict', async () => {
    const store = await boot(10000);
    await act(async () => void store.addAt(0.05));
    mock.failSave = 'conflict';
    await act(async () => void (await shell!.switcher.select('browser')));
    expect(shell!.switcher.kind.value).toBe('local');
    expect(shell!.current.value).toBe(store);
    expect(store.saveState.value).toBe('conflict');
  });

  it('waits for a save in flight and for the edit made while it ran, and leaves no save timer behind', async () => {
    const real = mock.saveEdits.bind(mock);
    mock.saveEdits = async (...a: Parameters<MockEngine['saveEdits']>) => {
      await new Promise((r) => setTimeout(r, 40));
      return real(...a);
    };
    const store = await boot(5);
    const n = store.edit.value.clips.length;
    await act(async () => void store.addAt(0.05));
    await tick(15); // the autosave timer fired: the first save is in flight
    expect(store.saveState.value).toBe('saving');
    await act(async () => void store.addAt(55)); // a second edit while it runs
    expect(store.edit.value.clips.length).toBe(n + 2);
    await act(async () => void (await shell!.switcher.select('browser')));
    expect(mock.edits?.clips.length).toBe(n + 2); // both edits were written before the switch finished
    expect(shell!.switcher.kind.value).toBe('browser');
    const done = saves();
    await tick(60);
    expect(saves()).toBe(done); // the disposed store saved nothing more
  });
});
