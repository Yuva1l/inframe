// UI edge cases: a busy export names the job, CC0 only drops a CC BY pick,
// an auto pick without a title is not "null", and closing the dialog ends an auto pick.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { EngineError, type Job, type MusicAutoResult } from '../src/engine/types';
import { createStore, type Store } from '../src/state';

let root: HTMLElement;
let store: Store;
const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
async function until(cond: () => unknown, what = 'condition') {
  for (let i = 0; i < 100; i++) {
    if (cond()) return;
    await tick(10);
  }
  throw new Error(`timed out waiting for ${what}`);
}
const q = <T extends HTMLElement = HTMLElement>(sel: string) => root.querySelector(sel) as T;
const text = (sel: string) => q(sel)?.textContent ?? '';
const withText = <T extends HTMLElement = HTMLElement>(tag: string, t: string, exact = true) =>
  [...root.querySelectorAll(tag)].find((e) => (exact ? e.textContent?.trim() === t : e.textContent?.includes(t))) as T;
const click = (el: Element) => act(() => void (el as HTMLElement).click());
const checkbox = (label: string) => withText<HTMLLabelElement>('label.check', label, false)?.querySelector('input') as HTMLInputElement;
const key = (k: string, extra: KeyboardEventInit = {}) =>
  act(() => void document.body.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...extra })));
const dlg = () => q('[role=dialog][aria-label=Export]');
const exportBtn = () => [...dlg().querySelectorAll('button')].find((e) => e.textContent?.trim() === 'Export') as HTMLButtonElement;

async function openReel(e: MockEngine) {
  e.caps.music = true;
  store = createStore(e, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5, pollMs: 5 });
  await store.init();
  await act(async () => void render(<App store={store} />, root));
  await act(async () => void (await store.openPath(MOCK_PATH)));
  await tick(5);
  await key('e', { metaKey: true });
  await until(() => dlg());
  await click(checkbox('Reel'));
}
const runningJob = (id: string): Job => ({ id, kind: 'music-auto', state: 'running', fraction: 0.1, message: 'choosing', started: 0, elapsed_s: 0, media_id: null, run_id: null, result: null, error: null });

beforeAll(() => {
  HTMLMediaElement.prototype.pause = () => {};
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
  root = document.createElement('div');
  document.body.appendChild(root);
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(
    () => ({ left: 0, top: 0, right: 800, bottom: 450, width: 800, height: 450, x: 0, y: 0, toJSON() {} }) as DOMRect,
  );
});
afterEach(() => {
  vi.restoreAllMocks();
  render(null, root);
  store.dispose();
  root.remove();
});

describe('lows9 ui', () => {
  it('a busy export names the job that runs, not the bare word busy', async () => {
    const e = new MockEngine();
    await openReel(e);
    const other: Job = { ...runningJob('j7'), kind: 'people', state: 'done' };
    e.startExport = async () => { throw new EngineError(409, 'busy', { error: 'busy', job: other }); };
    await click(exportBtn());
    await until(() => dlg().querySelector('[role=alert]'), 'the error');
    const msg = dlg().querySelector('[role=alert]')!.textContent!;
    expect(msg).not.toBe('busy');
    expect(msg).toContain('A people job is running');
  });

  it('ticking CC0 only drops a chosen CC BY search track, so the export cannot fail on its licence', async () => {
    const e = new MockEngine();
    await openReel(e);
    await click(withText('button', 'Search'));
    await until(() => root.querySelectorAll('ul[aria-label="Music results"] li').length === 2);
    await click(root.querySelectorAll('ul[aria-label="Music results"] li')[1].querySelectorAll('button')[1]); // Long Road, CC BY
    expect(text('[data-testid=music-choice]')).toContain('Long Road');
    await click(checkbox('CC0 only'));
    expect(text('[data-testid=music-choice]')).not.toContain('Long Road');
    await click(exportBtn());
    await until(() => e.lastExport);
    expect(e.lastExport?.music).toBeNull();
  });

  it('ticking CC0 only keeps a chosen CC0 search track', async () => {
    const e = new MockEngine();
    await openReel(e);
    await click(withText('button', 'Search'));
    await until(() => root.querySelectorAll('ul[aria-label="Music results"] li').length === 2);
    await click(root.querySelectorAll('ul[aria-label="Music results"] li')[0].querySelectorAll('button')[1]); // Bright Day, CC0
    await click(checkbox('CC0 only'));
    expect(text('[data-testid=music-choice]')).toContain('Bright Day');
  });

  it('an auto pick without a title is shown as an untitled song, never "null"', async () => {
    const e = new MockEngine();
    const orig = e.musicAuto.bind(e);
    e.musicAuto = async (...a) => {
      const r = await orig(...a);
      return { ...r, track: { ...r.track, title: null as unknown as string } };
    };
    await openReel(e);
    await click(q('[data-testid=music-auto]'));
    await until(() => q('[data-testid=auto-pick]'), 'the pick');
    expect(text('[data-testid=music-choice]')).toContain('Using: Untitled song');
    expect(root.textContent).not.toContain('null');
  });

  it('closing the dialog during an auto pick stops the polling and cancels the server job', async () => {
    const e = new MockEngine();
    let signal: AbortSignal | undefined;
    e.musicAuto = (_r, _q, onJob, sig) => {
      signal = sig;
      onJob?.(runningJob('mockjob-music-auto'));
      return new Promise<MusicAutoResult>(() => {});
    };
    const cancelled: string[] = [];
    e.cancel = (async (id: string) => { cancelled.push(id); return runningJob(id); }) as unknown as MockEngine['cancel'];
    await openReel(e);
    await click(q('[data-testid=music-auto]'));
    await until(() => q('[data-testid=auto-busy]'), 'the busy line');
    await key('Escape');
    await until(() => !dlg(), 'the dialog closed');
    expect(cancelled).toEqual(['mockjob-music-auto']);
    expect(signal?.aborted).toBe(true);
  });
});

describe('lows9 ui: LocalEngine.musicAuto', () => {
  it('stops polling the job once the signal is aborted', async () => {
    const { LocalEngine } = await import('../src/engine/local');
    const storage = { getItem: () => JSON.stringify({ token: 'tok-SESSION-1234567890', media_key: 'key-MEDIA-1234567890' }), setItem: () => {} };
    const calls: string[] = [];
    const f = (async (url: string | URL | Request) => {
      calls.push(String(url));
      return new Response(JSON.stringify({ job: runningJob('j1'), ...runningJob('j1') }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }) as unknown as typeof fetch;
    const e = (await LocalEngine.connect({ fetch: f, storage, hash: '', replaceUrl: () => {} }))!;
    const ctl = new AbortController();
    const p = e.musicAuto('r1', { clip_ids: null, mood: '', cc0_only: false, rank: 0 }, undefined, ctl.signal);
    setTimeout(() => ctl.abort(), 20);
    await expect(p).rejects.toMatchObject({ status: 499 });
    const n = calls.length;
    await new Promise((r) => setTimeout(r, 700));
    expect(calls.length).toBe(n); // no poll after the abort
  });
});
