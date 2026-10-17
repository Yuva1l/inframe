// Browser UI: the video never grows past the player box; a tap shows a pending state and blocks more taps.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
// the web package has no @types/node; vitest runs under node, so the file read is allowed to be untyped
// @ts-ignore
import { readFileSync } from 'node:fs';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { EngineError } from '../src/engine/types';
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
const withText = <T extends HTMLElement = HTMLElement>(tag: string, t: string) =>
  [...root.querySelectorAll(tag)].find((e) => e.textContent?.trim() === t) as T;
const clickAt = (x: number, y: number) =>
  act(() => void q('[data-testid=pick-layer]').dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: x, clientY: y })));

/** A tap whose answer the test releases by hand. */
class SlowTapEngine extends MockEngine {
  release: (() => void) | null = null;
  fail = false;
  async tap(...a: Parameters<MockEngine['tap']>) {
    await new Promise<void>((r) => (this.release = r));
    if (this.fail) throw new Error('the tracker gave up');
    return super.tap(...a);
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
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(
    () => ({ left: 0, top: 0, right: 800, bottom: 450, width: 800, height: 450, x: 0, y: 0, toJSON() {} }) as DOMRect,
  );
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
  await act(() => void withText('button[role=tab]', 'Tap').click());
  await until(() => text('[data-testid=pick-status]').includes('2 found'), 'boxes');
}

describe('ui-1: the video stays inside the player box', () => {
  it('takes the <video> out of the stage layout, so its intrinsic size cannot grow the box', () => {
    const css: string = readFileSync((globalThis as any).process.cwd() + '/src/styles.css', 'utf8');
    const rule = (sel: string) => new RegExp(`(?:^|\\n)${sel.replace('.', '\\.')}\\s*\\{([^}]*)\\}`).exec(css)?.[1] ?? '';
    const video = rule('.video');
    const stage = rule('.stage');
    // an in-flow <video> of 1080x1920 sizes its grid row to 1920 px; an absolutely placed one cannot
    expect(video).toMatch(/position:\s*absolute/);
    expect(video).toMatch(/inset:\s*0/);
    expect(video).toMatch(/object-fit:\s*contain/);
    expect(stage).toMatch(/position:\s*relative/);
    expect(stage).toMatch(/overflow:\s*hidden/);
  });
});

describe('ui-2: a tap in flight', () => {
  it('shows a pending state, ignores a second tap until the first returns, then clears', async () => {
    const e = new SlowTapEngine();
    await boot(e);
    await clickAt(100, 200);
    await until(() => e.release, 'the request');
    expect(text('[data-testid=tap-status]')).toMatch(/tapping/i);
    expect(q('[data-testid=pick-layer]').classList.contains('busy')).toBe(true);
    expect(store.pickBoxList.value?.map((b) => b.label)).toEqual(['1...', '2']); // the tapped box says so
    expect(withText<HTMLButtonElement>('button', 'Undo last tap').disabled).toBe(true);
    await clickAt(100, 200); // a second tap while the first runs
    await tick(20);
    expect(e.calls.filter((c) => c.startsWith('tap person')).length).toBe(0); // the mock logs only after the slow part
    e.release!();
    await until(() => text('[data-testid=tap-count]') === '1 tap', 'the tap');
    expect(q('[data-testid=tap-status]')).toBeNull();
    expect(q('[data-testid=pick-layer]').classList.contains('busy')).toBe(false);
    expect(e.calls.filter((c) => c.startsWith('tap person')).length).toBe(1);
    expect(store.tapPending.value).toBeNull();
  });

  it('lifts the pending state and toasts when the tap fails', async () => {
    const e = new SlowTapEngine();
    e.fail = true;
    await boot(e);
    await clickAt(100, 200);
    await until(() => e.release, 'the request');
    expect(store.tapPending.value).not.toBeNull();
    e.release!();
    await until(() => store.tapPending.value === null, 'the failure');
    expect(text('[data-testid=tap-count]')).toBe('0 taps');
    expect(store.toasts.value.some((t) => t.kind === 'error' && t.text.includes('the tracker gave up'))).toBe(true);
  });
});

/** The browser engine's answer when the models are not downloaded yet. */
class NoModelsEngine extends MockEngine {
  async boxes(): Promise<never> {
    throw new EngineError(409, 'The models are not downloaded yet. Use "Download models" first.', { error: 'models' });
  }
}

describe('boxes with the models missing', () => {
  it('shows the message on the Tap tab instead of spinning forever', async () => {
    store = createStore(new NoModelsEngine(), { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5, pollMs: 5 });
    await store.init();
    await act(async () => void render(<App store={store} />, root));
    await act(async () => void (await store.openPath(MOCK_PATH)));
    await tick(5);
    await act(() => void withText('button[role=tab]', 'Tap').click());
    await until(() => q('[role=alert]'), 'the models message');
    expect(text('[role=alert]')).toMatch(/not downloaded/i);
    expect(store.pickInfo.value.loading).toBe(false);
  });
});
