// main.tsx boot: a My Mac blip must not draw the "server down" banner on the browser store, and a failed start still offers This browser.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { start } from '../src/boot';
import { BrowserEngine } from '../src/engine/browser';
import type { Engines } from '../src/engine';
import { MockEngine } from '../src/engine/mock';
import { EngineError } from '../src/engine/types';
import type { Shell } from '../src/shell';

let root: HTMLElement;
let booted: Shell | undefined;
const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));

function memoryStorage(initial: Record<string, string> = {}) {
  const m = new Map(Object.entries(initial));
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, String(v)), removeItem: (k: string) => void m.delete(k), clear: () => m.clear(), key: () => null, get length() { return m.size; } };
}

beforeAll(() => {
  HTMLMediaElement.prototype.pause = () => {};
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
  root = document.createElement('div');
  document.body.appendChild(root);
});
afterEach(() => {
  render(null, root);
  booted?.current.value?.dispose();
  booted = undefined;
  root.remove();
  vi.unstubAllGlobals();
});

describe('the server-down banner', () => {
  it('is not drawn on the browser store when My Mac blips after the switch to This browser', async () => {
    vi.stubGlobal('localStorage', memoryStorage({ 'inframe.engine': 'browser' }));
    let onDown: ((d: boolean) => void) | undefined;
    const avail = async (o: { onDown?: (d: boolean) => void } = {}): Promise<Engines> => {
      onDown = o.onDown;
      return { local: new MockEngine(), browser: new BrowserEngine(), staticHost: false };
    };
    await act(async () => void (booted = await start(root, avail)));
    await tick();
    expect(booted!.current.value!.engine.kind).toBe('browser');
    act(() => onDown!(true));
    await tick();
    expect(booted!.current.value!.serverDown.value).toBe(false);
    expect(root.querySelector('.downbar')).toBeNull();
  });
  it('is still drawn when the store in use is My Mac', async () => {
    vi.stubGlobal('localStorage', memoryStorage());
    let onDown: ((d: boolean) => void) | undefined;
    const avail = async (o: { onDown?: (d: boolean) => void } = {}): Promise<Engines> => {
      onDown = o.onDown;
      return { local: new MockEngine(), browser: new BrowserEngine(), staticHost: false };
    };
    await act(async () => void (booted = await start(root, avail)));
    await tick();
    act(() => onDown!(true));
    await tick();
    expect(root.querySelector('.downbar')).not.toBeNull();
  });
});

describe('the fatal fallback', () => {
  it('shows the message and still has the engine switcher; This browser then opens a working store', async () => {
    vi.stubGlobal('localStorage', memoryStorage());
    const avail = async (): Promise<Engines> => { throw new EngineError(403, 'This link expired. Open the page inframe serve prints.'); };
    await act(async () => void (booted = await start(root, avail)));
    await tick();
    expect(root.querySelector('.errline')?.textContent).toContain('This link expired');
    const browserBtn = root.querySelector<HTMLButtonElement>('.engine-switch button[data-engine=browser]');
    expect(browserBtn).not.toBeNull();
    await act(() => void browserBtn!.click());
    await tick(10);
    expect(root.querySelector('.errline')).toBeNull();
    expect(booted!.current.value!.fatal.value).toBeNull();
    expect(booted!.current.value!.engine.kind).toBe('browser');
  });
});
