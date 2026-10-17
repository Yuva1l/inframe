// The engine switch (docs/T9B_DESIGN.md section 1): "My Mac" / "This browser", the choice kept in localStorage `inframe.engine`.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { BrowserEngine, browserCaps } from '../src/engine/browser';
import { LocalEngine } from '../src/engine/local';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { CHOICE_KEY, pickInitial, readChoice, STATIC_HOST_HINT, SWITCH_ASK } from '../src/engine/switch';
import { createShell, type Shell } from '../src/shell';

let root: HTMLElement;
let shell: Shell | undefined;
let mock: MockEngine;
const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
async function until(cond: () => unknown, what = 'condition') {
  for (let i = 0; i < 100; i++) {
    if (cond()) return;
    await tick(10);
  }
  throw new Error(`timed out waiting for ${what}`);
}
const q = <T extends HTMLElement = HTMLElement>(sel: string, from: ParentNode = root) => from.querySelector(sel) as T;
const click = (el: Element) => act(() => void (el as HTMLElement).click());
/** the switch inside `from` (the top bar or the open screen) */
const btn = (kind: 'local' | 'browser', from: ParentNode = root) => q<HTMLButtonElement>(`.engine-switch button[data-engine=${kind}]`, from);
const topBar = () => q('header.top');
const openCard = () => q('.opencard');
const dialog = () => q('[role=dialog][aria-label="Switch engine?"]');

/** an in-memory Storage */
function memoryStorage(initial: Record<string, string> = {}) {
  const m = new Map(Object.entries(initial));
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, String(v)), removeItem: (k: string) => void m.delete(k), clear: () => m.clear(), key: () => null, get length() { return m.size; } };
}
const throwingStorage = () => ({
  getItem() { throw new DOMException('blocked', 'SecurityError'); },
  setItem() { throw new DOMException('blocked', 'SecurityError'); },
  removeItem() { throw new DOMException('blocked', 'SecurityError'); },
});

async function boot(opts: { local?: boolean } = {}) {
  mock = new MockEngine();
  const sh = (shell = createShell({ local: opts.local === false ? null : mock, browser: new BrowserEngine() }, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5, pollMs: 5 }));
  await sh.start();
  await act(async () => void render(<sh.Root />, root));
}

beforeAll(() => {
  HTMLMediaElement.prototype.pause = () => {};
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
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

describe('choosing the engine to start with', () => {
  it('is My Mac when it exists and nothing is kept; this browser when that was kept; this browser when My Mac does not exist', () => {
    expect(pickInitial(true, null)).toBe('local');
    expect(pickInitial(true, 'local')).toBe('local');
    expect(pickInitial(true, 'browser')).toBe('browser');
    expect(pickInitial(false, null)).toBe('browser');
    expect(pickInitial(false, 'local')).toBe('browser');
  });
  it('reads only the two known values', () => {
    localStorage.setItem(CHOICE_KEY, 'local');
    expect(readChoice()).toBe('local');
    localStorage.setItem(CHOICE_KEY, 'cloud');
    expect(readChoice()).toBeNull();
  });
});

describe('both engines present (inframe serve)', () => {
  it('starts on My Mac with both buttons enabled, in the top bar and on the open screen', async () => {
    await boot();
    expect(shell!.current.value!.engine.kind).toBe('local');
    for (const from of [topBar(), openCard()]) {
      expect(btn('local', from).getAttribute('aria-pressed')).toBe('true');
      expect(btn('browser', from).getAttribute('aria-pressed')).toBe('false');
      expect(btn('local', from).disabled).toBe(false);
      expect(btn('browser', from).disabled).toBe(false);
    }
    expect(topBar().textContent).not.toContain('Local engine'); // the switch replaces the badge
    expect(q('[data-testid=engine-hint]')).toBeNull();
  });

  it('switches without asking when no video is open, and keeps the choice', async () => {
    await boot();
    await click(btn('browser', openCard()));
    await until(() => shell!.current.value!.engine.kind === 'browser', 'the browser engine');
    expect(dialog()).toBeNull();
    expect(localStorage.getItem(CHOICE_KEY)).toBe('browser');
    expect(btn('browser', topBar()).getAttribute('aria-pressed')).toBe('true');
    expect(btn('local', topBar()).getAttribute('aria-pressed')).toBe('false');
    expect(q('[data-testid=file-input]')).not.toBeNull(); // the browser engine opens a File from the page
    expect(q('[data-testid=browser-note]')).toBeNull();
    expect(q('input[aria-label="Path to a video"]')).toBeNull();
    expect(q('[data-testid=pick-layer]')).toBeNull();

    await click(btn('local', topBar()));
    await until(() => shell!.current.value!.engine.kind === 'local', 'My Mac again');
    expect(localStorage.getItem(CHOICE_KEY)).toBe('local');
    expect(q('input[aria-label="Path to a video"]')).not.toBeNull();
  });

  it('asks first when a video is open: Cancel keeps everything, Switch closes the video', async () => {
    await boot();
    await act(async () => void (await shell!.current.value!.openPath(MOCK_PATH)));
    await tick(5);
    expect(q('main.editor')).not.toBeNull();
    const before = shell!.current.value!;

    await click(btn('browser', topBar()));
    expect(dialog()).not.toBeNull();
    expect(q('[data-testid=engine-ask]').textContent).toBe(SWITCH_ASK);
    expect(SWITCH_ASK).toBe('The open video closes; runs stay with the engine that made them.');
    expect(shell!.current.value).toBe(before); // nothing happened yet
    expect(localStorage.getItem(CHOICE_KEY)).toBeNull();

    await click([...dialog().querySelectorAll('button')].find((b) => b.textContent === 'Cancel')!);
    expect(dialog()).toBeNull();
    expect(shell!.current.value).toBe(before);
    expect(q('main.editor')).not.toBeNull();

    await click(btn('browser', topBar()));
    await click([...dialog().querySelectorAll('button')].find((b) => b.textContent === 'Switch to This browser')!);
    await until(() => shell!.current.value!.engine.kind === 'browser', 'the browser engine');
    expect(dialog()).toBeNull();
    expect(shell!.current.value).not.toBe(before);
    expect(shell!.current.value!.media.value).toBeNull(); // the video closed
    expect(q('main.editor')).toBeNull();
    expect(q('.opencard')).not.toBeNull();
    expect(localStorage.getItem(CHOICE_KEY)).toBe('browser');
    expect(shell!.switcher.kind.value).toBe('browser');
  });

  it('does not ask for the engine already in use', async () => {
    await boot();
    await act(async () => void (await shell!.current.value!.openPath(MOCK_PATH)));
    await tick(5);
    const before = shell!.current.value;
    await click(btn('local', topBar()));
    expect(dialog()).toBeNull();
    expect(shell!.current.value).toBe(before);
  });

  it('keeps the choice for the next visit', async () => {
    await boot();
    await click(btn('browser', openCard()));
    await until(() => shell!.current.value!.engine.kind === 'browser', 'the browser engine');
    render(null, root);
    shell!.current.value!.dispose();
    await boot(); // a new page load: same localStorage
    expect(shell!.current.value!.engine.kind).toBe('browser');
    expect(btn('browser', topBar()).getAttribute('aria-pressed')).toBe('true');
  });
});

describe('this browser only (a static host)', () => {
  it('starts in the browser, disables My Mac and says how to get it', async () => {
    localStorage.setItem(CHOICE_KEY, 'local'); // a stale choice from another visit
    await boot({ local: false });
    expect(shell!.current.value!.engine.kind).toBe('browser');
    for (const from of [topBar(), openCard()]) {
      expect(btn('local', from).disabled).toBe(true);
      expect(btn('local', from).title).toBe(STATIC_HOST_HINT);
      expect(btn('browser', from).getAttribute('aria-pressed')).toBe('true');
    }
    expect(q('[data-testid=engine-hint]').textContent).toContain('Run inframe serve on your Mac and open the page it prints');
    expect(q('[data-testid=engine-hint] code').textContent).toBe('inframe serve');
    await click(btn('local', openCard()));
    expect(shell!.current.value!.engine.kind).toBe('browser');
    expect(shell!.switcher.kind.value).toBe('browser');
  });

  it('shows the open screen, not an error', async () => {
    await boot({ local: false });
    expect(q('[role=alert]')).toBeNull();
    expect(openCard().textContent).toContain('Open a video');
    expect(root.querySelector('.preflight')?.textContent ?? '').not.toMatch(/ffmpeg/); // ffmpeg is not what this engine needs: the checklist is WebGPU, decoding and the models
  });
});

describe('a localStorage that throws', () => {
  beforeEach(() => void vi.stubGlobal('localStorage', throwingStorage()));

  it('reads as no choice', () => {
    expect(readChoice()).toBeNull();
  });

  it('still starts and switches, the choice just does not last', async () => {
    await boot();
    expect(shell!.current.value!.engine.kind).toBe('local');
    await click(btn('browser', openCard()));
    await until(() => shell!.current.value!.engine.kind === 'browser', 'the browser engine');
    expect(shell!.switcher.kind.value).toBe('browser');
    expect(q('[role=alert]')).toBeNull();
    await click(btn('local', topBar()));
    await until(() => shell!.current.value!.engine.kind === 'local', 'My Mac again');
  });

  it('starts on a static host too', async () => {
    await boot({ local: false });
    expect(shell!.current.value!.engine.kind).toBe('browser');
  });
});

const exportDialog = () => q('[role=dialog][aria-label=Export]');
const legends = () => [...exportDialog().querySelectorAll('legend')].map((l) => l.textContent?.trim());
async function openExport(e: MockEngine) {
  const sh = (shell = createShell({ local: e, browser: new BrowserEngine() }, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5, pollMs: 5 }));
  await sh.start();
  await act(async () => void render(<sh.Root />, root));
  await act(async () => void (await shell!.current.value!.openPath(MOCK_PATH)));
  await tick(5);
  await act(() => void document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'e', metaKey: true, bubbles: true, cancelable: true })));
  await until(() => exportDialog(), 'the export dialog');
}

describe('the export dialog follows the caps', () => {
  it('shows Reel and Edit lists when the engine has them', async () => {
    await openExport(new MockEngine());
    expect(legends()).toContain('Edit lists');
    expect(legends().some((l) => l?.includes('Reel'))).toBe(true);
  });
  it('hides Reel and Edit lists when the caps are false', async () => {
    const e = new MockEngine();
    e.caps.reel = false;
    e.caps.editLists = false;
    await openExport(e);
    expect(legends()).not.toContain('Edit lists');
    expect(legends().some((l) => l?.includes('Reel'))).toBe(false);
    expect(legends().some((l) => l?.includes('Clips'))).toBe(true); // clips stay
  });
  it('hides just one of them', async () => {
    const e = new MockEngine();
    e.caps.reel = false;
    await openExport(e);
    expect(legends()).toContain('Edit lists');
    expect(legends().some((l) => l?.includes('Reel'))).toBe(false);
  });
});

describe('the local engine names reel and editLists even when an older server does not', () => {
  it('fills both with true, and leaves a value the server sends alone', async () => {
    const body = (caps: Record<string, unknown>) => ({ version: 'x', engine: 'local', platform: 'darwin', caps, presets: [], default_preset: 'balanced', classes: [], defaults: {}, preflight: {}, job: null, media: [], start_media: null });
    const old = { openDialog: true, output: 'server-folder', streamCopyCut: true };
    const mk = (caps: Record<string, unknown>) => new LocalEngine({ token: 't', mediaKey: 'm', fetch: (async () => new Response(JSON.stringify(body(caps)), { status: 200, headers: { 'Content-Type': 'application/json' } })) as typeof fetch });
    const a = mk(old);
    const s = await a.session();
    expect(s.caps).toMatchObject({ reel: true, editLists: true, openDialog: true });
    expect(a.caps.reel).toBe(true);
    expect((await mk({ ...old, reel: false, editLists: false }).session()).caps).toMatchObject({ reel: false, editLists: false });
  });
});

describe('browser caps', () => {
  it('turn everything native off, and name the output by what the browser has', () => {
    const c = browserCaps({});
    expect(c).toMatchObject({ openDialog: false, photosPicker: false, fileBrowser: false, locate: false, reveal: false, music: false, beatSync: false, savedTargets: false, people: false, proxy: false, streamCopyCut: true, chooseOutputFolder: false, reel: false, editLists: false });
    expect(c.output).toBe('download');
    expect(browserCaps({ showDirectoryPicker() {} }).output).toBe('granted-folder');
  });
  it('the local and mock engines keep the reel and the edit lists', async () => {
    expect((await new MockEngine().session()).caps).toMatchObject({ reel: true, editLists: true });
  });
});
