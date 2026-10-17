// The included songs: the music dialog lists the songs that ship with inframe before any search, offline, and Use / Play work on them.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import type { MusicTrack } from '../src/engine/types';
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
const rows = () => [...root.querySelectorAll('ul[aria-label="Included songs"] li')];

const SONGS: MusicTrack[] = [
  { id: 'b1', title: 'Harp Morning', creator: 'A. Harper', duration_s: 190, license: 'by', license_version: '3.0', bpm: 123.4, bundled: true },
  { id: 'b2', title: 'Plain Beat', creator: 'B. Drummer', duration_s: 100, license: 'cc0', license_version: null, bpm: 98, bundled: true },
];

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

describe('included songs in the music dialog', () => {
  it('lists them before any search, with length, BPM and license, and says they work offline', async () => {
    const e = new MockEngine();
    e.libraryTracks = SONGS;
    await openReel(e);
    await until(() => rows().length === 2, 'the included songs');
    expect(rows()[0].textContent).toContain('Harp Morning');
    expect(rows()[0].textContent).toContain('123 BPM');
    expect(rows()[0].textContent).toContain('CC BY 3.0');
    expect(rows()[1].textContent).toContain('CC0');
    expect(text('[data-testid=library-notice]')).toContain('work offline');
    expect(e.musicQueries).toHaveLength(0); // nothing was searched
    expect(withText('p', 'More songs: search online', false)).toBeTruthy();
  });

  it('Use picks an included song by its id for the export, and Play streams it from the server without a download', async () => {
    const e = new MockEngine();
    e.libraryTracks = SONGS;
    await openReel(e);
    await until(() => rows().length === 2);
    await click(rows()[0].querySelectorAll('button')[0]); // Play
    expect(q<HTMLAudioElement>('audio[aria-label=Preview]').getAttribute('src')).toBe('mock://music/b1');
    await click(rows()[0].querySelectorAll('button')[1]); // Use
    expect(text('[data-testid=music-choice]')).toContain('Harp Morning');
    await click(exportBtn());
    await until(() => e.lastExport);
    expect(e.lastExport?.music).toMatchObject({ track_id: 'b1' });
  });

  it('CC0 only asks the server again and shows only the CC0 songs; a chosen CC BY song is dropped', async () => {
    const e = new MockEngine();
    e.libraryTracks = SONGS;
    await openReel(e);
    await until(() => rows().length === 2);
    await click(rows()[0].querySelectorAll('button')[1]);
    await click(checkbox('CC0 only'));
    await until(() => rows().length === 1, 'the CC0 song');
    expect(e.libraryQueries).toEqual([false, true]);
    expect(rows()[0].textContent).toContain('Plain Beat');
    expect(text('[data-testid=music-choice]')).not.toContain('Harp Morning');
  });

  it('shows no list when the server has no included songs or the call fails', async () => {
    const e = new MockEngine();
    e.musicLibrary = async () => { throw new Error('404'); };
    await openReel(e);
    await tick(30);
    expect(rows()).toHaveLength(0);
    expect(withText('button', 'Search')).toBeTruthy();
  });

  it('an auto pick of an included song says it works offline', async () => {
    const e = new MockEngine();
    const orig = e.musicAuto.bind(e);
    e.musicAuto = async (...a) => { const r = await orig(...a); r.track.bundled = true; return r; };
    await openReel(e);
    await click(q('[data-testid=music-auto]'));
    await until(() => q('[data-testid=auto-bundled]'), 'the included-song note');
    expect(text('[data-testid=auto-bundled]')).toContain('works offline');
  });
});
