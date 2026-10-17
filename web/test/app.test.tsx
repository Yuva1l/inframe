import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { createStore, type Store } from '../src/state';

let root: HTMLElement;
let store: Store;
let engine: MockEngine;

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
const key = (k: string, extra: KeyboardEventInit = {}, target: EventTarget = document.body) =>
  act(() => void target.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...extra })));
const count = () => root.querySelector('[data-testid=clip-count]')?.textContent;
const rows = () => [...root.querySelectorAll('[data-testid=clip-list] li')];

async function boot(e = new MockEngine()) {
  engine = e;
  store = createStore(engine, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5 });
  await store.init();
  await act(async () => void render(<App store={store} />, root));
}

beforeAll(() => {
  // jsdom has no canvas: the overlay draws nothing here
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  root = document.createElement('div');
  document.body.appendChild(root);
});
afterEach(() => {
  render(null, root);
  store.dispose();
  root.remove();
});

describe('opening', () => {
  it('shows the open screen first, with the first-use help', async () => {
    await boot();
    expect(root.textContent).toContain('Open a video');
    expect(root.textContent).toContain('for shortcuts');
    expect(root.querySelector('.preflight')?.textContent).toContain('ffmpeg');
    expect(root.textContent).toContain('mock_clip.mp4'); // recent list
  });

  it('shows the editor after a path is opened: file facts, player, the newest run’s clips', async () => {
    await boot();
    const input = root.querySelector('input[aria-label="Path to a video"]') as HTMLInputElement;
    await act(() => { input.value = MOCK_PATH; input.dispatchEvent(new Event('input', { bubbles: true })); });
    await act(async () => void (root.querySelector('form.pathrow') as HTMLFormElement).dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    await tick(20);
    expect(engine.calls).toContain(`openPath ${MOCK_PATH}`);
    expect(root.querySelector('video')).not.toBeNull();
    expect(root.querySelector('header.top')?.textContent).toContain('mock_clip.mp4');
    expect(root.querySelector('header.top')?.textContent).toContain('Local engine');
    expect(count()).toBe('3');
    expect(rows()).toHaveLength(3);
    expect(root.querySelector('[data-testid=readout]')?.textContent).toBe('00:00.00 / 01:00.00');
  });

  it('says what went wrong when the file is not there', async () => {
    await boot();
    await act(async () => void (await store.openPath('/mock/missing.mp4')));
    expect(root.querySelector('[role=alert]')?.textContent).toBe('No such file: /mock/missing.mp4');
    expect(root.querySelector('video')).toBeNull();
  });

  it('opens the video named when the server starts', async () => {
    const e = new MockEngine();
    const session = e.session.bind(e);
    e.session = async () => ({ ...(await session()), media: [e.media_], start_media: e.media_.id });
    await boot(e);
    expect(root.querySelector('video')).not.toBeNull();
  });
});

describe('editing with the keys', () => {
  beforeEach(async () => {
    await boot();
    await act(async () => void (await store.openPath(MOCK_PATH)));
    await tick(5);
  });

  it('S splits at the playhead and Cmd+Z undoes it', async () => {
    await act(() => store.seek(8));
    expect(count()).toBe('3');
    await key('s');
    expect(count()).toBe('4');
    expect(store.edit.value.clips.slice(0, 2).map((c) => [c.start, c.end])).toEqual([[5, 8], [8, 12]]);
    expect(root.querySelector('[data-testid=diff]')?.textContent).toBe('edited: 1 changed, 1 added');
    await key('z', { metaKey: true });
    expect(count()).toBe('3');
    expect(root.querySelector('[data-testid=diff]')).toBeNull();
    await key('z', { metaKey: true, shiftKey: true });
    expect(count()).toBe('4');
  });

  it('saves the edits after the pause, with the stored revision', async () => {
    await act(() => store.seek(8));
    await key('s');
    expect(root.querySelector('.saved')?.textContent).toBe('Saving...');
    await tick(30);
    expect(engine.calls.filter((c) => c.startsWith('saveEdits'))).toEqual(['saveEdits mockrun00001 r0 n4']);
    expect(root.querySelector('.saved')?.textContent).toBe('Saved');
    expect(store.revision.value).toBe(1);
    await key('Delete');
    await tick(30);
    expect(engine.calls.at(-1)).toBe('saveEdits mockrun00001 r1 n3');
  });

  it('says why a split is refused instead of doing nothing', async () => {
    await act(() => store.seek(15));
    await key('s');
    expect(count()).toBe('3');
    expect(root.querySelector('.toast')?.textContent).toContain('not inside a clip');
  });

  it('moves between clips with the arrows, trims with I and O, deletes and adds', async () => {
    await key('ArrowDown');
    expect(store.selection.value).toEqual(['d1']);
    expect(store.time.value).toBe(5);
    await key('ArrowDown');
    expect(store.selection.value).toEqual(['d2']);
    expect(store.time.value).toBe(20);
    await act(() => store.seek(22));
    await key('i');
    expect(store.edit.value.clips[1].start).toBe(22);
    await act(() => store.seek(26));
    await key('o');
    expect(store.edit.value.clips[1].end).toBe(26);
    await key('Delete');
    expect(count()).toBe('2');
    await act(() => store.seek(30));
    await key('n');
    expect(count()).toBe('3');
    expect(store.edit.value.clips[1]).toMatchObject({ start: 29, end: 31, origin: 'added' });
    await key('ArrowUp');
    expect(store.selection.value).toEqual(['d1']);
  });

  it('merges the selected clips with M', async () => {
    await act(() => store.select(['d1', 'd2'], false));
    await key('m');
    expect(count()).toBe('2');
    expect(store.edit.value.clips[0]).toMatchObject({ start: 5, end: 28.5, origin: 'merge' });
  });

  it('steps by frames and seconds, and keeps the playhead inside the video', async () => {
    await act(() => store.seek(10));
    await key('ArrowRight');
    expect(store.time.value).toBeCloseTo(10 + 1 / 30, 6);
    await key('ArrowRight', { shiftKey: true });
    expect(store.time.value).toBeCloseTo(11 + 1 / 30, 6);
    await act(() => store.seek(-5));
    expect(store.time.value).toBe(0);
    await act(() => store.seek(500));
    expect(store.time.value).toBe(60);
  });

  it('toggles clips-only and boxes with P and B', async () => {
    await key('p');
    await key('b');
    expect(store.clipsOnly.value).toBe(true);
    expect(store.showBoxes.value).toBe(true);
    expect(root.querySelectorAll('.transport .tog.on')).toHaveLength(2);
  });

  it('ignores the keys while typing in a field', async () => {
    const input = document.createElement('input');
    root.appendChild(input);
    await act(() => store.seek(8));
    await key('s', {}, input);
    expect(count()).toBe('3');
  });

  it('opens the shortcut sheet with ? and closes it with Esc; keys do nothing meanwhile', async () => {
    await key('?', { shiftKey: true });
    const dlg = root.querySelector('[role=dialog]');
    expect(dlg?.textContent).toContain('Split at the playhead');
    await act(() => store.seek(8));
    await key('s');
    expect(count()).toBe('3');
    await key('Escape');
    expect(root.querySelector('[role=dialog]')).toBeNull();
  });

  it('shows a conflict and stops saving until the page is reloaded from the server', async () => {
    engine.failSave = 'conflict';
    await act(() => store.seek(8));
    await key('s');
    await tick(30);
    expect(root.querySelector('.saved.bad')?.textContent).toBe('Changed in another tab - Reload');
    await act(async () => void (root.querySelector('.saved.bad') as HTMLButtonElement).click());
    await tick(10);
    expect(count()).toBe('3');
    expect(store.canUndo.value).toBe(false); // the history is dropped
    expect(root.querySelector('.saved')?.textContent).toBe('Saved');
  });

  it('retries a failed save when the server answers again', async () => {
    engine.failSave = 'down';
    await act(() => store.seek(8));
    await key('s');
    await tick(30);
    expect(engine.calls.filter((c) => c.startsWith('saveEdits')).length).toBeGreaterThanOrEqual(2);
    expect(root.querySelector('.saved')?.textContent).toBe('Saved');
  });
});

describe('a run with edits already stored', () => {
  it('shows the stored clips and counts the difference to the detector', async () => {
    const e = new MockEngine();
    e.edits = { format: 1, base: { segments_sha256: 'x', n_clips: 3 }, revision: 4, updated: '', next_id: 9,
      clips: [{ id: 'd1', start: 5, end: 12, origin: 'detector' }, { id: 'u8', start: 30, end: 32, origin: 'added' }] };
    await boot(e);
    await act(async () => void (await store.openPath(MOCK_PATH)));
    expect(count()).toBe('2');
    expect(root.querySelector('[data-testid=diff]')?.textContent).toBe('edited: 2 deleted, 1 added');
    expect(store.revision.value).toBe(4);
  });
  it('has an empty state for a video with no run', async () => {
    await boot(new MockEngine({}, false));
    await act(async () => void (await store.openPath(MOCK_PATH)));
    expect(root.textContent).toContain('Pick a target and press Run.');
    expect(count()).toBe('0');
  });
});
