import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH, MOCK_RUN_ID } from '../src/engine/mock';
import type { EditsDoc, ResegmentParams, RunData } from '../src/engine/types';
import { createStore, type Store } from '../src/state';

let root: HTMLElement;
let store: Store;
let engine: MockEngine;

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
const key = (k: string, extra: KeyboardEventInit = {}, target: EventTarget = document.body) =>
  act(() => void target.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...extra })));
const ptr = (type: string, x: number) =>
  new ((window as any).PointerEvent ?? MouseEvent)(type, { clientX: x, button: 0, bubbles: true, cancelable: true });
// Preact listens for 'PointerDown' when the DOM has no onpointerdown property (jsdom), 'pointerdown' otherwise
const down = (el: Element, x: number) =>
  act(() => {
    el.dispatchEvent(ptr('pointerdown', x));
    if (!('onpointerdown' in el)) el.dispatchEvent(ptr('PointerDown', x));
  });
const block = (id: string) => root.querySelector(`[role=button][data-clip=${id}]`) as HTMLElement;
const handle = (id: string, edge: 'start' | 'end') => block(id).querySelector(`[data-edge=${edge}]`) as HTMLElement;
const count = () => root.querySelector('[data-testid=clip-count]')?.textContent;
const clips = () => store.edit.value.clips.map((c) => [c.start, c.end]);
const button = (label: string) => root.querySelector(`button[aria-label="${label}"]`) as HTMLButtonElement;
const clickShift = (el: Element) => act(() => void el.dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true })));

async function boot(e: MockEngine = new MockEngine(), width = 1200) {
  engine = e;
  store = createStore(engine, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5 });
  await store.init();
  await act(async () => void render(<App store={store} />, root));
  await act(async () => void (await store.openPath(MOCK_PATH)));
  await tick(5);
  await act(() => store.setTimelineWidth(width)); // 60 s in 1200 px: 20 px a second, left edge at 0
}

beforeAll(() => {
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
  root = document.createElement('div');
  document.body.appendChild(root);
});
afterEach(() => {
  vi.restoreAllMocks();
  render(null, root);
  store.dispose();
  root.remove();
});

describe('trimming by dragging an edge', () => {
  it('clamps at the neighbour, seeks to the dragged edge, and is one undo step', async () => {
    await boot();
    expect(store.view.value.pps).toBeCloseTo(20, 6);
    await down(handle('d2', 'start'), 400);
    expect(store.drag.value).toMatchObject({ id: 'd2', edge: 'start' });
    await act(() => void window.dispatchEvent(ptr('pointermove', 330))); // 16.5 s
    expect(store.edit.value.clips[1].start).toBeCloseTo(16.5, 1);
    expect(store.time.value).toBeCloseTo(16.5, 1); // the player follows the edge
    expect(root.querySelector('.tip')?.textContent).toContain('·');
    await act(() => void window.dispatchEvent(ptr('pointermove', 100))); // 5 s, far inside the clip before it
    await tick(40);
    expect(store.edit.value.clips[1].start).toBe(12); // the end of clip 1
    expect(store.time.value).toBe(12);
    await act(() => void window.dispatchEvent(ptr('pointerup', 100)));
    expect(store.drag.value).toBeNull();
    expect(store.edit.value.clips[1].origin).toBe('edited');
    await act(() => store.undo());
    expect(store.edit.value.clips[1]).toMatchObject({ start: 20, end: 28.5, origin: 'detector' });
    expect(store.canUndo.value).toBe(false); // the whole drag was one step
  });

  it('seeks at most once per frame while dragging, and ends on the latest edge', async () => {
    await boot();
    store.snapOn.value = false;
    await down(handle('d2', 'end'), 570);
    await act(() => void window.dispatchEvent(ptr('pointermove', 580))); // 29 s: seeks at once
    expect(store.time.value).toBeCloseTo(29, 6);
    await act(() => void window.dispatchEvent(ptr('pointermove', 590)));
    await act(() => void window.dispatchEvent(ptr('pointermove', 600))); // 30 s: waits for the frame
    expect(store.time.value).toBeCloseTo(29, 6);
    await tick(40);
    expect(store.time.value).toBeCloseTo(30, 6);
    await act(() => void window.dispatchEvent(ptr('pointerup', 600)));
    expect(store.edit.value.clips[1].end).toBeCloseTo(30, 6);
  });

  it('snaps to a neighbouring clip edge, and Alt skips the snap', async () => {
    await boot();
    await down(handle('d2', 'start'), 400);
    await act(() => void window.dispatchEvent(ptr('pointermove', 245))); // 12.25 s: within 8 px of 12 (the end of clip 1)
    expect(store.edit.value.clips[1].start).toBe(12);
    const free = new ((window as any).PointerEvent ?? MouseEvent)('pointermove', { clientX: 245, altKey: true, bubbles: true });
    await act(() => void window.dispatchEvent(free));
    await tick(40);
    expect(store.edit.value.clips[1].start).toBeCloseTo(12.25, 2);
    await act(() => void window.dispatchEvent(ptr('pointerup', 245)));
  });

  it('keeps the clip as it was when a drag ends where it began', async () => {
    await boot();
    await down(handle('d1', 'end'), 240);
    await act(() => void window.dispatchEvent(ptr('pointermove', 280)));
    await act(() => void window.dispatchEvent(ptr('pointermove', 240)));
    await act(() => void window.dispatchEvent(ptr('pointerup', 240)));
    expect(store.canUndo.value).toBe(false);
    expect(store.edit.value.clips[0]).toMatchObject({ end: 12, origin: 'detector' });
  });
});

describe('keys and buttons on the clips', () => {
  it('I and O set the edges to the playhead, clamped at the neighbours', async () => {
    await boot();
    await act(() => store.select(['d2'], false));
    await act(() => store.seek(3));
    await key('i');
    expect(store.edit.value.clips[1].start).toBe(12);
    await act(() => store.seek(23));
    await key('i');
    expect(store.edit.value.clips[1].start).toBe(23);
    await act(() => store.seek(55));
    await key('o');
    expect(store.edit.value.clips[1].end).toBe(40);
  });

  it('splits at the playhead from the toolbar and merges a shift-selection', async () => {
    await boot();
    await act(() => store.seek(8));
    await act(() => button('Split').click());
    expect(clips().slice(0, 2)).toEqual([[5, 8], [8, 12]]);
    await act(() => block('d1').click());
    expect(store.time.value).toBe(5);
    await clickShift(block('d3'));
    expect(store.selection.value).toEqual(['d1', expect.any(String), 'd2', 'd3']);
    await act(() => button('Merge').click());
    expect(clips()).toEqual([[5, 47]]);
    expect(store.edit.value.clips[0].origin).toBe('merge');
  });

  it('deletes the selected clip and adds one with a double click on an empty part of the track', async () => {
    await boot();
    await act(() => block('d2').click());
    await act(() => button('Delete').click());
    expect(count()).toBe('2');
    const lane = root.querySelector('.lane.clips') as HTMLElement;
    await act(() => void lane.dispatchEvent(new MouseEvent('dblclick', { clientX: 600, bubbles: true }))); // 30 s
    expect(clips()).toEqual([[5, 12], [29, 31], [40, 47]]);
    expect(store.edit.value.clips[1].origin).toBe('added');
    await act(() => void block('d1').dispatchEvent(new MouseEvent('dblclick', { clientX: 150, bubbles: true })));
    expect(count()).toBe('3'); // a double click on a clip adds nothing
  });

  it('reverts to the detector’s clips as one undoable step, after asking with the count', async () => {
    await boot();
    await act(() => store.seek(8));
    await key('s');
    expect(root.textContent).toContain('Revert to detector');
    await act(() => void ([...root.querySelectorAll('button')].find((b) => b.textContent === 'Revert to detector') as HTMLButtonElement).click());
    expect(root.textContent).toContain('Undo 2 edits?');
    await act(() => void ([...root.querySelectorAll('button')].find((b) => b.textContent === 'Revert') as HTMLButtonElement).click());
    expect(count()).toBe('3');
    expect(root.querySelector('[data-testid=diff]')).toBeNull();
    await act(() => store.undo());
    expect(count()).toBe('4');
  });
});

describe('accessibility', () => {
  it('labels each block and keeps them in time order for Tab', async () => {
    await boot();
    const blocks = [...root.querySelectorAll('[role=button][data-clip]')] as HTMLElement[];
    expect(blocks.map((b) => b.getAttribute('aria-label'))).toEqual([
      'Clip 1, 00:05.0 to 00:12.0, 7 seconds',
      'Clip 2, 00:20.0 to 00:28.5, 8.5 seconds',
      'Clip 3, 00:40.0 to 00:47.0, 7 seconds',
    ]);
    expect(blocks.every((b) => b.tabIndex === 0)).toBe(true);
    expect(root.querySelector('[role=listbox]')?.contains(blocks[0])).toBe(true);
    expect(root.querySelector('[role=listbox]')?.getAttribute('aria-label')).toBe('Clips');
  });

  it('Enter on a block selects it and seeks to its start', async () => {
    await boot();
    await key('Enter', {}, block('d3'));
    expect(store.selection.value).toEqual(['d3']);
    expect(store.time.value).toBe(40);
  });

  it('[ and ] move the focused block’s start and end by a frame, Shift by a second, Alt the other way', async () => {
    await boot();
    await key('[', { code: 'BracketLeft' }, block('d2'));
    expect(store.edit.value.clips[1].start).toBeCloseTo(20 - 1 / 30, 3);
    expect(store.time.value).toBeCloseTo(20 - 1 / 30, 3); // the player goes to the moved edge
    await key(']', { code: 'BracketRight' }, block('d2'));
    expect(store.edit.value.clips[1].end).toBeCloseTo(28.5 + 1 / 30, 3);
    await key('}', { code: 'BracketRight', shiftKey: true }, block('d2'));
    expect(store.edit.value.clips[1].end).toBeCloseTo(29.5 + 1 / 30, 3);
    await key('[', { code: 'BracketLeft', altKey: true }, block('d2'));
    expect(store.edit.value.clips[1].start).toBeCloseTo(20, 3);
    await key('{', { code: 'BracketLeft', shiftKey: true }, block('d2'));
    expect(store.edit.value.clips[1].start).toBeCloseTo(19, 3);
  });

  it('shows the keyframe a stream copy starts at, hatched when it is early', async () => {
    await boot();
    const ghost = root.querySelector('.kfghost') as HTMLElement;
    expect(ghost.title).toBe('stream copy starts 1.0 s early at the keyframe');
    expect(block('d1').classList.contains('hatched')).toBe(true);
  });
});

describe('inspector and clip strip', () => {
  it('shows the clip’s numbers, and the slow-motion line with source and output seconds', async () => {
    class SlowRun extends MockEngine {
      async run(id: string): Promise<RunData> {
        const r = await super.run(id);
        return { ...r, segments: { ...r.segments, slowmo: { factor: 8 }, clips: [{ index: 1, start: 5, end: 7, peak_score: 0.9 }] } };
      }
    }
    await boot(new SlowRun());
    expect(root.querySelector('[aria-label="Selected clip"]')?.textContent).toContain('Select a clip');
    await act(() => block('d1').click());
    const card = root.querySelector('[aria-label="Selected clip"]')!;
    expect(card.textContent).toContain('Clip 1 of 1');
    expect(card.textContent).toContain('00:05.00');
    expect(card.textContent).toContain('2 s');
    expect(card.textContent).toMatch(/Peak score0\.\d\d/); // from the samples
    expect(root.querySelector('[data-testid=slowmo]')?.textContent).toBe('slow-mo x8: source 2.0 s, output 16.0 s');
  });

  it('has a card per clip; clicking one selects it, seeks, and loads a sharper frame', async () => {
    await boot();
    const cards = [...root.querySelectorAll('[data-testid=clip-list] li button')] as HTMLElement[];
    expect(cards).toHaveLength(3);
    await act(() => cards[1].click());
    expect(store.selection.value).toEqual(['d2']);
    expect(store.time.value).toBe(20);
    expect(cards[1].classList.contains('sel')).toBe(true);
    expect(cards[1].style.backgroundImage).toContain('mock://frame/');
  });

  it('shows the run’s warnings', async () => {
    class Warned extends MockEngine {
      async run(id: string): Promise<RunData> {
        const r = await super.run(id);
        return { ...r, segments: { ...r.segments, warnings: ['The camera does not look static.'] } };
      }
    }
    await boot(new Warned());
    expect(root.querySelector('.warn')?.textContent).toBe('The camera does not look static.');
  });

  it('says where the empty states are: no run, and a run with no clips', async () => {
    await boot(new MockEngine({}, false));
    expect(root.querySelector('.lane .empty')?.textContent).toContain('Pick a target and press Run.');
    render(null, root);
    store.dispose();
    class NoClips extends MockEngine {
      async run(id: string): Promise<RunData> {
        const r = await super.run(id);
        return { ...r, segments: { ...r.segments, clips: [] } };
      }
    }
    await boot(new NoClips());
    expect(root.querySelector('.lane .empty')?.textContent).toMatch(/No stretch reached the enter threshold \(highest score 0\.\d\d of 0\.50\)/);
  });
});

describe('re-segment', () => {
  class Reseg extends MockEngine {
    asked: ResegmentParams[] = [];
    async resegment(id: string, p: ResegmentParams) {
      this.asked.push(p);
      return super.resegment(id, p);
    }
  }

  it('is seeded from the run, previews the clips, and Apply is one undo step', async () => {
    const e = new Reseg();
    await boot(e);
    await act(() => store.seek(8));
    await key('s');
    expect(count()).toBe('4');
    await act(() => void (root.querySelector('.resegment .head') as HTMLButtonElement).click());
    await tick(10);
    expect(e.asked[0]).toEqual({ enter: 0.5, exit: 0.35, merge_gap: 1.5, min_len: 1, pad: 1 });
    expect(root.querySelector('[data-testid=reseg-count]')?.textContent).toBe('3 clips (as the run made them)');
    const apply = [...root.querySelectorAll('.resegment button')].find((b) => b.textContent === 'Apply') as HTMLButtonElement;
    expect(apply.disabled).toBe(false);
    await act(() => apply.click());
    expect(count()).toBe('3');
    expect(store.edit.value.clips.every((c) => c.origin === 'resegment')).toBe(true);
    await act(() => store.undo());
    expect(count()).toBe('4'); // one step back to the split state
    expect(clips().slice(0, 2)).toEqual([[5, 8], [8, 12]]);
  });

  it('sends the slider values after a pause', async () => {
    const e = new Reseg();
    await boot(e);
    await act(() => void (root.querySelector('.resegment .head') as HTMLButtonElement).click());
    await tick(10);
    const pad = root.querySelector('input[aria-label="Pad"]') as HTMLInputElement;
    await act(() => { pad.value = '2.5'; pad.dispatchEvent(new Event('input', { bubbles: true })); });
    await tick(250);
    expect(e.asked.at(-1)).toMatchObject({ pad: 2.5, enter: 0.5 });
  });
});

describe('several runs of one video', () => {
  const RUN_B = 'mockrun00002';
  class TwoRuns extends MockEngine {
    editsB: EditsDoc = { format: 1, base: { segments_sha256: 'b', n_clips: 1 }, revision: 3, updated: '', next_id: 5,
      clips: [{ id: 'd1', start: 30, end: 34, origin: 'detector' }, { id: 'u4', start: 50, end: 52, origin: 'added' }] };
    async runs() {
      const [a] = await super.runs();
      return [a, { ...a, run_id: RUN_B, slug: 'saved-alice', target: 'saved:alice', n_clips: 1 }];
    }
    async run(id: string): Promise<RunData> {
      const r = await super.run(id);
      if (id !== RUN_B) return r;
      return { ...r, run_id: RUN_B, segments: { ...r.segments, clips: [{ index: 1, start: 30, end: 34 }] }, edits: this.editsB };
    }
  }

  it('lists the runs, draws a thin bar for the others, and switching loads the other run’s edits', async () => {
    await boot(new TwoRuns());
    const select = root.querySelector('select[aria-label=Run]') as HTMLSelectElement;
    expect([...select.options].map((o) => o.textContent)).toEqual([expect.stringMatching(/^tap · \d\d:\d\d$/), expect.stringMatching(/^saved: alice · \d\d:\d\d$/)]);
    expect(store.run.value?.run_id).toBe(MOCK_RUN_ID);
    expect(root.querySelectorAll('.tl-row.other')).toHaveLength(1);
    expect(Object.keys(store.samplesCache.value)).toContain(RUN_B);
    await act(() => { select.value = RUN_B; select.dispatchEvent(new Event('change', { bubbles: true })); });
    await tick(10);
    expect(store.run.value?.run_id).toBe(RUN_B);
    expect(clips()).toEqual([[30, 34], [50, 52]]);
    expect(store.revision.value).toBe(3);
    expect(store.canUndo.value).toBe(false);
    // the first run is now a thin bar; clicking it switches back
    const bar = root.querySelector('.tl-row.other .lane') as HTMLElement;
    expect(bar.getAttribute('aria-label')).toMatch(/^Switch to run tap/);
    await act(() => bar.click());
    await tick(10);
    expect(store.run.value?.run_id).toBe(MOCK_RUN_ID);
    expect(count()).toBe('3');
  });

  it('saves pending edits before it switches', async () => {
    await boot(new TwoRuns());
    await act(() => store.seek(8));
    await key('s');
    await act(async () => void (await store.switchRun(RUN_B)));
    expect(engine.calls.filter((c) => c.startsWith('saveEdits'))).toEqual(['saveEdits mockrun00001 r0 n4']);
    expect(store.run.value?.run_id).toBe(RUN_B);
  });
});

describe('undo history across a reload', () => {
  it('comes back from sessionStorage when the stored clips are the ones it ends in', async () => {
    const e = new MockEngine();
    await boot(e);
    await act(() => store.seek(8));
    await key('s');
    await act(() => store.seek(30));
    await key('n');
    await tick(30); // autosave
    expect(e.edits?.clips).toHaveLength(5);
    expect(sessionStorage.getItem(`inframe.history.${MOCK_RUN_ID}`)).toContain('"past"');
    render(null, root);
    store.dispose();
    await boot(e); // the "reloaded" page
    expect(count()).toBe('5');
    expect(store.canUndo.value).toBe(true);
    await act(() => store.undo());
    expect(count()).toBe('4');
    await act(() => store.undo());
    expect(count()).toBe('3');
    expect(store.canUndo.value).toBe(false);
    await act(() => store.redo());
    expect(count()).toBe('4');
  });

  it('starts empty when the stored clips are not the ones loaded, and says so in the undo tooltip', async () => {
    const e = new MockEngine();
    await boot(e);
    await act(() => store.seek(8));
    await key('s');
    // not saved yet: the server still holds the detector's clips
    render(null, root);
    store.dispose();
    await boot(e);
    expect(store.canUndo.value).toBe(false);
    expect(button('Undo').title).toContain('new tab starts with an empty history');
  });

  it('is dropped silently when storage throws', async () => {
    const e = new MockEngine();
    await boot(e);
    vi.spyOn(Object.getPrototypeOf(sessionStorage), 'setItem').mockImplementation(() => { throw new Error('quota'); });
    vi.spyOn(Object.getPrototypeOf(sessionStorage), 'getItem').mockImplementation(() => { throw new Error('blocked'); });
    await act(() => store.seek(8));
    await key('s');
    expect(count()).toBe('4'); // editing is not affected
    expect(store.canUndo.value).toBe(true);
    await tick(30);
    render(null, root);
    store.dispose();
    await boot(e);
    expect(count()).toBe('4');
    expect(store.canUndo.value).toBe(false);
    expect(root.querySelector('.toast')).toBeNull();
  });

  it('is cleared when the page reloads the run after a conflict', async () => {
    const e = new MockEngine();
    await boot(e);
    e.failSave = 'conflict';
    await act(() => store.seek(8));
    await key('s');
    await tick(30);
    expect(sessionStorage.getItem(`inframe.history.${MOCK_RUN_ID}`)).not.toBeNull();
    await act(async () => void (await store.reloadRun()));
    expect(sessionStorage.getItem(`inframe.history.${MOCK_RUN_ID}`)).toBeNull();
    expect(store.canUndo.value).toBe(false);
  });
});

describe('zoom', () => {
  it('zooms with the keys around the playhead, fits again, and keeps fit when the window resizes', async () => {
    await boot();
    await act(() => store.seek(30));
    await key('+');
    expect(store.view.value.pps).toBeCloseTo(30, 6);
    expect(store.view.value.t0).toBeCloseTo(30 - (30 - 0) * (20 / 30), 6); // the playhead keeps its x (600 px)
    await key('-');
    expect(store.view.value.pps).toBeCloseTo(20, 6);
    expect(store.view.value.t0).toBe(0);
    await key('+');
    await key('0');
    expect(store.view.value.pps).toBeCloseTo(20, 6);
    await act(() => store.setTimelineWidth(600));
    expect(store.view.value).toMatchObject({ t0: 0, width: 600 });
    expect(store.view.value.pps).toBeCloseTo(10, 6);
  });

  it('zooms at the pointer with Cmd+wheel and pans with a plain wheel', async () => {
    await boot();
    const stack = root.querySelector('.tl-stack') as HTMLElement;
    await act(() => void stack.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, metaKey: true, clientX: 92 + 400, bubbles: true, cancelable: true })));
    const v = store.view.value;
    expect(v.pps).toBeGreaterThan(20);
    expect((400 / 20 - v.t0) * v.pps).toBeCloseTo(400, 3); // 20 s stays under the pointer
    await act(() => void stack.dispatchEvent(new WheelEvent('wheel', { deltaY: 200, bubbles: true, cancelable: true })));
    expect(store.view.value.t0).toBeGreaterThan(v.t0);
  });

  it('reads deltaMode: a Firefox line wheel pans like the same pixels', async () => {
    await boot();
    const stack = root.querySelector('.tl-stack') as HTMLElement;
    await act(() => void stack.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, metaKey: true, clientX: 92, bubbles: true, cancelable: true })));
    await act(() => store.panBy(100));
    const t0 = store.view.value.t0;
    await act(() => void stack.dispatchEvent(new WheelEvent('wheel', { deltaY: 3, deltaMode: 1, bubbles: true, cancelable: true })));
    const lines = store.view.value.t0 - t0;
    await act(() => store.panBy(-48));
    const t1 = store.view.value.t0;
    await act(() => void stack.dispatchEvent(new WheelEvent('wheel', { deltaY: 48, deltaMode: 0, bubbles: true, cancelable: true })));
    expect(lines).toBeGreaterThan(0);
    expect(store.view.value.t0 - t1).toBeCloseTo(lines, 6);
  });

  it('scrubs the playhead from the ruler', async () => {
    await boot();
    const lane = root.querySelector('.tl-row .lane') as HTMLElement; // the ruler
    await down(lane, 300);
    expect(store.time.value).toBeCloseTo(15, 6);
    await act(() => void window.dispatchEvent(ptr('pointermove', 400)));
    expect(store.time.value).toBeCloseTo(20, 6);
    await act(() => void window.dispatchEvent(ptr('pointerup', 400)));
  });
});
