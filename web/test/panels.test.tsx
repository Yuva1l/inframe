import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import { EngineError, type EditsDoc, type Job, type MusicAutoResult } from '../src/engine/types';
import { createStore, type Store } from '../src/state';

let root: HTMLElement;
let store: Store;
let engine: MockEngine;

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
async function until(cond: () => unknown, what = 'condition') {
  for (let i = 0; i < 100; i++) {
    if (cond()) return;
    await tick(10);
  }
  throw new Error(`timed out waiting for ${what}`);
}
const key = (k: string, extra: KeyboardEventInit = {}) =>
  act(() => void document.body.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...extra })));
const q = <T extends HTMLElement = HTMLElement>(sel: string) => root.querySelector(sel) as T;
const text = (sel: string) => q(sel)?.textContent ?? '';
const withText = <T extends HTMLElement = HTMLElement>(tag: string, t: string, exact = true) =>
  [...root.querySelectorAll(tag)].find((e) => (exact ? e.textContent?.trim() === t : e.textContent?.includes(t))) as T;
const click = (el: Element) => act(() => void (el as HTMLElement).click());
const setValue = (el: HTMLInputElement | HTMLSelectElement, v: string, ev = 'change') =>
  act(() => {
    el.value = v;
    el.dispatchEvent(new Event(ev, { bubbles: true }));
  });
const setChecked = (el: HTMLInputElement, v: boolean) =>
  act(() => {
    el.checked = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
const tabBtn = (name: string) => withText('button[role=tab]', name);
const checkbox = (label: string) => withText<HTMLLabelElement>('label.check', label, false)?.querySelector('input') as HTMLInputElement;
const ptr = (type: string, x: number, y: number) =>
  new ((window as any).PointerEvent ?? MouseEvent)(type, { clientX: x, clientY: y, button: 0, bubbles: true, cancelable: true });
// Preact listens for 'PointerDown' when the DOM has no onpointerdown property (jsdom), 'pointerdown' otherwise
const down = (el: Element, x: number, y: number) =>
  act(() => {
    el.dispatchEvent(ptr('pointerdown', x, y));
    if (!('onpointerdown' in el)) el.dispatchEvent(ptr('PointerDown', x, y));
  });
const winMove = (x: number, y: number) => act(() => void window.dispatchEvent(ptr('pointermove', x, y)));
const winUp = (x: number, y: number) => act(() => void window.dispatchEvent(ptr('pointerup', x, y)));
const dlg = () => q('[role=dialog][aria-label=Export]');
const inDlg = <T extends HTMLElement = HTMLButtonElement>(tag: string, t: string) =>
  [...dlg().querySelectorAll(tag)].find((e) => e.textContent?.trim() === t) as T;
const runButton = () => withText<HTMLButtonElement>('button.runbtn', 'Run');

async function boot(e: MockEngine = new MockEngine()) {
  engine = e;
  store = createStore(engine, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5, pollMs: 5 });
  await store.init();
  await act(async () => void render(<App store={store} />, root));
  await act(async () => void (await store.openPath(MOCK_PATH)));
  await tick(5);
}

const EDITS: EditsDoc = {
  format: 1, base: { segments_sha256: 'x', n_clips: 3 }, revision: 1, updated: '', next_id: 2,
  clips: [{ id: 'd1', start: 6, end: 12, origin: 'edited' }, { id: 'd2', start: 20, end: 28.5, origin: 'detector' }, { id: 'u1', start: 50, end: 52, origin: 'added' }],
};

beforeAll(() => {
  HTMLMediaElement.prototype.pause = () => {};
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
  root = document.createElement('div');
  document.body.appendChild(root);
  // the player is 800x450 for a 1280x720 video: the frame fills it
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

describe('the target panel', () => {
  it('keeps taps when the tab changes, and the summary card names the one active target', async () => {
    await boot();
    expect(text('[data-testid=summary-title]')).toBe('Anyone: every person');
    await click(tabBtn('Tap'));
    expect(store.playing.value).toBe(false);
    await until(() => text('[data-testid=pick-status]').includes('2 found'), 'boxes');
    expect(q<HTMLImageElement>('img.pickimg').src).toContain('mock://pickframe/');
    // a click inside box 1 (x 10-30 %, y 10-90 % of the 800x450 frame) taps it
    await act(() => void q('[data-testid=pick-layer]').dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: 100, clientY: 200 })));
    await until(() => text('[data-testid=tap-count]') === '1 tap', 'the tap');
    expect(engine.calls.filter((c) => c.startsWith('tap person')).length).toBe(1);
    expect(text('[data-testid=summary-title]')).toBe('Tap: person, 1 tap');

    await click(tabBtn('Anyone'));
    expect(text('[data-testid=summary-title]')).toBe('Anyone: every person');
    expect(text('[data-testid=summary-kept]')).toContain('1 tap is kept in the Tap tab');
    await click(tabBtn('Saved'));
    await until(() => withText('button.crop', 'alice · 4'), 'saved targets');
    await click(withText('button.crop', 'alice · 4'));
    expect(text('[data-testid=summary-title]')).toBe('Saved: alice');
    expect(text('[data-testid=summary-kept]')).toContain('kept');
    await click(tabBtn('Tap'));
    expect(text('[data-testid=tap-count]')).toBe('1 tap');
    expect(text('[data-testid=summary-title]')).toBe('Tap: person, 1 tap');
    await click(withText('button', 'Undo last tap'));
    await until(() => text('[data-testid=tap-count]') === '0 taps', 'undo');
  });

  it('shows saved targets with a stale badge and an unreadable one as disabled', async () => {
    await boot();
    await click(tabBtn('Saved'));
    await until(() => root.querySelectorAll('ul[aria-label="Saved targets"] li').length === 3, 'cards');
    const cards = [...root.querySelectorAll<HTMLButtonElement>('ul[aria-label="Saved targets"] button')];
    expect(cards.map((c) => c.disabled)).toEqual([false, false, true]);
    expect(cards[0].textContent).not.toContain('stale');
    expect(cards[1].textContent).toContain('stale');
    expect(cards[2].textContent).toContain('unreadable');
  });

  it('runs a tap target with the detector it was picked with and sends the zone to the picker', async () => {
    await boot();
    await click(tabBtn('Tap'));
    await until(() => text('[data-testid=pick-status]').includes('2 found'), 'boxes');
    expect(runButton().disabled).toBe(true); // no tap yet
    await act(() => void q('[data-testid=pick-layer]').dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: 500, clientY: 200 })));
    await until(() => text('[data-testid=tap-count]') === '1 tap');
    await setValue(q<HTMLSelectElement>('select[aria-label=Preset]'), 'fast'); // refused: the tap forces balanced
    expect(q<HTMLSelectElement>('select[aria-label=Preset]').disabled).toBe(true);
    await click(withText('button', 'Draw zone'));
    await down(q('[data-testid=zone-surface]'), 80, 45);
    await winMove(400, 225);
    await winUp(400, 225);
    await click(withText('button', 'Done'));
    await click(runButton());
    await until(() => engine.lastRun, 'the run request');
    expect(engine.lastRun).toMatchObject({ preset: 'balanced', target: { kind: 'tap', class: 'person' }, zone: [0.1, 0.1, 0.4, 0.4], zone_overlap: null });
    expect(engine.calls.some((c) => c.startsWith('setZone person [0.1,0.1,0.4,0.4]'))).toBe(true);
  });

  it('the photos tab chooses a folder, runs, and asks "Is this you?" when the run cannot decide', async () => {
    const e = new MockEngine();
    e.preflight.identity = true;
    await boot(e);
    await click(tabBtn('Photos'));
    expect(runButton().disabled).toBe(true);
    await click(withText('button', 'Choose folder...'));
    await until(() => text('[data-testid=photo-count]') === '12 photos', 'photos');
    expect(engine.calls).toContain('choosePhotos dialog');
    expect(text('[data-testid=summary-title]')).toBe('Photos: 12 photos');
    engine.script = [{ state: 'needs_confirm', error: 'Two people match the photos equally.' }];
    await click(runButton());
    await until(() => text('.runpanel').includes('Two people match'), 'the needs_confirm note');
    expect(engine.lastRun?.target).toMatchObject({ kind: 'photos', photos_id: 'p1' });
    await click(withText('button', 'Find who it is'));
    await until(() => root.querySelectorAll('ul[aria-label=Candidates] li').length === 3, 'candidates');
    expect(engine.lastRank).toMatchObject({ photos_id: 'p1' });
    await click(root.querySelector('ul[aria-label=Candidates] button')!);
    await until(() => text('[data-testid=summary-title]') === 'Photos + 1 confirm tap', 'the confirm');
    await click(runButton());
    await until(() => engine.lastRun?.target.kind === 'tap', 'the second run');
    expect(engine.lastRun?.target).toMatchObject({ kind: 'tap', photos_id: 'p1' });
  });

  it('lists the people of the video and shows the chosen one in the Tap tab at their best moment', async () => {
    const e = new MockEngine();
    e.caps.people = true;
    await boot(e);
    await click(tabBtn('Tap'));
    await click(withText('button', 'Who is in it?'));
    await until(() => root.querySelectorAll('ul[aria-label="People in the video"] li').length === 2, 'people');
    await click(root.querySelectorAll('ul[aria-label="People in the video"] button')[1]);
    expect(store.time.value).toBeCloseTo(24, 3);
    expect(store.tab.value).toBe('tap');
  });
});

describe('the zone tool', () => {
  it('draws, resizes, moves, cancels with Esc and clears with Delete', async () => {
    await boot();
    expect(text('[data-testid=summary-zone]')).toBe('off');
    await click(withText('button', 'Draw zone'));
    await down(q('[data-testid=zone-surface]'), 80, 45); // 10 %, 10 %
    await winMove(400, 225);
    await winUp(400, 225);
    expect(store.zone.value).toEqual([0.1, 0.1, 0.4, 0.4]);
    expect(store.zoneMode.value).toBe('edit');
    expect(root.querySelectorAll('[data-handle]').length).toBe(8);

    await down(q('[data-handle=se]'), 400, 225);
    await winMove(480, 225);
    await winUp(480, 225);
    expect(store.zone.value).toEqual([0.1, 0.1, 0.5, 0.4]);
    await down(q('[data-testid=zone-box]'), 200, 100);
    await winMove(280, 145);
    await winUp(280, 145);
    expect(store.zone.value).toEqual([0.2, 0.2, 0.5, 0.4]);
    await down(q('[data-handle=nw]'), 160, 90);
    await winMove(10_000, 10_000); // far past the opposite corner: the minimum side holds
    await winUp(10_000, 10_000);
    expect(store.zone.value![2]).toBeCloseTo(0.02, 6);

    await key('Escape'); // cancel: back to no zone
    expect(store.zone.value).toBeNull();
    expect(store.zoneMode.value).toBe('off');

    await click(withText('button', 'Draw zone'));
    await down(q('[data-testid=zone-surface]'), 80, 45);
    await winMove(400, 225);
    await winUp(400, 225);
    await click(withText('button', 'Done'));
    expect(text('[data-testid=summary-zone]')).toContain('on');
    await click(withText('button', 'Edit'));
    expect(store.zoneMode.value).toBe('edit');
    await key('Delete');
    expect(store.zone.value).toBeNull();
    expect(store.edit.value.clips.length).toBe(3); // Delete went to the zone, not to a clip
    expect(text('[data-testid=summary-zone]')).toBe('off');
  });

  it('ignores a drag that is too small and takes the overlap choice into the run request', async () => {
    await boot();
    await click(withText('button', 'Draw zone'));
    await down(q('[data-testid=zone-surface]'), 80, 45);
    await winMove(81, 46);
    await winUp(81, 46);
    expect(store.zone.value).toBeNull();
    await winMove(400, 225); // listeners are gone after the pointer went up
    expect(store.zone.value).toBeNull();
    await down(q('[data-testid=zone-surface]'), 80, 45);
    await winMove(400, 225);
    await winUp(400, 225);
    await click(withText('button', 'Done'));
    await setValue(q<HTMLSelectElement>('select[aria-label="Zone overlap"]'), 'part');
    await click(runButton());
    await until(() => engine.lastRun, 'the request');
    expect(engine.lastRun).toMatchObject({ zone: [0.1, 0.1, 0.4, 0.4], zone_overlap: 0.5 });
  });
});

describe('the run panel', () => {
  it('disables Run for a preset whose model is missing and shows the fix command', async () => {
    await boot();
    expect(runButton().disabled).toBe(false);
    await setValue(q<HTMLSelectElement>('select[aria-label=Preset]'), 'accurate');
    expect(runButton().disabled).toBe(true);
    expect(text('.runpanel .preflight')).toContain('detector model: accurate');
    expect(text('.runpanel .preflight')).toContain('Run `inframe fetch-models`');
    await setValue(q<HTMLSelectElement>('select[aria-label=Preset]'), 'balanced');
    expect(runButton().disabled).toBe(false);
  });

  it('needs the embedder for saved and photo targets, and no model for motion mode', async () => {
    await boot();
    await click(tabBtn('Saved'));
    await until(() => withText('button.crop', 'alice · 4'));
    await click(withText('button.crop', 'alice · 4'));
    expect(runButton().disabled).toBe(true);
    expect(text('.runpanel .preflight')).toContain('Run `uv sync --extra identity`');
    await click(tabBtn('Anyone'));
    await setValue(q<HTMLSelectElement>('select[aria-label=Preset]'), 'accurate');
    expect(runButton().disabled).toBe(true);
    await click(root.querySelector('details.advanced summary')!);
    await setChecked(checkbox('Motion mode'), true); // not a model run: nothing to fetch
    expect(runButton().disabled).toBe(false);
    expect(text('[data-testid=summary-title]')).toContain('Motion');
  });

  it('reads "at least" before any run and "about" after one', async () => {
    await boot();
    expect(text('[data-testid=estimate]')).toMatch(/^at least .* \(estimate\)$/);
    await click(runButton());
    await until(() => text('[data-testid=estimate]').startsWith('about'), 'the measured estimate');
    expect(text('[data-testid=estimate]')).toBe('about 30 s (estimate)'); // 30 s of work for a 60 s video
    expect(engine.lastRun).toMatchObject({ preset: 'balanced', target: { kind: 'any' }, slowmo: 'off', advanced: { pad: 1, enter: null } });
  });

  it('shows progress with the elapsed and left time, a thin bar, and Cancel', async () => {
    await boot();
    engine.script = [{ fraction: 0.5, message: 'detect: 30 of 60 s', elapsed_s: 15 }];
    await click(runButton());
    await until(() => text('[data-testid=job-progress]').includes('50 %'), 'progress');
    expect(text('[data-testid=job-message]')).toBe('detect: 30 of 60 s · 0:15 elapsed · about 0:15 left');
    expect(q('.jobbar')).toBeTruthy();
    expect(runButton().disabled).toBe(true);
    expect(runButton().title).toBe('A job is running');
    expect(withText<HTMLButtonElement>('button', 'Who is in it?')).toBeUndefined();
    await click(withText('button', 'Cancel'));
    await until(() => text('.runpanel').includes('Run cancelled. Nothing was kept.'), 'the cancelled note');
    expect(engine.cancelled).toBe(1);
    expect(q('.jobbar')).toBeNull();
  });

  it('shows no ETA before 5 %', async () => {
    await boot();
    engine.script = [{ fraction: 0.02, elapsed_s: 3, message: 'probe' }];
    await click(runButton());
    await until(() => text('[data-testid=job-message]').includes('probe'));
    expect(text('[data-testid=job-message]')).not.toContain('left');
    await click(withText('button', 'Cancel'));
  });

  it('shows a failed run in a red panel, and a 409 busy as a toast that follows the running job', async () => {
    await boot();
    engine.script = [{ state: 'failed', error: 'score_detect: out of memory' }];
    await click(runButton());
    await until(() => q('.errpanel'), 'the failure');
    expect(text('.errpanel')).toContain('out of memory');
    engine.busy = { id: 'other', kind: 'export', state: 'running', fraction: 0.2, message: 'cutting clip 1 of 5', started: 0, elapsed_s: 4, media_id: null, run_id: null, result: null, error: null };
    engine.script = [{ state: 'running', fraction: 0.2 }];
    await click(runButton());
    await until(() => store.toasts.value.some((t) => t.text === 'A export job is running'), 'the busy toast');
    expect(text('[data-testid=job-progress]')).toContain('Export');
    expect(text('[data-testid=job-progress]')).toContain('20 %');
    await click(withText('button', 'Cancel'));
  });

  it('asks before running again over a run with edits, keeps them aside, and offers to restore them', async () => {
    const e = new MockEngine();
    e.slug = 'any-person';
    e.edits = EDITS;
    await boot(e);
    expect(store.edit.value.clips.map((c) => c.id)).toEqual(['d1', 'd2', 'u1']);
    await click(runButton());
    await until(() => store.rerunAsk.value, 'the question');
    expect(text('[data-testid=rerun-text]')).toBe('This run has 3 edited clips. Run again: your edits are kept aside and can be restored.');
    expect(engine.calls.some((c) => c.startsWith('startRun'))).toBe(false);
    await click(withText('button', 'Cancel'));
    expect(store.rerunAsk.value).toBeNull();
    expect(engine.calls.some((c) => c.startsWith('startRun'))).toBe(false);

    await click(runButton());
    await until(() => store.rerunAsk.value);
    await click(withText('button', 'Run again'));
    await until(() => withText('button', 'Restore my earlier edits'), 'the restore offer');
    expect(store.edit.value.clips.map((c) => c.id)).toEqual(['d1', 'd2', 'd3']); // the new run's own clips
    await click(withText('button', 'Restore my earlier edits'));
    expect(store.edit.value.clips.map((c) => [c.id, c.start, c.end])).toEqual([['d1', 6, 12], ['d2', 20, 28.5], ['u1', 50, 52]]);
    expect(store.edit.value.nextId).toBeGreaterThanOrEqual(2);
    await act(() => store.undo());
    expect(store.edit.value.clips.length).toBe(3);
    expect(store.edit.value.clips[0].id).toBe('d1');
    expect(store.edit.value.clips[0].start).toBe(5); // one undo step: the new run's clips are back
  });

  it('does not ask when the earlier run has no real edits', async () => {
    const e = new MockEngine();
    e.slug = 'any-person';
    e.edits = { ...EDITS, clips: [{ id: 'd1', start: 5, end: 12, origin: 'detector' }, { id: 'd2', start: 20, end: 28.5, origin: 'detector' }, { id: 'd3', start: 40, end: 47, origin: 'detector' }] };
    await boot(e);
    await click(runButton());
    await until(() => engine.lastRun);
    expect(store.rerunAsk.value).toBeNull();
  });

  it('offers the slow-motion controls only for a slow-motion capture', async () => {
    await boot();
    expect(root.querySelector('.slowmo')).toBeNull();
    render(null, root);
    store.dispose();
    await boot(new MockEngine({ capture_fps: 240, fps: 30 }));
    expect(text('.slowmo')).toContain('Slow down clips (auto, x8)');
    await setChecked(checkbox('Slow down clips'), true);
    await setChecked(checkbox('Keep sound'), true);
    await click(runButton());
    await until(() => engine.lastRun);
    expect(engine.lastRun).toMatchObject({ slowmo: 'auto', slowmo_audio: true });
  });

  it('shows the run that is still going after a reload', async () => {
    const e = new MockEngine();
    e.runningAtStart = { id: 'mockjob-run', kind: 'run', state: 'running', fraction: 0.4, message: 'detect: 24 of 60 s', started: 0, elapsed_s: 10, media_id: null, run_id: null, result: null, error: null };
    e.script = [{ fraction: 0.4, elapsed_s: 10, message: 'detect: 24 of 60 s' }];
    await boot(e);
    expect(text('[data-testid=job-progress]')).toContain('40 %');
    expect(runButton().disabled).toBe(true);
    await click(withText('button', 'Cancel'));
  });
});

describe('the export dialog', () => {
  const open = async () => {
    await key('e', { metaKey: true });
    await until(() => q('[role=dialog][aria-label=Export]'), 'the dialog');
  };

  it('opens with Cmd+E and says what it will write, for all clips or the selected ones', async () => {
    await boot();
    await act(() => store.select(['d1']));
    await open();
    expect(text('[data-testid=export-summary]')).toMatch(/^1 clip \(selected\) of 3 - 7 s output - about [\d.]+ [KM]B - 1 clip starts more than 0.5 s early at a keyframe$/);
    expect(withText('label.check', 'Selected only (1)', false).querySelector('input')!.checked).toBe(true);
    await click(withText('label.check', 'All (3)', false).querySelector('input')!);
    expect(text('[data-testid=export-summary]')).toMatch(/^3 clips - 22.5 s output - about [\d.]+ [KM]B - 3 clips start more than 0.5 s early at a keyframe$/);
    expect(text('[data-testid=export-folder]')).toBe('/mock/clips/mock_clip_inframe/tap/edited/');
    expect(text('[role=dialog][aria-label=Export]')).toContain('frame-exact cutting comes later');
  });

  it('Selected only is off without a selection, and Esc closes the dialog without clearing anything else', async () => {
    await boot();
    await open();
    expect(withText('label.check', 'Selected only', false).querySelector('input')!.disabled).toBe(true);
    await act(() => store.select(['d2'], false));
    await key('Escape');
    expect(q('[role=dialog][aria-label=Export]')).toBeNull();
    expect(store.selection.value).toEqual(['d2']);
  });

  it('exports the selected clip ids with the chosen outputs and lists what was written', async () => {
    await boot();
    await act(() => store.select(['d1', 'd3'], false));
    await open();
    await click(checkbox('Reel'));
    await click(checkbox('FCPXML'));
    await click(inDlg('button', 'Export'));
    await until(() => q('[data-testid=export-outputs]'), 'the result');
    expect(engine.lastExport).toMatchObject({ clip_ids: ['d1', 'd3'], out_dir: null, replace: false, clips: true, reel: true, music: null, beat_sync: false, edit_lists: ['fcpxml'] });
    expect(text('[data-testid=export-outputs]')).toContain('mock_clip_reel.mp4');
    expect(text('[data-testid=export-outputs]')).toContain('joins ok');
    expect(withText('button', 'Play')).toBeTruthy();
  });

  it('asks before replacing an earlier export (409 exists), then replaces', async () => {
    await boot();
    engine.exportExists = { clips: [1, 2], reel: { file: 'r.mp4' } };
    await open();
    await click(inDlg('button', 'Export'));
    await until(() => q('[data-testid=replace-text]'), 'the question');
    expect(text('[data-testid=replace-text]')).toBe('Replace the export (2 clips, a reel)?');
    expect(engine.lastExport).toBeNull();
    await click(withText('button', 'Replace'));
    await until(() => engine.lastExport, 'the second request');
    expect(engine.lastExport?.replace).toBe(true);
    await until(() => q('[data-testid=export-outputs]'));
  });

  it('saves unsaved edits first, and an empty clip list cannot be exported', async () => {
    await boot();
    await act(() => store.dispatch('delete'));
    await act(() => store.select(['d1'], false));
    await act(() => store.dispatch('delete'));
    await act(() => store.select(['d2'], false));
    await act(() => store.dispatch('delete'));
    await act(() => store.select(['d3'], false));
    await act(() => store.dispatch('delete'));
    expect(store.edit.value.clips.length).toBe(0);
    await open();
    expect(text('[role=dialog][aria-label=Export]')).toContain('Nothing to export: the clip list is empty.');
    expect(inDlg<HTMLButtonElement>('button', 'Export').disabled).toBe(true);
    await tick(20);
    expect(engine.edits?.clips.length).toBe(0); // the autosave wrote the empty list
  });

  it('saves pending edits before it starts the export job', async () => {
    await boot();
    await act(() => store.seek(8));
    await act(() => store.select(['d1'], false));
    await act(() => store.dispatch('split'));
    expect(store.saveState.value).toBe('saving');
    await act(async () => void (await store.startExport({ clip_ids: null, out_dir: null, replace: false, clips: true, reel: false, music: null, beat_sync: false, edit_lists: [] })));
    const calls = engine.calls;
    expect(calls.findIndex((c) => c.startsWith('saveEdits'))).toBeGreaterThanOrEqual(0);
    expect(calls.findIndex((c) => c.startsWith('saveEdits'))).toBeLessThan(calls.indexOf('startExport'));
    expect(engine.edits?.clips.length).toBe(4);
  });

  it('disables the edit lists a frame rate cannot have, with the reason', async () => {
    await boot(new MockEngine({ fps: 120, capture_fps: 120 }));
    await open();
    const edl = checkbox('EDL');
    expect(edl.disabled).toBe(true);
    expect(text('[role=dialog][aria-label=Export]')).toContain('needs 60 fps or less');
    expect(checkbox('FCPXML').disabled).toBe(false);
    expect(checkbox('OTIO').disabled).toBe(false);
  });

  it('hides the Clips section when the engine cannot stream-copy, and the folder button when it cannot choose one', async () => {
    const e = new MockEngine();
    e.caps.streamCopyCut = false;
    e.caps.chooseOutputFolder = false;
    await boot(e);
    await open();
    expect(checkbox('Clips')).toBeUndefined();
    expect(withText('button', 'Choose folder...')).toBeUndefined();
  });

  it('chooses an output folder with the built-in file browser', async () => {
    await boot();
    await open();
    await click(withText('button', 'Choose folder...'));
    await until(() => q('[role=dialog][aria-label="Choose a folder"]'), 'the browser');
    await click(withText('button', 'Choose this folder'));
    expect(text('[data-testid=export-folder]')).toBe('/mock');
    await click(inDlg('button', 'Export'));
    await until(() => engine.lastExport);
    expect(engine.lastExport?.out_dir).toBe('/mock');
  });
});

describe('music for the reel', () => {
  const openReel = async (e: MockEngine) => {
    await boot(e);
    await key('e', { metaKey: true });
    await until(() => q('[role=dialog][aria-label=Export]'));
    await click(checkbox('Reel'));
  };

  it('sends only the mood field (cleaned), the minimum length, a limit and the CC0 only choice (off by default)', async () => {
    const e = new MockEngine();
    e.caps.music = true;
    await openReel(e);
    expect(text('[data-testid=music-choice]')).toContain('Find music');
    const mood = q<HTMLInputElement>('input[aria-label=Mood]');
    expect(mood.value).toBe('upbeat');
    await setValue(mood, 'chill/../mix; alice', 'input');
    await click(withText('button', 'Search'));
    await until(() => e.musicQueries.length === 1, 'the search');
    expect(Object.keys(e.musicQueries[0]).sort()).toEqual(['cc0_only', 'limit', 'min_s', 'mood']);
    expect(e.musicQueries[0]).toEqual({ mood: 'chillmix alice', min_s: 23, limit: 10, cc0_only: false });
    expect(JSON.stringify(e.musicQueries)).not.toMatch(/mock_clip|\/mock|tap/);
    await until(() => q('[data-testid=music-notice]'), 'the notice');
    expect(text('[data-testid=music-notice]')).toContain('Openverse');
    expect(root.querySelectorAll('ul[aria-label="Music results"] li').length).toBe(2);
    expect(checkbox('CC0 only').checked).toBe(false); // CC0 and CC BY are both offered by default
    expect(root.textContent).not.toContain('Also allow CC BY');
    expect(text('ul[aria-label="Music results"]')).toContain('BY');
    await click(checkbox('CC0 only'));
    await click(withText('button', 'Search'));
    await until(() => e.musicQueries.length === 2);
    expect(e.musicQueries[1].cc0_only).toBe(true);
    await until(() => !text('ul[aria-label="Music results"]').includes('BY'), 'CC0 results only');
  });

  it('plays a track after fetching it, uses one for the export, and takes a file of your own', async () => {
    const e = new MockEngine();
    e.caps.music = true;
    await openReel(e);
    await click(withText('button', 'Search'));
    await until(() => root.querySelectorAll('ul[aria-label="Music results"] li').length === 2);
    const first = root.querySelectorAll('ul[aria-label="Music results"] li')[0];
    await click(first.querySelectorAll('button')[0]); // Play
    await until(() => q('audio[aria-label=Preview]'));
    expect(e.calls).toContain('musicGet trk1');
    expect(q<HTMLAudioElement>('audio[aria-label=Preview]').src).toBe('mock://music/trk1');
    await click(first.querySelectorAll('button')[1]); // Use
    expect(text('[data-testid=music-choice]')).toBe('Using: Bright Day (cc0)');
    await click(inDlg('button', 'Export'));
    await until(() => e.lastExport);
    expect(e.lastExport).toMatchObject({ reel: true, music: { track_id: 'trk1', file: null, cc0_only: false } });
  });

  it('shows the credit text and the credits file name after an export whose music is CC BY, and not for a CC0 track', async () => {
    const e = new MockEngine();
    e.caps.music = true;
    await openReel(e);
    await click(withText('button', 'Search'));
    await until(() => root.querySelectorAll('ul[aria-label="Music results"] li').length === 2);
    await click(root.querySelectorAll('ul[aria-label="Music results"] li')[0].querySelectorAll('button')[1]); // Use the CC0 track
    await click(inDlg('button', 'Export'));
    await until(() => q('[data-testid=export-outputs]'), 'the CC0 result');
    expect(q('[data-testid=music-credit]')).toBeNull();
  });

  it('names the credits file and prints the credit for a CC BY track', async () => {
    const e = new MockEngine();
    e.caps.music = true;
    await openReel(e);
    await click(withText('button', 'Search'));
    await until(() => root.querySelectorAll('ul[aria-label="Music results"] li').length === 2);
    await click(root.querySelectorAll('ul[aria-label="Music results"] li')[1].querySelectorAll('button')[1]); // Use the CC BY track
    expect(text('[data-testid=music-choice]')).toBe('Using: Long Road (by)');
    await click(inDlg('button', 'Export'));
    await until(() => q('[data-testid=music-credit]'), 'the credit');
    expect(e.lastExport).toMatchObject({ music: { track_id: 'trk2', cc0_only: false } });
    expect(text('[data-testid=music-credit]')).toContain('mock_clip_credits.txt');
    expect(text('[data-testid=music-credit-text]')).toContain('Long Road');
    expect(text('[data-testid=export-outputs]')).toContain('mock_clip_credits.txt');

    await click(withText('button', 'Use a music file...'));
    await until(() => text('[data-testid=music-choice]').includes('own.mp3'));
  });

  it('keeps beat sync disabled without the music extra, and enables it when a track is chosen with librosa present', async () => {
    const e = new MockEngine();
    e.caps.music = true;
    await openReel(e);
    await click(withText('button', 'Search'));
    await until(() => root.querySelectorAll('ul[aria-label="Music results"] li').length === 2);
    await click(root.querySelectorAll('ul[aria-label="Music results"] li')[0].querySelectorAll('button')[1]);
    expect(checkbox('Beat sync').disabled).toBe(true);
    expect(text('[role=dialog][aria-label=Export]')).toContain('needs the music extra: uv sync --extra music');

    render(null, root);
    store.dispose();
    const e2 = new MockEngine();
    e2.caps.music = true;
    e2.caps.beatSync = true;
    await openReel(e2);
    expect(checkbox('Beat sync').disabled).toBe(true); // nothing to sync to yet
    await click(withText('button', 'Search'));
    await until(() => root.querySelectorAll('ul[aria-label="Music results"] li').length === 2);
    await click(root.querySelectorAll('ul[aria-label="Music results"] li')[0].querySelectorAll('button')[1]);
    expect(checkbox('Beat sync').disabled).toBe(false);
    expect(text('[role=dialog][aria-label=Export]')).not.toContain('needs the music extra');
    await click(checkbox('Beat sync'));
    await click(inDlg('button', 'Export'));
    await until(() => e2.lastExport);
    expect(e2.lastExport?.beat_sync).toBe(true);
  });

  describe('auto music', () => {
    const autoReel = async (e: MockEngine) => {
      e.caps.music = true;
      e.caps.beatSync = true;
      await openReel(e);
    };
    const pickAuto = async () => {
      await click(root.querySelector('[data-testid=music-auto]') as HTMLInputElement);
      await until(() => q('[data-testid=auto-pick]'), 'the auto pick');
    };

    it('asks for rank 0, shows the title and license, and ticks Beat sync once', async () => {
      const e = new MockEngine();
      await autoReel(e);
      expect(checkbox('Beat sync').checked).toBe(false);
      await pickAuto();
      expect(e.autoQueries).toEqual([{ clip_ids: null, mood: '', cc0_only: false, rank: 0 }]);
      const card = text('[data-testid=auto-pick]');
      expect(card).toContain('Sunny Walk');
      expect(card).toContain('A. Composer');
      expect(card).toContain('2:34');
      expect(card).toContain('CC BY 4.0');
      expect(card).toContain('124 BPM');
      expect(card).toContain('fit 81%');
      expect(card).toContain('1 of 4');
      expect(text('[data-testid=music-choice]')).toBe('Using: Sunny Walk (CC BY 4.0)');
      expect(checkbox('Beat sync').checked).toBe(true);
      await click(checkbox('Beat sync')); // the user may untick it, and another song does not tick it again
      await click(q('[data-testid=another-song]'));
      await until(() => e.autoQueries.length === 2);
      expect(checkbox('Beat sync').checked).toBe(false);
    });

    it('Another song asks for rank 1, then wraps back; the built-in beat says so', async () => {
      const e = new MockEngine();
      await autoReel(e);
      await pickAuto();
      await click(q('[data-testid=another-song]'));
      await until(() => text('[data-testid=auto-pick]').includes('Night Drive'), 'rank 1');
      expect(e.autoQueries[1].rank).toBe(1);
      expect(text('[data-testid=auto-pick]')).toContain('CC0');
      expect(q('[data-testid=auto-credit]')).toBeNull();
      await click(q('[data-testid=another-song]'));
      await until(() => text('[data-testid=auto-pick]').includes('Slow Tide'), 'rank 2');
      await click(q('[data-testid=another-song]'));
      await until(() => text('[data-testid=auto-pick]').includes('built-in beat'), 'the built-in beat');
      expect(text('[data-testid=auto-pick]')).toContain('No song fitted, so this is inframe’s built-in beat (CC0).');
      await click(q('[data-testid=another-song]'));
      await until(() => text('[data-testid=auto-pick]').includes('Sunny Walk'), 'wrapped to rank 0');
      expect(e.autoQueries.map((x) => x.rank)).toEqual([0, 1, 2, 3, 4]);
    });

    it('sends music.auto and beat_sync true with the export, and shows the pick and the credit for a CC BY song', async () => {
      const e = new MockEngine();
      await autoReel(e);
      await setValue(q<HTMLInputElement>('input[aria-label="Auto music mood"]'), 'calm', 'input');
      await pickAuto();
      expect(e.autoQueries[0].mood).toBe('calm');
      expect(text('[data-testid=auto-credit]')).toContain('Music: "Sunny Walk" by A. Composer');
      expect(text('[data-testid=auto-pick]')).toContain('Its credit file is written next to the reel');
      await click(inDlg('button', 'Export'));
      await until(() => e.lastExport);
      expect(e.lastExport).toMatchObject({ reel: true, music: { auto: { mood: 'calm', rank: 0, track: 'auto1' }, cc0_only: false }, beat_sync: true });
      expect(e.lastExport?.music).not.toHaveProperty('track_id');
      await until(() => q('[data-testid=music-credit]'), 'the credit after the export');
      expect(text('[data-testid=music-auto-used]')).toBe('Music: Sunny Walk by A. Composer (auto pick 1 of 4)');
      expect(text('[data-testid=music-credit-text]')).toContain('Sunny Walk');
    });

    it('asks again with rank 0 when the CC0 only box changes, and shows an error without choosing', async () => {
      const e = new MockEngine();
      await autoReel(e);
      await pickAuto();
      await click(checkbox('CC0 only'));
      await until(() => e.autoQueries.length === 2);
      expect(e.autoQueries[1]).toEqual({ clip_ids: null, mood: '', cc0_only: true, rank: 0 });
      await until(() => text('[data-testid=auto-pick]').includes('CC0'), 'CC0');
      expect(q('[data-testid=auto-credit]')).toBeNull();
      e.autoError = 'music: Openverse is rate-limiting (test)';
      await click(q('[data-testid=another-song]'));
      await until(() => q('[role=alert]'), 'the error');
      expect(text('[role=dialog][aria-label=Export]')).toContain('rate-limiting');
    });

    it('keeps the pick, its card, Another song and the mood when Reel is turned off and on', async () => {
      const e = new MockEngine();
      await autoReel(e);
      await setValue(q<HTMLInputElement>('input[aria-label="Auto music mood"]'), 'calm', 'input');
      await pickAuto();
      await click(checkbox('Reel'));
      await click(checkbox('Reel'));
      expect(q('[data-testid=auto-pick]')).not.toBeNull();
      expect(q('[data-testid=another-song]')).not.toBeNull();
      expect(q<HTMLInputElement>('input[aria-label="Auto music mood"]').value).toBe('calm');
      await click(inDlg('button', 'Export'));
      await until(() => e.lastExport);
      expect(e.lastExport?.music).toMatchObject({ auto: { mood: 'calm', rank: 0, track: 'auto1' } });
      expect(e.autoQueries.length).toBe(1);
    });

    it('asks again for the new clips when Which clips changes after a pick', async () => {
      const e = new MockEngine();
      e.caps.music = true;
      e.caps.beatSync = true;
      await boot(e);
      await act(() => store.select(['d1', 'd3'], false));
      await key('e', { metaKey: true });
      await until(() => q('[role=dialog][aria-label=Export]'));
      await click(checkbox('Reel'));
      const radios = [...dlg().querySelectorAll<HTMLInputElement>('input[name=which]')];
      await click(radios[0]); // All
      await pickAuto();
      expect(e.autoQueries).toEqual([{ clip_ids: null, mood: '', cc0_only: false, rank: 0 }]);
      await click(radios[1]); // Selected only
      await until(() => e.autoQueries.length === 2, 'the second ask');
      expect(e.autoQueries[1]).toEqual({ clip_ids: ['d1', 'd3'], mood: '', cc0_only: false, rank: 0 });
      await until(() => !q('[data-testid=auto-busy]'), 'the second answer');
      expect(q('[data-testid=auto-pick]')).not.toBeNull();
    });

    it('drops the pick when a re-ask for the CC0 only box fails, so the export never sends a song that was not shown', async () => {
      const e = new MockEngine();
      await autoReel(e);
      await pickAuto();
      expect(text('[data-testid=auto-pick]')).toContain('CC BY 4.0');
      e.autoError = 'music: Openverse is rate-limiting (test)';
      await click(checkbox('CC0 only'));
      await until(() => q('[role=alert]'), 'the error');
      expect(q('[data-testid=auto-pick]')).toBeNull();
      expect(text('[data-testid=music-choice]')).not.toContain('Sunny Walk');
      await click(inDlg('button', 'Export'));
      await until(() => e.lastExport);
      expect(e.lastExport?.music).toBeNull();
    });

    it('disables Export while the auto job runs, and shows a busy server as the job that runs, not as the word busy', async () => {
      const e = new MockEngine();
      await autoReel(e);
      let release: (r: MusicAutoResult) => void = () => {};
      e.musicAuto = (_runId, _qy, onJob) => {
        onJob?.({ id: 'mockjob-music-auto', kind: 'music-auto', state: 'running', fraction: 0.05, message: 'choosing a song', started: 0, elapsed_s: 0, media_id: null, run_id: null, result: null, error: null });
        return new Promise<MusicAutoResult>((res) => (release = res));
      };
      await click(root.querySelector('[data-testid=music-auto]') as HTMLInputElement);
      await until(() => q('[data-testid=auto-busy]'), 'the busy line');
      expect(inDlg('button', 'Export').disabled).toBe(true);
      await act(async () => release(await new MockEngine().musicAuto('r', { clip_ids: null, mood: '', cc0_only: false, rank: 0 })));
      await until(() => q('[data-testid=auto-pick]'), 'the pick');
      expect(inDlg('button', 'Export').disabled).toBe(false);

      const other: Job = { id: 'j9', kind: 'export', state: 'done', fraction: 1, message: '', started: 0, elapsed_s: 0, media_id: null, run_id: null, result: null, error: null };
      e.musicAuto = async () => { throw new EngineError(409, 'busy', { error: 'busy', job: other }); };
      await click(q('[data-testid=another-song]'));
      await until(() => store.toasts.value.length > 0, 'the toast');
      expect(store.toasts.value.map((t) => t.text)).toEqual(['A export job is running']);
      expect(dlg().querySelector('[role=alert]')?.textContent ?? '').not.toBe('busy');
    });
  });
});

describe('the export dialog: Save to folder or Save to Photos (T3b)', () => {
  const open = async () => {
    await key('e', { metaKey: true });
    await until(() => q('[role=dialog][aria-label=Export]'), 'the dialog');
  };
  const radio = (label: string) => withText<HTMLLabelElement>('label.check', label, false)?.querySelector('input') as HTMLInputElement;

  it('offers both choices on My Mac, folder first, and sends no to_photos for the folder', async () => {
    await boot();
    await open();
    expect(radio('Save to folder').checked).toBe(true);
    expect(radio('Save to Photos').checked).toBe(false);
    expect(q('[data-testid=photos-note]')).toBeNull();
    await click(inDlg('button', 'Export'));
    await until(() => q('[data-testid=export-outputs]'), 'the result');
    expect(engine.lastExport).not.toHaveProperty('to_photos');
    expect(q('[data-testid=photos-result]')).toBeNull();
  });

  it('Save to Photos sends to_photos, names the add-only promise, and shows the saved sentence', async () => {
    await boot();
    await open();
    await click(radio('Save to Photos'));
    expect(text('[data-testid=photos-note]')).toContain('cannot read your library');
    expect(text('[data-testid=photos-note]')).toContain('iCloud Photos is on');
    expect(text('[data-testid=photos-note]')).toContain('leaves your Mac');
    expect(inDlg('button', 'Export')).toBeUndefined();
    await click(inDlg('button', 'Export to Photos'));
    await until(() => q('[data-testid=photos-result]'), 'the Photos line');
    expect(engine.lastExport).toMatchObject({ to_photos: true, clips: true });
    expect(text('[data-testid=photos-result]')).toBe('Saved 2 video(s) to the Photos library.');
    expect(q('[data-testid=photos-result] p')!.className).toBe('okline');
  });

  it('a denied permission is an alert that says where to allow it, and the export is still listed', async () => {
    await boot();
    engine.photosResult = { status: 'denied', message: 'Photos permission was denied, so nothing was saved to Photos. Allow it in System Settings > Privacy & Security > Photos.', saved: [], failed: [{ name: 'mock_clip_01.mp4', error: 'no permission to add to Photos' }] };
    await open();
    await click(radio('Save to Photos'));
    await click(inDlg('button', 'Export to Photos'));
    await until(() => q('[data-testid=photos-result]'), 'the Photos line');
    const p = q('[data-testid=photos-result] p')!;
    expect(p.className).toBe('errline');
    expect(p.getAttribute('role')).toBe('alert');
    expect(p.textContent).toContain('Privacy & Security > Photos');
    expect(text('[data-testid=export-outputs]')).toContain('mock_clip_01.mp4');
  });

  it('a partial save lists the file that failed', async () => {
    await boot();
    engine.photosResult = { status: 'partial', message: 'Saved 1 of 2 videos to the Photos library; 1 failed.', saved: ['mock_clip_01.mp4'], failed: [{ name: 'mock_clip_reel.mp4', error: 'disk full' }] };
    await open();
    await click(radio('Save to Photos'));
    await click(checkbox('Reel'));
    await click(inDlg('button', 'Export to Photos'));
    await until(() => q('[data-testid=photos-result]'), 'the Photos line');
    expect(q('[data-testid=photos-result] p')!.className).toBe('errline');
    expect(text('[data-testid=photos-result]')).toContain('mock_clip_reel.mp4: disk full');
  });

  it('Photos needs clips or a reel: with only an edit list ticked the button is off', async () => {
    await boot();
    await open();
    await click(radio('Save to Photos'));
    await click(checkbox('Clips'));
    await click(checkbox('FCPXML'));
    expect(inDlg<HTMLButtonElement>('button', 'Export to Photos').disabled).toBe(true);
  });

  it('shows no Save to choice where the engine cannot save to Photos (the browser engine keeps its download)', async () => {
    const e = new MockEngine();
    e.caps = { ...e.caps, savePhotos: false };
    await boot(e);
    await open();
    expect(radio('Save to Photos')).toBeUndefined();
    expect(inDlg('button', 'Export')).toBeTruthy();
  });
});
