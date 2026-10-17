// A music credit is one long line (title, creator, license, link). It must wrap inside the music picker and the export dialog instead of
// running past the edge: `.credit pre` is pre-wrap (docs/T9B_DESIGN.md section 0; the old <pre> had white-space: pre and no rule).
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so the file read is allowed to be untyped
// @ts-ignore
import { readFileSync } from 'node:fs';
import { App } from '../src/app';
import { MockEngine, MOCK_PATH } from '../src/engine/mock';
import type { Job, MusicAutoQuery, MusicAutoResult } from '../src/engine/types';
import { createStore, type Store } from '../src/state';

const CSS: string = readFileSync((globalThis as any).process.cwd() + '/src/styles.css', 'utf8');
const LINE = 'Music: "Long Road Home" by A. B. Composer, CC BY 4.0, https://example.test/'.padEnd(80, 'x');

/** An engine whose CC BY credit is one 80-character line. */
class LongCreditEngine extends MockEngine {
  async musicAuto(runId: string, q: MusicAutoQuery, onJob?: (job: Job) => void): Promise<MusicAutoResult> {
    const r = await super.musicAuto(runId, q, onJob);
    if (r.track.needs_credit) r.track.credit_text = LINE;
    return r;
  }
  async job(id: string): Promise<Job> {
    const j = await super.job(id);
    const music = (j.result as { reel?: { music?: { credit_text?: string } } } | null)?.reel?.music;
    if (music?.credit_text) music.credit_text = LINE;
    return j;
  }
}

let root: HTMLElement;
let store: Store;
let style: HTMLStyleElement;
const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));
async function until(cond: () => unknown, what = 'condition') {
  for (let i = 0; i < 100; i++) {
    if (cond()) return;
    await tick(10);
  }
  throw new Error(`timed out waiting for ${what}`);
}
const q = <T extends HTMLElement = HTMLElement>(sel: string) => root.querySelector(sel) as T;
const click = (el: Element) => act(() => void (el as HTMLElement).click());
const computed = (el: Element) => getComputedStyle(el);

beforeAll(() => {
  HTMLMediaElement.prototype.pause = () => {};
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
});
beforeEach(() => {
  sessionStorage.clear();
  root = document.createElement('div');
  document.body.appendChild(root);
  style = document.createElement('style'); // jsdom does not load the stylesheet the build imports: put the real file in
  style.textContent = CSS;
  document.head.appendChild(style);
});
afterEach(() => {
  render(null, root);
  store?.dispose();
  root.remove();
  style.remove();
});

async function openExport() {
  const e = new LongCreditEngine();
  e.caps.music = true;
  store = createStore(e, { playable: async () => ({ ok: true, clockOffset: 0 }), autosaveMs: 5, toastMs: 0, retryMs: 5, pollMs: 5 });
  await store.init();
  await act(async () => void render(<App store={store} />, root));
  await act(async () => void (await store.openPath(MOCK_PATH)));
  await tick(5);
  await act(() => void document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'e', metaKey: true, bubbles: true, cancelable: true })));
  await until(() => q('[role=dialog][aria-label=Export]'), 'the export dialog');
  const reel = [...root.querySelectorAll('label.check')].find((l) => l.textContent?.includes('Reel'))!.querySelector('input')!;
  await click(reel);
}

describe('credit text wraps', () => {
  it('has an 80-character credit line to test with', () => {
    expect(LINE).toHaveLength(80);
  });

  it('styles.css has the rule', () => {
    const body = /(?:^|\n)\.credit pre\s*\{([^}]*)\}/.exec(CSS)?.[1] ?? '';
    expect(body).toMatch(/white-space:\s*pre-wrap/);
    expect(body).toMatch(/overflow-wrap:\s*anywhere/);
    expect(body).toMatch(/max-width:\s*100%/);
  });

  it('MusicPicker: the auto pick credit is pre-wrap', async () => {
    await openExport();
    await click(q('[data-testid=music-auto]'));
    await until(() => q('[data-testid=auto-credit]'), 'the auto pick');
    const pre = q('[data-testid=auto-credit]');
    expect(pre.tagName).toBe('PRE');
    expect(pre.textContent).toBe(LINE);
    expect(pre.closest('.credit')).not.toBeNull();
    expect(computed(pre).whiteSpace).toBe('pre-wrap');
    expect(computed(pre).overflowWrap).toBe('anywhere');
    expect(computed(pre).maxWidth).toBe('100%');
  });

  it('ExportDialog: the credit shown after the export is pre-wrap', async () => {
    await openExport();
    await click(q('[data-testid=music-auto]'));
    await until(() => q('[data-testid=auto-credit]'), 'the auto pick');
    await click([...q('[role=dialog][aria-label=Export]').querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Export')!);
    await until(() => q('[data-testid=music-credit-text]'), 'the credit after the export');
    const pre = q('[data-testid=music-credit-text]');
    expect(pre.tagName).toBe('PRE');
    expect(pre.textContent).toBe(LINE);
    expect(pre.closest('.credit')).not.toBeNull();
    expect(computed(pre).whiteSpace).toBe('pre-wrap');
  });

  it('an element outside .credit keeps the default (the rule is scoped)', () => {
    const pre = document.createElement('pre');
    root.appendChild(pre);
    expect(computed(pre).whiteSpace).not.toBe('pre-wrap');
  });
});
