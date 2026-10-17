// T9c: under the Download button the page says the video stays in the browser and names the host the model files come from.
import { render } from 'preact';
import { act } from 'preact/test-utils';
import { afterEach, describe, expect, it } from 'vitest';
import type { ModelControl } from '../src/engine/types';
import type { Store } from '../src/state';
import { Preflight } from '../src/ui/Preflight';

let root: HTMLElement | null = null;
afterEach(() => { if (root) render(null, root); root?.remove(); root = null; });

async function draw(status: Awaited<ReturnType<ModelControl['status']>>) {
  const models: ModelControl = { status: async () => status, download: async () => undefined, remove: async () => undefined };
  const store = { rows: { value: [] }, session: { value: { preflight: { browser: {} } } }, engine: { models, session: async () => ({}) } } as unknown as Store;
  root = document.createElement('div');
  document.body.appendChild(root);
  await act(async () => void render(<Preflight store={store} />, root!));
  await act(() => new Promise<void>((r) => setTimeout(r, 0)));
  return root;
}
const base = { available: true, error: null, detectorReady: false, allReady: false, persisted: null as boolean | null };

describe('the privacy line', () => {
  it('names the host and the size of what is still to download', async () => {
    const r = await draw({ ...base, totalBytes: 238_700_000, cachedBytes: 0, host: 'huggingface.co', filesLeft: 3 });
    expect(r.querySelector('[data-testid="models-privacy"]')?.textContent).toBe('Your video never leaves this browser. Only the three model files (about 239 MB) are downloaded, once, from huggingface.co, which hands the files on to its own download servers.');
  });
  it('counts what is left and says "this site" for the page\'s own host', async () => {
    const r = await draw({ ...base, totalBytes: 238_700_000, cachedBytes: 123_500_000, host: 'this site', filesLeft: 2 });
    expect(r.querySelector('[data-testid="models-privacy"]')?.textContent).toContain('Only the two model files (about 115 MB) are downloaded, once, from this site.');
    expect(r.querySelector('[data-testid="models-privacy"]')?.textContent).not.toContain('hands the files on');
  });
  it('is not drawn when everything is downloaded', async () => {
    const r = await draw({ ...base, allReady: true, totalBytes: 10, cachedBytes: 10, host: 'huggingface.co', filesLeft: 0 });
    expect(r.querySelector('[data-testid="models-privacy"]')).toBeNull();
  });
});
