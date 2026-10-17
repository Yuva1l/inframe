// The manifest has its own address (VITE_MODEL_MANIFEST) apart from the files (VITE_MODEL_BASE), model requests carry no cookies and no
// referrer. The Pages build's host settings are in pages-host.test.ts (it needs the node environment, vite does not load in jsdom).
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CACHE_NAME, DETECTOR_FILE, EMBEDDER_FILE, ModelStore, WASM_FILE, defaultManifestUrl, sha256Hex, type Manifest } from '../src/browser/models';

const enc = (s: string) => new TextEncoder().encode(s);
const FILES: Record<string, Uint8Array> = { [DETECTOR_FILE]: enc('d'.repeat(40)), [EMBEDDER_FILE]: enc('e'.repeat(50)), [WASM_FILE]: enc('w'.repeat(30)) };
const BASE = 'https://models.test/x/resolve/' + 'a'.repeat(40) + '/';

async function manifest(): Promise<Manifest> {
  const files: Manifest['files'] = {};
  for (const [name, b] of Object.entries(FILES)) files[name] = { bytes: b.byteLength, sha256: await sha256Hex(b), license: 'MIT', source: 'test', role: name === DETECTOR_FILE ? 'detector' : name === EMBEDDER_FILE ? 'embedder' : 'runtime' };
  return { format: 1, files };
}
class FakeCache {
  m = new Map<string, Response>();
  async match(k: string) { return this.m.get(k)?.clone(); }
  async put(k: string, r: Response) { this.m.set(k, r); }
}
const fakeCaches = () => { const c = new FakeCache(); return { open: async (n: string) => { expect(n).toBe(CACHE_NAME); return c; }, delete: async () => true } as unknown as CacheStorage; };

afterEach(() => vi.unstubAllEnvs());

describe('model addresses (T9c)', () => {
  it('reads the manifest from VITE_MODEL_MANIFEST and the files from the base, with no cookies and no referrer', async () => {
    vi.stubEnv('VITE_MODEL_MANIFEST', './models/manifest.json');
    const man = await manifest();
    const seen: { url: string; init?: RequestInit }[] = [];
    const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
      seen.push({ url: String(url), init });
      const u = String(url);
      if (u === new URL('./models/manifest.json', document.baseURI).href) return new Response(JSON.stringify(man));
      const name = u.slice(BASE.length);
      return name in FILES && u.startsWith(BASE) ? new Response(FILES[name] as BodyInit) : new Response('no', { status: 404 });
    }) as typeof fetch;
    const s = new ModelStore(BASE, { fetch: fetchFn, caches: fakeCaches(), origin: 'https://page.test', persist: async () => true });
    expect(s.manifestUrl).toBe(new URL('./models/manifest.json', document.baseURI).href);
    await s.download();
    const urls = seen.map((x) => x.url);
    expect(urls[0]).toBe(new URL('./models/manifest.json', document.baseURI).href);
    expect(urls.slice(1).sort()).toEqual(Object.keys(FILES).map((n) => BASE + n).sort());
    expect(urls.some((u) => u === BASE + 'manifest.json')).toBe(false);
    for (const x of seen) {
      expect(x.init?.credentials).toBe('omit');
      expect(x.init?.referrerPolicy).toBe('no-referrer');
      expect(Object.keys(x.init ?? {})).not.toContain('headers');
    }
  });

  it('defaults the manifest to base + manifest.json and resolves VITE_MODEL_MANIFEST against the page', () => {
    expect(defaultManifestUrl(BASE)).toBe(BASE + 'manifest.json');
    vi.stubEnv('VITE_MODEL_MANIFEST', './models/manifest.json');
    expect(defaultManifestUrl(BASE)).toBe(new URL('./models/manifest.json', document.baseURI).href);
    expect(new ModelStore(BASE).manifestUrl).toBe(defaultManifestUrl(BASE));
  });

  it('a file the host replaced fails its digest from the page manifest and is not kept', async () => {
    const man = await manifest();
    const bad: Record<string, Uint8Array> = { ...FILES, [DETECTOR_FILE]: enc('x'.repeat(40)) };
    const fetchFn = (async (url: string | URL | Request) => {
      const u = String(url);
      if (u.endsWith('/manifest.json')) return new Response(JSON.stringify(man));
      return new Response(bad[u.slice(BASE.length)] as BodyInit);
    }) as typeof fetch;
    const caches = fakeCaches();
    const s = new ModelStore(BASE, { fetch: fetchFn, caches, origin: 'https://page.test', persist: async () => true }, 'https://page.test/models/manifest.json');
    await expect(s.download()).rejects.toThrow(/checksum/);
    expect((await s.status()).files.find((f) => f.name === DETECTOR_FILE)?.cached).toBe(false);
  });

  it('names the host the files come from, or "this site"', async () => {
    const man = await manifest();
    const fetchFn = (async () => new Response(JSON.stringify(man))) as unknown as typeof fetch;
    const far = new ModelStore('https://huggingface.co/o/m/resolve/' + 'b'.repeat(40) + '/', { fetch: fetchFn, caches: fakeCaches(), origin: 'https://page.test' });
    expect((await far.status()).host).toBe('huggingface.co');
    expect((await far.status()).filesLeft).toBe(3);
    expect(new ModelStore('https://page.test/models/', { origin: 'https://page.test' }).host).toBe('this site');
  });
});
