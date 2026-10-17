// Model fetch and cache (docs/T9B_DESIGN.md section 3) with a fake network and a Map for Cache Storage.
import { describe, expect, it } from 'vitest';
import { CACHE_NAME, DETECTOR_FILE, EMBEDDER_FILE, ModelStore, WASM_FILE, resolveBase, sha256Hex, validManifest, type Manifest } from '../src/browser/models';

const enc = (s: string) => new TextEncoder().encode(s);
const FILES: Record<string, Uint8Array> = { [DETECTOR_FILE]: enc('detector bytes '.repeat(50)), [EMBEDDER_FILE]: enc('embedder '.repeat(80)), [WASM_FILE]: enc('\0asm-runtime'.repeat(30)) };

async function manifestOf(over: Record<string, Partial<Manifest['files'][string]>> = {}): Promise<Manifest> {
  const files: Manifest['files'] = {};
  for (const [name, b] of Object.entries(FILES)) {
    files[name] = { bytes: b.byteLength, sha256: await sha256Hex(b), license: name.endsWith('.wasm') ? 'MIT' : 'Apache-2.0', source: 'test', role: name === DETECTOR_FILE ? 'detector' : name === EMBEDDER_FILE ? 'embedder' : 'runtime', ...over[name] };
  }
  return { format: 1, files };
}

class FakeCache {
  m = new Map<string, Response>();
  async match(k: string) { return this.m.get(k)?.clone(); }
  async put(k: string, r: Response) { this.m.set(k, r); }
}
class FakeCaches {
  stores = new Map<string, FakeCache>();
  async open(n: string) { let c = this.stores.get(n); if (!c) this.stores.set(n, (c = new FakeCache())); return c; }
  async delete(n: string) { return this.stores.delete(n); }
}

function network(man: Manifest, body: Record<string, Uint8Array> = FILES) {
  const seen: string[] = [];
  const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
    const u = String(url);
    seen.push(`${init?.method ?? 'GET'} ${u}`);
    const name = u.split('/').pop()!;
    if (name === 'manifest.json') return new Response(JSON.stringify(man));
    if (!(name in body)) return new Response('no', { status: 404 });
    if (init?.signal?.aborted) throw new DOMException('aborted', 'AbortError');
    // two chunks, so progress is reported twice
    const b = body[name];
    return new Response(new ReadableStream({ start(c) { c.enqueue(b.slice(0, 7)); c.enqueue(b.slice(7)); c.close(); } }));
  }) as typeof fetch;
  return { fetchFn, seen };
}

const store = (man: Manifest, caches = new FakeCaches(), body = FILES, persist = async () => true) => {
  const net = network(man, body);
  return { s: new ModelStore('https://page.test/models/', { fetch: net.fetchFn, caches: caches as unknown as CacheStorage, origin: 'https://page.test', persist }), net, caches };
};

describe('models', () => {
  it('resolves the base against the page and always ends it with a slash', () => {
    expect(resolveBase('./models/', 'https://a.test/app/index.html')).toBe('https://a.test/app/models/');
    expect(resolveBase('https://hf.test/x/resolve/abc', 'https://a.test/')).toBe('https://hf.test/x/resolve/abc/');
  });

  it('knows a good manifest from a bad one', async () => {
    expect(validManifest(await manifestOf())).toBe(true);
    expect(validManifest({ format: 2, files: {} })).toBe(false);
    expect(validManifest({ format: 1, files: { '../x.onnx': { bytes: 1, sha256: 'a'.repeat(64) } } })).toBe(false);
    expect(validManifest({ format: 1, files: { 'a.onnx': { bytes: 1, sha256: 'xyz' } } })).toBe(false);
  });

  it('downloads every file once, with progress, and keeps them under the digest in Cache Storage inframe-models-v1', async () => {
    const man = await manifestOf();
    const { s, net, caches } = store(man);
    expect((await s.status()).allReady).toBe(false);
    const prog: number[] = [];
    await s.download({ onProgress: (p) => prog.push(p.done) });
    const total = Object.values(FILES).reduce((a, b) => a + b.byteLength, 0);
    expect(prog.at(-1)).toBe(total);
    expect(prog.every((x, i) => i === 0 || x >= prog[i - 1])).toBe(true);
    expect(prog.length).toBeGreaterThan(3);
    const cache = caches.stores.get(CACHE_NAME)!;
    expect([...cache.m.keys()].sort()).toEqual(Object.values(man.files).map((f) => `https://page.test/__inframe_models__/${f.sha256}`).sort());
    const st = await s.status();
    expect(st.allReady && st.detectorReady).toBe(true);
    expect(st.cachedBytes).toBe(total);
    expect(Array.from(await s.bytes(DETECTOR_FILE))).toEqual(Array.from(FILES[DETECTOR_FILE]));
    const fetched = net.seen.length;
    await s.download(); // nothing left to fetch
    expect(net.seen.length).toBe(fetched);
    expect(net.seen.every((x) => x.startsWith('GET https://page.test/models/'))).toBe(true);
  });

  it('refuses a file whose digest is wrong and keeps nothing of it', async () => {
    const man = await manifestOf();
    const bad = { ...FILES, [EMBEDDER_FILE]: enc('X'.repeat(FILES[EMBEDDER_FILE].byteLength)) };
    const { s, caches } = store(man, new FakeCaches(), bad);
    await expect(s.download()).rejects.toThrow(/checksum/);
    const st = await s.status();
    expect(st.files.find((f) => f.name === EMBEDDER_FILE)!.cached).toBe(false);
    expect(st.allReady).toBe(false);
    expect([...caches.stores.get(CACHE_NAME)!.m.keys()].length).toBeLessThan(3);
  });

  it('refuses a file of the wrong length (too long, too short) before hashing', async () => {
    const man = await manifestOf();
    const longer = store(man, new FakeCaches(), { ...FILES, [DETECTOR_FILE]: enc('d'.repeat(FILES[DETECTOR_FILE].byteLength + 5)) });
    await expect(longer.s.download()).rejects.toThrow(/larger than/);
    const shorter = store(man, new FakeCaches(), { ...FILES, [DETECTOR_FILE]: FILES[DETECTOR_FILE].slice(0, 20) });
    await expect(shorter.s.download()).rejects.toThrow(/bytes/);
  });

  it('asks for persistent storage, and says when the browser refused', async () => {
    const man = await manifestOf();
    const a = store(man, new FakeCaches(), FILES, async () => false);
    await a.s.download();
    expect((await a.s.status()).persisted).toBe(false);
  });

  it('bytes() says to download first, and remove() forgets everything', async () => {
    const man = await manifestOf();
    const { s, caches } = store(man);
    await expect(s.bytes(DETECTOR_FILE)).rejects.toMatchObject({ status: 409 });
    await expect(s.bytes('nope.onnx')).rejects.toMatchObject({ status: 404 });
    await s.download();
    await s.remove();
    expect(caches.stores.has(CACHE_NAME)).toBe(false);
    expect((await s.status()).cachedBytes).toBe(0);
  });

  it('a changed file (new digest) is fetched again; the old one is not in the way', async () => {
    const man = await manifestOf();
    const { s, caches } = store(man);
    await s.download({ only: [DETECTOR_FILE] });
    const v2 = { ...FILES, [DETECTOR_FILE]: enc('detector v2 '.repeat(30)) };
    const man2 = await manifestOf({ [DETECTOR_FILE]: { bytes: v2[DETECTOR_FILE].byteLength, sha256: await sha256Hex(v2[DETECTOR_FILE]) } });
    const s2 = new ModelStore('https://page.test/models/', { fetch: network(man2, v2).fetchFn, caches: caches as unknown as CacheStorage, origin: 'https://page.test', persist: async () => true });
    expect((await s2.status()).files.find((f) => f.name === DETECTOR_FILE)!.cached).toBe(false);
    await s2.download({ only: [DETECTOR_FILE] });
    expect(Array.from(await s2.bytes(DETECTOR_FILE))).toEqual(Array.from(v2[DETECTOR_FILE]));
  });

  it('reports a missing or broken model list instead of throwing from status()', async () => {
    const gone = new ModelStore('https://page.test/models/', { fetch: (async () => new Response('', { status: 404 })) as typeof fetch, caches: new FakeCaches() as unknown as CacheStorage, origin: 'https://page.test' });
    const st = await gone.status();
    expect(st.available).toBe(false);
    expect(st.error).toMatch(/No models are published/);
    const junk = new ModelStore('https://page.test/models/', { fetch: (async () => new Response('{"format":9}')) as typeof fetch, caches: new FakeCaches() as unknown as CacheStorage, origin: 'https://page.test' });
    expect((await junk.status()).error).toMatch(/format/);
  });

  it('cancelling stops the download with 499', async () => {
    const man = await manifestOf();
    const { s } = store(man);
    const ac = new AbortController();
    ac.abort();
    await expect(s.download({ signal: ac.signal })).rejects.toMatchObject({ status: 499 });
  });
});
