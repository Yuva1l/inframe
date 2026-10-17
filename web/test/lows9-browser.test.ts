// Browser edge cases: the replaced zip survives a failed export, a tap picture does not wait for a run, the clearing note shows after a reload.
import { describe, expect, it, vi } from 'vitest';
import { chooseSink } from '../src/browser/exportzip';
import { ModelStore, sha256Hex, type Manifest } from '../src/browser/models';
import { createHandler } from '../src/browser/worker';

let gate: Promise<void> = Promise.resolve();
let runTag = 'slow';
let active = 0;
let maxActive = 0;
vi.mock('../src/browser/frames', async (orig) => {
  class FakeDecoder {
    private chain: Promise<unknown> = Promise.resolve();
    decode(v: { tag: string }, _runs: unknown, want: Set<number>): Promise<Map<number, unknown>> {
      const job = this.chain.then(async () => {
        if (v.tag === 'slow') await gate;
        return new Map([[[...want][0], { close() {} }]]);
      });
      this.chain = job.catch(() => undefined);
      return job;
    }
  }
  return { ...(await orig<typeof import('../src/browser/frames')>()), FrameDecoder: FakeDecoder, decodePlan: () => ({ first: 0, last: 0, target: 0 }) };
});
vi.mock('../src/browser/demux', () => ({ fileSource: (f: { tag: string }) => ({ tag: f.tag }) }));
vi.mock('../src/browser/detect', () => ({
  MIN_PERSON_SCORE: 0.3,
  createSession: async () => ({ provider: 'wasm' }),
  RfDetr: class { session = { provider: 'wasm' }; async detect() { maxActive = Math.max(maxActive, ++active); await new Promise((r) => setTimeout(r, 15)); active--; return []; } },
}));
vi.mock('../src/browser/models', async (orig) => ({
  ...(await orig<typeof import('../src/browser/models')>()),
  ModelStore: class extends (await orig<typeof import('../src/browser/models')>()).ModelStore { adopt() {} async bytes() { return new Uint8Array(1); } },
}));
vi.mock('../src/browser/pipeline', () => ({
  FrameRenderer: class { prepare = async () => ({}); pixels() { return null; } },
  scoreVideo: async (a: { detect: (t: unknown) => Promise<unknown>; decoder: { decode: (v: unknown, r: unknown, w: Set<number>) => Promise<unknown> } }) => {
    await a.decoder.decode({ tag: runTag }, [], new Set([0]));
    await a.detect({});
    return { ok: true };
  },
}));

describe('lows batch 9', () => {
  it('a run and a tap picture never use the detector at the same moment', async () => {
    runTag = 'fast';
    maxActive = 0;
    const done: number[] = [];
    const handle = createHandler((r) => { if (r.type === 'done' || r.type === 'error') done.push(r.id); });
    const common = { base: 'http://x/', manifest: { format: 1, files: {} } as unknown as Manifest, demuxed: { samples: { pts: [0] }, endTime: 1 } as never };
    handle({ op: 'run', id: 1, ...common, file: { tag: 'fast' } as unknown as File, target: { kind: 'any' }, fps: null, zone: null, zoneOverlap: null, minBox: 0 });
    handle({ op: 'boxes', id: 2, ...common, file: { tag: 'fast' } as unknown as File, t: 0, cls: 'person' } as never);
    await new Promise((r) => setTimeout(r, 100));
    expect(done.sort()).toEqual([1, 2]);
    expect(maxActive).toBe(1);
    runTag = 'slow';
  });

  it('a tap picture is decoded while a run holds its decoder', async () => {
    let release: () => void = () => undefined;
    gate = new Promise<void>((r) => (release = r));
    const done: number[] = [];
    const handle = createHandler((r) => { if (r.type === 'done' || r.type === 'error') done.push(r.id); });
    const common = { base: 'http://x/', manifest: { format: 1, files: {} } as unknown as Manifest, demuxed: { samples: { pts: [0] }, endTime: 1 } as never };
    handle({ op: 'run', id: 1, ...common, file: { tag: 'slow' } as unknown as File, target: { kind: 'any' }, fps: null, zone: null, zoneOverlap: null, minBox: 0 });
    handle({ op: 'boxes', id: 2, ...common, file: { tag: 'fast' } as unknown as File, t: 0, cls: 'person' } as never);
    await new Promise((r) => setTimeout(r, 50));
    expect(done).toEqual([2]); // the boxes answered, the run still waits
    release();
    await new Promise((r) => setTimeout(r, 50));
    expect(done).toEqual([2, 1]);
  });

  it('discard removes a file the picker made empty, not a zip the user chose to replace', async () => {
    const handleOf = (size: number, log: { removed: number }) => ({ remove: async () => void log.removed++, getFile: async () => ({ size }) });
    const fresh = { removed: 0 };
    await (await chooseSink({ showSaveFilePicker: async () => handleOf(0, fresh) }, 'a.zip', 0)).discard!();
    expect(fresh.removed).toBe(1);
    const old = { removed: 0 };
    await (await chooseSink({ showSaveFilePicker: async () => handleOf(5000, old) }, 'a.zip', 0)).discard!();
    expect(old.removed).toBe(0);
  });

  it('status after a reload asks the browser whether storage is persisted', async () => {
    const bytes = new TextEncoder().encode('detector');
    const man: Manifest = { format: 1, files: { 'rfdetr_small_v1.onnx': { bytes: bytes.byteLength, sha256: await sha256Hex(bytes), license: 'MIT', source: 't', role: 'detector' } } };
    const stored = new Map<string, Response>();
    const caches = { open: async () => ({ match: async (k: string) => stored.get(k)?.clone(), put: async (k: string, r: Response) => void stored.set(k, r) }), delete: async () => true } as unknown as CacheStorage;
    const fetchFn = (async (u: string) => (String(u).endsWith('manifest.json') ? new Response(JSON.stringify(man)) : new Response(bytes))) as typeof fetch;
    const mk = (persisted: () => Promise<boolean>) => new ModelStore('https://page.test/models/', { fetch: fetchFn, caches, origin: 'https://page.test', persist: async () => false, persisted });
    await mk(async () => false).download();
    const after = mk(async () => false); // a new page load: nothing in memory
    expect((await after.status()).cachedBytes).toBeGreaterThan(0);
    expect((await after.status()).persisted).toBe(false);
    expect((await mk(async () => true).status()).persisted).toBe(true);
  });
});
