// Fixes for the browser media review: filmstrip sheet size, cancelling a filmstrip, abandoned frame decodes, the .mov -> mp4 retry,
// and frames handed over at every flush. Each test fails on the code before the fix.
import { act } from 'preact/test-utils';
import { render } from 'preact';
import { describe, expect, it } from 'vitest';
import type { Demuxed } from '../src/browser/demux';
import { FrameDecoder, type DecoderDeps, type OpenVideo } from '../src/browser/frames';
import { FilmstripBuilder, planFilmstrip, type FilmOps } from '../src/browser/thumbs';
import { BlobUrlCache } from '../src/browser/urls';
import { BrowserEngine } from '../src/engine/browser';
import type { Engine } from '../src/engine/types';
import { useEngineUrl } from '../src/ui/useEngineUrl';

const tick = (ms = 0) => act(() => new Promise<void>((r) => setTimeout(r, ms)));

/** `n` pictures at 1 per second, every one a keyframe (a 1 s GOP at 1 fps, close enough for the plan) */
function demuxedOf(n: number, gop = 1): Demuxed {
  const pts = Float64Array.from({ length: n }, (_, i) => i);
  const sync = Uint8Array.from({ length: n }, (_, i) => (i % gop === 0 ? 1 : 0));
  const order = Uint32Array.from({ length: n }, (_, i) => i);
  return {
    codec: 'h264', profile: null, video: { trackId: 1, codecString: 'avc1.64001f', fourcc: 'avc1', description: null, width: 1920, height: 1080, timescale: 1000 },
    samples: { n, pts, dts: pts, offset: new Float64Array(n), size: new Uint32Array(n).fill(1), sync, order, tsUs: Float64Array.from(pts, (p) => p * 1e6) },
    keyframes: [], rotation: 0, mirrored: false, sar: 1, fps: 1, captureFps: 1, duration: n, offset: 0, endTime: n, formatStart: 0, hasAudio: false,
    audio: null, warnings: [], stats: { chunkBytes: 1, reads: 1, bytesRead: 1, peakHeldBytes: 1 },
  } as unknown as Demuxed;
}
const openOf = (n: number, gop = 1): OpenVideo => ({ demuxed: demuxedOf(n, gop), src: { size: 1, read: async () => new ArrayBuffer(1) } as never });

const fakeFrame = () => ({ closed: false, close() { this.closed = true; } });

describe('filmstrip sheets', () => {
  it('a sheet is always the whole rows x cell_h grid, even when fewer than 100 cells are used (ClipStrip sizes it as rows * cell_h)', async () => {
    const v = openOf(60);
    const sizes: [number, number][] = [];
    const drawn: [number, number][] = [];
    const ops: FilmOps = {
      makeCanvas: ((w: number, h: number) => { sizes.push([w, h]); return { width: w, height: h, getContext: () => ({ fillStyle: '', fillRect() {} }) }; }) as never,
      draw: ((_c: unknown, _f: unknown, _u: unknown, x: number, y: number) => { drawn.push([x, y]); }) as never,
      blob: async () => new Blob(['x']),
    };
    const dec = { decode: async (_v: OpenVideo, runs: [number, number][], _w: Set<number>, o: { onFrame?: (s: number, f: never) => void } = {}) => { for (const [, last] of runs) o.onFrame?.(last, fakeFrame() as never); return new Map(); } };
    const b = new FilmstripBuilder(v, dec, ops);
    expect(b.plan.picks).toHaveLength(60);
    b.start();
    const blob = await b.sheet(0);
    expect(blob).not.toBeNull();
    expect(sizes).toEqual([[b.plan.cols * b.plan.cell_w, b.plan.rows * b.plan.cell_h]]); // 1920 x 1080, not 1920 x 648
    expect(drawn).toHaveLength(60);
    expect(drawn[25]).toEqual([5 * b.plan.cell_w, 2 * b.plan.cell_h]);
    expect(planFilmstrip(v).rows).toBe(10);
  });

  it('cancel() stops the build after the batch in flight and answers a sheet that was waited for', async () => {
    const v = openOf(300);
    let calls = 0;
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const ops: FilmOps = { makeCanvas: ((w: number, h: number) => ({ width: w, height: h, getContext: () => ({ fillStyle: '', fillRect() {} }) })) as never, draw: (() => {}) as never, blob: async () => new Blob(['x']) };
    const dec = { decode: async () => { calls++; await gate; return new Map(); } };
    const b = new FilmstripBuilder(v, dec, ops);
    b.start();
    const waiting = b.sheet(2);
    await tick(5);
    expect(calls).toBe(1);
    b.cancel();
    expect(await waiting).toBeNull();
    release();
    await tick(10);
    expect(calls).toBe(1); // no later batch was queued behind it
    expect(b.state).toBe('failed');
  });
});

describe('BrowserEngine and the filmstrip it started', () => {
  const engineWith = (films: { cancelled: number }[], extra: Record<string, unknown> = {}) =>
    new BrowserEngine({
      demux: async () => demuxedOf(5),
      fileUrl: (() => { let n = 0; return () => `blob:${n++}`; })(),
      revokeFileUrl: () => {},
      makeFilmstrip: () => {
        const f = { cancelled: 0 };
        films.push(f);
        return { state: 'running', error: null, start() {}, sheet: async () => null, cancel() { f.cancelled++; }, filmstrip: () => ({ version: 1, cell_w: 192, cell_h: 108, cols: 10, rows: 10, cells: [], sheets: 1, done: false }) };
      },
      store: { putMedia: async () => {} } as never,
      ...extra,
    });

  it('picking the same video again as a new File cancels the first filmstrip build', async () => {
    const films: { cancelled: number }[] = [];
    const e = engineWith(films);
    const m = await e.openFile(new File([new Uint8Array(4)], 'a.mp4', { lastModified: 1 }));
    await e.filmstrip(m.id);
    await e.openFile(new File([new Uint8Array(4)], 'a.mp4', { lastModified: 1 })); // a new File object, same name, size and date
    expect(films[0].cancelled).toBe(1);
  });

  it('close() (the store was disposed on an engine switch) cancels the build and frees the urls', async () => {
    const films: { cancelled: number }[] = [];
    const revoked: string[] = [];
    const e = engineWith(films, { revokeFileUrl: (u: string) => revoked.push(u) });
    const m = await e.openFile(new File([new Uint8Array(4)], 'a.mp4'));
    await e.filmstrip(m.id);
    e.close();
    expect(films[0].cancelled).toBe(1);
    expect(revoked).toEqual(['blob:0']);
    expect(e.videoUrl(m.id)).toBe('');
  });

  it('the .mov retry gets a different URL whose bytes are typed video/mp4', async () => {
    const made: Blob[] = [];
    const e = new BrowserEngine({ demux: async () => demuxedOf(5), fileUrl: (b) => { made.push(b); return `blob:${made.length}`; }, revokeFileUrl: () => {}, store: { putMedia: async () => {} } as never });
    const m = await e.openFile(new File([new Uint8Array(8)], 'IMG.MOV', { type: 'video/quicktime' }));
    const first = e.videoUrl(m.id);
    const second = e.videoUrl(m.id, { asMp4: true });
    expect(second).not.toBe(first);
    expect(made[1].type).toBe('video/mp4');
    expect(made[1].size).toBe(8);
    expect(e.videoUrl(m.id, { asMp4: true })).toBe(second); // made once
  });
});

describe('FrameDecoder', () => {
  /** a VideoDecoder stand-in: every decode() makes a frame at flush(); `open` counts frames not yet closed */
  function deps() {
    const st = { constructed: 0, open: 0, maxOpen: 0, flushes: 0 };
    class VD {
      static isConfigSupported = async () => ({ supported: true });
      state = 'configured';
      private q: number[] = [];
      constructor(private o: { output: (f: unknown) => void }) { st.constructed++; }
      configure() {}
      decode(c: { timestamp: number }) { this.q.push(c.timestamp); }
      async flush() {
        st.flushes++;
        for (const ts of this.q) {
          st.open++;
          st.maxOpen = Math.max(st.maxOpen, st.open);
          this.o.output({ timestamp: ts, close: () => { st.open--; } });
        }
        this.q = [];
      }
      close() { this.state = 'closed'; }
    }
    class EC { timestamp: number; constructor(i: { timestamp: number }) { this.timestamp = i.timestamp; } }
    return { st, deps: { VideoDecoder: VD, EncodedVideoChunk: EC } as unknown as DecoderDeps };
  }

  it('hands each frame over right after its run is flushed, so 20 runs never hold 20 frames', async () => {
    const { st, deps: d } = deps();
    const dec = new FrameDecoder(d);
    const v = openOf(20);
    const got: number[] = [];
    const runs = Array.from({ length: 20 }, (_, i) => [i, i] as [number, number]);
    const out = await dec.decode(v, runs, new Set(runs.map((r) => r[1])), { onFrame: (i, f) => { got.push(i); f.close(); } });
    expect(got).toHaveLength(20);
    expect(out.size).toBe(0);
    expect(st.flushes).toBe(20);
    expect(st.maxOpen).toBe(1);
  });

  it('a queued job whose requester went away is dropped without touching the decoder', async () => {
    const { st, deps: d } = deps();
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    (d.VideoDecoder as unknown as { isConfigSupported: () => Promise<unknown> }).isConfigSupported = async () => { await gate; return { supported: true }; };
    const dec = new FrameDecoder(d);
    const v = openOf(10);
    const first = dec.decode(v, [[0, 0]], new Set([0]));
    const ctl = new AbortController();
    const second = dec.decode(v, [[3, 3]], new Set([3]), { signal: ctl.signal });
    const third = dec.decode(v, [[5, 5]], new Set([5]));
    ctl.abort();
    release();
    await expect(second).rejects.toMatchObject({ status: 499 });
    const f1 = await first;
    const f3 = await third;
    expect([...f1.keys()]).toEqual([0]);
    expect([...f3.keys()]).toEqual([5]);
    expect(st.constructed).toBe(2); // the first and the third; the second never got a decoder
  });
});

describe('BlobUrlCache holds', () => {
  it('does not revoke a URL whose requester is still showing it, and trims when it is let go', async () => {
    const revoked: string[] = [];
    let n = 0;
    const c = new BlobUrlCache(2, { create: () => `blob:${++n}`, revoke: (u) => revoked.push(u) });
    const ctls = [0, 1, 2, 3].map(() => new AbortController());
    const urls: string[] = [];
    for (let i = 0; i < 4; i++) urls.push(await c.get(`k${i}`, async () => new Blob(['x']), ctls[i].signal));
    expect(revoked).toEqual([]); // four on screen, max 2: all four stay
    expect(c.size).toBe(4);
    ctls[0].abort(); // the first scrolls away
    ctls[1].abort();
    expect(revoked).toEqual(['blob:1', 'blob:2']);
    expect(c.size).toBe(2);
    expect(await c.get('k3', async () => new Blob(['y']), new AbortController().signal)).toBe(urls[3]); // still cached
  });

  it('a request without a signal holds nothing: the oldest URL goes as before', async () => {
    const revoked: string[] = [];
    let n = 0;
    const c = new BlobUrlCache(2, { create: () => `blob:${++n}`, revoke: (u) => revoked.push(u) });
    for (let i = 0; i < 3; i++) await c.get(`k${i}`, async () => new Blob(['x']));
    expect(revoked).toEqual(['blob:1']);
  });
});

describe('BlobUrlCache and abandoned requests', () => {
  const cache = () => new BlobUrlCache(5, { create: () => 'blob:x', revoke: () => {} });

  it('aborts the render when its only requester leaves; a second requester keeps it alive', async () => {
    const c = cache();
    let sig!: AbortSignal;
    let done!: () => void;
    const make = (s: AbortSignal) => { sig = s; return new Promise<Blob>((r) => (done = () => r(new Blob(['a'])))); };
    const a = new AbortController();
    const b = new AbortController();
    const pa = c.get('k', make, a.signal);
    const pb = c.get('k', make, b.signal);
    a.abort();
    await expect(pa).rejects.toMatchObject({ name: 'AbortError' });
    expect(sig.aborted).toBe(false);
    done();
    expect(await pb).toBe('blob:x');

    const solo = new AbortController();
    const p = c.get('k2', make, solo.signal);
    solo.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
    expect(sig.aborted).toBe(true);
    done();
    await tick(1);
    expect(c.has('k2')).toBe(false); // an abandoned render is not cached, and the next asker starts afresh
  });

  it('useEngineUrl aborts the request when the component unmounts or asks for another key', async () => {
    const seen: { url: string; signal?: AbortSignal }[] = [];
    const engine = { kind: 'browser', resolveUrl: (url: string, signal?: AbortSignal) => { seen.push({ url, signal }); return new Promise<string>(() => {}); } } as unknown as Engine;
    const P = ({ url }: { url: string }) => <img src={useEngineUrl(engine, url)} />;
    const root = document.createElement('div');
    await act(() => { render(<P url="inframe-blob:frame/a/1.000/320" />, root); });
    await tick(1);
    await act(() => { render(<P url="inframe-blob:frame/a/2.000/320" />, root); });
    await tick(1);
    expect(seen).toHaveLength(2);
    expect(seen[0].signal?.aborted).toBe(true); // the midpoint moved on
    expect(seen[1].signal?.aborted).toBe(false);
    await act(() => { render(null, root); });
    expect(seen[1].signal?.aborted).toBe(true);
  });
});
