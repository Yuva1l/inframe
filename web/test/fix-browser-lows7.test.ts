// Browser edge cases: a merged run no longer holds all its wanted frames open until the flush, and a picture the decoder
// never gave back is reported and counted. Each test fails on the code before the fix. (The export findings are in browser-export.test.ts.)
import { describe, expect, it } from 'vitest';
import type { Demuxed } from '../src/browser/demux';
import { FrameDecoder, type DecoderDeps, type OpenVideo } from '../src/browser/frames';
import { buildRunDoc, resolveSettings } from '../src/browser/pipeline';

function demuxedOf(n: number): Demuxed {
  const pts = Float64Array.from({ length: n }, (_, i) => i / 30);
  return {
    codec: 'h264', profile: null, video: { trackId: 1, codecString: 'avc1.64001f', fourcc: 'avc1', description: null, width: 640, height: 360, timescale: 1000 },
    samples: { n, pts, dts: pts, offset: new Float64Array(n), size: new Uint32Array(n).fill(1), sync: Uint8Array.from({ length: n }, (_, i) => (i === 0 ? 1 : 0)), order: Uint32Array.from({ length: n }, (_, i) => i), tsUs: Float64Array.from(pts, (p) => Math.round(p * 1e6)) },
  } as unknown as Demuxed;
}

describe('a long merged run keeps few frames open', () => {
  it('64 wanted pictures in one run are handed over as the decoder gives them, not all at the flush', async () => {
    let open = 0;
    let peak = 0;
    class VD {
      static isConfigSupported = async () => ({ supported: true });
      state = 'configured';
      constructor(private o: { output: (f: unknown) => void }) {}
      configure() {}
      decode(c: { timestamp: number }) {
        open++;
        peak = Math.max(peak, open);
        this.o.output({ timestamp: c.timestamp, close: () => void open-- });
      }
      async flush() {}
      close() { this.state = 'closed'; }
    }
    class EC { timestamp: number; constructor(i: { timestamp: number }) { this.timestamp = i.timestamp; } }
    const n = 80;
    const v = { demuxed: demuxedOf(n), src: { size: 1, read: async () => new ArrayBuffer(n) } } as unknown as OpenVideo;
    const want = new Set(Array.from({ length: 64 }, (_, i) => i + 5));
    const got: number[] = [];
    await new FrameDecoder({ VideoDecoder: VD, EncodedVideoChunk: EC } as unknown as DecoderDeps).decode(v, [[0, n - 1]], want, {
      onFrame: async (i, f) => {
        got.push(i);
        await Promise.resolve();
        f.close();
      },
    });
    expect(got).toEqual([...want]);
    expect(open).toBe(0);
    expect(peak).toBeLessThanOrEqual(3); // before: 64 + the unwanted ones until each closed
  });
});

describe('pictures that did not come back', () => {
  const st = resolveSettings({ preset: 'balanced', advanced: {} });
  const info = { target: 'any-person', cls: 1, zone: null, zoneOverlap: null, detector: 'x', model: 'small', warnings: [] as string[], totalS: 0 };
  const src = { offset: 0, end_time: 5, keyframes: [0] };
  it('warns and counts when fewer pictures came back than were planned', () => {
    const doc = buildRunDoc({ times: [0, 1, 2], scores: [0, 0, 0], boxes: [null, null, null], detectS: 0, wallS: 0, planned: 5 }, st, src, info);
    expect(doc.segments.warnings.some((w) => /2 of the 5 planned pictures/.test(w))).toBe(true);
    expect(doc.segments.scores).toMatchObject({ n_samples: 3, n_planned: 5, n_dropped: 2 });
  });
  it('is silent when every planned picture came back', () => {
    const doc = buildRunDoc({ times: [0, 1, 2], scores: [0, 0, 0], boxes: [null, null, null], detectS: 0, wallS: 0, planned: 3 }, st, src, info);
    expect(doc.segments.warnings.some((w) => /planned pictures/.test(w))).toBe(false);
    expect(doc.segments.scores).toMatchObject({ n_dropped: 0 });
  });
  it('counts them when none came back', () => {
    const doc = buildRunDoc({ times: [], scores: [], boxes: [], detectS: 0, wallS: 0, planned: 4 }, st, src, info);
    expect(doc.segments.scores).toEqual({ n_samples: 0, n_planned: 4, n_dropped: 4 });
    expect(doc.segments.warnings.some((w) => /4 of the 4 planned/.test(w))).toBe(true);
  });
});

// WebCodecs delivers output() as a task, never inside decode(), and counts a chunk out of decodeQueueSize (with a 'dequeue' event) as the decoder takes it.
// The test above fires output synchronously, which a real decoder never does; this one does what Chrome does, so a drain that only yields microtasks fails it.
describe('a long merged run keeps few frames open with a decoder that answers as a task', () => {
  it('hands pictures over while the run is still being fed, with few open at once', async () => {
    let open = 0;
    let peak = 0;
    let flushed = false;
    let firstHandOverAfterFlush: boolean | null = null;
    class VD {
      static isConfigSupported = async () => ({ supported: true });
      state = 'configured';
      decodeQueueSize = 0;
      private listeners: (() => void)[] = [];
      constructor(private o: { output: (f: unknown) => void }) {}
      configure() {}
      addEventListener(_t: string, fn: () => void) { this.listeners.push(fn); }
      removeEventListener(_t: string, fn: () => void) { this.listeners = this.listeners.filter((l) => l !== fn); }
      decode(c: { timestamp: number }) {
        this.decodeQueueSize++;
        setTimeout(() => {
          this.decodeQueueSize--;
          open++;
          peak = Math.max(peak, open);
          this.o.output({ timestamp: c.timestamp, close: () => void open-- });
          [...this.listeners].forEach((l) => l());
        }, 0);
      }
      flush() { return new Promise<void>((r) => setTimeout(() => ((flushed = true), r()), 5)); }
      close() { this.state = 'closed'; }
    }
    class EC { timestamp: number; constructor(i: { timestamp: number }) { this.timestamp = i.timestamp; } }
    const n = 80;
    const v = { demuxed: demuxedOf(n), src: { size: 1, read: async () => new ArrayBuffer(n) } } as unknown as OpenVideo;
    const want = new Set(Array.from({ length: 64 }, (_, i) => i + 5));
    const got: number[] = [];
    await new FrameDecoder({ VideoDecoder: VD, EncodedVideoChunk: EC } as unknown as DecoderDeps).decode(v, [[0, n - 1]], want, {
      onFrame: async (i, f) => {
        firstHandOverAfterFlush ??= flushed;
        got.push(i);
        await Promise.resolve();
        f.close();
      },
    });
    expect(got).toEqual([...want]);
    expect(open).toBe(0);
    expect(firstHandOverAfterFlush).toBe(false); // before: nothing reached onFrame until dec.flush()
    expect(peak).toBeLessThanOrEqual(6); // before: 64 + the unwanted ones until each closed
  });
});
