// @vitest-environment node
// Edge cases: a clip past the end of the video is refused, the first AAC track (not just the first audio track) is
// kept, and one undecodable keyframe leaves only its cell dark. Each test fails on the code before the fix.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
// @ts-ignore
import { execFileSync } from 'node:child_process';
// @ts-ignore
import fs from 'node:fs';
// @ts-ignore
import os from 'node:os';
// @ts-ignore
import path from 'node:path';
import { demux, type ByteSource, type Demuxed } from '../src/browser/demux';
import type { OpenVideo } from '../src/browser/frames';
import { clipBytes, muxClip, planClip, readMuxSource } from '../src/browser/mux';
import { FilmstripBuilder, type FilmOps } from '../src/browser/thumbs';
import { EngineError } from '../src/engine/types';

const sh = (cmd: string, args: string[]): string => execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) as string;
const have = (() => {
  try {
    sh('ffmpeg', ['-version']);
    sh('ffprobe', ['-version']);
    return true;
  } catch {
    return false;
  }
})();
const ff = (...a: string[]) => sh('ffmpeg', ['-nostats', '-loglevel', 'error', '-y', ...a]);
const nodeSource = (file: string): ByteSource => {
  const fd = fs.openSync(file, 'r');
  return {
    size: fs.fstatSync(fd).size,
    read: async (start, end) => {
      const u8 = new Uint8Array(end - start);
      fs.readSync(fd, u8, 0, u8.length, start);
      return u8.buffer;
    },
  };
};

let dir = '';
let twoAudio = '';
beforeAll(() => {
  if (!have) return;
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'inframe-lows-'));
  twoAudio = path.join(dir, 'two_audio.mp4');
  // track 1 video, track 2 MP3 (not AAC), track 3 AAC
  ff('-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=30:duration=6', '-f', 'lavfi', '-i', 'sine=frequency=330:duration=6', '-f', 'lavfi', '-i', 'sine=frequency=550:duration=6',
    '-map', '0:v', '-map', '1:a', '-map', '2:a', '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '30', '-c:a:0', 'libmp3lame', '-c:a:1', 'aac', twoAudio);
});
afterAll(() => {
  if (dir) fs.rmSync(dir, { recursive: true, force: true });
});

const open = async (file: string) => {
  const src = nodeSource(file);
  const d = await demux(src);
  return { m: await readMuxSource(src, d, path.basename(file)), src, d };
};

describe.skipIf(!have)('lossless clips', () => {
  it('a clip whose range lies past the end of the video is refused, not exported as the last GOP', async () => {
    const { m } = await open(twoAudio);
    expect(() => planClip(m, 100, 105)).toThrow(EngineError);
    expect(() => planClip(m, 100, 105)).toThrow(/after the end of the video/);
    expect(() => planClip(m, 5.9, 8)).not.toThrow(); // starts inside the last picture's span: still a clip
  });

  it('the AAC track is kept when the first audio track is another codec', async () => {
    const { m, src } = await open(twoAudio);
    expect(m.audio).not.toBeNull();
    expect(m.audio!.fourcc).toBe('mp4a');
    expect(m.warnings).toEqual([]);
    const bytes = await clipBytes(src, muxClip(m, planClip(m, 1, 4)));
    const out = path.join(dir, 'clip.mp4');
    fs.writeFileSync(out, bytes);
    const j = JSON.parse(sh('ffprobe', ['-v', 'error', '-show_streams', '-of', 'json', out]));
    expect(j.streams.map((s: any) => s.codec_name).sort()).toEqual(['aac', 'h264']);
  });
});

describe('filmstrip with an undecodable keyframe', () => {
  const n = 30;
  const pts = Float64Array.from({ length: n }, (_, i) => i);
  const demuxed = {
    video: { width: 1920, height: 1080 }, rotation: 0, mirrored: false, sar: 1, duration: n, offset: 0,
    samples: { n, pts, dts: pts, sync: new Uint8Array(n).fill(1), order: Uint32Array.from({ length: n }, (_, i) => i), tsUs: Float64Array.from(pts, (p) => p * 1e6) },
  } as unknown as Demuxed;
  const v: OpenVideo = { demuxed, src: { size: 1, read: async () => new ArrayBuffer(1) } as never };
  const BAD = 7;

  it('leaves that cell dark and draws every other one', async () => {
    const drawn: number[] = [];
    const closed: number[] = [];
    const ops: FilmOps = {
      makeCanvas: ((w: number, h: number) => ({ width: w, height: h, getContext: () => ({ fillStyle: '', fillRect() {} }) })) as never,
      draw: ((_c: unknown, f: { sample: number }) => { drawn.push(f.sample); }) as never,
      blob: async () => new Blob(['x']),
    };
    // a decoder like the real one: a bad sample makes the whole decode() reject (after the runs before it were handed over)
    const dec = {
      decode: async (_v: OpenVideo, runs: [number, number][], _w: Set<number>, o: { onFrame?: (s: number, f: never) => void } = {}) => {
        for (const [, last] of runs) {
          if (last === BAD) throw new EngineError(500, 'Decoding failed: EncodingError');
          o.onFrame?.(last, { sample: last, close: () => closed.push(last) } as never);
        }
        return new Map();
      },
    };
    const b = new FilmstripBuilder(v, dec, ops);
    b.start();
    expect(await b.sheet(0)).not.toBeNull();
    expect(b.state).toBe('done');
    expect(b.error).toBeNull();
    expect([...new Set(drawn)].sort((x, y) => x - y)).toEqual(Array.from({ length: n }, (_, i) => i).filter((i) => i !== BAD));
    expect(closed.length).toBe(drawn.length);
  });

  it('a damaged start (the first three keyframes) leaves those cells dark; the filmstrip is still made from the rest', async () => {
    const BADS = new Set([0, 1, 2]);
    const drawn: number[] = [];
    const ops: FilmOps = {
      makeCanvas: ((w: number, h: number) => ({ width: w, height: h, getContext: () => ({ fillStyle: '', fillRect() {} }) })) as never,
      draw: ((_c: unknown, f: { sample: number }) => { drawn.push(f.sample); }) as never,
      blob: async () => new Blob(['x']),
    };
    const dec = {
      decode: async (_v: OpenVideo, runs: [number, number][], _w: Set<number>, o: { onFrame?: (s: number, f: never) => void } = {}) => {
        for (const [, last] of runs) {
          if (BADS.has(last)) throw new EngineError(500, 'Decoding failed: EncodingError');
          o.onFrame?.(last, { sample: last, close: () => {} } as never);
        }
        return new Map();
      },
    };
    const b = new FilmstripBuilder(v, dec, ops);
    b.start();
    expect(await b.sheet(0)).not.toBeNull();
    expect(b.state).toBe('done');
    expect(b.error).toBeNull();
    expect([...new Set(drawn)].sort((x, y) => x - y)).toEqual(Array.from({ length: n }, (_, i) => i).filter((i) => !BADS.has(i)));
  });

  it('a video whose keyframes all fail to decode fails the filmstrip with the error, after the batch and one single try per cell of that batch, not one decode per cell', async () => {
    const ops: FilmOps = { makeCanvas: ((w: number, h: number) => ({ width: w, height: h, getContext: () => ({ fillStyle: '', fillRect() {} }) })) as never, draw: (() => {}) as never, blob: async () => new Blob(['x']) };
    let calls = 0;
    const b = new FilmstripBuilder(v, { decode: async () => { calls++; throw new EngineError(500, 'Decoding failed: EncodingError'); } }, ops);
    b.start();
    expect(await b.sheet(0)).toBeNull();
    expect(b.state).toBe('failed');
    expect(b.error).toMatch(/Decoding failed/);
    expect(calls).toBeLessThanOrEqual(1 + n); // the batch, then each cell of it once with nothing ever drawn: stop (30 cells fit one batch of 20 and a second of 10; only the first runs)
  });

  it('an unsupported codec (415) still fails the whole filmstrip', async () => {
    const ops: FilmOps = { makeCanvas: ((w: number, h: number) => ({ width: w, height: h, getContext: () => ({ fillStyle: '', fillRect() {} }) })) as never, draw: (() => {}) as never, blob: async () => new Blob(['x']) };
    const b = new FilmstripBuilder(v, { decode: async () => { throw new EngineError(415, 'This browser cannot decode HEVC video.'); } }, ops);
    b.start();
    expect(await b.sheet(0)).toBeNull();
    expect(b.state).toBe('failed');
    expect(b.error).toMatch(/cannot decode/);
  });
});
