// The pure parts of frames.ts and thumbs.ts: which samples to decode for a time, the upright geometry, which keyframes make the filmstrip.
// The VideoDecoder and canvas parts need a browser and are not run here.
import { describe, expect, it } from 'vitest';
import type { Demuxed, SampleTable } from '../src/browser/demux';
import { decodePlan, sampleAt, syncBefore, uprightOf, FRAME_LONG_SIDE } from '../src/browser/frames';
import { openGopTable } from './open-gop-table';
import { CELL_H, CELL_MAX_W, planFilmstrip, SLOTS_MAX } from '../src/browser/thumbs';

/** `n` pictures at `fps`, a sync sample every `gop`, with a B-frame reorder: decode order 0,2,1,4,3,... (pts swaps in pairs) */
function table(n: number, fps: number, gop: number, reorder = false): SampleTable {
  const pts = new Float64Array(n);
  const sync = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let p = i;
    if (reorder && i % gop !== 0) p = i % 2 === 1 ? i + 1 : i - 1; // swap neighbours inside a GOP
    if (p >= n || (reorder && p % gop === 0 && i % gop !== 0)) p = i;
    pts[i] = p / fps;
    sync[i] = i % gop === 0 ? 1 : 0;
  }
  const order = Uint32Array.from({ length: n }, (_, i) => i).sort((a, b) => pts[a] - pts[b]);
  return { n, pts, dts: Float64Array.from({ length: n }, (_, i) => i / fps), offset: new Float64Array(n), size: new Uint32Array(n), sync, order, tsUs: new Float64Array(n) };
}

describe('decode plan', () => {
  const t = table(90, 30, 30);
  it('sampleAt is the picture on screen at t', () => {
    expect(sampleAt(t, 0)).toBe(0);
    expect(sampleAt(t, 1 / 30 - 0.001)).toBe(0);
    expect(sampleAt(t, 1 / 30)).toBe(1);
    expect(sampleAt(t, 1.5)).toBe(45);
    expect(sampleAt(t, 99)).toBe(89);
    expect(sampleAt(t, -5)).toBe(0);
  });
  it('syncBefore finds the keyframe the decode must start from', () => {
    expect(syncBefore(t, 0)).toBe(0);
    expect(syncBefore(t, 29)).toBe(0);
    expect(syncBefore(t, 30)).toBe(30);
    expect(syncBefore(t, 75)).toBe(60);
  });
  it('plans a run from the keyframe to the wanted sample', () => {
    expect(decodePlan(t, 2.2)).toEqual({ target: 66, first: 60, last: 66 });
    expect(decodePlan(t, 0)).toEqual({ target: 0, first: 0, last: 0 });
  });
  it('with reordered pictures the target is chosen by presentation time, the run by decode order', () => {
    const r = table(30, 30, 10, true);
    const p = decodePlan(r, 4 / 30); // the picture shown 4th was decoded at index 3
    expect(r.pts[p.target]).toBeCloseTo(4 / 30, 9);
    expect(p.first).toBe(0);
    expect(p.last).toBe(p.target);
  });
});

describe('decode plan of a leading picture (open GOP, open-GOP parity notes)', () => {
  // 6 groups of 12, 3 leading pictures each: decode order key, 3 leading (shown before the key), the rest
  const t = openGopTable(30, 6, 12, 3);
  const leading = (i: number) => {
    let g = i;
    while (!t.sync[g]) g--;
    return t.pts[i] < t.pts[g];
  };
  it('a leading picture of a later group starts its run one keyframe earlier', () => {
    const lead = [...Array(t.n).keys()].filter((i) => leading(i) && i >= 12);
    expect(lead.length).toBe(15);
    for (const i of lead) {
      const p = decodePlan(t, t.pts[i]);
      expect(p.target).toBe(i);
      expect(p.last).toBe(i);
      expect(p.first, `target ${i}`).toBe(Math.floor(i / 12) * 12 - 12);
      expect(t.sync[p.first]).toBe(1);
    }
  });
  it('the leading pictures of the first group have no keyframe before them and the key picture and the rest start at their own keyframe', () => {
    const first = decodePlan(t, t.pts[1]);
    expect(leading(first.target)).toBe(true);
    expect(first.first).toBe(0);
    for (const i of [12, 16, 20, 30]) expect(decodePlan(t, t.pts[i])).toEqual({ target: i, first: Math.floor(i / 12) * 12, last: i });
  });
});

describe('upright geometry', () => {
  it('no rotation: 1280x720 becomes 640x360, never enlarged', () => {
    const u = uprightOf(1280, 720, 0, false);
    expect([u.w, u.h, u.outW, u.outH, u.theta, u.flipY]).toEqual([1280, 720, 640, 360, -0, false]);
    expect([uprightOf(320, 180, 0, false).outW, uprightOf(320, 180, 0, false).outH]).toEqual([320, 180]);
  });
  it('a quarter turn swaps the sides; ffprobe rotation is counter-clockwise, the canvas turns clockwise', () => {
    const u = uprightOf(1920, 1080, -90, false);
    expect([u.w, u.h, u.outW, u.outH]).toEqual([1080, 1920, 360, FRAME_LONG_SIDE]);
    expect(u.theta).toBeCloseTo(Math.PI / 2, 9); // rotation -90 (an iPhone held upright): turn 90 degrees clockwise
    expect(uprightOf(1920, 1080, 90, false).theta).toBeCloseTo(-Math.PI / 2, 9);
    expect(uprightOf(1920, 1080, 270, false).outW).toBe(360);
    expect(uprightOf(1920, 1080, 180, false).outW).toBe(640);
  });
  it('the sample aspect widens the picture before the turn; a mirrored matrix flips vertically', () => {
    const a = uprightOf(1440, 1080, 0, false, 4 / 3);
    expect([a.w, a.h]).toEqual([1920, 1080]);
    const b = uprightOf(1440, 1080, 90, true, 4 / 3);
    expect([b.w, b.h, b.flipY]).toEqual([1080, 1920, true]);
    expect(uprightOf(100, 50, 0, false, 0).w).toBe(100); // an unknown sar counts as square
  });
});

describe('filmstrip plan', () => {
  function video(duration: number, keyEvery: number, over: Partial<Demuxed> = {}) {
    const fps = 30;
    const n = Math.round(duration * fps);
    const s = table(n, fps, Math.round(keyEvery * fps));
    return { demuxed: { samples: s, offset: 0, duration, rotation: 0, sar: 1, video: { width: 1280, height: 720 }, ...over } as unknown as Demuxed, src: { size: 0, read: async () => new ArrayBuffer(0) } };
  }
  it('a 60 s video with a keyframe every second: one cell per second, one sheet, 16:9 cells', () => {
    const p = planFilmstrip(video(60, 1));
    expect(p.picks).toHaveLength(60);
    expect(p.sheets).toBe(1);
    expect([p.cell_h, p.cell_w, p.cols, p.rows]).toEqual([CELL_H, 192, 10, 10]);
    expect(p.picks.map((x) => x.t)).toEqual([...Array(60).keys()]);
  });
  it('sparse keyframes are used once each; dense ones are thinned to the slots', () => {
    expect(planFilmstrip(video(60, 10)).picks).toHaveLength(6);
    const long = planFilmstrip(video(2400, 0.5)); // 40 min, a keyframe twice a second: slot = 4 s -> 600 cells, 6 sheets
    expect(long.picks.length).toBeLessThanOrEqual(SLOTS_MAX);
    expect(long.picks.length).toBeGreaterThan(SLOTS_MAX - 5);
    expect(long.sheets).toBe(Math.ceil(long.picks.length / 100));
  });
  it('a portrait file gets narrow cells, and none is wider than the maximum', () => {
    expect(planFilmstrip(video(10, 1, { rotation: 90 })).cell_w).toBe(61);
    expect(planFilmstrip(video(10, 1, { video: { width: 4000, height: 1000 } as never })).cell_w).toBe(CELL_MAX_W);
  });
  it('picks stay on keyframes and ascend', () => {
    const v = video(30, 2.5);
    const p = planFilmstrip(v);
    for (const x of p.picks) expect(v.demuxed.samples.sync[x.sample]).toBe(1);
    expect(p.picks.map((x) => x.t)).toEqual([...p.picks.map((x) => x.t)].sort((a, b) => a - b));
  });
});
