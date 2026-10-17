// Tap targets in the browser (docs/T9B_DESIGN.md section 5), with mock frames: the pick flow of BrowserEngine (tap, undo, zone, reset), the
// tap target (gallery, margin, continuity, growth) driven by scoreVideo over stand-in pictures, and the TapError texts. No model and no browser here.
import { describe, expect, it } from 'vitest';
import type { Demuxed } from '../src/browser/demux';
import type { Detection } from '../src/browser/detect';
import type { OpenVideo } from '../src/browser/frames';
import { buildRunDoc, resolveSettings, scoreVideo } from '../src/browser/pipeline';
import { BrowserTap, TapError, targetFor, ZoneTarget } from '../src/browser/targets';
import { BrowserStore, MemoryBackend } from '../src/browser/store';
import { BrowserEngine, type Runner } from '../src/engine/browser';
import { blocking, preflightRows } from '../src/model/preflight';
import type { RunRequest } from '../src/engine/types';

// ---------------------------------------------------------------- stand-in pictures
function demuxedOf(fps: number, seconds: number): Demuxed {
  const n = Math.round(fps * seconds);
  const pts = Float64Array.from({ length: n }, (_, i) => i / fps);
  const order = Uint32Array.from({ length: n }, (_, i) => i);
  const sync = Uint8Array.from({ length: n }, (_, i) => (i % fps === 0 ? 1 : 0));
  const samples = { n, pts, dts: pts, offset: new Float64Array(n), size: new Uint32Array(n).fill(1), sync, order, tsUs: Float64Array.from(pts, (p) => Math.round(p * 1e6)) };
  return {
    codec: 'hevc', profile: null, video: { trackId: 1, codecString: 'hvc1.1.6.L93.B0', fourcc: 'hvc1', description: null, width: 1280, height: 720, timescale: 1000 }, samples,
    keyframes: Array.from({ length: seconds }, (_, i) => i), rotation: 0, mirrored: false, sar: 1, fps, captureFps: fps, duration: seconds, offset: 0, endTime: seconds, formatStart: 0,
    hasAudio: false, audio: null, warnings: [], stats: { chunkBytes: 1, reads: 1, bytesRead: 1, peakHeldBytes: 1 },
  } as unknown as Demuxed;
}

/** A decoder that hands over the wanted pictures in order; the picture itself is a stand-in with a close(). */
const decoderOf = () => ({
  decode: async (_v: OpenVideo, runs: [number, number][], want: Set<number>, o: { onFrame?: (i: number, f: never) => Promise<void> | void } = {}) => {
    for (const [a, b] of runs) for (let i = a; i <= b; i++) if (want.has(i)) await o.onFrame?.(i, { close() {} } as never);
    return new Map();
  },
});

/** unit vector from coordinates (padded with zeros to 8 numbers) */
const unit = (...xs: number[]): Float32Array => {
  const v = new Float32Array(8);
  xs.forEach((x, i) => (v[i] = x));
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
};
/** a vector whose cosine with e0 is exactly c, in direction e1 */
const cosWith0 = (c: number): Float32Array => unit(c, Math.sqrt(1 - c * c));
const A_BOX: [number, number, number, number] = [0.1, 0.1, 0.3, 0.9];
const B_BOX: [number, number, number, number] = [0.6, 0.1, 0.8, 0.9];

interface Seen {
  score: number;
  box: [number, number, number, number];
  look: Float32Array | null;
}
/** Runs the tap target over scripted pictures: `script(t)` lists what the detector saw; the embedder returns each one's `look`. */
async function runTap(taps: { t: number; box: [number, number, number, number] }[], script: (t: number) => Seen[], seconds = 8, opts: { zone?: [number, number, number, number]; fps?: number } = {}) {
  const fps = opts.fps ?? 5;
  const dm = demuxedOf(fps, seconds);
  const video = { demuxed: dm, src: {} } as unknown as OpenVideo;
  let k = 0;
  const times = dm.samples.pts;
  let current: Seen[] = [];
  const target = opts.zone ? new ZoneTarget(new BrowserTap(taps), opts.zone) : new BrowserTap(taps);
  const scored = await scoreVideo(
    {
      video, decoder: decoderOf() as never, prepare: () => new Float32Array(1), target,
      detect: async () => { current = script(times[k++]); return current.map((c) => ({ score: c.score, box: c.box }) as Detection); },
      embed: async (dets) => { dets.forEach((d, i) => (d.embedding = current[i].look)); },
    },
    fps, seconds,
  );
  return scored;
}
const presentAt = (s: Awaited<ReturnType<typeof runTap>>, from: number, to: number) => s.times.map((t, i) => (t >= from - 1e-9 && t < to - 1e-9 ? s.scores[i] > 0 : null)).filter((x) => x !== null) as boolean[];

describe('BrowserTap over scripted pictures', () => {
  const E0 = unit(1);
  const E1 = unit(0, 1);

  it('follows the tapped person through a gap and ignores the other one', async () => {
    // A is in the picture from 0 to 3 s and from 5 s on (back after two seconds); B is always there. The tap is on A at 1 s.
    const s = await runTap([{ t: 1, box: A_BOX }], (t) => {
      const out: Seen[] = [];
      if (t < 3 || t >= 5) out.push({ score: 0.9, box: A_BOX, look: E0 });
      out.push({ score: 0.85, box: B_BOX, look: E1 });
      return out;
    });
    expect(s.times).toHaveLength(40);
    expect(presentAt(s, 0, 3).every(Boolean)).toBe(true);
    expect(presentAt(s, 3, 5).some(Boolean)).toBe(false);
    expect(presentAt(s, 5, 8).every(Boolean)).toBe(true); // found again by how they look
    s.boxes.forEach((b) => { if (b) expect(b).toEqual(A_BOX); }); // never B
    expect(s.scores.filter((x) => x > 0).every((x) => x === 0.9)).toBe(true); // the detector's own score
    expect(s.embedS).toBeTypeOf('number');
  });

  it('a person who looks different from the gallery (cosine under 0.62) is not taken, one just over 0.66 is', async () => {
    const lookalike = (c: number) => async () => {
      const s = await runTap([{ t: 0, box: A_BOX }], (t) => (t < 0.5 ? [{ score: 0.9, box: A_BOX, look: E0 }] : [{ score: 0.9, box: B_BOX, look: cosWith0(c) }]), 4);
      return s.scores.slice(3).some((x) => x > 0);
    };
    expect(await lookalike(0.6)()).toBe(false);
    expect(await lookalike(0.67)()).toBe(true);
  });

  it('a box that continues the last match is carried when its look is a little short (0.62 to 0.66); a far box is not', async () => {
    const run = async (box: [number, number, number, number]) => {
      const s = await runTap([{ t: 0, box: A_BOX }], (t) => (t < 0.5 ? [{ score: 0.9, box: A_BOX, look: E0 }] : [{ score: 0.9, box, look: cosWith0(0.64) }]), 3);
      return s.scores.map((x) => x > 0);
    };
    const near = await run([0.11, 0.1, 0.31, 0.9]); // overlaps A's last box
    expect(near.slice(3).every(Boolean)).toBe(true); // from 0.6 s on, after the pictures the tap was made on
    const far = await run(B_BOX);
    expect(far.slice(3).some(Boolean)).toBe(false);
    const faint = await runTap([{ t: 0, box: A_BOX }], (t) => (t < 0.5 ? [{ score: 0.9, box: A_BOX, look: E0 }] : [{ score: 0.9, box: [0.11, 0.1, 0.31, 0.9], look: cosWith0(0.5) }]), 3);
    expect(faint.scores.slice(3).some((x) => x > 0)).toBe(false); // the bonus cannot make a different look pass
  });

  it('two look-alikes in one picture: neither is taken (margin 0.04), a clear lead is', async () => {
    const pair = (a: number, b: number) => runTap([{ t: 0, box: A_BOX }], (t) => (t < 0.5 ? [{ score: 0.9, box: A_BOX, look: E0 }] : [{ score: 0.9, box: [0.4, 0.1, 0.5, 0.9], look: cosWith0(a) }, { score: 0.9, box: B_BOX, look: cosWith0(b) }]), 3);
    const close = await pair(0.9, 0.88);
    expect(close.scores.slice(3).some((x) => x > 0)).toBe(false);
    const clear = await pair(0.9, 0.7);
    expect(clear.scores.slice(3).every((x) => x > 0)).toBe(true);
    expect(clear.boxes[3]).toEqual([0.4, 0.1, 0.5, 0.9]);
  });

  it('the gallery grows: a person whose look drifts step by step is followed past where the first look no longer matches', async () => {
    const turn = (a: number) => unit(Math.cos(a), Math.sin(a));
    const s = await runTap([{ t: 0, box: A_BOX }], (t) => [{ score: 0.9, box: A_BOX, look: turn(t * 5 * 0.6) }], 4); // 0.6 rad per picture: cosine 0.83 to the last, negative to the first after 3
    expect(s.scores.every((x) => x > 0)).toBe(true);
  });

  it('the look can grow backwards too: pictures before the tap are found from the gallery grown after it', async () => {
    const turn = (a: number) => unit(Math.cos(a), Math.sin(a));
    const s = await runTap([{ t: 3, box: A_BOX }], (t) => [{ score: 0.9, box: A_BOX, look: turn((t - 3) * 5 * 0.6) }], 4);
    expect(s.scores.every((x) => x > 0)).toBe(true);
  });

  it('several taps all seed the gallery (a second outfit tapped again)', async () => {
    const other = unit(0, 0, 1);
    const script = (t: number): Seen[] => [{ score: 0.9, box: A_BOX, look: t < 2 ? E0 : other }];
    const one = await runTap([{ t: 0, box: A_BOX }], script, 4);
    expect(one.scores.slice(10).some((x) => x > 0)).toBe(false); // the second outfit shares nothing with the first
    const two = await runTap([{ t: 0, box: A_BOX }, { t: 3, box: A_BOX }], script, 4);
    expect(two.scores.every((x) => x > 0)).toBe(true);
  });

  it('a zone drops a match outside it, and the answer keeps the order the pictures came in', async () => {
    const s = await runTap([{ t: 0, box: A_BOX }], () => [{ score: 0.9, box: A_BOX, look: E0 }], 4, { zone: [0.5, 0, 0.5, 1] });
    expect(s.scores.every((x) => x === 0)).toBe(true); // A's centre is at x = 0.2
    const t = new BrowserTap([{ t: 0, box: A_BOX }]);
    const late = { score: 0.9, box: A_BOX, embedding: E0 };
    t.evaluate([late], 5);
    t.evaluate([late], 0); // shown out of time order
    t.evaluate([{ ...late, embedding: cosWith0(0.1) }], 2);
    expect(t.finalize().map((h) => h.score > 0)).toEqual([true, true, false]);
  });
});

describe('the TapError texts', () => {
  it('a tap with no detection within 1 s says what to do, with the Python wording', async () => {
    // the person is only in the picture from 5 s; the tap at 2 s has nothing within 1 s of it
    await expect(runTap([{ t: 2, box: A_BOX }], (t) => (t >= 5 ? [{ score: 0.9, box: A_BOX, look: unit(1) }] : []))).rejects.toThrow(
      'no person detected within 1 s of the tap at 2.0 s (nothing at that box); pick another moment or position',
    );
  });

  it('a synthetic run on pictures with no detections ends with that message, as a TapError (status 422), not a crash', async () => {
    let err: unknown;
    try {
      await runTap([{ t: 3.5, box: A_BOX }], () => []);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(TapError);
    expect((err as TapError).status).toBe(422);
    expect((err as TapError).message).toMatch(/^no person detected within 1 s of the tap at 3\.5 s/);
  });

  it('a detection that is somewhere else than the tapped box does not count; one 1 s away does', async () => {
    await expect(runTap([{ t: 2, box: A_BOX }], () => [{ score: 0.9, box: B_BOX, look: unit(1) }])).rejects.toThrow(/nothing at that box/);
    const ok = await runTap([{ t: 2, box: A_BOX }], (t) => (t === 3 ? [{ score: 0.9, box: A_BOX, look: unit(1) }] : []), 5);
    expect(ok.scores[15]).toBe(0.9);
  });

  it('names the class when it is not a person', async () => {
    const dm = demuxedOf(5, 2);
    const target = new BrowserTap([{ t: 1, box: A_BOX }], 18);
    await expect(
      scoreVideo({ video: { demuxed: dm, src: {} } as never, decoder: decoderOf() as never, prepare: () => new Float32Array(1), detect: async () => [], embed: async () => {}, target }, 5, 2),
    ).rejects.toThrow(/no dog detected within 1 s of the tap at 1\.0 s/);
  });

  it('a tapped person too small to recognise says so; no taps at all is refused when the target is made', async () => {
    await expect(runTap([{ t: 1, box: A_BOX }], () => [{ score: 0.9, box: A_BOX, look: null }])).rejects.toThrow(/too small to recognise/);
    expect(() => new BrowserTap([])).toThrow('Tap the person in the video first.');
    expect(() => targetFor({ kind: 'tap' }, { taps: [] })).toThrow(/Tap the person/);
    expect(() => targetFor({ kind: 'photos' })).toThrow(/not offered in the browser/);
  });

  it('a cancel during the run never reaches finalize', async () => {
    const ac = new AbortController();
    const dm = demuxedOf(5, 4);
    let n = 0;
    const target = new BrowserTap([{ t: 0, box: A_BOX }]);
    let finalized = false;
    const orig = target.finalize.bind(target);
    target.finalize = () => { finalized = true; return orig(); };
    await expect(
      scoreVideo({ video: { demuxed: dm, src: {} } as never, decoder: decoderOf() as never, prepare: () => new Float32Array(1), detect: async () => { if (++n === 3) ac.abort(); return []; }, embed: async () => {}, target }, 5, 4, ac.signal),
    ).rejects.toMatchObject({ status: 499 });
    expect(finalized).toBe(false);
  });
});

describe('the run document of a tap run', () => {
  it('names the target tap, records the embedder time and builds clips from the matched stretches', async () => {
    const s = await runTap([{ t: 1, box: A_BOX }], (t) => (t >= 2 && t < 6 ? [{ score: 0.9, box: A_BOX, look: unit(1) }] : t === 1 ? [{ score: 0.9, box: A_BOX, look: unit(1) }] : []), 8);
    const st = resolveSettings({ preset: 'balanced', advanced: {} });
    const doc = buildRunDoc(s, st, { offset: 0, end_time: 8, keyframes: [0, 1, 2, 3, 4, 5, 6, 7] }, { target: 'tap', cls: 1, zone: null, zoneOverlap: null, detector: 'x', model: 'small', warnings: [], totalS: 1 });
    expect(doc.segments.settings.target).toBe('tap');
    expect(doc.segments.timing_s).toHaveProperty('embed_model_s');
    expect(doc.segments.clips.length).toBeGreaterThan(0);
    expect(doc.samples.track).toBeNull();
  });
});

// ---------------------------------------------------------------- the engine's pick flow
const MODELS = (over: Record<string, unknown> = {}) => ({
  available: true, error: null, detectorReady: true, allReady: true, totalBytes: 10, cachedBytes: 10, persisted: true,
  files: [{ name: 'rfdetr_small_v1.onnx', cached: true }, { name: 'dinov2_s_v1.onnx', cached: true }], ...over,
});
const fakeModels = (over: Record<string, unknown> = {}) =>
  ({ base: 'https://page.test/models/', status: async () => MODELS(over), manifest: async () => ({ format: 1, files: {} }), download: async () => {}, remove: async () => {} }) as never;
function fakeRunner(script: (msg: any) => Promise<unknown>): Runner & { sent: any[] } {
  const sent: any[] = [];
  return { sent, dispose() {}, send: (async (msg: unknown) => { sent.push(msg); return script(msg); }) as never };
}
async function engineWith(runner: Runner, models = fakeModels(), store = new BrowserStore(new MemoryBackend())) {
  const e = new BrowserEngine({
    demux: async () => demuxedOf(30, 20), fileUrl: () => 'blob:x', revokeFileUrl: () => {}, store, runner, models, env: async () => ({ webgpu: true, webcodecs: true, hevc: true }),
  });
  const m = await e.openFile(new File([new Uint8Array(8)], 'synthetic.mp4', { lastModified: 5 }));
  return { e, m, store };
}
const req = (id: string, over: Partial<RunRequest> = {}): RunRequest => ({ media_id: id, target: { kind: 'tap', class: 'person' }, zone: null, zone_overlap: null, preset: 'balanced', slowmo: 'off', slowmo_audio: false, advanced: {}, ...over });
const until = async (f: () => Promise<boolean>) => { for (let i = 0; i < 100; i++) { if (await f()) return; await new Promise((r) => setTimeout(r, 5)); } throw new Error('timed out'); };
const idle = fakeRunner(async () => ({}));

describe('BrowserEngine pick flow: tap, undo, zone, reset', () => {
  it('starts empty, remembers taps with rounded numbers, undoes the last one and resets everything', async () => {
    const { e, m } = await engineWith(idle);
    expect(await e.pickState(m.id, 'person')).toEqual({ target_key: '', taps: [], zone: null, file: null, class: 'person', candidates: [] });
    const a = await e.tap(m.id, 'person', 3.0334, [0.123456, 0.2, 0.5, 0.9], 1);
    expect(a.taps).toEqual([{ time: 3.033, box: [0.1235, 0.2, 0.5, 0.9], track_id: null, index: 1 }]);
    const b = await e.tap(m.id, 'person', 7, [0.6, 0.2, 0.7, 0.5], 2);
    expect(b.taps.map((t) => t.index)).toEqual([1, 2]);
    expect((await e.pickState(m.id, 'person')).taps).toHaveLength(2);
    expect((await e.undoTap(m.id, 'person')).taps.map((t) => t.index)).toEqual([1]);
    expect((await e.undoTap(m.id, 'person')).taps).toEqual([]);
    expect((await e.undoTap(m.id, 'person')).taps).toEqual([]); // nothing to undo is not an error
    await e.tap(m.id, 'person', 1, [0.1, 0.1, 0.4, 0.9], 1);
    await e.setZone(m.id, 'person', [0.1, 0.2, 0.5, 0.5]);
    const r = await e.resetPick(m.id, 'person');
    expect(r).toMatchObject({ taps: [], zone: null, class: 'person' });
    expect(await e.pickState(m.id, 'person')).toMatchObject({ taps: [], zone: null });
  });

  it('keeps taps per class, and keeps them after the engine is built again over the same storage', async () => {
    const store = new BrowserStore(new MemoryBackend());
    const first = await engineWith(idle, fakeModels(), store);
    await first.e.tap(first.m.id, 'person', 2, [0.1, 0.1, 0.4, 0.9], 1);
    await first.e.tap(first.m.id, 'dog', 4, [0.2, 0.2, 0.5, 0.9], 3);
    const second = await engineWith(idle, fakeModels(), store);
    expect(second.m.id).toBe(first.m.id);
    expect((await second.e.pickState(second.m.id, 'person')).taps.map((t) => t.time)).toEqual([2]);
    expect((await second.e.pickState(second.m.id, 'dog')).taps.map((t) => t.time)).toEqual([4]);
    expect((await second.e.pickState(second.m.id, 'dog')).class).toBe('dog');
    await second.e.resetPick(second.m.id, 'dog');
    expect((await second.e.pickState(second.m.id, 'person')).taps).toHaveLength(1);
  });

  it('the zone is validated and rounded like the Mac', async () => {
    const { e, m } = await engineWith(idle);
    expect((await e.setZone(m.id, 'person', [0.123456, 0.2, 0.5, 0.3333333])).zone).toEqual([0.1235, 0.2, 0.5, 0.3333]);
    await expect(e.setZone(m.id, 'person', [0.9, 0.2, 0.3, 0.3])).rejects.toMatchObject({ status: 400, message: /fractions of the upright frame/ });
    await expect(e.setZone(m.id, 'person', [0.1, 0.2] as never)).rejects.toMatchObject({ status: 400 });
    expect((await e.pickState(m.id, 'person')).zone).toEqual([0.1235, 0.2, 0.5, 0.3333]); // a refused zone changes nothing
    expect((await e.setZone(m.id, 'person', null)).zone).toBeNull();
    expect(await e.setZone(m.id, 'person', [0, 0, 1, 1])).toMatchObject({ zone: [0, 0, 1, 1], taps: [] });
  });

  it('refuses a tap with a bad box, a time outside the video or a class that does not exist', async () => {
    const { e, m } = await engineWith(idle);
    await expect(e.tap(m.id, 'person', 1, [0.5, 0.1, 0.4, 0.9], 1)).rejects.toMatchObject({ status: 400, message: /x0 < x1/ });
    await expect(e.tap(m.id, 'person', 1, [0.1, 0.1, 1.2, 0.9], 1)).rejects.toMatchObject({ status: 400 });
    await expect(e.tap(m.id, 'person', 1, [0.1, 0.1, 0.2] as never, 1)).rejects.toMatchObject({ status: 400 });
    await expect(e.tap(m.id, 'person', 25, [0.1, 0.1, 0.4, 0.9], 1)).rejects.toMatchObject({ status: 400, message: /outside the video/ });
    await expect(e.tap(m.id, 'person', Number.NaN, [0.1, 0.1, 0.4, 0.9], 1)).rejects.toMatchObject({ status: 400 });
    await expect(e.tap(m.id, 'unicorn', 1, [0.1, 0.1, 0.4, 0.9], 1)).rejects.toThrow(/unknown class/);
    expect((await e.pickState(m.id, 'person')).taps).toEqual([]);
  });

  it('pickFrameUrl is the picture at t, the one the boxes are drawn over', async () => {
    const { e, m } = await engineWith(idle);
    expect(e.pickFrameUrl(m.id, 3.0334, 'person')).toBe(e.frameUrl(m.id, 3.0334, 640));
  });
});

const scoredOf = () => {
  const times = Array.from({ length: 100 }, (_, i) => i / 5);
  return { times, scores: times.map((t) => (t >= 5 && t < 12 ? 0.9 : 0)), boxes: times.map((t) => (t >= 5 && t < 12 ? [0.1, 0.1, 0.4, 0.9] : null)), detectS: 1, embedS: 0.5, wallS: 2, planned: 100 };
};

describe('a tap run through the engine', () => {
  it('sends the taps to the runner, stores the run as "tap" and lists its clips', async () => {
    const runner = fakeRunner(async () => ({ scored: scoredOf(), provider: 'webgpu', warnings: [], cls: 1, targetName: 'tap' }));
    const { e, m } = await engineWith(runner);
    await e.tap(m.id, 'person', 5.5, [0.1, 0.1, 0.4, 0.9], 1);
    await e.tap(m.id, 'person', 11, [0.12, 0.1, 0.42, 0.9], 2);
    const ref = await e.startRun(req(m.id, { zone: [0, 0, 1, 1], zone_overlap: null }));
    expect(ref.run_id).toBe(`${m.id}-tap`);
    await until(async () => (await e.currentJob()) === null);
    expect(await e.job(ref.job.id)).toMatchObject({ state: 'done' });
    expect(runner.sent[0]).toMatchObject({ op: 'run', target: { kind: 'tap', class: 'person' }, taps: [{ t: 5.5, box: [0.1, 0.1, 0.4, 0.9] }, { t: 11, box: [0.12, 0.1, 0.42, 0.9] }], zone: [0, 0, 1, 1] });
    const runs = await e.runs(m.id);
    expect(runs).toHaveLength(1);
    expect(runs[0]).toMatchObject({ slug: 'tap', target: 'tap', n_clips: 1 });
    expect((await e.run(ref.run_id!)).segments.timing_s).toHaveProperty('embed_model_s');
  });

  it('a run of another class uses that class\'s taps', async () => {
    const runner = fakeRunner(async () => ({ scored: scoredOf(), provider: 'webgpu', warnings: [], cls: 18, targetName: 'tap' }));
    const { e, m } = await engineWith(runner);
    await e.tap(m.id, 'person', 5, [0.1, 0.1, 0.4, 0.9], 1);
    await expect(e.startRun(req(m.id, { target: { kind: 'tap', class: 'dog' } }))).rejects.toMatchObject({ status: 422, message: /Tap the person/ });
    await e.tap(m.id, 'dog', 6, [0.2, 0.1, 0.5, 0.9], 1);
    await e.startRun(req(m.id, { target: { kind: 'tap', class: 'dog' } }));
    await until(async () => (await e.currentJob()) === null);
    expect(runner.sent[0]).toMatchObject({ target: { class: 'dog' }, taps: [{ t: 6 }] });
  });

  it('no tap yet is a clear 422 and leaves no job behind; the recognition model missing is the models 409', async () => {
    const runner = fakeRunner(async () => ({}));
    const { e, m } = await engineWith(runner);
    await expect(e.startRun(req(m.id))).rejects.toMatchObject({ status: 422, message: 'Tap the person in the video first.' });
    expect(await e.currentJob()).toBeNull();
    const none = await engineWith(runner, fakeModels({ files: [{ name: 'rfdetr_small_v1.onnx', cached: true }, { name: 'dinov2_s_v1.onnx', cached: false }] }));
    await none.e.tap(none.m.id, 'person', 5, [0.1, 0.1, 0.4, 0.9], 1);
    await expect(none.e.startRun(req(none.m.id))).rejects.toMatchObject({ status: 409, extra: { error: 'models' }, message: /recognition model/ });
    expect(await none.e.currentJob()).toBeNull();
    expect(runner.sent).toEqual([]);
    // any person needs only the detector
    const detOnly = await engineWith(fakeRunner(async () => ({ scored: scoredOf(), provider: 'webgpu', warnings: [], cls: 1, targetName: 'any-person' })), fakeModels({ files: [{ name: 'rfdetr_small_v1.onnx', cached: true }] }));
    await detOnly.e.startRun(req(detOnly.m.id, { target: { kind: 'any', class: 'person' } }));
  });

  it('a TapError from the Worker is the job\'s error text, and no run is kept', async () => {
    const text = 'no person detected within 1 s of the tap at 5.5 s (nothing at that box); pick another moment or position';
    const { e, m } = await engineWith(fakeRunner(async () => { throw new TapError(text); }));
    await e.tap(m.id, 'person', 5.5, [0.1, 0.1, 0.4, 0.9], 1);
    const ref = await e.startRun(req(m.id));
    await until(async () => (await e.job(ref.job.id)).state !== 'running');
    expect(await e.job(ref.job.id)).toMatchObject({ state: 'failed', error: text });
    expect(await e.runs(m.id)).toEqual([]);
    expect(await e.currentJob()).toBeNull();
  });

  it('the session says whether the recognition model is in, and the checklist asks for it only for a tap', async () => {
    const yes = await (await engineWith(idle)).e.session();
    expect(yes.preflight.browser?.models.embedderReady).toBe(true);
    const no = await (await engineWith(idle, fakeModels({ files: [{ name: 'rfdetr_small_v1.onnx', cached: true }] }))).e.session();
    expect(no.preflight.browser?.models).toMatchObject({ detectorReady: true, embedderReady: false });
    const rows = (p: typeof no.preflight, tracking: boolean) => preflightRows(p, { preset: 'balanced', identity: false, tracking }).map((r) => [r.key, r.ok]);
    expect(rows(no.preflight, false).map((r) => r[0])).not.toContain('embedder');
    expect(rows(no.preflight, true)).toContainEqual(['embedder', false]);
    expect(rows(yes.preflight, true)).toContainEqual(['embedder', true]);
  });

  it('activity riding does not ask for the recognition model, and the full checklist lists it', async () => {
    const no = await (await engineWith(idle, fakeModels({ files: [{ name: 'rfdetr_small_v1.onnx', cached: true }] }))).e.session();
    // an Anyone run with Advanced > Activity "riding": tracking is on (Mac) but nothing follows one person
    const riding = preflightRows(no.preflight, { preset: 'balanced', identity: false, tracking: true, follow: false });
    expect(riding.map((r) => r.key)).not.toContain('embedder');
    expect(blocking(riding)).toEqual([]);
    // a tap still needs it
    expect(preflightRows(no.preflight, { preset: 'balanced', identity: false, tracking: true, follow: true }).map((r) => r.key)).toContain('embedder');
    // the first-use screen shows every row
    expect(preflightRows(no.preflight, { preset: null, identity: null, tracking: null, follow: null })).toContainEqual(expect.objectContaining({ key: 'embedder', ok: false }));
  });
});
