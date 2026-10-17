// A run in the tab (docs/T9B_DESIGN.md section 2): the sample plan, scoring, the run document, and BrowserEngine's jobs with a stand-in for the Worker.
import { describe, expect, it } from 'vitest';
import type { Demuxed } from '../src/browser/demux';
import { FrameDecoder, type DecoderDeps, type OpenVideo } from '../src/browser/frames';
import { buildRunDoc, pickSamples, planRuns, resolveSettings, scoreVideo, snapToKeyframe } from '../src/browser/pipeline';
import { openGopTable } from './open-gop-table';
import { AnyPerson } from '../src/browser/targets';
import { BrowserStore, MemoryBackend } from '../src/browser/store';
import { BrowserEngine, type Runner } from '../src/engine/browser';
import type { RunRequest } from '../src/engine/types';

/** `seconds` of video at `fps`, a keyframe every `gop` pictures; with `bframes`, pictures are stored in the order I B B P (decode order differs from shown order). */
function table(fps: number, seconds: number, gop: number, bframes = false) {
  const n = Math.round(fps * seconds);
  const pts = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const g = Math.floor(i / gop) * gop;
    const k = i - g;
    // decode order within a group of four: 0, 3, 1, 2 -> shown at slots 0, 3, 1, 2 of the four
    const slot = bframes ? (k % 4 === 0 ? k : k % 4 === 1 ? k + 2 : k - 1) : k;
    pts[i] = (g + slot) / fps;
  }
  const order = Uint32Array.from({ length: n }, (_, i) => i).sort((a, b) => pts[a] - pts[b]);
  const sync = Uint8Array.from({ length: n }, (_, i) => (i % gop === 0 ? 1 : 0));
  return { n, pts, dts: Float64Array.from({ length: n }, (_, i) => i / fps), offset: new Float64Array(n), size: new Uint32Array(n).fill(1), sync, order, tsUs: Float64Array.from(pts, (p) => Math.round(p * 1e6)) };
}
const demuxedOf = (fps: number, seconds: number, gop: number, bframes = false): Demuxed =>
  ({
    codec: 'hevc', profile: null, video: { trackId: 1, codecString: 'hvc1.1.6.L93.B0', fourcc: 'hvc1', description: null, width: 1280, height: 720, timescale: 1000 },
    samples: table(fps, seconds, gop, bframes), keyframes: Array.from({ length: Math.ceil((fps * seconds) / gop) }, (_, i) => (i * gop) / fps), rotation: 0, mirrored: false, sar: 1, fps, captureFps: fps,
    duration: seconds, offset: 0, endTime: seconds, formatStart: 0, hasAudio: false, audio: null, warnings: [], stats: { chunkBytes: 1, reads: 1, bytesRead: 1, peakHeldBytes: 1 },
  }) as unknown as Demuxed;

describe('which samples a run looks at (decode.iter_samples)', () => {
  it('5 per second of a 30 fps video: 100 for 20 s, one every 6th picture', () => {
    const t = table(30, 20, 30);
    const picks = pickSamples(t, 5, 20);
    expect(picks).toHaveLength(100);
    expect(picks.slice(0, 4).map((i) => t.pts[i])).toEqual([0, 6 / 30, 12 / 30, 18 / 30]);
  });

  it('follows presentation order when pictures are stored out of order', () => {
    const t = table(30, 4, 12, true);
    const picks = pickSamples(t, 5, 4);
    const times = picks.map((i) => t.pts[i]);
    expect(times).toEqual([...times].sort((a, b) => a - b));
    expect(picks).toHaveLength(20);
    expect(new Set(times).size).toBe(20);
  });

  it('a stream sparser than the rate yields each picture once (slots are skipped, not repeated)', () => {
    const t = table(2, 10, 2); // 2 fps video, 5 per second asked
    expect(pickSamples(t, 5, 10)).toHaveLength(20);
  });

  it('keyframes only takes the sync samples, and nothing at or after the end', () => {
    const t = table(30, 10, 30);
    expect(pickSamples(t, null, 10).map((i) => t.pts[i])).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(pickSamples(t, null, 5)).toHaveLength(6); // the picture at the end time itself still counts (decode.iter_samples breaks at end + 1e-4)
    expect(pickSamples(t, null, 4.99)).toHaveLength(5);
    expect(pickSamples(t, 5, 3)).toHaveLength(16);
  });

  it('groups the picks into decoder runs that start at a sync sample', () => {
    const t = table(30, 6, 30);
    const { runs, want } = planRuns(t, pickSamples(t, 5, 6));
    expect(runs).toHaveLength(6);
    for (const [a, b] of runs) {
      expect(t.sync[a]).toBe(1);
      expect(b).toBeGreaterThan(a);
      expect(b - a).toBeLessThan(30);
    }
    expect(want.size).toBe(30);
    expect(planRuns(t, []).runs).toEqual([]);
  });
});

describe('leading pictures of open GOPs (open-GOP parity notes)', () => {
  // a decoder that starts at a keyframe drops the pictures shown before it: they only decode when the run starts a GOP earlier
  const leading = (t: ReturnType<typeof openGopTable>, i: number) => {
    let g = i;
    while (!t.sync[g]) g--;
    return t.pts[i] < t.pts[g];
  };
  const startsBefore = (t: ReturnType<typeof openGopTable>, a: number, i: number) => {
    let g = i;
    while (!t.sync[g]) g--;
    return a < g && t.sync[a] === 1;
  };

  it('a pick shown before its keyframe is in a run that starts at the previous keyframe', () => {
    const t = openGopTable(30, 6, 12, 3);
    const picks = pickSamples(t, 30, 6);
    const lead = picks.filter((i) => leading(t, i) && i >= 12); // the first group has no earlier keyframe to start from, so its leading pictures stay as they are
    expect(lead.length).toBe(5 * 3);
    const { runs } = planRuns(t, picks);
    for (const i of lead) {
      const run = runs.find(([a, b]) => a <= i && i <= b);
      expect(run, `pick ${i}`).toBeDefined();
      expect(startsBefore(t, run![0], i), `pick ${i} starts in ${run}`).toBe(true);
    }
  });

  it('every run starts at a sync sample, runs do not overlap, and every pick is inside one', () => {
    const t = openGopTable(30, 6, 12, 3);
    const picks = pickSamples(t, 5, 6);
    const { runs, want } = planRuns(t, picks);
    for (const [a] of runs) expect(t.sync[a]).toBe(1);
    for (let k = 1; k < runs.length; k++) expect(runs[k][0]).toBeGreaterThan(runs[k - 1][1]);
    for (const i of want) expect(runs.some(([a, b]) => a <= i && i <= b)).toBe(true);
  });

  it('a leading pick that is not shown first of its group only widens the run of the group before', () => {
    const t = openGopTable(30, 4, 12, 3);
    const only = [12 + 1]; // decode index 13 = the leading picture shown at slot 12 + 1 of group 1
    expect(leading(t, only[0])).toBe(true);
    expect(planRuns(t, only).runs).toEqual([[0, 13]]);
    expect(planRuns(t, [12 + 5]).runs).toEqual([[12, 17]]); // a picture after the keyframe is unchanged
  });

  it('streams without leading pictures plan as before', () => {
    const t = table(30, 6, 30, true);
    expect(planRuns(t, pickSamples(t, 5, 6)).runs).toHaveLength(6);
  });

  it('a long chain of leading pictures is split before a run would hold too many pictures, and the pieces still cover every pick', () => {
    const t = openGopTable(30, 60, 12, 3);
    const picks = pickSamples(t, 30, 24);
    const { runs, want } = planRuns(t, picks);
    expect(runs.length).toBeGreaterThan(1);
    for (const i of want) expect(runs.some(([a, b]) => a <= i && i <= b)).toBe(true);
    for (const i of picks.filter((j) => leading(t, j) && j >= 12)) {
      expect(runs.some(([a, b]) => a <= i && i <= b && startsBefore(t, a, i))).toBe(true);
    }
  });
});

describe('overlapping runs hand each picture over once', () => {
  it('two runs that decode the same group give the callee one picture per sample', async () => {
    class VD {
      static isConfigSupported = async () => ({ supported: true });
      state = 'configured';
      private q: number[] = [];
      constructor(private o: { output: (f: unknown) => void }) {}
      configure() {}
      decode(c: { timestamp: number }) { this.q.push(c.timestamp); }
      async flush() { for (const ts of this.q) this.o.output({ timestamp: ts, close() {} }); this.q = []; }
      close() { this.state = 'closed'; }
    }
    class EC { timestamp: number; constructor(i: { timestamp: number }) { this.timestamp = i.timestamp; } }
    const d = demuxedOf(30, 4, 12);
    const v = { demuxed: d, src: { size: 1, read: async () => new ArrayBuffer(1) } } as unknown as OpenVideo;
    const got: number[] = [];
    await new FrameDecoder({ VideoDecoder: VD, EncodedVideoChunk: EC } as unknown as DecoderDeps).decode(v, [[0, 5], [0, 9]], new Set([1, 5, 8]), { onFrame: (i) => { got.push(i); } });
    expect(got.sort((a, b) => a - b)).toEqual([1, 5, 8]);
  });
});

describe('scoring and the run document', () => {
  const frame = () => ({ closed: false, close() { this.closed = true; } });

  it('decodes the plan, runs the detector on every frame in order, closes every frame and builds clips', async () => {
    const dm = demuxedOf(30, 20, 30);
    const video = { demuxed: dm, src: {} } as unknown as OpenVideo;
    const frames: ReturnType<typeof frame>[] = [];
    const seenRuns: [number, number][][] = [];
    const decoder = {
      decode: async (_v: OpenVideo, runs: [number, number][], want: Set<number>, o: { onFrame?: (i: number, f: never) => Promise<void> | void } = {}) => {
        seenRuns.push(runs);
        for (const [a, b] of runs) for (let i = a; i <= b; i++) if (want.has(i)) { const f = frame(); frames.push(f); await o.onFrame?.(i, f as never); }
        return new Map();
      },
    };
    // a person is in the picture from 5 s to 12 s
    let k = 0;
    const times = pickSamples(dm.samples, 5, 20).map((i) => dm.samples.pts[i]);
    const prog: number[] = [];
    const scored = await scoreVideo(
      { video, decoder: decoder as never, prepare: () => new Float32Array(1), detect: async () => { const t = times[k++]; return t >= 5 && t < 12 ? [{ score: 0.9, box: [0.1, 0.1, 0.4, 0.9] }] : []; }, target: new AnyPerson() },
      5, 20, undefined, (f) => prog.push(f),
    );
    expect(scored.times).toHaveLength(100);
    expect(scored.scores.filter((s) => s > 0)).toHaveLength(35);
    expect(frames.every((f) => f.closed)).toBe(true);
    expect(prog.at(-1)).toBeLessThanOrEqual(0.98);
    const st = resolveSettings({ preset: 'balanced', advanced: {} });
    const doc = buildRunDoc(scored, st, { offset: 0, end_time: 20, keyframes: dm.keyframes }, { target: 'any-person', cls: 1, zone: null, zoneOverlap: null, detector: 'x', model: 'small', warnings: [], totalS: 3 });
    expect(doc.segments.clips).toHaveLength(1);
    expect(doc.segments.clips[0]).toMatchObject({ index: 1, start: 3.9, end: 12.9, peak_score: 0.9, expected_copy_start: 3 });
    expect(doc.segments.presence_intervals).toEqual([[4.9, 11.9]]);
    expect(doc.samples).toMatchObject({ enter: 0.5, exit: 0.3, merge_gap: 1.5, min_len: 1, pad: 1, mode: 'detect', track: null });
    expect(doc.samples.t).toHaveLength(100);
    expect(doc.samples.box.filter((b) => b).length).toBe(35);
  });

  it('a cancel between two pictures stops the run and still closes the frame it held', async () => {
    const dm = demuxedOf(30, 4, 30);
    const ac = new AbortController();
    const frames: ReturnType<typeof frame>[] = [];
    const decoder = {
      decode: async (_v: OpenVideo, runs: [number, number][], want: Set<number>, o: { onFrame?: (i: number, f: never) => Promise<void> | void } = {}) => {
        for (const [a, b] of runs) for (let i = a; i <= b; i++) if (want.has(i)) { const f = frame(); frames.push(f); await o.onFrame?.(i, f as never); }
        return new Map();
      },
    };
    let n = 0;
    await expect(scoreVideo({ video: { demuxed: dm, src: {} } as never, decoder: decoder as never, prepare: () => new Float32Array(1), detect: async () => { if (++n === 3) ac.abort(); return []; }, target: new AnyPerson() }, 5, 4, ac.signal))
      .rejects.toMatchObject({ status: 499 });
    expect(frames.length).toBeLessThan(20);
    expect(frames.every((f) => f.closed)).toBe(true);
  });

  it('no stretch found says so, with the highest score', () => {
    const st = resolveSettings({ preset: 'balanced', advanced: {} });
    const doc = buildRunDoc({ times: [0, 0.2, 0.4], scores: [0.1, 0.2, 0.3], boxes: [null, null, null], detectS: 0, wallS: 0, planned: 3 }, st, { offset: 0, end_time: 1, keyframes: [0] }, { target: 'any-person', cls: 1, zone: null, zoneOverlap: null, detector: 'x', model: 'small', warnings: [], totalS: 0 });
    expect(doc.segments.clips).toEqual([]);
    expect(doc.segments.warnings[0]).toMatch(/no stretch found: the highest score 0.30000/);
  });

  it('settings follow pipeline.Settings and refuse what the Mac refuses', () => {
    expect(resolveSettings({ preset: 'balanced', advanced: {} })).toEqual({ preset: 'balanced', fps: 5, pad: 1, min_len: 1, merge_gap: 1.5, enter: 0.5, exit: 0.3, min_box: 0 });
    expect(resolveSettings({ preset: 'fast', advanced: {} }).fps).toBeNull();
    expect(resolveSettings({ preset: 'balanced', advanced: { fps: 0 } }).fps).toBeNull();
    expect(resolveSettings({ preset: 'balanced', advanced: { enter: 0.2, fps: 2 } })).toMatchObject({ enter: 0.2, exit: 0.2, fps: 2 });
    expect(() => resolveSettings({ preset: 'accurate', advanced: {} })).toThrow(/not offered/);
    expect(() => resolveSettings({ preset: 'balanced', advanced: { exit: 0.9 } })).toThrow(/must not exceed/);
    expect(() => resolveSettings({ preset: 'balanced', advanced: { enter: 1.5 } })).toThrow(/between 0 and 1/);
  });

  it('snaps to the keyframe at or before', () => {
    expect(snapToKeyframe(4.5, [0, 2, 4, 6])).toBe(4);
    expect(snapToKeyframe(4, [0, 2, 4, 6])).toBe(4);
    expect(snapToKeyframe(-1, [0, 2])).toBe(0);
    expect(snapToKeyframe(3, [])).toBe(3);
  });
});

// ---------------------------------------------------------------- the engine
const MODELS_OK = { available: true, error: null, detectorReady: true, allReady: true, totalBytes: 10, cachedBytes: 10, persisted: true, files: [] };
const fakeModels = (over: Record<string, unknown> = {}) =>
  ({ base: 'https://page.test/models/', status: async () => ({ ...MODELS_OK, ...over }), manifest: async () => ({ format: 1, files: {} }), download: async () => {}, remove: async () => {} }) as never;

function fakeRunner(script: (msg: any, progress: (f: number, m: string) => void, signal?: AbortSignal) => Promise<unknown>): Runner & { sent: any[] } {
  const sent: any[] = [];
  return { sent, dispose() {}, send: (async (msg: unknown, progress: (f: number, m: string) => void, signal?: AbortSignal) => { sent.push(msg); return script(msg, progress, signal); }) as never };
}

async function engineWith(runner: Runner, models = fakeModels()) {
  const e = new BrowserEngine({
    demux: async () => demuxedOf(30, 20, 30), fileUrl: () => 'blob:x', revokeFileUrl: () => {}, store: new BrowserStore(new MemoryBackend()), runner, models,
    env: async () => ({ webgpu: true, webcodecs: true, hevc: true }),
  });
  const m = await e.openFile(new File([new Uint8Array(8)], 'synthetic.mp4', { lastModified: 5 }));
  return { e, m };
}
const req = (id: string, over: Partial<RunRequest> = {}): RunRequest => ({ media_id: id, target: { kind: 'any', class: 'person' }, zone: null, zone_overlap: null, preset: 'balanced', slowmo: 'off', slowmo_audio: false, advanced: {}, ...over });
const scoredOf = (hit: [number, number]) => {
  const times = Array.from({ length: 100 }, (_, i) => i / 5);
  return { times, scores: times.map((t) => (t >= hit[0] && t < hit[1] ? 0.9 : 0.05)), boxes: times.map((t) => (t >= hit[0] && t < hit[1] ? [0.1, 0.1, 0.5, 0.9] : null)), detectS: 1, wallS: 2, planned: 100 };
};
const until = async (f: () => Promise<boolean>) => { for (let i = 0; i < 100; i++) { if (await f()) return; await new Promise((r) => setTimeout(r, 5)); } throw new Error('timed out'); };

describe('BrowserEngine runs', () => {
  it('a run goes through the runner, ends done and is listed with its clips, samples and a stable id', async () => {
    const runner = fakeRunner(async (_m, progress) => { progress(0.5, 'detect: 10 of 20 s'); return { scored: scoredOf([5, 12]), provider: 'webgpu', warnings: [], cls: 1, targetName: 'any-person' }; });
    const { e, m } = await engineWith(runner);
    const ref = await e.startRun(req(m.id));
    expect(ref.job.state).toBe('running');
    expect(ref.run_id).toBe(`${m.id}-any-person`);
    await until(async () => (await e.job(ref.job.id)).state !== 'running');
    const j = await e.job(ref.job.id);
    expect(j).toMatchObject({ state: 'done', fraction: 1, error: null, kind: 'run' });
    expect(await e.currentJob()).toBeNull();
    const msg = runner.sent[0];
    expect(msg).toMatchObject({ op: 'run', fps: 5, target: { kind: 'any' }, base: 'https://page.test/models/' });
    const runs = await e.runs(m.id);
    expect(runs).toHaveLength(1);
    expect(runs[0]).toMatchObject({ slug: 'any-person', n_clips: 1, preset: 'balanced', target: 'any-person' });
    const data = await e.run(ref.run_id!);
    expect(data.segments.clips[0]).toMatchObject({ start: 3.9, end: 12.9 });
    expect((await e.samples(ref.run_id!)).t).toHaveLength(100);
    expect((await e.resegment(ref.run_id!, {})).same_as_run).toBe(true);
  });

  it('a second run of the same target replaces the first (same id), and a second job at once is 409 busy', async () => {
    let release: (() => void) | null = null;
    const runner = fakeRunner(() => new Promise((ok) => { release = () => ok({ scored: scoredOf([5, 12]), provider: 'webgpu', warnings: [], cls: 1, targetName: 'any-person' }); }));
    const { e, m } = await engineWith(runner);
    const a = await e.startRun(req(m.id));
    await expect(e.startRun(req(m.id))).rejects.toMatchObject({ status: 409, extra: { error: 'busy', job: { id: a.job.id } } });
    expect((await e.currentJob())?.id).toBe(a.job.id);
    release!();
    await until(async () => (await e.currentJob()) === null);
    const b = await e.startRun(req(m.id));
    expect(b.run_id).toBe(a.run_id);
    release!();
    await until(async () => (await e.currentJob()) === null);
    expect(await e.runs(m.id)).toHaveLength(1);
  });

  it('cancel aborts the runner, the job ends cancelled and nothing is stored', async () => {
    const runner = fakeRunner((_m, _p, signal) => new Promise((_ok, no) => signal!.addEventListener('abort', () => no(Object.assign(new Error('Cancelled.'), { status: 499 })))));
    const { e, m } = await engineWith(runner);
    const ref = await e.startRun(req(m.id));
    await e.cancel(ref.job.id);
    await until(async () => (await e.job(ref.job.id)).state !== 'running');
    expect((await e.job(ref.job.id)).state).toBe('cancelled');
    expect(await e.runs(m.id)).toEqual([]);
    expect(await e.currentJob()).toBeNull();
  });

  it('a failure reaches the job as its error and starts no run', async () => {
    const { e, m } = await engineWith(fakeRunner(async () => { throw new Error('The model could not start (webgpu): out of memory'); }));
    const ref = await e.startRun(req(m.id));
    await until(async () => (await e.job(ref.job.id)).state !== 'running');
    expect(await e.job(ref.job.id)).toMatchObject({ state: 'failed', error: 'The model could not start (webgpu): out of memory' });
    expect(await e.runs(m.id)).toEqual([]);
  });

  it('refuses before any work: models missing (409), a saved target (501), a tap with no tap made (422), a bad class (Error), an unopened video (409), an unknown job (404)', async () => {
    const runner = fakeRunner(async () => ({}));
    const { e, m } = await engineWith(runner, fakeModels({ detectorReady: false }));
    await expect(e.startRun(req(m.id))).rejects.toMatchObject({ status: 409, extra: { error: 'models' } });
    const ok = await engineWith(runner);
    await expect(ok.e.startRun(req(ok.m.id, { target: { kind: 'saved', name: 'x' } }))).rejects.toMatchObject({ status: 501 });
    await expect(ok.e.startRun(req(ok.m.id, { target: { kind: 'tap' } }))).rejects.toMatchObject({ status: 422, message: /Tap the person/ }); // no tap yet
    await expect(ok.e.startRun(req(ok.m.id, { target: { kind: 'class', class: 'unicorn' } }))).rejects.toThrow(/unknown class/);
    await expect(ok.e.startRun(req('nope0000'))).rejects.toMatchObject({ status: 409 });
    await expect(ok.e.job('x')).rejects.toMatchObject({ status: 404 });
    expect(runner.sent).toEqual([]);
  });

  it('a class run is slugged class-dog and sends the class to the Worker', async () => {
    const runner = fakeRunner(async () => ({ scored: scoredOf([1, 2]), provider: 'webgpu', warnings: [], cls: 18, targetName: 'class:dog' }));
    const { e, m } = await engineWith(runner);
    const ref = await e.startRun(req(m.id, { target: { kind: 'class', class: 'dog' }, zone: [0.1, 0.1, 0.5, 0.5], zone_overlap: 0.5 }));
    expect(ref.run_id).toBe(`${m.id}-class-dog`);
    await until(async () => (await e.currentJob()) === null);
    expect(runner.sent[0]).toMatchObject({ target: { kind: 'class', class: 'dog' }, zone: [0.1, 0.1, 0.5, 0.5], zoneOverlap: 0.5 });
    expect((await e.runs(m.id))[0]).toMatchObject({ slug: 'class-dog', target: 'class:dog' });
  });

  it('boxes asks the Worker and keeps the people shown from score 0.4 up', async () => {
    const runner = fakeRunner(async () => ({ t: 3.0333333, detections: [{ score: 0.93456, box: [0.1234567, 0.2, 0.5, 0.9] }, { score: 0.41, box: [0.6, 0.2, 0.7, 0.5] }, { score: 0.3, box: [0, 0, 0.1, 0.1] }] }));
    const { e, m } = await engineWith(runner);
    const r = await e.boxes(m.id, 3, 'person');
    expect(r).toEqual({ t: 3.033, boxes: [{ i: 1, box: [0.1235, 0.2, 0.5, 0.9], score: 0.935 }, { i: 2, box: [0.6, 0.2, 0.7, 0.5], score: 0.41 }] });
    expect(runner.sent[0]).toMatchObject({ op: 'boxes', t: 3, cls: 'person' });
  });

  it('the session lists the 80 classes and the checklist facts: WebGPU, decoding and whether the models are in', async () => {
    const { e } = await engineWith(fakeRunner(async () => ({})));
    const s = await e.session();
    expect(s.classes).toHaveLength(80);
    expect(s.classes).toContain('dog');
    expect(s.preflight.browser).toMatchObject({ webgpu: true, webcodecs: true, hevc: true, models: { detectorReady: true } });
    expect(s.preflight.models).toEqual({ fast: true, balanced: true });
    const none = await engineWith(fakeRunner(async () => ({})), fakeModels({ detectorReady: false, allReady: false, cachedBytes: 0 }));
    expect((await none.e.session()).preflight.models).toEqual({ fast: false, balanced: false });
  });
});

describe('fix: the run slot is taken before the models are checked', () => {
  it('two startRun calls at once start one job; the second is 409 busy', async () => {
    const runner = fakeRunner(() => new Promise(() => {}));
    const { e, m } = await engineWith(runner, fakeModels());
    const res = await Promise.allSettled([e.startRun(req(m.id)), e.startRun(req(m.id))]);
    expect(res.map((r) => r.status).sort()).toEqual(['fulfilled', 'rejected']);
    const bad = res.find((r) => r.status === 'rejected') as PromiseRejectedResult;
    expect(bad.reason).toMatchObject({ status: 409, extra: { error: 'busy' } });
    await new Promise((r) => setTimeout(r, 20));
    expect(runner.sent).toHaveLength(1);
  });

  it('a run refused for missing models leaves no busy job behind', async () => {
    const runner = fakeRunner(async () => ({}));
    const { e, m } = await engineWith(runner, fakeModels({ detectorReady: false }));
    await expect(e.startRun(req(m.id))).rejects.toMatchObject({ status: 409, extra: { error: 'models' } });
    expect(await e.currentJob()).toBeNull();
  });
});
