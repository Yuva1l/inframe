// window.__t for tests/browser (Playwright for Chrome; Safari opens the page with #token=..&auto=parity and the page POSTs its results back).
import { createSession } from '../src/browser/detect';
import { cropTensor, Dino } from '../src/browser/embed';
import { DETECTOR_FILE, EMBEDDER_FILE, ModelStore, WASM_FILE } from '../src/browser/models';
import { BrowserStore, openBackend } from '../src/browser/store';
import { BrowserEngine } from '../src/engine/browser';

const hash = new URLSearchParams(location.hash.slice(1));
const token = hash.get('token') ?? '';
const logs: string[] = [];
for (const k of ['log', 'info', 'warn', 'error', 'debug'] as const) {
  const o = console[k].bind(console);
  console[k] = (...a: unknown[]) => {
    if (logs.length < 4000) logs.push(a.map(String).join(' ').slice(0, 400));
    o(...a);
  };
}
const NOTE = /placed on|not assigned|VerifyEachNode|Rerun|fallback|CPUExecutionProvider|WebGPU|webgpu/i;

const models = new ModelStore();
let engine: BrowserEngine | null = null;
const eng = () => (engine ??= new BrowserEngine());
let exportEngine: BrowserEngine | null = null;
let exportStore: BrowserStore | null = null;

function lcg(n: number): Float32Array {
  const a = new Float32Array(n);
  let s = 12345;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    a[i] = (s / 4294967296) * 2 - 1;
  }
  return a;
}

const SHAPE: Record<string, number[]> = { rfdetr: [1, 3, 512, 512], dino: [1, 3, 224, 224] };
const FILE: Record<string, string> = { rfdetr: DETECTOR_FILE, dino: EMBEDDER_FILE };

async function download() {
  const t0 = performance.now();
  await models.download();
  const st = await models.status();
  return { ms: performance.now() - t0, cachedBytes: st.cachedBytes, allReady: st.allReady, persisted: st.persisted, files: st.files.map((f) => [f.name, f.bytes, f.cached]) };
}

async function parity(model: 'rfdetr' | 'dino', input: 'lcg' | 'frame') {
  const mark = logs.length;
  const dims = SHAPE[model];
  const n = dims.reduce((a, b) => a * b, 1);
  let x: Float32Array;
  if (input === 'lcg') x = lcg(n);
  else {
    const r = await fetch(`/api/selftest/data/${model}_frame.f32`, { headers: { 'X-Inframe-Token': token } });
    if (!r.ok) throw new Error(`frame data answered ${r.status}`);
    x = new Float32Array(await r.arrayBuffer());
    if (x.length !== n) throw new Error(`frame data has ${x.length} values, wanted ${n}`);
  }
  const t0 = performance.now();
  const session = await createSession(await models.bytes(FILE[model]), await models.bytes(WASM_FILE), { providers: ['webgpu'], logSeverity: 0 });
  const createMs = performance.now() - t0;
  const t1 = performance.now();
  const out = await session.run(x, dims);
  const runMs = performance.now() - t1;
  const outputs: Record<string, { values: number[]; n: number; dims: readonly number[] }> = {};
  for (const [k, v] of Object.entries(out)) {
    outputs[k] = { values: Array.from(v.data as Float32Array), n: v.data.length, dims: v.dims }; // every value: tests/browser/parity_compare.py compares them one by one
  }
  await session.release();
  const notes = logs.slice(mark).filter((l) => NOTE.test(l));
  return {
    model, input, provider: session.provider, outputs, createMs, runMs, hasGpu: 'gpu' in navigator, ua: navigator.userAgent,
    cpuPlaced: logs.slice(mark).filter((l) => /placed on \[?CPUExecutionProvider/.test(l)).length,
    notAssigned: logs.slice(mark).filter((l) => /not assigned to the preferred/.test(l)).length,
    notes: notes.slice(0, 12),
    cpuLines: logs.slice(mark).filter((l) => /placed on|not assigned to the preferred/.test(l)).map((l) => l.slice(0, 300)).slice(0, 6),
    kernelMissing: logs.slice(mark).filter((l) => /webgpu kernel not found/.test(l)).map((l) => l.replace(/^.*Op type: /, '').slice(0, 80)),
  };
}

/** DINOv2 embeddings of boxes of one picture (`frame.rgb`: raw RGB, w x h, written by the test), through the page's own crop, pad, resize and session code. */
async function embedCrops(w: number, h: number, boxes: [number, number, number, number][]) {
  const r = await fetch('/api/selftest/data/frame.rgb', { headers: { 'X-Inframe-Token': token } });
  if (!r.ok) throw new Error(`frame data answered ${r.status}`);
  const rgb = new Uint8Array(await r.arrayBuffer());
  if (rgb.length !== w * h * 3) throw new Error(`frame data has ${rgb.length} bytes, wanted ${w * h * 3}`);
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) rgba.set([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255], i * 4);
  const session = await createSession(await models.bytes(FILE.dino), await models.bytes(WASM_FILE), { providers: ['webgpu'] });
  const dino = new Dino(session);
  const out: (number[] | null)[] = [];
  const t0 = performance.now();
  for (const b of boxes) {
    const t = cropTensor(rgba, w, h, b);
    out.push(t ? Array.from(await dino.embed(t)) : null);
  }
  const ms = performance.now() - t0;
  await dino.release();
  return { embeddings: out, provider: session.provider, ms };
}

async function run(opts: { cancelAtFraction?: number; preset?: string; advanced?: Record<string, unknown>; kind?: string; cls?: string; tap?: { t: number; box: [number, number, number, number] } } = {}) {
  const e = eng();
  const f = (document.querySelector('#file') as HTMLInputElement).files?.[0];
  if (!f) throw new Error('no file chosen');
  const m = await e.openFile(f);
  const t0 = performance.now();
  if (opts.tap) {
    await e.resetPick(m.id, opts.cls ?? 'person'); // taps of an earlier test in this browser are kept in IndexedDB
    await e.tap(m.id, opts.cls ?? 'person', opts.tap.t, opts.tap.box, 1);
  }
  const ref = await e.startRun({
    media_id: m.id, target: { kind: (opts.kind ?? 'any') as 'any', class: opts.cls ?? 'person' }, zone: null, zone_overlap: null, preset: opts.preset ?? 'balanced',
    slowmo: 'off', slowmo_audio: false, advanced: opts.advanced ?? {},
  });
  const seen: number[] = [];
  let cancelledAt: number | null = null;
  let j = ref.job;
  while (j.state === 'running') {
    await new Promise((r) => setTimeout(r, 40));
    j = await e.job(ref.job.id);
    seen.push(j.fraction);
    if (opts.cancelAtFraction !== undefined && cancelledAt === null && j.fraction >= opts.cancelAtFraction) {
      cancelledAt = j.fraction;
      await e.cancel(j.id);
    }
  }
  const out: Record<string, unknown> = { state: j.state, error: j.error, fractions: seen.length, lastMessage: j.message, cancelledAt, ms: performance.now() - t0, duration: m.duration, codec: m.codec, rotation: m.rotation };
  const runs = await e.runs(m.id);
  out.runs = runs.length;
  if (j.state === 'done') {
    const s = await e.samples(ref.run_id!);
    const r = await e.run(ref.run_id!);
    Object.assign(out, { nSamples: s.t.length, nClips: r.segments.clips.length, maxScore: s.score.reduce((a, b) => Math.max(a, b), 0), warnings: r.segments.warnings, settings: r.segments.settings, timing: r.segments.timing_s, segments: r.segments, firstT: s.t[0], lastT: s.t[s.t.length - 1] });
  }
  const after = await e.currentJob();
  out.stillRunning = after !== null;
  return out;
}


/** The export of part 5 on a real file in the real Chrome: a run is stored by hand (the detector is not the point here), the zip goes through the
 *  File System Access API (the page's private folder, OPFS: a real FileSystemDirectoryHandle with a streaming writable), and the bytes come back as base64. */
async function exportZip(opts: { clips: [number, number][]; sink?: 'folder' | 'download' }) {
  const f = (document.querySelector('#file') as HTMLInputElement).files?.[0];
  if (!f) throw new Error('no file chosen');
  if (!exportEngine) {
    const b = await openBackend();
    exportStore = new BrowserStore(b.backend, b.persistent);
    const root = await navigator.storage.getDirectory();
    exportEngine = new BrowserEngine({
      store: exportStore,
      sink: () => (opts.sink === 'download' ? { download: (bl, name) => { (window as unknown as { __zipBlob: Blob }).__zipBlob = bl; return URL.createObjectURL(bl) + '#' + name; } } : { showDirectoryPicker: async () => root }),
    });
  }
  const e = exportEngine;
  const m = await e.openFile(f);
  const t = Array.from({ length: Math.floor(m.duration * 5) }, (_, i) => Math.round(i * 0.2 * 1000) / 1000);
  const inside = (x: number) => opts.clips.some(([a, c]) => x >= a && x <= c);
  const runId = `${m.id}-any-person`;
  await exportStore!.putRun({
    run_id: runId, media_id: m.id, slug: 'any-person', target: 'any person', preset: 'balanced',
    segments: {
      settings: { mode: 'detect', fps: 5, enter: 0.5, exit: 0.3, target: 'any-person', detector: 'onnxruntime-web webgpu', model: 'small' }, warnings: [],
      clips: opts.clips.map(([a, c], i) => ({ index: i + 1, start: a, end: c })),
    },
    samples: { t, score: t.map((x) => (inside(x) ? 0.93 : 0.01)), box: t.map((x) => (inside(x) ? [0.1, 0.1, 0.4, 0.9] : null)), track: null, enter: 0.5, exit: 0.3, merge_gap: 1, min_len: 0.5, pad: 1, mode: 'detect' },
    source: { offset: m.offset, end_time: m.end_time },
  });
  const t0 = performance.now();
  const ref = await e.startExport(runId, { clip_ids: null, out_dir: null, replace: false, clips: true, reel: false, music: null, beat_sync: false, edit_lists: [] });
  const seen: number[] = [];
  let j = ref.job;
  while (j.state === 'running') {
    await new Promise((r) => setTimeout(r, 20));
    j = await e.job(ref.job.id);
    seen.push(j.fraction);
  }
  const out: Record<string, unknown> = { state: j.state, error: j.error, ms: performance.now() - t0, result: j.result, fractions: seen.length, monotonic: seen.every((x, i) => i === 0 || x >= seen[i - 1]), media: m };
  if (j.state === 'done') {
    const res = j.result as { outputs: { name: string; kind: string }[] };
    const zipName = res.outputs[0].name;
    let zip: Blob;
    if (opts.sink === 'download') zip = (window as unknown as { __zipBlob: Blob }).__zipBlob;
    else {
      const root = await navigator.storage.getDirectory();
      zip = await (await root.getFileHandle(zipName)).getFile();
      out.opfsSize = zip.size;
    }
    const buf = new Uint8Array(await zip.arrayBuffer());
    let bin = '';
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    out.zipBase64 = btoa(bin);
    out.zipName = zipName;
  }
  return out;
}

async function boxes(t: number) {
  const e = eng();
  const f = (document.querySelector('#file') as HTMLInputElement).files![0];
  const m = await e.openFile(f);
  return e.boxes(m.id, t, 'person');
}

const T = { token, download, parity, run, boxes, embedCrops, exportZip, session: () => eng().session(), logs: () => logs.slice(), ready: true };
(window as unknown as { __t: typeof T }).__t = T;

// Safari: the page does the work by itself and posts the answer back (no safaridriver)
async function auto() {
  const post = (body: unknown) => fetch('/api/selftest/result', { method: 'POST', headers: { 'X-Inframe-Token': token, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  try {
    const res: unknown[] = [];
    const dl = await download();
    for (const model of ['rfdetr', 'dino'] as const) for (const input of ['lcg', 'frame'] as const) res.push(await parity(model, input));
    await post({ ok: true, download: dl, results: res });
  } catch (e) {
    await post({ ok: false, error: String((e as Error)?.stack ?? e), logs: logs.slice(-30) });
  }
}
if (hash.get('auto') === 'parity') void auto();
