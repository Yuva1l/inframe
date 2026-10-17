// @vitest-environment node
// BrowserEngine.startExport on a real (synthetic) file: the job, its progress, the result document, the blob sink, cancel and the refusals.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so these imports are allowed to be untyped
// @ts-ignore
import { execFileSync, spawnSync } from 'node:child_process';
// @ts-ignore
import fs from 'node:fs';
// @ts-ignore
import os from 'node:os';
// @ts-ignore
import path from 'node:path';
import { BrowserStore, MemoryBackend } from '../src/browser/store';
import { BrowserEngine } from '../src/engine/browser';
import type { ExportRequest, Job } from '../src/engine/types';
import type { SinkEnv } from '../src/browser/exportzip';

const sh = (cmd: string, args: string[]): string => execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) as string;
const ok = (cmd: string, args: string[]): boolean => {
  try {
    sh(cmd, args);
    return true;
  } catch {
    return false;
  }
};
const haveFfmpeg = ok('ffmpeg', ['-version']);
const haveUnzip = ok('unzip', ['-v']);

let dir = '';
let file: File;
beforeAll(async () => {
  if (!haveFfmpeg) return;
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'inframe-bexp-'));
  const p = path.join(dir, 'session.mov');
  sh('ffmpeg', ['-nostats', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=30:duration=12', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=12', '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '30', '-bf', '2', '-c:a', 'aac', '-shortest', p]);
  const buf = fs.readFileSync(p);
  file = new File([buf], 'session.mov', { type: 'video/quicktime', lastModified: 1_700_000_000_000 });
});
afterAll(() => {
  if (dir) fs.rmSync(dir, { recursive: true, force: true });
});

const REQ: ExportRequest = { clip_ids: null, out_dir: null, replace: false, clips: true, reel: false, music: null, beat_sync: false, edit_lists: [] };

async function setup(sink: SinkEnv) {
  const store = new BrowserStore(new MemoryBackend());
  const e = new BrowserEngine({ store, sink: () => sink, fileUrl: () => 'blob:video', revokeFileUrl: () => undefined });
  const info = await e.openFile(file);
  const t = Array.from({ length: 60 }, (_, i) => i * 0.2);
  await store.putRun({
    run_id: `${info.id}-any-person`, media_id: info.id, slug: 'any-person', target: 'any person', preset: 'balanced',
    segments: {
      settings: { mode: 'detect', fps: 5, enter: 0.5, exit: 0.3, target: 'any-person', detector: 'onnxruntime-web webgpu', model: 'small' }, warnings: [],
      clips: [{ index: 1, start: 2.4, end: 5.1, expected_copy_start: 2 }, { index: 2, start: 8.2, end: 11.9, expected_copy_start: 8 }],
    },
    samples: { t, score: t.map(() => 0.9), box: t.map(() => [0.1, 0.1, 0.4, 0.9]), track: null, enter: 0.5, exit: 0.3, merge_gap: 1, min_len: 0.5, pad: 1, mode: 'detect' },
    source: { offset: 0, end_time: 12 },
  });
  return { e, info, runId: `${info.id}-any-person`, store };
}
const finish = async (e: BrowserEngine, id: string): Promise<Job> => {
  let j = await e.job(id);
  while (j.state === 'running') {
    await new Promise((r) => setTimeout(r, 10));
    j = await e.job(id);
  }
  return j;
};
const blobSink = () => {
  const saved: { name: string; blob: Blob }[] = [];
  const env: SinkEnv = { download: (blob, name) => { saved.push({ name, blob }); return `blob:zip-${saved.length}`; } };
  return { env, saved };
};

describe.skipIf(!haveFfmpeg)('BrowserEngine export', () => {
  it('says it can cut by stream copy', () => {
    expect(new BrowserEngine().caps.streamCopyCut).toBe(true);
  });

  it.skipIf(!haveUnzip)('writes one zip of clips, sidecars and the CSV; the job reports it and the result stays', async () => {
    const { env, saved } = blobSink();
    const { e, runId } = await setup(env);
    const ref = await e.startExport(runId, REQ);
    expect(ref.job.kind).toBe('export');
    const j = await finish(e, ref.job.id);
    expect(j.state, j.error ?? '').toBe('done');
    expect(j.fraction).toBe(1);
    const res = j.result as { outputs: { name: string; kind: string; size: number; url: string }[]; clips: { file: string; sidecar?: string }[]; n_clips: number; warnings: string[] };
    expect(res.n_clips).toBe(2);
    expect(res.outputs.map((o) => o.kind)).toEqual(['zip', 'clip', 'sidecar', 'clip', 'sidecar', 'csv']);
    expect(res.outputs[0]).toMatchObject({ name: 'session_clips.zip', url: 'blob:zip-1' });
    expect(res.outputs[0].size).toBe(saved[0].blob.size);
    expect(saved[0].name).toBe('session_clips.zip');
    expect(await e.exportResult(runId)).toEqual(res);
    expect(e.runFileUrl(runId, 'session_clips.zip')).toBe('blob:zip-1');
    expect(e.runFileUrl(runId, 'session_01_0002-0005.mov')).toBe('');
    const zip = path.join(dir, 'out.zip');
    fs.writeFileSync(zip, Buffer.from(await saved[0].blob.arrayBuffer()));
    const t = spawnSync('unzip', ['-t', zip], { encoding: 'utf8' });
    expect(t.status, t.stdout).toBe(0);
    // the sizes the result lists are the sizes in the zip
    const listing = sh('unzip', ['-l', zip]);
    for (const o of res.outputs.slice(1)) {
      const m = new RegExp(`\\s(\\d+)\\s+\\S+\\s+\\S+\\s+${o.name.replace(/\./g, '\\.')}\\n`).exec(listing);
      expect(m, o.name).not.toBeNull();
      expect(Number(m![1]), o.name).toBe(o.size);
    }
  });

  it('uses the edited clips, and only the chosen ones', async () => {
    const { env, saved } = blobSink();
    const { e, runId, store } = await setup(env);
    await e.saveEdits(runId, 0, [{ id: 'd1', start: 2.4, end: 5.1, origin: 'detector' }, { id: 'u1', start: 6.1, end: 7.5, origin: 'added' }]);
    const j = await finish(e, (await e.startExport(runId, { ...REQ, clip_ids: ['u1'] })).job.id);
    expect(j.state, j.error ?? '').toBe('done');
    const res = j.result as { clips: { file: string }[] };
    expect(res.clips).toHaveLength(1);
    expect(res.clips[0].file).toBe('session_01_0006-0007.mov');
    expect(saved).toHaveLength(1);
    void store;
  });

  it('refuses what browser mode cannot do, and a second job while one runs', async () => {
    const { env } = blobSink();
    const { e, runId } = await setup(env);
    await expect(e.startExport(runId, { ...REQ, reel: true })).rejects.toMatchObject({ status: 501 });
    await expect(e.startExport(runId, { ...REQ, edit_lists: ['edl'] })).rejects.toMatchObject({ status: 501 });
    await expect(e.startExport(runId, { ...REQ, clips: false })).rejects.toMatchObject({ status: 422 });
    await expect(e.startExport(runId, { ...REQ, clip_ids: ['nope'] })).rejects.toMatchObject({ status: 422 });
    await expect(e.startExport('missing', REQ)).rejects.toMatchObject({ status: 404 });
    const ref = await e.startExport(runId, REQ);
    await expect(e.startExport(runId, REQ)).rejects.toMatchObject({ status: 409 });
    await finish(e, ref.job.id);
  });

  it('above the Safari cap it says use My Mac before anything is read', async () => {
    const { e, runId } = await setup({ cap: 1000, download: () => 'blob:x' });
    await expect(e.startExport(runId, REQ)).rejects.toMatchObject({ status: 413, message: expect.stringContaining('Use My Mac') });
    expect(await e.currentJob()).toBeNull();
  });

  it('a user who closes the save dialog cancels without a job', async () => {
    const { e, runId } = await setup({ showSaveFilePicker: async () => { throw new DOMException('x', 'AbortError'); } });
    await expect(e.startExport(runId, REQ)).rejects.toMatchObject({ status: 499 });
    expect(await e.currentJob()).toBeNull();
  });

  it('opens the save picker once the request is checked, before the file is read, while the click still counts', async () => {
    let asked: unknown = null;
    let readsWhenAsked = -1;
    const env: SinkEnv = { showSaveFilePicker: async (o) => { asked = o; readsWhenAsked = reads; throw new DOMException('x', 'AbortError'); } };
    const { e, runId, info } = await setup(env);
    const src = (e as unknown as { opened: Map<string, { video: { src: { read: (a: number, b: number) => Promise<ArrayBuffer> } } }> }).opened.get(info.id)!.video.src;
    const real = src.read.bind(src);
    let reads = 0;
    src.read = async (a, b) => {
      reads++;
      return real(a, b);
    };
    await expect(e.startExport(runId, REQ)).rejects.toMatchObject({ status: 499 });
    expect(asked).toMatchObject({ suggestedName: 'session_clips.zip' });
    expect(readsWhenAsked).toBe(0);
  });

  it('a request that fails its check never opens the save picker', async () => {
    let asked = 0;
    const env: SinkEnv = { showSaveFilePicker: async () => { asked++; throw new DOMException('x', 'AbortError'); } };
    const { e, runId } = await setup(env);
    await expect(e.startExport(runId, { ...REQ, clip_ids: ['nope'] })).rejects.toMatchObject({ status: 422 });
    await expect(e.startExport('missing', REQ)).rejects.toMatchObject({ status: 404 });
    await expect(e.startExport(runId, { ...REQ, reel: true })).rejects.toMatchObject({ status: 501 });
    await expect(e.startExport(runId, { ...REQ, clips: false })).rejects.toMatchObject({ status: 422 });
    await e.saveEdits(runId, 0, []);
    await expect(e.startExport(runId, REQ)).rejects.toMatchObject({ status: 422, message: expect.stringContaining('empty') });
    expect(asked).toBe(0);
  });

  it('a codec it cannot cut is refused before the save picker opens', async () => {
    let asked = 0;
    const { e, runId, info } = await setup({ showSaveFilePicker: async () => { asked++; throw new DOMException('x', 'AbortError'); } });
    const o = (e as unknown as { opened: Map<string, { video: { demuxed: { video: { fourcc: string } } } }> }).opened.get(info.id)!;
    o.video.demuxed.video.fourcc = 'av01';
    const p = e.startExport(runId, REQ);
    // refused at once: no job slot taken and nothing awaited yet (the store and the picker are untouched)
    expect((e as unknown as { jobs: Map<string, unknown> }).jobs.size).toBe(0);
    await expect(p).rejects.toMatchObject({ status: 415 });
    expect((e as unknown as { jobs: Map<string, unknown> }).jobs.size).toBe(0);
    expect(asked).toBe(0);
    expect(await e.currentJob()).toBeNull();
  });

  it('two quick clicks: the second meets busy at once, before the first picker is answered', async () => {
    let asked = 0;
    let answer: (e: unknown) => void = () => undefined;
    const { e, runId } = await setup({ showSaveFilePicker: () => { asked++; return new Promise((_, rej) => (answer = rej)); } });
    const first = e.startExport(runId, REQ);
    const second = e.startExport(runId, REQ);
    await expect(second).rejects.toMatchObject({ status: 409 });
    expect(asked).toBe(1);
    answer(new DOMException('x', 'AbortError'));
    await expect(first).rejects.toMatchObject({ status: 499 });
    expect(await e.currentJob()).toBeNull(); // the slot is free again
  });

  it('clips that cannot be cut leave no empty zip behind: the file the picker made is removed', async () => {
    let removed = 0;
    const env: SinkEnv = { showSaveFilePicker: async () => ({ remove: async () => void removed++, createWritable: async () => new WritableStream<Uint8Array>(), getFile: async () => ({ size: 0 }) }) };
    const { e, runId, info, store } = await setup(env);
    const run = (await store.getRun(runId))!;
    const late = `${info.id}-late`; // a run whose only clip lies past the end of the 12 s video
    await store.putRun({ ...run, run_id: late, slug: 'late', segments: { ...run.segments, clips: [{ index: 1, start: 30, end: 40, expected_copy_start: 30 }] } });
    await expect(e.startExport(late, REQ)).rejects.toMatchObject({ status: 422 });
    await new Promise((r) => setTimeout(r, 20));
    expect(removed).toBe(1);
    expect(await e.currentJob()).toBeNull();
  });

  it('a failed read of the file is not kept: the next export reads again', async () => {
    const { env, saved } = blobSink();
    const { e, runId, info } = await setup(env);
    const src = (e as unknown as { opened: Map<string, { video: { src: { read: (a: number, b: number) => Promise<ArrayBuffer> } } }> }).opened.get(info.id)!.video.src;
    const real = src.read.bind(src);
    let fail = true;
    src.read = async (a, b) => {
      if (fail) throw new Error('read failed once');
      return real(a, b);
    };
    await expect(e.startExport(runId, REQ)).rejects.toThrow('read failed once');
    fail = false;
    const j = await finish(e, (await e.startExport(runId, REQ)).job.id);
    expect(j.state, j.error ?? '').toBe('done');
    expect(saved).toHaveLength(1);
  });

  it('cancel stops the export and leaves no result', async () => {
    const chunks: number[] = [];
    let release: () => void = () => undefined;
    const gate = new Promise<void>((r) => (release = r));
    const env: SinkEnv = {
      showSaveFilePicker: async () => ({
        createWritable: async () => new WritableStream<Uint8Array>({ write: async (c) => { chunks.push(c.length); await gate; } }),
        getFile: async () => ({ size: 0 }),
      }),
    };
    const { e, runId } = await setup(env);
    const ref = await e.startExport(runId, REQ);
    await new Promise((r) => setTimeout(r, 50));
    await e.cancel(ref.job.id);
    release();
    const j = await finish(e, ref.job.id);
    expect(j.state).toBe('cancelled');
    expect(await e.exportResult(runId)).toBeNull();
  });

  const handleWith = (log: { removed: number; written: number }) => ({
    remove: async () => void log.removed++,
    createWritable: async () => new WritableStream<Uint8Array>({ write: (c) => void (log.written += c.length) }),
    getFile: async () => ({ size: log.written }),
  });

  it('a cancel while the save picker is open ends cancelled, removes the file the user then picks, and writes nothing', async () => {
    const log = { removed: 0, written: 0 };
    let pick: (h: unknown) => void = () => undefined;
    const { e, runId } = await setup({ showSaveFilePicker: () => new Promise((res) => (pick = res)) });
    const first = e.startExport(runId, REQ);
    const failed = first.then(() => null, (x) => x);
    await new Promise((r) => setTimeout(r, 30)); // the picker is open
    const cur = (await e.currentJob())!;
    await e.cancel(cur.id);
    pick(handleWith(log));
    expect(await failed).toMatchObject({ status: 499 });
    expect(log).toEqual({ removed: 1, written: 0 });
    expect((await e.job(cur.id)).state).toBe('cancelled');
    expect(await e.currentJob()).toBeNull();
  });

  it('a cancel while the sample tables are read ends cancelled and removes the picked file', async () => {
    const log = { removed: 0, written: 0 };
    const { e, runId, info } = await setup({ showSaveFilePicker: async () => handleWith(log) });
    const src = (e as unknown as { opened: Map<string, { video: { src: { read: (a: number, b: number) => Promise<ArrayBuffer> } } }> }).opened.get(info.id)!.video.src;
    const real = src.read.bind(src);
    let release: () => void = () => undefined;
    const gate = new Promise<void>((r) => (release = r));
    src.read = async (a, b) => {
      await gate;
      return real(a, b);
    };
    const failed = e.startExport(runId, REQ).then(() => null, (x) => x);
    await new Promise((r) => setTimeout(r, 30));
    const cur = (await e.currentJob())!;
    await e.cancel(cur.id);
    release();
    expect(await failed).toMatchObject({ status: 499 });
    expect(log).toEqual({ removed: 1, written: 0 });
    expect((await e.job(cur.id)).state).toBe('cancelled');
  });

  it('a second click that met busy follows a job that then ends cleanly, not a 404', async () => {
    let answer: (e: unknown) => void = () => undefined;
    const { e, runId } = await setup({ showSaveFilePicker: () => new Promise((_, rej) => (answer = rej)) });
    const first = e.startExport(runId, REQ);
    await new Promise((r) => setTimeout(r, 30)); // the picker is open
    const second = await e.startExport(runId, REQ).then(() => null, (x) => x);
    expect(second).toMatchObject({ status: 409 });
    const id = (second.extra as { job: Job }).job.id;
    answer(new DOMException('x', 'AbortError'));
    await expect(first).rejects.toMatchObject({ status: 499 });
    expect((await e.job(id)).state).toBe('cancelled');
    expect(await e.currentJob()).toBeNull();
  });

  it('a follower of a job whose clips cannot be cut sees it failed with the refusal, not a 404', async () => {
    const { e, runId, info, store } = await setup({ showSaveFilePicker: async () => handleWith({ removed: 0, written: 0 }) });
    const run = (await store.getRun(runId))!;
    const late = `${info.id}-late`;
    await store.putRun({ ...run, run_id: late, slug: 'late', segments: { ...run.segments, clips: [{ index: 1, start: 30, end: 40, expected_copy_start: 30 }] } });
    const first = e.startExport(late, REQ).then(() => null, (x) => x);
    const second = await e.startExport(late, REQ).then(() => null, (x) => x);
    expect(second).toMatchObject({ status: 409 });
    const id = (second.extra as { job: Job }).job.id;
    expect(await first).toMatchObject({ status: 422 });
    const j = await e.job(id);
    expect(j.state).toBe('failed');
    expect(j.error).toBeTruthy();
    expect(await e.currentJob()).toBeNull();
  });

  it('a cancel while writing removes the file the picker made', async () => {
    const log = { removed: 0, written: 0 };
    let release: () => void = () => undefined;
    const gate = new Promise<void>((r) => (release = r));
    const h = { ...handleWith(log), createWritable: async () => new WritableStream<Uint8Array>({ write: async () => { await gate; } }) };
    const { e, runId } = await setup({ showSaveFilePicker: async () => h });
    const ref = await e.startExport(runId, REQ);
    await new Promise((r) => setTimeout(r, 50));
    await e.cancel(ref.job.id);
    release();
    expect((await finish(e, ref.job.id)).state).toBe('cancelled');
    expect(log.removed).toBe(1);
  });

  it('a failure after the picked file is written removes it, and the job fails', async () => {
    const log = { removed: 0, written: 0 };
    const h = { ...handleWith(log), getFile: async () => { throw new Error('disk full'); } };
    const { e, runId } = await setup({ showSaveFilePicker: async () => h });
    const j = await finish(e, (await e.startExport(runId, REQ)).job.id);
    expect(j.state).toBe('failed');
    expect(log.removed).toBe(1);
  });

  it('a file or folder sink is used when the browser has one', async () => {
    const written: number[] = [];
    const env: SinkEnv = {
      showSaveFilePicker: async () => ({ createWritable: async () => new WritableStream<Uint8Array>({ write: (c) => void written.push(c.length) }), getFile: async () => ({ size: written.reduce((a, b) => a + b, 0) }) }),
    };
    const { e, runId } = await setup(env);
    const j = await finish(e, (await e.startExport(runId, REQ)).job.id);
    expect(j.state, j.error ?? '').toBe('done');
    const res = j.result as { outputs: { kind: string; size: number; url: string }[] };
    expect(res.outputs[0].url).toBe(''); // written to a file: nothing to offer again
    expect(res.outputs[0].size).toBe(written.reduce((a, b) => a + b, 0));
  });
});
