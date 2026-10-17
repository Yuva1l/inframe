// @vitest-environment node
// The export zip (src/browser/exportzip.ts) end to end in Node on synthetic media: `unzip -t`, the sidecars through tools/validate_sidecars.py,
// the LosslessCut CSV against the Python writer, the Python number rules, the streaming SHA-256 against node's crypto, and the sinks.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so these imports are allowed to be untyped
// @ts-ignore
import { createHash } from 'node:crypto';
// @ts-ignore
import { execFileSync, spawnSync } from 'node:child_process';
// @ts-ignore
import fs from 'node:fs';
// @ts-ignore
import os from 'node:os';
// @ts-ignore
import path from 'node:path';
import { demux, type ByteSource } from '../src/browser/demux';
import { chooseSink, clipName, csvName, exportStream, fmt3, losslessCutCsv, mmss, planExport, pyRound, sidecarName, type ExportInput, type SinkEnv } from '../src/browser/exportzip';
import { readMuxSource } from '../src/browser/mux';
import { Sha256, sha256Of } from '../src/browser/sha256';
import type { MediaInfo, Samples } from '../src/engine/types';

const sh = (cmd: string, args: string[], cwd?: string): string => execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], cwd }) as string;
const ok = (cmd: string, args: string[]): boolean => {
  try {
    sh(cmd, args);
    return true;
  } catch {
    return false;
  }
};
const haveFfmpeg = ok('ffmpeg', ['-version']);
const haveX265 = haveFfmpeg && sh('ffmpeg', ['-hide_banner', '-encoders']).includes('libx265');
const haveUv = ok('uv', ['--version']);
const haveUnzip = ok('unzip', ['-v']);
const REPO = path.resolve(process.cwd(), '..');
const py = (code: string, ...args: string[]): string => sh('uv', ['run', '--project', REPO, 'python', '-c', code, ...args], REPO);

function nodeSource(file: string): ByteSource {
  const fd = fs.openSync(file, 'r');
  return { size: fs.fstatSync(fd).size, read: async (a, b) => { const u = new Uint8Array(b - a); fs.readSync(fd, u, 0, u.length, a); return u.buffer; } };
}

describe('Python number rules', () => {
  it('pyRound and fmt3 equal Python on ties, halves and ordinary values', () => {
    const xs = [0.0625, 0.1875, 0.3125, 2.0625, 0.0005, 1.0005, 2.5, 0.123456, 18.0004, 7.9996, 1e-7, 123.4565, 0.5, 1.5, 0.00049999, 3.0];
    const want = JSON.parse(py('import json,sys; xs=json.loads(sys.argv[1]); print(json.dumps([[round(x,3), round(x,4), round(x,5), f"{x:.3f}"] for x in xs]))', JSON.stringify(xs)));
    xs.forEach((x, i) => {
      expect(pyRound(x, 3), `round(${x}, 3)`).toBe(want[i][0]);
      expect(pyRound(x, 4), `round(${x}, 4)`).toBe(want[i][1]);
      expect(pyRound(x, 5), `round(${x}, 5)`).toBe(want[i][2]);
      expect(fmt3(x), `${x}:.3f`).toBe(want[i][3]);
    });
  });
  it('clip names follow cut.clip_name', () => {
    expect(mmss(18.9)).toBe('0018');
    expect(mmss(3725.2)).toBe('6205');
    expect(clipName('IMG_0001', 1, 18.2, 36.9, '.mov')).toBe('IMG_0001_01_0018-0036.mov');
    expect(sidecarName('IMG_0001_01_0018-0036.mov')).toBe('IMG_0001_01_0018-0036.inframe.json');
    expect(csvName('IMG_0001')).toBe('IMG_0001_losslesscut.csv');
  });
  it('the CSV equals cut.write_losslesscut_csv for the same clips (also with a comma and a quote in a label)', () => {
    const rows: [number, number, string][] = [[2.4, 5.1, 'a_01_0002-0005.mov'], [8.2, 11.9, 'a_02_0008-0011.mov'], [0.0625, 1234.5678, 'we,ird "name".mov'], [59.9995, 3600, 'x.mov']];
    const want = py('import sys,json,tempfile,os; from inframe.cut import write_losslesscut_csv; rows=[tuple(r) for r in json.loads(sys.argv[1])]; p=os.path.join(tempfile.mkdtemp(),"x.csv"); write_losslesscut_csv(p, rows); sys.stdout.write(open(p,newline="").read().replace("\\r","<CR>").replace("\\n","<LF>"))', JSON.stringify(rows));
    expect(losslessCutCsv(rows).replace(/\r/g, '<CR>').replace(/\n/g, '<LF>')).toBe(want);
  });
});

describe('sha256', () => {
  it('equals node crypto for lengths around the block and padding edges and for a sliced read', async () => {
    for (const n of [0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 128, 1000, 100000]) {
      const b = new Uint8Array(n).map((_, i) => (i * 31 + 7) & 255);
      const want = createHash('sha256').update(b).digest('hex');
      expect(new Sha256().update(b).hex(), `n=${n}`).toBe(want);
      const h = new Sha256();
      for (let i = 0; i < n; i += 37) h.update(b.subarray(i, Math.min(n, i + 37)));
      expect(h.hex(), `pieces n=${n}`).toBe(want);
      expect(await sha256Of(n, async (s, e) => b.slice(s, e).buffer, { slice: 4096 })).toBe(want);
    }
  });
});

describe('sinks', () => {
  const stream = (n: number) => new ReadableStream<Uint8Array>({ start(c) { c.enqueue(new Uint8Array(n).fill(7)); c.close(); } });
  it('a save-file picker gets the bytes written to its handle', async () => {
    const got: Uint8Array[] = [];
    const handle = { createWritable: async () => new WritableStream<Uint8Array>({ write: (c) => void got.push(c) }), getFile: async () => ({ size: 10 }) };
    let asked: any;
    const env: SinkEnv = { showSaveFilePicker: async (o) => { asked = o; return handle; } };
    const sink = await chooseSink(env, 'a_clips.zip', 10);
    expect(sink.kind).toBe('file');
    expect(asked.suggestedName).toBe('a_clips.zip');
    expect((await sink.write(stream(10))).size).toBe(10);
    expect(got.length).toBe(1);
  });
  it('a user who closes the picker cancels the export', async () => {
    const env: SinkEnv = { showSaveFilePicker: async () => { throw new DOMException('x', 'AbortError'); } };
    await expect(chooseSink(env, 'a.zip', 1)).rejects.toMatchObject({ status: 499 });
  });
  it('a granted folder gets a file of the zip name', async () => {
    const files: Record<string, number> = {};
    const dir = { getFileHandle: async (name: string) => ({ createWritable: async () => new WritableStream<Uint8Array>({ write: (c) => void (files[name] = (files[name] ?? 0) + c.length) }), getFile: async () => ({ size: files[name] }) }) };
    const sink = await chooseSink({ showDirectoryPicker: async () => dir }, 'b_clips.zip', 5);
    expect(sink.kind).toBe('folder');
    await sink.write(stream(5));
    expect(files['b_clips.zip']).toBe(5);
  });
  it('without a picker the zip is a Blob download, capped (Safari): above the cap it says use My Mac', async () => {
    let saved: { name: string; size: number } | null = null;
    const env: SinkEnv = { cap: 100, download: (b, name) => { saved = { name, size: b.size }; return 'blob:x'; } };
    const sink = await chooseSink(env, 'c.zip', 50);
    expect(sink.kind).toBe('download');
    const r = await sink.write(stream(50));
    expect(r).toEqual({ size: 50, url: 'blob:x' });
    expect(saved).toEqual({ name: 'c.zip', size: 50 });
    await expect(chooseSink(env, 'c.zip', 101)).rejects.toMatchObject({ status: 413, message: expect.stringContaining('Use My Mac') });
    await expect((await chooseSink(env, 'c.zip', 50)).write(stream(150))).rejects.toMatchObject({ status: 413 });
  });
});

let dir = '';
let media = '';
beforeAll(() => {
  if (!haveFfmpeg) return;
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'inframe-zip-'));
  media = path.join(dir, haveX265 ? 'session.mp4' : 'session.mov');
  const v = 'testsrc2=size=640x360:rate=30:duration=12';
  const a = 'sine=frequency=440:duration=12';
  if (haveX265) {
    ff('-f', 'lavfi', '-i', v, '-f', 'lavfi', '-i', a, '-c:v', 'libx265', '-tag:v', 'hvc1', '-preset', 'ultrafast', '-g', '30', '-x265-params', 'open-gop=1:keyint=30:min-keyint=30:bframes=3:log-level=error', '-c:a', 'aac', '-shortest', path.join(dir, 'raw.mp4'));
    ff('-display_rotation', '90', '-i', path.join(dir, 'raw.mp4'), '-c', 'copy', '-tag:v', 'hvc1', media);
  } else ff('-f', 'lavfi', '-i', v, '-f', 'lavfi', '-i', a, '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '30', '-bf', '2', '-c:a', 'aac', '-shortest', media);
});
const ff = (...a: string[]) => sh('ffmpeg', ['-nostats', '-loglevel', 'error', '-y', ...a]);
afterAll(() => {
  if (dir) fs.rmSync(dir, { recursive: true, force: true });
});

async function input(): Promise<ExportInput> {
  const src = nodeSource(media);
  const d = await demux(src);
  const mux = await readMuxSource(src, d, path.basename(media));
  const t = Array.from({ length: 60 }, (_, i) => Math.round(i * 0.2 * 1000) / 1000);
  const samples: Samples = {
    t, score: t.map((x) => (x >= 3 && x <= 6 || x >= 8.4 && x <= 11 ? 0.93456 : 0.01)), box: t.map((x) => (x >= 3 && x <= 6 || x >= 8.4 && x <= 11 ? [0.1, -0.0004, 0.41234, 1.0004] : null)),
    track: null, enter: 0.5, exit: 0.3, merge_gap: 1, min_len: 0.5, pad: 1, mode: 'detect',
  };
  const rot = d.rotation;
  const m: ExportInput['media'] = { name: path.basename(media), size: src.size, duration: d.duration, fps: d.fps, rotation: rot, mirrored: d.mirrored, width: d.video.width, height: d.video.height, display_w: d.video.height, display_h: d.video.width } satisfies Partial<MediaInfo> as ExportInput['media'];
  return {
    media: m, hash: createHash('sha256').update(fs.readFileSync(media)).digest('hex'), mux, src, samples,
    settings: { mode: 'detect', fps: 5, enter: 0.5, exit: 0.3, target: 'any-person', detector: 'onnxruntime-web webgpu', model: 'small' },
    clips: [{ start: 2.4, end: 5.1 }, { start: 8.2, end: 11.9 }],
  };
}

describe.skipIf(!haveFfmpeg || !haveUnzip)('the zip of a browser export', () => {
  it('unzips (unzip -t), holds clips + sidecars + CSV, the sidecars pass validate_sidecars.py and the CSV equals the Python one', async () => {
    const inp = await input();
    const plan = planExport(inp);
    expect(plan.clips).toHaveLength(2);
    expect(plan.failed).toEqual([]);
    const zip = path.join(dir, plan.zip);
    let counted = 0;
    const chunks: Uint8Array[] = [];
    const rd = exportStream(plan, inp, { onBytes: (n) => (counted += n) }).getReader();
    for (;;) {
      const { done, value } = await rd.read();
      if (done) break;
      chunks.push(value);
    }
    fs.writeFileSync(zip, Buffer.concat(chunks));
    expect(counted).toBe(plan.clips.reduce((s, c) => s + c.muxed.size, 0));

    const t = spawnSync('unzip', ['-t', zip], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    expect(t.stdout).toMatch(/No errors detected/);

    const out = path.join(dir, 'unzipped');
    sh('unzip', ['-q', zip, '-d', out]);
    const names = fs.readdirSync(out).sort();
    const stem = path.basename(media).replace(/\.[^.]+$/, '');
    const ext = path.extname(media);
    expect(names).toEqual([
      `${stem}_01_0002-0005${ext}`, `${stem}_01_0002-0005.inframe.json`, `${stem}_02_0008-0011${ext}`, `${stem}_02_0008-0011.inframe.json`, `${stem}_losslesscut.csv`,
    ].sort());
    for (const c of plan.clips) expect(fs.statSync(path.join(out, c.name)).size).toBe(c.muxed.size);

    if (haveUv) {
      const v = spawnSync('uv', ['run', '--project', REPO, 'python', path.join(REPO, 'tools/validate_sidecars.py'), out], { encoding: 'utf8', cwd: REPO });
      expect(v.status, v.stdout + v.stderr).toBe(0);
      expect(v.stdout).toMatch(/PASS validate_sidecars: 2 of 2 sidecars valid/);
      const want = py('import sys,json,tempfile,os; from inframe.cut import write_losslesscut_csv; rows=[tuple(r) for r in json.loads(sys.argv[1])]; p=os.path.join(tempfile.mkdtemp(),"x.csv"); write_losslesscut_csv(p, rows); sys.stdout.write(open(p,newline="").read())',
        JSON.stringify(plan.clips.map((c) => [c.start, c.end, c.name])));
      expect(fs.readFileSync(path.join(out, `${stem}_losslesscut.csv`), 'utf8')).toBe(want);
    }

    // the sidecar says what the Python one would: the keyframe the clip really starts at, the box clamped to the frame, the source hash
    const sc = JSON.parse(fs.readFileSync(path.join(out, `${stem}_01_0002-0005.inframe.json`), 'utf8'));
    expect(sc.clip).toMatchObject({ file: `${stem}_01_0002-0005${ext}`, in_source_s: 2.4, out_source_s: 5.1, copy_start_s: 2, duration_s: 3.1 });
    expect(sc.source.hash).toEqual({ algorithm: 'sha256', value: inp.hash, size_bytes: fs.statSync(media).size });
    expect(sc.source.path).toBe(path.basename(media));
    expect(sc.tries_total).toBe(2);
    expect(sc.samples.every((s: any) => s.box === null || s.box.every((v: number) => v >= 0 && v <= 1))).toBe(true);
    expect(sc.samples[0].clip_time).toBeCloseTo(sc.samples[0].source_time - 2, 3);
    expect(sc.presence_in_clip.source_s).toEqual([[2.4, 5.1]]);
  });

  it('a clip that cannot be planned is reported and the others still go', async () => {
    const inp = await input();
    const plan = planExport({ ...inp, clips: [{ start: 5, end: 5 }, { start: 8.2, end: 11.9 }] });
    expect(plan.clips.map((c) => c.index)).toEqual([2]);
    expect(plan.failed).toHaveLength(1);
    expect(plan.failed[0]).toMatchObject({ index: 1 });
    expect(plan.warnings.join(' ')).toMatch(/clip 1 could not be cut/);
  });
});
