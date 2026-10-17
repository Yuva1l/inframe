// @vitest-environment node
// The browser's lossless cutter (src/browser/mux.ts) against ffprobe and against the Python `cut` (src/inframe/cut.py), on synthetic media ffmpeg
// makes here (testsrc2 and a sine, never footage): HEVC with B-frames, an open GOP, display rotation 90 and AAC, and an H.264 .mov.
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
import { demux, type ByteSource } from '../src/browser/demux';
import { clipBytes, muxClip, planClip, readMuxSource, type MuxSource } from '../src/browser/mux';

const sh = (cmd: string, args: string[], cwd?: string): string => execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], cwd }) as string;
const has = (cmd: string): boolean => {
  try {
    sh(cmd, ['-version']);
    return true;
  } catch {
    return false;
  }
};
const haveFfmpeg = has('ffmpeg') && has('ffprobe');
const encoders = haveFfmpeg ? sh('ffmpeg', ['-hide_banner', '-encoders']) : '';
const haveX265 = encoders.includes('libx265');
const ff = (...a: string[]) => sh('ffmpeg', ['-nostats', '-loglevel', 'error', '-y', ...a]);
const REPO = path.resolve(process.cwd(), '..');
const haveUv = (() => {
  try {
    sh('uv', ['--version']);
    return true;
  } catch {
    return false;
  }
})();

function nodeSource(file: string): ByteSource {
  const fd = fs.openSync(file, 'r');
  return {
    size: fs.fstatSync(fd).size,
    read: async (start, end) => {
      const u8 = new Uint8Array(end - start);
      fs.readSync(fd, u8, 0, u8.length, start);
      return u8.buffer;
    },
  };
}

// ---------------------------------------------------------------- a tiny box reader, to look at what the writer wrote
interface Box {
  type: string;
  start: number;
  size: number;
  body: number;
}
function boxes(b: Uint8Array, from = 0, to = b.length): Box[] {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const out: Box[] = [];
  let p = from;
  while (p + 8 <= to) {
    let size = dv.getUint32(p);
    let hdr = 8;
    if (size === 1) {
      size = Number(dv.getBigUint64(p + 8));
      hdr = 16;
    } else if (size === 0) size = to - p;
    out.push({ type: String.fromCharCode(b[p + 4], b[p + 5], b[p + 6], b[p + 7]), start: p, size, body: p + hdr });
    p += size;
  }
  return out;
}
const find = (b: Uint8Array, parent: Box | null, type: string): Box | undefined => boxes(b, parent ? parent.body : 0, parent ? parent.start + parent.size : b.length).find((x) => x.type === type);
const dig = (b: Uint8Array, ...types: string[]): Box | undefined => {
  let cur: Box | null = null;
  for (const t of types) {
    const nx = find(b, cur, t);
    if (!nx) return undefined;
    cur = nx;
  }
  return cur ?? undefined;
};
function tracks(b: Uint8Array): Box[] {
  const moov = dig(b, 'moov')!;
  return boxes(b, moov.body, moov.start + moov.size).filter((x) => x.type === 'trak');
}
function elstOf(b: Uint8Array, trak: Box): { segment: number; mediaTime: number }[] | null {
  const edts = find(b, trak, 'edts');
  const e = edts && find(b, edts, 'elst');
  if (!e) return null;
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const n = dv.getUint32(e.body + 4);
  return Array.from({ length: n }, (_, i) => ({ segment: dv.getUint32(e.body + 8 + i * 12), mediaTime: dv.getInt32(e.body + 12 + i * 12) }));
}
function entryOf(b: Uint8Array, trak: Box): { type: string; children: string[]; raw: Uint8Array } {
  const stbl = find(b, find(b, find(b, trak, 'mdia')!, 'minf')!, 'stbl')!;
  const stsd = find(b, stbl, 'stsd')!;
  const e = boxes(b, stsd.body + 8, stsd.start + stsd.size)[0];
  const skip = e.type === 'mp4a' ? 28 : 78; // AudioSampleEntry / VisualSampleEntry fields before the child boxes
  return { type: e.type, children: boxes(b, e.body + skip, e.start + e.size).map((x) => x.type), raw: b.slice(e.start, e.start + e.size) };
}

// ---------------------------------------------------------------- ffprobe
interface Facts {
  vTag: string;
  rotation: number;
  fps: number;
  frames: number;
  firstShown: number;
  duration: number;
  vStart: number;
  aStart: number | null;
  aRate: number;
}
function facts(file: string): Facts {
  const j = JSON.parse(sh('ffprobe', ['-v', 'error', '-of', 'json', '-count_frames', '-show_entries',
    'stream=codec_type,codec_tag_string,start_time,duration,r_frame_rate,nb_read_frames,sample_rate:stream_side_data=rotation:format=duration', file]));
  const v = j.streams.find((s: any) => s.codec_type === 'video');
  const a = j.streams.find((s: any) => s.codec_type === 'audio');
  const fr = JSON.parse(sh('ffprobe', ['-v', 'error', '-of', 'json', '-select_streams', 'v:0', '-show_entries', 'frame=pts_time', file])).frames as { pts_time: string }[];
  const [num, den] = String(v.r_frame_rate).split('/').map(Number);
  const rot = (v.side_data_list ?? []).find((x: any) => 'rotation' in x);
  return {
    vTag: v.codec_tag_string, rotation: rot ? Math.round(Number(rot.rotation)) : 0, fps: num / den, frames: Number(v.nb_read_frames),
    firstShown: Math.min(...fr.map((f) => Number(f.pts_time))), duration: Number(v.duration ?? j.format.duration), vStart: Number(v.start_time),
    aStart: a ? Number(a.start_time) : null, aRate: a ? Number(a.sample_rate) : 0,
  };
}
const decodeErrors = (file: string): string => {
  const r = spawnSync('ffmpeg', ['-nostats', '-v', 'error', '-i', file, '-f', 'null', '-'], { encoding: 'utf8' });
  return `${r.stdout}${r.stderr}`;
};

/** inframe.cut.cut_clip on the same stretch: the reference the browser clip must agree with. */
function pythonCut(src: string, out: string, start: number, end: number): void {
  sh('uv', ['run', '--project', REPO, 'python', '-c', 'import sys; from inframe.cut import cut_clip; cut_clip(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))', src, out, String(start), String(end)], REPO);
}

let dir = '';
const files: Record<string, string> = {};
const KINDS = ['hevc_open_gop_rot90.mp4', 'h264_bframes.mov'] as const;
const CLIPS: [number, number][] = [[2.4, 5.1], [0, 3], [8.2, 11.9]];

beforeAll(() => {
  if (!haveFfmpeg) return;
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'inframe-mux-'));
  const f = (n: string) => (files[n] = path.join(dir, n));
  const video = 'testsrc2=size=640x360:rate=30:duration=12';
  const sine = 'sine=frequency=440:duration=12';
  if (haveX265) {
    // open GOP: the pictures after each keyframe in decode order include some shown before it (RASL), which the cutter must drop
    ff('-f', 'lavfi', '-i', video, '-f', 'lavfi', '-i', sine, '-c:v', 'libx265', '-tag:v', 'hvc1', '-preset', 'ultrafast', '-g', '30',
      '-x265-params', 'open-gop=1:keyint=30:min-keyint=30:bframes=3:log-level=error', '-c:a', 'aac', '-shortest', f('raw.mp4'));
    ff('-display_rotation', '90', '-i', files['raw.mp4'], '-c', 'copy', '-tag:v', 'hvc1', f('hevc_open_gop_rot90.mp4'));
  }
  ff('-f', 'lavfi', '-i', video, '-f', 'lavfi', '-i', sine, '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '30', '-bf', '2', '-c:a', 'aac', '-shortest', f('h264_bframes.mov'));
});
afterAll(() => {
  if (dir) fs.rmSync(dir, { recursive: true, force: true });
});

async function open(file: string): Promise<{ m: MuxSource; src: ByteSource; keyframes: number[] }> {
  const src = nodeSource(file);
  const d = await demux(src);
  return { m: await readMuxSource(src, d, path.basename(file)), src, keyframes: d.keyframes };
}

/** cut.snap_to_keyframe */
const snap = (t: number, kf: number[]): number => {
  let best = kf[0];
  for (const k of kf) if (k <= t + 1e-6) best = k;
  return best;
};

if (!haveFfmpeg) {
  it.skip('skipped: ffmpeg and ffprobe are not on PATH, so the synthetic media cannot be made', () => {});
} else {
  for (const kind of KINDS) {
    const hevc = kind.startsWith('hevc');
    const run = hevc && !haveX265 ? describe.skip : describe;
    run(`${kind}: ${CLIPS.length} lossless clips`, () => {
      CLIPS.forEach(([start, end], i) => {
        it(`clip ${i + 1} (${start} to ${end} s) is a valid lossless cut that ffprobe, the plan and the Python cut agree on`, async () => {
          const file = files[kind];
          const { m, src, keyframes } = await open(file);
          const plan = planClip(m, start, end);
          const clip = muxClip(m, plan);
          const out = path.join(dir, `clip_${kind}_${i}${path.extname(file)}`);
          const bytes = await clipBytes(src, clip);
          fs.writeFileSync(out, bytes);
          expect(bytes.length).toBe(clip.size);

          const f = facts(out);
          const s = facts(file);
          const frame = 1 / s.fps;

          // 1. the first shown picture is at time 0 (one frame of slack)
          expect(Math.abs(f.firstShown), `first shown pts ${f.firstShown}`).toBeLessThanOrEqual(frame + 1e-6);
          expect(Math.abs(f.vStart)).toBeLessThanOrEqual(frame + 1e-6);
          // 2. an edit list in each track, 3. the display matrix, the sample entry (hvc1 + hvcC, or avc1 + avcC) and the tag are the source's
          const trs = tracks(bytes);
          expect(trs.length).toBe(2);
          for (const t of trs) expect(elstOf(bytes, t), 'elst').not.toBeNull();
          const e = entryOf(bytes, trs[0]);
          expect(e.type).toBe(hevc ? 'hvc1' : 'avc1');
          expect(e.children).toContain(hevc ? 'hvcC' : 'avcC');
          expect(f.vTag).toBe(hevc ? 'hvc1' : 'avc1');
          expect(f.rotation).toBe(s.rotation);
          expect(s.rotation).toBe(hevc ? 90 : 0);
          expect(Buffer.from(e.raw).equals(Buffer.from(m.video.entry))).toBe(true); // the sample entry is copied byte for byte
          // 4. frame count and duration equal the plan ...
          expect(Math.abs(f.frames - plan.frames)).toBeLessThanOrEqual(1);
          expect(Math.abs(f.duration - plan.shownS)).toBeLessThanOrEqual(frame + 1e-3);
          // ... and the Python cut of the same stretch, within a frame
          if (haveUv) {
            const ref = path.join(dir, `ref_${kind}_${i}${path.extname(file)}`);
            pythonCut(file, ref, start, end);
            const r = facts(ref);
            expect(Math.abs(f.frames - r.frames), `frames ${f.frames} vs python ${r.frames}`).toBeLessThanOrEqual(1);
            expect(Math.abs(f.duration - r.duration), `duration ${f.duration} vs python ${r.duration}`).toBeLessThanOrEqual(frame + 1e-3);
            expect(Math.abs((f.aStart ?? 0) - (r.aStart ?? 0))).toBeLessThanOrEqual(1024 / f.aRate + 1e-3);
          }
          // 5. audio starts within one AAC frame of the first picture
          expect(f.aStart).not.toBeNull();
          expect(Math.abs((f.aStart ?? 0) - f.vStart), `audio ${f.aStart} video ${f.vStart}`).toBeLessThanOrEqual(1024 / f.aRate + 1e-6);
          // 6. nothing to say when it is decoded end to end
          expect(decodeErrors(out)).toBe('');
          // the cut starts on the keyframe at or before the request, as snap_to_keyframe says
          expect(plan.copyStart).toBeLessThanOrEqual(start + 1e-6);
          expect(start - plan.copyStart).toBeLessThan(1.0001);
          expect(snap(start, keyframes)).toBeCloseTo(plan.copyStart, 6);
        });
      });
    });
  }

  describe.skipIf(!haveX265)('open GOP', () => {
    it('drops the leading pictures that follow a keyframe in decode order but are shown before it', async () => {
      const { m } = await open(files['hevc_open_gop_rot90.mp4']);
      const plan = planClip(m, 3.1, 5);
      expect(plan.dropped).toBeGreaterThan(0);
      expect(plan.kept[0]).toBe(plan.key);
      for (const j of plan.kept) expect(m.video.cts[j]).toBeGreaterThanOrEqual(m.video.cts[plan.key]);
    });
    it('the clip carries no track but the picture and the sound', async () => {
      const { m, src } = await open(files['hevc_open_gop_rot90.mp4']);
      const bytes = await clipBytes(src, muxClip(m, planClip(m, 1, 3)));
      const moov = dig(bytes, 'moov')!;
      expect(boxes(bytes, moov.body, moov.start + moov.size).map((b) => b.type)).toEqual(['mvhd', 'trak', 'trak']);
      expect(dig(bytes, 'udta')).toBeUndefined();
      expect(dig(bytes, 'moov', 'meta')).toBeUndefined();
    });
  });

  describe('a source without usable sound', () => {
    it('a silent file gives a clip with one track', async () => {
      const silent = path.join(dir, 'silent.mp4');
      ff('-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=30:duration=4', '-c:v', 'libx264', '-g', '30', '-an', silent);
      const { m, src } = await open(silent);
      const bytes = await clipBytes(src, muxClip(m, planClip(m, 0.5, 2.5)));
      expect(tracks(bytes).length).toBe(1);
      fs.writeFileSync(path.join(dir, 'silent_clip.mp4'), bytes);
      expect(decodeErrors(path.join(dir, 'silent_clip.mp4'))).toBe('');
    });
    it('non-AAC sound is left out with a warning', async () => {
      const f = path.join(dir, 'pcm.mov');
      ff('-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=30:duration=4', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=4', '-c:v', 'libx264', '-g', '30', '-c:a', 'pcm_s16le', f);
      const { m } = await open(f);
      expect(m.audio).toBeNull();
      expect(m.warnings[0]).toMatch(/not AAC/);
    });
  });
}
