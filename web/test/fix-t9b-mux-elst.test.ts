// @vitest-environment node
// A source whose video edit list starts past its first keyframe (`ffmpeg -ss 1.5 -i src -c copy`): the keyframe before the edit point is
// decode-only and must stay a sync sample, so a clip from 0 starts there (cut.snap_to_keyframe), not at the next keyframe.
import { afterAll, beforeAll, expect, it } from 'vitest';
// @ts-ignore
import { execFileSync } from 'node:child_process';
// @ts-ignore
import fs from 'node:fs';
// @ts-ignore
import os from 'node:os';
// @ts-ignore
import path from 'node:path';
import { demux, type ByteSource } from '../src/browser/demux';
import { clipBytes, muxClip, planClip, readMuxSource } from '../src/browser/mux';

const sh = (cmd: string, args: string[], cwd?: string): string => execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], cwd }) as string;
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
const REPO = path.resolve(process.cwd(), '..');
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
let trimmed = '';
beforeAll(() => {
  if (!have) return;
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'inframe-elst-'));
  const raw = path.join(dir, 'src.mp4');
  trimmed = path.join(dir, 'trimmed.mp4');
  ff('-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=30:duration=8', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=8', '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '30', '-bf', '2', '-c:a', 'aac', '-shortest', raw);
  ff('-ss', '1.5', '-i', raw, '-c', 'copy', trimmed);
});
afterAll(() => {
  if (dir) fs.rmSync(dir, { recursive: true, force: true });
});

(have ? it : it.skip)('a clip from 0 of a trimmed source starts at the decode-only keyframe, as the Python cut does', async () => {
  const src = nodeSource(trimmed);
  const d = await demux(src);
  const m = await readMuxSource(src, d, 'trimmed.mp4');
  const plan = planClip(m, 0, 2);
  expect(plan.copyStart).toBeLessThan(0.1); // not the next keyframe at 0.5 s
  const out = path.join(dir, 'clip.mp4');
  fs.writeFileSync(out, await clipBytes(src, muxClip(m, plan)));
  const py = path.join(dir, 'py.mp4');
  sh('uv', ['run', '--project', REPO, 'python', '-c', 'import sys; from inframe.cut import cut_clip; cut_clip(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))', trimmed, py, '0', '2'], REPO);
  const n = (f: string) => Number(JSON.parse(sh('ffprobe', ['-v', 'error', '-of', 'json', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', f])).streams[0].nb_read_frames);
  expect(Math.abs(n(out) - n(py)), `frames ${n(out)} vs python ${n(py)}`).toBeLessThanOrEqual(1); // the same one-frame slack as mux.test.ts
  expect(plan.frames).toBe(n(py));
}, 60000);
