// @vitest-environment node
// The browser demux (src/browser/demux.ts) against ffprobe, on media ffmpeg makes here (synthetic: testsrc2 and a sine, never footage).
// Each file is read in 8 MB chunks through a ByteSource that only Node's fs backs, so nothing but the demux decides what is read.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
// the web package has no @types/node; vitest runs under node, so these imports are allowed to be untyped
// @ts-ignore
import { execFileSync } from 'node:child_process';
// @ts-ignore
import fs from 'node:fs';
// @ts-ignore
import os from 'node:os';
// @ts-ignore
import path from 'node:path';
import { CHUNK_BYTES, demux, editShift, mirroredOf, rotationOf, type ByteSource } from '../src/browser/demux';

const sh = (cmd: string, args: string[]): string => execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) as string;
function has(cmd: string): boolean {
  try {
    sh(cmd, ['-version']);
    return true;
  } catch {
    return false;
  }
}
const haveFfmpeg = has('ffmpeg') && has('ffprobe');
const haveX265 = haveFfmpeg && sh('ffmpeg', ['-hide_banner', '-encoders']).includes('libx265');
const encoders = haveFfmpeg ? sh('ffmpeg', ['-hide_banner', '-encoders']) : '';
const haveAv1 = encoders.includes('libsvtav1');
const haveVp9 = encoders.includes('libvpx-vp9');
const ff = (...a: string[]) => sh('ffmpeg', ['-nostats', '-loglevel', 'error', '-y', ...a]);

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

/** What probe.probe reads: the same ffprobe call and the same rules (rotation rounded, offset = stream start - container start). */
function truth(file: string) {
  const d = JSON.parse(sh('ffprobe', ['-v', 'error', '-of', 'json', '-show_entries',
    'stream=index,codec_type,codec_name,duration,start_time:stream_side_data=rotation,displaymatrix:format=duration,start_time', file]));
  const v = d.streams.find((s: any) => s.codec_type === 'video');
  const fmtStart = Number(d.format.start_time);
  const start = Number(v.start_time);
  const duration = Number(v.duration ?? d.format.duration);
  const rot = (v.side_data_list ?? []).find((x: any) => 'rotation' in x);
  const packets = JSON.parse(sh('ffprobe', ['-v', 'error', '-of', 'json', '-select_streams', 'v:0', '-show_entries', 'packet=pts_time,flags', file])).packets as { pts_time: string; flags: string }[];
  return {
    codec: v.codec_name as string,
    rotation: rot ? Math.round(Number(rot.rotation)) : 0,
    duration,
    offset: start - fmtStart,
    end_time: start - fmtStart + duration,
    has_audio: d.streams.some((s: any) => s.codec_type === 'audio'),
    keyframes: packets.filter((p) => p.flags.includes('K')).map((p) => Number(p.pts_time) - fmtStart).sort((a, b) => a - b),
  };
}

let dir = '';
const files: Record<string, string> = {};

beforeAll(() => {
  if (!haveFfmpeg) return;
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'inframe-demux-'));
  const f = (name: string) => (files[name] = path.join(dir, name));
  const video = (size: string, dur: number, noise = false) => `testsrc2=size=${size}:rate=30:duration=${dur}${noise ? ',noise=alls=30:allf=t+u' : ''}`;
  const sine = (dur: number) => `sine=frequency=440:duration=${dur}`;
  if (haveX265) {
    // ffmpeg autorotates a re-encode, so the matrix is added by a second, stream-copy pass
    ff('-f', 'lavfi', '-i', video('1280x720', 10), '-f', 'lavfi', '-i', sine(10), '-c:v', 'libx265', '-tag:v', 'hvc1', '-preset', 'ultrafast', '-g', '30',
      '-x265-params', 'log-level=error', '-c:a', 'aac', '-shortest', f('hevc_raw.mp4'));
    ff('-display_rotation', '90', '-i', files['hevc_raw.mp4'], '-c', 'copy', '-tag:v', 'hvc1', f('hevc_rot90.mp4'));
  }
  // AV1 and VP9 in mp4: mp4box cannot write av1C / vpcC back, so the demux must not hand the decoder a description for them
  if (haveAv1) ff('-f', 'lavfi', '-i', video('320x180', 2), '-c:v', 'libsvtav1', '-g', '30', '-an', f('av1.mp4'));
  if (haveVp9) ff('-f', 'lavfi', '-i', video('320x180', 2), '-c:v', 'libvpx-vp9', '-g', '30', '-an', f('vp9.mp4'));
  // an H.264 .mov: ffmpeg writes its moov at the end; B-frames give it an edit list
  ff('-f', 'lavfi', '-i', video('640x360', 6), '-f', 'lavfi', '-i', sine(6), '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '25', '-bf', '2', '-c:a', 'aac', '-shortest', f('h264_moov_end.mov'));
  // the video starts 1.5 s into the file, the audio at 0: offset 1.5
  ff('-itsoffset', '1.5', '-f', 'lavfi', '-i', video('640x360', 5), '-f', 'lavfi', '-i', sine(6), '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '25', '-bf', '2', '-c:a', 'aac', f('offset.mp4'));
  // a file of about 30 MB (four 8 MB chunks) with the moov at the end, then the same with the moov first
  ff('-f', 'lavfi', '-i', video('1280x720', 10, true), '-f', 'lavfi', '-i', sine(10), '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-g', '30', '-c:a', 'aac', '-shortest', f('big_moov_end.mp4'));
  ff('-i', files['big_moov_end.mp4'], '-c', 'copy', '-movflags', '+faststart', f('big_faststart.mp4'));
  // no audio at all, then the same with a 180 degree matrix (ffprobe says -180)
  ff('-f', 'lavfi', '-i', video('320x180', 3), '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '30', '-an', f('silent.mp4'));
  ff('-display_rotation', '180', '-i', files['silent.mp4'], '-c', 'copy', f('rot180.mp4'));
});
afterAll(() => {
  if (dir) fs.rmSync(dir, { recursive: true, force: true });
});

function agrees(name: string) {
  return async () => {
    const t = truth(files[name]);
    const d = await demux(nodeSource(files[name]));
    expect(d.codec).toBe(t.codec);
    expect(d.rotation).toBe(t.rotation);
    expect(d.hasAudio).toBe(t.has_audio);
    expect(d.duration).toBeCloseTo(t.duration, 3);
    expect(d.offset).toBeCloseTo(t.offset, 3);
    expect(d.endTime).toBeCloseTo(t.end_time, 3);
    expect(d.keyframes.length).toBe(t.keyframes.length);
    d.keyframes.forEach((k, i) => expect(Math.abs(k - t.keyframes[i]), `keyframe ${i}: ${k} vs ${t.keyframes[i]}`).toBeLessThanOrEqual(0.001));
    expect(d.stats.peakHeldBytes, 'peak chunk memory').toBeLessThan(3 * CHUNK_BYTES);
    return { d, t };
  };
}

if (!haveFfmpeg) {
  it.skip('skipped: ffmpeg and ffprobe are not on PATH, so the synthetic media cannot be made', () => {});
} else {
  describe('demux equals ffprobe', () => {
    if (haveX265) {
      it('10 s HEVC hvc1 1280x720 30 fps, display rotation 90, AAC', async () => {
        const { d } = await agrees('hevc_rot90.mp4')();
        expect(d.rotation).toBe(90);
        expect(d.video.fourcc).toBe('hvc1');
        expect(d.video).toMatchObject({ width: 1280, height: 720 });
        expect(d.fps).toBeCloseTo(30, 3);
        expect(d.keyframes).toHaveLength(10);
        expect(d.video.description?.length).toBeGreaterThan(20); // the hvcC payload a VideoDecoder needs
        expect(d.audio?.sampleRate).toBe(44100);
      });
    } else {
      it.skip('skipped: this ffmpeg has no libx265, so the HEVC file cannot be made', () => {});
    }
    it.skipIf(!haveAv1)('AV1: no VideoDecoder description (mp4box cannot write av1C, it would be empty)', async () => {
      const { d } = await agrees('av1.mp4')();
      expect(d.video.fourcc).toBe('av01');
      expect(d.video.description).toBeNull();
    });
    it.skipIf(!haveVp9)('VP9: no VideoDecoder description (mp4box writes only the vpcC version and flags)', async () => {
      const { d } = await agrees('vp9.mp4')();
      expect(d.video.fourcc).toBe('vp09');
      expect(d.video.description).toBeNull();
    });
    it('H.264 .mov with the moov at the end', async () => {
      const { d } = await agrees('h264_moov_end.mov')();
      expect(d.codec).toBe('h264');
      expect(d.rotation).toBe(0);
      expect(d.profile).toBe('Main');
      expect(d.video.codecString.startsWith('avc1.')).toBe(true);
      expect(d.video.description?.length).toBeGreaterThan(5); // the avcC payload is still sent
    });
    it('a file whose video starts 1.5 s in (nonzero offset)', async () => {
      const { d, t } = await agrees('offset.mp4')();
      expect(t.offset).toBeCloseTo(1.5, 3);
      expect(d.offset).toBeCloseTo(1.5, 3);
      expect(d.keyframes[0]).toBeCloseTo(1.5, 3);
      expect(d.endTime).toBeCloseTo(6.5, 3);
    });
    it('a 180 degree matrix reads as ffprobe reads it', async () => {
      const { d, t } = await agrees('rot180.mp4')();
      expect(t.rotation).toBe(-180);
      expect(d.rotation).toBe(-180);
    });
    it('a file without audio', async () => {
      const { d } = await agrees('silent.mp4')();
      expect(d.hasAudio).toBe(false);
      expect(d.audio).toBeNull();
    });
  });

  describe('chunked reading', () => {
    it('moov at the end: reads the first chunk, jumps over the mdat, reads the tail; never holds 3 chunks', async () => {
      const src = nodeSource(files['big_moov_end.mp4']);
      expect(src.size).toBeGreaterThan(3 * CHUNK_BYTES);
      const { d } = await agrees('big_moov_end.mp4')();
      expect(d.stats.reads).toBeLessThanOrEqual(3);
      expect(d.stats.bytesRead).toBeLessThan(src.size / 2);
      expect(d.stats.chunkBytes).toBe(8 * 1024 * 1024);
    });
    it('moov first: one chunk is enough', async () => {
      const { d } = await agrees('big_faststart.mp4')();
      expect(d.stats.reads).toBe(1);
    });
    it('says why a file is not a movie', async () => {
      const junk = { size: 5000, read: async (a: number, b: number) => new Uint8Array(b - a).buffer };
      await expect(demux(junk)).rejects.toMatchObject({ status: 422 });
    });
  });
}

describe('pure helpers', () => {
  it('rotationOf follows ffmpeg display matrices', () => {
    const fx = (v: number[]) => v.map((x) => Math.round(x * 65536));
    expect(rotationOf(fx([1, 0, 0, 0, 1, 0, 0, 0, 0]))).toBe(0);
    expect(rotationOf(fx([0, -1, 0, 1, 0, 0, 0, 0, 0]))).toBe(90);
    expect(rotationOf(fx([0, 1, 0, -1, 0, 0, 0, 0, 0]))).toBe(-90); // an iPhone held upright
    expect(rotationOf(fx([-1, 0, 0, 0, -1, 0, 0, 0, 0]))).toBe(-180); // ffmpeg returns -atan2, so a half turn is -180
    expect(rotationOf([0, 0, 0, 0, 0, 0, 0, 0, 0])).toBe(0);
  });
  it('mirroredOf is the sign of the 2x2 determinant', () => {
    expect(mirroredOf([65536, 0, 0, 0, 65536, 0, 0, 0, 0])).toBe(false);
    expect(mirroredOf([65536, 0, 0, 0, -65536, 0, 0, 0, 0])).toBe(true);
    expect(mirroredOf([-65536, 0, 0, 0, 65536, 0, 0, 0, 0])).toBe(true);
  });
  it('editShift: an empty edit delays, the first real edit sets the media start', () => {
    expect(editShift(undefined, 1000, 4)).toEqual({ empty: 0, mediaTime: 0, duration: 4 });
    expect(editShift([{ segment_duration: 1500, media_time: -1 }, { segment_duration: 5000, media_time: 1024 }], 1000, 5.07)).toEqual({ empty: 1.5, mediaTime: 1024, duration: 5 });
    expect(editShift([{ segment_duration: 1500, media_time: 0xffffffff }, { segment_duration: 0, media_time: 0 }], 1000, 3)).toEqual({ empty: 1.5, mediaTime: 0, duration: 3 });
  });
});
