// @vitest-environment node
// What a stream says about its colours (src/browser/colour.ts), read from the demux of media ffmpeg makes here (synthetic: testsrc2, never footage), and
// what FrameRenderer then converts with. The truth is what the Mac's decoder reports for the same file (ffprobe's color_space / color_range /
// chroma_location), because the Mac converts with the frame's tags: an untagged stream is BT.601, limited range, and h264 / hevc chroma is sited LEFT.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
// @ts-ignore
import { execFileSync } from 'node:child_process';
// @ts-ignore
import fs from 'node:fs';
// @ts-ignore
import os from 'node:os';
// @ts-ignore
import path from 'node:path';
import { demux, type ByteSource } from '../src/browser/demux';
import { NO_STREAM_COLOUR, pictureColour, streamColour } from '../src/browser/colour';
import { FrameRenderer } from '../src/browser/pipeline';
import type { OpenVideo } from '../src/browser/frames';

const sh = (cmd: string, args: string[]): string => execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) as string;
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
const haveVp9 = encoders.includes('libvpx-vp9');
const haveAv1 = encoders.includes('libsvtav1');
const ff = (...a: string[]) => sh('ffmpeg', ['-nostats', '-loglevel', 'error', '-y', ...a]);
const src = (size = '1280x720') => ['-f', 'lavfi', '-i', `testsrc2=size=${size}:rate=30:duration=1`];

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

/** ffprobe's view of the decoded stream: what the Mac's PyAV frames carry. */
function mac(file: string) {
  const d = JSON.parse(sh('ffprobe', ['-v', 'error', '-of', 'json', '-select_streams', 'v:0', '-show_entries', 'stream=color_space,color_range,chroma_location', file]));
  return d.streams[0] as { color_space?: string; color_range?: string; chroma_location?: string };
}

/** An h264 SPS NAL (header byte included, baseline profile) with vui_parameters_present_flag cleared and the VUI cut off: x264 always writes a VUI, so the
 *  stream that names nothing at all is made by rewriting the SPS. Returns the NAL with emulation prevention applied. */
function withoutVui(nal: Uint8Array): Uint8Array {
  const rbsp: number[] = [];
  for (let i = 1, zeros = 0; i < nal.length; i++) {
    if (zeros >= 2 && nal[i] === 3) {
      zeros = 0;
      continue;
    }
    zeros = nal[i] === 0 ? zeros + 1 : 0;
    rbsp.push(nal[i]);
  }
  let pos = 24; // profile_idc, constraint flags, level_idc
  const bit = () => (rbsp[pos >> 3] >> (7 - (pos++ & 7))) & 1;
  const ue = () => {
    let z = 0;
    while (!bit()) z++;
    let v = 1;
    for (let i = 0; i < z; i++) v = (v << 1) | bit();
    return v - 1;
  };
  if (rbsp[0] !== 66) throw new Error('baseline only');
  ue(); // sps id
  ue(); // log2_max_frame_num_minus4
  const poc = ue();
  if (poc === 0) ue();
  else if (poc === 1) throw new Error('poc type 1');
  ue(); // max_num_ref_frames
  bit(); // gaps
  ue();
  ue();
  if (!bit()) bit(); // frame_mbs_only_flag
  bit(); // direct_8x8_inference_flag
  if (bit()) for (let i = 0; i < 4; i++) ue();
  const flagAt = pos;
  if (!bit()) throw new Error('the SPS has no VUI already');
  const bits: number[] = [];
  for (let i = 0; i < flagAt; i++) bits.push((rbsp[i >> 3] >> (7 - (i & 7))) & 1);
  bits.push(0, 1); // vui_parameters_present_flag = 0, rbsp_stop_one_bit
  while (bits.length % 8) bits.push(0);
  const out: number[] = [nal[0]];
  for (let i = 0, zeros = 0; i < bits.length; i += 8) {
    let b = 0;
    for (let j = 0; j < 8; j++) b = (b << 1) | bits[i + j];
    if (zeros >= 2 && b <= 3) {
      out.push(3);
      zeros = 0;
    }
    out.push(b);
    zeros = b === 0 ? zeros + 1 : 0;
  }
  return Uint8Array.from(out);
}

/** An Annex B h264 stream (as ffmpeg writes it) with each SPS NAL passed through fn. */
function mapSps(raw: Uint8Array, fn: (nal: Uint8Array) => Uint8Array): Uint8Array {
  const starts: number[] = [];
  for (let i = 0; i + 3 < raw.length; i++) if (raw[i] === 0 && raw[i + 1] === 0 && (raw[i + 2] === 1 || (raw[i + 2] === 0 && raw[i + 3] === 1))) starts.push(i);
  const parts: number[][] = [];
  starts.forEach((st, k) => {
    const code = raw[st + 2] === 1 ? 3 : 4;
    const end = k + 1 < starts.length ? starts[k + 1] : raw.length;
    const nal = raw.subarray(st + code, end);
    parts.push([0, 0, 0, 1, ...((nal[0] & 0x1f) === 7 ? fn(nal) : nal)]);
  });
  return Uint8Array.from(parts.flat());
}

let dir = '';
const files: Record<string, string> = {};
beforeAll(() => {
  if (!haveFfmpeg) return;
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'inframe-colour-'));
  const f = (name: string) => (files[name] = path.join(dir, name));
  const x264 = (name: string, ...extra: string[]) => ff(...src(), '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', ...extra, f(name));
  x264('h264_untagged.mp4');
  x264('h264_709.mp4', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv');
  x264('h264_601_full.mp4', '-colorspace', 'smpte170m', '-color_primaries', 'smpte170m', '-color_trc', 'smpte170m', '-color_range', 'pc');
  x264('h264_2020.mp4', '-colorspace', 'bt2020nc', '-color_primaries', 'bt2020', '-color_trc', 'bt709');
  x264('h264_centre.mp4', '-x264-params', 'chromaloc=1'); // chroma_sample_loc_type 1: centre
  x264('h264_cqm.mp4', '-x264-params', 'cqm=jvt', '-profile:v', 'high', '-colorspace', 'bt709'); // scaling lists in the SPS before the VUI
  {
    // the stream that names nothing at all: no VUI in its SPS
    const raw = path.join(dir, 'with_vui.h264');
    ff(...src('320x180'), '-c:v', 'libx264', '-preset', 'ultrafast', '-profile:v', 'baseline', '-pix_fmt', 'yuv420p', '-bsf:v', 'h264_mp4toannexb', '-f', 'h264', raw);
    const bare = path.join(dir, 'no_vui.h264');
    fs.writeFileSync(bare, mapSps(new Uint8Array(fs.readFileSync(raw)), withoutVui));
    ff('-f', 'h264', '-framerate', '30', '-i', bare, '-c:v', 'copy', f('h264_novui.mp4'));
  }
  x264('h264_interlaced.mp4', '-x264-params', 'tff=1', '-colorspace', 'bt709', '-flags', '+ildct'); // frame_mbs_only_flag 0 where x264 allows it
  if (haveX265) {
    const x265 = (name: string, ...extra: string[]) => ff(...src(), '-c:v', 'libx265', '-tag:v', 'hvc1', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', ...extra, f(name));
    x265('hevc_untagged.mp4', '-x265-params', 'log-level=error');
    x265('hevc_709.mp4', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-x265-params', 'log-level=error');
    x265('hevc_601_full.mp4', '-colorspace', 'smpte170m', '-color_range', 'pc', '-x265-params', 'log-level=error');
    x265('hevc_centre.mp4', '-x265-params', 'log-level=error:chromaloc=1');
  }
  if (haveVp9) ff(...src('320x180'), '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuv420p', f('vp9_untagged.mp4'));
  if (haveAv1) ff(...src('320x180'), '-c:v', 'libsvtav1', '-pix_fmt', 'yuv420p', f('av1_untagged.mp4'));
});
afterAll(() => {
  if (dir) fs.rmSync(dir, { recursive: true, force: true });
});

const colourOf = async (name: string) => (await demux(nodeSource(files[name]))).colour;

describe.skipIf(!haveFfmpeg)('the colours a stream names, read from its SPS', () => {
  it('an h264 stream that names nothing is BT.601, limited range, chroma left (not what Chrome reports)', async () => {
    expect(await colourOf('h264_untagged.mp4')).toEqual({ matrix: 'smpte170m', fullRange: false, chromaLoc: 'left' });
    expect(mac(files['h264_untagged.mp4']).color_space ?? 'unknown').toMatch(/unknown|unspecified/);
    expect(mac(files['h264_untagged.mp4']).chroma_location).toBe('left');
  });
  it.skipIf(!haveX265)('an hevc stream that names nothing is the same', async () => {
    expect(await colourOf('hevc_untagged.mp4')).toEqual({ matrix: 'smpte170m', fullRange: false, chromaLoc: 'left' });
    expect(mac(files['hevc_untagged.mp4']).color_space ?? 'unknown').toMatch(/unknown|unspecified/);
  });
  it('a named matrix and range are kept', async () => {
    expect(await colourOf('h264_709.mp4')).toEqual({ matrix: 'bt709', fullRange: false, chromaLoc: 'left' });
    expect((await colourOf('h264_601_full.mp4'))).toEqual({ matrix: 'smpte170m', fullRange: true, chromaLoc: 'left' });
    expect((await colourOf('h264_2020.mp4')).matrix).toBe('bt2020-ncl');
    expect(mac(files['h264_601_full.mp4']).color_range).toBe('pc');
  });
  it.skipIf(!haveX265)('the same for hevc', async () => {
    expect(await colourOf('hevc_709.mp4')).toEqual({ matrix: 'bt709', fullRange: false, chromaLoc: 'left' });
    expect(await colourOf('hevc_601_full.mp4')).toEqual({ matrix: 'smpte170m', fullRange: true, chromaLoc: 'left' });
  });
  it('a chroma siting named in the VUI is read from it', async () => {
    expect((await colourOf('h264_centre.mp4')).chromaLoc).toBe('center');
    expect(mac(files['h264_centre.mp4']).chroma_location).toBe('center');
  });
  it.skipIf(!haveX265)('and for hevc', async () => {
    expect((await colourOf('hevc_centre.mp4')).chromaLoc).toBe('center');
    expect(mac(files['hevc_centre.mp4']).chroma_location).toBe('center');
  });
  it('an SPS with no VUI at all is BT.601, limited range, and its chroma is unspecified: centred, as the Mac converts it', async () => {
    const c = await colourOf('h264_novui.mp4');
    expect(c).toEqual({ matrix: 'smpte170m', fullRange: false, chromaLoc: 'center' });
    expect(mac(files['h264_novui.mp4']).color_space ?? 'unknown').toMatch(/unknown|unspecified/);
    expect(mac(files['h264_novui.mp4']).chroma_location ?? 'unspecified').toMatch(/unspecified|unknown/);
    // and the frame's own matrix (Chrome says bt709) does not override it
    expect(pictureColour({ matrix: 'bt709', fullRange: false }, c)).toEqual({ matrix: 'smpte170m', fullRange: false, chromaLoc: 'center' });
  });
  it('an SPS with scaling lists before its VUI is still read', async () => {
    expect((await colourOf('h264_cqm.mp4')).matrix).toBe('bt709');
  });
  it('an SPS with interlaced coding before its VUI is still read', async () => {
    expect((await colourOf('h264_interlaced.mp4')).matrix).toBe('bt709');
  });
  it.skipIf(!haveVp9)('vp9 has no siting for swscale: centred, and the frame keeps its own colour space', async () => {
    expect(await colourOf('vp9_untagged.mp4')).toEqual({ matrix: null, fullRange: null, chromaLoc: 'center' });
    expect(mac(files['vp9_untagged.mp4']).chroma_location ?? 'unspecified').toMatch(/unspecified|unknown/);
  });
  it.skipIf(!haveAv1)('and av1', async () => {
    expect((await colourOf('av1_untagged.mp4')).chromaLoc).toBe('center');
  });
});

describe('what a frame is converted with', () => {
  it('an SPS that cannot be read keeps the frame colour, and the stream wins over Chrome when it can be', () => {
    const unreadable = streamColour('h264', null, null);
    expect(unreadable.matrix).toBeNull();
    const stream = { matrix: 'smpte170m' as const, fullRange: false, chromaLoc: 'left' as const };
    expect(pictureColour({ matrix: 'bt709', fullRange: false }, stream)).toEqual({ matrix: 'smpte170m', fullRange: false, chromaLoc: 'left' });
  });
  it('a stream whose tags are not read keeps the frame matrix and range', () => {
    expect(pictureColour({ matrix: 'bt709', fullRange: true }, { matrix: null, fullRange: null, chromaLoc: 'center' })).toEqual({ matrix: 'bt709', fullRange: true, chromaLoc: 'center' });
    expect(pictureColour(undefined, NO_STREAM_COLOUR)).toEqual({ matrix: null, fullRange: false, chromaLoc: 'center' });
  });
  it('the container colr box is used when the SPS names nothing, and for codecs without an SPS', () => {
    const colr = { matrix_coefficients: 1, full_range: true };
    expect(streamColour('vp9', null, colr)).toEqual({ matrix: 'bt709', fullRange: true, chromaLoc: 'center' });
    expect(streamColour('h264', null, colr)).toEqual({ matrix: 'bt709', fullRange: true, chromaLoc: 'left' });
  });
});

/** A decoded NV12 frame as WebCodecs gives it: flat Y120 U90 V200, limited range, colorSpace as Chrome reports an untagged H.264 clip. */
function fakeFrame(matrix: string, fullRange = false) {
  const w = 32;
  const h = 32;
  const bytes = new Uint8Array(w * h + w * (h / 2));
  bytes.fill(120, 0, w * h);
  for (let i = w * h; i < bytes.length; i += 2) {
    bytes[i] = 90;
    bytes[i + 1] = 200;
  }
  return {
    format: 'NV12', codedWidth: w, codedHeight: h, visibleRect: { width: w, height: h }, colorSpace: { matrix, fullRange },
    allocationSize: () => bytes.length,
    copyTo: async (into: Uint8Array) => {
      into.set(bytes);
      return [{ offset: 0, stride: w }, { offset: w * h, stride: w }];
    },
  } as unknown as VideoFrame;
}

describe('FrameRenderer on a frame of a stream that names nothing', () => {
  const video = (colour: ReturnType<typeof streamColour>) =>
    ({ demuxed: { video: { width: 32, height: 32 }, rotation: 0, mirrored: false, sar: 1, colour } }) as unknown as OpenVideo;
  const rgbOf = async (colour: ReturnType<typeof streamColour>) => {
    const r = new FrameRenderer(video(colour), 32, 640, true);
    await r.prepare(fakeFrame('bt709'));
    const { rgba } = r.pixels();
    return [rgba[0], rgba[1], rgba[2]];
  };
  it('converts as BT.601 (PyAV 19 reformat to rgb24 of the same planes: unspecified and smpte170m give [236, 77, 44], bt709 gives [250, 91, 41])', async () => {
    const got = await rgbOf({ matrix: 'smpte170m', fullRange: false, chromaLoc: 'left' });
    expect(got.every((v, i) => Math.abs(v - [236, 77, 44][i]) <= 2)).toBe(true);
  });
  it('a stream that really names bt709 converts as BT.709', async () => {
    const got = await rgbOf({ matrix: 'bt709', fullRange: false, chromaLoc: 'left' });
    expect(got.every((v, i) => Math.abs(v - [250, 91, 41][i]) <= 2)).toBe(true);
  });
});

describe('the chroma siting reaches the conversion', () => {
  // a 800 x 64 frame is scaled to 640, so the chroma filter runs; U ramps along x, so a quarter-sample shift changes the picture
  function ramp() {
    const w = 800;
    const h = 64;
    const bytes = new Uint8Array(w * h + w * (h / 2));
    bytes.fill(120, 0, w * h);
    for (let r = 0; r < h / 2; r++) for (let c = 0; c < w / 2; c++) {
      bytes[w * h + r * w + c * 2] = 40 + ((c * 7) % 160);
      bytes[w * h + r * w + c * 2 + 1] = 128;
    }
    return {
      format: 'NV12', codedWidth: w, codedHeight: h, visibleRect: { width: w, height: h }, colorSpace: { matrix: 'smpte170m', fullRange: false },
      allocationSize: () => bytes.length,
      copyTo: async (into: Uint8Array) => (into.set(bytes), [{ offset: 0, stride: w }, { offset: w * h, stride: w }]),
    } as unknown as VideoFrame;
  }
  const picture = async (colour: ReturnType<typeof streamColour>) => {
    const v = { demuxed: { video: { width: 800, height: 64 }, rotation: 0, mirrored: false, sar: 1, colour } } as unknown as OpenVideo;
    const r = new FrameRenderer(v, 64, 640, true);
    await r.prepare(ramp());
    return r.pixels().rgba.slice();
  };
  it('a vp9 stream (centred) is not converted like an h264 one (left)', async () => {
    const left = await picture({ matrix: 'smpte170m', fullRange: false, chromaLoc: 'left' });
    const centre = await picture(streamColour('vp9', null, { matrix_coefficients: 6, full_range: false }));
    let differing = 0;
    for (let i = 0; i < left.length; i++) if (left[i] !== centre[i]) differing++;
    expect(differing).toBeGreaterThan(left.length / 10);
  });
});
