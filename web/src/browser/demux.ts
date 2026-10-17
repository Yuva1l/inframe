// Chunked MP4/MOV demux with mp4box.js (BSD-3, docs/T9B_DESIGN.md section 2): the file is read in 8 MB slices and never held whole.
// mp4box skips the mdat payload (createFile() discards it) and tells us where to read next, so a file with its moov at the end costs
// two slices, not the whole file. The result mirrors what src/inframe/probe.py reads with ffprobe, in the same file time (the time
// `ffmpeg -ss` counts: first picture at `offset`, last at `end_time`), plus the sample table a decoder needs.
import { createFile, DataStream, Endianness, MP4BoxBuffer } from 'mp4box';
import { EngineError } from '../engine/types';
import { streamColour, type StreamColour } from './colour';

export const CHUNK_BYTES = 8 * 1024 * 1024;

/** Where bytes come from: a File in the page, a file read by Node in the tests. */
export interface ByteSource {
  size: number;
  read(start: number, end: number): Promise<ArrayBuffer>;
}

export function fileSource(blob: Blob): ByteSource {
  return {
    size: blob.size,
    read: (start, end) => {
      const part = blob.slice(start, end);
      if (typeof part.arrayBuffer === 'function') return part.arrayBuffer();
      return new Promise<ArrayBuffer>((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result as ArrayBuffer);
        r.onerror = () => reject(r.error ?? new Error('read failed'));
        r.readAsArrayBuffer(part);
      });
    },
  };
}

/** The video track's samples in decode order, as typed arrays (a one-hour phone video has about 108 000). Times are file time, in seconds. */
export interface SampleTable {
  n: number;
  /** presentation time, minus the container start (file time) */
  pts: Float64Array;
  /** decode time on the same clock */
  dts: Float64Array;
  /** byte offset in the file */
  offset: Float64Array;
  size: Uint32Array;
  sync: Uint8Array;
  /** sample indices sorted by pts (presentation order) */
  order: Uint32Array;
  /** the media-timescale cts of each sample, for EncodedVideoChunk timestamps (microseconds) */
  tsUs: Float64Array;
}

export interface VideoTrackInfo {
  trackId: number;
  /** the MIME codecs string WebCodecs wants, e.g. hvc1.1.6.L93.90 */
  codecString: string;
  fourcc: string;
  /** hvcC / avcC payload (without the box header), the VideoDecoder `description`; null for AV1 and VP9, which take none */
  description: Uint8Array | null;
  /** stored size, before rotation */
  width: number;
  height: number;
  timescale: number;
}

export interface Demuxed {
  /** ffprobe's codec_name: hevc, h264, ... */
  codec: string;
  profile: string | null;
  video: VideoTrackInfo;
  samples: SampleTable;
  /** sync samples, sorted, file time */
  keyframes: number[];
  /** ffprobe's rotation (degrees counter-clockwise as the display matrix says), 0 when none */
  rotation: number;
  mirrored: boolean;
  /** sample aspect ratio, 1 when square or unknown */
  sar: number;
  /** what the stream says about its colours (colour.ts), for converting a decoded frame the way the Mac does */
  colour: StreamColour;
  fps: number;
  captureFps: number;
  duration: number;
  /** where the video stream starts in file time (> 0 when the audio, say, starts first) */
  offset: number;
  endTime: number;
  /** the container start time that file time counts from */
  formatStart: number;
  hasAudio: boolean;
  audio: { trackId: number; codecString: string; sampleRate: number; channels: number; start: number; timescale: number } | null;
  warnings: string[];
  stats: DemuxStats;
}

export interface DemuxStats {
  chunkBytes: number;
  reads: number;
  bytesRead: number;
  /** most bytes mp4box held (what it kept plus the slice being parsed) at any one time */
  peakHeldBytes: number;
}

const FF_CODEC: Record<string, string> = { hvc1: 'hevc', hev1: 'hevc', avc1: 'h264', avc3: 'h264', av01: 'av1', vp09: 'vp9', mp4v: 'mpeg4', apch: 'prores', apcn: 'prores', apcs: 'prores', apco: 'prores', ap4h: 'prores', ap4x: 'prores' };
const H264_PROFILE: Record<number, string> = { 66: 'Baseline', 77: 'Main', 88: 'Extended', 100: 'High', 110: 'High 10', 122: 'High 4:2:2', 244: 'High 4:4:4 Predictive' };
const HEVC_PROFILE: Record<number, string> = { 1: 'Main', 2: 'Main 10', 3: 'Main Still Picture', 4: 'Rext' };

const fail = (status: number, msg: string): never => {
  throw new EngineError(status, msg);
};

/** ffmpeg's av_display_rotation_get on a 3x3 tkhd matrix (16.16 fixed point), rounded like probe.parse_rotation. */
export function rotationOf(m: ArrayLike<number>): number {
  const s0 = Math.hypot(m[0], m[3]);
  const s1 = Math.hypot(m[1], m[4]);
  if (!s0 || !s1) return 0;
  const r = Math.round(-(Math.atan2(m[1] / s1, m[0] / s0) * 180) / Math.PI);
  return r === 0 ? 0 : r; // no -0
}

/** probe.parse_mirrored: the determinant of the matrix's 2x2 part is negative. */
export const mirroredOf = (m: ArrayLike<number>): boolean => m[0] * m[4] - m[1] * m[3] < 0;

interface Edit {
  segment_duration: number;
  media_time: number;
}

/** What the edit list does to a track: a leading empty edit delays it (`empty`, seconds), the first real edit starts at `mediaTime`
 *  (media timescale) so earlier samples (the B-frame delay, AAC priming) are not shown, and `duration` is the sum of the real edits. */
export function editShift(edits: Edit[] | undefined, movieTs: number, mediaDurationS: number): { empty: number; mediaTime: number; duration: number } {
  let empty = 0;
  let mediaTime = 0;
  let duration = 0;
  let seenReal = false;
  for (const e of edits ?? []) {
    const isEmpty = e.media_time < 0 || e.media_time === 0xffffffff;
    if (isEmpty && !seenReal) {
      empty += e.segment_duration / movieTs;
    } else if (!isEmpty) {
      if (!seenReal) mediaTime = e.media_time;
      seenReal = true;
      duration += e.segment_duration === 0 ? mediaDurationS : e.segment_duration / movieTs;
    }
  }
  return { empty, mediaTime, duration: seenReal ? duration : mediaDurationS };
}

function boxBytes(box: any): Uint8Array | null {
  if (!box) return null;
  const ds = new DataStream(undefined, 0, Endianness.BIG_ENDIAN);
  box.write(ds);
  return new Uint8Array(ds.buffer as ArrayBuffer, 8, ds.getPosition() - 8);
}

/** Feed `src` to mp4box in `chunkBytes` slices until the movie header (moov) is parsed. Used by `demux` and by the export (mux.ts), which
 *  needs the raw sample tables of the audio and the sample entries too. */
export async function openMp4(src: ByteSource, opts: { chunkBytes?: number; signal?: AbortSignal } = {}): Promise<{ file: ReturnType<typeof createFile>; stats: DemuxStats }> {
  const chunk = opts.chunkBytes ?? CHUNK_BYTES;
  const stats: DemuxStats = { chunkBytes: chunk, reads: 0, bytesRead: 0, peakHeldBytes: 0 };
  const file = createFile();
  let ready = false;
  let error: string | null = null;
  file.onReady = () => {
    ready = true;
  };
  file.onError = (m: unknown) => {
    error = String(m);
  };
  const held = () => ((file as any).stream?.buffers ?? []).reduce((s: number, b: ArrayBuffer) => s + b.byteLength, 0);

  let pos = 0;
  let steps = 0;
  const maxSteps = Math.ceil(src.size / chunk) * 3 + 8;
  while (!ready && !error) {
    if (opts.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    if (pos >= src.size || ++steps > maxSteps) fail(422, 'This file has no movie header (moov), so it cannot be read. Is it a complete MP4 or MOV?');
    const end = Math.min(src.size, pos + chunk);
    const ab = await src.read(pos, end);
    stats.reads++;
    stats.bytesRead += ab.byteLength;
    const before = held();
    const next = file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(ab, pos), end >= src.size);
    stats.peakHeldBytes = Math.max(stats.peakHeldBytes, before + ab.byteLength);
    if (ready || error) break;
    if (next === undefined || next < 0) fail(422, 'This file could not be read as MP4 or MOV.');
    pos = next as number;
  }
  if (error) fail(422, `This file could not be read as MP4 or MOV (${error}).`);
  stats.peakHeldBytes = Math.max(stats.peakHeldBytes, held());
  return { file, stats };
}

export async function demux(src: ByteSource, opts: { chunkBytes?: number; signal?: AbortSignal } = {}): Promise<Demuxed> {
  const { file, stats } = await openMp4(src, opts);

  const info = file.getInfo();
  if (info.isFragmented) fail(415, 'Fragmented MP4 files are not supported in this browser yet. Use My Mac.');
  const movieTs = info.timescale || 1000;
  const traks = (info.tracks as any[]).map((t) => ({ info: t, trak: file.getTrackById(t.id) as any }));
  const handler = (t: any): string => t.trak?.mdia?.hdlr?.handler ?? '';
  const v = traks.find((t) => handler(t) === 'vide' || t.info.type === 'video' || t.info.video);
  if (!v) return fail(422, 'No video track in this file.');
  const a = traks.find((t) => handler(t) === 'soun' || t.info.type === 'audio' || t.info.audio);
  const warnings: string[] = [];

  const trak = v.trak;
  const ts: number = v.info.timescale;
  const raw: any[] = trak.samples ?? [];
  if (!raw.length) fail(422, 'The video track has no samples.');
  const shift = editShift(trak.edts?.elst?.entries, movieTs, (trak.mdia?.mdhd?.duration ?? v.info.duration) / ts);
  if ((trak.edts?.elst?.entries ?? []).filter((e: Edit) => e.media_time >= 0 && e.media_time !== 0xffffffff).length > 1) warnings.push('The video has more than one edit; only the first is applied.');

  // the container start: the earliest first picture of the video and the audio (ffmpeg's format start_time)
  const startOf = (t: any, shiftT: { empty: number; mediaTime: number }): number => {
    const s: any[] = t.trak.samples ?? [];
    let min = Infinity;
    for (const x of s) if (x.cts >= shiftT.mediaTime) min = Math.min(min, (x.cts - shiftT.mediaTime) / x.timescale + shiftT.empty);
    return Number.isFinite(min) ? min : shiftT.empty;
  };
  const videoStart = startOf(v, { empty: shift.empty, mediaTime: shift.mediaTime });
  let audioInfo: Demuxed['audio'] = null;
  let formatStart = videoStart;
  if (a) {
    const ash = editShift(a.trak.edts?.elst?.entries, movieTs, (a.trak.mdia?.mdhd?.duration ?? a.info.duration) / a.info.timescale);
    const astart = startOf(a, ash);
    formatStart = Math.min(videoStart, astart);
    audioInfo = { trackId: a.info.id, codecString: a.info.codec, sampleRate: a.info.audio?.sample_rate ?? 0, channels: a.info.audio?.channel_count ?? 0, start: astart - formatStart, timescale: a.info.timescale };
  }

  const n = raw.length;
  const table: SampleTable = { n, pts: new Float64Array(n), dts: new Float64Array(n), offset: new Float64Array(n), size: new Uint32Array(n), sync: new Uint8Array(n), order: new Uint32Array(n), tsUs: new Float64Array(n) };
  const keyframes: number[] = [];
  const durCount = new Map<number, number>();
  let durSum = 0;
  for (let i = 0; i < n; i++) {
    const s = raw[i];
    const pts = (s.cts - shift.mediaTime) / ts + shift.empty - formatStart;
    table.pts[i] = pts;
    table.dts[i] = (s.dts - shift.mediaTime) / ts + shift.empty - formatStart;
    table.offset[i] = s.offset;
    table.size[i] = s.size;
    table.sync[i] = s.is_sync ? 1 : 0;
    table.tsUs[i] = Math.round((s.cts / ts) * 1e6);
    durSum += s.duration;
    durCount.set(s.duration, (durCount.get(s.duration) ?? 0) + 1);
    if (s.is_sync && s.cts >= shift.mediaTime) keyframes.push(pts);
  }
  keyframes.sort((x, y) => x - y);
  table.order = Uint32Array.from({ length: n }, (_, i) => i).sort((x, y) => table.pts[x] - table.pts[y] || x - y);

  const entry = trak.mdia.minf.stbl.stsd.entries[0];
  const fourcc: string = entry.type ?? entry.box_name ?? '';
  // WebCodecs wants a description only for avcC and hvcC; AV1 and VP9 carry their config in the codec string (and the stream), and
  // mp4box has no write() for av1C / vpcC (it would give an empty array or just the FullBox header), so send none for them.
  const description = boxBytes(entry.hvcC ?? entry.avcC ?? null);
  const codec = FF_CODEC[fourcc] ?? fourcc;
  const profile = entry.avcC ? (H264_PROFILE[entry.avcC.AVCProfileIndication] ?? null) : entry.hvcC ? (HEVC_PROFILE[entry.hvcC.general_profile_idc] ?? null) : null;
  const colr = entry.colr?.colour_type === 'nclx' ? { matrix_coefficients: entry.colr.matrix_coefficients as number, full_range: entry.colr.full_range_flag === 1 } : null;
  const colour = streamColour(codec, description, colr);
  const pasp = entry.pasp;
  const sar = pasp && pasp.hSpacing > 0 && pasp.vSpacing > 0 ? pasp.hSpacing / pasp.vSpacing : 1;

  const matrix = trak.tkhd?.matrix ?? [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824];
  const mediaS = durSum / ts;
  const fps = mediaS > 0 ? n / mediaS : 0;
  let common = 0;
  let commonN = 0;
  for (const [d, c] of durCount) if (c > commonN || (c === commonN && d < common)) [common, commonN] = [d, c];
  const rFps = common > 0 ? ts / common : fps;
  const captureFps = fps > 0 && rFps > 2 * fps ? fps : rFps; // probe.capture_fps: an irregular file's average is the honest rate
  const duration = shift.duration;
  const offset = videoStart - formatStart;

  return {
    codec, profile,
    video: { trackId: v.info.id, codecString: v.info.codec, fourcc, description, width: v.info.video?.width ?? entry.width, height: v.info.video?.height ?? entry.height, timescale: ts },
    samples: table, keyframes,
    rotation: rotationOf(matrix), mirrored: mirroredOf(matrix), sar, colour, fps, captureFps,
    duration, offset, endTime: offset + duration, formatStart,
    hasAudio: !!a, audio: audioInfo, warnings, stats,
  };
}
