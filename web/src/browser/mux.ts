// Lossless clip cutting in the page (docs/T9B_DESIGN.md section 6): our own small ISO-BMFF writer. mp4box only parses; every byte of a clip's
// moov is written here, and the picture and sound bytes are copied from the source file untouched (nothing is decoded or re-encoded).
//
// What a clip is, to match `inframe cut` (src/inframe/cut.py: `-ss` before `-i`, then the align pass):
//  - video starts at the sync sample at or before the requested start (cut.snap_to_keyframe), ends at the last sample whose decode time is
//    before the end (ffmpeg's `-t` test is on dts), and carries the source's display matrix and its sample entry (hvcC / avcC, colr, pasp) byte for byte;
//  - the pictures that follow the keyframe in decode order but are shown before it (RASL, open GOP) cannot be decoded without the previous
//    GOP and are dropped. The decode times are packed again, the presentation times are not touched, and the edit list starts at the
//    keyframe's presentation time, so the first picture a player shows is the keyframe at time 0;
//  - AAC audio starts with the first packet within half an AAC frame of that picture (the cut point sits half a frame early) and an edit
//    list delays or trims it by the exact remainder. Other audio codecs are dropped with a warning. Every other track, container tags and
//    the location data a phone writes are never read, so none of it can get into a clip.
import { EngineError } from '../engine/types';
import { CHUNK_BYTES, editShift, openMp4, type ByteSource, type Demuxed } from './demux';

/** The movie timescale of every clip (ffmpeg's mov muxer uses 1000 too). */
export const MOVIE_TS = 1000;
/** Samples are written in chunks of about this many seconds per track, video and audio taking turns. */
export const CHUNK_S = 1;
const VIDEO_ENTRIES = new Set(['hvc1', 'hev1', 'avc1', 'avc3']);

/** Why a video whose sample entry is `fourcc` cannot be cut by stream copy here, or null when it can. Needs no read of the file, so it can run before anything is asked of the user. */
export function cutRefusal(fourcc: string): EngineError | null {
  return VIDEO_ENTRIES.has(fourcc) ? null : new EngineError(415, `A lossless cut of ${fourcc || 'this'} video is not possible in the browser. Use My Mac.`);
}
const EPS = 1e-9;

/** One track of the source as the writer needs it: the raw sample table in the media's own timescale, the sample entry bytes and the tkhd facts. */
export interface MuxTrack {
  kind: 'video' | 'audio';
  timescale: number;
  n: number;
  /** decode and composition time in media timescale units, as the container holds them */
  dts: Float64Array;
  cts: Float64Array;
  dur: Float64Array;
  size: Uint32Array;
  offset: Float64Array;
  sync: Uint8Array;
  /** presentation and decode time in file time (seconds, the clock Demuxed uses) */
  pts: Float64Array;
  dtsS: Float64Array;
  /** the sample entry (hvc1, avc1, mp4a, ...) with its box header, copied byte for byte */
  entry: Uint8Array;
  fourcc: string;
  tkhd: { flags: number; layer: number; altGroup: number; volume: number; width: number; height: number; matrix: number[] };
  language: number;
}

export interface MuxSource {
  video: MuxTrack;
  audio: MuxTrack | null;
  /** why the audio was left out (not AAC), if it was */
  warnings: string[];
  /** the source file's container brand family: 'qt' for .mov, else 'isom' */
  brand: 'qt' | 'isom';
}

const trackOf = async (src: ByteSource, kind: 'video' | 'audio', trak: any, d: Demuxed, movieTs: number): Promise<MuxTrack> => {
  const raw: any[] = trak.samples ?? [];
  const n = raw.length;
  const ts: number = trak.mdia.mdhd.timescale;
  const shift = editShift(trak.edts?.elst?.entries, movieTs, (trak.mdia?.mdhd?.duration ?? 0) / ts);
  const t: MuxTrack = {
    kind, timescale: ts, n, dts: new Float64Array(n), cts: new Float64Array(n), dur: new Float64Array(n), size: new Uint32Array(n), offset: new Float64Array(n),
    sync: new Uint8Array(n), pts: new Float64Array(n), dtsS: new Float64Array(n), entry: new Uint8Array(0), fourcc: '',
    tkhd: {
      flags: trak.tkhd?.flags ?? 3, layer: trak.tkhd?.layer ?? 0, altGroup: trak.tkhd?.alternate_group ?? 0, volume: trak.tkhd?.volume ?? (kind === 'audio' ? 0x100 : 0),
      width: trak.tkhd?.width ?? 0, height: trak.tkhd?.height ?? 0, matrix: Array.from(trak.tkhd?.matrix ?? [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824]),
    },
    language: trak.mdia?.mdhd?.language ?? 0x55c4,
  };
  for (let i = 0; i < n; i++) {
    const s = raw[i];
    t.dts[i] = s.dts;
    t.cts[i] = s.cts;
    t.dur[i] = s.duration;
    t.size[i] = s.size;
    t.offset[i] = s.offset;
    t.sync[i] = s.is_sync ? 1 : 0;
    t.pts[i] = (s.cts - shift.mediaTime) / ts + shift.empty - d.formatStart;
    t.dtsS[i] = (s.dts - shift.mediaTime) / ts + shift.empty - d.formatStart;
  }
  // The sync flags stay as the file has them: a keyframe below the first edit's media_time (a source trimmed by passthrough) is decode-only, yet
  // it is where a lossless cut must start (cut.snap_to_keyframe); the output's own edit list hides what is not wanted.
  const entry = trak.mdia.minf.stbl.stsd.entries[0];
  t.fourcc = String(entry.type ?? entry.box_name ?? '');
  const bytes = new Uint8Array(await src.read(entry.start, entry.start + entry.size));
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const type = String.fromCharCode(bytes[4], bytes[5], bytes[6], bytes[7]);
  if (bytes.length !== entry.size || dv.getUint32(0) !== entry.size || (t.fourcc && type !== t.fourcc)) {
    throw new EngineError(422, 'The sample description of this file could not be read back.');
  }
  t.entry = bytes;
  return t;
};

/** Read what the export needs from the source (a few slices again: the movie header, then the two sample entries). `d` is the open video's Demuxed. */
export async function readMuxSource(src: ByteSource, d: Demuxed, name = ''): Promise<MuxSource> {
  const { file } = await openMp4(src);
  const info = file.getInfo();
  const movieTs = info.timescale || 1000;
  const vt = (info.tracks as any[]).find((t) => t.id === d.video.trackId);
  if (!vt) throw new EngineError(422, 'No video track in this file.');
  const video = await trackOf(src, 'video', file.getTrackById(vt.id), d, movieTs);
  const refusal = cutRefusal(video.fourcc);
  if (refusal) throw refusal;
  const warnings: string[] = [];
  let audio: MuxTrack | null = null;
  if (d.audio) {
    // the first AAC track wins: a phone's first audio track can be something else with an AAC one after it
    const audios = (info.tracks as any[]).filter((t) => t.audio || t.type === 'audio' || t.id === d.audio!.trackId);
    const at = audios.find((t) => String(t.codec).startsWith('mp4a.40.'));
    if (at) audio = await trackOf(src, 'audio', file.getTrackById(at.id), d, movieTs);
    else warnings.push(`The sound (${audios[0]?.codec ?? d.audio.codecString ?? 'unknown'}) is not AAC, so the clips have no sound. Use My Mac to keep it.`);
  }
  return { video, audio, warnings, brand: /\.mov$/i.test(name) ? 'qt' : 'isom' };
}

// ---------------------------------------------------------------- the plan of one clip
export interface ClipPlan {
  /** the requested stretch, file time */
  start: number;
  end: number;
  /** the sync sample the cut starts at */
  key: number;
  /** the video samples that are kept, in decode order (the keyframe first) */
  kept: Uint32Array;
  /** how many pictures the source had between the keyframe and the end that were dropped (RASL) */
  dropped: number;
  /** file time of the keyframe's presentation: where the clip really starts (cut.snap_to_keyframe) */
  copyStart: number;
  /** pictures a player shows, and how long they last */
  frames: number;
  shownS: number;
  audio: { first: number; last: number } | null;
  /** first audio packet's presentation time minus the first picture's (seconds, within half an AAC frame) */
  audioShift: number;
}

/** cut.snap_to_keyframe over the sync samples: the last one at or before `t`, else the first. */
export function keyframeIndex(v: MuxTrack, t: number): number {
  let best = -1;
  let first = -1;
  let bestPts = -Infinity;
  let firstPts = Infinity;
  for (let i = 0; i < v.n; i++) {
    if (!v.sync[i]) continue;
    const p = v.pts[i];
    if (p < firstPts) [first, firstPts] = [i, p];
    if (p <= t + 1e-6 && p > bestPts) [best, bestPts] = [i, p];
  }
  if (first < 0) throw new EngineError(422, 'The video has no keyframe, so it cannot be cut without re-encoding. Use My Mac.');
  return best >= 0 ? best : first;
}

export function planClip(m: MuxSource, start: number, end: number): ClipPlan {
  const v = m.video;
  if (!(end > start)) throw new EngineError(422, 'The clip has no length.');
  // a range past the last picture would otherwise snap to the last keyframe and export that GOP as if it were the clip
  let videoEnd = 0;
  for (let i = 0; i < v.n; i++) videoEnd = Math.max(videoEnd, v.pts[i] + v.dur[i] / v.timescale);
  if (start >= videoEnd - 1e-6) throw new EngineError(422, 'The clip starts after the end of the video, so there is nothing to cut.');
  const key = keyframeIndex(v, start);
  const kept: number[] = [];
  let dropped = 0;
  for (let j = key; j < v.n; j++) {
    if (j > key) {
      if (v.dtsS[j] >= end - EPS) break; // ffmpeg's -t test is on the decode time
      if (v.cts[j] < v.cts[key]) {
        dropped++; // shown before the keyframe: a leading picture that needs the previous GOP
        continue;
      }
    }
    kept.push(j);
  }
  let maxEnd = 0;
  let total = 0;
  for (const j of kept) {
    maxEnd = Math.max(maxEnd, v.cts[j] + v.dur[j]);
    total += v.dur[j];
  }
  // the last pictures can be decoded before the end yet shown after it (B-frames), leaving a hole before a lone picture; as in the Python cut,
  // that picture stays, but the clip lasts as long as the pictures it holds (ffprobe clamps the edit to this too)
  const shownUnits = Math.min(maxEnd - v.cts[key], total);
  const copyStart = v.pts[key];
  const plan: ClipPlan = { start, end, key, kept: Uint32Array.from(kept), dropped, copyStart, frames: kept.length, shownS: shownUnits / v.timescale, audio: null, audioShift: 0 };
  const a = m.audio;
  if (a && a.n) {
    let first = -1;
    for (let i = 0; i < a.n; i++) {
      if (a.pts[i] >= copyStart - a.dur[i] / a.timescale / 2 - EPS) {
        first = i;
        break;
      }
    }
    let last = -1;
    for (let i = Math.max(first, 0); i < a.n && first >= 0; i++) {
      if (a.dtsS[i] >= end - EPS) break;
      last = i;
    }
    if (first >= 0 && last >= first) {
      plan.audio = { first, last };
      plan.audioShift = a.pts[first] - copyStart;
    }
  }
  return plan;
}

// ---------------------------------------------------------------- bytes
const enc = new TextEncoder();
const cat = (parts: Uint8Array[]): Uint8Array => {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
};
/** big-endian 32-bit words (negative numbers wrap to their two's complement) */
const words = (...v: number[]): Uint8Array => {
  const out = new Uint8Array(v.length * 4);
  const dv = new DataView(out.buffer);
  v.forEach((x, i) => dv.setUint32(i * 4, x >>> 0));
  return out;
};
const u16 = (v: number): Uint8Array => Uint8Array.of((v >> 8) & 255, v & 255);
const zeros = (n: number): Uint8Array => new Uint8Array(n);
const fourcc = (s: string): Uint8Array => enc.encode(s.padEnd(4, ' ').slice(0, 4));
const box = (type: string, ...parts: Uint8Array[]): Uint8Array => {
  const body = cat(parts);
  return cat([words(body.length + 8), fourcc(type), body]);
};
const full = (type: string, version: number, flags: number, ...parts: Uint8Array[]): Uint8Array => box(type, words(((version & 255) << 24) | (flags & 0xffffff)), ...parts);
const table = (a: ArrayLike<number>): Uint8Array => {
  const out = new Uint8Array(a.length * 4);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < a.length; i++) dv.setUint32(i * 4, a[i] >>> 0);
  return out;
};
const table64 = (a: ArrayLike<number>): Uint8Array => {
  const out = new Uint8Array(a.length * 8);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < a.length; i++) dv.setBigUint64(i * 8, BigInt(a[i]));
  return out;
};
/** run-length pairs [count, value, count, value, ...] */
const runs = (a: ArrayLike<number>): number[] => {
  const out: number[] = [];
  for (let i = 0; i < a.length; i++) {
    if (out.length && out[out.length - 1] === a[i]) out[out.length - 2]++;
    else out.push(1, a[i]);
  }
  return out;
};

interface Edit {
  /** movie timescale */
  segment: number;
  /** media timescale, -1 for an empty edit */
  mediaTime: number;
}

/** One track of a clip: what goes into its moov and where its samples go. */
interface OutTrack {
  src: MuxTrack;
  id: number;
  idx: Uint32Array;
  /** output decode time of each sample and its composition offset (media units) */
  dts: Float64Array;
  ctsOff: Float64Array;
  mediaDuration: number;
  edits: Edit[];
  movieDuration: number;
  /** chunk table: [first sample, count] and the byte offset of each chunk in the mdat payload (set at layout) */
  chunks: { first: number; count: number; bytes: number; at: number; t: number }[];
}

function outVideo(m: MuxSource, p: ClipPlan): OutTrack {
  const v = m.video;
  const n = p.kept.length;
  const dts = new Float64Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    dts[i] = acc;
    acc += v.dur[p.kept[i]];
  }
  const rel = new Float64Array(n);
  let delay = 0;
  for (let i = 0; i < n; i++) {
    rel[i] = v.cts[p.kept[i]] - v.cts[p.key];
    delay = Math.max(delay, dts[i] - rel[i]); // the decode times are packed, so the first picture waits this long
  }
  const ctsOff = new Float64Array(n);
  for (let i = 0; i < n; i++) ctsOff[i] = rel[i] + delay - dts[i];
  const shown = Math.round((p.shownS * MOVIE_TS));
  return { src: v, id: 1, idx: p.kept, dts, ctsOff, mediaDuration: acc, edits: [{ segment: shown, mediaTime: delay }], movieDuration: shown, chunks: [] };
}

function outAudio(m: MuxSource, p: ClipPlan): OutTrack | null {
  const a = m.audio;
  if (!a || !p.audio) return null;
  const n = p.audio.last - p.audio.first + 1;
  const idx = new Uint32Array(n);
  const dts = new Float64Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    idx[i] = p.audio.first + i;
    dts[i] = acc;
    acc += a.dur[idx[i]];
  }
  const total = Math.round((acc / a.timescale) * MOVIE_TS);
  const edits: Edit[] = [];
  let movieDuration: number;
  if (p.audioShift >= 0) {
    const gap = Math.round(p.audioShift * MOVIE_TS);
    if (gap > 0) edits.push({ segment: gap, mediaTime: -1 });
    edits.push({ segment: total, mediaTime: 0 });
    movieDuration = gap + total;
  } else {
    const skip = Math.min(acc, Math.round(-p.audioShift * a.timescale));
    const seg = Math.round(((acc - skip) / a.timescale) * MOVIE_TS);
    edits.push({ segment: seg, mediaTime: skip });
    movieDuration = seg;
  }
  return { src: a, id: 2, idx, dts, ctsOff: new Float64Array(n), mediaDuration: acc, edits, movieDuration, chunks: [] };
}

function planChunks(t: OutTrack, startOffsetS: number): void {
  const ts = t.src.timescale;
  let first = 0;
  let bytes = 0;
  const n = t.idx.length;
  for (let i = 0; i < n; i++) {
    if (i > first && t.dts[i] - t.dts[first] >= CHUNK_S * ts) {
      t.chunks.push({ first, count: i - first, bytes, at: 0, t: startOffsetS + t.dts[first] / ts });
      first = i;
      bytes = 0;
    }
    bytes += t.src.size[t.idx[i]];
  }
  if (n > first) t.chunks.push({ first, count: n - first, bytes, at: 0, t: startOffsetS + t.dts[first] / ts });
}

const matrixBytes = (m: number[]): Uint8Array => words(...m);
const IDENTITY = [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824];

function stbl(t: OutTrack): Uint8Array {
  const n = t.idx.length;
  const durs = new Float64Array(n);
  for (let i = 0; i < n; i++) durs[i] = t.src.dur[t.idx[i]];
  const parts: Uint8Array[] = [full('stsd', 0, 0, words(1), t.src.entry)];
  const stts = runs(durs);
  parts.push(full('stts', 0, 0, words(stts.length / 2), table(stts)));
  if (t.src.kind === 'video') {
    if (t.ctsOff.some((x) => x !== 0)) {
      const c = runs(t.ctsOff);
      parts.push(full('ctts', 0, 0, words(c.length / 2), table(c)));
    }
    const sync: number[] = [];
    for (let i = 0; i < n; i++) if (t.src.sync[t.idx[i]]) sync.push(i + 1);
    if (sync.length !== n) parts.push(full('stss', 0, 0, words(sync.length), table(sync)));
  }
  parts.push(full('stsz', 0, 0, words(0, n), table(Array.from(t.idx, (j) => t.src.size[j]))));
  const stsc: number[] = [];
  t.chunks.forEach((c, i) => {
    if (!stsc.length || stsc[stsc.length - 2] !== c.count) stsc.push(i + 1, c.count, 1);
  });
  // the entries are [first_chunk, samples_per_chunk, description]; a run only starts where the count changes
  parts.push(full('stsc', 0, 0, words(stsc.length / 3), table(stsc)));
  parts.push(full('co64', 0, 0, words(t.chunks.length), table64(t.chunks.map((c) => c.at))));
  return box('stbl', ...parts);
}

function trak(t: OutTrack): Uint8Array {
  const k = t.src;
  const isV = k.kind === 'video';
  const tkhd = full('tkhd', 0, k.tkhd.flags, words(0, 0, t.id, 0, t.movieDuration), zeros(8), u16(k.tkhd.layer), u16(k.tkhd.altGroup), u16(isV ? 0 : 0x100), zeros(2),
    matrixBytes(isV ? k.tkhd.matrix : IDENTITY), words(isV ? k.tkhd.width : 0, isV ? k.tkhd.height : 0));
  const elst = full('elst', 0, 0, words(t.edits.length), ...t.edits.map((e) => words(e.segment, e.mediaTime < 0 ? 0xffffffff : e.mediaTime, 0x00010000)));
  const mdhd = full('mdhd', 0, 0, words(0, 0, k.timescale, t.mediaDuration), u16(k.language), u16(0));
  const hdlr = full('hdlr', 0, 0, words(0), fourcc(isV ? 'vide' : 'soun'), zeros(12), enc.encode(isV ? 'VideoHandler\0' : 'SoundHandler\0'));
  const head = isV ? full('vmhd', 0, 1, zeros(8)) : full('smhd', 0, 0, zeros(4));
  const dinf = box('dinf', full('dref', 0, 0, words(1), full('url ', 0, 1)));
  return box('trak', tkhd, box('edts', elst), box('mdia', mdhd, hdlr, box('minf', head, dinf, stbl(t))));
}

export interface MuxedClip {
  /** ftyp, moov and the mdat header */
  head: Uint8Array;
  /** byte ranges of the source file that follow the head, in order (the mdat payload) */
  ranges: [number, number][];
  size: number;
  /** pictures a player shows and the seconds they last */
  frames: number;
  shownS: number;
  audioPackets: number;
  /** where the clip really starts, file time */
  copyStart: number;
  dropped: number;
}

/** Build the clip of `plan`: its head (ftyp, moov, mdat header) and the source ranges its mdat is made of. */
export function muxClip(m: MuxSource, plan: ClipPlan): MuxedClip {
  const v = outVideo(m, plan);
  const a = outAudio(m, plan);
  const tracks = a ? [v, a] : [v];
  planChunks(v, 0);
  if (a) planChunks(a, plan.audioShift); // the audio's own start decides where its chunks sit among the picture's
  // layout: chunks by start time (the picture first on a tie)
  const order: { t: OutTrack; c: OutTrack['chunks'][number]; k: number }[] = [];
  tracks.forEach((t, k) => t.chunks.forEach((c) => order.push({ t, c, k })));
  order.sort((x, y) => x.c.t - y.c.t || x.k - y.k);
  let at = 0;
  for (const o of order) {
    o.c.at = at;
    at += o.c.bytes;
  }
  const payload = at;
  const movieDuration = Math.max(...tracks.map((t) => t.movieDuration));
  const mvhd = full('mvhd', 0, 0, words(0, 0, MOVIE_TS, movieDuration, 0x00010000), u16(0x0100), zeros(2), zeros(8), matrixBytes(IDENTITY), zeros(24), words(tracks.length + 1));
  const moov = box('moov', mvhd, ...tracks.map(trak));
  const ftyp = m.brand === 'qt' ? box('ftyp', fourcc('qt  '), words(0), fourcc('qt  ')) : box('ftyp', fourcc('isom'), words(512), fourcc('isom'), fourcc('iso2'), fourcc('mp41'));
  const wide = payload + 8 > 0xffffffff;
  const mdatHead = wide ? cat([words(1), fourcc('mdat'), (() => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(payload + 16)); return b; })()]) : cat([words(payload + 8), fourcc('mdat')]);
  const base = ftyp.length + moov.length + mdatHead.length;
  // the co64 entries were written as offsets into the payload; moving them by the head's size is a second pass over the same boxes
  const shifted = shiftOffsets(moov, base);
  const head = cat([ftyp, shifted, mdatHead]);

  const ranges: [number, number][] = [];
  const push = (s: number, e: number) => {
    const last = ranges[ranges.length - 1];
    if (last && last[1] === s) last[1] = e;
    else ranges.push([s, e]);
  };
  for (const o of order) {
    for (let i = o.c.first; i < o.c.first + o.c.count; i++) {
      const j = o.t.idx[i];
      push(o.t.src.offset[j], o.t.src.offset[j] + o.t.src.size[j]);
    }
  }
  return { head, ranges, size: head.length + payload, frames: plan.frames, shownS: plan.shownS, audioPackets: a ? a.idx.length : 0, copyStart: plan.copyStart, dropped: plan.dropped };
}

/** Add `by` to every co64 offset of a moov (the head's size is only known once the moov is built; co64 entries have a fixed size). */
function shiftOffsets(moov: Uint8Array, by: number): Uint8Array {
  const out = moov.slice();
  const dv = new DataView(out.buffer);
  const walk = (start: number, end: number): void => {
    let p = start;
    while (p + 8 <= end) {
      const size = dv.getUint32(p);
      const type = String.fromCharCode(out[p + 4], out[p + 5], out[p + 6], out[p + 7]);
      if (size < 8) break;
      if (type === 'co64') {
        const n = dv.getUint32(p + 12);
        for (let i = 0; i < n; i++) {
          const q = p + 16 + i * 8;
          dv.setBigUint64(q, dv.getBigUint64(q) + BigInt(by));
        }
      } else if (type === 'moov' || type === 'trak' || type === 'mdia' || type === 'minf' || type === 'stbl') walk(p + 8, p + size);
      p += size;
    }
  };
  walk(0, out.length);
  return out;
}

/** The clip as a stream: the head, then the source ranges read in slices of at most CHUNK_BYTES, one at a time as the reader asks. */
export function clipStream(src: ByteSource, clip: MuxedClip, signal?: AbortSignal): ReadableStream<Uint8Array> {
  let r = 0;
  let pos = clip.ranges.length ? clip.ranges[0][0] : 0;
  let sent = false;
  return new ReadableStream<Uint8Array>({
    async pull(ctl) {
      if (signal?.aborted) return ctl.error(new DOMException('Aborted', 'AbortError'));
      if (!sent) {
        sent = true;
        return ctl.enqueue(clip.head);
      }
      if (r >= clip.ranges.length) return ctl.close();
      const [, end] = clip.ranges[r];
      const to = Math.min(end, pos + CHUNK_BYTES);
      const buf = await src.read(pos, to);
      if (buf.byteLength !== to - pos) return ctl.error(new EngineError(500, 'The video file could not be read back (it changed or moved).'));
      ctl.enqueue(new Uint8Array(buf));
      pos = to;
      if (pos >= end) {
        r++;
        if (r < clip.ranges.length) pos = clip.ranges[r][0];
      }
    },
  });
}

/** The clip's bytes in one piece (the tests, and small clips). */
export async function clipBytes(src: ByteSource, clip: MuxedClip): Promise<Uint8Array> {
  const parts: Uint8Array[] = [];
  const rd = clipStream(src, clip).getReader();
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    parts.push(value);
  }
  return cat(parts);
}
