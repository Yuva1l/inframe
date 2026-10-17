// Export in the page (docs/T9B_DESIGN.md section 6): the clips cut by mux.ts, a JSON sidecar per clip (docs/sidecar.schema.json, the same
// content as src/inframe/sidecar.py) and the LosslessCut CSV (cut.write_losslesscut_csv), streamed into one zip with client-zip (MIT, Zip64).
// The zip goes to a file chosen with showSaveFilePicker (or a folder the user grants) so memory stays bounded; a browser with neither
// (Safari) builds a Blob, which is capped at 2 GB, and says "use My Mac" above that.
import { makeZip } from 'client-zip';
import { EngineError, type MediaInfo, type Samples } from '../engine/types';
import { BLOB_CAP } from '../model/limits';
import { clipStream, muxClip, planClip, type ClipPlan, type MuxedClip, type MuxSource } from './mux';
import type { ByteSource } from './demux';

export const SCHEMA_VERSION = '1.2';
export const SIDECAR_SUFFIX = '.inframe.json';
export const INFRAME_VERSION = '0.1.0 (browser)';
/** Safari builds the zip in memory; WebKit's Blob limit makes anything near 2 GB unreliable. */
export const SAFARI_BLOB_CAP = BLOB_CAP;

export const TIME_BASE = 'source_time is the file\'s own timeline in seconds, the one `ffmpeg -ss` counts, and slow motion is NOT divided out (a baked slow-motion clip stays slow). '
  + 'clip_time = source_time - clip.copy_start_s, the time a player shows in the cut clip; the clip starts on the keyframe at copy_start_s. When the `slowmo` block is '
  + 'present the clip was retimed: the time a player shows is clip_time * slowmo.factor.';

// ---------------------------------------------------------------- numbers and names (the Python rules)
/** Python's round(x, d) for the values the sidecar holds: correctly rounded, a tie (only possible for odd multiples of 2^-(d+1)) to even. */
export function pyRound(x: number, d: number): number {
  if (!Number.isFinite(x)) return x;
  const k = x * 2 ** (d + 1);
  if (Number.isInteger(k) && Math.abs(k) % 2 === 1) {
    const y = x * 10 ** d; // exact: a half-integer
    const lo = Math.floor(y);
    return Number(((lo % 2 === 0 ? lo : lo + 1) / 10 ** d).toFixed(d));
  }
  return Number(x.toFixed(d));
}

/** Python's f"{x:.3f}" */
export function fmt3(x: number): string {
  const k = x * 16;
  if (Number.isInteger(k) && Math.abs(k) % 2 === 1) return pyRound(x, 3).toFixed(3);
  return x.toFixed(3);
}

/** cut.mmss */
export const mmss = (t: number): string => {
  const s = Math.trunc(t);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}${String(s % 60).padStart(2, '0')}`;
};

/** cut.clip_name: `IMG_0001_01_0018-0036.mov` */
export const clipName = (stem: string, index: number, start: number, end: number, suffix = '.mov'): string => `${stem}_${String(index).padStart(2, '0')}_${mmss(start)}-${mmss(end)}${suffix}`;
/** sidecar.sidecar_name */
export const sidecarName = (clip: string): string => `${clip.replace(/\.[^.]*$/, '')}${SIDECAR_SUFFIX}`;
export const csvName = (stem: string): string => `${stem}_losslesscut.csv`;
export const zipName = (stem: string): string => `${stem}_clips.zip`;

export const stemOf = (name: string): string => {
  const i = name.lastIndexOf('.');
  return i > 0 ? name.slice(0, i) : name;
};
export const suffixOf = (name: string): string => {
  const i = name.lastIndexOf('.');
  return i > 0 ? name.slice(i).toLowerCase() : '.mov';
};

const csvField = (s: string): string => (/[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);

/** cut.write_losslesscut_csv: start,end,label per row, seconds with 3 decimals, no header, csv.writer's \r\n line ends. */
export function losslessCutCsv(rows: readonly (readonly [number, number, string])[]): string {
  return rows.map(([s, e, label]) => `${fmt3(s)},${fmt3(e)},${csvField(label)}\r\n`).join('');
}

// ---------------------------------------------------------------- sidecars
const box4 = (b: readonly number[] | null): number[] | null => (b === null ? null : b.map((v) => pyRound(Math.min(1, Math.max(0, v)), 4)));

export interface SidecarInput {
  media: Pick<MediaInfo, 'name' | 'size' | 'duration' | 'fps' | 'rotation' | 'mirrored' | 'width' | 'height' | 'display_w' | 'display_h'>;
  hash: string;
  settings: Record<string, any>;
  samples: Samples | null;
}

/** sidecar.build_sidecar for one clip of the export. */
/** `ranges` are all the clips of the export: the presence stretches, cut to each clip. */
export function buildSidecar(i: SidecarInput, ranges: readonly (readonly [number, number])[], index: number, nClips: number, file: string, start: number, end: number, copyStart: number): Record<string, unknown> {
  const s = i.samples;
  const samples: Record<string, unknown>[] = [];
  const tids = new Set<number>();
  if (s) {
    for (let k = 0; k < s.t.length; k++) {
      const t = s.t[k];
      if (t < copyStart - 1e-9 || t > end + 1e-9) continue;
      const tid = s.track ? s.track[k] : null;
      const box = box4(s.box[k]);
      samples.push({ source_time: pyRound(t, 3), clip_time: pyRound(t - copyStart, 3), box, confidence: pyRound(s.score[k], 5), track_id: tid === null || tid === undefined ? null : Math.trunc(tid) });
      if (tid !== null && tid !== undefined && box !== null) tids.add(Math.trunc(tid));
    }
  }
  const inside = ranges.filter(([a, b]) => b > copyStart && a < end).map(([a, b]) => [pyRound(Math.max(a, copyStart), 3), pyRound(Math.min(b, end), 3)]);
  const st = i.settings;
  const mode = st.mode ?? 'detect';
  const m = i.media;
  return {
    schema_version: SCHEMA_VERSION,
    inframe_version: INFRAME_VERSION,
    source: { path: m.name, hash: { algorithm: 'sha256', value: i.hash, size_bytes: m.size }, duration_s: pyRound(m.duration, 3) },
    clip: { file, in_source_s: pyRound(start, 3), out_source_s: pyRound(end, 3), copy_start_s: pyRound(copyStart, 3), duration_s: pyRound(end - copyStart, 3) },
    try_number: index,
    tries_total: nClips,
    video: { fps: pyRound(m.fps, 4), rotation: Math.trunc(m.rotation), mirrored: !!m.mirrored, stored_width: m.width, stored_height: m.height, display_width: m.display_w, display_height: m.display_h },
    time_base: TIME_BASE,
    coordinate_frame: {
      description: 'Upright display frame: the stored picture with the display matrix (rotation, then any mirror) applied, as a player shows it.',
      box_format: 'xyxy', units: 'normalized', origin: 'top-left', x_axis: 'right', y_axis: 'down', scale_to_pixels: 'x * display_width, y * display_height',
    },
    target: { name: st.target ?? 'unknown' },
    embedder: st.target_info?.embedder ?? (st.tap ? 'dinov2-small' : null),
    detector: { mode, backend: st.detector ?? null, model: st.model ?? null, tracker: st.tracker ?? null },
    score: {
      kind: mode === 'motion' ? 'motion_foreground_fraction' : mode === 'detect' ? 'detector_confidence' : 'unknown', enter: st.enter ?? null, exit: st.exit ?? null,
      presence: 'a sample is in frame from the moment confidence >= enter until it stays below exit; compare confidence with these thresholds, never with a fixed number such as 0.5',
    },
    sampling: { fps: st.fps ?? null, n_samples: samples.length },
    presence_in_clip: { source_s: inside, clip_s: inside.map(([a, b]) => [pyRound(a - copyStart, 3), pyRound(b - copyStart, 3)]) },
    track_ids: [...tids].sort((a, b) => a - b),
    samples,
  };
}

// ---------------------------------------------------------------- the export
export interface ExportClip {
  start: number;
  end: number;
}

export interface ExportInput extends SidecarInput {
  clips: readonly ExportClip[];
  mux: MuxSource;
  src: ByteSource;
}

export interface PlannedClip {
  index: number;
  name: string;
  sidecar: string;
  start: number;
  end: number;
  plan: ClipPlan;
  muxed: MuxedClip;
  /** the sidecar's size in bytes (the same with the real hash: it has the same length) */
  sidecarBytes: number;
}

export interface ExportPlan {
  zip: string;
  clips: PlannedClip[];
  /** clips that could not be planned (no keyframe, ...): the reason; nothing is written for them */
  failed: { index: number; name: string; start: number; end: number; error: string }[];
  csv: string;
  csvName: string;
  /** bytes of everything in the zip but the zip's own headers */
  bytes: number;
  warnings: string[];
}

const enc = new TextEncoder();
const rangesOf = (i: ExportInput): [number, number][] => i.clips.map((c) => [c.start, c.end]);

export function planExport(i: ExportInput): ExportPlan {
  const stem = stemOf(i.media.name);
  const suffix = suffixOf(i.media.name);
  const clips: PlannedClip[] = [];
  const failed: ExportPlan['failed'] = [];
  const warnings = [...i.mux.warnings];
  const ranges = rangesOf(i);
  i.clips.forEach((c, k) => {
    const index = k + 1;
    const name = clipName(stem, index, c.start, c.end, suffix);
    try {
      const plan = planClip(i.mux, c.start, c.end);
      const side = JSON.stringify(buildSidecar(i, ranges, index, i.clips.length, name, c.start, c.end, pyRound(plan.copyStart, 3)), null, 1);
      clips.push({ index, name, sidecar: sidecarName(name), start: c.start, end: c.end, plan, muxed: muxClip(i.mux, plan), sidecarBytes: enc.encode(side).length });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      failed.push({ index, name, start: c.start, end: c.end, error: msg });
      warnings.push(`clip ${index} could not be cut: ${msg}`);
    }
  });
  const csv = losslessCutCsv(clips.map((c) => [c.start, c.end, c.name] as const));
  const bytes = clips.reduce((s, c) => s + c.muxed.size + c.sidecarBytes, 0) + csv.length;
  return { zip: zipName(stem), clips, failed, csv, csvName: csvName(stem), bytes, warnings };
}

/** The zip as a stream: each clip is read from the source as the zip asks for it. `onBytes` hears every piece of the clips. */
export function exportStream(p: ExportPlan, i: ExportInput, opts: { signal?: AbortSignal; now?: Date; onBytes?: (n: number) => void } = {}): ReadableStream<Uint8Array> {
  const when = opts.now ?? new Date();
  const ranges = rangesOf(i);
  async function* entries() {
    // client-zip's cancel() throws the reason into this generator and does not wait for it; a rejected promise nobody holds would be reported
    // as unhandled, so a cancelled export ends the generator quietly
    try {
      for (const c of p.clips) {
        const raw = clipStream(i.src, c.muxed, opts.signal);
        const input = opts.onBytes ? raw.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({ transform: (chunk, ctl) => { opts.onBytes!(chunk.length); ctl.enqueue(chunk); } })) : raw;
        yield { name: c.name, lastModified: when, input };
        const side = buildSidecar(i, ranges, c.index, i.clips.length, c.name, c.start, c.end, pyRound(c.plan.copyStart, 3));
        yield { name: c.sidecar, lastModified: when, input: JSON.stringify(side, null, 1) };
      }
      yield { name: p.csvName, lastModified: when, input: p.csv };
    } catch (e) {
      if (!opts.signal?.aborted) throw e;
    }
  }
  return makeZip(entries());
}

// ---------------------------------------------------------------- where the zip goes
export interface SinkResult {
  /** bytes written */
  size: number;
  /** a blob URL the page can offer again (only for the Blob sink) */
  url?: string;
}

export interface ZipSink {
  kind: 'file' | 'folder' | 'download';
  write(stream: ReadableStream<Uint8Array>, signal?: AbortSignal): Promise<SinkResult>;
  /** Gives back what choosing the place already made: a save picker creates the empty file when the user picks it. Best effort, never throws. */
  discard?(): Promise<void>;
}

export interface SinkEnv {
  showSaveFilePicker?: (o: unknown) => Promise<any>;
  showDirectoryPicker?: (o: unknown) => Promise<any>;
  /** the page's blob download (an anchor click); the default one is in `downloadBlob` */
  download?: (blob: Blob, name: string) => string;
  /** the Blob cap, bytes */
  cap?: number;
}

const cancelled = (): never => {
  throw new EngineError(499, 'Cancelled.');
};
const isAbort = (e: unknown): boolean => e instanceof DOMException && e.name === 'AbortError';

export function downloadBlob(blob: Blob, name: string): string {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  a.remove();
  return url;
}

async function writeToHandle(handle: any, stream: ReadableStream<Uint8Array>, signal?: AbortSignal): Promise<SinkResult> {
  const w = await handle.createWritable();
  try {
    await stream.pipeTo(w, { signal });
  } catch (e) {
    try {
      await w.abort?.();
    } catch { /* already closed */ }
    throw e;
  }
  const f = await handle.getFile?.();
  return { size: f?.size ?? 0 };
}

/** Pick where the zip goes. Needs a click behind it (the pickers ask for one), so call it first thing in the click's handler chain: the picker opens
 *  before the first await. `estimate` is the zip's expected size, or a promise of it (only the in-memory download needs it, so it may arrive later). */
export async function chooseSink(env: SinkEnv, name: string, estimate: number | Promise<number>): Promise<ZipSink> {
  const cap = env.cap ?? SAFARI_BLOB_CAP;
  if (env.showSaveFilePicker) {
    let handle: any;
    try {
      handle = await env.showSaveFilePicker({ suggestedName: name, types: [{ description: 'Zip file', accept: { 'application/zip': ['.zip'] } }] });
    } catch (e) {
      return isAbort(e) ? cancelled() : Promise.reject(e);
    }
    // A picker makes an empty file for a new name, but hands back the existing file when the user picks a zip to replace; a failed export must not delete that one
    // (the old bytes stay until a write finishes). Only a file that was empty when it was picked is the picker's own.
    let existing = false;
    try {
      existing = ((await handle.getFile?.())?.size ?? 0) > 0;
    } catch { /* cannot tell: treat it as new, as before */ }
    return { kind: 'file', write: (s, sig) => writeToHandle(handle, s, sig), discard: async () => { if (existing) return; try { await handle.remove?.(); } catch { /* not allowed here: the empty file stays */ } } };
  }
  if (env.showDirectoryPicker) {
    let dir: any;
    try {
      dir = await env.showDirectoryPicker({ mode: 'readwrite' });
    } catch (e) {
      return isAbort(e) ? cancelled() : Promise.reject(e);
    }
    return { kind: 'folder', write: async (s, sig) => writeToHandle(await dir.getFileHandle(name, { create: true }), s, sig) };
  }
  if ((await estimate) > cap) throw new EngineError(413, 'These clips come to more than 2 GB, which this browser cannot build in memory. Use My Mac, or export fewer clips.');
  return {
    kind: 'download',
    write: async (s, sig) => {
      let n = 0;
      const counted = s.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
        transform: (chunk, ctl) => {
          n += chunk.length;
          if (n > cap) ctl.error(new EngineError(413, 'These clips come to more than 2 GB, which this browser cannot build in memory. Use My Mac, or export fewer clips.'));
          else ctl.enqueue(chunk);
        },
      }), { signal: sig });
      const blob = await new Response(counted).blob();
      return { size: blob.size, url: (env.download ?? downloadBlob)(blob, name) };
    },
  };
}

/** The pickers of this page. */
export const pageSinkEnv = (): SinkEnv => {
  const w = globalThis as unknown as SinkEnv;
  return { showSaveFilePicker: w.showSaveFilePicker?.bind(globalThis), showDirectoryPicker: w.showDirectoryPicker?.bind(globalThis) };
};
