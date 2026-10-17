// The clip-edit model (docs/T9A_DESIGN.md 2.6): pure functions over an immutable EditState. Each op returns
// { state, changed, reason? }; the UI shows `reason` as a toast when nothing changed. The rules are the server's
// (serve_edits.validate_clips): at most 2000 clips, finite numbers, sorted, offset <= start < end <= end_time (1 ms slack),
// at least 0.2 s long, no overlap, unique ids; times rounded to 1 ms.
import type { DetectorClip, EditClip, EditsDoc, Origin } from '../engine/types';

export const MIN_LEN = 0.2;
export const MAX_CLIPS = 2000;
export const ADD_LEN = 2;
export const HISTORY_CAP = 200;
const SLACK = 0.001;
const EPS = 1e-6;
const ID_RE = /^[a-z0-9]{1,16}$/;
const ORIGINS: readonly Origin[] = ['detector', 'edited', 'split', 'merge', 'added', 'resegment'];

export interface EditState {
  clips: EditClip[];
  /** the number the next new id (`u<nextId>`) takes */
  nextId: number;
}

/** The video's span in file time: `offset` to `end_time`. */
export interface Bounds {
  offset: number;
  end: number;
}

export interface OpResult {
  state: EditState;
  changed: boolean;
  reason?: string;
  /** the clips an op made or kept, for the selection */
  ids?: string[];
}

export const r3 = (t: number): number => Math.round(t * 1000) / 1000;
const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);
const same = (state: EditState, reason: string): OpResult => ({ state, changed: false, reason });
const copy = (c: EditClip): EditClip => ({ id: c.id, start: c.start, end: c.end, origin: c.origin });
const suffix = (id: string): number => {
  const m = /^[a-z]*([0-9]+)$/.exec(id);
  return m ? Number(m[1]) : 0;
};

// ---------------------------------------------------------------- building a state
/** The detector's clips as editable clips: ids `d<index>` (the clip's own number in segments.json). With `bounds`, a clip is cut back
 * to the video's span (a run written before the cutter clamped at the first frame can start before `offset`) and one left shorter
 * than MIN_LEN is dropped, so the first save is never refused as "outside the video". */
export function fromDetector(clips: readonly DetectorClip[], bounds?: Bounds): EditState {
  let out = clips.map((c, i) => ({ id: `d${c.index ?? i + 1}`, start: r3(c.start), end: r3(c.end), origin: 'detector' as Origin }));
  if (bounds && bounds.end > bounds.offset) {
    out = out
      .map((c) => ({ ...c, start: r3(Math.max(c.start, bounds.offset)), end: r3(Math.min(c.end, bounds.end)) }))
      .filter((c) => c.end - c.start >= MIN_LEN - EPS);
  }
  return { clips: out, nextId: out.reduce((m, c) => Math.max(m, suffix(c.id)), 0) + 1 };
}

/** The stored edits.json as a state. */
export function fromDoc(doc: Pick<EditsDoc, 'clips' | 'next_id'>): EditState {
  const clips = doc.clips.map(copy);
  return { clips, nextId: Math.max(doc.next_id || 1, clips.reduce((m, c) => Math.max(m, suffix(c.id)), 0) + 1) };
}

// ---------------------------------------------------------------- validation
/** Every rule the server checks; an empty list means the clip list is acceptable. */
export function validate(clips: readonly EditClip[], bounds: Bounds): string[] {
  const problems: string[] = [];
  if (clips.length > MAX_CLIPS) problems.push(`too many clips (${clips.length}, at most ${MAX_CLIPS})`);
  const seen = new Set<string>();
  clips.forEach((c, i) => {
    const at = `clip ${i + 1}`;
    if (typeof c.id !== 'string' || !ID_RE.test(c.id)) problems.push(`${at}: bad id`);
    else if (seen.has(c.id)) problems.push(`${at}: duplicate id ${c.id}`);
    else seen.add(c.id);
    if (!ORIGINS.includes(c.origin)) problems.push(`${at}: bad origin`);
    if (!Number.isFinite(c.start) || !Number.isFinite(c.end)) {
      problems.push(`${at}: times must be finite numbers`);
      return;
    }
    if (c.start < bounds.offset - SLACK || c.end > bounds.end + SLACK) problems.push(`${at}: outside the video`);
    if (c.end - c.start < MIN_LEN - EPS) problems.push(`${at}: shorter than ${MIN_LEN} s`);
    const prev = clips[i - 1];
    if (prev && Number.isFinite(prev.start) && Number.isFinite(prev.end)) {
      if (c.start < prev.start) problems.push(`${at}: not sorted by start`);
      else if (prev.end > c.start + EPS) problems.push(`${at}: overlaps the clip before it`);
    }
  });
  return problems;
}

// ---------------------------------------------------------------- ops
export function trim(state: EditState, id: string, edge: 'start' | 'end', t: number, bounds: Bounds): OpResult {
  const i = state.clips.findIndex((c) => c.id === id);
  if (i < 0) return same(state, 'No such clip');
  if (!Number.isFinite(t)) return same(state, 'Not a time');
  const c = state.clips[i];
  const lo = state.clips[i - 1]?.end ?? bounds.offset;
  const hi = state.clips[i + 1]?.start ?? bounds.end;
  const next = copy(c);
  if (edge === 'start') next.start = r3(clamp(t, lo, c.end - MIN_LEN));
  else next.end = r3(clamp(t, c.start + MIN_LEN, hi));
  if (next.start === c.start && next.end === c.end) return same(state, edge === 'start' ? 'The start cannot move there' : 'The end cannot move there');
  if (c.origin === 'detector') next.origin = 'edited';
  const clips = state.clips.slice();
  clips[i] = next;
  return { state: { ...state, clips }, changed: true, ids: [id] };
}

/** Split the clip `id`, or the one that holds `t`, at `t`. The first part keeps the id. */
export function split(state: EditState, t: number, id?: string): OpResult {
  const c = id ? state.clips.find((x) => x.id === id) : state.clips.find((x) => t > x.start && t < x.end);
  if (!c) return same(state, id ? 'No such clip' : 'The playhead is not inside a clip');
  if (!(t > c.start && t < c.end)) return same(state, 'The playhead is not inside that clip');
  if (t - c.start < MIN_LEN - EPS || c.end - t < MIN_LEN - EPS) return same(state, 'Too short to split here');
  if (state.clips.length >= MAX_CLIPS) return same(state, `At most ${MAX_CLIPS} clips`);
  const at = r3(t);
  const second: EditClip = { id: `u${state.nextId}`, start: at, end: c.end, origin: 'split' };
  const clips = state.clips.flatMap((x) => (x.id === c.id ? [{ ...x, end: at, origin: 'split' as Origin }, second] : [x]));
  return { state: { clips, nextId: state.nextId + 1 }, changed: true, ids: [second.id] };
}

/** Merge a clip with the one after it (the gap becomes part of the clip). */
export function mergeNext(state: EditState, id: string): OpResult {
  const i = state.clips.findIndex((c) => c.id === id);
  if (i < 0) return same(state, 'No such clip');
  const n = state.clips[i + 1];
  if (!n) return same(state, 'There is no clip after this one');
  const merged: EditClip = { id, start: state.clips[i].start, end: n.end, origin: 'merge' };
  const clips = state.clips.slice();
  clips.splice(i, 2, merged);
  return { state: { ...state, clips }, changed: true, ids: [id] };
}

/** Merge the selected clips into one: from the first selected to the last, so any clip between them goes into it too. */
export function mergeIds(state: EditState, ids: readonly string[]): OpResult {
  const set = new Set(ids);
  const idx = state.clips.flatMap((c, i) => (set.has(c.id) ? [i] : []));
  if (idx.length < 2) return same(state, 'Select two or more clips to merge');
  const a = idx[0];
  const b = idx[idx.length - 1];
  const merged: EditClip = { id: state.clips[a].id, start: state.clips[a].start, end: state.clips[b].end, origin: 'merge' };
  const clips = state.clips.slice();
  clips.splice(a, b - a + 1, merged);
  return { state: { ...state, clips }, changed: true, ids: [merged.id] };
}

export function remove(state: EditState, ids: readonly string[]): OpResult {
  const set = new Set(ids);
  const clips = state.clips.filter((c) => !set.has(c.id));
  if (clips.length === state.clips.length) return same(state, 'Nothing to delete');
  return { state: { ...state, clips }, changed: true, ids: [] };
}

/** A 2 s clip centred on t, shrunk to the gap it lands in. */
export function add(state: EditState, t: number, bounds: Bounds): OpResult {
  if (!Number.isFinite(t)) return same(state, 'Not a time');
  if (state.clips.length >= MAX_CLIPS) return same(state, `At most ${MAX_CLIPS} clips`);
  if (state.clips.some((c) => t > c.start && t < c.end)) return same(state, 'The playhead is inside a clip');
  const prev = [...state.clips].reverse().find((c) => c.end <= t);
  const next = state.clips.find((c) => c.start >= t);
  const lo = prev?.end ?? bounds.offset;
  const hi = next?.start ?? bounds.end;
  const start = r3(Math.max(lo, t - ADD_LEN / 2));
  const end = r3(Math.min(hi, t + ADD_LEN / 2));
  if (end - start < MIN_LEN - EPS) return same(state, 'No room for a clip here');
  const c: EditClip = { id: `u${state.nextId}`, start, end, origin: 'added' };
  const clips = state.clips.slice();
  clips.splice(next ? clips.indexOf(next) : clips.length, 0, c);
  return { state: { clips, nextId: state.nextId + 1 }, changed: true, ids: [c.id] };
}

/** Replace the clip list by the intervals of a re-segment preview (origin `resegment`). */
export function applyResegment(state: EditState, intervals: readonly (readonly [number, number])[], bounds?: Bounds): OpResult {
  const clips: EditClip[] = intervals.map(([s, e], i) => ({ id: `u${state.nextId + i}`, start: r3(s), end: r3(e), origin: 'resegment' }));
  if (clips.length === state.clips.length && clips.every((c, i) => c.start === state.clips[i].start && c.end === state.clips[i].end)) {
    return same(state, 'That is the clip list already');
  }
  if (bounds) {
    const problems = validate(clips, bounds);
    if (problems.length) return same(state, problems[0]);
  }
  return { state: { clips, nextId: state.nextId + clips.length }, changed: true, ids: clips.map((c) => c.id) };
}

/** "Revert to detector": the clip list becomes the detector's again. An ordinary step, so it can be undone. */
export function revert(state: EditState, detector: readonly EditClip[]): OpResult {
  const clips = detector.map(copy);
  const equal = clips.length === state.clips.length && clips.every((c, i) => {
    const s = state.clips[i];
    return c.id === s.id && c.start === s.start && c.end === s.end && c.origin === s.origin;
  });
  if (equal) return same(state, 'Already the detector’s clips');
  return { state: { ...state, clips }, changed: true, ids: [] };
}

// ---------------------------------------------------------------- summary
export interface Diff {
  changed: number;
  deleted: number;
  added: number;
  /** "edited: 3 changed, 1 deleted, 2 added", or "" when nothing differs */
  text: string;
}

export function diffSummary(clips: readonly EditClip[], detector: readonly EditClip[]): Diff {
  const base = new Map(detector.map((c) => [c.id, c]));
  const kept = new Set<string>();
  let changed = 0;
  let added = 0;
  for (const c of clips) {
    const d = base.get(c.id);
    if (!d) added++;
    else {
      kept.add(c.id);
      if (Math.abs(d.start - c.start) > EPS || Math.abs(d.end - c.end) > EPS) changed++;
    }
  }
  const deleted = detector.length - kept.size;
  const parts = [changed && `${changed} changed`, deleted && `${deleted} deleted`, added && `${added} added`].filter(Boolean);
  return { changed, deleted, added, text: parts.length ? `edited: ${parts.join(', ')}` : '' };
}

// ---------------------------------------------------------------- history
/** Undo and redo over whole states. Push the state BEFORE a change; a drag is one push. */
export class History {
  past: EditState[] = [];
  future: EditState[] = [];
  readonly cap: number;

  constructor(cap = HISTORY_CAP) {
    this.cap = cap;
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }
  get canRedo(): boolean {
    return this.future.length > 0;
  }

  push(before: EditState): void {
    this.past.push(before);
    if (this.past.length > this.cap) this.past.shift();
    this.future = [];
  }

  /** The state to go back to (ids stay unique: nextId never goes down), or null. */
  undo(current: EditState): EditState | null {
    const p = this.past.pop();
    if (!p) return null;
    this.future.push(current);
    return { ...p, nextId: Math.max(p.nextId, current.nextId) };
  }

  redo(current: EditState): EditState | null {
    const n = this.future.pop();
    if (!n) return null;
    this.past.push(current);
    return { ...n, nextId: Math.max(n.nextId, current.nextId) };
  }

  clear(): void {
    this.past = [];
    this.future = [];
  }
}
