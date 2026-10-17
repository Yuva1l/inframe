// What the export dialog says before anything is written (docs/T9A_DESIGN.md 1.7).
import { formatBytes, formatLength } from './format';

export const EARLY_S = 0.5;
export const EDIT_LISTS = [
  { key: 'fcpxml', label: 'FCPXML (Final Cut, FilmCraft)' },
  { key: 'fcp7', label: 'FCP7 XML (Premiere)' },
  { key: 'kdenlive', label: 'Kdenlive' },
  { key: 'edl', label: 'EDL (CMX3600)' },
  { key: 'otio', label: 'OTIO' },
] as const;

/** editlist.SMPTE_RATES and EDL_MAX_FPS: the same rules as editlist.write_edit_lists. */
const SMPTE = [23.976, 24, 25, 29.97, 30, 50, 59.94, 60];

/** Why an edit-list format cannot be written for this frame rate, or null. */
export function editListIssue(key: string, fps: number): string | null {
  if (key === 'edl' && fps > 60 + 0.01) return `needs 60 fps or less (this video is ${Math.round(fps * 100) / 100} fps)`;
  if ((key === 'edl' || key === 'fcp7') && !SMPTE.some((r) => Math.abs(fps - r) < 0.01)) return `needs a standard frame rate (this video is ${Math.round(fps * 100) / 100} fps)`;
  return null;
}

export interface PlanClip {
  start: number;
  end: number;
}

export interface PlanInput {
  clips: readonly PlanClip[];
  /** the selected ones; used when `selectedOnly` */
  selected: readonly PlanClip[];
  selectedOnly: boolean;
  /** slow-motion factor of the run (1 = none): output lengths are source lengths times this */
  factor: number;
  keyframeOf: (t: number) => number;
  fileSize: number;
  duration: number;
}

export interface Plan {
  count: number;
  total: number;
  outSeconds: number;
  bytes: number;
  early: number;
  text: string;
}

/** "8 clips (selected) of 23 - 1 min 12 s output - about 640 MB - 3 clips start more than 0.5 s early at a keyframe" */
export function plan(i: PlanInput): Plan {
  const used = i.selectedOnly ? i.selected : i.clips;
  const total = i.clips.length;
  const out = used.reduce((s, c) => s + (c.end - c.start) * Math.max(1, i.factor), 0);
  // a stream copy starts at the keyframe at or before the clip start, so it carries those extra seconds too
  const copied = used.reduce((s, c) => s + (c.end - Math.min(i.keyframeOf(c.start), c.start)), 0);
  const bytes = i.duration > 0 ? (i.fileSize * copied) / i.duration : 0;
  const early = used.filter((c) => c.start - i.keyframeOf(c.start) > EARLY_S).length;
  const noun = (n: number) => `${n} clip${n === 1 ? '' : 's'}`;
  const parts = [`${noun(used.length)}${i.selectedOnly ? ` (selected) of ${total}` : ''}`, `${formatLength(out)} output`, `about ${formatBytes(bytes)}`];
  if (early) parts.push(`${noun(early)} start${early === 1 ? 's' : ''} more than ${EARLY_S} s early at a keyframe`);
  return { count: used.length, total, outSeconds: out, bytes, early, text: parts.join(' - ') };
}
