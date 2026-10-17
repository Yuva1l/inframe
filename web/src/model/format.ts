// Text for the top bar, transport and left panel. Times are file time.
import type { MediaInfo } from '../engine/types';

const pad = (n: number, w = 2) => String(n).padStart(w, '0');

/** mm:ss.cc, the transport readout (03:14.20); hours are shown when needed. */
export function formatClock(t: number): string {
  const total = Math.max(0, Math.round(t * 100));
  const cs = total % 100;
  const s = Math.floor(total / 100) % 60;
  const m = Math.floor(total / 6000) % 60;
  const h = Math.floor(total / 360000);
  return `${h > 0 ? `${h}:` : ''}${pad(m)}:${pad(s)}.${pad(cs)}`;
}

/** 20:14 or 1:02:03, whole seconds. */
export function formatDuration(t: number): string {
  const total = Math.max(0, Math.round(t));
  const s = total % 60;
  const m = Math.floor(total / 60) % 60;
  const h = Math.floor(total / 3600);
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/** "7.5 s", "1 min 12 s" */
export function formatLength(sec: number): string {
  if (Math.round(sec * 10) / 10 < 60) return `${Math.round(sec * 10) / 10} s`;
  const total = Math.round(sec); // round first, then split, so 119.6 s is "2 min 0 s", never "1 min 60 s"
  return `${Math.floor(total / 60)} min ${total % 60} s`;
}

export function formatBytes(n: number): string {
  if (n < 1000) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n;
  let i = -1;
  do {
    v /= 1000;
    i++;
  } while (v >= 1000 && i < units.length - 1);
  return `${v >= 100 ? Math.round(v) : Math.round(v * 10) / 10} ${units[i]}`;
}

export function formatFps(fps: number): string {
  return `${Math.round(fps * 100) / 100} fps`;
}

/** "20:14 · HEVC 3840x2160 · 60 fps · 1.2 GB" */
export function mediaLine(m: MediaInfo): string {
  return [formatDuration(m.duration), `${m.codec.toUpperCase()} ${m.display_w}x${m.display_h}`, formatFps(m.fps), formatBytes(m.size)].join(' · ');
}

export function slowmoLabel(m: MediaInfo): string | null {
  return m.capture_fps >= 100 ? `slow motion ${Math.round(m.capture_fps)} fps` : null;
}

/** The aria-label of a clip block: "Clip 7, 07:12.4 to 07:19.9, 7.5 seconds". */
export function clipLabel(n: number, start: number, end: number): string {
  const f = (t: number) => {
    const total = Math.max(0, Math.round(t * 10));
    return `${pad(Math.floor(total / 600))}:${pad(Math.floor(total / 10) % 60)}.${total % 10}`;
  };
  return `Clip ${n}, ${f(start)} to ${f(end)}, ${Math.round((end - start) * 10) / 10} seconds`;
}

/** "tap · 10:12", "saved: alice · 09:40": a run in the switcher (target, then the time it was made). */
export function runLabel(r: { target: string; slug: string; created: string | number }): string {
  const target = (r.target || r.slug).replace(/^([a-z]+):(?=\S)/i, '$1: ');
  const d = typeof r.created === 'number' ? new Date(r.created * 1000) : new Date(r.created);
  if (Number.isNaN(d.getTime())) return target;
  return `${target} · ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** "slow-mo x8: source 2.0 s, output 16.0 s" for a clip of a run that was slowed down by `factor`. */
export function slowmoLine(sourceSeconds: number, factor: number): string {
  return `slow-mo x${Math.round(factor * 100) / 100}: source ${sourceSeconds.toFixed(1)} s, output ${(sourceSeconds * factor).toFixed(1)} s`;
}
