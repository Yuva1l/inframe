// "Can this browser play it?" is answered by playing it (docs/T9A_DESIGN.md 2.4): set src, wait for loadedmetadata, seek to the
// middle, wait for one presented frame. canPlayType is only a hint. The same load measures the clock offset (model/clock.ts).
import { clockOffset } from '../model/clock';

export interface PlayableResult {
  ok: boolean;
  /** file time minus player time; 0 when rVFC is missing */
  clockOffset: number;
  error?: string;
}

export type PlayableFn = (video: HTMLVideoElement, url: string, firstFrameFileTime: number) => Promise<PlayableResult>;

type RVFC = (cb: (now: number, meta: { mediaTime: number }) => void) => number;

function once(el: HTMLVideoElement, ok: string, fail: string[], ms: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const done = (fn: () => void) => {
      clearTimeout(timer);
      el.removeEventListener(ok, onOk);
      for (const f of fail) el.removeEventListener(f, onFail);
      fn();
    };
    const onOk = () => done(resolve);
    const onFail = () => done(() => reject(new Error(el.error?.message || 'the browser cannot play this file')));
    const timer = setTimeout(() => done(() => reject(new Error('timed out'))), ms);
    el.addEventListener(ok, onOk);
    for (const f of fail) el.addEventListener(f, onFail);
  });
}

/** The first presented frame's mediaTime, or null when the browser has no requestVideoFrameCallback. */
function frameTime(el: HTMLVideoElement, ms: number): Promise<number | null> {
  const rvfc = (el as unknown as { requestVideoFrameCallback?: RVFC }).requestVideoFrameCallback;
  if (!rvfc) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('no frame was shown')), ms);
    rvfc.call(el, (_now, meta) => {
      clearTimeout(timer);
      resolve(meta.mediaTime);
    });
  });
}

export const loadPlayable: PlayableFn = async (el, url, firstFrameFileTime) => {
  const deadline = Date.now() + 6000;
  const left = () => Math.max(500, deadline - Date.now());
  try {
    el.muted = true;
    el.src = url;
    el.load();
    await once(el, 'loadedmetadata', ['error'], left());
    // the middle: proves seeking and decoding work, not just the container
    const mid = el.duration > 1 ? el.duration / 2 : 0;
    const shown = frameTime(el, left());
    const seeked = once(el, 'seeked', ['error'], left());
    el.currentTime = mid;
    await seeked;
    await shown;
    // the start: the first presented frame gives m0
    const first = frameTime(el, left());
    const back = once(el, 'seeked', ['error'], left());
    el.currentTime = 0;
    await back;
    const m0 = await first;
    el.muted = false;
    return { ok: true, clockOffset: m0 === null ? 0 : clockOffset(firstFrameFileTime, m0) };
  } catch (e) {
    el.removeAttribute('src');
    el.load();
    return { ok: false, clockOffset: 0, error: e instanceof Error ? e.message : String(e) };
  }
};
