// Shared bits of the timeline lanes: a row with its label cell, canvas drawing with the device pixel ratio, scrubbing.
import type { ComponentChildren } from 'preact';
import { xToTime } from '../model/timeline';
import type { Store } from '../state';

export const COLORS = {
  tick: '#3a3d44', text: '#9aa0aa', teal: '#2dd4bf', tealDim: 'rgba(45, 212, 191, 0.38)', grey: '#4a4f57', line: 'rgba(231, 232, 235, 0.35)',
  cell: '#1b1d21', violet: '#a78bfa', amber: '#f59e0b', rose: '#fb7185',
};

/** One row of the timeline: a label cell at the left and the lane (the time axis) at the right. */
export function Row({ label, height, children, class: cls = '' }: { label: string; height: number; children: ComponentChildren; class?: string }) {
  return (
    <div class={`tl-row ${cls}`} style={{ height: `${height}px` }}>
      <div class="tl-label" title={label}>{label}</div>
      {children}
    </div>
  );
}

/** Size a canvas to w x h CSS pixels at the device pixel ratio and hand its context to `draw`. A page without canvas (jsdom) draws nothing. */
export function drawCanvas(el: HTMLCanvasElement | null, w: number, h: number, draw: (ctx: CanvasRenderingContext2D) => void) {
  const ctx = el?.getContext?.('2d');
  if (!el || !ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const pw = Math.max(1, Math.round(w * dpr));
  const ph = Math.max(1, Math.round(h * dpr));
  if (el.width !== pw) el.width = pw;
  if (el.height !== ph) el.height = ph;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  draw(ctx);
}

/** The x of a pointer inside the lane element, in lane pixels. */
export const laneX = (el: Element | null, clientX: number): number => clientX - (el?.getBoundingClientRect().left ?? 0);

/** pointerdown handler for lanes that move the playhead: click or drag to scrub. */
export function scrubber(store: Store, getLane: () => Element | null) {
  return (e: PointerEvent) => {
    if (e.button !== 0) return;
    const at = (cx: number) => xToTime(store.view.value, laneX(getLane(), cx));
    store.seek(at(e.clientX));
    const move = (m: PointerEvent) => store.seek(at(m.clientX));
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  };
}
