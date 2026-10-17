// A canvas over the video: the run's target box for the current sample, the zone, and (for the pick tab) numbered candidate boxes.
// It places everything with the server's display size, never videoWidth/videoHeight (docs/T9A_DESIGN.md 1.1).
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import type { Box, Zone } from '../engine/types';
import { boxToRect, containRect, zoneToRect } from '../model/geometry';
import { sampleAt } from '../model/presence';
import type { Store } from '../state';

export interface PickBox {
  box: Box;
  label: string;
}

export function Overlay({ store, zone, pickBoxes }: { store: Store; zone?: Zone | null; pickBoxes?: PickBox[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const m = store.media.value;
  const t = store.time.value;
  const showBoxes = store.showBoxes.value;
  const samples = store.samples.value;

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useLayoutEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext?.('2d');
    if (!el || !ctx || !m) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.max(1, Math.round(size.w * dpr));
    el.height = Math.max(1, Math.round(size.h * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    const frame = containRect(size.w, size.h, m.display_w, m.display_h);
    if (zone) {
      const z = zoneToRect(frame, zone);
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.fillStyle = 'rgba(250, 204, 21, 0.08)';
      ctx.fillRect(z.x, z.y, z.w, z.h);
      ctx.strokeRect(z.x, z.y, z.w, z.h);
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = '#4ade80';
    ctx.fillStyle = '#4ade80';
    ctx.lineWidth = 2;
    ctx.font = '11px system-ui, sans-serif';
    for (const p of pickBoxes ?? []) {
      const r = boxToRect(frame, p.box);
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.fillRect(r.x - 1, r.y - 18, ctx.measureText(p.label).width + 12, 18);
      ctx.fillStyle = '#04150a';
      ctx.fillText(p.label, r.x + 5, r.y - 5);
      ctx.fillStyle = '#4ade80';
    }
    if (showBoxes && samples) {
      const i = sampleAt(samples.t, t, 1);
      const b = i >= 0 ? samples.box[i] : null;
      if (b) {
        const r = boxToRect(frame, b);
        ctx.strokeRect(r.x, r.y, r.w, r.h);
      }
    }
  }, [size.w, size.h, m?.display_w, m?.display_h, t, showBoxes, samples, zone, pickBoxes]);

  return <canvas ref={canvas} class="overlay" aria-hidden="true" />;
}
