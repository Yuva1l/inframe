import { useLayoutEffect, useRef } from 'preact/hooks';
import { formatTime, tickStep, timeToX } from '../model/timeline';
import type { Store } from '../state';
import { COLORS, drawCanvas, Row, scrubber } from './lane';

export const RULER_H = 22;

/** Time ticks of the visible range. The times are source (file) time. */
export function Ruler({ store }: { store: Store }) {
  const lane = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const v = store.view.value;
  useLayoutEffect(() => {
    drawCanvas(canvas.current, v.width, RULER_H, (ctx) => {
      const step = tickStep(v.pps);
      ctx.font = '10px system-ui, sans-serif';
      ctx.textBaseline = 'top';
      for (let t = Math.ceil(v.t0 / step) * step; timeToX(v, t) < v.width; t += step) {
        const x = Math.round(timeToX(v, t)) + 0.5;
        ctx.fillStyle = COLORS.tick;
        ctx.fillRect(x, RULER_H - 8, 1, 8);
        ctx.fillStyle = COLORS.text;
        ctx.fillText(formatTime(t, step), x + 3, 3);
      }
    });
  });
  return (
    <Row label="source time" height={RULER_H}>
      <div class="lane" ref={lane} onPointerDown={scrubber(store, () => lane.current)}>
        <canvas ref={canvas} style={{ width: '100%', height: `${RULER_H}px` }} aria-hidden="true" />
      </div>
    </Row>
  );
}
