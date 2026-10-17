import { useLayoutEffect, useRef } from 'preact/hooks';
import { bucketMax, holdColumns, maxOf } from '../model/presence';
import { timeToX, viewEnd } from '../model/timeline';
import type { Store } from '../state';
import { COLORS, drawCanvas, Row, scrubber } from './lane';

export const PRESENCE_H = 56;

/** The score of the active run along the timeline: teal above enter, dim teal between exit and enter, grey below exit. */
export function PresenceLane({ store }: { store: Store }) {
  const lane = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const v = store.view.value;
  const s = store.samples.value;
  const intervals = store.run.value?.segments.presence_intervals;
  useLayoutEffect(() => {
    drawCanvas(canvas.current, v.width, PRESENCE_H, (ctx) => {
      if (!s || !s.t.length) return;
      const top = 5; // below the interval bar
      const area = PRESENCE_H - top;
      // detector scores are 0..1; motion scores are not, so that lane scales to the data and the enter line
      const ymax = s.mode === 'motion' ? maxOf(s.score, s.enter * 1.25) : 1;
      const cols = Math.max(1, Math.floor(v.width));
      const dt = s.t.length > 1 ? (s.t[s.t.length - 1] - s.t[0]) / (s.t.length - 1) : 1;
      const vals = holdColumns(bucketMax(s.t, s.score, v.t0, viewEnd(v), cols), Math.ceil(dt * v.pps * 1.5));
      for (let c = 0; c < cols; c++) {
        const val = vals[c];
        if (Number.isNaN(val)) continue;
        const h = Math.max(1, Math.min(1, val / ymax) * area);
        ctx.fillStyle = val >= s.enter ? COLORS.teal : val >= s.exit ? COLORS.tealDim : COLORS.grey;
        ctx.fillRect(c, PRESENCE_H - h, 1, h);
      }
      ctx.strokeStyle = COLORS.line;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      for (const level of [s.enter, s.exit]) {
        const y = Math.round(PRESENCE_H - Math.min(1, level / ymax) * area) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(v.width, y);
        ctx.stroke();
        ctx.globalAlpha = 0.6;
      }
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.teal;
      for (const [a, b] of intervals ?? []) {
        const x0 = Math.max(0, timeToX(v, a));
        const x1 = Math.min(v.width, timeToX(v, b));
        if (x1 > x0) ctx.fillRect(x0, 0, Math.max(1, x1 - x0), 3);
      }
    });
  });
  return (
    <Row label="presence" height={PRESENCE_H}>
      <div class="lane" ref={lane} onPointerDown={scrubber(store, () => lane.current)}>
        <canvas ref={canvas} style={{ width: '100%', height: `${PRESENCE_H}px` }} aria-label="Presence of the target over time" role="img" />
      </div>
    </Row>
  );
}
