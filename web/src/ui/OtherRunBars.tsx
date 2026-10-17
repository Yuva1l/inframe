import { useLayoutEffect, useRef } from 'preact/hooks';
import type { RunSummary, Samples } from '../engine/types';
import { runLabel } from '../model/format';
import { bucketMax, holdColumns } from '../model/presence';
import { viewEnd } from '../model/timeline';
import type { Store } from '../state';
import { COLORS, drawCanvas, Row } from './lane';

export const OTHER_H = 14;
const BAR_COLORS = [COLORS.violet, COLORS.amber, COLORS.rose];

function Bar({ store, run, samples, color }: { store: Store; run: RunSummary; samples: Samples | undefined; color: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const v = store.view.value;
  useLayoutEffect(() => {
    drawCanvas(canvas.current, v.width, OTHER_H, (ctx) => {
      if (!samples || !samples.t.length) return;
      const cols = Math.max(1, Math.floor(v.width));
      const dt = samples.t.length > 1 ? (samples.t[samples.t.length - 1] - samples.t[0]) / (samples.t.length - 1) : 1;
      const vals = holdColumns(bucketMax(samples.t, samples.score, v.t0, viewEnd(v), cols), Math.ceil(dt * v.pps * 1.5));
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.65;
      for (let c = 0; c < cols; c++) if (vals[c] >= samples.enter) ctx.fillRect(c, 3, 1, OTHER_H - 6);
    });
  });
  const label = runLabel(run);
  return (
    <Row label={label} height={OTHER_H} class="other">
      <div class="lane" role="button" tabIndex={0} aria-label={`Switch to run ${label}`} title={`Switch to run ${label}`}
        onClick={() => void store.switchRun(run.run_id)}
        onKeyDown={(e) => { if (e.key === 'Enter') void store.switchRun(run.run_id); }}>
        <canvas ref={canvas} style={{ width: '100%', height: `${OTHER_H}px` }} aria-hidden="true" />
      </div>
    </Row>
  );
}

/** Thin read-only presence bars of up to three other runs of this video. A click makes that run the active one. */
export function OtherRunBars({ store }: { store: Store }) {
  const active = store.run.value?.run_id;
  const cache = store.samplesCache.value;
  const others = store.runs.value.filter((r) => r.run_id !== active).slice(0, 3);
  return (
    <>
      {others.map((r, i) => <Bar key={r.run_id} store={store} run={r} samples={cache[r.run_id]} color={BAR_COLORS[i]} />)}
    </>
  );
}
