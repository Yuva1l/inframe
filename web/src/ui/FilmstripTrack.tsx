import { useLayoutEffect, useRef } from 'preact/hooks';
import type { Filmstrip } from '../engine/types';
import { xToTime } from '../model/timeline';
import type { Store } from '../state';
import { COLORS, drawCanvas, Row, scrubber } from './lane';
import { useEngineUrls } from './useEngineUrl';

export const FILM_H = 56;

// sheet images by url: shared by every redraw
const sheets = new Map<string, HTMLImageElement>();

/** The index of the cell nearest t (cells are sorted by time). */
export function nearestCell(cells: Filmstrip['cells'], t: number): number {
  let lo = 0;
  let hi = cells.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cells[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && (lo === cells.length || t - cells[lo - 1].t <= cells[lo].t - t)) lo--;
  return lo;
}

/** Frames along the timeline: for each slot at this zoom, the nearest filmstrip cell, drawn from its sheet. */
export function FilmstripTrack({ store }: { store: Store }) {
  const lane = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const v = store.view.value;
  const f = store.filmstrip.value;
  const m = store.media.value;
  // a browser-engine sheet is a key until it is rendered: resolve every sheet the filmstrip lists, then draw from the ready URLs
  const sheetRaw = m && f ? Array.from({ length: f.cells.length ? f.cells[f.cells.length - 1].sheet + 1 : 0 }, (_, n) => store.engine.sheetUrl(m.id, n)) : [];
  const ready = useEngineUrls(store.engine, sheetRaw);
  useLayoutEffect(() => {
    const redraw = () => {
      drawCanvas(canvas.current, v.width, FILM_H, (ctx) => {
        ctx.fillStyle = COLORS.cell;
        ctx.fillRect(0, 0, v.width, FILM_H);
        if (!f || !m || !f.cells.length) return;
        const cw = Math.max(8, Math.round((f.cell_w / f.cell_h) * FILM_H));
        for (let x = 0; x < v.width; x += cw) {
          const c = f.cells[nearestCell(f.cells, xToTime(v, x + cw / 2))];
          const url = ready(store.engine.sheetUrl(m.id, c.sheet));
          if (!url) continue;
          let img = sheets.get(url);
          if (!img) {
            img = new Image();
            img.onload = redraw;
            img.src = url;
            sheets.set(url, img);
          }
          if (!img.complete || !img.naturalWidth) continue;
          ctx.drawImage(img, (c.i % f.cols) * f.cell_w, Math.floor(c.i / f.cols) * f.cell_h, f.cell_w, f.cell_h, x, 0, cw, FILM_H);
        }
      });
    };
    redraw();
  });
  return (
    <Row label="frames" height={FILM_H}>
      <div class={`lane${f && !f.done ? ' shimmer' : ''}`} ref={lane} onPointerDown={scrubber(store, () => lane.current)}>
        <canvas ref={canvas} style={{ width: '100%', height: `${FILM_H}px` }} aria-hidden="true" />
      </div>
    </Row>
  );
}
