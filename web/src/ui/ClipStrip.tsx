import { useEffect, useRef } from 'preact/hooks';
import type { Filmstrip } from '../engine/types';
import { formatLength } from '../model/format';
import type { Store } from '../state';
import { useEngineUrl } from './useEngineUrl';

/** What a card shows: the sharp frame for the selected clip, else its cell of the filmstrip sheet. */
function cardImage(store: Store, f: Filmstrip | null, start: number, end: number, sharp: boolean): { url: string; cell?: Filmstrip['cells'][number] } | null {
  const m = store.media.value;
  if (!m) return null;
  const peak = store.peakOf({ start, end });
  const at = peak?.t ?? (start + end) / 2;
  if (sharp) return { url: store.engine.frameUrl(m.id, at, 320) };
  if (!f || !f.cells.length) return null;
  let best = f.cells[0];
  for (const c of f.cells) if (Math.abs(c.t - at) < Math.abs(best.t - at)) best = c;
  return { url: store.engine.sheetUrl(m.id, best.sheet), cell: best };
}

function ClipCard({ store, id, index, start, end, origin, selected, sharp, f }: {
  store: Store; id: string; index: number; start: number; end: number; origin: string; selected: boolean; sharp: boolean; f: Filmstrip | null;
}) {
  const img = cardImage(store, f, start, end, sharp);
  const url = useEngineUrl(store.engine, img?.url);
  let style: Record<string, string> = {};
  if (img && url) {
    if (!img.cell || !f) style = { backgroundImage: `url("${url}")` };
    else {
      const k = 70 / f.cell_h;
      style = {
        backgroundImage: `url("${url}")`,
        backgroundSize: `${f.cols * f.cell_w * k}px ${f.rows * f.cell_h * k}px`,
        backgroundPosition: `${-(img.cell.i % f.cols) * f.cell_w * k}px ${-Math.floor(img.cell.i / f.cols) * f.cell_h * k}px`,
      };
    }
  }
  return (
    <li>
      <button class={`cc${selected ? ' sel' : ''}${origin !== 'detector' ? ' edit' : ''}`} data-clip={id}
        aria-pressed={selected} aria-label={`Clip ${index + 1}, ${formatLength(end - start)}`}
        style={style} onClick={(e) => store.selectClick(id, e.shiftKey)}>
        <span>{index + 1}</span>
        <em>{formatLength(end - start)}</em>
      </button>
    </li>
  );
}

/** One card per clip, in time order: the filmstrip frame at its highest-scoring moment. The selected card loads a sharper frame. */
export function ClipStrip({ store }: { store: Store }) {
  const clips = store.edit.value.clips;
  const sel = store.selection.value;
  const f = store.filmstrip.value;
  const list = useRef<HTMLOListElement>(null);

  useEffect(() => {
    const first = sel[0];
    if (!first) return;
    (list.current?.querySelector(`[data-clip="${first}"]`) as HTMLElement | null)?.scrollIntoView?.({ inline: 'nearest', block: 'nearest' });
  }, [sel[0]]);

  return (
    <section class="strip" aria-label="Clip thumbnails">
      <span class="lbl">CLIPS</span>
      <ol class="cards" data-testid="clip-list" ref={list}>
        {clips.map((c, i) => (
          <ClipCard key={c.id} store={store} id={c.id} index={i} start={c.start} end={c.end} origin={c.origin} selected={sel.includes(c.id)} sharp={sel.length === 1 && sel[0] === c.id} f={f} />
        ))}
      </ol>
    </section>
  );
}
