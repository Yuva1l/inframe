import { useRef } from 'preact/hooks';
import { clipLabel, formatClock, formatLength } from '../model/format';
import { maxOf } from '../model/presence';
import { timeToX, xToTime } from '../model/timeline';
import type { Store } from '../state';
import { laneX, Row } from './lane';

export const CLIPS_H = 52;
// the click that ends an edge drag must not reach the block: it would select the clip and seek back to its start
const swallow = (e: Event) => e.stopPropagation();
const GHOST_MIN = 0.1; // a stream copy starting more than this early is shown hatched

/** One block per clip. Edges drag to trim; [ and ] move the focused block's start and end by a frame (Shift: a second). */
export function ClipTrack({ store }: { store: Store }) {
  const lane = useRef<HTMLDivElement>(null);
  const v = store.view.value;
  const clips = store.edit.value.clips;
  const sel = store.selection.value;
  const run = store.run.value;
  const s = store.samples.value;
  const d = store.drag.value;
  const fps = store.media.value?.fps || 30;

  const startDrag = (e: PointerEvent, id: string, edge: 'start' | 'end') => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    store.beginTrim(id, edge);
    const at = (cx: number) => xToTime(store.view.value, laneX(lane.current, cx));
    const move = (m: PointerEvent) => store.dragTrim(at(m.clientX), m.altKey);
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      store.endTrim();
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  };

  const onKey = (e: KeyboardEvent, id: string) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      return store.select([id]);
    }
    const left = e.code === 'BracketLeft' || e.key === '[' || e.key === '{';
    const right = e.code === 'BracketRight' || e.key === ']' || e.key === '}';
    if (!left && !right) return;
    e.preventDefault();
    e.stopPropagation();
    const step = e.shiftKey ? 1 : 1 / fps;
    // [ moves the start earlier and ] the end later (growing the clip); Alt turns each around
    if (left) store.nudge(id, 'start', e.altKey ? step : -step);
    else store.nudge(id, 'end', e.altKey ? -step : step);
  };

  const onDouble = (e: MouseEvent) => {
    if ((e.target as HTMLElement).closest('.blk')) return;
    store.addAt(xToTime(store.view.value, laneX(lane.current, e.clientX)));
  };

  let empty = '';
  if (!run) empty = 'Pick a target and press Run. The presence of the target over time will show here.';
  else if (!clips.length) {
    const best = s && s.score.length ? maxOf(s.score) : null;
    empty = `No stretch reached the enter threshold${best !== null && s ? ` (highest score ${best.toFixed(2)} of ${s.enter.toFixed(2)})` : ''}. Try Re-segment, another target or preset.`;
  }

  return (
    <Row label="clips" height={CLIPS_H}>
      <div class="lane clips" ref={lane} role="listbox" aria-label="Clips" aria-orientation="horizontal" onDblClick={onDouble}>
        {empty && <p class="empty">{empty}</p>}
        {clips.map((c, i) => {
          const x0 = timeToX(v, c.start);
          const x1 = timeToX(v, c.end);
          if (x1 < -40 || x0 > v.width + 40) return null;
          const kf = store.keyframeOf(c.start);
          const early = c.start - kf;
          const label = clipLabel(i + 1, c.start, c.end);
          const selected = sel.includes(c.id);
          return (
            <div key={c.id}>
              {early > 1e-3 && (
                <div class="kfghost" style={{ left: `${timeToX(v, kf)}px`, width: `${Math.max(1, x0 - timeToX(v, kf))}px` }}
                  title={`stream copy starts ${early.toFixed(1)} s early at the keyframe`} />
              )}
              <div class={`blk${selected ? ' sel' : ''}${early > GHOST_MIN ? ' hatched' : ''}`} role="button" tabIndex={0} aria-label={label} aria-pressed={selected}
                data-clip={c.id} style={{ left: `${x0}px`, width: `${Math.max(2, x1 - x0)}px` }}
                title={`${formatClock(c.start)} - ${formatClock(c.end)} (${formatLength(c.end - c.start)})${early > GHOST_MIN ? `; stream copy starts ${early.toFixed(1)} s early at the keyframe` : ''}`}
                onClick={(e) => store.selectClick(c.id, e.shiftKey)} onKeyDown={(e) => onKey(e, c.id)}>
                <span class="bn">{i + 1}</span>
                {c.origin !== 'detector' && <i class="dot" title={c.origin} />}
                <span class="handle h-start" data-edge="start" onPointerDown={(e) => startDrag(e, c.id, 'start')} onClick={swallow} />
                <span class="handle h-end" data-edge="end" onPointerDown={(e) => startDrag(e, c.id, 'end')} onClick={swallow} />
              </div>
            </div>
          );
        })}
        {d && (
          <div class="tip mono" style={{ left: `${Math.min(Math.max(timeToX(v, d.t), 60), v.width - 60)}px` }} role="status">
            {formatClock(d.t)} · {formatLength(d.len)}
          </div>
        )}
      </div>
    </Row>
  );
}
