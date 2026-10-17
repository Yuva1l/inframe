// Step 3: draw the zone (docs/T9A_DESIGN.md 1.4). The panel holds the buttons and the overlap choice; ZoneSurface is the drawing layer
// over the player: drag a rectangle, then resize with 8 handles or move by the body. Esc cancels, Delete clears (the hotkey handler).
import { useLayoutEffect, useRef, useState } from 'preact/hooks';
import type { Zone } from '../engine/types';
import { containRect, type Rect } from '../model/geometry';
import { fromDrag, HANDLES, move, resize, zoneText, type Handle } from '../model/zone';
import type { Store } from '../state';

export function ZoneTool({ store }: { store: Store }) {
  const mode = store.zoneMode.value;
  const z = store.zone.value;
  const overlap = store.zoneOverlap.value;
  const m = store.media.value;
  return (
    <div class="zonetool">
      <div class="btns wrap">
        {mode === 'off' ? (
          <>
            <button disabled={!m} onClick={() => store.zoneBegin()}>{z ? 'Draw again' : 'Draw zone'}</button>
            {z && <button onClick={() => store.zoneBegin(true)}>Edit</button>}
            {z && <button class="ghost" onClick={() => store.zoneClear()}>Clear</button>}
          </>
        ) : (
          <>
            <button class="primary" onClick={() => store.zoneDone()}>Done</button>
            <button onClick={() => store.zoneCancel()}>Cancel</button>
            <button class="ghost" onClick={() => store.zoneClear()}>Clear</button>
          </>
        )}
      </div>
      <p class="muted" data-testid="zone-text">
        {mode === 'draw' ? 'Drag a rectangle over the video. Esc cancels.' : z ? `Zone on: ${zoneText(z)}.` : 'No zone: the whole frame counts.'}
      </p>
      {z && (
        <label class="field">
          <span class="muted">Count a person when</span>
          <select aria-label="Zone overlap" value={overlap === null ? 'centre' : 'part'}
            onChange={(e) => (store.zoneOverlap.value = (e.target as HTMLSelectElement).value === 'centre' ? null : 0.5)}>
            <option value="centre">the box centre is inside</option>
            <option value="part">at least N % of the box is inside</option>
          </select>
        </label>
      )}
      {z && overlap !== null && (
        <label class="field row">
          <span class="muted">N %</span>
          <input type="number" min={1} max={100} step={5} aria-label="Zone overlap percent" value={Math.round(overlap * 100)}
            onInput={(e) => {
              const n = Number((e.target as HTMLInputElement).value);
              if (n >= 1 && n <= 100) store.zoneOverlap.value = n / 100;
            }} />
        </label>
      )}
      <p class="muted small">Only people inside the zone count. Needs a fixed camera: if the camera moves, the run ignores the zone and says so.</p>
    </div>
  );
}

type Drag =
  | { kind: 'draw'; anchor: [number, number] }
  | { kind: 'resize'; handle: Handle; from: [number, number]; zone: Zone }
  | { kind: 'move'; from: [number, number]; zone: Zone };

const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);

/** The drawing layer over the player. Pointer positions are turned into frame fractions with the same object-fit: contain maths
 *  as the overlay (the server's display size, never videoWidth/videoHeight). */
export function ZoneSurface({ store }: { store: Store }) {
  const el = useRef<HTMLDivElement>(null);
  const [frame, setFrame] = useState<Rect>({ x: 0, y: 0, w: 0, h: 0 });
  const m = store.media.value;
  const mode = store.zoneMode.value;
  const z = store.zone.value;

  const measure = (): { left: number; top: number; f: Rect } | null => {
    const r = el.current?.getBoundingClientRect();
    if (!r || !m) return null;
    return { left: r.left, top: r.top, f: containRect(r.width, r.height, m.display_w, m.display_h) };
  };

  useLayoutEffect(() => {
    const m2 = measure();
    if (m2) setFrame(m2.f);
    const node = el.current;
    if (!node || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      const mm = measure();
      if (mm) setFrame(mm.f);
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, [m?.display_w, m?.display_h, mode]);

  if (mode === 'off' || !m) return null;

  const frac = (cx: number, cy: number): [number, number] | null => {
    const mm = measure();
    if (!mm || !(mm.f.w > 0 && mm.f.h > 0)) return null;
    return [(cx - mm.left - mm.f.x) / mm.f.w, (cy - mm.top - mm.f.y) / mm.f.h];
  };

  const begin = (e: PointerEvent, d: (p: [number, number]) => Drag) => {
    if (e.button !== 0) return;
    const p = frac(e.clientX, e.clientY);
    if (!p) return;
    e.preventDefault();
    e.stopPropagation();
    const drag = d(p);
    const onMove = (ev: PointerEvent) => {
      const q = frac(ev.clientX, ev.clientY);
      if (!q) return;
      if (drag.kind === 'draw') {
        const next = fromDrag(drag.anchor.map(clamp01) as [number, number], [clamp01(q[0]), clamp01(q[1])]);
        if (next) store.zone.value = next;
      } else if (drag.kind === 'resize') store.zone.value = resize(drag.zone, drag.handle, q[0] - drag.from[0], q[1] - drag.from[1]);
      else store.zone.value = move(drag.zone, q[0] - drag.from[0], q[1] - drag.from[1]);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      if (store.zone.value && drag.kind === 'draw') store.zoneMode.value = 'edit';
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  };

  const pct = (v: number) => `${v * 100}%`;
  return (
    <div ref={el} class={`zonesurf ${mode}`} data-testid="zone-surface"
      onPointerDown={(e) => mode === 'draw' && begin(e, (p) => ({ kind: 'draw', anchor: p }))}>
      <div class="zoneframe" style={{ left: `${frame.x}px`, top: `${frame.y}px`, width: `${frame.w}px`, height: `${frame.h}px` }}>
        {z && (
          <div class="zonebox" data-testid="zone-box" style={{ left: pct(z[0]), top: pct(z[1]), width: pct(z[2]), height: pct(z[3]) }}
            onPointerDown={(e) => mode === 'edit' && begin(e, (p) => ({ kind: 'move', from: p, zone: z }))}>
            {mode === 'edit' && HANDLES.map((h) => (
              <span key={h} class={`zh zh-${h}`} data-handle={h}
                onPointerDown={(e) => begin(e, (p) => ({ kind: 'resize', handle: h, from: p, zone: z }))} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
