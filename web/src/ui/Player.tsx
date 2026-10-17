import { useCallback } from 'preact/hooks';
import type { Store } from '../state';
import { containRect, hitBox } from '../model/geometry';
import { Overlay } from './Overlay';
import { useEngineUrl } from './useEngineUrl';
import { ZoneSurface } from './ZoneTool';

export function Player({ store }: { store: Store }) {
  // a stable ref callback: a new function on every render would reattach the video each time
  const ref = useCallback((el: HTMLVideoElement | null) => store.attachVideo(el), [store]);
  const b = store.banner.value;
  const m = store.media.value;
  const picked = store.pickMode.value ? store.pickInfo.value.boxes : null;
  // a click in pick mode taps the smallest numbered box under it
  const onPick = (e: MouseEvent) => {
    if (!picked || !m || store.zoneMode.value !== 'off' || store.tapPending.value !== null) return;
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const f = containRect(r.width, r.height, m.display_w, m.display_h);
    if (!(f.w > 0 && f.h > 0)) return;
    const fx = (e.clientX - r.left - f.x) / f.w;
    const fy = (e.clientY - r.top - f.y) / f.h;
    const i = hitBox(picked.boxes.map((x) => x.box), fx, fy);
    if (i >= 0) void store.tapBox(picked.boxes[i].i);
  };
  return (
    <div class="stage">
      <video ref={ref} class="video" playsInline preload="auto" onClick={() => store.playPause()} aria-label="Video" />
      {picked && m && <PickImage store={store} t={picked.t} />}
      <Overlay store={store} zone={store.zone.value} pickBoxes={store.pickBoxList.value ?? undefined} />
      {picked && <div class={`picklayer${store.tapPending.value !== null ? ' busy' : ''}`} data-testid="pick-layer" aria-busy={store.tapPending.value !== null} onClick={onPick} />}
      <ZoneSurface store={store} />
      {b && (
        <div class={`banner ${b.kind}`} role="status">
          <span>{b.text}</span>
          {b.fraction !== undefined && <progress max={1} value={b.fraction} />}
        </div>
      )}
    </div>
  );
}

/** The frame the pick boxes sit on. */
function PickImage({ store, t }: { store: Store; t: number }) {
  const m = store.media.value;
  const src = useEngineUrl(store.engine, m ? store.engine.pickFrameUrl(m.id, t, store.cls.value) : null);
  return src ? <img class="pickimg" alt="" src={src} /> : null;
}
