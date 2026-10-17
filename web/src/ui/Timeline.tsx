import { useEffect, useRef, useState } from 'preact/hooks';
import { MAX_PPS, timeToX, xToTime } from '../model/timeline';
import type { Store } from '../state';
import { ClipTrack } from './ClipTrack';
import { FilmstripTrack } from './FilmstripTrack';
import { Icon } from './Icons';
import { laneX } from './lane';
import { OtherRunBars } from './OtherRunBars';
import { PresenceLane } from './PresenceLane';
import { Ruler } from './Ruler';
import { RunSwitcher } from './RunSwitcher';

const LABEL_W = 92;

function Toolbar({ store }: { store: Store }) {
  const [confirm, setConfirm] = useState(false);
  const diff = store.diff.value;
  const edits = diff.changed + diff.deleted + diff.added;
  const v = store.view.value;
  const b = store.bounds.value;
  const minPps = v.width / Math.max(b.end - b.offset, 1e-6);
  const level = Math.max(MAX_PPS, minPps) > minPps ? Math.log(v.pps / minPps) / Math.log(Math.max(MAX_PPS, minPps) / minPps) : 0;
  const undoTip = store.canUndo.value ? 'Undo (Cmd+Z)' : store.historyNote.value === 'fresh' ? 'Nothing to undo (a new tab starts with an empty history)' : 'Nothing to undo';
  const btn = (label: string, tip: string, icon: Parameters<typeof Icon>[0]['name'], on: () => void, disabled = false) => (
    <button class="icon ghost" aria-label={label} title={tip} disabled={disabled} onClick={on}><Icon name={icon} /></button>
  );
  return (
    <div class="tl-bar">
      {btn('Undo', undoTip, 'undo', () => store.undo(), !store.canUndo.value)}
      {btn('Redo', 'Redo (Shift+Cmd+Z)', 'redo', () => store.redo(), !store.canRedo.value)}
      <span class="sep" />
      {btn('Split', 'Split at the playhead (S)', 'split', () => store.dispatch('split'))}
      {btn('Merge', 'Merge with the next clip (M)', 'merge', () => store.dispatch('merge'))}
      {btn('Delete', 'Delete the selected clips (Delete)', 'trash', () => store.dispatch('delete'))}
      {btn('Add clip', 'Add a 2 s clip at the playhead (N)', 'plus', () => store.dispatch('add'))}
      <span class="sep" />
      <button class={`tog${store.snapOn.value ? ' on' : ''}`} aria-pressed={store.snapOn.value} title="Snap to the playhead, presence edges, clip edges and keyframes (hold Alt while dragging to skip once)"
        onClick={() => (store.snapOn.value = !store.snapOn.value)}><Icon name="magnet" /> Snap</button>
      <span class="sep" />
      <RunSwitcher store={store} />
      <span class="spacer" />
      <span class="muted"><span data-testid="clip-count">{store.edit.value.clips.length}</span> clips</span>
      {diff.text && <span class="badge" data-testid="diff">{diff.text}</span>}
      {edits > 0 && !confirm && <button class="link" onClick={() => setConfirm(true)}>Revert to detector</button>}
      {confirm && (
        <span class="row">
          <span>Undo {edits} edit{edits === 1 ? '' : 's'}?</span>
          <button class="primary" onClick={() => { setConfirm(false); store.revertToDetector(); }}>Revert</button>
          <button class="ghost" onClick={() => setConfirm(false)}>Keep</button>
        </span>
      )}
      <span class="sep" />
      {btn('Zoom out', 'Zoom out (-)', 'zoomOut', () => store.dispatch('zoomOut'))}
      <input class="zoomer" type="range" min={0} max={1} step={0.01} value={level} aria-label="Zoom" onInput={(e) => store.zoomTo(Number((e.target as HTMLInputElement).value))} />
      {btn('Zoom in', 'Zoom in (+)', 'zoomIn', () => store.dispatch('zoomIn'))}
      <button class="ghost" title="Fit the whole video (0)" onClick={() => store.zoomFit()}>Fit</button>
      {btn('Shortcuts', 'Keyboard shortcuts (?)', 'help', () => (store.shortcutsOpen.value = true))}
    </div>
  );
}

function Playhead({ store }: { store: Store }) {
  const x = timeToX(store.view.value, store.time.value);
  useEffect(() => store.follow());
  return (
    <div class="tl-ph" aria-hidden="true">
      {x >= 0 && x <= store.view.value.width && <div class="ph" style={{ transform: `translateX(${x}px)` }} />}
    </div>
  );
}

/** The editing timeline: toolbar, ruler, filmstrip, presence lane, other runs' bars and the clip blocks, with one playhead across them. */
export function Timeline({ store }: { store: Store }) {
  const stack = useRef<HTMLDivElement>(null);

  // the lane width is the stack's width less the label column
  useEffect(() => {
    const el = stack.current;
    if (!el) return;
    const measure = () => store.setTimelineWidth(el.clientWidth - LABEL_W);
    measure();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure);
      return () => window.removeEventListener('resize', measure);
    }
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [store]);

  // Cmd/Ctrl + wheel (and a trackpad pinch) zooms at the pointer; a plain wheel pans
  useEffect(() => {
    const el = stack.current;
    if (!el) return;
    // Firefox wheels report lines (1) or pages (2), Chrome and Safari pixels (0): bring all to pixels
    const px = (d: number, mode: number) => (mode === 1 ? d * 16 : mode === 2 ? d * el.clientWidth : d);
    const onWheel = (e: WheelEvent) => {
      const deltaX = px(e.deltaX, e.deltaMode);
      const deltaY = px(e.deltaY, e.deltaMode);
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const anchor = xToTime(store.view.value, laneX(el, e.clientX) - LABEL_W);
        store.zoom(Math.exp(-deltaY * 0.01), anchor);
        return;
      }
      const dx = Math.abs(deltaX) > Math.abs(deltaY) ? deltaX : deltaY;
      if (dx) {
        e.preventDefault();
        store.panBy(dx);
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [store]);

  return (
    <section class="timeline" aria-label="Timeline">
      <Toolbar store={store} />
      <div class="tl-stack" ref={stack}>
        <Ruler store={store} />
        <FilmstripTrack store={store} />
        <PresenceLane store={store} />
        <OtherRunBars store={store} />
        <ClipTrack store={store} />
        <Playhead store={store} />
      </div>
    </section>
  );
}
