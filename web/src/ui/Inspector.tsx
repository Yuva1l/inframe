import { useState } from 'preact/hooks';
import { formatClock, formatLength, slowmoLine } from '../model/format';
import type { Store } from '../state';
import { ReSegment } from './ReSegment';
import { useEngineUrl } from './useEngineUrl';

/** The clip's peak frame. A key from the browser engine is rendered first; until then (or when it fails) the box stays empty. */
function Poster({ store, url }: { store: Store; url: string }) {
  const src = useEngineUrl(store.engine, url);
  return <img class="poster" alt="" src={src} style={src ? undefined : { visibility: 'hidden' }} onError={(e) => ((e.target as HTMLElement).style.visibility = 'hidden')} />;
}

function SaveAsTarget({ store }: { store: Store }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const slug = store.runs.value.find((r) => r.run_id === store.run.value?.run_id)?.slug ?? '';
  const keepsOnePerson = slug === 'tap' || slug === 'photos' || slug.startsWith('saved-');
  if (!store.caps.value.savedTargets) return null;
  if (!open) {
    return (
      <button class="ghost" disabled={!keepsOnePerson} title={keepsOnePerson ? 'Add crops of the person in this run to a saved target' : 'Only tap, photo and saved runs follow one person'}
        onClick={() => setOpen(true)}>Save as target...</button>
    );
  }
  const go = () => {
    void store.saveAsTarget(name);
    setOpen(false);
  };
  return (
    <form class="row" onSubmit={(e) => { e.preventDefault(); go(); }}>
      <input class="grow" value={name} placeholder="target name" aria-label="Saved target name" onInput={(e) => setName((e.target as HTMLInputElement).value)} />
      <button class="primary" disabled={!name.trim()} type="submit">Save</button>
      <button type="button" class="ghost" onClick={() => setOpen(false)}>Cancel</button>
    </form>
  );
}

/** The selected clip: its times, peak score, where a stream copy starts, origin; the editing buttons; the run's warnings; re-segment. */
export function Inspector({ store }: { store: Store }) {
  const clips = store.edit.value.clips;
  const sel = store.selection.value;
  const run = store.run.value;
  const m = store.media.value;
  const one = sel.length === 1 ? clips.find((c) => c.id === sel[0]) : undefined;
  const n = one ? clips.indexOf(one) + 1 : 0;
  const peak = one ? store.peakOf(one) : null;
  const kf = one ? store.keyframeOf(one.start) : 0;
  const factor = run?.segments.slowmo?.factor;

  return (
    <>
      {one && m ? (
        <section class="card" aria-label="Selected clip">
          <h4>Clip {n} of {clips.length}</h4>
          <Poster store={store} url={store.engine.frameUrl(m.id, peak?.t ?? (one.start + one.end) / 2, 320)} />
          <div class="kv">
            <span>Start</span><span class="mono">{formatClock(one.start)}</span>
            <span>End</span><span class="mono">{formatClock(one.end)}</span>
            <span>Length</span><span class="mono">{formatLength(one.end - one.start)}</span>
            <span>Peak score</span><span class="mono">{peak ? peak.score.toFixed(2) : '-'}</span>
            <span>Copy starts</span><span class="mono">{formatClock(kf)} (keyframe{one.start - kf > 0.1 ? `, ${(one.start - kf).toFixed(1)} s early` : ''})</span>
            <span>Origin</span><span>{one.origin}</span>
          </div>
          {factor && factor > 1 && <p class="mono muted" data-testid="slowmo">{slowmoLine(one.end - one.start, factor)}</p>}
          <div class="btns wrap">
            <button onClick={() => store.dispatch('split')}>Split</button>
            <button onClick={() => store.dispatch('merge')}>Merge with next</button>
            <button onClick={() => store.dispatch('delete')}>Delete</button>
          </div>
          <div class="btns wrap"><SaveAsTarget store={store} /></div>
        </section>
      ) : (
        <section class="card" aria-label="Selected clip">
          <h4>Clip</h4>
          {sel.length > 1 ? (
            <>
              <p>{sel.length} clips selected</p>
              <div class="btns wrap">
                <button onClick={() => store.dispatch('merge')}>Merge them</button>
                <button onClick={() => store.dispatch('delete')}>Delete them</button>
              </div>
            </>
          ) : (
            <p class="muted">Select a clip to see its details. Press ? for the shortcuts.</p>
          )}
        </section>
      )}
      {run && run.edits && !run.edits_base_ok && <div class="warn">These edits were made on an earlier run.</div>}
      {(run?.segments.warnings ?? []).map((w) => <div class="warn" key={w}>{w}</div>)}
      {run && <ReSegment key={run.run_id} store={store} />}
    </>
  );
}
