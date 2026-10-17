import { useEffect, useRef, useState } from 'preact/hooks';
import type { ResegmentParams, ResegmentResult } from '../engine/types';
import { maxOf } from '../model/presence';
import type { Store } from '../state';

type Key = 'enter' | 'exit' | 'merge_gap' | 'min_len' | 'pad';
const FALLBACK: Record<Key, number> = { enter: 0.5, exit: 0.35, merge_gap: 1.5, min_len: 1, pad: 1 };

const num = (x: unknown, d: number) => (typeof x === 'number' && Number.isFinite(x) ? x : d);

/** Sliders for the segment logic, seeded from the run's own settings; previews the clips they would make, Apply replaces the clip list. */
export function ReSegment({ store }: { store: Store }) {
  const run = store.run.value!;
  const s = store.samples.value;
  const settings = run.segments.settings;
  const seed: Record<Key, number> = {
    enter: num(s?.enter, num(settings.enter, FALLBACK.enter)),
    exit: num(s?.exit, num(settings.exit, FALLBACK.exit)),
    merge_gap: num(s?.merge_gap, num(settings.merge_gap, FALLBACK.merge_gap)),
    min_len: num(s?.min_len, num(settings.min_len, FALLBACK.min_len)),
    pad: num(s?.pad, num(settings.pad, FALLBACK.pad)),
  };
  const top = maxOf(s?.score ?? [], Math.max(1, seed.enter * 2));
  const [open, setOpen] = useState(false);
  const [v, setV] = useState(seed);
  const [preview, setPreview] = useState<ResegmentResult | null>(null);
  const [error, setError] = useState('');
  const first = useRef(true);

  useEffect(() => {
    if (!open) return;
    let stale = false;
    const params: ResegmentParams = { ...v };
    const timer = setTimeout(async () => {
      try {
        const r = await store.engine.resegment(run.run_id, params);
        if (stale) return;
        setPreview(r);
        setError('');
      } catch (e) {
        if (!stale) setError(e instanceof Error ? e.message : String(e));
      }
    }, first.current ? 0 : 200);
    first.current = false;
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  }, [open, v.enter, v.exit, v.merge_gap, v.min_len, v.pad]);

  const slider = (key: Key, label: string, min: number, max: number, step: number, unit = '') => (
    <label class="slider">
      <span class="row between"><span class="muted">{label}</span><span class="mono">{v[key].toFixed(step < 0.1 ? 2 : 1)}{unit}</span></span>
      <input type="range" min={min} max={max} step={step} value={v[key]} aria-label={label}
        onInput={(e) => setV({ ...v, [key]: Number((e.target as HTMLInputElement).value) })} />
    </label>
  );

  const current = store.edit.value.clips;
  const same = !!preview && preview.clips.length === current.length && preview.clips.every(([a, b], i) => Math.abs(a - current[i].start) < 5e-4 && Math.abs(b - current[i].end) < 5e-4);

  return (
    <section class="card resegment">
      <button class="link head" aria-expanded={open} onClick={() => setOpen(!open)}>Re-segment {open ? '▾' : '▸'}</button>
      {open && (
        <>
          {s?.sound === 'snap' && <p class="muted">Re-segment drops the sound snap.</p>}
          {slider('enter', 'Enter', 0, Math.round(top * 100) / 100, 0.01)}
          {slider('exit', 'Exit', 0, Math.round(top * 100) / 100, 0.01)}
          {slider('merge_gap', 'Merge gap', 0, 10, 0.1, ' s')}
          {slider('min_len', 'Min length', 0, 10, 0.1, ' s')}
          {slider('pad', 'Pad', 0, 5, 0.1, ' s')}
          {error && <p class="errline">{error}</p>}
          <div class="row">
            <span class="muted" data-testid="reseg-count">{preview ? `${preview.clips.length} clips${preview.same_as_run ? ' (as the run made them)' : ''}` : '...'}</span>
            <span class="spacer" />
            <button disabled={!preview || same} onClick={() => preview && store.applyIntervals(preview.clips)}>Apply</button>
          </div>
        </>
      )}
    </section>
  );
}
