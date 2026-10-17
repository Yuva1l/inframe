import { runLabel } from '../model/format';
import type { Store } from '../state';

/** Every run of this video; the chosen one owns the presence lane, the clips and the inspector. */
export function RunSwitcher({ store }: { store: Store }) {
  const runs = store.runs.value;
  const active = store.run.value?.run_id ?? '';
  if (!runs.length) return null;
  return (
    <label class="muted runsw">
      Run{' '}
      <select aria-label="Run" value={active} onChange={(e) => void store.switchRun((e.target as HTMLSelectElement).value)}>
        {runs.map((r) => <option key={r.run_id} value={r.run_id}>{runLabel(r)}</option>)}
      </select>
    </label>
  );
}
