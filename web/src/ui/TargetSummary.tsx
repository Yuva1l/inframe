// The card at the top of step 4: the one active target, what is kept in the other tabs, the zone and the preset it forces
// (docs/T9A_DESIGN.md 1.3).
import { formatClock } from '../model/format';
import { zoneText } from '../model/zone';
import type { Store } from '../state';
import { TAP_PRESET } from '../workflow';

export function TargetSummary({ store }: { store: Store }) {
  const kind = store.targetKind.value;
  const tab = store.tab.value;
  const cls = store.cls.value;
  const taps = store.currentPick.value?.taps ?? [];
  const photos = store.photos.value;
  const confirms = store.photoConfirms.value;
  const zone = store.zone.value;

  let title: string;
  switch (kind) {
    case 'motion': title = 'Motion: no model, fixed camera'; break;
    case 'any': title = 'Anyone: every person'; break;
    case 'class': title = `Anyone: every ${cls}`; break;
    case 'saved': title = store.savedName.value ? `Saved: ${store.savedName.value}` : 'Saved: nothing chosen yet'; break;
    case 'photos': title = photos ? `Photos: ${photos.n_photos} ${photos.n_photos === 1 ? 'photo' : 'photos'}` : 'Photos: none chosen yet'; break;
    default:
      title = tab === 'photos' ? `Photos + ${confirms} confirm ${confirms === 1 ? 'tap' : 'taps'}` : `Tap: ${cls}, ${taps.length} ${taps.length === 1 ? 'tap' : 'taps'}`;
  }
  // taps are kept while another tab is active: say that they are not what Run uses now
  const kept = kind !== 'tap' && taps.length > 0;
  const preset = kind === 'tap' ? `${TAP_PRESET} (set by the taps)` : kind === 'motion' ? 'none (motion mode)' : store.effectivePreset.value;
  return (
    <div class="card summary" data-testid="target-summary" aria-label="Active target">
      <h4>Active target</h4>
      <p class="title" data-testid="summary-title">{title}</p>
      {kind === 'tap' && tab === 'tap' && taps.length > 0 && (
        <div class="chips">
          {taps.map((t, i) => <button key={i} class="chip" onClick={() => store.seek(t.time)}>{formatClock(t.time)}</button>)}
        </div>
      )}
      {kept && <p class="muted small" data-testid="summary-kept">{taps.length} {taps.length === 1 ? 'tap is' : 'taps are'} kept in the Tap tab (not used now).</p>}
      <div class="kv">
        <span>Zone</span><span data-testid="summary-zone">{zone ? `on, ${zoneText(zone)}` : 'off'}</span>
        <span>Preset</span><span data-testid="summary-preset">{preset}</span>
      </div>
    </div>
  );
}
