// Step 4: the run controls, the estimate, progress with Cancel, and what happened (docs/T9A_DESIGN.md 1.5).
import type { ComponentChildren } from 'preact';
import type { Job } from '../engine/types';
import { estimateText } from '../model/estimate';
import { formatDuration } from '../model/format';
import type { Store } from '../state';
import { slugOf } from '../workflow';
import { macComparison, Preflight, UseMyMac } from './Preflight';
import { RunSwitcher } from './RunSwitcher';
import { Modal } from './ShortcutSheet';
import { Candidates } from './TargetPanel';
import { TargetSummary } from './TargetSummary';

const KIND_LABEL: Record<Job['kind'], string> = { run: 'Run', rank: 'Finding who it is', people: 'Looking for people', export: 'Export', 'target-add': 'Saving the target', 'music-auto': 'Choosing a song' };

/** Seconds left from the rate so far; shown after 5 % to avoid wild guesses. */
export function etaSeconds(j: Pick<Job, 'fraction' | 'elapsed_s'>): number | null {
  return j.fraction >= 0.05 && j.fraction < 1 && j.elapsed_s > 0 ? (j.elapsed_s * (1 - j.fraction)) / j.fraction : null;
}

export function JobProgress({ store, kinds }: { store: Store; kinds?: Job['kind'][] }) {
  const j = store.job.value;
  if (!j || j.state !== 'running' || (kinds && !kinds.includes(j.kind))) return null;
  const eta = etaSeconds(j);
  return (
    <div class="progress" data-testid="job-progress">
      <div class="row between">
        <strong>{KIND_LABEL[j.kind]}</strong>
        <span class="mono muted">{Math.round(j.fraction * 100)} %</span>
      </div>
      <progress max={1} value={j.fraction} aria-label={`${KIND_LABEL[j.kind]} progress`} />
      <p class="muted small" data-testid="job-message">
        {j.message || 'Working...'} · {formatDuration(j.elapsed_s)} elapsed{eta !== null ? ` · about ${formatDuration(eta)} left` : ''}
      </p>
      <button onClick={() => void store.cancelJob()}>Cancel</button>
    </div>
  );
}

/** The thin bar over the timeline while any job runs. */
export function JobBar({ store }: { store: Store }) {
  const j = store.job.value;
  if (!j || j.state !== 'running') return null;
  return <div class="jobbar" role="progressbar" aria-label={KIND_LABEL[j.kind]} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(j.fraction * 100)}><i style={{ width: `${Math.round(j.fraction * 100)}%` }} /></div>;
}

function Num({ label, value, onChange, step = 0.1, min = 0, blank }: { label: string; value: number | null; onChange: (v: number | null) => void; step?: number; min?: number; blank?: string }) {
  return (
    <label class="field">
      <span class="muted">{label}</span>
      <input type="number" step={step} min={min} value={value ?? ''} placeholder={blank} aria-label={label}
        onInput={(e) => {
          const raw = (e.target as HTMLInputElement).value;
          if (raw === '') return onChange(blank !== undefined ? null : value);
          const n = Number(raw);
          if (Number.isFinite(n) && n >= min) onChange(n);
        }} />
    </label>
  );
}

function Check({ checked, onChange, children, disabled }: { checked: boolean; onChange: (v: boolean) => void; children: ComponentChildren; disabled?: boolean }) {
  return (
    <label class="check">
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange((e.target as HTMLInputElement).checked)} /> {children}
    </label>
  );
}

function WritesTo({ store }: { store: Store }) {
  const m = store.media.value!;
  const slug = slugOf({ kind: store.targetKind.value, class: store.cls.value, name: store.savedName.value ?? '' });
  const stem = m.name.replace(/\.[^.]+$/, '');
  const dir = m.picked_copy ? '~/Movies/inframe' : m.path.replace(/[/\\][^/\\]*$/, '') || '/';
  return <p class="muted small mono" data-testid="writes-to">Writes to {dir}/{stem}_inframe/{slug}/</p>;
}

export function RunPanel({ store }: { store: Store }) {
  const m = store.media.value!;
  const blockers = store.blockers.value;
  const note = store.jobNote.value;
  const adv = store.adv.value;
  const kind = store.targetKind.value;
  const motion = store.motion.value;
  const running = store.busy.value;
  const sess = store.session.value;
  const presets = sess?.presets?.length ? sess.presets : ['fast', 'balanced', 'accurate'];
  const tapForced = kind === 'tap';
  return (
    <div class="runpanel">
      <TargetSummary store={store} />
      <RunSwitcher store={store} />
      <Preflight store={store} />
      <WritesTo store={store} />
      <p data-testid="estimate">{estimateText(store.estimate.value)}{macComparison(store) ? `, ${macComparison(store)}` : ''}</p>
      <UseMyMac store={store} />

      <label class="field">
        <span class="muted">Preset</span>
        <select aria-label="Preset" value={store.effectivePreset.value} disabled={tapForced || motion} onChange={(e) => (store.preset.value = (e.target as HTMLSelectElement).value)}>
          {presets.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </label>
      {tapForced && <p class="muted small">A tap target runs with the detector it was picked with.</p>}

      {m.capture_fps >= 100 && (
        <div class="slowmo">
          <Check checked={store.slowmoOn.value} onChange={(v) => (store.slowmoOn.value = v)}>Slow down clips (auto, x{store.autoFactor.value})</Check>
          {store.slowmoOn.value && (
            <>
              <label class="field">
                <span class="muted">Factor</span>
                <input type="number" min={1.01} step={0.5} placeholder="auto" aria-label="Slow-motion factor" value={store.slowFactor.value}
                  onInput={(e) => (store.slowFactor.value = (e.target as HTMLInputElement).value)} />
              </label>
              <Check checked={store.slowmoAudio.value} onChange={(v) => (store.slowmoAudio.value = v)}>Keep sound (stretched)</Check>
            </>
          )}
        </div>
      )}

      <details class="advanced">
        <summary>Advanced</summary>
        <div class="advgrid">
          <Num label="Pad (s)" value={adv.pad} onChange={(v) => store.setAdv('pad', v ?? adv.pad)} />
          <Num label="Min length (s)" value={adv.min_len} onChange={(v) => store.setAdv('min_len', v ?? adv.min_len)} />
          <Num label="Merge gap (s)" value={adv.merge_gap} onChange={(v) => store.setAdv('merge_gap', v ?? adv.merge_gap)} />
          <Num label="Enter" value={adv.enter} step={0.05} blank="auto" onChange={(v) => store.setAdv('enter', v)} />
          <Num label="Exit" value={adv.exit} step={0.05} blank="auto" onChange={(v) => store.setAdv('exit', v)} />
          <Num label="Min box" value={adv.min_box} step={0.01} onChange={(v) => store.setAdv('min_box', v ?? adv.min_box)} />
          <Num label="Samples per second" value={adv.fps} step={1} blank="preset" onChange={(v) => store.setAdv('fps', v)} />
          <label class="field">
            <span class="muted">Tracker</span>
            <select aria-label="Tracker" value={adv.tracker} onChange={(e) => store.setAdv('tracker', (e.target as HTMLSelectElement).value)}>
              <option value="ocsort">ocsort</option>
              <option value="botsort">botsort (camera pans)</option>
            </select>
          </label>
          <label class="field">
            <span class="muted">Activity</span>
            <select aria-label="Activity" value={adv.activity} onChange={(e) => store.setAdv('activity', (e.target as HTMLSelectElement).value)}>
              <option value="any">any</option>
              <option value="riding">riding</option>
            </select>
          </label>
        </div>
        <Check checked={motion} onChange={(v) => (store.motion.value = v)}>Motion mode (no model, fixed camera)</Check>
      </details>

      <button class="primary runbtn" disabled={blockers.length > 0} title={running ? 'A job is running' : blockers.join('. ')} onClick={() => void store.startRun()}>Run</button>
      {blockers.length > 0 && !running && <p class="muted small" data-testid="run-blockers">{blockers[0]}</p>}

      <JobProgress store={store} />
      {note?.kind === 'failed' && <div class="errpanel" role="alert">{note.text}</div>}
      {note?.kind === 'cancelled' && <p class="muted" role="status">{note.text}</p>}
      {note?.kind === 'done' && <p class="okline" role="status">{note.text}</p>}
      {note?.kind === 'needs_confirm' && (
        <div class="warn" role="status">
          <p>{note.text}</p>
          {store.photos.value && <button disabled={running} title={running ? 'A job is running' : 'Reads the whole video once, about as long as a run'} onClick={() => void store.findWho()}>Find who it is</button>}
        </div>
      )}
      <Candidates store={store} />
      {store.restore.value && (
        <div class="note" role="status">
          <p>Your earlier edits were kept aside.</p>
          <button onClick={() => store.restoreEdits()}>Restore my earlier edits</button>
        </div>
      )}
    </div>
  );
}

/** "This run has 6 edited clips. Run again: your edits are kept aside and can be restored." */
export function RerunDialog({ store }: { store: Store }) {
  const ask = store.rerunAsk.value;
  if (!ask) return null;
  return (
    <Modal title="Run again?" onClose={() => store.cancelRerun()}>
      <p data-testid="rerun-text">This run has {ask.count} edited {ask.count === 1 ? 'clip' : 'clips'}. Run again: your edits are kept aside and can be restored.</p>
      <div class="modal-foot btns">
        <button onClick={() => store.cancelRerun()}>Cancel</button>
        <button class="primary" onClick={() => void store.startRun(true)}>Run again</button>
      </div>
    </Modal>
  );
}
