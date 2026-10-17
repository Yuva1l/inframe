import { useContext, useState } from 'preact/hooks';
import { STATIC_HOST_HINT, SWITCH_ASK, SwitcherContext, type EngineKind } from '../engine/switch';
import { mediaLine, slowmoLabel } from '../model/format';
import type { Store } from '../state';
import { Icon } from './Icons';
import { Modal } from './ShortcutSheet';

function SaveState({ store }: { store: Store }) {
  const s = store.saveState.value;
  if (!store.run.value) return null;
  if (s === 'error') return <button class="saved bad" onClick={() => store.retrySave()}>Not saved - retry</button>;
  if (s === 'conflict') return <button class="saved bad" onClick={() => void store.reloadRun()}>Changed in another tab - Reload</button>;
  return <span class={`saved${s === 'saving' ? ' busy' : ''}`}>{s === 'saving' ? 'Saving...' : 'Saved'}</span>;
}

const LABEL: Record<EngineKind, string> = { local: 'My Mac', browser: 'This browser' };

/** "My Mac" / "This browser" (docs/T9B_DESIGN.md section 1). Draws nothing without a switcher (main.tsx provides one). With a video open,
 *  switching asks first, because the open video closes. On a static host "My Mac" is disabled and `hint` says how to get it. */
export function EngineSwitch({ store, hint }: { store: Store; hint?: boolean }) {
  const sw = useContext(SwitcherContext);
  const [ask, setAsk] = useState<EngineKind | null>(null);
  if (!sw) return null;
  const now = sw.kind.value;
  const busy = sw.switching.value;
  const choose = (k: EngineKind) => {
    if (k === now || busy) return;
    if (store.media.value) setAsk(k);
    else void sw.select(k);
  };
  const off = (k: EngineKind) => k === 'local' && !sw.localAvailable;
  return (
    <>
      <div class="engine-switch" role="group" aria-label="Where inframe works" data-testid="engine-switch">
        {(['local', 'browser'] as EngineKind[]).map((k) => (
          <button key={k} type="button" class={k === now ? 'on' : ''} aria-pressed={k === now} disabled={busy || off(k)} data-engine={k}
            title={off(k) ? STATIC_HOST_HINT : k === 'local' ? 'Work on this Mac through inframe serve' : 'Work inside this browser tab'} onClick={() => choose(k)}>
            {LABEL[k]}
          </button>
        ))}
      </div>
      {hint && !sw.localAvailable && <p class="muted small" data-testid="engine-hint">{LABEL.local} is off here. Run <code>inframe serve</code> on your Mac and open the page it prints.</p>}
      {ask && (
        <Modal title="Switch engine?" onClose={() => setAsk(null)}>
          <p data-testid="engine-ask">{SWITCH_ASK}</p>
          <div class="modal-foot btns">
            <button onClick={() => setAsk(null)}>Cancel</button>
            <button class="primary" onClick={() => { const k = ask; setAsk(null); void sw.select(k); }}>Switch to {LABEL[ask]}</button>
          </div>
        </Modal>
      )}
    </>
  );
}

export function TopBar({ store }: { store: Store }) {
  const m = store.media.value;
  const slow = m ? slowmoLabel(m) : null;
  const switchable = useContext(SwitcherContext) !== null;
  return (
    <header class="top">
      <button class="icon ghost only-narrow" aria-label="Show the video panel" onClick={() => (store.drawerLeft.value = !store.drawerLeft.value)}>
        <Icon name="panelLeft" />
      </button>
      <div class="logo"><b />inframe</div>
      {m && (
        <div class="file">
          <strong title={m.path}>{m.name}</strong>
          <span class="muted mono">{mediaLine(m)}</span>
          {slow && <span class="badge">{slow}</span>}
        </div>
      )}
      <span class="spacer" />
      <SaveState store={store} />
      {switchable ? <EngineSwitch store={store} /> : <span class="badge ok">{store.engine.kind === 'local' ? 'Local engine' : 'Browser engine'}</span>}
      <button class="icon ghost" aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)" onClick={() => (store.shortcutsOpen.value = true)}>
        <Icon name="help" />
      </button>
      <button class="icon ghost only-narrow" aria-label="Show the clip panel" onClick={() => (store.drawerRight.value = !store.drawerRight.value)}>
        <Icon name="panelRight" />
      </button>
      {m && <button onClick={() => (store.openScreen.value = true)}>Open...</button>}
      {m && (
        <button class="primary" disabled={store.busy.value && store.job.value?.kind !== 'export'} title={store.busy.value && store.job.value?.kind !== 'export' ? 'A job is running' : 'Export (Cmd+E)'}
          onClick={() => store.dispatch('export')}>Export</button>
      )}
    </header>
  );
}
