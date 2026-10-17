// The engine switch (docs/T9B_DESIGN.md section 1): which engine the page works with, "My Mac" (LocalEngine, `inframe serve`) or
// "This browser" (BrowserEngine). The choice is kept in localStorage `inframe.engine`; storage can be missing or throw (private
// window, blocked site data), and then the choice only lasts for the tab.
import { createContext } from 'preact';
import { signal, type Signal } from '@preact/signals';

export type EngineKind = 'local' | 'browser';

export const CHOICE_KEY = 'inframe.engine';
/** Shown next to the disabled "My Mac" button on a static host (Safari blocks public pages from calling localhost). */
export const STATIC_HOST_HINT = 'Run inframe serve on your Mac and open the page it prints';
export const SWITCH_ASK = 'The open video closes; runs stay with the engine that made them.';

/** Said when the switch is refused because the open video's clip edits could not be saved. */
export const SWITCH_BLOCKED = 'Not switched: your clip edits are not saved yet. Fix the save (or reload the run), then switch.';

export function readChoice(): EngineKind | null {
  try {
    const v = localStorage.getItem(CHOICE_KEY);
    return v === 'local' || v === 'browser' ? v : null;
  } catch {
    return null;
  }
}

export function saveChoice(kind: EngineKind): void {
  try {
    localStorage.setItem(CHOICE_KEY, kind);
  } catch {
    /* the choice just does not last */
  }
}

/** The kept choice when that engine exists here; else My Mac when it exists (the editor's home), else this browser. */
export function pickInitial(localAvailable: boolean, stored: EngineKind | null): EngineKind {
  if (stored === 'browser') return 'browser';
  return localAvailable ? 'local' : 'browser';
}

export interface EngineSwitcher {
  /** the engine in use */
  readonly kind: Signal<EngineKind>;
  /** false on a static host: there is no `inframe serve` to talk to */
  readonly localAvailable: boolean;
  readonly switching: Signal<boolean>;
  /** Make `kind` the engine in use: `apply` rebuilds the page's store (it may answer false to stay), then the choice is kept. A no-op for the engine already in use. */
  select(kind: EngineKind): Promise<void>;
}

export function createSwitcher(o: { localAvailable: boolean; initial: EngineKind; apply: (kind: EngineKind) => Promise<boolean | void> }): EngineSwitcher {
  const kind = signal<EngineKind>(o.initial);
  const switching = signal(false);
  return {
    kind,
    localAvailable: o.localAvailable,
    switching,
    async select(next) {
      if (next === kind.value || switching.value) return;
      if (next === 'local' && !o.localAvailable) return;
      switching.value = true;
      try {
        if ((await o.apply(next)) === false) return; // refused (edits that could not be saved): stay on the engine in use
        kind.value = next;
        saveChoice(next);
      } finally {
        switching.value = false;
      }
    },
  };
}

/** Provided by main.tsx; TopBar and OpenScreen draw the control only when a switcher is there (the tests that build a store alone have none). */
export const SwitcherContext = createContext<EngineSwitcher | null>(null);
