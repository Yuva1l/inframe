// The page above the store (docs/T9B_DESIGN.md section 1): one store per engine, rebuilt when the user switches engine. main.tsx and the
// engine-switch tests both build it here.
import { signal } from '@preact/signals';
import { App } from './app';
import type { Engines } from './engine';
import { createSwitcher, pickInitial, readChoice, SWITCH_BLOCKED, SwitcherContext, type EngineKind, type EngineSwitcher } from './engine/switch';
import { createStore, type Store, type StoreOptions } from './state';

export interface Shell {
  /** the store in use (null until `start` ends) */
  current: ReturnType<typeof signal<Store | null>>;
  switcher: EngineSwitcher;
  /** Build the first store, with the engine the kept choice (or the default) names. */
  start(): Promise<Store>;
  Root(): preact.JSX.Element;
}

export interface ShellOptions extends StoreOptions {
  /** The start failed with this message (an expired `inframe serve` link): the first store shows it, My Mac stays the engine "in use" so that
   *  the switch to This browser is open, and that switch builds a store without the message. */
  failed?: string;
}

export function createShell(engines: Pick<Engines, 'local' | 'browser'>, { failed, ...opts }: ShellOptions = {}): Shell {
  const current = signal<Store | null>(null);
  const build = async (kind: EngineKind) => {
    const s = createStore(kind === 'local' && engines.local ? engines.local : engines.browser, opts);
    await s.init();
    return s;
  };
  const initial = failed !== undefined ? 'local' : pickInitial(engines.local !== null, readChoice());
  const switcher = createSwitcher({
    localAvailable: engines.local !== null,
    initial,
    apply: async (next) => {
      const old = current.value;
      if (old) {
        // the open video closes: write its edits first, waiting for a save in flight and for the edits made meanwhile
        await old.settleSave().catch(() => {});
        if (old.saveState.value === 'error' || old.saveState.value === 'conflict') {
          // keep the old store: its edits exist only there
          old.toast(SWITCH_BLOCKED, 'error');
          return false;
        }
      }
      current.value = await build(next);
      old?.dispose();
    },
  });
  const Root = () => (
    <SwitcherContext.Provider value={switcher}>
      {current.value ? <App key={current.value.engine.kind} store={current.value} /> : null}
    </SwitcherContext.Provider>
  );
  const first = async () => {
    if (failed === undefined) return build(initial);
    const s = createStore(engines.browser, opts); // not initialised: it only carries the message
    s.fatal.value = failed;
    return s;
  };
  return { current, switcher, async start() { return (current.value = await first()); }, Root };
}
