import { useEffect } from 'preact/hooks';
import type { Store } from './state';
import { actionForKey, type Action } from './model/keys';
import { formatBytes, formatDuration, formatFps, slowmoLabel } from './model/format';
import { ExportDialog } from './ui/ExportDialog';
import { FileBrowser } from './ui/FileBrowser';
import { ClipStrip } from './ui/ClipStrip';
import { Inspector } from './ui/Inspector';
import { Timeline } from './ui/Timeline';
import { OpenScreen } from './ui/OpenScreen';
import { Player } from './ui/Player';
import { JobBar, RerunDialog, RunPanel } from './ui/RunPanel';
import { ShortcutSheet } from './ui/ShortcutSheet';
import { TargetPanel } from './ui/TargetPanel';
import { Toasts } from './ui/Toasts';
import { TopBar } from './ui/TopBar';
import { Transport } from './ui/Transport';
import { ZoneTool } from './ui/ZoneTool';

const REPEATABLE: Action[] = ['stepBack', 'stepFwd', 'stepBack1s', 'stepFwd1s'];

function VideoStep({ store }: { store: Store }) {
  const m = store.media.value!;
  const slow = slowmoLabel(m);
  return (
    <>
      <section class="step">
        <h3><span class="num done">1</span> Video</h3>
        <div class="kv">
          <span>Name</span><span class="ellipsis" title={m.path}>{m.name}</span>
          <span>Length</span><span>{formatDuration(m.duration)}</span>
          <span>Codec</span><span>{m.codec.toUpperCase()} {m.display_w}x{m.display_h}</span>
          <span>Frame rate</span><span>{formatFps(m.fps)}</span>
          <span>Size</span><span>{formatBytes(m.size)}</span>
        </div>
        {slow && <p><span class="badge">{slow}</span></p>}
      </section>
      <section class="step">
        <h3><span class="num">2</span> Target</h3>
        <TargetPanel store={store} />
      </section>
      <section class="step">
        <h3><span class="num">3</span> Zone</h3>
        <ZoneTool store={store} />
      </section>
      <section class="step">
        <h3><span class="num">4</span> Run</h3>
        <RunPanel store={store} />
      </section>
    </>
  );
}

function Editor({ store }: { store: Store }) {
  const m = store.media.value!;
  return (
    <main class={`editor${store.drawerLeft.value ? ' show-left' : ''}${store.drawerRight.value ? ' show-right' : ''}`}>
      <aside class="left"><VideoStep store={store} /></aside>
      <section class="center">
        <Player key={m.id} store={store} />
        <Transport store={store} />
      </section>
      <aside class="right"><Inspector store={store} /></aside>
      <ClipStrip store={store} />
      <JobBar store={store} />
      <Timeline store={store} />
    </main>
  );
}

export function App({ store }: { store: Store }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const a = actionForKey(e, store.dialogOpen.value);
      if (!a) return;
      if (!store.showEditor.value && a !== 'shortcuts' && a !== 'escape') return;
      if (e.repeat && !REPEATABLE.includes(a)) return;
      e.preventDefault();
      store.dispatch(a);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [store]);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    if (store.engine.openFile) void store.openFile(f);
    else if (store.caps.value.locate) void store.openDropped({ name: f.name, size: f.size, lastModified: f.lastModified });
  };
  const fatal = store.fatal.value;
  const browse = store.browserOpen.value;
  return (
    <div class="app" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      {store.serverDown.value && <div class="downbar" role="alert">inframe serve is not running. Start it again in the terminal.</div>}
      <TopBar store={store} />
      {fatal ? (
        <div class="open-wrap"><section class="opencard" role="alert"><h1>inframe</h1><p class="errline">{fatal}</p></section></div>
      ) : store.showEditor.value ? (
        <Editor store={store} />
      ) : (
        <OpenScreen store={store} />
      )}
      <Toasts store={store} />
      {store.shortcutsOpen.value && <ShortcutSheet store={store} />}
      {store.exportOpen.value && <ExportDialog store={store} />}
      <RerunDialog store={store} />
      {browse && (
        <FileBrowser
          engine={store.engine}
          mode={browse}
          onClose={() => store.closeBrowser()}
          onPick={(p) => store.browsePicked(p)}
        />
      )}
    </div>
  );
}
