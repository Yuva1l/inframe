import { useRef, useState } from 'preact/hooks';
import { SOURCE_URL } from '../model/about';
import type { Store } from '../state';
import { Preflight } from './Preflight';
import { EngineSwitch } from './TopBar';

export function OpenScreen({ store }: { store: Store }) {
  const [path, setPath] = useState('');
  const caps = store.caps.value;
  const busy = store.opening.value;
  const hasMedia = store.media.value !== null;
  const browserMode = store.engine.kind === 'browser';
  // an engine that reads files in the page (the browser engine) takes a File from the input or a drop
  const canReadFile = !!store.engine.openFile;
  const fileInput = useRef<HTMLInputElement>(null);
  return (
    <div class="open-wrap">
      <section class="opencard" aria-label="Open a video">
        <h1>Open a video</h1>
        <p class="muted">The file stays where it is. Nothing is uploaded.</p>
        <EngineSwitch store={store} hint />
        <div class="btnrow">
          {canReadFile && <button class="primary" disabled={busy} onClick={() => fileInput.current?.click()}>Choose a video...</button>}
          {canReadFile && <input ref={fileInput} type="file" accept="video/*,.mov,.mp4,.m4v" hidden data-testid="file-input" aria-label="Choose a video file"
            onChange={(e) => {
              const input = e.target as HTMLInputElement;
              const f = input.files?.[0];
              input.value = ''; // the same file can be chosen again
              if (f) void store.openFile(f);
            }} />}
          {caps.openDialog && <button class="primary" disabled={busy} onClick={() => void store.openDialog()}>Open file...</button>}
          {caps.photosPicker && <button disabled={busy} onClick={() => void store.openFromPhotos()}>From Photos...</button>}
          {caps.fileBrowser && <button class={caps.openDialog ? '' : 'primary'} disabled={busy} onClick={() => store.browse('video')}>Browse...</button>}
          {hasMedia && <button class="ghost" onClick={() => (store.openScreen.value = false)}>Cancel</button>}
        </div>
        {browserMode && !canReadFile && <p class="muted" data-testid="browser-note">Opening a video in this browser is not built yet.</p>}
        {!browserMode && <form class="pathrow" onSubmit={(e) => { e.preventDefault(); if (path.trim()) void store.openPath(path); }}>
          <input class="input grow" value={path} aria-label="Path to a video" placeholder="Paste a full path, for example /Users/you/Movies/clip.mov"
            onInput={(e) => setPath((e.target as HTMLInputElement).value)} />
          <button type="submit" disabled={busy || !path.trim()}>Open</button>
        </form>}
        {(caps.locate || canReadFile) && <div class="dropzone">or drop a video file here</div>}
        {busy && <div class="muted" role="status"><span class="spinner" /> Reading video...</div>}
        {store.openError.value && <div class="errline" role="alert">{store.openError.value}</div>}
        {store.recent.value.length > 0 && (
          <div class="recent">
            <div class="row between">
              <h3>Recent</h3>
              <button class="link" onClick={() => void store.clearRecent()}>Clear</button>
            </div>
            <ul>
              {store.recent.value.map((r) => (
                <li key={r.path}>
                  <button class="entry" title={r.path} onClick={() => void store.openPath(r.path)}>
                    <span class="grow">{r.name}</span>
                    <span class="muted mono ellipsis">{r.path}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
      <section class="firststeps">
        <ol>
          <li><b>Open</b> a video.</li>
          <li><b>Pick</b> who or what to follow.</li>
          <li><b>Run</b> it and fix the clips.</li>
        </ol>
        <Preflight store={store} all />
        <p class="muted">Press <kbd>?</kbd> for shortcuts.</p>
        <p class="muted small" data-testid="source-link"><a href={SOURCE_URL} target="_blank" rel="noopener noreferrer">Source code</a> (Apache-2.0; one file LGPL-2.1)</p>
      </section>
    </div>
  );
}
