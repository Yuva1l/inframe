// The built-in file browser (the cross-platform twin of the native dialogs): home folder and volumes, folders and video or audio
// files only. The server never lists hidden entries or ~/Library.
import { useEffect, useState } from 'preact/hooks';
import type { DirListing, Engine } from '../engine/types';
import { formatBytes } from '../model/format';
import { Icon } from './Icons';
import { Modal } from './ShortcutSheet';

export type BrowseMode = 'video' | 'folder' | 'audio';

const TITLES: Record<BrowseMode, string> = { video: 'Open a video', folder: 'Choose a folder', audio: 'Choose a music file' };

export function joinPath(dir: string, name: string): string {
  const sep = dir.includes('\\') && !dir.includes('/') ? '\\' : '/';
  return dir.endsWith(sep) ? dir + name : dir + sep + name;
}

export function FileBrowser({ engine, mode, onPick, onClose }: { engine: Engine; mode: BrowseMode; onPick: (path: string) => void; onClose: () => void }) {
  const [listing, setListing] = useState<DirListing | null>(null);
  const [dir, setDir] = useState<string | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [typed, setTyped] = useState('');

  useEffect(() => {
    let gone = false;
    setError(null);
    engine.listDir(dir, mode).then(
      (l) => {
        if (gone) return;
        setListing(l);
        setTyped(l.dir);
      },
      (e) => !gone && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      gone = true;
    };
  }, [engine, dir, mode]);

  const go = () => {
    const p = typed.trim();
    if (!p) return;
    if (mode === 'folder') setDir(p);
    else engine.listDir(p, mode).then(() => setDir(p), () => onPick(p)); // not a folder: take it as a file path
  };

  return (
    <Modal title={TITLES[mode]} onClose={onClose} wide>
      <form class="pathrow" onSubmit={(e) => { e.preventDefault(); go(); }}>
        <button type="button" class="icon" aria-label="Up one folder" disabled={!listing?.parent} onClick={() => listing?.parent && setDir(listing.parent)}>
          <Icon name="up" />
        </button>
        <input class="input grow" value={typed} placeholder="/path/to/folder or file" aria-label="Path" onInput={(e) => setTyped((e.target as HTMLInputElement).value)} />
        <button type="submit">Go</button>
      </form>
      {listing && listing.roots.length > 0 && (
        <div class="roots">
          {listing.roots.map((r) => (
            <button key={r.path} class="ghost" onClick={() => setDir(r.path)}>{r.name}</button>
          ))}
        </div>
      )}
      {error && <div class="errline" role="alert">{error}</div>}
      <ul class="entries" role="listbox" aria-label="Files">
        {listing?.entries.length === 0 && <li class="muted pad">Nothing to show here.</li>}
        {listing?.entries.map((e) => (
          <li key={e.name}>
            <button class="entry" onClick={() => (e.kind === 'dir' ? setDir(joinPath(listing.dir, e.name)) : onPick(joinPath(listing.dir, e.name)))}>
              <Icon name={e.kind === 'dir' ? 'folder' : 'play'} />
              <span class="grow">{e.name}</span>
              {e.kind !== 'dir' && typeof e.size === 'number' && <span class="muted mono">{formatBytes(e.size)}</span>}
            </button>
          </li>
        ))}
      </ul>
      {mode === 'folder' && listing && (
        <div class="modal-foot">
          <button class="primary" onClick={() => onPick(listing.dir)}>Choose this folder</button>
        </div>
      )}
    </Modal>
  );
}
