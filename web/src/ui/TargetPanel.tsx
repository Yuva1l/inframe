// Step 2: who or what to follow (docs/T9A_DESIGN.md 1.3). Four tabs, one active target; switching tabs never deletes taps.
import { useState } from 'preact/hooks';
import { formatClock } from '../model/format';
import type { Store } from '../state';
import type { Tab } from '../workflow';

function ClassSelect({ store, label }: { store: Store; label: string }) {
  const classes = store.session.value?.classes?.length ? store.session.value.classes : ['person'];
  return (
    <label class="field">
      <span class="muted">{label}</span>
      <select aria-label="Class" value={store.cls.value} onChange={(e) => store.setClass((e.target as HTMLSelectElement).value)}>
        {classes.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
    </label>
  );
}

function AnyonePanel({ store }: { store: Store }) {
  const c = store.cls.value;
  return (
    <>
      <ClassSelect store={store} label="Follow every" />
      <p class="muted small">{c === 'person' ? 'Every person in the video counts.' : `Every ${c} in the video counts.`} Use Tap, Saved or Photos to follow one of them.</p>
    </>
  );
}

function People({ store }: { store: Store }) {
  const list = store.people.value;
  const m = store.media.value;
  if (!store.caps.value.people || !m) return null;
  return (
    <div class="people">
      <button disabled={store.busy.value} title={store.busy.value ? 'A job is running' : 'Scan the whole video once for the people in it'} onClick={() => void store.scanPeople()}>
        Who is in it?
      </button>
      {list && list.length === 0 && <p class="muted small">Nobody was seen long enough.</p>}
      {list && list.length > 0 && (
        <ul class="cards3" aria-label="People in the video">
          {list.map((p) => (
            <li key={p.person}>
              <button class="crop" title={`Show them at ${formatClock(p.time)}, then tap them`} onClick={() => store.showPerson(p)}>
                <img alt="" src={store.engine.personUrl(m.id, p.person)} />
                <span>{formatClock(p.time)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function TapPanel({ store }: { store: Store }) {
  const info = store.pickInfo.value;
  const pick = store.currentPick.value;
  const taps = pick?.taps ?? [];
  const c = store.cls.value;
  const pending = store.tapPending.value;
  let status: string;
  if (store.playing.value) status = 'Pause the video to see who can be tapped.';
  else if (info.loading && info.firstLoad) status = 'Loading the detector, about 5 s the first time...';
  else if (info.loading) status = 'Looking...';
  else if (info.boxes && info.boxes.boxes.length === 0) status = c === 'person' ? 'Nobody found in this frame: try another moment.' : `No ${c} found in this frame: try another moment.`;
  else if (info.boxes) status = `${info.boxes.boxes.length} found. Click one in the video to tap it.`;
  else status = 'Move the playhead to a moment where they are clear.';
  return (
    <>
      <ClassSelect store={store} label="Class" />
      <p class="muted small" data-testid="pick-status">{status}</p>
      {info.error && <p class="errline" role="alert">{info.error}</p>}
      <div class="row between">
        <span data-testid="tap-count">{taps.length} {taps.length === 1 ? 'tap' : 'taps'}</span>
        <button class="ghost" disabled={taps.length === 0 || pending !== null} onClick={() => void store.undoTap()}>Undo last tap</button>
      </div>
      {pending !== null && <p class="muted small" data-testid="tap-status" role="status"><span class="spinner" aria-hidden="true" /> Tapping box {pending}, following them for a moment. Wait before the next tap.</p>}
      {taps.length > 0 && (
        <div class="chips" aria-label="Taps">
          {taps.map((t, i) => <button key={i} class="chip" title="Go to this tap" onClick={() => store.seek(t.time)}>{formatClock(t.time)}</button>)}
        </div>
      )}
      <p class="muted small">Tap the same person again in another stretch if they leave and come back.</p>
      {store.engine.kind === 'browser' && (
        <p class="muted small" data-testid="tap-browser-note">
          Browser tap matching is simpler than My Mac: it follows one person by how they look, with no tracker. If it loses them, tap them again where it did.
        </p>
      )}
      <People store={store} />
    </>
  );
}

function SavedPanel({ store }: { store: Store }) {
  const list = store.saved.value;
  const chosen = store.savedName.value;
  if (store.savedError.value) return <p class="errline" role="alert">{store.savedError.value}</p>;
  if (!list) return <p class="muted small">Loading...</p>;
  if (list.length === 0) return <p class="muted small">No saved targets yet. After a run, use Save as target in the inspector, or <code>inframe target add</code>.</p>;
  return (
    <ul class="cards3" aria-label="Saved targets">
      {list.map((t) => {
        const stale = !t.error && !Object.values(t.embeddings ?? {}).some((e) => e.current && e.valid);
        return (
          <li key={t.name}>
            <button class={`crop${chosen === t.name ? ' sel' : ''}`} disabled={!!t.error} aria-pressed={chosen === t.name} title={t.error || t.name}
              onClick={() => store.chooseSaved(t.name)}>
              {t.error ? <span class="errline small">{t.error}</span> : <img alt="" src={store.engine.targetCropUrl(t.name, 0)} onError={(e) => ((e.target as HTMLElement).style.visibility = 'hidden')} />}
              <span>{t.name}{t.n_crops !== undefined ? ` · ${t.n_crops}` : ''}</span>
              {stale && <em class="badge" title="The thumbnails are embedded again on the next run">stale</em>}
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function PhotosPanel({ store }: { store: Store }) {
  const caps = store.caps.value;
  const p = store.photos.value;
  const [name, setName] = useState('');
  const folder = () => (caps.openDialog ? void store.choosePhotos('dialog') : store.browse('folder', (path) => void store.choosePhotos('path', path)));
  return (
    <>
      <div class="btns wrap">
        <button onClick={folder}>Choose folder...</button>
        {caps.photosPicker && <button onClick={() => void store.choosePhotos('photos')}>From Photos...</button>}
        {caps.openDialog && caps.fileBrowser && <button class="ghost" onClick={() => store.browse('folder', (path) => void store.choosePhotos('path', path))}>Browse...</button>}
      </div>
      <p data-testid="photo-count">{p ? `${p.n_photos} ${p.n_photos === 1 ? 'photo' : 'photos'}` : 'No photos chosen yet.'}</p>
      <p class="muted small">Photos of the same person, one per file. They stay on this Mac.</p>
      {p && caps.savedTargets && (
        <form class="row" onSubmit={(e) => { e.preventDefault(); void store.savePhotosAsTarget(name); setName(''); }}>
          <input class="grow" value={name} placeholder="save as target..." aria-label="Name for the saved target" onInput={(e) => setName((e.target as HTMLInputElement).value)} />
          <button type="submit" disabled={!name.trim() || store.busy.value} title={store.busy.value ? 'A job is running' : ''}>Save</button>
        </form>
      )}
    </>
  );
}

/** "Is this you?": up to five candidate crops with time and score; a click writes the confirm into the target. */
export function Candidates({ store }: { store: Store }) {
  const list = store.candidates.value;
  const m = store.media.value;
  if (!m || list.length === 0) return null;
  return (
    <div>
      <p class="muted small">Is this you? Click the right one, then press Run again.</p>
      <ul class="cards3" aria-label="Candidates">
        {list.map((cand, i) => (
          <li key={i}>
            <button class="crop" onClick={() => void store.confirmCandidate(i)}>
              <img alt="" src={store.engine.candidateUrl(m.id, i)} />
              <span>{formatClock(cand.t)} · {cand.score.toFixed(2)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

const TABS: [Tab, string][] = [['any', 'Anyone'], ['tap', 'Tap'], ['saved', 'Saved'], ['photos', 'Photos']];

export function TargetPanel({ store }: { store: Store }) {
  const tab = store.tab.value;
  const caps = store.caps.value;
  // saved targets and photos need the Mac (a folder of photos, the saved-target files): the browser engine hides both; People is hidden by caps.people
  const tabs = TABS.filter(([k]) => (k !== 'saved' || caps.savedTargets) && (k !== 'photos' || caps.openDialog || caps.fileBrowser));
  return (
    <div class="targetpanel">
      <div class="tabs" role="tablist" aria-label="Who to follow">
        {tabs.map(([k, label]) => (
          <button key={k} role="tab" class={`tab${tab === k ? ' on' : ''}`} aria-selected={tab === k} onClick={() => store.setTab(k)}>{label}</button>
        ))}
      </div>
      <div role="tabpanel" class="tabpanel">
        {tab === 'any' && <AnyonePanel store={store} />}
        {tab === 'tap' && <TapPanel store={store} />}
        {tab === 'saved' && <SavedPanel store={store} />}
        {tab === 'photos' && <PhotosPanel store={store} />}
      </div>
    </div>
  );
}
