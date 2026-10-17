// Export (Cmd+E): clips, a reel with music, edit lists (docs/T9A_DESIGN.md 1.7). Shows what will be written before anything is,
// asks before replacing an earlier export, and lists what was written with its checks afterwards.
import { useMemo, useState } from 'preact/hooks';
import { EngineError, type ExportOutput, type ExportRequest, type PhotosSave } from '../engine/types';
import { EDIT_LISTS, editListIssue, plan } from '../model/exportplan';
import { formatBytes } from '../model/format';
import { BLOB_CAP } from '../model/limits';
import type { Store } from '../state';
import { MusicPicker, type MusicChoice } from './MusicPicker';
import { JobProgress } from './RunPanel';
import { Modal } from './ShortcutSheet';
import { useEngineUrl } from './useEngineUrl';

/** The exported reel; a browser-engine key becomes a blob URL first. */
function ReelPlayer({ store, url }: { store: Store; url: string }) {
  const src = useEngineUrl(store.engine, url);
  return <video class="reelplayer" controls autoPlay src={src} aria-label="Reel" />;
}

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

interface ExportDoc {
  outputs?: ExportOutput[];
  warnings?: string[];
  reel?: { file?: string | null; ok?: boolean; problem?: string; check?: { ok?: boolean; problems?: string[]; lufs?: number };
    /** `credits`: the credits file's name and `credit_text` its text, set when the music's license asks for a credit */
    music?: { credits?: string | null; credit_text?: string | null; title?: string | null; creator?: string | null; auto?: { rank: number; n: number } | null } | null } | null;
  edit_lists?: { file?: string; format?: string; ok?: boolean | null; max_err_frames?: number | null; problem?: string }[];
  clips?: { file?: string; error?: string }[];
  /** set when the export was asked to save to Photos (My Mac): what that did, in a sentence */
  photos?: PhotosSave | null;
}

/** One line per output: ok or FAILED, and the check that belongs to it. */
export function outputRows(doc: ExportDoc): { out: ExportOutput; ok: boolean; check: string }[] {
  return (doc.outputs ?? []).map((out) => {
    let ok = true;
    const checks: string[] = [];
    const clip = doc.clips?.find((c) => c.file === out.name);
    if (clip?.error) {
      ok = false;
      checks.push(clip.error);
    }
    if (out.kind === 'reel' && doc.reel) {
      const r = doc.reel;
      if (r.ok === false) {
        ok = false;
        checks.push(...(r.check?.problems ?? [r.problem ?? 'the reel failed its join check']));
      } else checks.push('joins ok');
      if (typeof r.check?.lufs === 'number') checks.push(`${r.check.lufs} LUFS`);
    }
    const list = doc.edit_lists?.find((l) => l.file === out.name);
    if (list) {
      if (list.ok === false) {
        ok = false;
        checks.push(list.problem ?? 'could not be written');
      } else if (typeof list.max_err_frames === 'number') checks.push(`largest cut error ${list.max_err_frames} frames`);
    }
    return { out, ok, check: checks.join(' · ') };
  });
}

/** "Replace the export from 10:12 today (23 clips, a reel)?" from the export.json the server named in its 409. */
export function replaceText(x: Record<string, unknown> | null): string {
  const when = [x?.created, x?.exported, x?.updated, x?.time].find((v) => typeof v === 'string' || typeof v === 'number');
  const d = typeof when === 'number' ? new Date(when * 1000) : typeof when === 'string' ? new Date(when) : null;
  const at = d && !Number.isNaN(d.getTime()) ? ` from ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}` : '';
  const n = Array.isArray(x?.clips) ? (x!.clips as unknown[]).length : null;
  const bits = [n !== null ? `${n} clips` : null, x?.reel ? 'a reel' : null].filter(Boolean).join(', ');
  return `Replace the export${at}${bits ? ` (${bits})` : ''}?`;
}

export function ExportDialog({ store }: { store: Store }) {
  const run = store.run.value;
  const m = store.media.value;
  const caps = store.caps.value;
  const clips = store.edit.value.clips;
  const sel = store.selection.value;
  const job = store.job.value;
  const [selectedOnly, setSelectedOnly] = useState(sel.length > 0);
  const [doClips, setDoClips] = useState(caps.streamCopyCut);
  const [reel, setReel] = useState(false);
  const [music, setMusic] = useState<MusicChoice | null>(null);
  const [cc0Only, setCc0Only] = useState(false);
  const [autoBusy, setAutoBusy] = useState(false);
  const [beat, setBeat] = useState(false);
  const [beatAutoTicked, setBeatAutoTicked] = useState(false);
  const [lists, setLists] = useState<string[]>([]);
  const [outDir, setOutDir] = useState<string | null>(null);
  /** where the result goes on My Mac: the folder only, or the folder and then the Photos library (add-only, no album) */
  const [dest, setDest] = useState<'folder' | 'photos'>('folder');
  const [error, setError] = useState('');
  const [ask, setAsk] = useState<Record<string, unknown> | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);

  const selected = clips.filter((c) => sel.includes(c.id));
  const factor = run?.segments.slowmo?.factor ?? 1;
  const p = useMemo(
    () => plan({ clips, selected, selectedOnly, factor, keyframeOf: (t) => store.keyframeOf(t), fileSize: m?.size ?? 0, duration: m?.duration ?? 0 }),
    [clips, sel.join(','), selectedOnly, factor, m?.id],
  );
  if (!run || !m) return null;
  // a browser that has no file picker builds the zip in memory (Safari): above the cap it cannot
  const tooBig = caps.output === 'download' && doClips && p.bytes > BLOB_CAP;

  const mine = job && job.kind === 'export' ? job : null;
  const running = mine?.state === 'running';
  const otherRunning = store.busy.value && !running;
  const toPhotos = !!caps.savePhotos && caps.output === 'server-folder' && dest === 'photos';
  const nothing = !(doClips || reel || lists.length) || (toPhotos && !(doClips || reel));
  const close = () => (store.exportOpen.value = false);
  const folder = outDir ?? `${run.dir}/edited/`;

  const go = async (replace = false) => {
    setError('');
    setAsk(null);
    const req: ExportRequest = {
      clip_ids: selectedOnly ? selected.map((c) => c.id) : null,
      out_dir: outDir,
      replace,
      clips: doClips,
      reel,
      music: reel && music ? (music.auto ? { auto: music.auto, cc0_only: cc0Only } : { track_id: music.track_id, file: music.file, cc0_only: cc0Only }) : null,
      beat_sync: reel && !!music && beat,
      edit_lists: lists,
      ...(toPhotos ? { to_photos: true } : {}),
    };
    try {
      await store.startExport(req);
    } catch (e) {
      if (e instanceof EngineError && e.status === 409 && e.extra.error === 'exists') setAsk((e.extra.export as Record<string, unknown>) ?? {});
      else if (e instanceof EngineError && e.status === 409 && e.extra.error === 'busy') {
        const kind = (e.extra.job as { kind?: string } | undefined)?.kind;
        setError(`${kind ? `A ${kind}` : 'Another'} job is running. Wait for it to finish, then export again.`);
        store.busyOrToast(e); // follows the job, so its progress shows
      } else {
        store.noteDown(e);
        setError(message(e));
      }
    }
  };

  // the first auto pick ticks Beat sync (when it is possible); after that the user's choice stays
  const chooseMusic = (c: MusicChoice | null) => {
    setMusic(c);
    if (c?.auto && caps.beatSync && !beatAutoTicked) {
      setBeat(true);
      setBeatAutoTicked(true);
    }
  };
  const toggleList = (k: string, on: boolean) => setLists(on ? [...lists, k] : lists.filter((x) => x !== k));
  const done = mine && mine.state === 'done' ? (mine.result as ExportDoc | null) ?? {} : null;
  const rows = done ? outputRows(done) : [];

  return (
    <Modal title="Export" onClose={close} wide>
      <p class="summaryline" data-testid="export-summary">{p.text}</p>
      {p.count === 0 && <p class="errline" role="alert">Nothing to export: the clip list is empty.</p>}

      <fieldset>
        <legend>Which clips</legend>
        <label class="check"><input type="radio" name="which" checked={!selectedOnly} disabled={autoBusy} onChange={() => setSelectedOnly(false)} /> All ({clips.length})</label>
        <label class="check"><input type="radio" name="which" checked={selectedOnly} disabled={sel.length === 0 || autoBusy} onChange={() => setSelectedOnly(true)} /> Selected only ({sel.length})</label>
      </fieldset>

      {caps.output === 'server-folder' && caps.savePhotos && (
        <fieldset>
          <legend>Save to</legend>
          <label class="check"><input type="radio" name="dest" checked={dest === 'folder'} disabled={autoBusy} onChange={() => setDest('folder')} /> Save to folder</label>
          <label class="check"><input type="radio" name="dest" checked={dest === 'photos'} disabled={autoBusy} onChange={() => setDest('photos')} /> Save to Photos</label>
          {dest === 'photos' && (
            <p class="muted small" data-testid="photos-note">
              Adds the clips and the reel to your Photos library (no album). macOS asks once for permission to add; inframe cannot read your library. The files are also kept in the folder below.
              {' '}<strong>If iCloud Photos is on, Apple uploads everything saved here to iCloud, so this footage leaves your Mac.</strong> Turn iCloud Photos off first if that is not what you want.
            </p>
          )}
        </fieldset>
      )}

      {caps.output === 'server-folder' ? (
        <div class="row">
          <span class="muted">Output folder</span>
          <code class="grow ellipsis" title={folder} data-testid="export-folder">{folder}</code>
          {caps.chooseOutputFolder && <button onClick={() => store.browse('folder', (path) => setOutDir(path))}>Choose folder...</button>}
          {outDir && <button class="ghost" onClick={() => setOutDir(null)}>Default</button>}
        </div>
      ) : (
        <p class="muted small" data-testid="export-zip-note">
          {caps.output === 'granted-folder' ? 'One zip file; you choose where to save it when you press Export.' : 'One zip file; your browser saves it to your downloads.'}
        </p>
      )}
      {tooBig && <p class="errline" role="alert" data-testid="export-too-big">About {formatBytes(p.bytes)}: more than this browser can build in memory ({formatBytes(BLOB_CAP)}). Export fewer clips, or use My Mac.</p>}

      {caps.streamCopyCut && (
        <fieldset>
          <legend><label class="check"><input type="checkbox" checked={doClips} onChange={(e) => setDoClips((e.target as HTMLInputElement).checked)} /> Clips</label></legend>
          <p class="muted small">{caps.output === 'server-folder' ? 'One file per clip' : 'One zip with a file per clip'} (stream copy, no re-encode), a JSON sidecar each, and the LosslessCut CSV. Clips start at the keyframe before the clip; frame-exact cutting comes later.</p>
        </fieldset>
      )}

      {caps.reel && (
        <fieldset>
          <legend><label class="check"><input type="checkbox" checked={reel} onChange={(e) => setReel((e.target as HTMLInputElement).checked)} /> Reel</label></legend>
          {caps.music && (
            <div hidden={!reel}>
              {/* always mounted, so turning Reel off and on keeps the auto pick, its card and the mood */}
              <MusicPicker store={store} runId={run.run_id} clipIds={selectedOnly ? selected.map((c) => c.id) : null} choice={music} onChoice={chooseMusic} cc0Only={cc0Only} onCc0Only={setCc0Only}
                reelSeconds={p.outSeconds} autoBusy={autoBusy} onAutoBusy={setAutoBusy} />
              <label class="check">
                <input type="checkbox" checked={beat} disabled={!caps.beatSync || !music} onChange={(e) => setBeat((e.target as HTMLInputElement).checked)} /> Beat sync
                {!caps.beatSync && <span class="muted"> - needs the music extra: uv sync --extra music</span>}
                {caps.beatSync && <span class="muted"> - re-encodes the reel</span>}
              </label>
            </div>
          )}
          {reel && !caps.music && <p class="muted small">Music is not available here: the reel keeps the clips’ sound.</p>}
        </fieldset>
      )}

      {caps.editLists && (
        <fieldset>
          <legend>Edit lists</legend>
          {EDIT_LISTS.map((f) => {
            const why = editListIssue(f.key, m.fps);
            return (
              <label class="check" key={f.key}>
                <input type="checkbox" checked={lists.includes(f.key)} disabled={!!why} onChange={(e) => toggleList(f.key, (e.target as HTMLInputElement).checked)} /> {f.label}
                {why && <span class="muted"> - {why}</span>}
              </label>
            );
          })}
        </fieldset>
      )}

      {ask && (
        <div class="warn" role="alertdialog" aria-label="Replace the export">
          <p data-testid="replace-text">{replaceText(ask)}</p>
          <div class="btns"><button onClick={() => setAsk(null)}>Cancel</button><button class="primary" onClick={() => void go(true)}>Replace</button></div>
        </div>
      )}
      {error && <p class="errline" role="alert">{error}</p>}
      <JobProgress store={store} kinds={['export']} />
      {mine && mine.state === 'failed' && <p class="errline" role="alert">{mine.error || 'The export failed.'}</p>}
      {mine && mine.state === 'cancelled' && <p class="muted">Export cancelled.</p>}

      {done && (
        <section aria-label="Exported files">
          <h3>Written</h3>
          {(done.warnings ?? []).map((w) => <div class="warn" key={w}>{w}</div>)}
          <ul class="outputs" data-testid="export-outputs">
            {rows.map(({ out, ok, check }) => (
              <li key={out.name}>
                <span class={ok ? 'okline' : 'errline'}>{ok ? 'ok' : 'FAILED'}</span>
                <span class="grow ellipsis" title={out.name}>{out.name}</span>
                <span class="muted mono">{formatBytes(out.size)}</span>
                {check && <span class="muted small">{check}</span>}
                {out.kind === 'reel' && <button onClick={() => setPlaying(playing === out.name ? null : out.name)}>Play</button>}
                {out.kind === 'zip' && out.url && <a href={out.url} download={out.name}>Save again</a>}
                {caps.reveal && <button class="ghost" onClick={() => void store.engine.reveal(run.run_id, out.name).catch((e) => setError(message(e)))}>{store.session.value?.platform === 'darwin' ? 'Reveal in Finder' : 'Show in file manager'}</button>}
              </li>
            ))}
          </ul>
          {done.reel?.music?.auto && done.reel.music.title && (
            <p class="small" data-testid="music-auto-used">Music: {done.reel.music.title}{done.reel.music.creator ? ` by ${done.reel.music.creator}` : ''} (auto pick {done.reel.music.auto.rank + 1} of {done.reel.music.auto.n})</p>
          )}
          {done.reel?.music?.credits && (
            <div class="credit" data-testid="music-credit">
              <p class="small">This music needs a credit. It is written to <strong>{done.reel.music.credits}</strong> next to the reel; put it in the video's description:</p>
              {done.reel.music.credit_text && <pre class="mono small" data-testid="music-credit-text">{done.reel.music.credit_text}</pre>}
            </div>
          )}
          {done.photos && (
            <div data-testid="photos-result">
              <p class={done.photos.status === 'saved' ? 'okline' : 'errline'} role={done.photos.status === 'saved' ? undefined : 'alert'}>{done.photos.message}</p>
              {done.photos.failed.length > 0 && (
                <ul class="small">{done.photos.failed.map((f) => <li key={f.name}>{f.name}: {f.error}</li>)}</ul>
              )}
            </div>
          )}
          {playing && <ReelPlayer store={store} url={store.engine.runFileUrl(run.run_id, playing)} />}
        </section>
      )}

      <div class="modal-foot btns">
        <button onClick={close}>Close</button>
        <button class="primary" disabled={running || otherRunning || autoBusy || nothing || p.count === 0 || tooBig}
          title={running || otherRunning || autoBusy ? 'A job is running' : nothing ? 'Choose clips, a reel or an edit list' : ''} onClick={() => void go(false)}>{toPhotos ? 'Export to Photos' : 'Export'}</button>
      </div>
    </Modal>
  );
}

