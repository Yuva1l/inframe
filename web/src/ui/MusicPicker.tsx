// Music for the reel (docs/T9A_DESIGN.md 1.7): the songs that ship with inframe come first (listed offline, no search); "more songs": search by mood (CC0 and CC BY unless the CC0 only box is ticked), listen, use; or use a file of your own.
// The only text that leaves the page is the mood field; it is never filled from file, folder or target names.
// "Auto music" asks the server (a job, polled until it ends) for a free song that fits the reel; "Another song" asks for the next rank.
import { useEffect, useRef, useState } from 'preact/hooks';
import { EngineError, type Job, type MusicAutoResult, type MusicTrack } from '../engine/types';
import { formatDuration } from '../model/format';
import type { Store } from '../state';

export interface MusicChoice {
  track_id: string | null;
  file: string | null;
  title: string;
  license: string | null;
  /** an auto pick: the mood ("" = from the reel), the rank the server answered with and the id of the song it showed (the export refuses another) */
  auto?: { mood: string; rank: number; track: string };
}

/** "CC0", "CC BY 4.0" */
export const licenseLabel = (t: { license: string; license_version: string | null }): string =>
  t.license === 'cc0' ? 'CC0' : `CC ${t.license.toUpperCase()}${t.license_version ? ` ${t.license_version}` : ''}`;

/** The server accepts at most 40 characters of letters, digits, space, `_` and `-`. */
export const cleanMood = (s: string): string => s.replace(/[^\w -]/g, '').slice(0, 40);

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export function MusicPicker({ store, runId, clipIds, choice, onChoice, cc0Only, onCc0Only, reelSeconds, autoBusy, onAutoBusy }: {
  store: Store;
  /** the run and the clips the reel will hold (null = all of them): what auto music fits a song to */
  runId: string;
  clipIds: string[] | null;
  choice: MusicChoice | null;
  onChoice: (c: MusicChoice | null) => void;
  cc0Only: boolean;
  onCc0Only: (v: boolean) => void;
  /** the reel's length: the default minimum length of a track */
  reelSeconds: number;
  /** an auto pick is being chosen (the dialog keeps it, so Export and Which clips wait for it) */
  autoBusy: boolean;
  onAutoBusy: (v: boolean) => void;
}) {
  const [mood, setMood] = useState('upbeat');
  const [minS, setMinS] = useState(Math.round(reelSeconds));
  const [tracks, setTracks] = useState<MusicTrack[] | null>(null);
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [playing, setPlaying] = useState<string | null>(null);
  const [autoMood, setAutoMood] = useState('');
  const [auto, setAuto] = useState<MusicAutoResult | null>(null);
  const [autoJob, setAutoJob] = useState<Job | null>(null);
  // the songs that ship with inframe: listed without a search and without the network
  const [library, setLibrary] = useState<MusicTrack[]>([]);
  const [libraryNotice, setLibraryNotice] = useState('');
  const engine = store.engine;
  // the dialog closing ends an auto pick: stop the polling and cancel the server job (it would keep downloading songs and the next Export would get 409 busy)
  const alive = useRef(true);
  const abort = useRef<AbortController | null>(null);
  const jobRef = useRef<Job | null>(null);
  useEffect(() => () => {
    alive.current = false;
    abort.current?.abort();
    const j = jobRef.current;
    if (j && j.state === 'running') void engine.cancel(j.id).catch(() => undefined);
  }, []);
  const trackTitle = (t: { title: string | null }): string => t.title || 'Untitled song';

  // Ask for the song at `rank` (0 = the first pick, +1 = another song); the server answers with the rank it used.
  // `again`: the ask follows a change of the box, the mood or the clips, so the song shown no longer fits and a failure drops it.
  const askAuto = async (rank: number, cc0 = cc0Only, again = false) => {
    const mood = cleanMood(autoMood).trim();
    setBusy(false);
    setError('');
    onAutoBusy(true);
    const ctl = new AbortController();
    abort.current = ctl;
    try {
      const r = await engine.musicAuto(runId, { clip_ids: clipIds, mood, cc0_only: cc0, rank }, (j) => { jobRef.current = j; if (alive.current) setAutoJob(j); }, ctl.signal);
      if (!alive.current) return;
      setAuto(r);
      setNotice(r.notice);
      onChoice({ track_id: null, file: null, auto: { mood, rank: r.rank, track: r.track.id }, title: trackTitle(r.track), license: licenseLabel(r.track) });
    } catch (e) {
      if (!alive.current) return;
      if (again) {
        setAuto(null);
        onChoice(null);
      }
      if (e instanceof EngineError && e.status === 409 && e.extra.error === 'busy') store.busyOrToast(e);
      else {
        store.noteDown(e);
        setError(message(e));
      }
    } finally {
      jobRef.current = null;
      if (abort.current === ctl) abort.current = null;
      if (alive.current) setAutoJob(null);
      onAutoBusy(false);
    }
  };

  // the included songs, again when the CC0 box changes; a server without the route (older inframe) simply shows none
  useEffect(() => {
    let live = true;
    engine.musicLibrary(cc0Only).then((r) => { if (live) { setLibrary(r.tracks); setLibraryNotice(r.notice); } }, () => { if (live) setLibrary([]); });
    return () => { live = false; };
  }, [cc0Only]);

  // other clips need another song (the export fits the song to its own clips)
  const idsKey = JSON.stringify(clipIds);
  const lastIds = useRef(idsKey);
  useEffect(() => {
    if (lastIds.current === idsKey) return;
    lastIds.current = idsKey;
    if (choice?.auto) void askAuto(0, cc0Only, true);
  }, [idsKey]);

  const search = async () => {
    setBusy(true);
    setError('');
    try {
      const r = await engine.musicSearch({ mood: cleanMood(mood), min_s: minS, limit: 10, cc0_only: cc0Only });
      setTracks(r.tracks);
      setNotice(r.notice);
    } catch (e) {
      store.noteDown(e);
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };

  // Play downloads the track into the music cache first, then plays it from the server
  const play = async (t: MusicTrack) => {
    setError('');
    try {
      await engine.musicGet(t.id, cc0Only);
      setPlaying(t.id);
    } catch (e) {
      store.noteDown(e);
      setError(message(e));
    }
  };

  const ownFile = async () => {
    if (store.caps.value.openDialog) {
      try {
        const r = await engine.musicChooseFile();
        if (r) onChoice({ track_id: null, file: r.file, title: r.file.split(/[/\\]/).pop() ?? r.file, license: r.license });
      } catch (e) {
        setError(message(e));
      }
    } else store.browse('audio', (path) => onChoice({ track_id: null, file: path, title: path.split(/[/\\]/).pop() ?? path, license: null }));
  };

  return (
    <div class="music">
      <label class="check">
        <input type="radio" name="music" checked={choice === null} onChange={() => onChoice(null)} /> No music (keep the clips’ sound)
      </label>
      <label class="check">
        <input type="radio" name="music" data-testid="music-auto" checked={!!choice?.auto} disabled={autoBusy}
          onChange={() => { if (!choice?.auto) void askAuto(0); }} /> Auto music: pick a free song that fits the reel (the included songs first)
      </label>
      <label class="field row"><span class="muted">Auto mood</span>
        <input class="grow" maxLength={40} value={autoMood} aria-label="Auto music mood" placeholder="from the reel" disabled={autoBusy}
          onInput={(e) => setAutoMood(cleanMood((e.target as HTMLInputElement).value))} onChange={() => { if (choice?.auto) void askAuto(0, cc0Only, true); }} />
      </label>
      {autoBusy && (
        <p class="muted" role="status" data-testid="auto-busy"><span class="spinner" /> Choosing a song (the included songs need no download; more songs come from the internet)... {autoJob?.message}
          {autoJob && <> <button class="ghost" onClick={() => void engine.cancel(autoJob.id).catch((e) => setError(message(e)))}>Cancel</button></>}
        </p>
      )}
      {choice?.auto && auto && (
        <div class="credit" data-testid="auto-pick">
          <p>
            <strong>{trackTitle(auto.track)}</strong> <span class="muted">{auto.track.creator} · {formatDuration(auto.track.duration_s)} · {licenseLabel(auto.track)}
              {auto.track.bpm !== null && ` · ${Math.round(auto.track.bpm)} BPM`}{auto.track.score !== null && ` · fit ${Math.round(auto.track.score * 100)}%`} · {auto.rank + 1} of {auto.n}</span>
          </p>
          {auto.track.builtin && <p class="small">No song fitted, so this is inframe’s built-in beat (CC0).</p>}
          {auto.track.bundled && <p class="muted small" data-testid="auto-bundled">An included song: it works offline.</p>}
          {auto.warnings.map((w) => <div class="warn" key={w}>{w}</div>)}
          {auto.track.needs_credit && auto.track.credit_text && (
            <>
              <pre class="mono small" data-testid="auto-credit">{auto.track.credit_text}</pre>
              <p class="muted small">Its credit file is written next to the reel.</p>
            </>
          )}
          <div class="row">
            <button onClick={() => setPlaying(auto.track.id)}>Play</button>
            <button data-testid="another-song" disabled={autoBusy} onClick={() => void askAuto(auto.rank + 1)}>Another song</button>
          </div>
        </div>
      )}
      <p class="muted small" data-testid="music-choice">{choice ? `Using: ${choice.title}${choice.license ? ` (${choice.license})` : ''}` : 'Find music or use a file of your own.'}</p>
      {library.length > 0 && (
        <>
          <h4 class="small">Included songs <span class="muted">(no download, they work offline)</span></h4>
          <ul class="tracks" aria-label="Included songs" data-testid="library-tracks">
            {library.map((t) => (
              <li key={t.id}>
                <span class="grow"><strong>{trackTitle(t)}</strong> <span class="muted">{t.creator} · {formatDuration(t.duration_s)}
                  {typeof t.bpm === 'number' && ` · ${Math.round(t.bpm)} BPM`} · {licenseLabel({ license: t.license, license_version: t.license_version ?? null })}</span></span>
                <button onClick={() => setPlaying(t.id)}>Play</button>
                <button class={choice?.track_id === t.id ? 'primary' : ''} onClick={() => onChoice({ track_id: t.id, file: null, title: t.title, license: t.license })}>Use</button>
              </li>
            ))}
          </ul>
          {libraryNotice && <p class="muted small" data-testid="library-notice">{libraryNotice}</p>}
        </>
      )}
      <p class="muted small">More songs: search online (needs the internet).</p>
      <form class="row wrap" onSubmit={(e) => { e.preventDefault(); void search(); }}>
        <input class="grow" maxLength={40} value={mood} aria-label="Mood" placeholder="upbeat" onInput={(e) => setMood(cleanMood((e.target as HTMLInputElement).value))} />
        <label class="field row"><span class="muted">min s</span>
          <input type="number" min={0} step={1} value={minS} aria-label="Minimum length in seconds" onInput={(e) => setMinS(Math.max(0, Number((e.target as HTMLInputElement).value) || 0))} />
        </label>
        <button type="submit" disabled={busy}>Search</button>
      </form>
      <label class="check"><input type="checkbox" checked={cc0Only} disabled={autoBusy}
        onChange={(e) => { const v = (e.target as HTMLInputElement).checked; onCc0Only(v);
          if (choice?.auto) void askAuto(0, v, true);
          else if (v && choice?.track_id && String(choice.license).toLowerCase() !== 'cc0') onChoice(null); // a CC BY pick would fail the export
          if (v) setTracks((ts) => ts && ts.filter((t) => String(t.license).toLowerCase() === 'cc0')); }} /> CC0 only (otherwise CC BY tracks too; their credit file is written)</label>
      {busy && <p class="muted" role="status"><span class="spinner" /> Searching...</p>}
      {error && <p class="errline" role="alert">{error}</p>}
      {tracks && tracks.length === 0 && <p class="muted">No track found. Try another mood.</p>}
      {tracks && tracks.length > 0 && (
        <ul class="tracks" aria-label="Music results">
          {tracks.map((t) => (
            <li key={t.id}>
              <span class="grow"><strong>{t.title}</strong> <span class="muted">{t.creator} · {formatDuration(t.duration_s)} · {String(t.license).toUpperCase()}</span></span>
              <button onClick={() => void play(t)}>Play</button>
              <button class={choice?.track_id === t.id ? 'primary' : ''} onClick={() => onChoice({ track_id: t.id, file: null, title: t.title, license: t.license })}>Use</button>
            </li>
          ))}
        </ul>
      )}
      {playing && <audio controls autoPlay src={engine.musicUrl(playing)} aria-label="Preview" />}
      {notice && <p class="muted small" data-testid="music-notice">{notice}</p>}
      <button class="ghost" onClick={() => void ownFile()}>Use a music file...</button>
    </div>
  );
}
