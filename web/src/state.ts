// The store: what the page knows (the opened video, the active run, the clip edits, the playhead) and what it can do.
// Built on signals; one store per page, made from an Engine so tests can give it the mock engine.
import { batch, computed, signal } from '@preact/signals';
import { loadPlayable, type PlayableFn } from './engine/playability';
import {
  EditConflict,
  EngineError,
  type EditClip,
  type Engine,
  type Filmstrip,
  type MediaInfo,
  type RecentEntry,
  type RunData,
  type RunSummary,
  type Samples,
  type Session,
} from './engine/types';
import { toFileTime, toMediaTime } from './model/clock';
import {
  add as addClip,
  applyResegment,
  diffSummary,
  fromDetector,
  fromDoc,
  History,
  mergeIds,
  mergeNext,
  remove as removeClips,
  revert as revertClips,
  split as splitClip,
  trim,
  validate,
  type Bounds,
  type EditState,
  type OpResult,
} from './model/edits';
import type { Action } from './model/keys';
import { fit, keyframeStart, snap, zoomAt, MAX_PPS, type View } from './model/timeline';
import { peakIn } from './model/presence';
import { createWorkflow } from './workflow';

export type SaveState = 'saved' | 'saving' | 'error' | 'conflict';
export interface Toast {
  id: number;
  text: string;
  kind: 'info' | 'error' | 'ok';
}
export interface Banner {
  kind: 'info' | 'error';
  text: string;
  /** 0..1 for a progress bar */
  fraction?: number;
}

export interface StoreOptions {
  /** the playability test (tests pass a stub: jsdom has no media pipeline) */
  playable?: PlayableFn;
  autosaveMs?: number;
  retryMs?: number;
  toastMs?: number;
  /** how often a running job is asked for its progress (500 ms; tests lower it) */
  pollMs?: number;
}

const SHUTTLE = [1, 2, 4];
type RVFC = (cb: (now: number, meta: { mediaTime: number }) => void) => number;

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export function createStore(engine: Engine, opts: StoreOptions = {}) {
  const playable = opts.playable ?? loadPlayable;
  const autosaveMs = opts.autosaveMs ?? 400;
  const retryMs = opts.retryMs ?? 3000;

  // ---------------------------------------------------------------- state
  const fatal = signal<string | null>(null);
  const session = signal<Session | null>(null);
  const caps = computed(() => session.value?.caps ?? engine.caps);
  const serverDown = signal(false);
  const recent = signal<RecentEntry[]>([]);

  const media = signal<MediaInfo | null>(null);
  const opening = signal(false);
  const openError = signal<string | null>(null);
  const runs = signal<RunSummary[]>([]);
  const run = signal<RunData | null>(null);
  const samples = signal<Samples | null>(null);

  const edit = signal<EditState>({ clips: [], nextId: 1 });
  const revision = signal(0);
  const selection = signal<string[]>([]);
  const history = new History();
  const historyVersion = signal(0);
  const saveState = signal<SaveState>('saved');

  const time = signal(0); // file time of the playhead
  const playing = signal(false);
  const rate = signal(1);
  const clipsOnly = signal(false);
  const showBoxes = signal(false);
  const clockOffset = signal(0);
  const playSrc = signal<{ url: string; kind: 'original' | 'mp4' | 'proxy' } | null>(null);
  const banner = signal<Banner | null>(null);

  const toasts = signal<Toast[]>([]);
  const shortcutsOpen = signal(false);
  const browserOpen = signal<null | 'video' | 'folder' | 'audio'>(null);
  const openScreen = signal(false);
  const drawerLeft = signal(false);
  const drawerRight = signal(false);

  // timeline: what the visible window shows, snapping, the trim drag in progress, filmstrip and the other runs' samples
  const viewWidth = signal(1000);
  const view = signal<View>(fit(1, 1000, 0));
  const snapOn = signal(true);
  const drag = signal<{ id: string; edge: 'start' | 'end'; t: number; len: number } | null>(null);
  const filmstrip = signal<Filmstrip | null>(null);
  const samplesCache = signal<Record<string, Samples>>({});
  /** 'fresh': this tab has no history for the run; 'restored': it came back from sessionStorage */
  const historyNote = signal<'fresh' | 'restored'>('fresh');

  const bounds = computed<Bounds>(() => ({ offset: media.value?.offset ?? 0, end: media.value?.end_time ?? 0 }));
  const detectorClips = computed(() => (run.value ? fromDetector(run.value.segments.clips, bounds.value).clips : []));
  const diff = computed(() => diffSummary(edit.value.clips, detectorClips.value));
  const canUndo = computed(() => (historyVersion.value, history.canUndo));
  const canRedo = computed(() => (historyVersion.value, history.canRedo));
  const showEditor = computed(() => media.value !== null && !openScreen.value);

  // target, zone, run and export (workflow.ts); the functions it is given are declared below (hoisted)
  const wf = createWorkflow({
    engine, media, time, playing, session, caps, runs, run, edit, saveState, bounds, detectorClips,
    loadRun: (id) => loadRun(id), flushSave: () => flushSave(), commit: (r) => commit(r), pause: () => pause(), seek: (t) => seek(t),
    toast: (t, k) => toast(t, k), noteDown: (e) => noteDown(e), pollMs: opts.pollMs ?? 500,
  });
  const dialogOpen = computed(() => shortcutsOpen.value || browserOpen.value !== null || wf.exportOpen.value || wf.rerunAsk.value !== null);

  // ---------------------------------------------------------------- small helpers
  let toastId = 0;
  function toast(text: string, kind: Toast['kind'] = 'info') {
    const id = ++toastId;
    toasts.value = [...toasts.value, { id, text, kind }];
    const ms = opts.toastMs ?? 4500;
    if (ms > 0) setTimeout(() => dismissToast(id), ms);
  }
  function dismissToast(id: number) {
    toasts.value = toasts.value.filter((t) => t.id !== id);
  }

  let pollTimer: ReturnType<typeof setTimeout> | undefined;
  let polling = false;
  function noteDown(e: unknown) {
    if (!(e instanceof EngineError) || e.status !== 0 || polling) return;
    polling = true;
    serverDown.value = true;
    const poll = async () => {
      try {
        await engine.session();
        polling = false;
        serverDown.value = false;
        if (saveState.value === 'error') scheduleSave();
      } catch {
        pollTimer = setTimeout(poll, retryMs);
      }
    };
    pollTimer = setTimeout(poll, retryMs);
  }

  // ---------------------------------------------------------------- starting and opening
  async function init() {
    try {
      const s = await engine.session();
      session.value = s;
      wf.adoptDefaults(s);
      recent.value = await engine.recent().catch(() => []);
      if (s.start_media) {
        const info = s.media.find((m) => m.id === s.start_media) ?? (await engine.media(s.start_media));
        await openMedia(info);
      }
      wf.resume(s.job);
    } catch (e) {
      if (e instanceof EngineError && e.status === 0) noteDown(e);
      fatal.value = message(e);
    }
  }

  async function openWith(f: () => Promise<MediaInfo | null>, notFound?: string) {
    opening.value = true;
    openError.value = null;
    try {
      const info = await f();
      if (info) await openMedia(info);
      else if (notFound) openError.value = notFound;
    } catch (e) {
      noteDown(e);
      openError.value = message(e);
    } finally {
      opening.value = false;
    }
  }
  const openPath = (path: string) => openWith(() => engine.openPath(path.trim()));
  const openDialog = () => openWith(() => engine.openDialog());
  const openFromPhotos = () => openWith(() => engine.openFromPhotos());
  const openDropped = (f: { name: string; size: number; lastModified: number }) =>
    openWith(() => engine.locate(f), 'Browsers hide where a dropped file lives. Use Open file... instead.');
  /** A File from the file input or a drop; only an engine that reads files in the page (the browser engine) has openFile. */
  const openFile = (f: File) => openWith(() => (engine.openFile ? engine.openFile(f) : Promise.resolve(null)));
  async function clearRecent() {
    recent.value = await engine.clearRecent().catch(() => recent.value);
  }

  // the built-in file browser also picks folders (photos, export) and music files: whoever opens it says what happens with the pick
  let browserCb: ((path: string) => void) | null = null;
  function browse(mode: 'video' | 'folder' | 'audio', cb: ((path: string) => void) | null = null) {
    browserCb = cb;
    browserOpen.value = mode;
  }
  function browsePicked(path: string) {
    const cb = browserCb;
    browserCb = null;
    browserOpen.value = null;
    if (cb) cb(path);
    else void openPath(path);
  }
  function closeBrowser() {
    browserCb = null;
    browserOpen.value = null;
  }

  let sourceToken = 0;
  async function openMedia(info: MediaInfo) {
    await settleSave(); // the pending edits belong to the run being left
    sourceToken++;
    stopReverse();
    batch(() => {
      media.value = info;
      openScreen.value = false;
      banner.value = null;
      playSrc.value = null;
      playing.value = false;
      clockOffset.value = 0;
      time.value = info.offset;
      filmstrip.value = null;
      samplesCache.value = {};
      resetRun();
      wf.resetForMedia();
      zoomFit();
    });
    void loadFilmstrip(info.id);
    engine.recent().then((r) => (recent.value = r)).catch(() => {});
    await loadRuns(info.id);
  }

  function resetRun() {
    epoch++;
    clearTimeout(saveTimer);
    run.value = null;
    runs.value = [];
    samples.value = null;
    edit.value = { clips: [], nextId: 1 };
    revision.value = 0;
    selection.value = [];
    history.clear();
    historyVersion.value++;
    historyNote.value = 'fresh';
    saveState.value = 'saved';
  }

  async function loadRuns(mediaId: string) {
    try {
      const list = await engine.runs(mediaId);
      if (media.value?.id !== mediaId) return;
      runs.value = list;
      if (list.length) await loadRun(list[0].run_id);
    } catch (e) {
      noteDown(e);
      toast(message(e), 'error');
    }
  }

  /** Make a run the active one. `restore`: take its undo history back from sessionStorage when it still fits the stored clips. */
  async function loadRun(runId: string, restore = true) {
    let data!: RunData;
    let s: Samples | null = null;
    // A pending autosave goes to the run it belongs to, before this one replaces it. Edits made while the fetch
    // is out are still the old run's: save them and fetch again rather than overwrite them below.
    for (let tries = 0; tries < 5; tries++) {
      await settleSave();
      [data, s] = await Promise.all([engine.run(runId), engine.samples(runId).catch(() => null)]);
      if (saveState.peek() !== 'saving') break;
    }
    if (media.value && data.media_id !== media.value.id) return;
    batch(() => {
      epoch++;
      clearTimeout(saveTimer);
      run.value = data;
      samples.value = s;
      if (s) samplesCache.value = { ...samplesCache.value, [runId]: s };
      edit.value = data.edits ? fromDoc(data.edits) : fromDetector(data.segments.clips, bounds.value);
      revision.value = data.edits?.revision ?? 0;
      selection.value = [];
      history.clear();
      historyNote.value = 'fresh';
      if (restore) restoreHistory(runId);
      else dropHistory(runId);
      historyVersion.value++;
      saveState.value = 'saved';
    });
    void loadOthers();
  }

  // ---------------------------------------------------------------- undo history in sessionStorage (per tab, per run)
  const HISTORY_BYTES = 2_000_000;
  const historyKey = (runId: string) => `inframe.history.${runId}`;
  const sameClips = (a: readonly EditClip[], b: readonly EditClip[]) =>
    a.length === b.length && a.every((c, i) => c.id === b[i].id && c.start === b[i].start && c.end === b[i].end && c.origin === b[i].origin);
  const isState = (x: unknown): x is EditState =>
    !!x && typeof x === 'object' && Array.isArray((x as EditState).clips) && typeof (x as EditState).nextId === 'number';

  function dropHistory(runId: string) {
    try { sessionStorage.removeItem(historyKey(runId)); } catch { /* storage is optional */ }
  }

  function restoreHistory(runId: string) {
    try {
      const raw = sessionStorage.getItem(historyKey(runId));
      if (!raw) return;
      const o = JSON.parse(raw);
      if (!o || o.v !== 1 || !isState(o.current) || !Array.isArray(o.past) || !Array.isArray(o.future)) return;
      // the stored clips must be the ones the history ends in: otherwise another tab or a re-run moved on and every step is stale
      if (!sameClips(o.current.clips, edit.value.clips) || !o.past.every(isState) || !o.future.every(isState)) return;
      history.past = o.past.slice(-history.cap);
      history.future = o.future.slice(0, history.cap);
      edit.value = { ...edit.value, nextId: Math.max(edit.value.nextId, o.current.nextId) };
      historyNote.value = 'restored';
    } catch { /* unreadable or no storage: start empty */ }
  }

  function persistHistory() {
    const r = run.value;
    if (!r) return;
    try {
      // roughly 70 bytes a clip: keep the arrays short enough that one stringify stays near the cap
      const room = Math.max(1, Math.floor(HISTORY_BYTES / (70 * (edit.value.clips.length + 1))));
      let past = history.past.slice(-room);
      let future = history.future.slice(0, Math.max(0, room - past.length));
      let text = JSON.stringify({ v: 1, current: edit.value, past, future });
      while (text.length > HISTORY_BYTES && (past.length || future.length)) {
        past = past.slice(Math.ceil(past.length / 2));
        future = future.slice(0, Math.floor(future.length / 2));
        text = JSON.stringify({ v: 1, current: edit.value, past, future });
      }
      if (text.length > HISTORY_BYTES) sessionStorage.removeItem(historyKey(r.run_id));
      else sessionStorage.setItem(historyKey(r.run_id), text);
    } catch { /* storage is optional */ }
  }

  /** Another run of this video becomes the active one: the pending edits are saved first. */
  async function switchRun(runId: string) {
    if (run.value?.run_id === runId) return;
    if (saveState.value === 'saving') await flushSave();
    if (saveState.value === 'saving' || saveState.value === 'error') return toast('The edits of this run are not saved yet', 'error');
    try {
      await loadRun(runId);
    } catch (e) {
      noteDown(e);
      toast(message(e), 'error');
    }
  }

  /** Samples of up to three other runs, fetched once, for the thin bars under the presence lane. */
  async function loadOthers() {
    const active = run.value?.run_id;
    const want = runs.value.filter((r) => r.run_id !== active).slice(0, 3).filter((r) => !samplesCache.value[r.run_id]);
    await Promise.all(want.map(async (r) => {
      try {
        const s = await engine.samples(r.run_id);
        samplesCache.value = { ...samplesCache.value, [r.run_id]: s };
      } catch { /* a run without samples has no bar */ }
    }));
  }

  let filmTimer: ReturnType<typeof setTimeout> | undefined;
  async function loadFilmstrip(mediaId: string) {
    clearTimeout(filmTimer);
    try {
      const f = await engine.filmstrip(mediaId);
      if (media.value?.id !== mediaId) return;
      filmstrip.value = f;
      if (!f.done && f.state !== 'failed') filmTimer = setTimeout(() => void loadFilmstrip(mediaId), 1500);
    } catch (e) {
      noteDown(e); // the lane shows grey cells; a missing filmstrip is not worth a toast
    }
  }

  // ---------------------------------------------------------------- the video element
  let video: HTMLVideoElement | null = null;
  let detachVideo: (() => void) | null = null;
  let settingUp = false;

  function attachVideo(el: HTMLVideoElement | null) {
    detachVideo?.();
    detachVideo = null;
    video = el;
    if (!el) return;
    const onPlay = () => (playing.value = true);
    const onPause = () => (playing.value = false);
    const onFallback = () => {
      if (!settingUp && !hasRvfc(el)) onFrame(el.currentTime);
    };
    el.addEventListener('play', onPlay);
    el.addEventListener('pause', onPause);
    el.addEventListener('ended', onPause);
    el.addEventListener('timeupdate', onFallback);
    el.addEventListener('seeked', onFallback);
    let handle = 0;
    if (hasRvfc(el)) {
      const loop = (_now: number, meta: { mediaTime: number }) => {
        if (!settingUp) onFrame(meta.mediaTime);
        handle = (el as unknown as { requestVideoFrameCallback: RVFC }).requestVideoFrameCallback(loop);
      };
      handle = (el as unknown as { requestVideoFrameCallback: RVFC }).requestVideoFrameCallback(loop);
    }
    detachVideo = () => {
      el.removeEventListener('play', onPlay);
      el.removeEventListener('pause', onPause);
      el.removeEventListener('ended', onPause);
      el.removeEventListener('timeupdate', onFallback);
      el.removeEventListener('seeked', onFallback);
      const c = (el as unknown as { cancelVideoFrameCallback?: (h: number) => void }).cancelVideoFrameCallback;
      if (handle && c) c.call(el, handle);
    };
    void prepareSource();
  }

  function hasRvfc(el: HTMLVideoElement): boolean {
    return typeof (el as unknown as { requestVideoFrameCallback?: unknown }).requestVideoFrameCallback === 'function';
  }

  /** A presented frame (or a time update): move the playhead; in "clips only" mode jump over gaps. */
  function onFrame(mediaTime: number) {
    const t = toFileTime(mediaTime, clockOffset.value);
    time.value = t;
    if (!clipsOnly.value || !playing.value || reverseTimer !== null) return;
    const clips = edit.value.clips;
    const tol = 1 / (media.value?.fps || 30); // a frame is presented at its pts, which can be up to a frame before the clip start
    if (clips.some((c) => t >= c.start - tol && t < c.end)) return;
    const next = clips.find((c) => c.start > t);
    if (next) seek(next.start);
    else video?.pause();
  }

  /** Find a source the browser can play: the original, `?as=mp4` for a .mov, then a server-made preview copy. */
  async function prepareSource() {
    const m = media.value;
    const el = video;
    if (!m || !el) return;
    const token = sourceToken;
    const stale = () => token !== sourceToken || media.value?.id !== m.id || video !== el;
    settingUp = true;
    try {
      const tries: { url: string; kind: 'original' | 'mp4' }[] = [{ url: engine.videoUrl(m.id), kind: 'original' }];
      if (/\.mov$/i.test(m.name)) tries.push({ url: engine.videoUrl(m.id, { asMp4: true }), kind: 'mp4' });
      for (const t of tries) {
        const r = await playable(el, t.url, m.offset);
        if (stale()) return;
        if (r.ok) return ready(t.url, t.kind, r.clockOffset);
      }
      if (!caps.value.proxy) {
        banner.value = { kind: 'error', text: 'This browser cannot play this file here, and no preview copy can be made.' };
        return;
      }
      banner.value = { kind: 'info', text: 'This browser cannot play this file here. Making a preview copy (H.264, 720p)...', fraction: 0 };
      let task = await engine.makeProxy(m.id);
      while (task.state === 'running' || task.state === 'none') {
        if (stale()) return;
        banner.value = { kind: 'info', text: 'This browser cannot play this file here. Making a preview copy (H.264, 720p)...', fraction: task.fraction };
        await new Promise((r) => setTimeout(r, 500));
        task = await engine.mediaTask(m.id, 'proxy');
      }
      if (stale()) return;
      // POST /proxy on a copy that is already made answers {state: done} without the url: read it with GET
      if (task.state === 'done' && !task.url) task = await engine.mediaTask(m.id, 'proxy');
      if (stale()) return;
      if (task.state === 'failed' || !task.url) {
        banner.value = { kind: 'error', text: `No preview: ffmpeg failed (${task.error || 'no message'}). Clips and export still work; the timeline shows filmstrip frames only.` };
        return;
      }
      const r = await playable(el, task.url, task.first_frame_file_time ?? m.offset);
      if (stale()) return;
      if (r.ok) return ready(task.url, 'proxy', r.clockOffset);
      banner.value = { kind: 'error', text: `No preview: the preview copy would not play either (${r.error || 'unknown'}).` };
    } catch (e) {
      noteDown(e);
      if (!stale()) banner.value = { kind: 'error', text: `No preview: ${message(e)}` };
    } finally {
      settingUp = false;
    }

    function ready(url: string, kind: 'original' | 'mp4' | 'proxy', offset: number) {
      batch(() => {
        clockOffset.value = offset;
        playSrc.value = { url, kind };
        banner.value = null;
      });
      settingUp = false;
      seek(time.value);
    }
  }

  // ---------------------------------------------------------------- playback
  let reverseTimer: ReturnType<typeof setInterval> | null = null;
  let shuttleIdx = 0;

  function stopReverse() {
    if (reverseTimer !== null) clearInterval(reverseTimer);
    reverseTimer = null;
  }

  function seek(t: number) {
    const b = bounds.value;
    const v = b.end > b.offset ? Math.min(Math.max(t, b.offset), b.end) : t;
    time.value = v;
    if (video && playSrc.value) {
      try {
        video.currentTime = toMediaTime(v, clockOffset.value);
      } catch {
        /* not seekable yet */
      }
    }
  }

  function pause() {
    stopReverse();
    shuttleIdx = 0;
    if (video) {
      if (video.playbackRate !== 1) video.playbackRate = 1;
      try { video.pause(); } catch { /* ignore */ }
    }
    rate.value = 1;
    playing.value = false;
  }

  function play() {
    if (!video || !playSrc.value) return;
    stopReverse();
    const p = video.play() as Promise<void> | undefined;
    p?.catch?.(() => (playing.value = false));
  }

  function playPause() {
    if (playing.value || reverseTimer !== null) pause();
    else play();
  }

  function shuttle(dir: 1 | -1) {
    if (!video || !playSrc.value) return;
    const wasForward = playing.value && reverseTimer === null && dir === 1;
    const wasBack = reverseTimer !== null && dir === -1;
    shuttleIdx = wasForward || wasBack ? Math.min(shuttleIdx + 1, SHUTTLE.length - 1) : 0;
    const r = SHUTTLE[shuttleIdx];
    rate.value = r;
    if (dir === 1) {
      stopReverse();
      video.playbackRate = r;
      play();
    } else {
      try { video.pause(); } catch { /* ignore */ }
      stopReverse();
      playing.value = true;
      let last = performance.now();
      reverseTimer = setInterval(() => {
        const now = performance.now();
        const dt = (now - last) / 1000;
        last = now;
        const t = time.value - dt * r;
        if (t <= bounds.value.offset) {
          seek(bounds.value.offset);
          pause();
        } else seek(t);
      }, 50);
    }
  }

  function setRate(r: number) {
    if (video) video.playbackRate = r;
    rate.value = r;
  }

  function setVolume(v: number) {
    if (video) video.volume = Math.min(Math.max(v, 0), 1);
  }

  function step(frames: number) {
    if (playing.value || reverseTimer !== null) pause();
    const fps = media.value?.fps || 30;
    seek(time.value + frames / fps);
  }

  // ---------------------------------------------------------------- clips
  const sortedClips = () => edit.value.clips;

  function clipAt(t: number) {
    return sortedClips().find((c) => t >= c.start && t < c.end);
  }

  function select(ids: string[], seekToStart = true) {
    selection.value = ids;
    if (seekToStart && ids.length) {
      const c = sortedClips().find((x) => x.id === ids[0]);
      if (c) seek(c.start);
    }
  }

  function gotoClip(dir: -1 | 1) {
    const clips = sortedClips();
    if (!clips.length) return;
    const sel = selection.value.length === 1 ? clips.findIndex((c) => c.id === selection.value[0]) : -1;
    let target;
    if (sel >= 0) target = clips[sel + dir];
    else if (dir < 0) target = [...clips].reverse().find((c) => c.start < time.value - 0.001);
    else target = clips.find((c) => c.start > time.value + 0.001);
    if (target) select([target.id]);
  }

  /** Apply an edit op: record undo, select what it made, schedule the save; or toast why nothing happened. */
  function commit(r: OpResult): boolean {
    if (!r.changed) {
      if (r.reason) toast(r.reason);
      return false;
    }
    history.push(edit.value);
    batch(() => {
      edit.value = r.state;
      historyVersion.value++;
      if (r.ids) selection.value = r.ids;
      historyNote.value = 'restored';
    });
    persistHistory();
    scheduleSave();
    return true;
  }

  /** A click on a clip block or card: plain selects and seeks; Shift selects the range from the first selected clip. */
  function selectClick(id: string, shift: boolean) {
    const clips = sortedClips();
    if (shift && selection.value.length) {
      const a = clips.findIndex((c) => c.id === selection.value[0]);
      const b = clips.findIndex((c) => c.id === id);
      if (a >= 0 && b >= 0) return select(clips.slice(Math.min(a, b), Math.max(a, b) + 1).map((c) => c.id), false);
    }
    select([id]);
  }

  // ---- trimming by dragging an edge: one history step per drag, the player follows the dragged edge
  let dragBase: EditState | null = null;
  let dragCands: number[] = [];
  let seekPending: number | null = null;
  let seekBusy = false;

  /** Seek at most once per animation frame: the first call goes straight through, later ones in the same frame keep only the latest. */
  function seekThrottled(t: number) {
    if (seekBusy) {
      seekPending = t;
      return;
    }
    seek(t);
    seekBusy = true;
    const raf = globalThis.requestAnimationFrame ?? ((cb: () => void) => setTimeout(cb, 16));
    raf(() => {
      seekBusy = false;
      if (seekPending !== null) {
        const p = seekPending;
        seekPending = null;
        seek(p);
      }
    });
  }

  /** Snap targets: the playhead as it was, presence edges, other clips' edges and keyframes. */
  function snapCandidates(excludeId?: string): number[] {
    const out: number[] = [time.value];
    for (const [a, b] of run.value?.segments.presence_intervals ?? []) out.push(a, b);
    for (const c of sortedClips()) if (c.id !== excludeId) out.push(c.start, c.end);
    if (media.value) out.push(...media.value.keyframes);
    return out;
  }

  function beginTrim(id: string, edge: 'start' | 'end') {
    const c = sortedClips().find((x) => x.id === id);
    if (!c) return;
    dragBase = edit.value;
    dragCands = snapCandidates(id);
    drag.value = { id, edge, t: edge === 'start' ? c.start : c.end, len: c.end - c.start };
  }

  /** The pointer is at file time `rawT`. `free`: Alt is held, no snapping this time. */
  function dragTrim(rawT: number, free = false) {
    const d = drag.value;
    if (!dragBase || !d) return;
    const m = media.value;
    const t = free || !snapOn.value ? rawT : snap(rawT, dragCands, view.value.pps, 8, m?.fps, m?.offset ?? 0);
    const r = trim(dragBase, d.id, d.edge, t, bounds.value);
    const state = r.changed ? r.state : dragBase;
    edit.value = state;
    const c = state.clips.find((x) => x.id === d.id);
    if (!c) return;
    const et = d.edge === 'start' ? c.start : c.end;
    drag.value = { id: d.id, edge: d.edge, t: et, len: c.end - c.start };
    seekThrottled(et);
  }

  function endTrim() {
    const base = dragBase;
    dragBase = null;
    drag.value = null;
    if (!base) return;
    if (sameClips(base.clips, edit.value.clips)) return void (edit.value = base);
    history.push(base);
    batch(() => {
      historyVersion.value++;
      historyNote.value = 'restored';
    });
    persistHistory();
    scheduleSave();
  }

  /** `[` and `]` on a focused block: move one edge by `dt` seconds (a frame, or a second with Shift). */
  function nudge(id: string, edge: 'start' | 'end', dt: number) {
    const c = sortedClips().find((x) => x.id === id);
    if (!c) return;
    const from = edge === 'start' ? c.start : c.end;
    if (commit(trim(edit.value, id, edge, from + dt, bounds.value))) {
      const n = sortedClips().find((x) => x.id === id);
      if (n) seek(edge === 'start' ? n.start : n.end);
    }
  }

  function addAt(t: number) {
    commit(addClip(edit.value, t, bounds.value));
  }

  /** "Revert to detector" is an ordinary undoable step. */
  function revertToDetector() {
    commit(revertClips(edit.value, detectorClips.value));
  }

  /** The clip list becomes the intervals of a re-segment preview; one undo step. */
  function applyIntervals(intervals: readonly (readonly [number, number])[]) {
    if (commit(applyResegment(edit.value, intervals, bounds.value))) selection.value = [];
  }

  // ---- the timeline's window
  const span = () => ({ offset: bounds.value.offset, end: bounds.value.end });
  let fitMode = true;

  function zoomFit() {
    fitMode = true;
    const b = bounds.value;
    view.value = fit(b.end - b.offset, viewWidth.value, b.offset);
  }

  function setTimelineWidth(w: number) {
    if (!(w > 0) || w === viewWidth.value) return;
    viewWidth.value = w;
    if (fitMode) return zoomFit();
    view.value = zoomAt({ ...view.value, width: w }, view.value.t0, 1, span());
  }

  /** Zoom by `factor` around file time `anchorT` (the playhead when omitted). */
  function zoom(factor: number, anchorT = time.value) {
    const v = view.value;
    const inside = anchorT >= v.t0 && anchorT <= v.t0 + v.width / v.pps;
    const next = zoomAt(v, inside ? anchorT : v.t0 + v.width / v.pps / 2, factor, span());
    fitMode = next.pps <= viewWidth.value / Math.max(bounds.value.end - bounds.value.offset, 1e-6) + 1e-9;
    view.value = next;
  }

  /** Slider: 0 is the whole video, 1 is 400 px a second, in between on a log scale. */
  function zoomTo(level: number) {
    const b = bounds.value;
    const minPps = viewWidth.value / Math.max(b.end - b.offset, 1e-6);
    const pps = minPps * Math.pow(Math.max(MAX_PPS, minPps) / minPps, Math.min(Math.max(level, 0), 1));
    zoom(pps / view.value.pps);
  }

  function panBy(dx: number) {
    const v = view.value;
    const b = bounds.value;
    const t0 = Math.min(Math.max(v.t0 + dx / v.pps, b.offset), Math.max(b.offset, b.end - v.width / v.pps));
    if (t0 === v.t0) return;
    fitMode = false;
    view.value = { ...v, t0 };
  }

  /** Keep the playhead on screen while the video plays. */
  function follow() {
    const v = view.value;
    const t = time.value;
    if (!playing.value || (t >= v.t0 && t <= v.t0 + v.width / v.pps)) return;
    panBy((t - v.t0 - (v.width / v.pps) * 0.1) * v.pps);
  }

  // ---- facts the panels show
  const peakOf = (c: { start: number; end: number }) => {
    const s = samples.value;
    return s ? peakIn(s.t, s.score, c.start, c.end) : null;
  };
  const keyframeOf = (start: number) => keyframeStart(media.value?.keyframes ?? [], start);

  /** Add crops of this run's person to a saved target (a job; polled here until it ends). */
  async function saveAsTarget(name: string) {
    const r = run.value;
    if (!r || !name.trim()) return;
    try {
      let job = (await engine.startTargetAdd(name.trim(), { from: 'run', run_id: r.run_id })).job;
      toast(`Adding this run's crops to the saved target "${name.trim()}"...`);
      while (job.state === 'running') {
        await new Promise((res) => setTimeout(res, 500));
        job = await engine.job(job.id);
      }
      if (job.state === 'done') toast(`Saved target "${name.trim()}" updated`, 'ok');
      else toast(job.error || `Saving the target ${job.state}`, 'error');
    } catch (e) {
      noteDown(e);
      toast(message(e), 'error');
    }
  }

  function undo() {
    const s = history.undo(edit.value);
    if (!s) return toast('Nothing to undo');
    applyHistory(s);
  }
  function redo() {
    const s = history.redo(edit.value);
    if (!s) return toast('Nothing to redo');
    applyHistory(s);
  }
  function applyHistory(s: EditState) {
    batch(() => {
      edit.value = s;
      historyVersion.value++;
      selection.value = selection.value.filter((id) => s.clips.some((c) => c.id === id));
    });
    persistHistory();
    scheduleSave();
  }

  const selected = () => selection.value.map((id) => sortedClips().find((c) => c.id === id)).filter((c) => !!c);

  function doTrim(edge: 'start' | 'end') {
    const sel = selected();
    if (sel.length !== 1) return toast('Select one clip first');
    commit(trim(edit.value, sel[0]!.id, edge, time.value, bounds.value));
  }

  function doSplit() {
    const sel = selected();
    const hovered = sel.length === 1 ? sel[0]!.id : clipAt(time.value)?.id;
    commit(splitClip(edit.value, time.value, hovered));
  }

  function doMerge() {
    if (selection.value.length > 1) return void commit(mergeIds(edit.value, selection.value));
    const id = selection.value[0] ?? clipAt(time.value)?.id;
    if (!id) return toast('Select a clip first');
    commit(mergeNext(edit.value, id));
  }

  function doDelete() {
    if (!selection.value.length) return toast('Select a clip first');
    commit(removeClips(edit.value, selection.value));
  }

  function dispatch(a: Action) {
    switch (a) {
      case 'playPause': return playPause();
      case 'back': return shuttle(-1);
      case 'forward': return shuttle(1);
      case 'stepBack': return step(-1);
      case 'stepFwd': return step(1);
      case 'stepBack1s': return step(-(media.value?.fps || 30));
      case 'stepFwd1s': return step(media.value?.fps || 30);
      case 'prevClip': return gotoClip(-1);
      case 'nextClip': return gotoClip(1);
      case 'setStart': return doTrim('start');
      case 'setEnd': return doTrim('end');
      case 'split': return doSplit();
      case 'merge': return doMerge();
      case 'delete': return wf.zoneMode.value !== 'off' ? wf.zoneClear() : doDelete();
      case 'add': return void commit(addClip(edit.value, time.value, bounds.value));
      case 'undo': return undo();
      case 'redo': return redo();
      case 'clipsOnly': return void (clipsOnly.value = !clipsOnly.value);
      case 'showBoxes': return void (showBoxes.value = !showBoxes.value);
      case 'export': return wf.openExport();
      case 'shortcuts': return void (shortcutsOpen.value = !shortcutsOpen.value);
      case 'escape':
        if (browserOpen.value) closeBrowser();
        else if (shortcutsOpen.value) shortcutsOpen.value = false;
        else if (wf.rerunAsk.value) wf.cancelRerun();
        else if (wf.exportOpen.value) wf.exportOpen.value = false;
        else if (wf.zoneMode.value !== 'off') wf.zoneCancel();
        else if (selection.value.length) selection.value = [];
        return;
      case 'zoomIn': return zoom(1.5);
      case 'zoomOut': return zoom(1 / 1.5);
      case 'zoomFit': return zoomFit();
    }
  }

  // ---------------------------------------------------------------- autosave
  let saveTimer: ReturnType<typeof setTimeout> | undefined;
  let saving = false;
  let again = false;
  let inflight: Promise<void> | null = null;
  let disposed = false; // after dispose() no save timer is set again
  let epoch = 0; // bumped whenever another run becomes the active one: a save that started before it must not touch the new run

  /** Before the active run is replaced: wait for a save in flight, write what is still pending, and drop the timers (they would hit the new run). */
  async function settleSave() {
    clearTimeout(saveTimer);
    if (inflight) await inflight;
    clearTimeout(saveTimer);
    if (run.value && (saveState.value === 'saving' || saveState.value === 'error')) await flushSave();
    if (inflight) await inflight;
    clearTimeout(saveTimer);
  }

  function scheduleSave() {
    if (disposed || !run.value || saveState.value === 'conflict') return;
    saveState.value = 'saving';
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => void flushSave(), autosaveMs);
  }

  /** Write the clip list now (the autosave timer calls this; tests too). */
  async function flushSave() {
    clearTimeout(saveTimer);
    const r = run.value;
    if (!r) return;
    if (saving) {
      again = true;
      return;
    }
    const snapshot = edit.value;
    const problems = validate(snapshot.clips, bounds.value);
    if (problems.length) {
      saveState.value = 'error';
      toast(`Not saved: ${problems[0]}`, 'error');
      return;
    }
    saving = true;
    const mine = epoch;
    let release!: () => void;
    inflight = new Promise<void>((res) => (release = res));
    try {
      const doc = await engine.saveEdits(r.run_id, revision.value, snapshot.clips);
      if (mine !== epoch) return; // another run is active now: this revision is not its
      revision.value = doc.revision;
      saveState.value = edit.value === snapshot && !again ? 'saved' : 'saving';
      if (edit.value !== snapshot || again) scheduleSave();
    } catch (e) {
      if (mine !== epoch) return;
      if (e instanceof EditConflict) saveState.value = 'conflict';
      else {
        saveState.value = 'error';
        noteDown(e);
        if (!disposed) saveTimer = setTimeout(() => void flushSave(), retryMs);
      }
    } finally {
      saving = false;
      again = false;
      inflight = null;
      release();
    }
  }

  function retrySave() {
    clearTimeout(saveTimer);
    void flushSave();
  }

  /** After a 409: take the stored document and drop the history (every step in it is older than the stored edits). */
  async function reloadRun() {
    if (run.value) await loadRun(run.value.run_id, false);
  }

  return {
    engine, fatal, session, caps, serverDown, recent, media, opening, openError, runs, run, samples,
    edit, revision, selection, historyVersion, saveState, time, playing, rate, clipsOnly, showBoxes, clockOffset, playSrc, banner,
    toasts, shortcutsOpen, browserOpen, openScreen, drawerLeft, drawerRight,
    dialogOpen, bounds, detectorClips, diff, canUndo, canRedo, showEditor,
    viewWidth, view, snapOn, drag, filmstrip, samplesCache, historyNote,
    ...wf, browse, browsePicked, closeBrowser, noteDown,
    selectClick, beginTrim, dragTrim, endTrim, nudge, addAt, revertToDetector, applyIntervals, switchRun,
    setTimelineWidth, zoom, zoomTo, zoomFit, panBy, follow, peakOf, keyframeOf, saveAsTarget,
    init, openPath, openDialog, openFromPhotos, openDropped, openFile, openMedia, clearRecent, loadRun, reloadRun,
    attachVideo, seek, step, setRate, setVolume, playPause, pause, select, dispatch, commit, undo, redo,
    flushSave, settleSave, retrySave, toast, dismissToast,
    dispose() {
      disposed = true;
      wf.dispose();
      clearTimeout(saveTimer);
      clearTimeout(pollTimer);
      clearTimeout(filmTimer);
      stopReverse();
      detachVideo?.();
      engine.close?.();
    },
  };
}

export type Store = ReturnType<typeof createStore>;
