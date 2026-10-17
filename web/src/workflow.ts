// Target, zone, run and export: what the left panel and the export dialog know and do (docs/T9A_DESIGN.md 1.3-1.7). A second factory beside
// the store: it is given the few things of the store it needs (the opened video, the runs, the clip edits) and returns signals and actions
// that createStore passes on, so the UI still imports only the store.
import { batch, computed, effect, signal, type ReadonlySignal, type Signal } from '@preact/signals';
import {
  EngineError,
  type Candidate,
  type Caps,
  type EditClip,
  type Engine,
  type ExportRequest,
  type FrameBoxes,
  type Job,
  type MediaInfo,
  type PhotoSet,
  type PickState,
  type RunData,
  type RunRequest,
  type RunSummary,
  type SavedTarget,
  type Session,
  type Zone,
} from './engine/types';
import { diffSummary, fromDetector, revert, type Bounds, type EditState, type OpResult } from './model/edits';
import { estimate as estimateRun, median, pushTiming, type Estimate } from './model/estimate';
import { browserKindOf, estimate as browserEstimate, suggestLocal, type BrowserEstimate, type LocalSuggestion } from './model/limits';
import { blocking, preflightRows } from './model/preflight';

export type Tab = 'any' | 'tap' | 'saved' | 'photos';
export type ZoneMode = 'off' | 'draw' | 'edit';
/** The detector a tap was picked with: serve_pick.PRESET_NAME. `tap.resolve_preset` refuses another model, so a tap target forces it. */
export const TAP_PRESET = 'balanced';
const TIMINGS_KEY = 'inframe.timings';
const DEBOUNCE_MS = 250;

export interface Advanced {
  pad: number;
  min_len: number;
  merge_gap: number;
  enter: number | null;
  exit: number | null;
  min_box: number;
  tracker: string;
  activity: string;
  fps: number | null;
}

export const ADVANCED_FALLBACK: Advanced = { pad: 1, min_len: 1, merge_gap: 1.5, enter: null, exit: null, min_box: 0, tracker: 'ocsort', activity: 'any', fps: null };

export interface PickInfo {
  boxes: FrameBoxes | null;
  loading: boolean;
  /** the first detector request of this page: the model loads */
  firstLoad: boolean;
  error: string | null;
}

export interface Person {
  person: number;
  time: number;
  seen: number;
}

export interface JobNote {
  kind: 'cancelled' | 'failed' | 'needs_confirm' | 'done';
  text: string;
}

export interface RerunAsk {
  slug: string;
  count: number;
  clips: EditClip[];
}

export interface WorkflowCtx {
  engine: Engine;
  media: Signal<MediaInfo | null>;
  time: Signal<number>;
  playing: Signal<boolean>;
  session: Signal<Session | null>;
  caps: ReadonlySignal<Caps>;
  runs: Signal<RunSummary[]>;
  run: Signal<RunData | null>;
  edit: Signal<EditState>;
  saveState: ReadonlySignal<string>;
  bounds: ReadonlySignal<Bounds>;
  detectorClips: ReadonlySignal<EditClip[]>;
  loadRun(runId: string): Promise<void>;
  flushSave(): Promise<void>;
  commit(r: OpResult): boolean;
  pause(): void;
  seek(t: number): void;
  toast(text: string, kind?: 'info' | 'error' | 'ok'): void;
  noteDown(e: unknown): void;
  pollMs: number;
}

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function readTimings(): Record<string, number[]> {
  try {
    const o = JSON.parse(localStorage.getItem(TIMINGS_KEY) ?? '{}');
    return o && typeof o === 'object' ? o : {};
  } catch {
    return {};
  }
}

function writeTimings(t: Record<string, number[]>) {
  try { localStorage.setItem(TIMINGS_KEY, JSON.stringify(t)); } catch { /* storage is optional */ }
}

const slugPart = (s: string) => s.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48) || 'x'; // serve_runs._slug_part

/** The folder name the server gives a run of this target (docs/T9A_DESIGN.md 2.6). */
export function slugOf(target: RunRequest['target']): string {
  switch (target.kind) {
    case 'any': return 'any-person';
    case 'class': return `class-${slugPart(target.class ?? '')}`;
    case 'saved': return `saved-${slugPart(target.name ?? '')}`;
    case 'motion': return 'motion-ui'; // not 'motion': the CLI default <stem>_inframe/motion (serve_runs.py)
    default: return target.kind;
  }
}

export function createWorkflow(c: WorkflowCtx) {
  const { engine } = c;
  let disposed = false;

  // ---------------------------------------------------------------- the target
  const tab = signal<Tab>('any');
  const cls = signal('person');
  const picks = signal<Record<string, PickState>>({});
  const saved = signal<SavedTarget[] | null>(null);
  const savedError = signal<string | null>(null);
  const savedName = signal<string | null>(null);
  const photos = signal<PhotoSet | null>(null);
  const photoConfirms = signal(0);
  const candidates = signal<Candidate[]>([]);
  const people = signal<Person[] | null>(null);
  const motion = signal(false);
  const pickInfo = signal<PickInfo>({ boxes: null, loading: false, firstLoad: true, error: null });
  /** the number of the box whose tap request is in flight (the server runs the tracker over a couple of seconds); null when none */
  const tapPending = signal<number | null>(null);

  const pickMode = computed(() => tab.value === 'tap' && c.media.value !== null);
  const currentPick = computed(() => picks.value[cls.value] ?? null);
  const tapCount = computed(() => currentPick.value?.taps.length ?? 0);
  const pickBoxList = computed(() => pickInfo.value.boxes?.boxes.map((b) => ({ box: b.box, label: tapPending.value === b.i ? `${b.i}...` : String(b.i) })) ?? null);

  const targetKind = computed<RunRequest['target']['kind']>(() => {
    if (motion.value) return 'motion';
    switch (tab.value) {
      case 'any': return cls.value === 'person' ? 'any' : 'class';
      case 'tap': return 'tap';
      case 'saved': return 'saved';
      case 'photos': return photoConfirms.value > 0 ? 'tap' : 'photos';
    }
  });

  function setPick(st: PickState) {
    picks.value = { ...picks.value, [st.class]: st };
    if (st.zone && !zone.peek()) zone.value = st.zone;
  }

  async function ensurePick(name: string) {
    const m = c.media.value;
    if (!m || picks.peek()[name]) return;
    try {
      setPick(await engine.pickState(m.id, name));
    } catch (e) {
      c.noteDown(e);
      c.toast(message(e), 'error');
    }
  }

  function setTab(t: Tab) {
    if (tab.peek() === t) return;
    tab.value = t;
    if (t === 'tap') {
      if (c.playing.peek()) c.pause();
      void ensurePick(cls.peek());
    } else if (t === 'saved') void loadSaved();
  }

  function setClass(name: string) {
    cls.value = name;
    if (tab.peek() === 'tap') void ensurePick(name);
  }

  async function loadSaved() {
    try {
      saved.value = await engine.savedTargets();
      savedError.value = null;
    } catch (e) {
      c.noteDown(e);
      savedError.value = message(e);
    }
  }

  function chooseSaved(name: string) {
    savedName.value = name;
  }

  // ---- tapping: server frames with numbered boxes while paused in the Tap tab
  let boxTimer: ReturnType<typeof setTimeout> | undefined;
  let boxAbort: AbortController | null = null;
  const stopBoxEffect = effect(() => {
    const on = pickMode.value;
    const t = c.time.value;
    const playing = c.playing.value;
    const name = cls.value;
    const mid = c.media.value?.id;
    const fps = c.media.value?.fps || 30;
    clearTimeout(boxTimer);
    const cur = pickInfo.peek();
    if (!on || !mid || playing) {
      boxAbort?.abort();
      if (cur.boxes || cur.loading || cur.error) pickInfo.value = { ...cur, boxes: null, loading: false, error: null };
      return;
    }
    if (cur.boxes && Math.abs(t - cur.boxes.t) <= 1.5 / fps) return; // still the frame the boxes belong to
    boxAbort?.abort();
    if (cur.boxes || cur.error) pickInfo.value = { ...cur, boxes: null, error: null };
    boxTimer = setTimeout(() => void fetchBoxes(mid, t, name), DEBOUNCE_MS);
  });

  async function fetchBoxes(mid: string, t: number, name: string) {
    boxAbort?.abort();
    const ctl = (boxAbort = new AbortController());
    pickInfo.value = { ...pickInfo.peek(), loading: true, error: null };
    try {
      const b = await engine.boxes(mid, t, name, ctl.signal);
      if (ctl.signal.aborted || disposed) return;
      pickInfo.value = { boxes: b, loading: false, firstLoad: false, error: null };
    } catch (e) {
      if (ctl.signal.aborted || disposed) return;
      if (e instanceof DOMException && e.name === 'AbortError') return;
      if (e instanceof EngineError && e.status === 409 && e.extra.error !== 'models') {
        pickInfo.value = { ...pickInfo.peek(), loading: false }; // superseded by a newer request (another tab): stop the spinner, show nothing
        return;
      } // a missing-models 409 is shown
      c.noteDown(e);
      pickInfo.value = { ...pickInfo.peek(), loading: false, error: message(e) };
    }
  }

  async function tapBox(i: number) {
    const m = c.media.value;
    const b = pickInfo.peek().boxes;
    const hit = b?.boxes.find((x) => x.i === i);
    if (!m || !b || !hit || tapPending.peek() !== null) return; // one tap at a time: a second click while the first runs is dropped
    tapPending.value = i;
    try {
      setPick(await engine.tap(m.id, cls.peek(), b.t, hit.box, hit.i));
    } catch (e) {
      c.noteDown(e);
      c.toast(message(e), 'error');
    } finally {
      tapPending.value = null;
    }
  }

  async function undoTap() {
    const m = c.media.value;
    if (!m || !tapCount.peek() || tapPending.peek() !== null) return;
    try {
      setPick(await engine.undoTap(m.id, cls.peek()));
    } catch (e) {
      c.noteDown(e);
      c.toast(message(e), 'error');
    }
  }

  // ---- photos
  async function choosePhotos(source: 'dialog' | 'photos' | 'path', path?: string) {
    try {
      const p = await engine.choosePhotos(source, path);
      if (!p) return;
      batch(() => {
        photos.value = p;
        photoConfirms.value = 0;
        candidates.value = [];
      });
    } catch (e) {
      c.noteDown(e);
      c.toast(message(e), 'error');
    }
  }

  async function findWho() {
    const m = c.media.value;
    const p = photos.value;
    if (!m || !p) return;
    try {
      const ref = await engine.startRank(m.id, { photos_id: p.id, cls: cls.value });
      await follow(ref.job, (j) => {
        if (j.state === 'done') candidates.value = ((j.result as { candidates?: Candidate[] } | null)?.candidates ?? []).slice(0, 5);
        else if (j.state === 'failed') c.toast(j.error || 'Could not find who it is', 'error');
      });
    } catch (e) {
      busyOrToast(e);
    }
  }

  async function confirmCandidate(i: number) {
    const m = c.media.value;
    if (!m) return;
    try {
      const st = await engine.confirmCandidate(m.id, i, cls.peek());
      photoConfirms.value = st.taps.length;
      jobNote.value = null;
      c.toast('Confirmed. Press Run again.', 'ok');
    } catch (e) {
      c.noteDown(e);
      c.toast(message(e), 'error');
    }
  }

  async function savePhotosAsTarget(name: string) {
    const p = photos.value;
    if (!p || !name.trim()) return;
    try {
      const ref = await engine.startTargetAdd(name.trim(), { from: 'photos', photos_id: p.id });
      await follow(ref.job, async (j) => {
        if (j.state === 'done') {
          c.toast(`Saved target "${name.trim()}" updated`, 'ok');
          await loadSaved();
        } else c.toast(j.error || `Saving the target ${j.state}`, 'error');
      });
    } catch (e) {
      busyOrToast(e);
    }
  }

  async function scanPeople() {
    const m = c.media.value;
    if (!m) return;
    try {
      const ref = await engine.startPeople(m.id);
      await follow(ref.job, (j) => {
        if (j.state !== 'done') return void (j.state === 'failed' && c.toast(j.error || 'The people scan failed', 'error'));
        const list = (j.result as { people?: Record<string, unknown>[] } | null)?.people ?? [];
        people.value = list.map((p) => ({
          person: Number(p.person),
          time: Number(p.thumbnail_time_s ?? 0),
          seen: Number(p.seen_s ?? 0),
        }));
      });
    } catch (e) {
      busyOrToast(e);
    }
  }

  /** A person card: show that person at their best moment in the Tap tab, where one tap makes them the target. */
  function showPerson(p: Person) {
    cls.value = 'person';
    setTab('tap');
    c.seek(p.time);
  }

  // ---------------------------------------------------------------- the zone
  const zone = signal<Zone | null>(null);
  const zoneMode = signal<ZoneMode>('off');
  const zoneOverlap = signal<number | null>(null);
  let zoneBackup: Zone | null = null;

  function zoneBegin(edit = false) {
    zoneBackup = zone.peek();
    zoneMode.value = edit && zoneBackup ? 'edit' : 'draw';
  }
  function zoneDone() {
    if (zoneMode.peek() === 'off') return;
    zoneMode.value = 'off';
    void syncZone();
  }
  function zoneCancel() {
    if (zoneMode.peek() === 'off') return;
    zone.value = zoneBackup;
    zoneMode.value = 'off';
  }
  function zoneClear() {
    zone.value = null;
    zoneMode.value = 'off';
    void syncZone();
  }

  /** In tap mode the zone also goes to the Picker, so target.json carries it as the CLI pick page's does. */
  async function syncZone() {
    const m = c.media.value;
    if (!m || targetKind.peek() !== 'tap') return;
    try {
      setPickKeepZone(await engine.setZone(m.id, cls.peek(), zone.peek()));
    } catch (e) {
      c.noteDown(e);
      c.toast(message(e), 'error');
    }
  }
  function setPickKeepZone(st: PickState) {
    picks.value = { ...picks.value, [st.class]: st };
  }

  // ---------------------------------------------------------------- run settings
  const preset = signal('balanced');
  const slowmoOn = signal(false);
  const slowFactor = signal('');
  const slowmoAudio = signal(false);
  const adv = signal<Advanced>({ ...ADVANCED_FALLBACK });
  const timings = signal<Record<string, number[]>>(readTimings());

  function setAdv<K extends keyof Advanced>(k: K, v: Advanced[K]) {
    adv.value = { ...adv.value, [k]: v };
  }

  /** Defaults from the server (`pipeline.Settings()`), once the session is known. */
  function adoptDefaults(s: Session) {
    const d = s.defaults ?? {};
    const num = (k: string, f: number) => (typeof d[k] === 'number' ? (d[k] as number) : f);
    const opt = (k: string) => (typeof d[k] === 'number' ? (d[k] as number) : null);
    adv.value = {
      pad: num('pad', ADVANCED_FALLBACK.pad), min_len: num('min_len', ADVANCED_FALLBACK.min_len), merge_gap: num('merge_gap', ADVANCED_FALLBACK.merge_gap),
      enter: opt('enter'), exit: opt('exit'), min_box: num('min_box', 0), tracker: typeof d.tracker === 'string' ? d.tracker : 'ocsort',
      activity: typeof d.activity === 'string' ? d.activity : 'any', fps: opt('fps'),
    };
    preset.value = s.default_preset || 'balanced';
  }

  const effectivePreset = computed(() => (targetKind.value === 'tap' ? TAP_PRESET : preset.value));
  const needsIdentity = computed(() => targetKind.value === 'saved' || targetKind.value === 'photos');
  const needsTracking = computed(() => needsIdentity.value || targetKind.value === 'tap' || adv.value.activity === 'riding');
  // the browser is much slower than the Mac and the two keep separate timings
  const presetKey = computed(() => (motion.value ? 'motion' : effectivePreset.value));
  const timingKey = computed(() => `${engine.kind === 'browser' ? 'browser:' : ''}${presetKey.value}`);

  const rows = computed(() => {
    const p = c.session.value?.preflight;
    return p ? preflightRows(p, { preset: effectivePreset.value, motion: motion.value, identity: needsIdentity.value, tracking: needsTracking.value && !motion.value, follow: needsIdentity.value || targetKind.value === 'tap' }) : [];
  });

  const busy = computed(() => job.value?.state === 'running');

  /** Why Run cannot start right now (empty = it can). */
  const blockers = computed<string[]>(() => {
    const out: string[] = [];
    if (!c.media.value) return ['Open a video first'];
    if (busy.value) out.push('A job is running');
    for (const r of blocking(rows.value)) out.push(`${r.text}: ${r.fix ?? 'missing'}`);
    const k = targetKind.value;
    if (k === 'tap' && tab.value === 'tap' && tapCount.value === 0) out.push('Tap the person in the video first');
    if (k === 'saved' && !savedName.value) out.push('Choose a saved target');
    if (k === 'photos' && !photos.value) out.push('Choose the photos');
    return out;
  });

  /** The browser engine's figures from the constants of model/limits.ts (null on My Mac). */
  const browserEst = computed<BrowserEstimate | null>(() => {
    const m = c.media.value;
    if (engine.kind !== 'browser' || !m) return null;
    const target = motion.value ? 'motion' : targetKind.value === 'tap' ? 'tap' : targetKind.value === 'class' ? 'class' : 'any';
    return browserEstimate({ duration: m.duration, preset: effectivePreset.value, target, browser: browserKindOf(typeof navigator === 'undefined' ? '' : navigator.userAgent), keyframes: m.keyframes.length, fps: adv.value.fps });
  });

  const estimate = computed<Estimate | null>(() => {
    const m = c.media.value;
    if (!m) return null;
    const measured = median(timings.value[timingKey.value] ?? []);
    if (browserEst.value) return { seconds: measured !== null ? (measured * m.duration) / 60 : browserEst.value.seconds, basis: 'measured' }; // the browser's constants are measured runs, not a lower bound
    return estimateRun({ preset: presetKey.value, duration: m.duration, keyframes: m.keyframes.length, fps: adv.value.fps, motion: motion.value, timings: timings.value[timingKey.value] ?? [] });
  });

  /** "Use My Mac" for this video, in the browser engine: set when it cannot decode the video here, has no WebGPU, would take long, or the video is long. */
  const localSuggestion = computed<LocalSuggestion | null>(() => {
    const m = c.media.value;
    const b = c.session.value?.preflight.browser;
    const be = browserEst.value;
    const est = estimate.value;
    if (!m || !b || !be || !est) return null;
    return suggestLocal({
      duration: m.duration, codec: m.codec, browser: browserKindOf(typeof navigator === 'undefined' ? '' : navigator.userAgent), est: { ...be, seconds: est.seconds },
      caps: { webcodecs: b.webcodecs, hevc: b.hevc, webgpu: b.webgpu },
    });
  });

  /** The slow-motion factor `auto` would use: capture rate over 30 fps (240 -> 8, 120 -> 4). */
  const autoFactor = computed(() => Math.max(1, Math.round((c.media.value?.capture_fps ?? 30) / 30)));

  function buildRequest(): RunRequest {
    const m = c.media.value!;
    const kind = targetKind.value;
    const target: RunRequest['target'] = { kind };
    if (kind === 'any' || kind === 'class' || kind === 'tap' || kind === 'photos') target.class = cls.value;
    if (kind === 'saved') target.name = savedName.value ?? undefined;
    if (kind === 'photos' || (kind === 'tap' && tab.value === 'photos')) target.photos_id = photos.value?.id;
    const slow = slowmoOn.value && m.capture_fps >= 100;
    const f = Number(slowFactor.value);
    return {
      media_id: m.id, target, zone: zone.value, zone_overlap: zone.value ? zoneOverlap.value : null, preset: effectivePreset.value,
      slowmo: slow ? (f > 1 ? String(f) : 'auto') : 'off', slowmo_audio: slow && slowmoAudio.value, advanced: { ...adv.value },
    };
  }

  // ---------------------------------------------------------------- jobs
  const job = signal<Job | null>(null);
  const jobNote = signal<JobNote | null>(null);
  const rerunAsk = signal<RerunAsk | null>(null);
  const restore = signal<{ clips: EditClip[] } | null>(null);
  let pending: RerunAsk | null = null;
  let followToken = 0;

  /** Poll a job every pollMs (twice as slow when the tab is hidden) until it ends, then call `onEnd`. */
  async function follow(first: Job, onEnd?: (j: Job) => void | Promise<void>): Promise<Job> {
    const mine = ++followToken;
    let j = first;
    job.value = j;
    while (j.state === 'running') {
      await sleep(typeof document !== 'undefined' && document.hidden ? c.pollMs * 2 : c.pollMs);
      if (disposed || mine !== followToken) return j;
      try {
        j = await engine.job(first.id);
        job.value = j;
      } catch (e) {
        c.noteDown(e);
        if (e instanceof EngineError && e.status !== 0) {
          j = { ...j, state: 'failed', error: message(e) };
          job.value = j;
        }
      }
    }
    await onEnd?.(j);
    return j;
  }

  /** A 409 busy answer: say which job runs and follow it, so its progress shows. */
  function busyOrToast(e: unknown) {
    if (e instanceof EngineError && e.status === 409 && e.extra.error === 'busy') {
      const j = e.extra.job as Job | undefined;
      c.toast(`A ${j?.kind ?? 'job'} job is running`, 'error');
      if (j && j.state === 'running') void follow(j, onEndFor(j.kind));
      return;
    }
    c.noteDown(e);
    c.toast(message(e), 'error');
  }

  function onEndFor(kind: Job['kind']): ((j: Job) => void | Promise<void>) | undefined {
    return kind === 'run' ? (j) => runEnded(j) : undefined;
  }

  async function runEnded(j: Job, runId?: string) {
    const m = c.media.value;
    if (j.state === 'done' || j.state === 'failed') {
      const rid = j.run_id ?? runId;
      try {
        if (m) {
          const list = await engine.runs(m.id);
          if (c.media.value?.id === m.id) {
            c.runs.value = list;
            if (rid && list.some((r) => r.run_id === rid)) await c.loadRun(rid);
          }
        }
      } catch (e) {
        c.noteDown(e);
        if (j.state === 'done') c.toast(message(e), 'error');
      }
    }
    if (j.state === 'done') {
      const total = c.run.value?.segments.timing_s?.total;
      if (m && typeof total === 'number' && m.duration > 0) {
        const key = timingKey.peek();
        timings.value = { ...timings.value, [key]: pushTiming(timings.value[key] ?? [], (total / m.duration) * 60) };
        writeTimings(timings.value);
      }
      jobNote.value = { kind: 'done', text: `Done: ${c.run.value?.segments.clips.length ?? 0} clips.` };
      if (pending) restore.value = { clips: pending.clips };
    } else if (j.state === 'failed') jobNote.value = { kind: 'failed', text: j.error || 'The run failed.' };
    else if (j.state === 'cancelled') jobNote.value = { kind: 'cancelled', text: 'Run cancelled. Nothing was kept.' };
    else if (j.state === 'needs_confirm') jobNote.value = { kind: 'needs_confirm', text: j.error || j.message || 'The photos match more than one person.' };
    pending = null;
  }

  /** Re-running a target whose run has edits: the count, or null when there is nothing to lose. */
  async function checkRerun(slug: string): Promise<RerunAsk | null> {
    const old = c.runs.value.find((r) => r.slug === slug);
    if (!old || !old.has_edits) return null;
    if (old.run_id === c.run.value?.run_id && c.saveState.value !== 'saved') await c.flushSave();
    const data = await engine.run(old.run_id);
    if (!data.edits) return null;
    const d = diffSummary(data.edits.clips, fromDetector(data.segments.clips, c.bounds.value).clips);
    const count = d.changed + d.deleted + d.added;
    return count ? { slug, count, clips: data.edits.clips } : null;
  }

  async function startRun(confirmed = false) {
    const m = c.media.value;
    if (!m || (blockers.peek().length && !confirmed)) return;
    if (busy.peek()) return c.toast('A job is running', 'error');
    if (c.playing.peek()) c.pause();
    const req = buildRequest();
    const slug = slugOf(req.target);
    try {
      if (!confirmed) {
        const ask = await checkRerun(slug);
        if (ask) {
          rerunAsk.value = ask;
          return;
        }
      } else pending = rerunAsk.peek();
      rerunAsk.value = null;
      if (req.target.kind === 'tap' && tab.peek() === 'tap') await syncZone();
      jobNote.value = null;
      restore.value = null;
      const ref = await engine.startRun(req);
      await follow(ref.job, (j) => runEnded(j, ref.run_id));
    } catch (e) {
      pending = null;
      busyOrToast(e);
    }
  }

  function cancelRerun() {
    rerunAsk.value = null;
  }

  async function cancelJob() {
    const j = job.peek();
    if (!j || j.state !== 'running') return;
    try {
      await engine.cancel(j.id);
    } catch (e) {
      c.noteDown(e);
      c.toast(message(e), 'error');
    }
  }

  /** After a reload the server may still be running a job: show its progress again. */
  function resume(j: Job | null) {
    if (j && j.state === 'running') void follow(j, onEndFor(j.kind));
  }

  /** "Restore my earlier edits": the clips kept aside become the clip list, as one undo step. */
  function restoreEdits() {
    const r = restore.peek();
    if (!r) return;
    const clips = r.clips.map((x) => ({ ...x, origin: x.origin === 'detector' ? ('edited' as const) : x.origin }));
    const res = revert(c.edit.peek(), clips);
    if (res.changed) {
      const top = clips.reduce((n, x) => Math.max(n, Number(/\d+/.exec(x.id)?.[0] ?? 0)), 0);
      res.state = { ...res.state, nextId: Math.max(res.state.nextId, top + 1) };
    }
    if (c.commit(res)) restore.value = null;
  }

  // ---------------------------------------------------------------- export
  const exportOpen = signal(false);

  function openExport() {
    if (!c.run.value) return c.toast('Nothing to export yet: pick a target and press Run first');
    exportOpen.value = true;
  }

  /** Save pending edits, then start the export job; a 409 `exists` reaches the caller (the dialog asks to replace). */
  async function startExport(req: ExportRequest): Promise<Job | null> {
    const r = c.run.value;
    if (!r) return null;
    if (c.saveState.value !== 'saved') await c.flushSave();
    if (c.saveState.value !== 'saved') {
      c.toast('The edits are not saved yet', 'error');
      return null;
    }
    const ref = await engine.startExport(r.run_id, req);
    return follow(ref.job);
  }

  // ---------------------------------------------------------------- per video
  function resetForMedia() {
    clearTimeout(boxTimer);
    boxAbort?.abort();
    batch(() => {
      picks.value = {};
      zone.value = null;
      zoneMode.value = 'off';
      zoneOverlap.value = null;
      photos.value = null;
      photoConfirms.value = 0;
      candidates.value = [];
      people.value = null;
      pickInfo.value = { boxes: null, loading: false, firstLoad: pickInfo.peek().firstLoad, error: null };
      jobNote.value = null;
      restore.value = null;
      rerunAsk.value = null;
      pending = null;
    });
    if (tab.peek() === 'tap') void ensurePick(cls.peek());
  }

  return {
    tab, cls, picks, saved, savedError, savedName, photos, photoConfirms, candidates, people, motion, pickInfo, tapPending, pickMode, currentPick, tapCount, pickBoxList,
    targetKind, setTab, setClass, loadSaved, chooseSaved, tapBox, undoTap, choosePhotos, findWho, confirmCandidate, savePhotosAsTarget, scanPeople, showPerson,
    zone, zoneMode, zoneOverlap, zoneBegin, zoneDone, zoneCancel, zoneClear,
    preset, slowmoOn, slowFactor, slowmoAudio, adv, timings, setAdv, adoptDefaults, effectivePreset, rows, blockers, busy, estimate, browserEst, localSuggestion, autoFactor, buildRequest,
    job, jobNote, rerunAsk, restore, busyOrToast, startRun, cancelRerun, cancelJob, resume, restoreEdits,
    exportOpen, openExport, startExport, resetForMedia,
    dispose() {
      disposed = true;
      clearTimeout(boxTimer);
      boxAbort?.abort();
      stopBoxEffect();
    },
  };
}

export type Workflow = ReturnType<typeof createWorkflow>;
