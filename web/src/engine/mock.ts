// MockEngine: an in-memory engine with one synthetic 60 s video, one run with three clips and generated samples. Used by vitest and by
// `npm run dev -- --mode mock`, so the UI can be built without the Python server. Nothing here touches the network or a real file.
import {
  EditConflict,
  EngineError,
  type Box,
  type Caps,
  type PhotosSave,
  type Candidate,
  type DetectorClip,
  type DirListing,
  type EditClip,
  type EditsDoc,
  type Engine,
  type ExportResult,
  type Filmstrip,
  type FrameBoxes,
  type Job,
  type JobRef,
  type MediaInfo,
  type MediaTask,
  type MusicAutoQuery,
  type MusicLibraryResult,
  type MusicTrack,
  type MusicAutoResult,
  type MusicQuery,
  type MusicSearchResult,
  type PhotoSet,
  type PickState,
  type RankRequest,
  type RecentEntry,
  type ResegmentParams,
  type ResegmentResult,
  type RunData,
  type RunRequest,
  type RunSummary,
  type SavedTarget,
  type Samples,
  type Session,
  type Zone,
  type ExportRequest,
  type Preflight,
} from './types';

export const MOCK_MEDIA_ID = 'mock00000001';
export const MOCK_RUN_ID = 'mockrun00001';
export const MOCK_PATH = '/mock/clips/mock_clip.mp4';

const CAPS: Caps = {
  openDialog: true, photosPicker: false, fileBrowser: true, locate: true, reveal: false, music: false, beatSync: false,
  savedTargets: true, people: false, proxy: true, output: 'server-folder', streamCopyCut: true, chooseOutputFolder: true, savePhotos: true,
  reel: true, editLists: true,
};

export function mockMedia(over: Partial<MediaInfo> = {}): MediaInfo {
  return {
    id: MOCK_MEDIA_ID, path: MOCK_PATH, name: 'mock_clip.mp4', size: 12_345_678, mtime: 1_760_000_000, codec: 'h264', profile: 'High', pix_fmt: 'yuv420p',
    width: 1280, height: 720, display_w: 1280, display_h: 720, rotation: 0, mirrored: false, fps: 30, capture_fps: 30, slowmo_suggested: false,
    duration: 60, offset: 0, end_time: 60, has_audio: false, keyframes: [0, 2, 4, 6, 8, 10], video_url: `api/media/${MOCK_MEDIA_ID}/video`,
    proxy: { state: 'none', fraction: 0, url: null, first_frame_file_time: null, error: null }, filmstrip_state: 'none', picked_copy: false, ...over,
  };
}

const DETECTOR: DetectorClip[] = [
  { index: 1, start: 5, end: 12, peak_score: 0.93 },
  { index: 2, start: 20, end: 28.5, peak_score: 0.88 },
  { index: 3, start: 40, end: 47, peak_score: 0.91 },
];

function mockSamples(): Samples {
  const t: number[] = [];
  const score: number[] = [];
  const box: Samples['box'] = [];
  for (let i = 0; i <= 300; i++) {
    const x = i * 0.2;
    const inside = DETECTOR.some((c) => x >= c.start && x <= c.end);
    t.push(Math.round(x * 1000) / 1000);
    score.push(inside ? 0.8 + 0.1 * Math.sin(x) : 0.1 + 0.05 * Math.cos(x));
    box.push(inside ? [0.3, 0.2, 0.5, 0.8] : null);
  }
  return { t, score, box, track: null, enter: 0.5, exit: 0.35, merge_gap: 1.5, min_len: 1, pad: 1, sound: null, mode: 'detect' };
}

/** Three fixed songs and then the built-in beat; the rank wraps (`rank % 4`). With CC0 only the CC BY songs answer as CC0. */
const AUTO_TRACKS = [
  { id: 'auto1', title: 'Sunny Walk', creator: 'A. Composer', license: 'by', bpm: 124, score: 0.81, duration_s: 154 },
  { id: 'auto2', title: 'Night Drive', creator: 'B. Player', license: 'cc0', bpm: 120, score: 0.74, duration_s: 201 },
  { id: 'auto3', title: 'Slow Tide', creator: 'C. Writer', license: 'by', bpm: 100, score: 0.55, duration_s: 187 },
  { id: '00000000-0000-0000-0000-000000000128', title: 'inframe built-in beat, 128 BPM', creator: 'inframe', license: 'cc0', bpm: 128, score: null, duration_s: 120 },
];

function autoResult(mood: string, rank: number, cc0Only: boolean): MusicAutoResult {
  const t = AUTO_TRACKS[rank % AUTO_TRACKS.length];
  const license = cc0Only ? 'cc0' : t.license;
  const by = license === 'by';
  return {
    rank: rank % AUTO_TRACKS.length, n: AUTO_TRACKS.length, wrapped: rank >= AUTO_TRACKS.length, mood: mood || 'energetic', speed: 'fast', warnings: [],
    notice: 'Found through Openverse (openverse.org); inframe is not endorsed by Openverse.',
    track: { id: t.id, title: t.title, creator: t.creator, license, license_version: by ? '4.0' : '1.0', license_url: null, duration_s: t.duration_s, bpm: t.bpm, score: t.score,
      needs_credit: by, credit_text: by ? `Music: "${t.title}" by ${t.creator}\nLicense: CC BY 4.0` : null, builtin: t.id.startsWith('0000') },
  };
}

export class MockEngine implements Engine {
  readonly kind = 'local' as const;
  /** a copy, so a test can switch capabilities on before the store starts */
  caps: Caps = { ...CAPS };
  preflight: Preflight = { ffmpeg: true, models: { fast: true, balanced: true, accurate: false }, identity: false, tracking: true, music: false };
  /** what the tests look at */
  calls: string[] = [];
  /** the next saveEdits fails this way */
  failSave: 'conflict' | 'down' | null = null;
  edits: EditsDoc | null = null;
  /** the folder name of the mock run (a target's slug) */
  slug = 'tap';
  media_: MediaInfo;
  private recentList: RecentEntry[] = [{ path: MOCK_PATH, name: 'mock_clip.mp4', opened: 1_760_000_000 }];

  constructor(media: Partial<MediaInfo> = {}, public hasRun = true) {
    this.media_ = mockMedia(media);
  }

  private log(s: string) {
    this.calls.push(s);
  }

  async session(): Promise<Session> {
    return {
      version: '0.0.0-mock', engine: 'local', platform: 'darwin', caps: this.caps, presets: ['fast', 'balanced', 'accurate'], default_preset: 'balanced',
      classes: ['person', 'dog'], defaults: { pad: 1, min_len: 1, merge_gap: 1.5, enter: null, exit: null, min_box: 0, tracker: 'ocsort', activity: 'any', fps: null },
      preflight: this.preflight, job: this.beginRunning(), media: [], start_media: null,
    };
  }
  async openPath(path: string) {
    this.log(`openPath ${path}`);
    if (path.includes('missing')) throw new EngineError(404, `No such file: ${path}`);
    return this.media_;
  }
  async openDialog() { this.log('openDialog'); return this.media_; }
  async openFromPhotos() { this.log('openFromPhotos'); return null; }
  async locate(f: { name: string; size: number; lastModified: number }) {
    this.log(`locate ${f.name}`);
    return f.name === this.media_.name ? this.media_ : null;
  }
  async listDir(dir = '/mock'): Promise<DirListing> {
    this.log(`listDir ${dir}`);
    // the shapes of serve_media.fs_list: roots are {name, path}, a folder's size is null, a root has no parent
    const roots = [{ name: 'Home', path: '/mock' }, { name: 'Card', path: '/Volumes/Card' }];
    return {
      dir, parent: roots.some((r) => r.path === dir) ? null : dir.replace(/\/[^/]*$/, '') || '/', roots,
      entries: dir === '/mock' ? [{ name: 'clips', kind: 'dir', size: null, mtime: 1_760_000_000 }] : [{ name: 'mock_clip.mp4', kind: 'video', size: 12_345_678, mtime: 1_760_000_000 }],
    };
  }
  async recent() { return this.recentList; }
  async clearRecent() { this.log('clearRecent'); this.recentList = []; return this.recentList; }
  async media(id: string) {
    if (id !== MOCK_MEDIA_ID) throw new EngineError(404, `no such media: ${id}`);
    return this.media_;
  }
  videoUrl(id: string, opts: { asMp4?: boolean } = {}) { return `mock://video/${id}${opts.asMp4 ? '?as=mp4' : ''}`; }
  // POST /proxy answers {state, fraction} only (serve_media._proxy_brief); the url comes with GET /proxy once the copy is done
  async makeProxy(): Promise<MediaTask> { this.log('makeProxy'); return { state: 'running', fraction: 0 }; }
  async mediaTask(): Promise<MediaTask> { return { state: 'done', fraction: 1, url: 'mock://proxy', first_frame_file_time: 0, error: null }; }
  async filmstrip(): Promise<Filmstrip> { return { version: 1, cell_w: 192, cell_h: 108, cols: 10, rows: 10, cells: [], sheets: 0, done: true }; }
  sheetUrl(id: string, n: number) { return `mock://sheet/${id}/${n}`; }
  frameUrl(id: string, t: number) { return `mock://frame/${id}/${t}`; }
  pickFrameUrl(id: string, t: number, cls: string) { return `mock://pickframe/${id}/${cls}/${t.toFixed(3)}`; }
  /** what `boxes` finds in every frame; a test sets it */
  mockBoxes: { i: number; box: Box; score: number }[] = [{ i: 1, box: [0.1, 0.1, 0.3, 0.9], score: 0.93 }, { i: 2, box: [0.5, 0.1, 0.8, 0.9], score: 0.88 }];
  async boxes(_id: string, t: number, cls: string): Promise<FrameBoxes> {
    this.log(`boxes ${t.toFixed(3)} ${cls}`);
    return { t, boxes: this.mockBoxes.map((b) => ({ ...b })) };
  }

  private picks = new Map<string, PickState>();
  private pickFor(cls: string): PickState {
    let p = this.picks.get(cls);
    if (!p) this.picks.set(cls, (p = { target_key: '', taps: [], zone: null, file: null, class: cls, candidates: [] }));
    return p;
  }
  private snap(p: PickState): PickState {
    return JSON.parse(JSON.stringify({ ...p, file: p.taps.length ? `/mock/clips/mock_clip_inframe/targets/target-${p.class}.json` : null }));
  }
  async pickState(_id: string, cls: string) { this.log(`pickState ${cls}`); return this.snap(this.pickFor(cls)); }
  async tap(_id: string, cls: string, t: number, box: Box, index: number) {
    this.log(`tap ${cls} ${t.toFixed(3)} #${index}`);
    const p = this.pickFor(cls);
    p.taps.push({ time: t, box, track_id: null, index });
    return this.snap(p);
  }
  async undoTap(_id: string, cls: string) {
    this.log(`undoTap ${cls}`);
    const p = this.pickFor(cls);
    p.taps.pop();
    return this.snap(p);
  }
  async setZone(_id: string, cls: string, zone: Zone | null) {
    this.log(`setZone ${cls} ${JSON.stringify(zone)}`);
    const p = this.pickFor(cls);
    p.zone = zone;
    return this.snap(p);
  }
  async resetPick(_id: string, cls: string) {
    const p = this.pickFor(cls);
    p.taps = [];
    p.zone = null;
    return this.snap(p);
  }
  saved: SavedTarget[] = [
    { name: 'alice', n_crops: 4, embeddings: { v1: { embedder: 'dinov2-s', current: true, valid: true } } },
    { name: 'old', n_crops: 2, embeddings: {} },
    { name: 'broken', error: 'target.json is unreadable' },
  ];
  async savedTargets() { return this.saved.map((s) => ({ ...s })); }
  targetCropUrl(name: string, i: number) { return `mock://crop/${name}/${i}`; }
  /** what the next photos choice gives; null = cancelled */
  nextPhotos: PhotoSet | null = { id: 'p1', dir: '/mock/photos', n_photos: 12 };
  async choosePhotos(source: 'dialog' | 'photos' | 'path', path?: string) {
    this.log(`choosePhotos ${source}${path ? ` ${path}` : ''}`);
    return this.nextPhotos;
  }

  private job_: Job = { id: 'mockjob', kind: 'run', state: 'done', fraction: 1, message: '', started: 0, elapsed_s: 0, media_id: null, run_id: null, result: null, error: null };
  /** a job that is already running when the page starts (session.job) */
  runningAtStart: Job | null = null;
  /** what the next job() polls answer, in order (the last one repeats); each is merged over a base running job */
  script: Partial<Job>[] = [];
  lastRun: RunRequest | null = null;
  lastExport: ExportRequest | null = null;
  lastRank: RankRequest | null = null;
  lastTargetAdd: { name: string; from: string } | null = null;
  /** what `photos` says in the result of an export that asked for to_photos (the default is a full save) */
  photosResult: PhotosSave = { status: 'saved', message: 'Saved 2 video(s) to the Photos library.', saved: ['mock_clip_01.mp4', 'mock_clip_reel.mp4'], failed: [] };
  /** the next startExport answers 409 exists with this export.json */
  exportExists: Record<string, unknown> | null = null;
  /** the next startRun answers 409 busy */
  busy: Job | null = null;
  cancelled = 0;
  private current: Job | null = null;
  private base(kind: Job['kind']): Job {
    return { id: `mockjob-${kind}`, kind, state: 'running', fraction: 0, message: '', started: 0, elapsed_s: 0, media_id: MOCK_MEDIA_ID, run_id: null, result: null, error: null };
  }
  private begin(kind: Job['kind'], plan: Partial<Job>[], extra: Partial<Job> = {}): JobRef {
    const job = { ...this.base(kind), ...extra };
    if (!this.script.length) this.script = plan;
    this.current = job;
    return { job: { ...job }, run_id: extra.run_id ?? undefined };
  }
  private beginRunning(): Job | null {
    if (this.runningAtStart) this.current = { ...this.runningAtStart };
    return this.runningAtStart;
  }
  async startRun(req: RunRequest) {
    this.log(`startRun ${req.target.kind}`);
    if (this.busy) {
      this.current = { ...this.busy };
      throw new EngineError(409, 'busy', { error: 'busy', job: this.busy });
    }
    this.lastRun = JSON.parse(JSON.stringify(req));
    this.hasRun = true;
    this.edits = null; // the server moves the edits of the run it replaces aside
    return this.begin('run', [{ fraction: 0.5, message: 'detect: 30 of 60 s', elapsed_s: 15 }, { state: 'done', fraction: 1, elapsed_s: 30, run_id: MOCK_RUN_ID }], { run_id: MOCK_RUN_ID });
  }
  async startRank(_id: string, req: RankRequest) {
    this.log('startRank');
    this.lastRank = req;
    const cands: Candidate[] = [1, 2, 3].map((n) => ({ rank: n, t: 10 * n, box: [0.2, 0.2, 0.4, 0.8], track_id: n, score: 0.9 - n * 0.1 }));
    return this.begin('rank', [{ state: 'done', fraction: 1, result: { candidates: cands } }]);
  }
  async startPeople() {
    this.log('startPeople');
    const people = [1, 2].map((n) => ({ person: n, thumbnail_time_s: 12 * n, seen_s: 20 * n, crop_url: `mock://person/${n}`, target: `tap:/mock/person_00${n}.json` }));
    return this.begin('people', [{ state: 'done', fraction: 1, result: { people } }]);
  }
  async confirmCandidate(_id: string, i: number, cls = 'person') {
    this.log(`confirmCandidate ${i}`);
    return { target_key: 'photos:p1', taps: [{ time: 10 * (i + 1), box: [0.2, 0.2, 0.4, 0.8] as Box, track_id: null, index: i }], zone: null, file: '/mock/target.json', class: cls, candidates: [] };
  }
  candidateUrl(id: string, i: number) { return `mock://cand/${id}/${i}`; }
  personUrl(id: string, n: number) { return `mock://person/${id}/${n}`; }
  async startExport(_id: string, req: ExportRequest) {
    this.log('startExport');
    if (this.exportExists && !req.replace) throw new EngineError(409, 'exists', { error: 'exists', export: this.exportExists });
    this.lastExport = JSON.parse(JSON.stringify(req));
    const auto = req.music && 'auto' in req.music ? autoResult(req.music.auto.mood, req.music.auto.rank, req.music.cc0_only) : null;
    const byMusic = !!req.reel && (auto ? auto.track.needs_credit : (req.music as { track_id?: string | null } | null)?.track_id === 'trk2'); // trk2 is the CC BY track of musicSearch
    const outputs = [{ name: 'mock_clip_01.mp4', kind: 'clip', size: 1_000_000, url: 'api/runs/x/files/mock_clip_01.mp4' }, ...(req.reel ? [{ name: 'mock_clip_reel.mp4', kind: 'reel', size: 3_000_000, url: 'api/runs/x/files/mock_clip_reel.mp4' }] : []),
      ...(byMusic ? [{ name: 'mock_clip_credits.txt', kind: 'credits', size: 300, url: 'api/runs/x/files/mock_clip_credits.txt' }] : [])];
    return this.begin('export', [{ fraction: 0.5, message: 'cutting clip 1 of 2' }, { state: 'done', fraction: 1, result: { outputs, warnings: [], reel: req.reel ? { file: 'mock_clip_reel.mp4', ok: true, check: { ok: true, problems: [], lufs: -14.1 },
      ...(auto && req.reel ? { music: { title: auto.track.title, creator: auto.track.creator, auto: { rank: auto.rank, n: auto.n }, credits: byMusic ? 'mock_clip_credits.txt' : null, credit_text: auto.track.credit_text } }
        : byMusic ? { music: { credits: 'mock_clip_credits.txt', credit_text: 'Music: "Long Road" by B. Player\nLicense: CC BY 4.0' } } : {}) } : null, edit_lists: [], ...(req.to_photos ? { photos: this.photosResult } : {}) } }]);
  }
  async startTargetAdd(name: string, req: { from: string }) {
    this.log(`startTargetAdd ${name}`);
    this.lastTargetAdd = { name, from: req.from };
    return this.begin('target-add', [{ state: 'done', fraction: 1 }]);
  }
  async job(id: string) {
    this.log(`job ${id}`);
    const base = this.current ?? this.job_;
    const next = this.script.length > 1 ? this.script.shift()! : (this.script[0] ?? { state: 'done' as const, fraction: 1 });
    const job = { ...base, ...next };
    if (job.state !== 'running') this.script = [];
    this.current = job;
    return { ...job };
  }
  async currentJob() { return this.current && this.current.state === 'running' ? { ...this.current } : null; }
  async cancel() {
    this.cancelled++;
    this.script = [{ state: 'cancelled', fraction: 0.4 }];
    return { ...(this.current ?? this.job_) };
  }

  async runs(): Promise<RunSummary[]> {
    if (!this.hasRun) return [];
    return [{ run_id: MOCK_RUN_ID, dir: '/mock/clips/mock_clip_inframe/tap', slug: this.slug, target: this.slug, preset: 'balanced', created: 1_760_000_100,
      n_clips: DETECTOR.length, error: null, has_edits: !!this.edits, edits_base_ok: true }];
  }
  async run(runId: string): Promise<RunData> {
    this.log(`run ${runId}`);
    return {
      run_id: runId, media_id: MOCK_MEDIA_ID, dir: '/mock/clips/mock_clip_inframe/tap',
      segments: { settings: { enter: 0.5, exit: 0.35, merge_gap: 1.5, min_len: 1, pad: 1, target: 'tap' }, warnings: [], timing_s: { total: 30 }, clips: DETECTOR.map((c) => ({ ...c })), error: null },
      edits: this.edits, edits_base_ok: true, export: null,
    };
  }
  async samples() { return mockSamples(); }
  async saveEdits(runId: string, revision: number, clips: EditClip[]): Promise<EditsDoc> {
    this.log(`saveEdits ${runId} r${revision} n${clips.length}`);
    const fail = this.failSave;
    this.failSave = null;
    if (fail === 'down') throw new EngineError(0, 'inframe serve is not running. Start it again in the terminal.');
    if (fail === 'conflict' || revision !== (this.edits?.revision ?? 0)) throw new EditConflict(this.edits);
    this.edits = {
      format: 1, base: { segments_sha256: 'mock', n_clips: DETECTOR.length }, revision: revision + 1, updated: new Date(0).toISOString(),
      next_id: clips.reduce((m, c) => Math.max(m, Number(c.id.replace(/\D/g, '')) || 0), 0) + 1, clips: clips.map((c) => ({ ...c })),
    };
    return this.edits;
  }
  async revertEdits() { this.edits = null; }
  async resegment(_id: string, _p: ResegmentParams): Promise<ResegmentResult> {
    const iv = DETECTOR.map((c) => [c.start, c.end] as [number, number]);
    return { presence: iv, clips: iv, same_as_run: true };
  }
  async exportResult(): Promise<ExportResult | null> { return null; }
  runFileUrl(runId: string, name: string) { return `mock://file/${runId}/${name}`; }
  async reveal() {}

  /** the included songs `musicLibrary` answers with (none by default) */
  libraryTracks: MusicTrack[] = [];
  libraryQueries: boolean[] = [];
  async musicLibrary(cc0Only: boolean): Promise<MusicLibraryResult> {
    this.libraryQueries.push(cc0Only);
    return { tracks: this.libraryTracks.filter((t) => !cc0Only || t.license === 'cc0'), notice: 'These songs come with inframe and work offline.' };
  }
  musicQueries: MusicQuery[] = [];
  async musicSearch(q: MusicQuery): Promise<MusicSearchResult> {
    this.musicQueries.push({ ...q });
    return {
      tracks: [{ id: 'trk1', title: 'Bright Day', creator: 'A. Composer', duration_s: 95, license: 'cc0' }, { id: 'trk2', title: 'Long Road', creator: 'B. Player', duration_s: 200, license: q.cc0_only ? 'cc0' : 'by' }],
      notice: 'Found through Openverse (openverse.org); inframe is not endorsed by Openverse.',
    };
  }
  async musicGet(id: string) { this.log(`musicGet ${id}`); return { id, file: '/mock/music.mp3', license: 'cc0' }; }
  /** every musicAuto query, in order */
  autoQueries: MusicAutoQuery[] = [];
  /** the next musicAuto fails with this message */
  autoError: string | null = null;
  async musicAuto(_runId: string, q: MusicAutoQuery, onJob?: (job: Job) => void, _signal?: AbortSignal): Promise<MusicAutoResult> {
    this.autoQueries.push(JSON.parse(JSON.stringify(q)));
    onJob?.({ ...this.base('music-auto'), message: 'choosing a song' });
    if (this.autoError) {
      const m = this.autoError;
      this.autoError = null;
      throw new EngineError(502, m);
    }
    return autoResult(q.mood, q.rank, q.cc0_only);
  }
  musicUrl(id: string) { return `mock://music/${id}`; }
  async musicChooseFile() { this.log('musicChooseFile'); return { file: '/mock/own.mp3', license: null }; }
  async quit() {}
}
