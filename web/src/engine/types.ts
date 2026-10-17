// The seam between the UI and whatever does the work (docs/T9A_DESIGN.md 2.10). The types mirror the API JSON of section 3.
// UI code imports only this file and the pure modules in ../model. Times are file time (seconds); boxes are [x0, y0, x1, y1] and zones
// [x, y, w, h], both as fractions of the upright frame.

export type Box = [number, number, number, number];
export type Zone = [number, number, number, number];

export interface Caps {
  openDialog: boolean;
  photosPicker: boolean;
  fileBrowser: boolean;
  locate: boolean;
  reveal: boolean;
  music: boolean;
  beatSync: boolean;
  savedTargets: boolean;
  people: boolean;
  proxy: boolean;
  output: 'server-folder' | 'download' | 'granted-folder';
  streamCopyCut: boolean;
  chooseOutputFolder: boolean;
  /** "Save to Photos" in the export dialog (My Mac on macOS: the add-only Photos permission, library only) */
  savePhotos?: boolean;
  /** the Reel section of the export dialog (true on My Mac; the browser engine has no reel yet) */
  reel: boolean;
  /** the Edit lists section of the export dialog (true on My Mac; the browser engine writes none yet) */
  editLists: boolean;
}

/** What a browser can do for a run (BrowserEngine.session): shown instead of the ffmpeg and model rows of the Mac. */
export interface BrowserPreflight {
  webgpu: boolean;
  webcodecs: boolean;
  /** null = not checked */
  hevc: boolean | null;
  models: { available: boolean; error: string | null; detectorReady: boolean; /** the DINOv2 file tap targets need is cached too */ embedderReady?: boolean; allReady: boolean; totalBytes: number; cachedBytes: number; /** where the files come from, and how many are still to download (the privacy line) */ host?: string; filesLeft?: number };
}

/** The downloaded models of the browser engine (Preflight shows the buttons). */
export interface ModelControl {
  status(): Promise<BrowserPreflight['models'] & { persisted: boolean | null }>;
  download(opts?: { onProgress?: (p: { name: string; done: number; total: number }) => void; signal?: AbortSignal }): Promise<void>;
  remove(): Promise<void>;
}

export interface Preflight {
  /** set by the browser engine; the rows then come from it */
  browser?: BrowserPreflight;
  ffmpeg: boolean;
  ffprobe?: boolean;
  models: Record<string, boolean>;
  identity: boolean;
  tracking: boolean;
  music: boolean;
}

export interface Session {
  version: string;
  engine: 'local' | 'browser';
  platform: string;
  caps: Caps;
  presets: string[];
  default_preset: string;
  classes: string[];
  defaults: Record<string, unknown>;
  preflight: Preflight;
  job: Job | null;
  media: MediaInfo[];
  start_media: string | null;
}

export interface MediaTask {
  state: 'none' | 'running' | 'done' | 'failed';
  fraction: number;
  url?: string | null;
  first_frame_file_time?: number | null;
  error?: string | null;
}

export interface MediaInfo {
  id: string;
  path: string;
  name: string;
  size: number;
  mtime: number;
  codec: string;
  profile: string | null;
  pix_fmt: string | null;
  width: number;
  height: number;
  display_w: number;
  display_h: number;
  rotation: number;
  mirrored: boolean;
  fps: number;
  capture_fps: number;
  slowmo_suggested: boolean;
  duration: number;
  offset: number;
  end_time: number;
  has_audio: boolean;
  keyframes: number[];
  video_url: string;
  proxy: MediaTask;
  filmstrip_state: string;
  picked_copy: boolean;
}

export interface Filmstrip {
  version: number;
  cell_w: number;
  cell_h: number;
  cols: number;
  rows: number;
  cells: { t: number; sheet: number; i: number }[];
  sheets: number;
  done: boolean;
  state?: string;
}

export interface DirEntry {
  name: string;
  kind: 'dir' | 'video' | 'audio';
  /** null for a folder */
  size?: number | null;
  mtime?: number;
}

/** A folder the file browser may start from (the home folder, a mounted volume). */
export interface DirRoot {
  name: string;
  path: string;
}

export interface DirListing {
  dir: string;
  parent: string | null;
  roots: DirRoot[];
  entries: DirEntry[];
}

export interface RecentEntry {
  path: string;
  name: string;
  opened: number | string;
}

export interface FrameBoxes {
  t: number;
  boxes: { i: number; box: Box; score: number }[];
}

export interface Tap {
  time: number;
  box: Box;
  track_id: number | null;
  index: number;
}

export interface Candidate {
  rank: number;
  t: number;
  box: Box;
  track_id: number | null;
  score: number;
}

export interface PickState {
  target_key: string;
  taps: Tap[];
  zone: Zone | null;
  file: string | null;
  class: string;
  candidates: Candidate[];
}

export interface SavedTarget {
  name: string;
  error?: string;
  /** Saved.summary(): how many crops, and one entry per embedder version (`current` and `valid` decide the "stale" badge) */
  n_crops?: number;
  embeddings?: Record<string, { embedder?: string | null; current?: boolean; valid?: boolean }>;
  [k: string]: unknown;
}

export interface PhotoSet {
  id: string;
  dir: string;
  n_photos: number;
}

export interface RunRequest {
  media_id: string;
  target: { kind: 'any' | 'class' | 'tap' | 'saved' | 'photos' | 'motion'; class?: string; name?: string; photos_id?: string };
  zone: Zone | null;
  zone_overlap: number | null;
  preset: string;
  slowmo: string;
  slowmo_audio: boolean;
  advanced: Record<string, unknown>;
}

export interface RankRequest {
  photos_id?: string;
  saved?: string;
  cls?: string;
}

export interface ExportRequest {
  clip_ids: string[] | null;
  out_dir: string | null;
  replace: boolean;
  clips: boolean;
  reel: boolean;
  /** exactly one of a track id, a file or an auto pick (`mood` "" = from the reel) */
  music: { track_id: string | null; file: string | null; cc0_only: boolean } | { auto: { mood: string; rank: number; track?: string }; cc0_only: boolean } | null;
  beat_sync: boolean;
  edit_lists: string[];
  /** My Mac only: after the export, add its clips and reel to the Photos library (add-only permission, no album) */
  to_photos?: boolean;
}

/** `photos` in an export result: what saving to Photos did. `message` is the sentence to show, whatever the status. */
export interface PhotosSave {
  status: 'saved' | 'partial' | 'denied' | 'failed' | 'interrupted' | 'skipped';
  authorization?: string;
  message: string;
  saved: string[];
  failed: { name: string; error: string }[];
}

export interface TargetAddRequest {
  from: 'run' | 'photos';
  run_id?: string;
  photos_id?: string;
}

export interface Job {
  id: string;
  kind: 'run' | 'rank' | 'people' | 'export' | 'target-add' | 'music-auto';
  state: 'running' | 'done' | 'failed' | 'cancelled' | 'needs_confirm';
  fraction: number;
  message: string;
  started: number;
  elapsed_s: number;
  media_id: string | null;
  run_id: string | null;
  result: unknown;
  error: string | null;
}

export interface JobRef {
  job: Job;
  run_id?: string;
}

export interface RunSummary {
  run_id: string;
  dir: string;
  slug: string;
  target: string;
  preset: string;
  created: string | number;
  n_clips: number;
  error: string | null;
  has_edits: boolean;
  edits_base_ok: boolean;
}

/** A clip as the detector wrote it in segments.json. */
export interface DetectorClip {
  index?: number;
  start: number;
  end: number;
  peak_score?: number;
  file?: string;
  expected_copy_start?: number;
  [k: string]: unknown;
}

export type Origin = 'detector' | 'edited' | 'split' | 'merge' | 'added' | 'resegment';

export interface EditClip {
  id: string;
  start: number;
  end: number;
  origin: Origin;
}

export interface EditsDoc {
  format: number;
  base: { segments_sha256: string; n_clips: number };
  revision: number;
  updated: string;
  next_id: number;
  clips: EditClip[];
}

export interface RunData {
  run_id: string;
  media_id: string;
  dir: string;
  segments: {
    settings: Record<string, any>;
    warnings: string[];
    scores?: unknown;
    timing_s?: Record<string, number>;
    presence_intervals?: [number, number][];
    clips: DetectorClip[];
    slowmo?: { factor: number; [k: string]: unknown } | null;
    error?: string | null;
  };
  edits: EditsDoc | null;
  edits_base_ok: boolean;
  export: unknown | null;
}

export interface Samples {
  t: number[];
  score: number[];
  box: (Box | null)[];
  track: (number | null)[] | null;
  enter: number;
  exit: number;
  merge_gap: number;
  min_len: number;
  pad: number;
  sound?: string | null;
  mode: string;
}

export interface ResegmentParams {
  enter?: number;
  exit?: number;
  merge_gap?: number;
  min_len?: number;
  pad?: number;
}

export interface ResegmentResult {
  presence: [number, number][];
  clips: [number, number][];
  same_as_run: boolean;
}

export interface ExportOutput {
  name: string;
  kind: 'clip' | 'reel' | 'sidecar' | 'editlist' | 'csv' | 'credits' | 'zip';
  size: number;
  url: string;
}

export interface ExportResult {
  outputs: ExportOutput[];
  [k: string]: unknown;
}

export interface MusicQuery {
  mood: string;
  min_s?: number;
  limit?: number;
  cc0_only?: boolean;
}

export interface MusicTrack {
  id: string;
  title: string;
  creator: string;
  duration_s: number;
  license: string;
  /** a song that ships inside inframe (the included-songs list only) */
  bundled?: boolean;
  bpm?: number | null;
  moods?: string[];
  license_version?: string | null;
  [k: string]: unknown;
}

/** GET api/music/library: the songs that come with inframe (no network); `notice` says so */
export interface MusicLibraryResult {
  tracks: MusicTrack[];
  notice: string;
}

export interface MusicSearchResult {
  tracks: MusicTrack[];
  /** music.OPENVERSE_NOTICE, shown under the results */
  notice: string;
}

/** POST api/runs/{id}/music/auto: the clips the reel would hold (null = all), the mood ("" = from the reel), CC0 only, the rank (0 = first pick) */
export interface MusicAutoQuery {
  clip_ids: string[] | null;
  mood: string;
  cc0_only: boolean;
  rank: number;
}

/** The result of the `music-auto` job. */
export interface MusicAutoResult {
  rank: number;
  /** the length of the ranked list (the last entry is inframe's built-in beat) */
  n: number;
  wrapped: boolean;
  mood: string;
  speed: string;
  warnings: string[];
  notice: string;
  track: {
    id: string;
    title: string;
    creator: string;
    license: string;
    license_version: string | null;
    license_url: string | null;
    duration_s: number;
    bpm: number | null;
    /** the fit, 0 to 1; null for the built-in beat */
    score: number | null;
    needs_credit: boolean;
    credit_text: string | null;
    builtin: boolean;
    /** one of the songs that ship with inframe (works offline); absent from an older server */
    bundled?: boolean;
  };
}

export interface MusicFile {
  id: string;
  file: string;
  license: string | null;
  attribution?: string | null;
  duration_s?: number;
  needs_credit?: boolean;
}

export class EngineError extends Error {
  readonly status: number;
  readonly extra: Record<string, unknown>;
  constructor(status: number, message: string, extra: Record<string, unknown> = {}) {
    super(message);
    this.name = 'EngineError';
    this.status = status;
    this.extra = extra;
  }
}

/** saveEdits met a newer stored document (409): `stored` is that document. */
export class EditConflict extends EngineError {
  readonly stored: EditsDoc | null;
  constructor(stored: EditsDoc | null, message = 'Changed in another tab') {
    super(409, message, { error: 'conflict' });
    this.name = 'EditConflict';
    this.stored = stored;
  }
}

export interface Engine {
  readonly kind: 'local' | 'browser';
  readonly caps: Caps;
  session(): Promise<Session>;
  // media
  openPath(path: string): Promise<MediaInfo>;
  openDialog(): Promise<MediaInfo | null>;
  openFromPhotos(): Promise<MediaInfo | null>;
  locate(f: { name: string; size: number; lastModified: number }): Promise<MediaInfo | null>;
  openFile?(file: File): Promise<MediaInfo>;
  listDir(dir?: string, mode?: 'video' | 'folder' | 'audio'): Promise<DirListing>;
  recent(): Promise<RecentEntry[]>;
  clearRecent(): Promise<RecentEntry[]>;
  media(id: string): Promise<MediaInfo>;
  videoUrl(id: string, opts?: { asMp4?: boolean }): string;
  makeProxy(id: string): Promise<MediaTask>;
  mediaTask(id: string, kind: 'proxy' | 'filmstrip'): Promise<MediaTask>;
  filmstrip(id: string): Promise<Filmstrip>;
  sheetUrl(id: string, n: number): string;
  frameUrl(id: string, t: number, width?: number): string;
  boxes(id: string, t: number, cls: string, signal?: AbortSignal): Promise<FrameBoxes>;
  /** exactly the frame `boxes` returned for `t` (row 20) */
  pickFrameUrl(id: string, t: number, cls: string): string;
  // target
  pickState(id: string, cls: string): Promise<PickState>;
  tap(id: string, cls: string, t: number, box: Box, index: number): Promise<PickState>;
  undoTap(id: string, cls: string): Promise<PickState>;
  setZone(id: string, cls: string, zone: Zone | null): Promise<PickState>;
  resetPick(id: string, cls: string): Promise<PickState>;
  savedTargets(): Promise<SavedTarget[]>;
  targetCropUrl(name: string, i: number): string;
  choosePhotos(source: 'dialog' | 'photos' | 'path', path?: string): Promise<PhotoSet | null>;
  // jobs
  startRun(req: RunRequest): Promise<JobRef>;
  startRank(mediaId: string, req: RankRequest): Promise<JobRef>;
  startPeople(mediaId: string): Promise<JobRef>;
  confirmCandidate(mediaId: string, i: number, cls?: string): Promise<PickState>;
  candidateUrl(mediaId: string, i: number): string;
  /** the thumbnail of person n of the people scan (row 33) */
  personUrl(mediaId: string, n: number): string;
  startExport(runId: string, req: ExportRequest): Promise<JobRef>;
  startTargetAdd(name: string, req: TargetAddRequest): Promise<JobRef>;
  job(id: string): Promise<Job>;
  currentJob(): Promise<Job | null>;
  cancel(id: string): Promise<Job>;
  // runs and edits
  runs(mediaId: string): Promise<RunSummary[]>;
  run(runId: string): Promise<RunData>;
  samples(runId: string): Promise<Samples>;
  saveEdits(runId: string, revision: number, clips: EditClip[]): Promise<EditsDoc>;
  revertEdits(runId: string): Promise<void>;
  resegment(runId: string, p: ResegmentParams): Promise<ResegmentResult>;
  exportResult(runId: string): Promise<ExportResult | null>;
  runFileUrl(runId: string, name: string): string;
  reveal(runId: string, name?: string): Promise<void>;
  // music
  /** the songs that ship with inframe, no network; CC0 ones only with `cc0Only` */
  musicLibrary(cc0Only: boolean): Promise<MusicLibraryResult>;
  musicSearch(q: MusicQuery): Promise<MusicSearchResult>;
  musicGet(id: string, cc0Only: boolean): Promise<MusicFile>;
  /** Pick a free song that fits the reel (a `music-auto` job, polled here until it ends); `onJob` sees each poll (progress, Cancel). Throws when it fails or is cancelled. */
  /** `signal` stops the polling (the caller cancels the job itself) */
  musicAuto(runId: string, q: MusicAutoQuery, onJob?: (job: Job) => void, signal?: AbortSignal): Promise<MusicAutoResult>;
  musicUrl(id: string): string;
  /** the native audio-file dialog (row 52); null = cancelled */
  musicChooseFile(): Promise<{ file: string; license: string | null } | null>;
  quit(): Promise<void>;
  /** Turn a URL this engine handed out (an `inframe-blob:` key) into one an <img> can load. LocalEngine has none: its URLs are real and pass through. */
  resolveUrl?(url: string, signal?: AbortSignal): Promise<string>;
  /** The browser engine's downloadable models (undefined on the Mac). */
  models?: ModelControl;
  /** The page is done with this engine (the store was disposed, e.g. on an engine switch): stop background work and free what it holds. */
  close?(): void;
}
