# T9a design: review UI and local mode (`inframe serve`)

Status: design.

What T9a builds: `inframe serve [VIDEO]` starts a loopback web server that serves a dark, CapCut-like editor. Open a video, pick the
target (tap in a frame, a saved target, or photos), draw a zone, run, see a presence timeline, fix the clips (drag edges, split, merge,
delete, add, undo/redo), and export clips, a reel with CC0 music, or edit lists. Everything runs on the Mac; no request carries video,
frames or embeddings off the machine. T9b later swaps the server for in-browser inference behind the same TypeScript `Engine` interface.

## 0. Facts checked against the code

Read in `src/inframe/`. Each row was opened, not assumed.

| Claim | Verdict | Where |
|---|---|---|
| pick.py: stdlib `ThreadingHTTPServer` on 127.0.0.1, per-run token in the URL, Host and Origin checks | Right, with limits: the token travels only as `?k=` on every request; Origin is checked on POST only; there is no Range support and no static-file serving (one inline page). `QuietServer` and `CLIENT_GONE` handle cancelled fetches. | `pick.py` `make_handler._allowed`, `QuietServer` |
| Picker: one detect worker that serves the newest request, frame/box cache | Right. `Picker.frame(t)` is latest-wins (`Superseded`), cache of 48 entries keyed by rounded time and `("ft", frame time)`. `frame_image(t)` returns the exact frame the boxes came from. The class (`cls`) is fixed at construction, so changing class means a new Picker. | `pick.py` `Picker` |
| Pick page uses CPU_AND_GPU, pipeline runs CPU_ONLY | Right. `PICK_UNITS = "CPU_AND_GPU"` unless `INFRAME_DETECT_UNITS` is set; `run_cut` uses `load_detector(model)` with the default units. T9a changes neither. | `pick.py` `pick_units`, `pipeline.py` `score_detect` |
| `Picker.rank` (photo "Is this you?") can report progress | **Wrong.** It calls `score_detect(..., None, "ocsort")`: no progress, no cancel. The server overrides it (section 2.8). | `pick.py` `Picker.rank` |
| Per-sample scores via `write_samples` | Right: `samples.csv` with header `t,score,x0,y0,x1,y1[,track]`, times in file time (a real run on the Central Park test clip starts at `t=0.007`, the container offset). Written before the cuts, and also on a failed run. | `pipeline.py` `write_samples` |
| `run_cut` progress and cancel | Progress callback `(fraction, message)`: probe 0-2 %, scoring 2-92 % (one call per batch of 8 samples), cutting 92-100 %. Cancel works by raising from the callback (the MCP server does this with `_Cancelled`); during the cuts `run_cut` writes a failed-run `segments.json` and re-raises. No callback runs during probe or `check_static`, so a cancel there waits for the first batch. | `pipeline.py` `run_cut`, `mcp_server.py` `_run_exclusive` |
| Several targets per run | **Not in the code.** `identity.parse_targets` folds every `--target` spec into ONE target (taps plus photos are one identity; two classes are refused). Several people means several runs. T9a: one target per run, several runs per video, with a run switcher and a lane per run (1.5a; section 6, Q4). | `identity.py` `parse_targets` |
| `Picker.save()` writes one file | Right, and it **unlinks** that file when the tap list is empty. `out` is fixed at construction, and so are `photos`, `embedder` and the preset model. Two Pickers with the same `out` overwrite or delete each other's taps; choosing photos or a saved target needs a new Picker (2.8). | `pick.py` `Picker.__init__`, `Picker.save` |
| `pick.py` POST body | `n = int(Content-Length)` then `rfile.read(n)` with no cap and no socket timeout. `serve_http` parses the body strictly instead (section 4). | `pick.py` `do_POST` |
| `run_cut` thresholds | `run_cut` picks enter/exit itself (motion mode: `auto_thresholds`; detect: `DETECT_ENTER`/`DETECT_EXIT`) and records the values used in `segments.json` `settings.enter/exit/merge_gap/min_len/pad`; `settings.sound == "snap"` then moves clip edges after `build_segments`. Re-segment reads those settings (3.4 row 41). | `pipeline.py` `run_cut` |
| `music.get` download | Checks the track id (`ID_RE`), the licence from the per-track Openverse record (`parse_track`, CC0 unless `allow_by`), a size cap (`MAX_BYTES`) and that ffprobe reads it as audio. It does **not** check the host of `t.url`, the scheme, redirects or the file's magic bytes. Part B3 adds that (section 4). | `music.py` `get`, `_get`, `_check_audio` |
| ffprobe/ffmpeg input formats | No call in `src/inframe` passes `-protocol_whitelist` or `-f`; ffprobe picks the demuxer from the content, so a text file named `.mp4` holding an HLS or ffconcat playlist would be followed. The server gates every path it accepts by extension and magic bytes (3.4 Path safety). | `grep -rn protocol_whitelist src/inframe` (no hits) |
| `run_cut` time range | **None.** `Settings` has no start/end, so "run on a short range first" is not possible without a pipeline change; not in T9a. | `pipeline.py` `Settings` |
| Picker temp folder | `picker.new_export_dir()` is `tempfile.mkdtemp(prefix="inframe-picked-")`, so already mode 0700. | `picker.py` |
| Apple photo picker through a Swift helper, no Full Disk Access | Right, but **videos only** (`cfg.filter = .videos`) and it **exports a copy** of each picked original into a temp folder. `picker.pick(out_dir, limit, announce)` has no kind argument. Picking target photos needs an `--images` option (part A0). | `picker.py`, `swift/photo_picker.swift` |
| Export of a reel and edit lists from segments.json | Right: `export.export_run(doc, clips_dir, out_dir, reel, formats, music, beat_sync)`. Edit lists read only `doc["clips"]` times and names (`editlist.cuts`); the reel needs the clip files. So an edited clip list can be exported by building a new doc (section 2.7). | `export.py`, `editlist.py` |
| Music CC0 only by default | Right: `music.licenses_allowed(allow_by=False) == ("cc0",)`; `search`, `get`, `resolve`; cache in `~/Library/Caches/inframe/music`. Only the query text goes to Openverse. | `music.py` |
| Dropping a video onto the page | Browsers never give a dropped file's path. Drop is handled by a stat-only lookup (section 2.10); the main way is an Open button that runs a native dialog on the server. |  |
| A Next.js static build on Vercel | Superseded by the later decision (Cloudflare Pages, open-source license, no accounts). A Vite static build serves both `inframe serve` and Pages; Next.js is not needed. |  |
| Times | Every time in samples, segments, keyframes and cuts is **file time** (pts minus container start, the `ffmpeg -ss` timeline; `VideoInfo.offset`/`end_time`). The UI keeps file time everywhere and converts only at the `<video>` element (section 2.4). | `probe.py` `VideoInfo` |
| `.gitignore` blocks `*.jpg *.png *.webm *.mov ... out/` | Matters for shipping the built UI: the dist folder must hold no `.png`/`.jpg` (hatchling drops git-ignored files from the wheel), so icons are inline SVG; the dist folder must not be named `out`. | `.gitignore`, `pyproject.toml` (hatchling, `packages = ["src/inframe"]`) |

## 1. Screens and flows

### 1.1 Layout (one window, dark, 1440x900 reference; works from 1180x720 up)

```
+--------------------------------------------------------------------------------------------------+
| top bar 48px: [inframe] file name - 20:14 - HEVC 3840x2160 60 fps  | Local engine | Open  Export  |
+-------------------+----------------------------------------------------+-------------------------+
| LEFT 300px        | PLAYER (fills)                                     | INSPECTOR 300px         |
| 1 Video           |  <video> + overlay canvas (boxes, zone, target box)|  selected clip:         |
| 2 Target          |  letterboxed, vertical or horizontal               |  #7  03:12.4 - 03:19.9  |
|   Tap|Saved|Photos|                                                    |  7.5 s, peak 0.91       |
| 3 Zone            |  transport: |< <  [play]  > >|  03:14.20 / 20:14.00 |  starts at key 03:12.0  |
| 4 Run  [Run]      |  speed 1x  [clips only]  [boxes]  volume           |  [split][merge][delete] |
| progress / result |                                                    |  run warnings           |
+-------------------+----------------------------------------------------+-------------------------+
| CLIP STRIP 92px: [thumb #1 0:04.1] [thumb #2 0:07.5] [#3] ...  (selected card outlined)            |
+--------------------------------------------------------------------------------------------------+
| TIMELINE TOOLBAR 36px: undo redo | split merge delete add | snap | run [tap - alice v] | zoom fit ?|
| ruler 22px     0:00     0:30 ...   (labelled "source time")    playhead (white line, full height) |
| filmstrip 56px [][][][][][][][][][][][][][][]                                                      |
| presence 56px  score area of the ACTIVE run (teal above enter, dim between exit and enter)          |
| other runs 14px each (0-3): thin presence bars of the video's other runs, click one to switch      |
| clips 52px     [=====]   [==]      [==========]   blocks with drag handles, keyframe ghost         |
+--------------------------------------------------------------------------------------------------+
```

- Below 1180 px wide the inspector folds into a drawer (button in the top bar). Below 900 px the left panel becomes a drawer too. The
  editor is a desktop tool; phone layout is not a goal for T9a (T9b may revisit).
- **Run switcher** (timeline toolbar, also in step 4): every run of this video (`GET /api/media/{id}/runs`), labelled by target
  (`tap`, `saved: alice`, `class: dog`...) and time. The active run owns the presence lane, the clips track and the inspector; up to 3
  other runs show as thin read-only presence bars under it (their `samples` are fetched once and bucketed the same way), so two targets
  can be compared on one timeline. Clicking a thin bar makes that run active; its own edits load.
- Vertical video (9:16) sits centred in the player area with dark side bars; horizontal fills the width. The player box uses the
  server's `display_w/display_h` (rotation applied), never `videoWidth/videoHeight`, to place overlays (`object-fit: contain` math in
  `model/geometry.ts`).
- Colours and type are CSS custom properties in `web/src/styles.css` (see the mockup): background `#0e0f11`, panels `#16171a`, raised
  `#1e2024`, borders `#2a2d33`, text `#e7e8eb`, muted `#9aa0aa`, accent cyan `#22c5e5`, presence teal `#2dd4bf`, clip blocks
  `#1d6fd8 -> #2a8cf0`, zone yellow `#facc15` and boxes green `#4ade80` (same as the pick page), danger `#f87171`. System font stack,
  no web fonts (no network). Dark only.

### 1.2 Open the video (empty state of the whole app)

The editor area shows one centred card: "Open a video" with three buttons and a drop zone:

1. **Open file...** (`POST /api/media/open-dialog`): macOS native open panel via `osascript` (no build, no Full Disk Access). On other
   systems the button opens the built-in **file browser** (the cross-platform twin, `GET /api/fs/list`).
2. **From Photos...** (macOS only, `POST /api/media/photos-pick`): the existing PHPicker helper; it exports a copy of the one picked
   original into a temp folder (removed when the server stops unless `--keep-picked`).
3. **Browse...**: the built-in file browser (home folder and `/Volumes/*`, folders and video files only, hidden folders and `~/Library`
   never listed). A path field accepts a pasted absolute path.
4. **Drop a file** anywhere: the browser gives name, size and modified time but no path. The UI calls `POST /api/media/locate`; the
   server looks for exactly one file with that name, size and mtime in the recent folders and in `~/Movies`, `~/Downloads`, `~/Desktop`,
   `~/Documents` (depth 2, `stat` only). One match opens it; none or several shows "Browsers hide where a dropped file lives. Use Open
   file... instead." No upload, no copy.
5. **Recent** list under the card (last 20 opened paths, stored in the serve state folder, "Clear" link).
6. `inframe serve VIDEO` opens that file at start.

After opening: the player loads the original (section 2.4). The left panel shows step 1 as done (name, duration, codec, size,
"slow motion 240 fps" badge when `capture_fps >= 100`). The filmstrip fills in as sheets are made. If earlier runs of this video exist
(`<stem>_inframe/*/segments.json` whose `source.path` is this file), a "Runs" dropdown in step 4 lists them and the newest is loaded,
with its edits.

Loading and error states of this step:

| State | What the user sees |
|---|---|
| Probing | Card button spinner "Reading video..." (ffprobe takes under 1 s; keyframe list for a 20-min HEVC takes a few seconds [unverified]) |
| Not a file / gone | Red line under the card: "No such file: PATH" |
| Not a video (ffprobe fails) | "This file has no video stream ffprobe can read." |
| Path refused (hidden or under ~/Library) | "inframe serve does not open files there." |
| Dialog cancelled | Nothing changes |
| Photos helper missing Xcode tools / build fails | The picker's own message (PickerError text) |
| Playback test fails | Banner over the player: "This browser cannot play VP9/HEVC/Theora here. Making a preview copy (H.264, 720p)... 34 %" with a progress bar; editing and picking work meanwhile (they use server frames) |
| Proxy fails | Banner: "No preview: ffmpeg failed (message). Clips and export still work; the timeline shows filmstrip frames only." |
| Filmstrip making | Grey shimmering cells, filled sheet by sheet |

### 1.3 Pick the target (step 2): four choices, one active

Tabs in step 2: **Anyone** (default: every person, or every object of a class from a class dropdown, `class:dog`), **Tap**, **Saved**,
**Photos**. Advanced users also get **Motion** (no model, static camera) under the Run panel's Advanced section.

**One active target, shown in a Target summary card** (top of step 4, always visible): mode, class, taps count with time chips or
photos count or saved name, zone on/off, and the preset it forces (a tap target records its detector preset). Combination rule, the
same as the CLI: exactly one of Anyone / class / Tap / Saved / Photos is the target; a zone combines with any of them; the only
mixed target is the one `identity` already makes, photos confirmed by a tap ("Is this you?" writes the tap into the photos target's
target.json, and the card then reads "Photos + 1 confirm tap"). Switching tabs never deletes taps; it only changes which target Run
uses, and the card says which.

**Tap** (Picker, latest-wins worker):
- Entering the tab pauses the video and switches the player to *pick mode*: the overlay shows the server frame (`frame.jpg` for the
  time `boxes` returned) with numbered green boxes. Scrubbing or playing hides them; pausing again asks for boxes at the new time
  (debounced 250 ms; older requests are aborted with AbortController, `409 superseded` is ignored).
- Click inside a box: `POST /pick/tap`. The tap list under the tabs shows "3 taps" with time chips (click a chip to seek there).
  Undo last tap. Hint: "Tap the same person again in another stretch if they leave and come back."
- The class dropdown (person by default) uses the Picker of that class; each class keeps its own taps (2.8).
- Optional **Who is in it?** button (people scan, `people.scan_people`, one detect+track pass, about 2 min for 20 min of video):
  results show as cards with crops; clicking one sets the target to that person's `tap:person_NNN.json`. This is a job (one at a
  time) and can be cancelled. [Part A3 builds the routes; the UI card is in C3.]
- States: detector loading ("Loading the detector, about 5 s the first time"), no box above 0.4 ("Nobody found in this frame: try
  another moment"), error from the worker (red line with the message).

**Saved** (`targets.list_targets`):
- A grid of saved targets: name, first crop thumbnail, number of crops, "stale" badge when `summary().embeddings` has no current valid
  entry (it can still work: the thumbnails are embedded again).
- Choosing one sets the target to `saved:NAME`. Empty state: "No saved targets yet. After a run, use Save as target in the
  inspector, or `inframe target add`."
- Error states: unreadable target (the `error` field of `list_targets`), shown as a disabled card with the message.

**Photos** (photo mode, `identity.PhotoTarget`):
- Buttons: **Choose folder...** (macOS `osascript choose folder`; twin: the file browser in folder mode) and **From Photos...**
  (PHPicker with `--images`, exported copies in a temp folder). The panel shows "12 photos" (from `identity.photo_files`) and never
  shows the photos themselves unless the user clicks "Show" (they are the user's photos; nothing is uploaded either way).
- Run starts photo mode directly. If the run ends with `identity.Ambiguous` (the CLI's exit 3), the job ends in state
  `needs_confirm` and the panel says why (the message) and offers **Find who it is** (rank job: reads the whole video once, about as
  long as a run). Its result is the "Is this you?" row: up to 5 candidate crops (`/cands/{i}.jpg`) with time and score; clicking one
  writes the confirm into target.json (`Picker.confirm`) and the user presses Run again (target becomes `tap:<target.json>`, which
  carries the photos folder, as the CLI does).
- **Save these photos as a target** (name field): `targets.add_crops(name, crops_from("photos:DIR"))` as a job.

### 1.4 Draw a zone (step 3)

- **Draw zone** turns the player overlay into a drawing surface (crosshair cursor). Drag a rectangle; it stays with 8 handles (corners
  and edges) to resize and a body to move; Esc cancels; Delete clears. Stored as `[x, y, w, h]` fractions of the upright frame, the
  same as `zone.Zone` (min 0.02 per side, clamped to the frame; rounding as `pick._round_zone`).
- A note under it: "Only people inside the zone count. Needs a fixed camera: if the camera moves, the run ignores the zone and says so."
  (`run_cut` checks `camera.check_static` and puts that warning in `segments.json`.)
- Option "Count a person when: centre inside (default) / at least N % of the box inside" (`zone_overlap`).
- In tap mode the zone is also sent to the Picker (`/pick/zone`), so target.json carries it like the CLI pick page does.

### 1.5 Run (step 4)

- **Preflight** (a checklist at the top of step 4, from `GET /api/session` `preflight`): ffmpeg/ffprobe found, detector models present
  for the chosen preset (`models` lookup, no load), embedder extra importable (only for Saved/Photos), tracking extra (only when the
  target needs tracks), librosa (only for beat sync, shown in Export). A missing item turns red with the fix command ("Run `inframe
  fetch-models`", "Run `uv sync --extra identity`") and disables Run before it is pressed, not after.
- **Where it writes**: one line under Run, "Writes to ~/Videos/clip_inframe/tap/" with **Change...**. The first run into a folder
  that has no `<stem>_inframe/` yet asks once ("Make clip_inframe here? / Use ~/Movies/inframe instead"); a folder that is not
  writable (`os.access`) goes straight to the `~/Movies/inframe` choice.
- **Estimate** before Run: "about 4 min (estimate)". Source: the seconds per source minute of this preset measured by earlier UI runs
  on this Mac (`timing_s.total` / duration, kept in the serve state folder, median of the last 5); before any run, the model-call
  cost measured in development times the number of samples, labelled "at least". It never claims more than that.
- Controls: preset (fast / balanced / accurate; default balanced, or the preset recorded in the tap's target.json, which `cut` would
  enforce anyway via `tap.resolve_preset`), slow motion (shown when `capture_fps >= 100`: "Slow down clips (auto, x8)" maps to
  `slowmo="auto"`; a factor field and **Keep sound (stretched)** for `slowmo_audio`), and **Advanced** (pad, min length, merge gap, enter, exit, min box, tracker ocsort/botsort, activity riding,
  mode motion). Defaults are `pipeline.Settings()`.
- **Run** starts the job (`POST /api/runs`). Only one job runs at a time; while one runs, Run, Export and the rank/people jobs are
  disabled with the tooltip "A job is running".
- Progress: a bar in step 4 and a thin bar over the timeline: `fraction`, the message from `run_cut` ("detect: 312 of 1214 s"), elapsed
  and an ETA computed in the UI from the fraction rate (shown after 5 % to avoid wild guesses). **Cancel** stops at the next progress
  report (one batch, normally under 2 s; during probe and the camera check it waits for the first batch).
- Done: the presence lane and clip blocks appear; warnings from `segments.json` show as yellow lines in the inspector ("the camera does
  not look static..."); "no stretch found" shows as an empty-state message on the clips track with the highest score.
- Failed: red panel with the message (`run_cut` messages already say what was saved, e.g. partial `samples.csv`); if samples exist the
  presence lane still shows them.
- Cancelled: "Run cancelled. Nothing was kept." (the run folder holds a failed-run record; its earlier edits file is kept aside, see
  2.6).
- Re-running the same target into a folder whose run has edits asks first: "This run has 6 edited clips. Run again: your edits are
  kept aside and can be restored. / Cancel". Then the earlier edits move to `edits.prev-<UTC time>.json` and a banner offers "Restore
  my earlier edits" (applies them on top of the new run as user clips).

### 1.5a Several targets on one video

One run = one target (the code folds every spec into one identity, section 0). To follow two people, the user picks the second target
and presses Run again: each run gets its own folder (slug), its own edits and its own lane. The run switcher (1.1) moves between them;
export always works on the active run and writes into that run's folder, so two targets give two clip sets and two reels. A combined
reel of several runs is not in T9a (section 6, Q4).

### 1.6 Review and edit clips

Selection: click a block (or its card in the clip strip) selects it and seeks to its start; Shift-click adds a range of blocks for
merge; Esc clears. Hover shows start, end, duration.

Keyboard and access rules (`model/keys.ts`):
- Hotkeys are ignored while focus is in an `input`, `textarea`, `select` or a `contenteditable` element, and while a dialog is open
  (except Esc). Cmd/Ctrl combinations other than the ones in the table pass through to the browser.
- **?** opens a shortcut sheet (the table below, generated from the same key map) and the empty state links to it.
- Timeline blocks are `role="button"` with `aria-label="Clip 7, 07:12.4 to 07:19.9, 7.5 seconds"`, reachable with Tab in time order;
  Enter selects and seeks, `[`/`]` with the block focused move its start/end by one frame (Shift: 1 s). The clips track is
  `role="listbox"`, the playhead position is announced through an `aria-live="polite"` region at most once a second while paused.


| Action | Mouse | Key | Rule |
|---|---|---|---|
| Play / pause | transport button, click on video | Space or K | |
| Back / forward | J / L (1x, 2x, 4x on repeat) | J, L | |
| Frame step | | Left / Right (paused) | one frame, using `rVFC` mediaTime; falls back to `1/fps` |
| 1 s step | | Shift+Left / Shift+Right | |
| Previous / next clip | | Up / Down | seeks to the clip start and selects it |
| Trim start / end | drag a block edge | I / O set the selected clip's start / end to the playhead | clamped by neighbours and the video bounds; min length 0.2 s; while dragging, the player seeks to the dragged edge (throttled to one seek per animation frame, on whatever source is playing: original or proxy) and a tooltip shows the edge's time and the clip length |
| Split | toolbar or inspector | S | at the playhead, inside the selected (or the hovered) clip; both parts at least 0.2 s |
| Merge | toolbar | M | the selected clip with the next one, or all shift-selected clips into one (gaps become part of the clip) |
| Delete | toolbar | Delete / Backspace | |
| Add clip | toolbar, or double-click an empty part of the clips track | N | 2 s centred on the playhead, shrunk to fit the gap |
| Undo / redo | toolbar | Cmd+Z / Shift+Cmd+Z (Ctrl on other systems) | 200 steps; a drag is one step; the history of each run is kept in `sessionStorage` (per tab, survives a reload, capped at 2 MB, dropped silently when storage throws); a fresh tab starts with an empty history and says so in the undo tooltip |
| Snap on/off | toolbar magnet | hold Alt while dragging to disable once | snaps to playhead, presence edges, other clip edges, keyframes; then to the frame grid |
| Zoom | slider, Cmd+scroll, pinch | + / - / 0 (fit) | anchored at the pointer (or playhead for keys) |
| Scroll | trackpad / wheel | | horizontal pan |
| Play clips only | transport toggle | P | playback jumps over gaps: a preview of the edit |
| Show target box | transport toggle | B | draws the run's box for the current sample (from samples.csv) over the video |
| Export | top bar | Cmd+E | |

- Each clip block shows its number, a faint keyframe ghost where a stream-copy cut really starts (`cut.snap_to_keyframe`, the
  `expected_copy_start` rule), and a hatched left edge when the ghost is more than 0.1 s before the start ("stream copy starts 0.6 s
  early at the keyframe"). Detector clips are blue; user-made or changed clips get a small dot.
- Times on the ruler, the transport and the inspector are **source (file) time**. For a run with slow motion the inspector adds
  "slow-mo x8: source 2.0 s, output 16.0 s" (`cut.retimed_facts`), and the export dialog totals use output durations.
- The inspector shows the selected clip: start, end, duration, peak score in the clip (from samples), the keyframe start, origin
  (detector / edited / split / merge / added), and buttons Split, Merge with next, Delete, **Save as target** (adds crops of the person
  in this run to a saved target: `targets.add_crops(name, targets.crops_from_run(segments.json))`; enabled for tap, photo and saved
  runs, which keep one person).
- **Re-segment** (inspector, collapsed): sliders for enter, exit, merge gap, min length, pad, seeded from the run's own
  `segments.json` `settings` (`enter`, `exit` as used, `merge_gap`, `min_len`, `pad`), previewing the clips that
  `segments.build_segments` would make from the same samples (`POST /resegment`, no write). With the seeded values and
  `settings.sound != "snap"` the preview equals the run's clips exactly (a test checks it); for a sound-snapped run the panel says
  "Re-segment drops the sound snap". **Apply** replaces the clip list (one undo
  step; origin `resegment`).
- Edits autosave 400 ms after each committed change (`PUT /edits`); the top bar shows "Saved" / "Saving..." / "Not saved - retry". A 409
  (another tab saved) shows "Changed in another tab - Reload".
- A badge "edited: 3 changed, 1 deleted, 2 added" next to the clip count; **Revert to detector** is an ordinary undoable step (it sets
  the clip list to the detector clips and saves); the confirm dialog names the count ("Undo 6 edits?"). A 409 reload replaces the
  state and clears the history (the stored doc is newer than every step in it).
- Empty states: no run yet ("Pick a target and press Run. The presence of the target over time will show here."), run with 0 clips
  ("No stretch reached the enter threshold (highest score 0.41 of 0.50). Try Re-segment, another target or preset.").

### 1.7 Export (Cmd+E, dialog)

Three sections with check boxes; each says where it writes. A **summary line** on top: "8 clips (selected) of 23 - 1 min 12 s output -
about 640 MB - 3 clips start more than 0.5 s early at a keyframe". **Which clips**: All / Selected only. **Output folder**: default
`<run folder>/edited/`, or **Choose folder...** (native dialog; twin: the file browser in folder mode). When the folder already holds
an export, a confirm names it ("Replace the export from 10:12 today (23 clips, a reel)?"); export never deletes anything else.
Frame-exact smart cut is not in T9a (section 6, Q7): the dialog shows "Clips start at the keyframe before the clip; frame-exact
cutting comes later".

1. **Clips**: one file per clip, stream copy (no re-encode; starts at the keyframe at or before the clip start), retimed when the run
   used slow motion, plus a JSON sidecar per clip and the LosslessCut CSV.
2. **Reel**: the clips joined into `<stem>_reel.<ext>`. Music: "No music (keep the clips' sound)" or **Find music**: a mood field
   ("upbeat" default), minimum length (default: the reel's length), Search (`music.search`, CC0 only; a "Also allow CC BY (credit file
   is written)" check box sets `allow_by`). Results: title, creator, length, licence, **Play** (downloads into the music cache first,
   then plays from the server) and **Use**. "Use a music file..." opens the native dialog (audio types). **Beat sync** check box
   (disabled with "needs the music extra: uv sync --extra music" when librosa is missing); it re-encodes the reel with
   hevc_videotoolbox. The Openverse notice (`music.OPENVERSE_NOTICE`) is shown under the results.
3. **Edit lists**: FCPXML (Final Cut, FilmCraft), FCP7 XML (Premiere), Kdenlive, EDL (CMX3600), OTIO. Disabled formats show why
   (EDL above 60 fps; EDL/FCP7 at a non-SMPTE rate), with the same rules as `editlist.write_edit_lists`.

Browser engine (T9b): the dialog shows only what `Engine.caps.output` allows (2.10): clips and edit lists are offered as downloads
or written to a folder the user grants (File System Access), the reel without beat sync, and no Reveal.

**Export** runs a job with progress ("cutting clip 4 of 23", "reel", "edit lists"). The result list shows each output with ok / FAILED
and its check (reel joins, largest cut error in frames, LUFS for a music reel), warnings, **Reveal in Finder** (or the file manager), and
for the reel a **Play** button (served from the run folder). Errors: "music: Openverse is rate-limiting..." (MusicError text), reel join
check failed (problems listed), no clips ("Nothing to export: the clip list is empty").

### 1.8 States that apply everywhere

| Situation | Behaviour |
|---|---|
| First start | The empty state shows three numbered steps (Open, Pick, Run), the preflight list (1.5) and "Press ? for shortcuts". |
| Server stopped (fetch fails) | Full-width red bar: "inframe serve is not running. Start it again in the terminal." Edits not yet saved stay in memory and are retried when the server answers. |
| Token wrong (403) | "This page belongs to an older inframe serve. Open the address the terminal printed." |
| Job already running (409 `busy`) | Toast with the running job's kind and a link to its progress |
| Models missing (`models.ModelsMissing`) | The message, plus "Run `inframe fetch-models`" |
| Embedder missing (`identity.EmbedderUnavailable`) | The message, plus "Run `uv sync --extra identity`" |
| Tracking extra missing for photo/saved mode | The message from the run |
| Disk full / ffmpeg failure | The job error text |
| 20-min 1.2 GB HEVC .mov | Plays from the original with Range requests; filmstrip in the background; samples (about 6000 at 5 fps) drawn per pixel column; keyframe list loaded once; nothing loads the whole file in the browser |

## 2. Architecture

### 2.1 Front-end stack

| Package | Version (pin exact in package.json) | Licence | Why |
|---|---|---|---|
| vite | 6.x (latest 6 minor at build time, at least two weeks old) | MIT | builds a plain SPA into one folder that both the Python server and Cloudflare Pages serve unchanged |
| preact | 10.x | MIT | 4 KB React-compatible UI core; keeps a path to React libraries via preact/compat |
| @preact/signals | 1.x or 2.x | MIT | small reactive store for player time, selection and edits without prop drilling |
| @preact/preset-vite | 2.x | MIT | JSX for Preact in Vite |
| typescript | 5.x | Apache-2.0 | types; build-time only |
| vitest | 3.x | MIT | unit tests for the edit model, timeline math, presence downsampling and the local engine; also the runner T9b's golden vectors need |
| jsdom | 25.x or newer | MIT | DOM for the few component tests (vitest environment) |

Everything else is our own code: the timeline (canvas for ruler, presence lane and filmstrip; DOM for clip blocks and handles), the player
controls, zone drawing, dialogs, icons (inline SVG, about 20 hand-drawn 24x24 paths; no icon package, `.gitignore` blocks png/jpg).

Components looked at and not used:
- `@xzdarcy/react-timeline-editor` (MIT): closest fit, but React-only (peer react>=18, plus react-virtualized and interactjs), quiet
  since v1.0.0 in Jan 2026, and it gives only the draggable rows; the filmstrip, presence lane, keyframe ghosts, split/merge rules and
  snapping are ours anyway. Our clip lane is about 300 lines.
- wavesurfer.js (BSD-3) regions: audio-waveform tool, no filmstrip or split semantics.
- OpenCut, Omniclip (MIT): whole apps, not libraries; read for patterns only (trim/split/undo).
- Media Chrome (MIT): only skins `<video>`; our transport is a few buttons and needs frame stepping it does not give.
- Vidstack (maintenance mode), Video.js (heavier, streaming features unused on loopback): no.
- peaks.js (LGPL-3.0), Twick (Sustainable Use License), openvideodev/react-video-editor (custom licence): licence not allowed.
- Next.js: not needed (no SSR); Vite's static output already suits Cloudflare Pages.

### 2.2 Shipping the UI inside the Python package (node only to build)

- Source: `web/` at the repo root (`package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/`, `test/`).
- Build: `npm --prefix web ci && npm --prefix web run build` writes `src/inframe/web/dist/` (`build.outDir: "../src/inframe/web/dist"`,
  `emptyOutDir: true`, `base: "./"`, `build.assetsInlineLimit: 0`, no source maps in dist). Output: `index.html`, `assets/*.js`,
  `assets/*.css`, `favicon.svg`. No png/jpg.
- **The built dist is committed.** Hatchling includes everything under `src/inframe` that git does not ignore, so `uv tool install`,
  `uv sync` and a detached stable worktree all get the UI without node. Same pattern as marimo/Gradio but with the
  output in git, because the project is installed from a git checkout, not from PyPI.
- Node is pinned: `web/.nvmrc` holds one exact Node version and `package.json` `engines.node` the same major; `npm ci` with the
  committed lock. Licences: `rollup-plugin-license` (MIT) in `vite.config.ts` writes `THIRD_PARTY_LICENSES.txt` into dist (the
  notices Vite's minifier strips) and fails the build when a bundled package's licence is not in MIT, ISC, BSD-2-Clause,
  BSD-3-Clause, Apache-2.0, 0BSD.
- Staleness guard: `tools/check_web_dist.sh` checks `node --version` against `.nvmrc` (exit 2 with a message when it differs), builds
  into a fresh temp dir (`vite build --outDir "$tmp" --emptyOutDir`) and runs `diff -r "$tmp" src/inframe/web/dist` (exit 1 on any
  difference, untracked or not). `tests/test_serve_dist.py` checks `index.html` exists and references only files in `dist/assets`,
  that `THIRD_PARTY_LICENSES.txt` exists, and that no file in `dist` matches `https?://` outside an allowlist (`www.w3.org/2000/svg`,
  licence URLs inside `THIRD_PARTY_LICENSES.txt`). It skips when dist is absent unless `INFRAME_REQUIRE_DIST=1`, which parts C1-C3 set.
- Until part C1 commits dist, `serve.py` answers `/` with a small built-in page ("The web UI is not built: run `npm --prefix web ci &&
  npm --prefix web run build`"). Only parts C1-C3 write `src/inframe/web/dist/`.
- `web/node_modules/` goes into `.gitignore`.
- At run time the server reads files with `importlib.resources.files("inframe") / "web" / "dist"`.

### 2.2a Cloudflare Pages readiness (T9c; nothing deployed in T9a)

- Build command `npm --prefix web ci && npm --prefix web run build -- --outDir web/pages-dist --mode pages`, output dir
  `web/pages-dist`; the `pages` mode selects the browser engine only and leaves out the local engine's session probe.
- `web/public/_headers` (copied into the output; `serve.py` sends the same headers itself and does not serve files starting with
  `_`): the CSP of section 4 with T9b's additions, `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy:
  require-corp` (needed only if onnxruntime-web uses wasm threads / SharedArrayBuffer; [unverified] whether the WebGPU path needs
  them), `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`.
- `LICENSE` (the project's open-source licence, chosen at T9c) and `THIRD_PARTY_LICENSES.txt` in the output.
- Check: the same `test_serve_dist.py` URL scan runs on `pages-dist` (no off-origin URL except the pinned model host of T9b), and the
  output holds no media file (`find web/pages-dist -type f ! -name '*.js' ! -name '*.css' ! -name '*.html' ! -name '*.svg' ! -name
  '*.txt' ! -name '_headers'` prints nothing).

### 2.3 The server

- **stdlib `ThreadingHTTPServer`, HTTP/1.1** (`protocol_version = "HTTP/1.1"`, exact `Content-Length` on every response), in a new module
  `serve.py`, reusing pick.py's `QuietServer`, `CLIENT_GONE`, `Picker`, `grab_frame` and the token/Host/Origin rules (made stricter, 4.).
  Reasons: zero new Python dependencies; the threading model already fits Picker's worker; Range for one file is about 40 lines; progress
  is polled, so no SSE or async is needed. Starlette+uvicorn (BSD) is the upgrade path if streaming ever matters; not needed now.
- **`serve_http.py`** (part A1, imports nothing from other `serve_*` modules): `Route(method, regex, handler, media=False)`,
  `Request` (`method, path, params, query, headers, json, body`), `Response`, `json_response(obj, code=200)`, `error(code, message,
  **extra)` (body `{"error": message, ...extra}`), `bytes_response(data, ctype)`, `file_response(path, ctype, req)` (Range-aware,
  below), `read_body(handler, limit)` (section 4). `media=True` marks the routes that accept the media key in `?m=` (section 4).
  Handlers are `handler(app, req) -> Response`; route modules annotate `app: "App"` under `TYPE_CHECKING` only.
- **Route modules.** `serve.py` imports, in this order, `serve_media`, `serve_pick`, `serve_runs`, `serve_export`, `serve_music` with
  `importlib.import_module`; each exports `ROUTES: list[Route]` and `attach(app) -> None`. A module that is absent
  (`ModuleNotFoundError` naming exactly that module) is skipped, so each part is testable before the later ones exist. Route modules
  import `serve_http`, `serve_paths`, `serve_jobs` and core `inframe` modules, never `serve.py`, so there is no import cycle.
- **`App` (in `serve.py`) fields the route modules may read or set**, and nothing else:
  `token: str`, `media_key: str`, `port: int`, `offline: bool`, `cache_root: Path`, `state_root: Path`, `lock: threading.RLock`,
  `media: dict[str, Media]` (A1 `open_media` fills it), `jobs: serve_jobs.JobRunner` (A1), `photos_dirs: dict[str, Path]` (A3 sets),
  `pickers: dict[tuple[str, str, str], ServePicker]` (A3 sets; key `(media_id, cls, target_key)`), `media_tasks` (A2 sets),
  `runs: dict[str, Path]` (B1 sets), `stop()`. Methods: `open_media(path) -> Media` (path rules and magic check of 3.4, probe,
  register), `media_or_404(id) -> Media`, `run_or_404(rid) -> Path` (B1 attaches it), `out_root(media) -> Path` (the
  `<stem>_inframe` folder after the 1.5 confirmation, or `~/Movies/inframe/<stem>_inframe`).
- `Media` (dataclass, `serve.py`, part A1): `id, path (resolved), size, mtime_ns, info (probe.VideoInfo), capture_fps, cache_dir,
  picked_copy: bool`.
  `id = sha1(f"{path}\0{size}\0{mtime_ns}")[:12]`: stable across restarts so caches are reused, different when the file changes.
- Range (`file_response`): `Accept-Ranges: bytes`; one range `bytes=a-b`, `bytes=a-`, `bytes=-n`; 206 with `Content-Range`; 416 with
  `Content-Range: bytes */size` when unsatisfiable; multi-range requests get the full 200 (browsers do not send them for media);
  `If-Range` ignored (files are immutable while open; a changed file gets a new media id). Copy with `os.sendfile` when available,
  else 1 MiB chunks; `CLIENT_GONE` ends quietly. `HEAD` supported for the same paths.
- Content-Type by extension: `.mp4 .m4v -> video/mp4`, `.mov -> video/quicktime`, `.webm -> video/webm`, `.mkv -> video/x-matroska`,
  `.ogv -> video/ogg`; `?as=mp4` forces `video/mp4` for `.mov` (same ISO-BMFF family; a fallback if a browser refuses
  `video/quicktime`, [unverified] which one does).
- State folders: cache root = `$INFRAME_CACHE/serve` else `~/Library/Caches/inframe/serve` (macOS) / `~/.cache/inframe/serve`
  (`models.cache_dir` rule). Per media: `<cache_root>/<media_id>/` holding `filmstrip/sheet_NNN.jpg`, `filmstrip.json`, `proxy.mp4`,
  `proxy.json`. Serve state (recent list): `~/Library/Application Support/inframe/serve/recent.json` (macOS) /
  `$XDG_DATA_HOME/inframe/serve/` (the `targets.store_dir` rule; `$INFRAME_SERVE_STATE` overrides it, tests use it). These hold derived
  frames and paths of the user's videos and never leave the Mac.

### 2.4 Video playback

- Default: the original, served by Range (`/api/media/{id}/video`). HEVC .mov from an iPhone plays in Safari and in Chrome on macOS
  (hardware decode; Chrome 107+ per secondary sources, [unverified] for Main10 at 240 fps).
- **Playability test** in the UI on every open (`engine/playability.ts`): set `src`, wait for `loadedmetadata`, seek to the middle,
  wait for one `requestVideoFrameCallback` (rVFC) frame, all within 6 s; `canPlayType` is only a hint. Failure on `.mov` retries once
  with `?as=mp4`; failure again (or `.ogv`/Theora, which Safari never plays) asks the server for a **proxy**.
- Proxy (`POST /api/media/{id}/proxy`, a media task, not "the job"): `ffmpeg -nostdin -nostats -loglevel error -i SRC -map 0:v:0
  -map 0:a:0? -vf "scale=..."` to a 720-px short side (never upscale; rotation baked upright by ffmpeg's autorotate), H.264 with
  `h264_videotoolbox -b:v 4M` on macOS (twin: `libx264 -preset veryfast -crf 26` where present, else `mpeg4 -q:v 5`), `-g 15`
  (short GOP for scrubbing), AAC 128k, `-movflags +faststart`, `-progress pipe:1` parsed for `out_time_us` (progress =
  out_time / duration). Output `<cache>/proxy.mp4` (`.part` then rename). `proxy.json` records `first_frame_file_time` measured
  with `probe.first_packet_pts`-style ffprobe on the proxy and the source's `offset`. Cost on a 20-min 4K HEVC: [unverified, measure on
  a quiet machine]. Only one proxy at a time; proxies older than the 3 newest are deleted.
- **Clock mapping** (the only place file time meets player time): `fileTime = video.currentTime + clockOffset`. After the playable
  source loads, the UI seeks to 0, reads the first rVFC `mediaTime` m0 and sets `clockOffset = first_frame_file_time - m0`, where
  `first_frame_file_time` is `info.offset` for the original (the proxy's own value from `proxy.json` for a proxy). Default 0 if rVFC is
  missing. Browser check: the `late` synthetic clip (video starts at 2.0 s in the file, white flash at file time
  7.0-7.4 s) must show white at fileTime 7.2 and black at 6.8 and 7.6. [unverified until that check runs]
- Frame-exact readout uses rVFC `mediaTime`; `timeupdate` is the fallback. Frame stepping seeks by `1/fps` from the last rVFC
  `mediaTime`. Playback rate 0.25x-2x (slow-motion review).
- Rotation and mirroring: browsers apply the display matrix of mp4/mov; overlays use `display_w/display_h`. Server frames
  (`frame.jpg`, filmstrip) are upright (decode.upright).

### 2.5 Filmstrip, clip thumbnails, presence lane

- **Filmstrip** (media task, started at open, part A2): PyAV decode of keyframes only (`decode.iter_samples(path, info, fps=None,
  long_side=...)`, which already picks VideoToolbox for HEVC and applies rotation), one decode pass, no ffmpeg tile filter (we need the
  exact time of each cell, which the filter does not report). Cell height 108 px (width from the display aspect, even, at most 192);
  one cell per slot of `max(1 s, duration / 600)` seconds, using the first keyframe at or after the slot start (slots a sparse GOP skips
  reuse the previous cell). Sheets of 10x10 cells, JPEG quality 70, written as each sheet fills; `filmstrip.json`:
  `{"version":1,"cell_w":..,"cell_h":..,"cols":10,"rows":10,"cells":[{"t":12.012,"sheet":0,"i":7},...],"sheets":3,"done":true}`. Cost on
  a 20-min HEVC: [unverified]; the UI shows partial sheets meanwhile. Cache key is the media id.
- The filmstrip track draws, for each visible cell slot at the current zoom, the cell nearest that time (CSS-sprite draw into the canvas
  with `drawImage` from the sheet `Image`).
- **Clip thumbnails** (clip strip cards): the filmstrip cell nearest each clip's peak-score sample (no extra decode). The selected clip
  card also loads `frame.jpg?t=<peak time>&w=320` for a sharper poster.
- **Presence lane** (`GET /api/runs/{id}/samples`): arrays `t`, `score`, `box`, `track`, plus `enter`, `exit`, `mode`. The UI buckets
  samples into pixel columns for the visible range (`model/presence.ts`: max score per column) and draws an area: teal where the column
  max >= enter, dim teal between exit and enter, grey below exit; dashed lines at enter and exit; the `presence_intervals` as a 3-px bar
  along the top. Motion-mode scores are not 0..1: the lane scales to `max(score, enter * 1.25)`.

### 2.6 Jobs and edits (`JobRunner` part A1; runs and edits part B1)

**One job at a time.** `serve_jobs.JobRunner` holds at most one running job (`run`, `rank`, `people`, `export`, `target-add`); a second
start returns `409 {"error": "busy", "job": <running job>}`. A job runs in a daemon thread; its `progress(fraction, message)` callback
stores both (with a monotonic timestamp) and raises `JobCancelled` when the cancel event is set (the MCP `_run_exclusive` pattern).
Finished jobs stay readable for 10 minutes (`GET /api/jobs/{id}`). The UI polls every 500 ms while a job runs (1000 ms when the tab is
hidden); no SSE.

**Run folder.** `<video dir>/<stem>_inframe/<slug>/` where slug is `any-person`, `class-<name>`, `tap`, `saved-<name>`, `photos`,
`motion`. Same parent as the CLI default (`<stem>_inframe/<mode>`), so `inframe export <folder>` works on UI runs, but different leaf
names so the UI never overwrites a CLI run. The tap's target.json lives at `<stem>_inframe/target.json`, the `inframe pick`
default, so `inframe cut --target tap:` finds it too. For a Photos-picked copy (temp
folder), the run folder goes to `~/Movies/inframe/<stem>_inframe/<slug>/` (macOS; `~/inframe` elsewhere) so it survives the temp
folder's deletion. `run_id = sha1(str(run_dir))[:12]`. The folder is created with `mkdir(mode=0o700)` the first time (an existing
folder keeps its mode).

**Run folders are untrusted input** (a shared or downloaded `<stem>_inframe/` can hold anything). `serve_edits` opens
`segments.json`, `samples.csv`, `edits.json` and `target.json` only after `os.lstat` shows a regular file (no symlink) inside the run
folder; a run is registered only when its `source.path` resolves to the path of an opened media; any path read from inside those files
(target.json `photos`, `video`) goes through `serve_paths.check_read` (regular file or folder, not hidden, not under `~/Library`)
before use, and crops or candidates are only ever produced by our code, never read from a path named in the JSON.

**Edit model, on disk.** The detector's `segments.json` is never written by the UI. User edits live next to it in `edits.json`:

```json
{
  "format": 1,
  "base": {"segments_sha256": "<sha256 of segments.json bytes>", "n_clips": 23},
  "revision": 7,
  "updated": "1970-01-01T00:00:00Z",
  "next_id": 31,
  "clips": [
    {"id": "d1", "start": 12.207, "end": 19.53, "origin": "detector"},
    {"id": "u27", "start": 40.0, "end": 42.0, "origin": "added"}
  ]
}
```

- `origin` is one of `detector` (unchanged from segments.json clip N, id `dN`), `edited` (a detector clip whose edges moved), `split`,
  `merge`, `added`, `resegment`. Ids are unique strings `[a-z0-9]{1,16}`; new ids are `u<next_id>`.
- Rules checked by the server (`serve_edits.validate_clips`) and the TS model alike: at most 2000 clips; finite numbers; sorted by
  start; `offset <= start < end <= end_time` (with 1 ms slack); `end - start >= 0.2`; no overlap (`end_i <= start_{i+1}`); unique ids.
  Times rounded to 1 ms.
- Writes are atomic (`edits.json.tmp` then `os.replace`) and guarded by `revision`: `PUT` carries the revision it edited; a mismatch
  is `409` with the stored document. No edits file = the detector clips.
- When a new run replaces `segments.json` in a folder that has `edits.json`, the run job first renames it to
  `edits.prev-<YYYYmmddTHHMMSSZ>.json` (at most 5 kept). When the UI loads a run whose `edits.json` has a different
  `base.segments_sha256` (a CLI re-run, say), the edits are still shown, with a banner "These edits were made on an earlier run".

**Edit model, in the browser** (`web/src/model/edits.ts`, pure functions, vitest-tested): `fromDetector(clips)`, `trim(state, id, edge,
t, bounds)`, `split(state, t, id?)`, `mergeNext(state, id)`, `mergeIds(state, ids)`, `remove(state, ids)`, `add(state, t, bounds)`,
`applyResegment(state, intervals)`, `validate(clips, bounds)`, `diffSummary(clips, detectorClips)`, and `History` (`push`, `undo`, `redo`,
cap 200). Each op returns `{state, changed: boolean, reason?: string}`; the UI shows `reason` as a toast ("Too short to split here").
Trim clamps to `[prev.end, next.start]`, the bounds and the 0.2 s minimum; merge keeps the first id and sets origin `merge`.

### 2.7 Export from the edits (part B2, `serve_export.export_edits`)

1. Read `segments.json` (`doc`) and the current clip list (edits or detector clips). Refuse when `doc` has `error` (failed run).
2. `edited_dir = run_dir / "edited"` (or the folder chosen in the dialog, checked by `serve_paths.check_write`). Clear what an
   earlier export left there, and nothing else: candidates come from a directory scan of `edited_dir`, a file is removed only when it
   is a regular file (lstat), its name matches the `cut.clip_name` / sidecar / reel / edit-list / LosslessCut patterns for this
   stem, and `path.resolve().parent == edited_dir.resolve()`. Names from `segments.json` are never used to delete.
3. Build `edited_doc = deepcopy(doc)` with `clips = [{index, start, end, file: cut.clip_name(stem, index, start, end, suffix),
   peak_score (max score of samples in [start, end]), expected_copy_start: cut.snap_to_keyframe(start, keyframes)}]` where keyframes
   come from `probe.probe(video)`; markers of the original clips that fall inside the new ranges are kept; `edited: {"from":
   "../segments.json", "revision": N, "segments_sha256": ...}`; warnings start from the run's.
4. If clips or a reel are wanted: `cut.cut_clip(video, edited_dir / file, start, end, factor, slowmo_audio)` per clip, with progress
   and the same slow-motion bookkeeping as `run_cut` (`factor = doc["slowmo"]["factor"]` when present; `cut.retimed_facts`). A clip that
   fails gets `error` and the export goes on (as `run_cut`).
5. `cut.write_losslesscut_csv(edited_dir / f"{stem}_losslesscut.csv", ...)` and `sidecar.write_sidecars(edited_dir, edited_doc,
   times=times, scores=scores, boxes=boxes, track_ids=track_ids, presence=[[c["start"], c["end"]] for c in edited clips],
   target=doc["settings"]["target"], embedder=doc["settings"].get("target_info", {}).get("embedder"), detector=dict(mode=...,
   backend=..., model=..., tracker=...))` (only when clip files were cut). The arrays come from ONE helper,
   `serve_edits.load_samples_arrays(run_dir) -> (times, scores, boxes, track_ids)`, the inverse of `pipeline.write_samples` (boxes
   `None` where the csv has none, `track_ids` `None` when the column is absent); route 38 serialises the same tuple. **Presence for
   edited clips is the clip ranges themselves** (a user clip is by definition where the target is); `peak_score` and
   `expected_copy_start` are computed by `serve_edits.edited_clip_facts(start, end, times, scores, keyframes)`, one function with its
   own test. The part's test runs `uv run python tools/validate_sidecars.py <run>/edited` and needs its last line `PASS`.
6. If a reel or edit lists are wanted: `export.export_run(edited_doc, edited_dir, edited_dir, reel, formats, music, beat_sync)` with
   `music = music.resolve(track_id or file, allow_by)`.
7. Write `edited_dir / "segments.json"` (the edited doc with `reel`/`edit_lists` filled in) and `edited_dir / "export.json"` (as the CLI
   does). So `inframe export <run>/edited --reel --music ID` also works on the edited result.

### 2.8 Pick, rank and targets on the server

- `ServePicker(pick.Picker)` (part A3, `serve_pick.py`) adds `rank(self, announce, progress=None)`: the body of `Picker.rank` with
  `progress` passed to `score_detect` (so the rank job reports and cancels) and `reset()` (clear taps and zone). One ServePicker per
  `(media_id, cls, target_key)` where `target_key` is `""` for plain taps, `photos:<photos_id>` or `saved:<name>`: `Picker` fixes
  class, photos, embedder and preset model at construction, so choosing another class, photos folder or saved target creates (or
  reuses) another ServePicker; nothing is mutated in place. Each writes its own file, `out = <stem>_inframe/targets/target-<cls>[-
  <target_key slug>].json`, so `save()` (which unlinks its file when the taps are empty) never touches another picker's taps. At Run
  time the active picker's file is copied (atomic replace) to `<stem>_inframe/target.json`, the `inframe pick` default, and the run
  uses `tap:<that copy>`; `inframe cut --target tap:` therefore sees the last target run from the UI. `warm()` in a thread on first
  use. Units stay `pick.pick_units()` (CPU_AND_GPU).
- Photos runs use `identity.parse_targets([f"photos:{dir}"], embedder, offline, identity.lazy_detect(model))`; saved runs `["saved:NAME"]`;
  tap runs `[f"tap:{target_json}"]` with `tap.resolve_preset` deciding the preset; class runs `["class:NAME"]`; any person `None`.
  `models.set_offline(app.offline)` once at start. The default embedder is `identity.DEFAULT_EMBEDDER`.

### 2.9 Opening files when browsers hide paths

The server only opens what the user pointed at: the CLI argument, a native dialog result, a path typed or picked in the file browser,
a PHPicker export, or a `locate` match for a dropped file. Native dialogs: macOS `osascript -e 'POSIX path of (choose file with prompt
"Open a video" of type {"public.movie"})'` (and `choose folder`, and `{"public.audio"}` for music), run with a 10-minute timeout; exit
code 1 with "User canceled" is `{"cancelled": true}`. The cross-platform twin is the built-in file browser (every Mac-only path has it).
No upload copy in T9a: a 1.2 GB upload over loopback would double disk use and buy nothing when the server can read the file in place.
T9b (browser-only) will use `<input type=file>`/File System Access with OPFS instead.

### 2.10 The seam for T9b: `Engine`

`web/src/engine/types.ts` (types mirror the API JSON of section 3 exactly):

```ts
export interface Engine {
  readonly kind: 'local' | 'browser';
  readonly caps: { openDialog: boolean; photosPicker: boolean; fileBrowser: boolean; locate: boolean; reveal: boolean;
                   music: boolean; beatSync: boolean; savedTargets: boolean; people: boolean; proxy: boolean;
                   output: 'server-folder' | 'download' | 'granted-folder'; streamCopyCut: boolean; chooseOutputFolder: boolean };
  session(): Promise<Session>;                       // Session.preflight: {ffmpeg, models: {fast, balanced, accurate}, identity, tracking, music}
  // media
  openPath(path: string): Promise<MediaInfo>;
  openDialog(): Promise<MediaInfo | null>;           // null = cancelled
  openFromPhotos(): Promise<MediaInfo | null>;
  locate(f: { name: string; size: number; lastModified: number }): Promise<MediaInfo | null>;
  openFile?(file: File): Promise<MediaInfo>;          // T9b only
  listDir(dir?: string, mode?: 'video' | 'folder' | 'audio'): Promise<DirListing>;
  recent(): Promise<RecentEntry[]>;
  media(id: string): Promise<MediaInfo>;
  videoUrl(id: string, opts?: { asMp4?: boolean }): string;   // T9b: blob URL
  makeProxy(id: string): Promise<MediaTask>;
  mediaTask(id: string, kind: 'proxy' | 'filmstrip'): Promise<MediaTask>;
  filmstrip(id: string): Promise<Filmstrip>;
  sheetUrl(id: string, n: number): string;
  frameUrl(id: string, t: number, width?: number): string;
  boxes(id: string, t: number, cls: string, signal?: AbortSignal): Promise<FrameBoxes>;
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
  confirmCandidate(mediaId: string, i: number): Promise<PickState>;
  candidateUrl(mediaId: string, i: number): string;
  startExport(runId: string, req: ExportRequest): Promise<JobRef>;
  startTargetAdd(name: string, req: TargetAddRequest): Promise<JobRef>;
  job(id: string): Promise<Job>;
  currentJob(): Promise<Job | null>;
  cancel(id: string): Promise<Job>;
  // runs and edits
  runs(mediaId: string): Promise<RunSummary[]>;
  run(runId: string): Promise<RunData>;
  samples(runId: string): Promise<Samples>;
  saveEdits(runId: string, revision: number, clips: EditClip[]): Promise<EditsDoc>;   // throws EditConflict on 409
  revertEdits(runId: string): Promise<void>;
  resegment(runId: string, p: ResegmentParams): Promise<ResegmentResult>;
  exportResult(runId: string): Promise<ExportResult | null>;          // ExportResult.outputs: ExportOutput[]
  runFileUrl(runId: string, name: string): string;
  reveal(runId: string, name?: string): Promise<void>;
  // music
  musicSearch(q: MusicQuery): Promise<MusicTrack[]>;
  musicGet(id: string, allowBy: boolean): Promise<MusicFile>;
  musicUrl(id: string): string;
  quit(): Promise<void>;
}
```

- **Output** is where local and browser differ most. `startExport` returns a job whose result lists `ExportOutput {name, kind:
  'clip'|'reel'|'sidecar'|'editlist'|'csv'|'credits', size, url}`. Local (`output: 'server-folder'`): files are written by the server
  into the run's `edited/` or the chosen folder; `url` plays them; Reveal and Choose folder exist. Browser (T9b, `output: 'download'`
  or `'granted-folder'` when File System Access is available): each output is a Blob URL the UI offers as a download or writes into
  the granted folder; music comes from a file the user picks (no server cache, no Play-before-download). Omitted in browser mode:
  Reveal, native dialogs, Photos picker, locate, beat sync (librosa), the server-side proxy, and stream-copy cutting until T9b proves a
  WebCodecs/Mediabunny remux (`streamCopyCut: false` hides the Clips section and keeps Edit lists).
- `engine/local.ts`: `LocalEngine implements Engine` over `fetch` (session token in `X-Inframe-Token` on every API call; media URLs
  carry the read-only media key `?m=`, section 4). Errors become
  `EngineError {status, message, extra}`.
- `engine/mock.ts`: an in-memory engine with a synthetic 60 s video description and generated samples; used by vitest and by
  `npm run dev -- --mode mock` so the UI can be built without the Python server.
- `engine/browser.ts`: a stub class whose methods throw `EngineError(501, "browser mode is T9b")`; T9b fills it with WebCodecs,
  mp4box.js and onnxruntime-web, plus a TypeScript port of `segments.py` for `resegment`.
- Selection (`engine/index.ts`): if `GET ./api/session` answers with `engine: "local"`, use LocalEngine; otherwise BrowserEngine
  (Cloudflare Pages has no `/api`). UI code imports only `Engine` and the pure modules (`model/*`).

## 3. API

Conventions: JSON bodies (`Content-Type: application/json`), times in file-time seconds (3 decimals), boxes `[x0, y0, x1, y1]` and
zones `[x, y, w, h]` as fractions of the upright frame. Every `/api/*` request needs the session token in the `X-Inframe-Token`
header, except the GET/HEAD media routes (`Route.media=True`: rows 12, 15, 17, 18, 20, 27, 30, 33, 44, 51) used by `<video>`, `<img>`,
`<audio>`, which take the read-only media key `?m=` instead (section 4); `?k=` no longer exists. Errors are `{"error": "<message>", ...}` with 400 (bad input), 403 (forbidden), 404
(unknown id or file), 409 (busy, superseded, revision conflict), 413 (body over 1 MB), 415 (not a video), 422 (valid JSON, refused by
the domain code, e.g. a `ValueError` from `run_cut`'s checks), 500 (`"<Type>: <message>"`), 501 (not on this platform). The Part
column names the part (A, B or C) that owns the route.

### 3.1 Session, static, files

| # | Method, path | Request | Response | Calls | Part |
|---|---|---|---|---|---|
| 1 | GET `/`, `/index.html`, `/assets/*`, `/favicon.svg` | | static files from `dist`, or the built-in "UI not built" page (no token: they hold no user data; Host and Sec-Fetch checks still apply; names starting with `_` or `.` are 404) | `importlib.resources` | A1 |
| 1a | POST `/api/session/claim` | `{"nonce": "..."}` (no token) | `{"token", "media_key"}` once; 403 for a used, unknown or expired (30 s) nonce | `App.claim` | A1 |
| 2 | GET `/api/session` | | `{"version": "0.1.0", "engine": "local", "platform": "darwin", "caps": {...as Engine.caps}, "presets": ["fast","balanced","accurate"], "default_preset": "balanced", "classes": ["person","dog",...], "defaults": <pipeline.Settings() as dict>, "job": <current job or null>, "media": [<MediaInfo of opened files>], "start_media": "<id or null>"}` | `inframe.__version__`, `pipeline.PRESETS`, `pipeline.Settings`, `classes.COCO_91` (names, sorted), `importlib.util.find_spec("librosa")` for beatSync; `preflight` (1.5): `shutil.which("ffmpeg"/"ffprobe")`, model files by `models` lookup (no load), `find_spec` of the identity, tracking and music extras | A1 |
| 3 | POST `/api/quit` | `{}` | `{"ok": true}` then shutdown | `App.stop` | A1 |

### 3.2 Media (part A2; `App.open_media` behind row 4 is A1)

`MediaInfo`: `{"id","path","name","size","mtime","codec","profile","pix_fmt","width","height","display_w","display_h","rotation","mirrored","fps","capture_fps","slowmo_suggested": bool (capture_fps >= cut.AUTO_MIN_CAPTURE_FPS),"duration","offset","end_time","has_audio","keyframes":[...],"video_url","proxy": {"state": "none|running|done|failed","fraction","url","first_frame_file_time","error"},"filmstrip_state": "...","picked_copy": bool}`.

| # | Method, path | Request | Response | Calls |
|---|---|---|---|---|
| 4 | POST `/api/media/open` | `{"path": "/abs/file.mov"}` | `MediaInfo` (starts the filmstrip task, adds to recent) | `App.open_media` (path rules, extension and magic check, `probe.probe`, `probe.capture_fps`); errors 404/415/403 |
| 5 | POST `/api/media/open-dialog` | `{"kind": "video"}` | `MediaInfo` or `{"cancelled": true}`; 501 off macOS | `subprocess.run(["osascript", ...])`, then as 4 |
| 6 | POST `/api/media/photos-pick` | `{}` | `MediaInfo` (of the exported copy) or `{"cancelled": true}`; 501 off macOS | `picker.pick(export_dir, limit=1)`, `picker.new_export_dir` |
| 7 | POST `/api/media/locate` | `{"name","size","last_modified_ms"}` | `MediaInfo`, or 404 `{"error","matches": n}` | `os.scandir` over recent folders and `~/Movies ~/Downloads ~/Desktop ~/Documents` (depth 2), `os.stat` |
| 8 | GET `/api/fs/list?dir=&mode=video\|folder\|audio` | | `{"dir","parent","roots":[...],"entries":[{"name","kind":"dir\|video\|audio","size","mtime"}]}` sorted dirs first; no hidden entries, no `~/Library` | `os.scandir`, `Path.resolve` + `is_relative_to(root)` |
| 9 | GET `/api/recent` | | `{"recent":[{"path","name","opened"}]}` (missing files dropped) | serve state `recent.json` |
| 10 | DELETE `/api/recent` | | `{"recent": []}` | |
| 11 | GET `/api/media/{id}` | | `MediaInfo` | |
| 12 | GET/HEAD `/api/media/{id}/video?m=&as=mp4` | Range | 200/206 bytes of the original | `file_response` |
| 13 | POST `/api/media/{id}/proxy` | `{}` | `{"state","fraction"}` (idempotent; 409 if another proxy runs) | `subprocess.Popen(ffmpeg ...)` in `serve_media.make_proxy` |
| 14 | GET `/api/media/{id}/proxy` | | `{"state","fraction","url","first_frame_file_time","error"}` | `proxy.json` |
| 15 | GET/HEAD `/api/media/{id}/proxy.mp4?m=` | Range | bytes | `file_response` |
| 16 | GET `/api/media/{id}/filmstrip` | | `filmstrip.json` contents plus `"state"` | `serve_media.make_filmstrip` (uses `decode.iter_samples(fps=None)`) |
| 17 | GET `/api/media/{id}/filmstrip/{n}.jpg?m=` | | JPEG sheet | file |
| 18 | GET `/api/media/{id}/frame.jpg?t=&w=&m=` | | JPEG of the frame at or after t, long side `w` (default 960, max 1280) | `pick.grab_frame` (LRU of 32 encoded JPEGs) |

### 3.3 Pick and targets

`PickState`: `{"target_key": "" | "photos:p1" | "saved:alice", "taps":[{"time","box","track_id","index"}],"zone":[x,y,w,h]|null,"file":"/abs/target.json"|null,"class":"person","candidates":[{"rank","t","box","track_id","score"}]}`.

| # | Method, path | Request | Response | Calls | Part |
|---|---|---|---|---|---|
| 19 | GET `/api/media/{id}/boxes?t=&cls=person` | | `{"t": 12.345, "boxes":[{"i":1,"box":[...],"score":0.93}]}` (score >= 0.4, as pick.py); 409 `{"error":"superseded"}` | `ServePicker.frame` | A3 |
| 20 | GET `/api/media/{id}/pickframe.jpg?t=&cls=&m=` | | JPEG of exactly the frame `boxes` returned (t as returned) | `ServePicker.frame_image` | A3 |
| 21 | GET `/api/media/{id}/pick?cls=` | | `PickState` | `ServePicker` fields | A3 |
| 22 | POST `/api/media/{id}/pick/tap` | `{"cls","t","box","index"}` | `PickState` | `Picker.add_tap` (writes target.json) | A3 |
| 23 | POST `/api/media/{id}/pick/undo` | `{"cls"}` | `PickState` | `taps.pop()`, `Picker.save` | A3 |
| 24 | POST `/api/media/{id}/pick/zone` | `{"cls","zone": [x,y,w,h] \| null}` | `PickState` | `Picker.set_zone` (validates via `zone.Zone.of`) | A3 |
| 25 | POST `/api/media/{id}/pick/reset` | `{"cls"}` | `PickState` | `ServePicker.reset` | A3 |
| 26 | GET `/api/targets` | | `{"targets":[<Saved.summary()> or {"name","error"}]}` | `targets.list_targets` | A3 |
| 27 | GET `/api/targets/{name}/crops/{i}.jpg?m=` | | JPEG crop | `targets.Saved(name).crop_paths()[i]` (name checked by `targets.check_name`) | A3 |
| 28 | POST `/api/photos/choose` | `{"source":"dialog"\|"photos"\|"path","path"?}` | `{"id","dir","n_photos"}` or `{"cancelled":true}` | `osascript choose folder` / `picker.pick(tempfile.mkdtemp(prefix="inframe-picked-"), kind="images")` (A0) / path through `serve_paths.check_read`; `identity.photo_files` | A3 |
| 29 | POST `/api/media/{id}/rank` | `{"photos_id"}` or `{"saved":"NAME"}`, `"cls"` | `{"job": Job}` (result: `{"candidates":[...]}`) | `ServePicker(photos=...).rank(progress=...)` | A3 |
| 30 | GET `/api/media/{id}/cands/{i}.jpg?m=` | | JPEG crop | `Picker.thumb(i)` | A3 |
| 31 | POST `/api/media/{id}/pick/confirm` | `{"cls","i"}` | `PickState` | `Picker.confirm(i)` | A3 |
| 32 | POST `/api/media/{id}/people` | `{}` | `{"job": Job}` (result: people.json doc with `"crop_url"` per person) | `people.scan_people(video, out, progress)` | A3 |
| 33 | GET `/api/media/{id}/people/{n}.jpg?m=` | | JPEG | the n-th person's `person_NNN_<scan id>.jpg` in `<stem>_inframe/people/` (named in people.json; the client never sends a file name) | A3 |
| 34 | POST `/api/targets/{name}/add` | `{"from":"run","run_id"}` or `{"from":"photos","photos_id"}` | `{"job": Job}` (result: `add_crops` dict) | `targets.check_name`, `targets.crops_from_run` / `targets.crops_from("photos:DIR")`, `targets.add_crops` | B1 |

### 3.4 Runs, jobs, edits, export, music (rows 35-41 part B1, 42-45 B2, 46-48 A1, 49-52 B3)

`Job`: `{"id","kind":"run|rank|people|export|target-add","state":"running|done|failed|cancelled|needs_confirm","fraction","message","started","elapsed_s","media_id","run_id","result","error"}`.

`RunRequest`:
```json
{"media_id": "a1b2c3d4e5f6",
 "target": {"kind": "any|class|tap|saved|photos|motion", "class": "person", "name": "alice", "photos_id": "p1"},
 "zone": [0.1, 0.2, 0.5, 0.6], "zone_overlap": null,
 "preset": "balanced", "slowmo": "off|auto|4", "slowmo_audio": false,
 "advanced": {"pad": 1.0, "min_len": 1.0, "merge_gap": 1.5, "enter": null, "exit": null, "min_box": 0.0,
              "tracker": "ocsort", "activity": "any", "fps": null}}
```

| # | Method, path | Request | Response | Calls |
|---|---|---|---|---|
| 35 | POST `/api/runs` | `RunRequest` | `{"job": Job, "run_id"}`; 409 busy; 422 for bad settings | `identity.parse_targets` / `tap.resolve_preset` (as `cli.check_target`), `pipeline.Settings`, `pipeline.check_ranges`, `pipeline.run_cut(video, out, settings, target, progress)`; `identity.Ambiguous` -> state `needs_confirm` |
| 36 | GET `/api/media/{id}/runs` | | `{"runs":[{"run_id","dir","slug","target","preset","created","n_clips","error","has_edits","edits_base_ok"}]}` newest first | `json.load(segments.json)` of `<stem>_inframe/*/` whose `source.path` resolves to the media path |
| 37 | GET `/api/runs/{rid}` | | `{"run_id","media_id","dir","segments":{"settings","warnings","scores","timing_s","presence_intervals","clips","slowmo","error"},"edits": <edits.json or null>,"edits_base_ok": bool,"export": <export.json or null>}` | `serve_edits.load_edits`, `segments_sha256` |
| 38 | GET `/api/runs/{rid}/samples` | | `{"t":[...],"score":[...],"box":[[...]\|null],"track":[...]\|null,"enter","exit","merge_gap","min_len","pad","sound","mode"}` (the last six from `segments.json` `settings`) | `serve_edits.load_samples_arrays` (csv) |
| 39 | PUT `/api/runs/{rid}/edits` | `{"revision": 6, "clips": [EditClip]}` | `edits.json` doc (revision 7); 409 `{"error":"conflict","edits":<stored>}`; 422 rule broken | `serve_edits.validate_clips`, `serve_edits.save_edits` |
| 40 | DELETE `/api/runs/{rid}/edits` | | `{"ok": true}` (file renamed to `edits.prev-...json`) | `serve_edits.revert_edits` |
| 41 | POST `/api/runs/{rid}/resegment` | `{"enter","exit","merge_gap","min_len","pad"}`, each optional (missing = the run's `segments.json` `settings` value) | `{"presence":[[s,e]],"clips":[[s,e]],"same_as_run": bool}` (no write) | `segments.build_segments(times, scores, doc["source"]["end_time"], ...)` |
| 42 | POST `/api/runs/{rid}/export` | `{"clip_ids": null \| ["d3","u27"], "out_dir": null \| "/abs/folder", "replace": false, "clips": true, "reel": true, "music": {"track_id": "...", "file": null, "allow_by": false} \| null, "beat_sync": false, "edit_lists": ["fcpxml","edl"]}` | `{"job": Job}` (result: as export.json plus `"outputs"`); 409 `{"error": "exists", "export": <export.json>}` when the folder holds an export and `replace` is false | `serve_export.export_edits` -> `cut.cut_clip`, `sidecar.write_sidecars`, `cut.write_losslesscut_csv`, `export.export_run`, `music.resolve`, `editlist.parse_formats` |
| 43 | GET `/api/runs/{rid}/export` | | `export.json` of `edited/` or 404 | file |
| 44 | GET/HEAD `/api/runs/{rid}/files/{name}?m=` | Range | a clip, reel or credits file from the run folder or the export folder; `name == os.path.basename(name)`, no leading `.` or `-`, no `/` or NUL, at most 255 bytes, and present in a server-built list from `os.scandir` of that folder (regular files only, by lstat) whose names match the clip/reel/credits patterns; the JSON is not trusted for this | `file_response` |
| 45 | POST `/api/runs/{rid}/reveal` | `{"name": null}` | `{"ok": true}` | same name rule as 44; the argument is always an absolute path: macOS `open -R -- /abs/path`, Linux `xdg-open /abs/dir`, Windows `explorer /select,C:\abs\path` |
| 46 | GET `/api/jobs/current` | | `{"job": Job \| null}` | `JobRunner` |
| 47 | GET `/api/jobs/{jid}` | | `Job` | `JobRunner` |
| 48 | POST `/api/jobs/{jid}/cancel` | `{}` | `Job` (state stays `running` until the next progress report, then `cancelled`) | `JobRunner.cancel` |
| 49 | GET `/api/music/search?mood=&min_s=&limit=&allow_by=0` | (`mood`: the dialog's mood field only, at most 40 chars of `[\w -]`; the UI never fills it from file or target names) | `{"tracks":[Track.to_json()],"notice": OPENVERSE_NOTICE}`; 502 on MusicError | `music.search` |
| 50 | POST `/api/music/get` | `{"id","allow_by"}` | `{"id","file","license","attribution","duration_s","needs_credit"}` | `music.get`, `music.load_meta` |
| 51 | GET/HEAD `/api/music/{id}/audio?m=` | Range | the cached audio file | `music.cache_dir()` lookup of `<id>.*` (id checked by `music.ID_RE`) |
| 52 | POST `/api/music/choose-file` | `{}` | `{"file","license": null}` or cancelled | `osascript choose file of type {"public.audio"}`; `music.load_meta` |

**Path safety (all parts; rules in `serve_paths.py`, part A1).** The server takes a path only in rows 4, 8, 28 (`source: "path"`),
42 (`out_dir`), the dialog results and the CLI argument; everything else refers to ids.
- `check_read(path, kind)`: `resolve(strict=True)`; refuse a path with a component starting with `.`; refuse anything inside
  `~/Library` except the inframe cache and the picker temp folders; `os.stat` must show a regular file (or a folder for `kind="dir"`).
- `check_video(path)`: `check_read`, then an extension allowlist (`.mp4 .m4v .mov .mkv .webm .ogv`) and a magic check on the first
  64 bytes: ISO-BMFF `ftyp` (or `moov`/`mdat`/`wide`/`free` box) at offset 4 for mp4/m4v/mov, EBML `1A 45 DF A3` for mkv/webm,
  `OggS` for ogv. A text file (HLS, ffconcat, SDP) never reaches ffprobe, ffmpeg or PyAV. `check_audio(path)` does the same for
  `.mp3 .m4a .aac .wav .flac .ogg .opus` (ID3/MPEG sync, `ftyp`, `RIFF....WAVE`, `fLaC`, `OggS`).
- Every ffmpeg/ffprobe command the `serve_*` modules build themselves (proxy, music check) passes `-protocol_whitelist file,pipe`
  and, for its own output, `-f mp4`. The core modules (`probe`, `decode`, `cut`) are not changed in T9a; the magic gate is what
  protects them.
- Reads of outputs go only through registered run folders (`app.runs`), their export folder, the media cache, the music cache and the
  targets store; the final `resolve()` must be `is_relative_to` that folder and lstat must show a regular file.
- `check_write(folder)`: writes go only to run folders (created after the 1.5 confirmation), a chosen export folder (not hidden, not
  under `~/Library`, `os.access(W_OK)`), the media cache and the serve state folder. Picker temp folders are `tempfile.mkdtemp`
  (0700); at start the server removes `inframe-picked-*` folders in the temp dir older than 24 h that no running server owns (a
  `owner.pid` file inside, checked with `os.kill(pid, 0)`).

## 4. Security and privacy

- **Loopback only.** Bind `127.0.0.1` (IPv4); `--host` does not exist. Port 0 (free port) by default, `--port N` to choose.
- **Secrets.** Per server start: a session token and a media key, each `secrets.token_urlsafe(16)` (128 bits), compared with
  `hmac.compare_digest`. The session token is needed, in the `X-Inframe-Token` header only, on every `/api/*` route except 1a and the
  media routes. The media key (`?m=`) is accepted only on GET/HEAD routes marked `media=True` (3, conventions): it can read a picture,
  a video or an audio file of something already opened, nothing else (no JSON, no job, no write). A custom header also forces a CORS
  preflight for any cross-origin caller, which the server never approves.
- **Hand-over without a token in a URL.** The server makes a one-use **nonce** (`token_urlsafe(16)`, expires 30 s after it is made,
  void after one claim) and opens `http://127.0.0.1:PORT/#n=NONCE`. A fragment is never sent to a server and never in a Referer. The
  page reads it, calls `POST /api/session/claim {"nonce"}` (row 1a), keeps the token and media key in `sessionStorage` and removes the
  fragment with `history.replaceState`. The terminal prints the same link and "Press Enter for a new link" (a stdin thread makes a
  fresh nonce and prints it; with no tty, `inframe serve` prints that a new link needs a restart). A reload keeps working from
  `sessionStorage`; a new tab needs a new link. What another program of the same user can see (the argv of `open`/`xdg-open` on
  Linux, the browser history, the terminal) is at most a spent or expiring nonce, never the session token. The earlier claim "`ps`
  shows no token" was wrong for the URL handed to the browser launcher; it is now true because that URL holds only the nonce.
- **Host check (DNS rebinding).** `Host` must be exactly `127.0.0.1:PORT`, on every request including static files. `localhost` is
  not accepted (the server never prints it). A rebinding name never matches.
- **Origin check.** If a request has an `Origin` header it must be exactly `http://127.0.0.1:PORT`; POST, PUT and DELETE must have one.
- **Fetch metadata.** If `Sec-Fetch-Site` is present it must be `same-origin` or `none`, else 403. When it is absent (Safari before
  16.4, curl, other clients) nothing is relaxed: API routes still need the header token, media routes the media key. So what stands
  behind the secrets alone is: GET media routes (media key), every API route for a client that sends no fetch metadata (session
  token). Static files need neither and hold no user data.
- **Response headers on everything:** `Cross-Origin-Resource-Policy: same-origin`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, `Cache-Control: no-store` on `/api/*`. No `Access-Control-Allow-*` header,
  ever; `OPTIONS` answers 403.
- **CSP on the HTML:** `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src
  'self' blob:; connect-src 'self'; worker-src 'self' blob:; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action
  'none'`. `connect-src 'self'` means the page cannot send anything to any other host even by mistake. T9b adds
  `'wasm-unsafe-eval'` and ONE pinned model host to `connect-src`, and checks a SHA-256 of each weight file before use.
- **Request bodies** (`serve_http.read_body`, before any read): `Transfer-Encoding` present -> 400 (no chunked bodies);
  `Content-Length` must match `^[0-9]{1,8}$` (else 400; missing on POST/PUT -> 411); above 1 MB -> 413 without reading; then exactly
  that many bytes are read. Handler sockets get `timeout = 15` s (a slow body ends with 408 and a closed connection); at most 32
  handler threads at once (a `BoundedSemaphore` in `process_request`; over -> 503 and close). Clip lists over 2000 are 422.
- **Outbound network.** The page talks only to the loopback server. The server's own outbound requests go through ONE module,
  `net.py` (part B3): `net.fetch(url, *, kind, max_bytes)` allows `https` only; `kind="api"` allows only `api.openverse.org`;
  `kind="audio"` (the track URL that `music.get` takes from the per-track Openverse record) allows any https host whose every
  `getaddrinfo` address is global (`ipaddress.ip_address(a).is_global`: no loopback, private, link-local), refuses a redirect to
  another host or to http, stops at 50 MB, and needs a `Content-Type` of `audio/*` or `application/octet-stream`; the bytes must then
  pass `serve_paths.check_audio`'s magic check before ffprobe (with `-protocol_whitelist file,pipe`) looks at them. `music._get` calls
  `net.fetch`. The music query is built only from the mood field (row 49), never from file, folder or target names. Jamendo stays an
  Openverse source: the licence is read per track from Openverse's record (`parse_track`, CC0 unless `allow_by`) and written into the
  track's meta and the reel's credits file. Model weights: the embedder may download them from Hugging Face on first use through the
  core `models`/embedder code (unchanged in T9a; `--offline` stops it). A pytest (`tests/test_serve_offline.py`, part B2) patches
  `socket.socket.connect` and `socket.create_connection` and fails on any non-loopback address during open, a motion run, edits and
  an export.
- **Licences.** No new Python dependency (stdlib server). npm: build-time packages are listed in README with their licence; bundled
  ones are checked and their notices shipped by `rollup-plugin-license` (2.2). ffmpeg is the user's own binary started as a
  subprocess; `libx264`, when the proxy uses it, is reached only through that external binary, never linked. GPL editors (Kdenlive,
  LosslessCut) only read the edit-list files we write; `inframe serve` never launches them. The licence of the FFmpeg build inside
  the PyAV wheel is a project-level question that predates T9a (section 6, Q13).
- **Process:** jobs run in threads of the server process; ffmpeg and osascript are started with argument lists (no shell). Names from
  requests are checked against the existing regexes (`targets.NAME_RE`, `music.ID_RE`) and the file-name rule of row 44.
- **Not handled (accepted for local use):** another program run by the same macOS user can read the session token from the
  browser's `sessionStorage` on disk or attach a debugger; that user can already read the files. A web page can learn that some
  server answers on a loopback port (the static page needs no secret) but cannot read it (no CORS, CORP same-origin). The gap between
  the `getaddrinfo` check and the connect in `net.fetch` (DNS rebinding of a music host) is narrow and only reaches a download capped
  at 50 MB that must still sniff as audio.

## 5. Build split

The work was built in parts (A: server core, media and picking; B: runs, export and music; C: the web UI). The Part column in section 3 names the part that owns each route.

## 6. Open questions for the maintainer (each with the default the build uses)

1. **Where UI runs write.** Default: `<video folder>/<stem>_inframe/<target>/` (e.g. `clip_inframe/saved-alice/`), exports in
   `.../edited/`; Photos-picked copies write to `~/Movies/inframe/`. Alternative: always `~/Movies/inframe/`.
2. **Port and bookmarks.** Default: a new port and token each start, the browser opens by itself with a one-use link (a new tab needs
   a new link: press Enter in the terminal). Alternative: a fixed port and a saved token so a bookmark works (weaker: the token then
   lives on disk and in history).
3. **Photos of the target from Apple Photos.** Default: yes, the picker helper gets an images mode (copies go to a temp folder, deleted
   when the server stops).
4. **Several people.** Default: one target per run; a video can have several runs (one per person), a run switcher, and thin lanes
   for up to 3 other runs under the active one (1.5a). Each run exports on its own. Later: one reel from clips of several runs.
5. **Photo mode cannot decide.** Default: the run says so and offers "Find who it is", which reads the video once more (about the run's
   time). Alternative: always rank first (slower when not needed).
6. **Preview copy.** Default: made only when the browser cannot play the file (Theora, some VP9 or HEVC cases), 720p H.264, the 3 newest
   kept in `~/Library/Caches/inframe/serve`. Alternative: always make one for very large files.
7. **Clip starts.** Default: stream copy as today (a clip can start up to one GOP early; the UI shows where). Frame-exact smart cut is
   not in T9a.
8. **Dropped files.** Default: look up the dropped file by name, size and time in Movies, Downloads, Desktop, Documents and recent
   folders (no copy). Alternative: no drop support, Open button only.
9. **Music.** Default: CC0 only; Play downloads the track into the music cache first. CC BY only when you tick the box (a credits file is
   written).
10. **Recent files.** Default: the last 20 paths are remembered on this Mac, with a Clear button. Alternative: remember nothing.
11. **Re-run over edits.** Default: earlier edits are kept aside (`edits.prev-*.json`) and you can restore them on the new run.
12. **Detector units.** Settled outside T9a: CPU_AND_GPU with the fp32-topk package is the default for runs; tapping already used the GPU. The UI's pre-run estimate uses those GPU timings.
13. **FFmpeg inside PyAV.** PyAV (BSD) is already a core dependency; the FFmpeg its wheel bundles may be LGPL or GPL-built. Default: not
    changed by T9a; a later task logs `av.library_versions` and the build configuration and decides. [unverified which build the
    wheel uses]
14. **Try on a short range first.** `run_cut` has no start/end option, so the UI cannot run on a part of the video yet. Default: not in
    T9a (the estimate before Run is shown instead). Alternative: add `Settings.start/end` to the pipeline as its own task.
15. **First run in a folder.** Default: ask once before making `<stem>_inframe/` next to the video, offer `~/Movies/inframe` instead.
