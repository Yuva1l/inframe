# Auto music ("Auto-pick + fit"): design

The feature: one option makes inframe pick a free song that suits
the reel and fit it: trim or loop, fades, -14 LUFS and cuts on the beat. "Another song" moves to the next song on a fixed ranked list.
The product uses no LLM and adds no new dependency. The only thing that leaves the machine is the Openverse search text (a mood word) and
the track downloads.

The design builds on the code in which CC0 and CC BY are allowed by default.

## 0. Facts checked against the code and the API

- `music.search()` returns `Track` objects without `tags`, `genres` or `mature`, and stops at `limit`. Auto music therefore reads the raw
  rows itself, through `music._get` and `music.parse_track`, and leaves `music.py` unchanged. `search` sends one query per license, in the
  order `licenses_allowed(cc0_only)` (`cc0` first, then `by`). The module docstring says Openverse has no CC0 tracks in its music category.
- `music.get(id, cc0_only)` checks the cache first. It globs `cache_dir()/<id>.*`, skips files ending in `.json`, and requires `<id>.json`
  beside the audio. Otherwise it fetches the per-track record from Openverse, checks the license with `parse_track`, downloads the file
  (at most 50 MB, sniffed, then `ffprobe`'d) and writes `<id>.<ext>` plus `<id>.json`. Any id that matches
  `ID_RE = ^[0-9a-fA-F-]{8,64}$` and has a cached CC0 `.json` is served from the cache with no network request.
- `beats.beat_times(path)` decodes the whole file at 22.05 kHz. It computes a low-band onset envelope (n_fft 512, hop 64, fmax 300 Hz) and
  runs `librosa.beat.beat_track(trim=True)` on it, returning `BeatGrid(tempo, times, duration)`. It does not return the envelope.
- `beats.plan_beat_cuts(available_frames, slack_frames, beats, fps)` is pure Python. It returns a `Plan` with `joins[i].on_beat` and with
  warnings `"clip N: no beat in its padding..."` and `"clip N: shorter than the gap between beats..."`. **Auto music reuses it to score
  how well a song fits.**
- `export.tail_slack(doc, clip)` gives each clip's padding in seconds (already multiplied by the slow-motion factor).
  `reel.shown_pictures(path)` gives the number of pictures in a clip file.
- `export.export_run(doc, clips_dir, out_dir, reel, formats, music=(path, meta), beat_sync)` writes the credits file when
  `needs_credit(meta.license)` and fills `reel.music.{file, license, credits, credit_text}`.
- CLI: `export` has `--music FILE|ID`, `--cc0-only`, `--allow-by` (ignored) and `--beat-sync`. `--music` needs `--reel`, and
  `--beat-sync` needs `--music`. **`cut` has `--reel` but no `--music`.** `run_cut` calls `export_run(doc, out, out, reel, formats)` and
  never passes music.
- **MCP (`mcp_server.py`) has no music tool and no export tool.** `cut_video` has `reel` and `edit_lists` only.
- Serve: `serve_music.py` has `GET /api/music/search`, `POST /api/music/get`, `GET /api/music/{id}/audio` (from the cache, the id must
  match ID_RE) and `POST /api/music/choose-file`. It validates mood with `MOOD_RE = [\w -]{0,40}` and uses a lock `_download`.
  `serve_export.plan_export` accepts `music: {track_id | file, cc0_only}`, and `job_fn` calls `resolve_music` before anything is
  cleared. The caps `music: True` and `beatSync: _has("librosa")` are in `serve.py`.
- UI: `MusicPicker.tsx` offers the radio "No music", a mood and minimum-length search with Play and Use, a "CC0 only" box, and "Use a music
  file...". `ExportDialog.tsx` sends `music: {track_id, file, cc0_only}` and `beat_sync`, and shows `reel.music.credit_text` after the
  export. The built UI is committed in `src/inframe/web/dist` (vite `outDir: '../src/inframe/web/dist'`), so it must be rebuilt.
- Tests stub Openverse with `monkeypatch.setattr(music_mod, "_get", fake)` and set `INFRAME_CACHE` to a temporary folder.
  `tools/synth_music.py` renders click and kick-and-hat tracks.
- **Openverse, checked live** with one call `?q=calm&source=jamendo&category=music&license=by&page_size=20&length=medium`:
  - HTTP 200, with headers `x-ratelimit-limit-anon_burst: 20/min` and `x-ratelimit-limit-anon_sustained: 200/day`.
  - 88 results in 5 pages.
  - `duration` is in milliseconds; this call returned 157 to 437 s.
  - `tags` hold mood, instrument and speed words, e.g. `speed_verylow`, `speed_low`, `instrumental`, `calm`, `piano`. `genres` is a list.
    There is no BPM field.
  - The research sweep reports that anonymous requests allow `page_size` up to 20 [unverified].
- **ffmpeg synthetic audio, checked locally** and run through `beat_times`:
  - The command is `aevalsrc='(0.9*sin(2*PI*80*t)*exp(-mod(t-0.3,P)*30)*gte(t,0.3)+0.3*sin(2*PI*1250*t))*gte(t,INTRO)'` (40 s, 44.1 kHz).
  - P = 0.5 gave a tempo of 120.2 and P = 0.667 gave 89.9.
  - INTRO = 4 put the first beat at 4.304 s.
  - The three analyses took 4.1 s in total, including the imports (not a benchmark).

## 1. Selection

### 1.1 The reel's shape (offline, before any request)

`automusic.ReelShape(durations: list[float], slack: list[float], fps: float)` holds one entry per clip, in reel order.

- **From clip files** (CLI `export`, MCP, and `cut --music`):
  - `fps = musicreel.video_params(first clip)["fps"]`
  - `durations[i] = shown_pictures(clip_i) / fps`
  - `slack[i] = tail_slack(doc, clip_i)`
- **From the document only** (serve, before the export job cuts the clips):
  - `durations[i] = (end - (expected_copy_start or start)) * factor`, where `factor` is the slow-motion factor (`doc.slowmo.factor`, or 1)
  - `slack[i] = tail_slack(doc, clip)`
  - `fps = media.info.fps`
  - For retimed slow-motion clips this frame-rate estimate is [unverified]. The score only ranks songs; the real plan is made later by
    `build_music_reel` from the files.
- **Derived values:**
  - `T = sum(durations)`
  - `n = len(durations)`
  - `rate = 60 * n / T` (clips per minute)
  - `speed`: `slow` if rate < 4, `medium` if 4 <= rate < 10, `fast` if rate >= 10
- **Per speed:**

| speed | default mood (query) | wanted Jamendo speed tag | tempo band (folded BPM) | built-in beat BPM choices |
|---|---|---|---|---|
| slow | `calm` | low (1) | 70-100 | 80, 90, 100 |
| medium | `upbeat` | medium (2) | 95-130 | 100, 112, 120, 128 |
| fast | `energetic` | high (3) | 120-160 | 120, 128, 140, 150 |

- Speed tag values: `speed_verylow` = 0, `low` = 1, `medium` = 2, `high` = 3, `veryhigh` = 4.
- Folded BPM: double while below 70, halve while 180 or more.

### 1.2 Candidates: the search

- **Query text:** the user's mood if given, else the default mood for the speed. It must match `MOOD_RE = [\w -]{0,40}` (the same rule as
  serve; an empty string means "from the reel"). Nothing else ever goes into `q`. The reel shape never leaves the machine.
- **Requests:** `GET https://api.openverse.org/v1/audio/?q=<mood>&category=music&source=jamendo&license=<lic>&page_size=20&page=<p>`.
  - Send one query per license in `music.licenses_allowed(cc0_only)` order (`cc0`, then `by`), pages 1 and 2.
  - Stop a license early when its `page_count` is reached or 30 rows have passed the filters.
  - **At most 4 search requests per query.** Do not send a `length=` filter: its buckets are not documented, and the live `medium` call
    returned 157-437 s. Filter on duration locally instead.
- **Fixed fallback query:** if fewer than 5 rows pass the filters, run the same search once for `q=instrumental` (at most 4 more requests)
  and append its new rows after the first query's rows.
- **Filters on each row, before any download:**
  - `music.parse_track(row, cc0_only)` is not None. This keeps only `cc0` or `by` with a `license_url`, and only `cc0` with `cc0_only`.
  - 30 s <= `duration_s` <= 600 s.
  - `row.get("mature")` is not true.
  - The id has not been seen already.
- **Search cache:** keep the kept rows (Track fields plus `tags` names, `genres`, the query and the page) in
  `cache_dir()/auto/search/<sha256(json [v, mood, cc0_only]) [:16]>.json` for 7 days. Every reel with the same mood reuses it at no cost.
- **When the search fails:** on a `MusicError` (network, 429, bad JSON), use a stale search cache if one exists and add a warning.
  Otherwise the candidate list is empty, which leads to the built-in beat (1.5).

### 1.3 Metadata score (no download), used to order the shortlist

For a row with duration `dur`, let `N = T + 3` (allowance for the song's first beat):

- **Length:** `L = 1.0` if `N <= dur <= 3N`, else `max(0.6, 1 - 0.1*(dur/N - 3))` when longer. When `dur < N` the song must loop:
  `L = 0.2 + 0.3*dur/N`.
- **Speed tag:** `S = max(0, 1 - 0.35*|tag - want|)`, where `tag` is the row's `speed_*` value. A row with no speed tag gets `S = 0.5`.
- **Mood:** `M = 1.0` if any word of the query is in the row's tags or genres (lower case), else `0.6`.
- **Vocals:** `I = 1.0` with tag `instrumental`, `0.6` with tag `vocal`, else `0.8`.
- **`meta = 0.45 L + 0.30 S + 0.15 M + 0.10 I`.**
- **Order:** `meta` descending, then CC0 before CC BY (no credit needed), then `id` ascending. The order is total and deterministic.

### 1.4 Audio analysis and the final score (only with librosa, i.e. when beat sync is possible)

Downloads go in fixed batches of the metadata order, so ranks never move once given:

- Batch 1 is meta positions 1-4, batch 2 is positions 5-6, batch 3 is positions 7-8.
- **At most 8 downloads per (reel shape, mood, cc0_only).** A batch is fetched only when the asked rank goes past the ranks already scored.
- Each batch is sorted by its final score (ties broken by the meta order) and appended to the ranked list.
- **Fetching:** each candidate is fetched with `music.get(id, cc0_only=cc0_only)`, which reuses the cache, the license check and the
  `.json` record, and costs one detail request plus the file.
- **Failures:** a candidate that fails is dropped from its batch and not replaced. This applies to `LicenseError`, a `MusicError` other
  than a network or 429 error, and a `NoBeatError`. Record the failure in its analysis file (`{"error": ...}`) so the drop is
  deterministic next time. Network and 429 errors are not recorded; they end the batch with a warning.

**Analysis.** Write it to `cache_dir()/auto/analysis/<id>.json` (`v: 1`) and run it again only if `analysed_s` is shorter than the reel
now needs.

- Decode `min(full length, T + 40 s)`, or the full song when `full length < T + 40`, capped at 600 s.
- Run the beat analysis below. Its result is `bpm`, `beats` (seconds), `duration_s` of the full file (ffprobe) and `analysed_s`.
- **Beat analysis:** factor `beats.beat_times` into `beats.analyse_signal(y) -> (BeatGrid, oenv)`. `beat_times` then calls it, with
  unchanged behaviour. Add `max_seconds` to `_load_mono` (an ffmpeg `-t`).
- **`pulse`:** the mean of `oenv` at the beat frames (`round(t * 22050 / 64)`) divided by `mean(oenv)`.
- **`intro_ratio`:** the mean RMS over the 10 s after the first beat divided by the median RMS of the analysed part. RMS uses
  `librosa.feature.rms`, frame 2048, hop 512.
- **`intro_s`:** the first time the RMS is within 20 dB of its 90th percentile (reported only).

**Final score**, all terms between 0 and 1:

- **J, plan fit:** run `plan_beat_cuts([round(d*fps)], [min(a-1, round(s*fps))], beats, fps)` on the reel shape.
  `J = clean / n`, where a clean join is `on_beat` and its clip has no warning. This is the "tempo against the clip rhythm" term: it asks
  whether a beat falls inside each clip's padding, which is what beat sync needs to trim only padding.
- **Lp:** `L` from 1.3, with `N = T + beats[0]` and the analysed full `duration_s`.
- **Tb, tempo band:** 1 inside the band of 1.1 (folded BPM), else `max(0, 1 - distance/25)`.
- **P, beat strength:** `clamp((pulse - 1) / 1.5, 0, 1)`.
- **Q, intro:** `clamp(intro_ratio / 0.6, 0, 1)`. Silence before the first beat costs nothing, because the beat-synced reel starts at the
  first beat (`Plan.music_start`). A quiet intro after the first beat does cost.
- **`score = 0.35 J + 0.20 Lp + 0.15 Tb + 0.15 P + 0.10 Q + 0.05 meta`**, rounded to 3 decimals in the outputs.
- If the rank-0 score is below 0.50, the song is still used, with the warning "no song fits this reel well (best fit 0.42); try another
  mood".

**Without librosa** (no beat sync possible):

- There is no audio analysis. The ranked list is the metadata order, and only the chosen rank is downloaded.
- Beat sync stays off, with the warning "beat sync needs the music extra".
- Auto music still trims, fades and sets -14 LUFS.

### 1.5 The fixed fallback: the built-in beat

The last entry of every ranked list is **inframe's built-in beat**, so the list is never empty. When nothing was found, or Openverse could
not be reached, it is the only entry.

- **Source:** move `render` and `write_wav` from `tools/synth_music.py` into a new `src/inframe/synthbeat.py`, with numpy only.
  `tools/synth_music.py` then imports them and keeps its command line.
- **Tempo:** for each BPM choice of the reel's speed (1.1), compute J with the exact synthetic beat times (no librosa needed). Take the
  highest J; ties go to the lower BPM.
- **Rendering:** `render(bpm, 120, 0.0, "beat")` makes a 120 s WAV, which loops between beats for longer reels.
- **Where it lives:** in the music cache as `cache_dir()/00000000-0000-0000-0000-000000000BBB.wav`, where BBB is the BPM with 3 digits.
  The id matches ID_RE, so the existing `music.get` cache hit, `/api/music/{id}/audio` and `resolve` all work unchanged.
- **Its `<id>.json`:** `{id, title: "inframe built-in beat, BBB BPM", creator: "inframe", license: "cc0", license_version: "1.0",
  license_url: "https://creativecommons.org/publicdomain/zero/1.0/", attribution: null, source: "inframe", duration_s: 120, builtin: true}`.
  It is CC0, so no credit is needed. It is allowed with `--cc0-only`.
- **Warnings:** "no free song was found for mood X" or "Openverse could not be reached".

### 1.6 Determinism and "another song"

- `choose(shape, mood, cc0_only, rank, analyse)` returns entry `rank % len(list)` of the ranked list. It reports `wrapped: true` when
  `rank >= len`.
- **The ranked list:** the analysed batches in order (1.4), then, once 8 songs have been fetched, the built-in beat.
- When batches 2 and 3 cannot be filled (too few candidates), the built-in beat comes right after the last scored song.
- Inputs that fix the list: the cached search rows (7 days), the reel shape, the mood, `cc0_only`, and the cached analyses. Re-asking for
  the same rank gives the same song.
- "Another song" means asking for `rank + 1`.

## 2. Caching and limits

Everything stays in `music.cache_dir()` (`$INFRAME_CACHE/music`, or `~/Library/Caches/inframe/music`):

- **Audio and license records:** `cache_dir()/<id>.<ext>` and `<id>.json`, written by `music.get` as today. Auto music writes nothing
  there except the built-in beat.
- **`cache_dir()/auto/`:**
  - `search/<key>.json` (7 days)
  - `analysis/<id>.json`
  - `index.json`: the ids that auto music downloaded, with `last_used`
  - `music.get` only globs the top level, so these files never look like tracks.
- **Caps:**
  - At most 8 downloads per (shape, mood, cc0_only).
  - At most 600 s per candidate, so a Jamendo `mp32` file is roughly 10-20 MB [unverified]; the existing 50 MB cap per file stays.
  - Files that auto music downloaded may total **500 MB or 60 tracks**. Past either, evict by least-recent `last_used`: remove the audio,
    `<id>.json` and its analysis. Never evict the song being returned, and never evict files that are not in `index.json` (`music get`
    downloads made by the user).
- **Request budget** (anonymous Openverse: 20/min burst, 200/day):
  - A new mood: at most 8 search requests plus at most 8 detail requests. The typical case is about 3 search plus 4 detail requests.
  - The same mood within 7 days: no search request; at most the detail requests of tracks not in the cache.
  - A 429 gives the existing one-line message, and the ranking falls back as in 1.2.
- **`INFRAME_MUSIC_OFFLINE=1`** (environment): auto music sends no request and uses only the caches and the built-in beat. Tests and the
  Done-when use it.

## 3. Surfaces

### 3.1 Core: `src/inframe/automusic.py` (new)

The public API:

- `ReelShape`
- `shape_from_files(doc, clips_dir)`
- `shape_from_doc(doc, clips, fps, factor)`
- `parse_spec(arg) -> mood | None`, which returns `""` for `auto`, `"calm"` for `auto:calm`, and None for anything else. A mood that
  fails `MOOD_RE` raises `ValueError`.
- `meta_score(row, shape, mood)`
- `audio_score(analysis, shape, meta)`, which returns a dict with the terms and `score`
- `rank_list(shape, mood, cc0_only, upto, analyse)`
- `choose(...) -> Choice`
- `builtin_beat(shape)`
- `can_analyse()`, which is `importlib.util.find_spec("librosa") is not None`

`Choice` holds:

- `audio: Path`
- `meta: dict` (the `<id>.json`)
- `rank`, `n`, `wrapped`
- `mood` (the query used)
- `speed`
- `score: dict` (the terms)
- `bpm`
- `warnings: list[str]`
- `summary()`, the dict stored as `meta["auto"]`: `{rank, n, mood, speed, score, bpm, builtin, wrapped}`

`export.py` changes:

- `export_run` adds `r["music"]["title"]`, `["creator"]` and `["auto"] = meta.get("auto")`.
- Nothing else changes: fitting stays the existing trim or loop, fades, -14 LUFS and beat sync of `build_music_reel`.

### 3.2 CLI (`cli.py`)

- **`inframe export RUN --reel --music auto[:MOOD] [--music-rank N] [--cc0-only] [--no-beat-sync] [--out DIR]`:**
  - With `auto`, beat sync is **on by default** when `can_analyse()`. `--no-beat-sync` turns it off; `--beat-sync` still works.
  - `--music-rank` (default 0, from 0 to 63) is "another song". It is refused without `--music auto...`.
  - The CLI prints one extra line after the reel line:
    `music: auto pick 1 of 5 ("Title" by Creator, CC BY 4.0, 124 BPM, fit 0.81); another song: --music-rank 1`.
  - The auto choice's warnings are printed as `[WARNING]` lines.
- **`inframe cut VIDEO ... --music auto[:MOOD]|FILE|ID [--music-rank N] [--cc0-only] [--beat-sync|--no-beat-sync]`** (new on `cut`):
  - It implies `--reel`. `run_cut` runs with `reel=False` (edit lists unchanged).
  - The CLI then calls the same helper as `export` on `out` (`export_run(doc, out, out, True, [], track, beat)`), writes `export.json`
    and prints through `print_exports`. Put the shared code in one function, `_music_reel(doc, clips_dir, out, args)`, used by both
    commands.
  - A cut with no clips prints "reel not built: no clips" and makes no music request.
- **`inframe music auto RUN [--mood M] [--rank N] [--cc0-only] [--json]`** (new): prints the ranked list up to rank N without building a
  reel. Each line has rank, score, BPM, length, license, title and creator, and the line of rank N is marked `*`. It is for previews and
  checks.

### 3.3 MCP (`mcp_server.py`): a new tool `export_reel`

`export_reel(run, music=None, music_rank=0, beat_sync=None, cc0_only=False, edit_lists=None, out=None)` builds the reel of a finished run:

- `run` is an absolute `segments.json` or run folder (`_segments_file`).
- `music` is `"auto"`, `"auto:MOOD"`, a track id, or an absolute audio path (`_existing_file`).
- `beat_sync=None` means on for auto music when librosa is installed, off otherwise.
- It runs through `_run_exclusive`.
- It returns `_export_result`-style absolute paths plus `reel.music.auto`. When `music` is auto it also returns
  `another_song: "export_reel(run=..., music=..., music_rank=N+1)"`.
- The docstring says that only the mood word leaves the machine.

### 3.4 Serve

- **New route `POST /api/runs/{RID}/music/auto`**, in `serve_export.py`, which owns `load_run`, `current_clips` and `_select`:
  - Body: `{clip_ids: [..]|null, mood: str ("" = from the reel; MOOD_RE), cc0_only: bool, rank: int 0..63}`. Errors are 400 for shape,
    422 for a run that failed, and 502 for a `MusicError` (this should be rare, given the fallback).
  - It builds `shape_from_doc` from the selected clips and runs `automusic.choose` under `serve_music._download`. It is synchronous like
    `/api/music/get`, which already downloads synchronously. Whether the server handles a 30-90 s request well is [unverified]: check the
    request timeout in `serve_http`.
  - Answer: `{rank, n, wrapped, mood, speed, warnings, notice: OPENVERSE_NOTICE, track: {id, title, creator, license, license_version,
    license_url, duration_s, bpm, score, needs_credit, credit_text|null, builtin}}`.
- **Export request:** `music` may also be `{auto: {mood: str, rank: int}, cc0_only}`, with exactly one of `track_id`, `file` or `auto`.
  - `plan_export` validates it the same way.
  - `resolve_music` calls `automusic.choose` with `shape_from_doc(plan.doc, plan.clips, plan.media.info.fps, factor)`. That is the same
    function and the same cache as the preview, so the export uses the song that was shown.
  - `meta["auto"] = choice.summary()`.
  - `beat_sync` stays what the client sends.

### 3.5 UI (`web/src`)

- **`types.ts`:**
  - `ExportRequest.music` becomes `{track_id, file, cc0_only} | {auto: {mood: string; rank: number}; cc0_only: boolean} | null`.
  - New types `MusicAutoQuery` and `MusicAutoResult`.
  - `Engine.musicAuto(runId, q)`.
- **`local.ts`:** `POST api/runs/${enc(runId)}/music/auto`.
- **`mock.ts`:** three fixed tracks plus the built-in beat; `rank % 4`; records its queries.
- **`MusicPicker.tsx`:**
  - A new radio, first under "No music": **"Auto music: pick a free song that fits the reel"** (`data-testid="music-auto"`), with an
    optional mood input whose placeholder is "from the reel".
  - Selecting it calls `musicAuto` with `rank: 0` and shows a spinner, "Choosing a song (the first time downloads up to 4 songs)...".
  - Then it shows a card (`data-testid="auto-pick"`): the title, the creator, `m:ss`, `CC BY 4.0` or `CC0`, the BPM, "fit 81%" and
    "1 of 5".
  - The card has a **Play** button (`musicUrl(id)`) and an **"Another song"** button (`data-testid="another-song"`, which asks for
    `rank + 1`).
  - When `needs_credit` is true, the card shows the credit text in a `<pre>` with "Its credit file is written next to the reel".
  - For the built-in beat it says "No song fitted, so this is inframe's built-in beat (CC0)."
  - Changing the mood or the CC0-only box asks again with rank 0.
  - New props: `runId` and `clipIds`.
- **`ExportDialog.tsx`:**
  - With an auto choice, it sends `music: {auto: {mood, rank}, cc0_only}`.
  - When auto is chosen and `caps.beatSync` is true, it ticks Beat sync once; the user may untick it.
  - After the export it shows "Music: Title by Creator (auto pick k of n)" above the existing credit block.
- **The built page:** run `cd web && npm run build` (it writes `src/inframe/web/dist`) and commit the new dist.

## 4. Licenses

- The default is CC0 and CC BY.
  - The search, `parse_track` and `music.get` refuse everything else (BY-SA, BY-ND, NC) and check the license on the per-track record.
  - With `--cc0-only` (CLI), `cc0_only` (MCP and serve) or the UI box, only CC0 is searched and accepted. Openverse music has no CC0
    tracks, so auto music then normally ends on the built-in beat (CC0). The `--cc0-only` help text and the README say so.
- **Credit:** a CC BY pick goes through `export_run`. That writes `<stem>_credits.txt`, the reel's comment tag and
  `reel.music.credit_text`, as today. The UI shows the credit before the export (`track.credit_text`) and after it.
- The ranking only orders tracks that are already allowed. CC0 wins ties on the meta score (1.3).
- No new dependency: librosa (ISC) is already in the `music` extra, and numpy (BSD) is already a dependency. Add one README line under
  Licenses: "Auto music adds no dependency."

## 5. Tests (no network; synthetic audio only)

Every test sets `INFRAME_CACHE=tmp_path`. Openverse is replaced by `monkeypatch.setattr(music_mod, "_get", fake)`. The fake answers:

- **Search URLs:** canned rows with `tags`, `genres` and `duration`.
- **`/v1/audio/<id>/`:** the row.
- **`binary=True`:** the bytes of a WAV made by ffmpeg.

The synthetic tracks are made with ffmpeg `aevalsrc` (checked, 0):

- **A:** 120 BPM, 60 s (P = 0.5)
- **B:** 90 BPM, 60 s (P = 0.6667)
- **C:** 120 BPM whose first 12 s are 10x quieter: multiply the expression by `if(lt(t,12),0.1,1)`

The reel is 6 clips of 5.0 s with 0.6 s of slack at 30 fps: T = 30 s, rate 12 per minute, speed `fast`.

`tests/test_automusic.py` (new; the librosa parts use `pytest.importorskip("librosa")`):

1. `meta_score`:
   - A 60 s row with `speed_high` beats a 20 s row (a loop) and a `speed_low` row.
   - Equal scores fall back to CC0 first, then id order.
   - `parse_spec("auto:calm") == "calm"`, `parse_spec("auto") == ""`, `parse_spec("x.mp3") is None`, and `parse_spec("auto:a/b")`
     raises.
2. Shape: 6 x 5 s gives speed `fast` and default mood `energetic`; 2 x 30 s gives `slow` and `calm`.
3. `audio_score` on the real analyses of A, B and C:
   - Order A > C > B.
   - J(A) == 1.0, J(B) < 1.0, Tb(B) == 0, Q(C) < 0.5.
4. `choose` end to end with the stub (rows A, B and C as `by`, plus one `by-sa` and one `cc0` row):
   - rank 0 is A, and rank 1 is not A.
   - The `by-sa` id is never fetched.
   - A second identical call makes **no** stub calls (count them) and gives the same ids.
   - `rank = len` wraps to 0 with `wrapped`.
5. Request audit: every non-binary URL the stub saw has host `api.openverse.org`. Its query keys are a subset of
   `{q, category, license, source, page_size, page}` and `q` equals the mood, or the path is `/v1/audio/<id>/`. No URL contains `.mov`,
   a path, or a number with a decimal point (the shape).
6. Caps: with 30 rows stubbed, asking ranks 0..9 makes at most 8 binary fetches, and the last entry is the built-in beat.
7. Fallback:
   - With a stub that raises `MusicError`, or with `INFRAME_MUSIC_OFFLINE=1` and an empty cache, the list is `[builtin]`.
   - The file is a WAV of 120 s.
   - Its `.json` says `cc0`.
   - Its BPM is in the `fast` choices.
   - `music.get(builtin_id)` returns it with no request.
8. With `cc0_only`, `by` rows are never fetched, and only `cc0` rows and the built-in beat appear.
9. CLI with the `clips` fixture of `test_music.py`: copy or import it into a `conftest`, or rebuild it the same way.
   - `main(["export", run, "--reel", "--music", "auto:calm", "--out", d])` returns 0.
   - `export.json` has `reel.music.auto.rank == 0` and `reel.music.beat_sync == true`.
   - The credits file exists for a `by` pick.
   - `--music-rank 1` gives a different `reel.music.file`.
   - `--music-rank` without auto gives exit code 2.
   - `music auto RUN --json` lists the same order.
10. `cut --music auto` with `INFRAME_MUSIC_OFFLINE=1` on a synthetic video (motion mode, testsrc2 plus a moving box, as other cut tests do)
    writes the reel with the built-in beat and `export.json`.

Additions to existing files:

- **`tests/test_serve_export.py`:**
  - `POST /api/runs/{rid}/music/auto` with the stub returns `track.id`; with `rank: 1` it returns another id.
  - A bad mood gives 400, and a bad rank gives 400.
  - An export with `music: {auto: {mood: "", rank: 0}}` and `beat_sync: true` uses the same id as the preview.
- **`tests/test_mcp.py`:** `export_reel(run, music="auto")` with `INFRAME_MUSIC_OFFLINE=1` returns `reel.music.auto.builtin == true` and
  an `another_song` string.
- **`web/test/panels.test.tsx`:**
  - Choosing "Auto music" calls `musicAuto` with rank 0 and shows the title and license.
  - "Another song" calls it with rank 1.
  - The export request carries `music.auto` and `beat_sync: true` when `caps.beatSync` is set.
  - The credit text is shown for a CC BY pick.

## 6. Files

The work has two parts.

**Part A, core, CLI and MCP:**

| File | Status |
|---|---|
| `src/inframe/automusic.py` | new, about 350 lines |
| `src/inframe/synthbeat.py` | new; code moved from tools |
| `tools/synth_music.py` | imports from `synthbeat` |
| `src/inframe/beats.py` | `analyse_signal`, `max_seconds` |
| `src/inframe/export.py` | `music.title`, `creator`, `auto` |
| `src/inframe/cli.py` | `export` flags, `cut --music`, `music auto`, print line |
| `src/inframe/mcp_server.py` | `export_reel` |
| `tests/test_automusic.py` | new |
| `tests/test_mcp.py` | additions |
| `README.md` | "Music for the reel": an "Auto music" bullet with the scoring rule, caps and request budget; the Licenses line |

**Part B, serve and UI (needs Part A):**

| File | Status |
|---|---|
| `src/inframe/serve_export.py` | route plus auto spec |
| `web/src/engine/types.ts` | |
| `web/src/engine/local.ts` | |
| `web/src/engine/mock.ts` | |
| `web/src/ui/MusicPicker.tsx` | |
| `web/src/ui/ExportDialog.tsx` | |
| `src/inframe/web/dist/**` | rebuilt |
| `tests/test_serve_export.py` | |
| `web/test/panels.test.tsx` | |

Not touched: `music.py`, `musicreel.py`, `soundtrack.py`, `serve_music.py` (only its `_download` lock is imported), `pipeline.py` and
`pyproject.toml`.
