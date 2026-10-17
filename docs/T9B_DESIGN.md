# T9b design: browser mode as a second engine ("This browser" / "My Mac")

Status: built (segments port and engine switch; browser media; models, detector and runs; tap; lossless export and limits).

## 0. Facts read

- `web/src/engine/types.ts` already defines `Engine`, `Caps` and `openFile?(file)`. `engine/browser.ts` was a stub that throws 501. `engine/index.ts` picked `LocalEngine` when `/api/session` answers and `BrowserEngine` otherwise; `--mode pages` forces `BrowserEngine`. TopBar showed a badge with the engine kind but no switch.
- `segments.py` is pure, 78 lines: `sample_edges`, `hysteresis`, `merge_gaps`, `drop_short`, `pad`, `build_segments(lo = offset)`. `serve_runs.resegment` fills any missing field from the run's settings.
- Balanced preset (the reference-clip gate): RF-DETR small at 5 fps, frames decoded at 640 px on the long side, squashed to 512, ImageNet mean/std, sigmoid on `logits[:, :, cls]`, min score, boxes from cxcywh. DINOv2 identity: `pad_square`, then the HF processor (short edge to 256, centre-crop 224), `pooler_output` L2-normalised, session threshold 0.66, margin 0.04 (`identity.SPECS`).
- The T9-0 spike: both ONNX models run fully on WebGPU (ort-web 1.30). RF-DETR small takes 61 ms in Chrome and 116 ms in Safari; DINOv2-S 19 / 30 ms. Downloads are 123 MB and 88 MB fp32. HEVC decodes in both browsers. Reading the whole file into memory would need several GB for a long phone clip, so we stream instead. The streaming path was not measured.
- No `.credit` CSS rule existed. MusicPicker shows credit text in a `<pre>`, and `white-space: pre` stops it wrapping.
- The CSP in `serve.py` already allows `blob:` and `worker-src blob:`. It lacks `'wasm-unsafe-eval'`.

## 1. Engine switch

- `main.tsx` builds every engine that is available. Under `inframe serve` that is `LocalEngine` and `BrowserEngine`; on a static host it is `BrowserEngine` only.
- A two-button control replaces the TopBar badge and also sits on the OpenScreen: "My Mac" and "This browser". The choice is kept in localStorage `inframe.engine` (wrapped in try/catch).
- Switching while a video is open asks first ("The open video closes; runs stay with the engine that made them"), then rebuilds the store with the other engine.
- On a static host "My Mac" is disabled, with the text "Run `inframe serve` on your Mac and open the page it prints". Safari blocks public pages from calling localhost.
- New Caps: `reel` and `editLists`. Both are true for local and false for browser in T9b, so ExportDialog hides those sections. Browser caps: output `'granted-folder'` when `showDirectoryPicker` exists, else `'download'`. `streamCopyCut` becomes true in part 5. Everything native is off: dialogs, Photos, locate, reveal, music, beat sync, saved targets, people, proxy.
- Presets in the browser: balanced (small, 5 fps) and fast (small, keyframes only; the Mac uses nano). Accurate is not offered because the base model is not shipped.

### Part 1 as built

- `engine/index.ts`: `availableEngines()` returns `{ local, browser, staticHost }`; `selectEngine()` stays and returns `local ?? browser`. In mock mode both exist, so the switch can be tried in `npm run dev -- --mode mock`.
- `engine/switch.ts`: the choice (`readChoice`, `saveChoice`, `pickInitial`), `createSwitcher`, and `SwitcherContext`. TopBar and OpenScreen draw the control only when a switcher is in the context, so a store built alone (the older tests) keeps the badge.
- `shell.tsx`: `createShell(engines)` holds the current store and rebuilds it on a switch (the old store's edits are settled first, waiting for a save in flight; if that save fails or conflicts, the switch is refused with a toast and the old store stays; otherwise it is disposed). `main.tsx` and `test/engine-switch.test.tsx` both use it.
- `LocalEngine` fills `caps.reel` and `caps.editLists` with true when an older server does not send them. The server itself (`serve.py`) is unchanged in part 1.
- `BrowserEngine`: `recent()` and `clearRecent()` answer `[]` so the open screen draws instead of the 501 error card; its open screen says that opening a video comes with the next part (part 2 adds `openFile`). `Engine.resolveUrl?` is declared; `BrowserEngine` implements it in part 2.

## 2. Data flow (all inside the tab; nothing is uploaded)

- Opening: File (input or drop) -> `openFile` -> `demux.ts`. mp4box.js (BSD-3) is fed 8 MB `file.slice()` chunks with `fileStart`, and follows mp4box's next-position hint when moov is at the end. It gives: the sample table, sync samples (the keyframes), codec, the raw hvcC/avcC description, the tkhd matrix (rotation and mirroring), offset and end_time, and the audio track.
- The media id is a fingerprint (`name|size|lastModified`). `video_url` is `URL.createObjectURL(file)`.
- A run (Worker `browser/worker.ts`):
  - The Worker gets the File (structured clone) and a `RunRequest`. Samples are picked as the Python pipeline picks them (5 fps or keyframes).
  - `VideoDecoder` streams with backpressure (`decodeQueueSize` <= 30, then `releaseUsedSamples`). Only the frame nearest each sample time is kept.
  - Each kept frame is drawn upright with the rotation transform onto an `OffscreenCanvas` at 640 px on the long side. From there: the 512 squash goes to RF-DETR, and padded crops go to DINOv2 when the target is a tap.
  - Then `target.evaluate` -> `Samples` -> `segments.ts` `buildSegments` -> `RunData`. The result is stored in IndexedDB.
  - Progress and cancel go through the existing `Job` shape (`AbortController`). If the Worker cannot get WebGPU in Safari, the run moves to the main thread and logs a warning [unverified for Safari 26.5].
- Storage: IndexedDB `inframe` holds media (`MediaInfo` without URLs), runs (segments, samples, edits) and pick state. It never holds video, frames or embeddings. Re-picking the same file brings its runs back.
- Images: `frameUrl`, `pickFrameUrl`, `sheetUrl`, `candidateUrl` and `personUrl` return `inframe-blob:` keys. A new optional method `Engine.resolveUrl(url): Promise<string>` plus a `useEngineUrl` hook in the 5 image components (Player, FilmstripTrack, ClipStrip, Inspector, ExportDialog) render on demand into blob URLs (LRU, revoked on evict). `LocalEngine` has no `resolveUrl`, so its URLs pass through unchanged.

## 3. Models (fetch, cache, licenses)

- `tools/export_onnx.py` is the T9-0 `export.py`, run in a bench-only venv: rfdetr 1.5.2 for RF-DETR small, transformers for DINOv2-S at opset 17. It writes `$INFRAME_MODELS/onnx/{rfdetr_small_v1.onnx, dinov2_s_v1.onnx, manifest.json}`. The manifest lists file, bytes, sha256, license, source, input shape and output names.
- The onnxruntime-web `.wasm` (MIT; the `asyncify` file, because `onnxruntime-web/webgpu`, the build T9-0 measured, loads that one; the default `jsep` build would need a different file) is listed in the same manifest and served from the same place. This keeps the committed dist small; the size is checked in part 3.
- The base URL is set at build time: `VITE_MODEL_BASE`, default `./models/`. `inframe serve` adds GET `/models/<manifest name>`, read-only, allowlisted names, Range, nosniff, and adds `'wasm-unsafe-eval'` to script-src. T9c later points the base at a pinned Hugging Face commit URL. No tokens are involved; the files are public.
- Cache: Cache Storage `inframe-models-v1`, keyed by sha256. The digest is checked with `crypto.subtle` after download, and a bad file is deleted. `navigator.storage.persist()` is requested.
- Preflight shows "Download models (about 211 MB, once)" with progress and a "Remove downloaded models" button.
- ort sessions use `executionProviders ['webgpu']`. Wasm with 1 thread is the fallback (about 19x slower), with a warning. COOP/COEP is not needed.
- Licenses: RF-DETR small weights Apache-2.0 [verify per release in part 3], DINOv2 Apache-2.0, onnxruntime-web MIT, mp4box BSD-3, client-zip MIT. Each goes into README Licenses, and rollup-plugin-license still gates the bundle. Mediabunny (MPL) is not used, so no decision needs logging.
- fp16 weights would halve the download. Use them only if part 3 parity holds; otherwise stay fp32.

### Part 3 as built

- Files: `tools/export_onnx.py` (bench venv; writes `$INFRAME_MODELS/onnx/{rfdetr_small_v1.onnx 123.5 MB, dinov2_s_v1.onnx 88.4 MB, ort-wasm-simd-threaded.asyncify.wasm 26.8 MB, manifest.json}`, fp32), `serve.py` (`GET /models/<name>` for the names in manifest.json only, no symlinks, Range, `Cache-Control: no-cache`; CSP script-src adds `'wasm-unsafe-eval'`), `web/public/_headers` (same CSP, test-checked), `browser/models.ts`, `detect.ts`, `targets.ts`, `pipeline.ts`, `worker.ts`, `engine/browser.ts`, `ui/Preflight.tsx`.
- The ort WASM runtime is handed to ort as bytes (`env.wasm.wasmBinary`, read from Cache Storage), so ort never fetches it and `connect-src 'self'` needs no `blob:`. `vite.config.ts` `ortPlugin` stops Vite copying ort's own 26 MB wasm into dist and removes the one https:// string the dist scan would flag.
- One run: `pickSamples` (decode.iter_samples' loop over the sample table) -> `planRuns` (one decoder run per GOP) -> `FrameDecoder.decode` with an async `onFrame` (the next picture waits for the detector) -> `FrameRenderer` (upright 640 px canvas, squash to 512, `preprocess`) -> `RfDetr.detect` -> target -> `buildRunDoc` (segments.ts). The run id is `<media id>-<slug>`, so a re-run of the same target replaces the earlier one (the UI keeps the edits aside client-side, as for the Mac).
- Jobs: `BrowserEngine.startRun/job/currentJob/cancel/boxes`; one job at a time (409 `busy` with the job); cancel aborts between two pictures, and a Worker that has not stopped 5 s later is terminated. The Worker holds the detector session between jobs. A browser whose Workers have no `navigator.gpu` makes the same handler run on the page's thread (`NO_WEBGPU_HERE`); wasm (1 thread) is the last fallback, with a warning.
- Targets in the browser so far: any person and any COCO class, optionally in a zone. Tap, saved targets, photos and motion answer 501 until part 4.
- Preflight in the browser engine: WebGPU (a warning, not a stop), WebCodecs, whether the detector model is downloaded, and the Download / Remove buttons (Preflight.tsx, shown on the open screen and in the run panel).
- Tests: golden detect/targets vectors (pytest + vitest), vitest for models/pipeline/engine, and `tests/browser` (see its conftest): the real engine, Worker and models in Chrome through the real `inframe serve` server (`web/selftest` is the page), Safari through the open-and-POST-back page.

## 4. Segments port and golden vectors

- `web/src/model/segments.ts` ports all five functions line by line, with the same order of operations so the doubles match. It also exports `resegment(samples, settings, params, bounds)`, which follows the `serve_runs` fill-in rule (a missing or null field comes from the run's settings; none left, or exit above enter, is an error). `bounds` is `{ offset, end }`, the video's first picture and end in file time, which `serve_runs` reads from the run document.
- `tools/gen_golden.py` (stdlib `random` with fixed seeds, no numpy) writes `tests/golden/segments_v1.json`, which both test suites read. It holds 35 hand-made cases and 45 random series (80 in all), plus 10 `resegment` cases. Each case stores its inputs and the result of every stage (edges, hysteresis, merged, kept, presence, clips):
  - hand cases: empty, one sample, all above, exit == enter, lo > 0, clamping at duration, a gap exactly equal to merge_gap, padding that causes a merge, two error cases;
  - random series: varied lengths, rates, jitter and parameter sets.
- pytest `tests/test_golden.py` runs the generator with `--check` and fails if the committed file drifts from `segments.py`; it also replays every stage and holds hand-written anchors, so a regenerated file cannot quietly bless a changed rule. vitest `web/test/golden-segments.test.ts` reads the same file and compares every stage within 1e-9.
- Later golden files use the same pattern:
  - `detect_v1.json`: postprocess on synthetic logits and boxes, `person_crop`, `pad_square` geometry;
  - `targets_v1.json`: `AnyOfClass`, `ZoneTarget` and `min_box` on synthetic candidate lists;
  - `gallery_v1.json`: `Gallery.add`/dedup/evict and `unit_score` on seeded unit vectors.

## 5. Targets in the browser

- `any`, `class` (COCO-91 table from `classes.py` exported into the golden file) and `zone` are exact ports.
- Tap is simplified: there is no OC-SORT, because `trackers` is OpenCV/Python. Each candidate's DINOv2 crop is compared with the tap gallery. A match needs cosine >= 0.66 and a lead of at least 0.04 over the runner-up; the gallery grows at 0.62 with dedup 0.95; an IoU continuity bonus carries the match between samples. The UI note says "Browser tap matching is simpler than My Mac".
- Photos, saved targets and people stay off.

### Part 4 as built

- `browser/embed.ts`: `personCropRect` (identity.person_crop geometry), `padSquare`, `resizeBicubic` (a port of Pillow's two-pass 8-bit bicubic, so every byte equals PIL's), `centerCrop`, `toTensor`, `Dino` (pooler_output, L2) and `embedDetections` (score >= 0.3, best 8 per picture, boxes of at least 48 x 16 px at the 640 px picture). `browser/gallery.ts`: `Gallery` (add, dedup, evict, optional colour signatures) and `unitScore`, ports of identity.py. Both are held to `tests/golden/gallery_v1.json` (Gallery members after every add, unit_score, crop rectangles, pad_square, resized bytes, the 224 input tensor within 1e-6) in vitest and pytest. The processor of transformers 5.19 differs from the Pillow path by at most one grey level; `tests/test_golden.py` checks that when the model card is in the local Hugging Face cache.
- `BrowserTap` (targets.ts) is a stateful target: `evaluate(candidates, t)` records, `finalize()` answers for every picture. Each tap lands on the detection with IoU >= 0.3 on the nearest picture within 1 s that has one, else a `TapError` with the Python wording ("no person detected within 1 s of the tap at 5.0 s (nothing at that box); pick another moment or position"). The anchors seed a `Gallery` (DINOv2 spec, cap 24). The pictures are then walked forwards and backwards: the best candidate matches when its cosine plus 0.05 for a box that continues the last match (IoU >= 0.3 within 1 s) reaches 0.66, its own cosine is at least 0.62, and it leads the runner-up by 0.04; a match joins the gallery. A matched picture scores with the detector's score. Wrapped in a `ZoneTarget` like any target.
- `scoreVideo` calls `embed(dets)` (the Worker crops the 640 px picture it just drew, one DINOv2 run per crop, batch 1) for targets with `needsEmbeddings`, then `finalize()` once, never after a cancel. The run document gets `timing_s.embed_model_s`. Memory: one 384-float vector per embedded detection is held until the end (about 1.5 KB each; 60 minutes with 8 people in every picture would be about 200 MB); part 5's limits measure it.
- `BrowserEngine`: `pickState/tap/undoTap/setZone/resetPick` keep the pick in IndexedDB per video and class (taps rounded as `pick.add_tap`, the zone validated and rounded as the Mac; a tap needs a box in 0..1 and a time inside the video). `startRun` accepts `tap`: the taps of that class go to the Worker; no tap is a 422 "Tap the person in the video first."; a missing DINOv2 file is the models 409. The checklist shows "Recognition model downloaded" for a tap.
- UI: the Tap tab says "Browser tap matching is simpler than My Mac"; Saved needs `caps.savedTargets`, Photos needs `caps.openDialog` or `caps.fileBrowser`, People needs `caps.people` (all off in the browser).
- Not measured: how well it follows a real person (not yet tried on real footage), the embedder's cost per picture on a crowded scene, Safari.

## 6. Export

- `browser/mux.ts` is our own small ISO-BMFF writer; mp4box only parses. It writes `ftyp`, then `moov` with `mvhd` and one `trak` per stream:
  - `tkhd` with the source matrix;
  - `edts/elst` whose `media_time` is the keyframe's cts (this covers B-frame delay);
  - `stbl` with the source `stsd` entry copied byte for byte (hvc1 + hvcC, colr, pasp kept), `stts`, `ctts`, `stss`, `stsc`, `stsz` and `co64`.
  Then `mdat`.
- Video starts at the sync sample at or before the clip start. RASL leading pictures (decoded after the keyframe but shown before it) are dropped. AAC audio packets from the keyframe time to the end are copied; other audio codecs are dropped with a warning. Nothing is re-encoded, and slow-motion retiming is not in T9b.
- `browser/exportzip.ts` streams the clips, the LosslessCut CSV and the per-clip sidecar JSON (same schema) through client-zip, which handles Zip64.
  - Chrome writes to a file from `showSaveFilePicker` or into the granted folder, so memory stays bounded.
  - Safari builds a Blob and is capped at 2 GB; above that the dialog says "use My Mac".
- `ExportOutput` URLs are blob URLs.

### Part 5 as built (export)

- `browser/mux.ts` (own writer, `readMuxSource` + `planClip` + `muxClip` + `clipStream`). Per clip it writes `ftyp` (qt for .mov, isom otherwise), `moov` (`mvhd`, one `trak` per kept track: `tkhd` with the source matrix, `edts/elst`, `mdhd`, `hdlr`, `vmhd`/`smhd`, `dinf`, `stbl` with the sample entry copied byte for byte from the source file, `stts`, `ctts`, `stss`, `stsc`, `stsz`, `co64`) and `mdat`. The head is built in memory; the sample bytes are streamed from the source file in slices of at most 8 MB, so a clip of any size costs a few MB of memory. Chunks are about one second long per track, picture and sound taking turns.
  - The cut follows `inframe cut`: the keyframe is `cut.snap_to_keyframe` (last sync sample at or before the start, else the first), the end is the first sample whose decode time is at or after the clip end (ffmpeg's `-t` tests dts), and RASL pictures (after the keyframe in decode order, shown before it) are dropped. The decode times of the kept samples are packed again and their presentation times are not touched, so the elst `media_time` is the smallest delay that keeps every composition offset non-negative (the keyframe's B-frame delay).
  - After a B-frame tail the last pictures can be decoded before the end yet shown after it, leaving a hole before one lone picture. The Python cut has the same hole; the clip keeps that picture and lasts as long as the pictures it holds (`planClip.shownS = min(last end - keyframe cts, sum of durations)`), which is also what ffprobe reports.
  - Audio: AAC (`mp4a.40.*`) only. The first packet is the first whose presentation time is at or after the keyframe's minus half an AAC frame (the Python align pass), and an edit (an empty edit if it starts late, a `media_time` skip if it starts early) places it at the remainder, so it starts within half a frame of the picture. Other audio is left out with a warning. Timecode, metadata (`mebx`, where a phone keeps its location) and container tags are never read.
  - H.264 and HEVC (`avc1`, `avc3`, `hvc1`, `hev1`) only; another codec says "Use My Mac" (415).
- `browser/exportzip.ts`: `clipName`, `sidecarName`, `losslessCutCsv` (Python's `f"{x:.3f}"`, `csv.writer` line ends and quoting; `pyRound`/`fmt3` follow Python's round-half-even on exact ties), `buildSidecar` (schema 1.2, the key order and content of `sidecar.build_sidecar`; `source.path` is the file name only, `inframe_version` is `0.1.0 (browser)`), `planExport`, `exportStream` (client-zip 2.5.0 streams the entries; Zip64 is automatic) and the sinks. The SHA-256 of the source (`source.hash`, required by the schema) is computed in the page in 4 MB slices by `browser/sha256.ts` (WebCrypto cannot digest a stream), once per open file.
- Sinks (`chooseSink`, called first in the click's handler chain because the pickers need a click): `showSaveFilePicker` writes through `FileSystemWritableFileStream` (memory stays bounded), else `showDirectoryPicker` (a granted folder, the zip is written into it), else a Blob download. The Blob path is capped at 2 GB (`BLOB_CAP`); above it `startExport` answers 413 "Use My Mac" before reading anything, and the export dialog disables Export with the same words. A closed save dialog is a 499 with no job.
- `BrowserEngine.startExport/exportResult/runFileUrl`: the clips are the run's edited clips (else the detector's), narrowed by `clip_ids`; only `clips` is accepted (reel, music and edit lists are 501). One job at a time. The result document has the fields `ExportDialog` reads (`clips`, `outputs`, `warnings`); `outputs` lists the zip (kind `zip`, with a blob URL when the browser downloaded it, shown as "Save again") and each file inside it. A cancel aborts the zip stream (client-zip's `cancel` throws into the entries generator, which ends quietly) and the writable is aborted, so no partial file is kept.
- Tests: `web/test/mux.test.ts` (3 clips each from an open-GOP HEVC with B-frames, rotation 90 and AAC, and from an H.264 .mov with B-frames; ffprobe, the plan and the Python `cut_clip` agree; `ffmpeg -v error -f null` prints nothing), `exportzip.test.ts` (`unzip -t`, `tools/validate_sidecars.py`, the CSV against Python, SHA-256, sinks), `browser-export.test.ts` (the engine's job), `limits.test.ts`, `browser-limits-ui.test.tsx`; `tests/browser/test_export_ffprobe.py` does the ffprobe checks on a zip made in Chrome through the real File System Access API.

## 7. Limits

- `web/src/model/limits.ts` holds measured constants per browser: decode fps, detector and embedder ms, peak MB per video minute.
- `estimate(duration, preset, target, browser)` feeds the Preflight and RunPanel. It suggests local mode ("about 25 min here, about 3 min on My Mac: run `inframe serve`") when the estimate passes 10 minutes or the duration passes the longest measured duration that stayed under 4 GB peak RSS.
- With no HEVC WebCodecs (Firefox) or no WebGPU, the panel says why and suggests My Mac.
- The constants come from `tools/browser_limits.py`: synthetic 1080p HEVC at iPhone bitrate, 5, 20 and 60 minutes, in Chrome and Safari, measuring peak RSS and seconds per video minute.

## 8. Testing without any model looking at footage

- Every automated test uses ffmpeg synthetic media (testsrc2, sine audio, `-display_rotation`), LCG noise tensors or golden vectors produced by Python code.
- Browser tests: Python playwright (Apache-2.0) in a new `browser` dependency group, using `channel="chrome"` so no browser is downloaded, plus the T9-0 open-Safari-and-POST-back pattern. They live in `tests/browser/` and are skipped unless `INFRAME_BROWSER_TESTS=1`.
- Every browser test checks that each request is a GET to the page origin or the model base.
- No screenshots are taken except with the mock engine or synthetic video.
- Model parity: ort-web output against python onnxruntime CPU on the same ONNX, within 1e-3 relative, using LCG noise and a synthetic frame.
- Mux check: ffprobe confirms the first shown pts is 0 (+-1 frame), the elst is present, rotation matches the source, the tag is hvc1 with hvcC, the frame count and duration match the plan, and audio starts within 1 AAC frame. Results are compared with the Python cut of the same synthetic file.
- Reference-clip parity is `tools/browser_regress.py`. It takes the regress lock, starts serve, and feeds the file in place with `setInputFiles`. It runs balanced any-person, saves `segments.json` to `~/.cache/inframe/regress-browser/`, then calls `tools/regress.py --from-run`. It prints only interval times and PASS/FAIL. Run it yourself; the automated tests do not.
