"""inframe as an MCP server (stdio). Tools take and return local paths and numbers, never video bytes.

    probe_video(path)                      facts about a video
    list_people(path)                      who is in the video: tracks, times, thumbnail paths, a target for each
    list_targets()                         the saved targets (people kept by name): cut_video(target="saved:NAME")
    add_target_crops(name, run)            add crops of the person a finished run found to a saved target
    cut_video(path, mode, preset, target)  clips of the stretches with a person / motion / the chosen person in frame
    compare_runs(a, b)                     how two cut runs of the same video agree

Run it with `inframe-mcp`. Logs go to stderr; stdout belongs to the protocol.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import anyio
import anyio.from_thread
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import identity, models
from .activity import check_activity
from .editlist import parse_formats
from .cut import parse_slowmo
from .names import safe_stem
from .pipeline import MODES, PRESETS, Settings, run_cut
from .tap import TapError, is_point_spec, resolve_preset
from .zone import Zone, check_overlap, zone_from_target_files

mcp = MCPServer(
    "inframe",
    instructions=(
        "Cuts a video into clips of the stretches where a person (detect) or anything moving (motion, static camera only) "
        "is in frame. To keep one person: list_people(video) shows who is there (thumbnails, times, a `target` string for each), then "
        "cut_video(video, target=<that string>). A person kept by name is a saved target: list_targets() names them, and "
        "cut_video(video, target='saved:NAME') finds that person with no tap. All paths are absolute local paths. cut_video and list_people on a 20 minute video "
        "take about 2 minutes and report progress."
    ),
)

_cut_lock = threading.Lock()  # one cut at a time: it uses the GPU and writes one output folder


class _Cancelled(Exception):
    """Raised inside the worker thread, from the progress callback, once the client has cancelled the call."""


def _abs_path(value: str, what: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{what} must be a non-empty path")
    p = Path(value.strip()).expanduser()
    if not p.is_absolute():
        raise ToolError(f"{what} must be an absolute path (got {value!r})")
    return p


def _existing_file(value: str, what: str) -> Path:
    p = _abs_path(value, what)
    if not p.exists():
        raise ToolError(f"{what}: no such file: {p}")
    if not p.is_file():
        raise ToolError(f"{what}: not a file: {p}")
    return p


def _non_negative(name: str, v: float) -> float:
    if not math.isfinite(v) or v < 0:
        raise ToolError(f"{name} must be a number, 0 or more (got {v!r})")
    return float(v)


def _probe(p: Path):
    from .probe import ProbeError, probe

    try:
        return probe(str(p))
    except FileNotFoundError:
        raise ToolError("ffprobe not found on PATH; install ffmpeg (brew install ffmpeg)") from None
    except ProbeError as e:
        raise ToolError(str(e)) from None
    except (ValueError, KeyError) as e:
        raise ToolError(f"{p} is not a usable video: {e}") from None


def _source_summary(src: dict) -> dict:
    keys = ("codec", "profile", "width", "height", "display_width", "display_height", "rotation", "fps", "duration",
            "has_audio", "audio_codec", "n_keyframes", "keyframe_gap_max_s")
    return {k: src[k] for k in keys if k in src}


@mcp.tool()
async def probe_video(path: str) -> dict[str, Any]:
    """Facts about a local video file: codec, stored and displayed size, display rotation, fps, duration (seconds),
    audio, keyframe spacing. Reads stream facts only; container tags (such as location) are never returned.

    Args:
        path: absolute path of the video file.
    """
    p = _existing_file(path, "path")
    info = await anyio.to_thread.run_sync(_probe, p)  # the keyframe scan reads every packet: not on the event loop
    return dict(path=str(p), **_source_summary(info.to_dict()))


async def _run_exclusive(job, ctx: Context | None):
    """Run job(report) in a worker thread, one job at a time, with MCP progress; a cancelled call stops the job at its next report."""
    last = [-1.0]
    cancelled = threading.Event()

    def report(fraction: float, message: str) -> None:  # runs in the worker thread
        if cancelled.is_set():  # the client gave up: stop at the next progress step and free the lock
            raise _Cancelled()
        if ctx is None or fraction - last[0] < 0.01 and fraction < 1.0:
            return
        last[0] = fraction
        try:
            anyio.from_thread.run(ctx.report_progress, round(fraction * 100, 1), 100, message)
        except Exception:  # progress is a courtesy; never fail a run over it
            pass

    def work():
        if not _cut_lock.acquire(blocking=False):
            raise ToolError("another cut_video or list_people run is in progress in this server; wait for it to finish")
        try:
            return job(report)
        finally:
            _cut_lock.release()

    try:
        return await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        cancelled.set()  # the thread cannot be killed; it stops at its next progress report and then releases the lock
        raise


def _parse_target(target: str | list[str] | None, mode: str, preset: str = "balanced"):
    """The `--target` values of the CLI as a Target object (None = every person)."""
    if target is None or target == "" or target == []:
        return None
    if mode != "detect":
        raise ToolError("target needs mode \"detect\"")
    specs = [target] if isinstance(target, str) else list(target)
    if not all(isinstance(x, str) for x in specs):
        raise ToolError("target must be a string such as \"tap:/abs/person_001.json\", or a list of them")
    for x in specs:
        kind, _, rest = x.partition(":")
        if kind in ("tap", "photos") and rest and not is_point_spec(rest) and not Path(rest.strip()).expanduser().is_absolute():
            raise ToolError(f"target {x!r}: the path must be absolute")
    try:
        return identity.parse_targets(specs, identity.DEFAULT_EMBEDDER, False, identity.lazy_detect(PRESETS[preset].model if preset in PRESETS else "small"))
    except TapError as e:
        raise ToolError(str(e)) from None
    except identity.EmbedderUnavailable as e:
        raise ToolError(str(e)) from None


def _run(video: Path, out: Path, s: Settings, report, target=None) -> dict:
    try:
        return run_cut(str(video), str(out), s, target, progress=report)
    except _Cancelled:
        raise
    except identity.Ambiguous as e:  # photo mode cannot decide alone: confirm with list_people / `inframe pick --photos`
        raise ToolError(str(e)) from None
    except (models.ModelsMissing, identity.EmbedderUnavailable, TapError) as e:
        raise ToolError(str(e)) from None
    except ValueError as e:
        raise ToolError(str(e)) from None
    except RuntimeError as e:  # a failed run: the message says what was saved and where
        raise ToolError(str(e)) from None
    except Exception as e:
        raise ToolError(f"cut failed: {type(e).__name__}: {e}") from None


@mcp.tool()
async def list_people(path: str, out: str | None = None, min_seen: float = 1.0, max_people: int = 20,
                      ctx: Context | None = None) -> dict[str, Any]:
    """Who is in a local video: one detect + track pass (balanced preset, 5 samples per second; needs the tracking extra).
    Returns the people the tracker followed, longest in frame first. Per person: `person` (1, 2, ...), `first_s` and `last_s`,
    `seen_s` (seconds in frame), `stretches` ([start, end] seconds), `thumbnail` (absolute path of a JPEG crop; look at it to
    tell who is who) and `target`, a string to pass to cut_video(target=...) to keep only that person. A person who leaves
    and comes back, or whom the tracker loses, may appear as two entries; pass both targets as a list to cut_video.
    A 20 minute video takes about 2 minutes; progress is reported while it runs.

    Args:
        path: absolute path of the video file. It is only read.
        out: absolute folder for people.json, person_NNN.jpg and person_NNN.json; default <video stem>_inframe/people next to the video.
        min_seen: leave out people seen for less than this many seconds (0 or more).
        max_people: keep at most this many people (1 or more).
    """
    video = _existing_file(path, "path")
    min_seen = _non_negative("min_seen", min_seen)
    if not isinstance(max_people, int) or isinstance(max_people, bool) or max_people < 1:
        raise ToolError(f"max_people must be a whole number, 1 or more (got {max_people!r})")
    out_dir = video.with_name(video.stem + "_inframe") / "people" if not out else _abs_path(out, "out")
    if out_dir.exists() and not out_dir.is_dir():
        raise ToolError(f"out: exists and is not a folder: {out_dir}")
    await anyio.to_thread.run_sync(_probe, video)

    def job(report) -> dict:
        from .people import scan_people

        try:
            return scan_people(str(video), out_dir, report, min_seen, max_people)
        except _Cancelled:
            raise
        except (ValueError, RuntimeError) as e:
            raise ToolError(str(e)) from None
        except models.ModelsMissing as e:
            raise ToolError(str(e)) from None
        except Exception as e:
            raise ToolError(f"list_people failed: {type(e).__name__}: {e}") from None

    doc = await _run_exclusive(job, ctx)
    return dict(out_dir=str(out_dir), people_json=str(out_dir / "people.json"), **{k: v for k, v in doc.items() if k != "video"})


@mcp.tool()
async def cut_video(
    path: str,
    mode: str = "detect",
    preset: str | None = None,
    pad: float = 1.0,
    min_len: float = 1.0,
    merge_gap: float = 1.5,
    reel: bool = False,
    edit_lists: str | None = None,
    out: str | None = None,
    target: str | list[str] | None = None,
    zone: str | None = None,
    zone_overlap: float | None = None,
    activity: str = "any",
    slowmo: str = "off",
    slowmo_audio: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Cut a local video into lossless clips (stream copy, no re-encode) of the stretches where the target is in frame.

    mode "detect" (default): RF-DETR person detector, any camera. mode "motion": no model, static camera only.
    preset "balanced" (5 samples per second), "accurate" (the same rate with the larger RF-DETR base: finds smaller people, about 1.5 times
    the detector time) or "fast" (keyframes only / 2 per second). A 20 minute video takes about
    2 minutes; progress is reported while it runs. Returns the clip list (absolute file paths, start and end seconds,
    peak score), the unpadded presence intervals, and the path of segments.json.

    Args:
        path: absolute path of the video file. It is only read.
        mode: "detect" or "motion".
        preset: "balanced" (default), "accurate" or "fast". With a target picked by `inframe pick`, the default is the preset recorded in
            that target.json, and a preset with another detector is refused (the tapped box would not be found again).
        pad: seconds added before and after each stretch (0 or more).
        min_len: drop stretches shorter than this many seconds (0 or more).
        merge_gap: join stretches closer than this many seconds (0 or more).
        reel: also join the clips, in time order and without re-encoding, into <stem>_reel.<ext> (result key "reel").
        edit_lists: comma separated formats of edit lists of the cuts on the original video: fcpxml, fcp7, kdenlive, edl, otio, or "all"
            (edl only for sources of 60 fps or less, fcp7 for the usual SMPTE rates). Result key "edit_lists".
        out: absolute output folder; default <video stem>_inframe/<mode> next to the video. Clips of an earlier run in it are replaced.
        target: who to keep instead of every person (mode "detect" only): the `target` string of a person from list_people
            ("tap:<absolute person_NNN.json>"), "tap:T,x,y" (seconds, then x and y as fractions of the frame), or
            "photos:<absolute folder of photos of the person>", "saved:<name>" for a person kept with list_targets / `inframe target add`
            (no tap needed; fails with a message when the saved thumbnails are gone), or "class:NAME" (every dog, car, ... of the 80 COCO
            classes; add a tap string to follow one of them). A list of strings adds appearances of the same person.
            "photos:" that cannot decide alone fails with a message; confirm it with list_people instead.
        zone: "x,y,w,h", fractions of the upright frame as a player shows it (0,0 is the top left): only people whose box centre is
            inside count. Static camera only: when the camera moves, a warning says so and the zone is ignored. Motion mode counts
            the changed pixels inside the zone.
        zone_overlap: with zone, a person counts when at least this fraction (above 0, up to 1) of their box is inside, instead of the centre.
        activity: "any" (default) or "riding": a person counts only while their box moves at least about 1.65 box heights per second
            (detect mode, needs 5+ samples per second, static camera).
        slowmo: "off" (default), "auto" (the file's capture rate over 30 fps: 240 fps gives 8, 120 gives 4; no change under 100 fps) or a
            factor such as "4" or "8": retime every clip to play that many times slower, without re-encoding the picture. Each clip then lasts
            factor times longer; its `slowmo` entry in the result and in segments.json holds the factor and the clip-time map.
        slowmo_audio: with slowmo, keep the audio stretched to match (re-encoded to AAC); by default the retimed clips have no audio.
    """
    video = _existing_file(path, "path")
    if mode not in MODES:
        raise ToolError(f"mode must be one of {list(MODES)} (got {mode!r})")
    if preset is not None and preset not in PRESETS:
        raise ToolError(f"preset must be one of {sorted(PRESETS)} (got {preset!r})")
    try:
        preset = resolve_preset(None if not target else [target] if isinstance(target, str) else [x for x in target if isinstance(x, str)], preset)
    except TapError as e:
        raise ToolError(str(e)) from None
    try:
        formats = parse_formats(edit_lists)
    except ValueError as e:
        raise ToolError(str(e)) from None
    target_obj = _parse_target(target, mode, preset)
    pad, min_len, merge_gap = _non_negative("pad", pad), _non_negative("min_len", min_len), _non_negative("merge_gap", merge_gap)
    if out is None or out == "":
        out_dir = video.with_name(video.stem + "_inframe") / mode
    else:
        out_dir = _abs_path(out, "out")
    if out_dir.exists() and not out_dir.is_dir():
        raise ToolError(f"out: exists and is not a folder: {out_dir}")
    await anyio.to_thread.run_sync(_probe, video)  # a bad file fails here, in one line, before any output folder is made

    try:
        zone_obj = Zone.parse(zone) if zone else zone_from_target_files(target if isinstance(target, list) else [target] if target else None)
        check_overlap(zone_overlap)
        check_activity(activity)
        if activity != "any" and mode != "detect":
            raise ValueError("activity needs mode detect")
        slowmo = parse_slowmo(slowmo)
    except ValueError as e:
        raise ToolError(str(e)) from None
    if slowmo_audio and slowmo == "off":
        raise ToolError("slowmo_audio needs slowmo set to auto or a factor")
    settings = Settings(mode=mode, preset=preset, pad=pad, min_len=min_len, merge_gap=merge_gap, reel=bool(reel), edit_lists=tuple(formats),
                        zone=tuple(zone_obj.as_list()) if zone_obj else None, zone_overlap=zone_overlap if zone_obj else None, activity=activity,
                        slowmo=slowmo, slowmo_audio=bool(slowmo_audio))
    doc = await _run_exclusive(lambda report: _run(video, out_dir, settings, report, target_obj), ctx)
    res = _cut_result(doc, out_dir)
    specs = [target] if isinstance(target, str) else list(target or [])
    saved = next((x[len("saved:"):] for x in specs if x.startswith("saved:")), None)
    if saved and doc["clips"]:  # offer to grow the saved target with what this run found
        res["saved_target"] = saved
        res["add_crops"] = f"add_target_crops(name={saved!r}, run={str(out_dir / 'segments.json')!r})"
    return res


@mcp.tool()
async def list_targets() -> list[dict[str, Any]]:
    """The saved targets: people kept by name (crops and embeddings of them, never video). Pass `saved:<name>` as the target of
    cut_video to keep only that person. Per target: `name`, `n_crops`, `created`, `updated`, `sources` (photos, tap, run), `embeddings`
    (stored embedder versions) and `path`. Add and remove targets with the `inframe target` command."""
    from . import targets

    return await anyio.to_thread.run_sync(targets.list_targets)


@mcp.tool()
async def add_target_crops(name: str, run: str, embedder: str = identity.DEFAULT_EMBEDDER) -> dict[str, Any]:
    """Add crops of the person a finished cut found to the saved target `name` (created when new). Use it after a cut_video run whose
    clips you confirmed show the right person. Crops that look like ones already kept are dropped, the set is capped, and different
    outfits stay. Returns how many were added.

    Args:
        name: the saved target (letters, digits, - and _).
        run: absolute path of the run's segments.json (or its folder); the run must have been made with a target, not every person.
        embedder: "dinov2" (default), "siglip2" or "color".
    """
    from . import targets

    p = _segments_file(run, "run")
    if embedder not in identity.SPECS:
        raise ToolError(f"embedder must be one of {sorted(identity.SPECS)} (got {embedder!r})")

    def job() -> dict:
        try:
            return targets.add_crops(name, targets.crops_from_run(str(p)), "run", embedder)
        except (targets.TargetError, identity.EmbedderUnavailable, models.ModelsMissing, ValueError, OSError) as e:
            raise ToolError(str(e)) from None

    return await anyio.to_thread.run_sync(job)


def _cut_result(doc: dict, out_dir: Path) -> dict[str, Any]:
    clips = [dict(index=c["index"], start=c["start"], end=c["end"], duration=round(c["end"] - c["start"], 3),
                  file=str(out_dir / c["file"]), peak_score=c["peak_score"], expected_copy_start=c["expected_copy_start"],
                  **({"sidecar": str(out_dir / c["sidecar"])} if c.get("sidecar") else {}),
                  **({"slowmo": c["slowmo"]} if c.get("slowmo") else {}))
             for c in doc["clips"]]
    stem = Path(doc["source"]["path"]).stem
    return dict(
        mode=doc["mode"],
        preset=doc["settings"]["preset_name"],
        out_dir=str(out_dir),
        segments_json=str(out_dir / "segments.json"),
        losslesscut_csv=str(out_dir / f"{safe_stem(stem)}_losslesscut.csv"),
        samples_csv=str(out_dir / "samples.csv"),
        source=_source_summary(doc["source"]),
        n_clips=len(clips),
        kept_s=round(sum(c["duration"] for c in clips), 2),
        clips=clips,
        presence_intervals=doc["presence_intervals"],
        warnings=doc["warnings"],
        timing_s=doc["timing_s"],
        **_export_result(doc, out_dir),
    )


def _export_result(doc: dict, out_dir: Path) -> dict[str, Any]:
    """The reel and edit lists with absolute paths; keys only when they were asked for."""
    res: dict[str, Any] = {}
    if doc.get("reel"):
        r = dict(doc["reel"])
        r["file"] = str(out_dir / r["file"]) if r.get("file") else None
        res["reel"] = r
    if doc.get("edit_lists"):
        res["edit_lists"] = [dict(r, file=str(out_dir / r["file"]) if r.get("file") else None) for r in doc["edit_lists"]]
    return res


def _segments_file(value: str, what: str) -> Path:
    p = _abs_path(value, what)
    if p.is_dir():
        p = p / "segments.json"
    return _existing_file(str(p), what)


@mcp.tool()
async def export_reel(run: str, music: str | None = None, music_rank: int = 0, beat_sync: bool | None = None, cc0_only: bool = False,
                      edit_lists: str | None = None, out: str | None = None, to_photos: bool = False,
                      ctx: Context | None = None) -> dict[str, Any]:
    """Build the reel of a finished cut_video run, optionally with music, and optionally edit lists. With music="auto" (or "auto:calm",
    a mood word) inframe picks a free song (CC0 or CC BY) that fits the reel, trims or loops it, fades it, sets it to -14 LUFS and, with
    librosa installed, puts the clip changes on beats. Only the mood word leaves the machine (the Openverse search text) and the track
    downloads; the clips and the reel's shape never do. A CC BY song's credit file is written next to the reel. The result's
    `reel.music.auto` says which song of how many was picked, and `another_song` is the call that moves to the next one.

    Args:
        run: absolute path of the run's segments.json, or of its folder (the clips are read from that folder).
        music: "auto", "auto:MOOD" (letters, digits, spaces, dashes; up to 40), a track id from Openverse, or an absolute path of an audio file.
            Default: the reel keeps the clips' own sound.
        music_rank: with music="auto": 0 is the best song, 1 is another song, and so on (0 to 63; past the end of the list it wraps).
        beat_sync: every clip change lands on a beat (the clips are re-encoded). Default: on for auto music when librosa is installed, else off.
        cc0_only: refuse CC BY music. Openverse has no CC0 music, so auto music then ends on inframe's built-in beat.
        edit_lists: comma separated edit list formats: fcpxml, fcp7, kdenlive, edl, otio, or "all".
        out: absolute output folder for the reel, credits and export.json; default the run folder.
        to_photos: also add the run's clip files and the reel to the macOS Photos library (library only, no album, the files stay where
            they are). It uses the add-only permission: macOS shows a dialog the first time. The result's `photos` says `saved`, `partial`,
            `denied` (permission refused: `message` says where to allow it), `failed`, or `interrupted` (the save timed out or the helper
            died partway: the listed files not confirmed may already be in Photos, so check before calling again, a repeat duplicates
            them), with the names saved and not saved. A denied or failed save never undoes the reel, and is also added to `warnings`.
            PRIVACY: with iCloud Photos on, Apple uploads everything saved to the library, so the footage leaves this Mac. Once the add-only
            permission has been granted this tool saves without any click: pass to_photos=true only when the person asked for Photos.
    """
    import json

    from . import automusic
    from . import music as music_mod
    from .export import export_run

    seg = _segments_file(run, "run")
    try:
        doc = json.loads(seg.read_text())
        doc["source"]["path"], doc["clips"]
    except (ValueError, KeyError, TypeError, OSError):
        raise ToolError(f"{seg} is not a segments.json written by inframe") from None
    if not isinstance(music_rank, int) or isinstance(music_rank, bool) or not 0 <= music_rank <= 63:
        raise ToolError(f"music_rank must be a whole number from 0 to 63 (got {music_rank!r})")
    mood = None
    if music is not None:
        try:
            mood = automusic.parse_spec(music)
        except ValueError as e:
            raise ToolError(f"music: {e}") from None
    if music_rank and mood is None:
        raise ToolError('music_rank needs music="auto"')
    if beat_sync and music is None:
        raise ToolError("beat_sync needs music")
    try:
        formats = parse_formats(edit_lists)
    except ValueError as e:
        raise ToolError(str(e)) from None
    out_dir = seg.parent if out is None else _abs_path(out, "out")
    if out_dir.exists() and not out_dir.is_dir():
        raise ToolError(f"out: exists and is not a folder: {out_dir}")
    music_path = _existing_file(music, "music") if music is not None and mood is None and Path(music).expanduser().is_absolute() else None
    beat = bool(beat_sync) if beat_sync is not None else (mood is not None and automusic.can_analyse())

    def job(report) -> tuple[dict, Any]:
        report(0.05, "choosing the music" if music is not None else "building the reel")
        track, auto = None, None
        try:
            if mood is not None:
                good = [c for c in doc["clips"] if "error" not in c]
                if good and all(c.get("file") and (seg.parent / c["file"]).is_file() for c in good):  # otherwise export_run says why the reel was not built
                    auto = automusic.choose(automusic.shape_from_files(doc, seg.parent), mood, cc0_only, music_rank)
                    track = (auto.audio, dict(auto.meta, auto=auto.summary()))
            elif music is not None:
                track = music_mod.resolve(str(music_path) if music_path else music, cc0_only=cc0_only)
        except music_mod.MusicError as e:
            raise ToolError(str(e)) from None
        except (RuntimeError, OSError) as e:
            raise ToolError(f"could not choose the music: {e}") from None
        report(0.2, "building the reel")
        ex = export_run(doc, seg.parent, out_dir, True, formats, track, beat)
        (out_dir / "export.json").write_text(json.dumps({k: v for k, v in ex.items() if k != "ok"}, indent=1))
        report(1.0, "done")
        return ex, auto

    ex, auto = await _run_exclusive(job, ctx)
    res: dict[str, Any] = dict(out_dir=str(out_dir), export_json=str(out_dir / "export.json"), ok=ex["ok"],
                               warnings=(auto.warnings if auto else []) + ex["warnings"], **_export_result(ex, out_dir))
    if auto:
        res["another_song"] = f"export_reel(run={str(seg)!r}, music={music!r}, music_rank={auto.rank + 1})"
    if to_photos:
        res["photos"] = await anyio.to_thread.run_sync(_save_to_photos, doc, seg.parent, out_dir, ex.get("reel"))
        if res["photos"]["status"] not in ("saved", "skipped"):
            res["warnings"] = list(res["warnings"]) + [res["photos"]["message"]]
    return res


def _save_to_photos(doc: dict, clips_dir: Path, out_dir: Path, reel: dict | None) -> dict[str, Any]:
    """The `photos` block of export_reel: the run's clips and the reel just built, added to the Photos library (add-only). Never raises."""
    from . import photosave

    files = photosave.videos_of_export(clips_dir, {"clips": doc.get("clips")}) + photosave.videos_of_export(out_dir, {"reel": reel})
    if not files:
        return {"status": "failed", "authorization": "unknown", "message": "No clip or reel file to save to Photos.", "saved": [], "failed": []}
    try:
        return photosave.save(files).to_json()
    except photosave.PhotosSaveError as e:
        return {"status": "failed", "authorization": "unknown", "message": f"Nothing was saved to Photos: {e}", "saved": [],
                "failed": [{"name": f.name, "error": str(e)} for f in files]}


@mcp.tool()
def compare_runs(a: str, b: str, min_len: float = 1.0, out: str | None = None) -> dict[str, Any]:
    """Compare two cut runs of the same video (for example detect and motion): seconds kept by both, only A, only B,
    and every stretch where they disagree for min_len seconds or more. Refuses two different videos.
    Writes compare.json and compare.md into `out` (default: the parent folder of A's folder) and returns the comparison.

    Args:
        a: absolute path of a segments.json, or of the run folder that holds it.
        b: the same for the second run.
        min_len: report disagreements at least this many seconds long (0 or more).
        out: absolute folder for compare.json and compare.md.
    """
    from .compare import compare_files

    pa, pb = _segments_file(a, "a"), _segments_file(b, "b")
    min_len = _non_negative("min_len", min_len)
    out_dir = _abs_path(out, "out") if out else None
    if out_dir is not None and out_dir.exists() and not out_dir.is_dir():
        raise ToolError(f"out: exists and is not a folder: {out_dir}")
    try:
        return compare_files(pa, pb, min_len, out_dir)
    except ValueError as e:
        raise ToolError(str(e)) from None
    except OSError as e:
        raise ToolError(f"could not write the comparison: {e}") from None


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
