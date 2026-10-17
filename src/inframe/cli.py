"""inframe command line."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import classes, identity, models, tracks
from .cut import parse_slowmo
from .editlist import EDL_MAX_FPS, FORMATS, parse_formats
from .activity import ACTIVITIES, THRESHOLD as ACTIVITY_THRESHOLD, check_activity
from .zone import Zone, check_overlap, zone_from_target_files
from .pipeline import DETECT_ENTER, MARKER_KINDS, MODES, PRESETS, SOUND_MODES, Settings, check_ranges, run_cut


def _slowmo_arg(v: str) -> str:
    try:
        return parse_slowmo(v)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e).removeprefix("--slowmo ")) from None


def _add_music_options(p: argparse.ArgumentParser, *, needs_reel: bool) -> None:
    """--music and its companions, the same on `cut` and `export`."""
    p.add_argument("--music", default=None, metavar="FILE|ID|auto[:MOOD]",
                   help=("with --reel: " if needs_reel else "implies --reel: ") + "the reel's only sound is this music (an audio file, a track id from `inframe music search`, "
                         "or auto = pick a free song that fits the reel, with a mood word after a colon, e.g. auto:calm), trimmed or looped, "
                         "faded in and out, at -14 LUFS; the credits file is written when the license asks for one. auto searches Openverse "
                         "with the mood word only (default: calm, upbeat or energetic, from how fast the clips change), and with librosa also beat-syncs")
    p.add_argument("--music-rank", type=int, default=None, metavar="N",
                   help="with --music auto: 0 (default) is the best song, 1 is another song, and so on (0 to 63; past the end of the list it wraps)")
    p.add_argument("--cc0-only", action="store_true",
                   help="with --music: refuse CC BY music and use CC0 only (default: CC0 and CC BY; a CC BY track's credits file is written next to the reel). "
                        "Openverse has no CC0 music, so --music auto then ends on inframe's built-in beat")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--beat-sync", action="store_true",
                   help="with --music: every clip change lands on a beat (each clip's tail is trimmed inside its padding; the clips are re-encoded "
                        "with hevc_videotoolbox at high quality). Needs the music extra (librosa). On by default with --music auto when librosa is installed")
    g.add_argument("--no-beat-sync", action="store_true", help="with --music auto: do not put the clip changes on beats")


def check_music_options(a: argparse.Namespace) -> str | None:
    """One-line message for the first bad --music option of `cut` or `export`, or None."""
    from . import automusic

    if a.beat_sync and not a.music:
        return "--beat-sync needs --music"
    try:
        auto = a.music is not None and automusic.parse_spec(a.music) is not None
    except ValueError as e:
        return f"--music: {e}"
    if a.music_rank is not None and not auto:
        return "--music-rank needs --music auto"
    if a.music_rank is not None and not 0 <= a.music_rank <= 63:
        return f"--music-rank must be 0 to 63 (got {a.music_rank})"
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inframe", description="Cut a video into the stretches where the target is in frame.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut", help="find the stretches where the target is in frame and stream-copy them to clips")
    c.add_argument("video", help="a video file, or photos:<uuid> for a video in the Photos library (see `inframe photos list`; "
                                 "the original is read in place and the output goes to ./<uuid>_inframe/<mode> unless --out is given)")
    c.add_argument("--library", default=None, help="photos:<uuid> only: the .photoslibrary to read (default: the last library Photos opened)")
    c.add_argument("--mode", choices=MODES, default="detect",
                   help="detect (default): RF-DETR person detector, any camera. motion: no model, static camera only")
    c.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="balanced (default), accurate or fast; with --target tap:<target.json> the default is the preset that `inframe pick` recorded there, and another detector is refused")
    c.add_argument("--fps", type=float, default=None, help="samples per second, more than 0; default from the preset (fast = keyframes only)")
    c.add_argument("--pad", type=float, default=1.0, help="seconds added before and after each stretch")
    c.add_argument("--min-len", type=float, default=1.0, help="drop stretches shorter than this (seconds)")
    c.add_argument("--merge-gap", type=float, default=1.5, help="join stretches closer than this (seconds)")
    c.add_argument("--enter", type=float, default=None, help="score that starts a stretch (motion: chosen from the video's own scores)")
    c.add_argument("--exit", type=float, default=None, help="score below which a stretch ends")
    c.add_argument("--min-box", type=float, default=0.0, help="detect: ignore persons (or the --target class) shorter than this fraction of frame height")
    c.add_argument("--tracker", choices=tracks.KINDS, default="ocsort",
                   help="identity between samples for targets that need it: ocsort, or botsort when the camera pans (runs at 5+ samples/s only)")
    c.add_argument("--target", action="append", default=None, metavar="SPEC",
                   help="who to keep, instead of every person: class:NAME (every dog, car, ... of the 80 COCO classes; with a tap below it follows that one dog), "
                        "tap:<target.json from `inframe pick`> or tap:T,x,y (seconds, then x and y as "
                        "fractions 0..1 of the frame). Repeat for each appearance of the same person. photos:<folder> follows the person in the photos "
                        "(add the confirm that `inframe pick --photos` wrote, or tap:T,x,y; alone it picks only above the embedder's margin, "
                        "else exit 3). saved:<name> uses a target kept with `inframe target add` (no tap; exit 3 when its thumbnails are gone). "
                        "detect mode only; exit 2 when nothing of the class was detected within 1 s of a tap")
    c.add_argument("--add-crops", action="store_true",
                   help="with --target saved:NAME: after the run, add deduplicated, outfit-diverse crops of the person it found to NAME "
                        "(without it the run only prints the command that does this)")
    c.add_argument("--zone", default=None, metavar="X,Y,W,H",
                   help="only people in this rectangle count: x,y,w,h as fractions 0..1 of the upright frame (0,0 = top left, as you see it), "
                        "box centre inside. Static camera only (a moving camera: warning, zone ignored). Motion mode counts changed pixels inside. "
                        "Default: the zone drawn on the `inframe pick` page, if the --target tap: file holds one")
    c.add_argument("--zone-overlap", type=float, default=None, metavar="F",
                   help="with a zone: a person counts when at least this fraction (0..1] of the box is inside, instead of the centre")
    c.add_argument("--activity", choices=ACTIVITIES, default="any",
                   help=f"riding: a person counts only while their box moves at {ACTIVITY_THRESHOLD:g} box heights per real second or more "
                        "(median over 1 s; detect mode, 2+ samples per second, static camera). any (default): standing and walking count too")
    c.add_argument("--activity-threshold", type=float, default=None, metavar="V", help="box heights per real second for --activity riding")
    c.add_argument("--slowmo-factor", type=float, default=1.0, metavar="F",
                   help="the file plays F times slower than real time (a baked slow-motion file): speeds are per real second")
    c.add_argument("--embedder", choices=sorted(identity.SPECS), default=identity.DEFAULT_EMBEDDER,
                   help="photo mode: how tracks are compared with the photos (siglip2 default, dinov2 faster, color needs no model)")
    c.add_argument("--offline", action="store_true", help="never download a model; fail with a clear message when one is missing")
    c.add_argument("--reel", action="store_true", help="also join the clips, in time order and without re-encoding, into <stem>_reel.<ext>")
    _add_music_options(c, needs_reel=False)
    c.add_argument("--edit-list", default=None, metavar="FORMATS",
                   help=f"also write edit lists of the cuts on the original video: {', '.join(FORMATS)} (comma separated) or all; "
                        f"edl only for sources of {EDL_MAX_FPS:g} fps or less")
    c.add_argument("--slowmo", type=_slowmo_arg, default="off", metavar="auto|4|8|off",
                   help="play the clips in slow motion by retiming them, never re-encoding the picture: a factor (4, 8), or auto = the file's "
                        "capture rate over 30 fps (240 fps gives 8, 120 gives 4, no change under 100 fps). One constant factor: the speed ramp "
                        "of a Photos slow-motion edit cannot be read. The clips last factor times longer; the audio is dropped unless --slowmo-audio. Default off")
    c.add_argument("--slowmo-audio", action="store_true",
                   help="with --slowmo: keep the audio, stretched to the slow-motion length with atempo (the audio, and only the audio, is re-encoded to AAC)")
    c.add_argument("--sound", choices=SOUND_MODES, default="off",
                   help="snap: move each try's start to just before a pop and its end to just after a landing, within 1.5 s and never "
                        "more than 1 s (clips still start on a keyframe); every clip then also gets `impacts` and a raised `confidence`. "
                        "No audio track: a warning, exit 0")
    c.add_argument("--markers", choices=[k for k in MARKER_KINDS if k], default="",
                   help="clap: a marker at every sound impact inside each clip (segments.json, sidecars, FCPXML and OTIO edit lists)")
    c.add_argument("--sound-speed", default="auto", metavar="auto|normal|FACTOR",
                   help="whether the audio is baked slow motion: auto decides from how fast impacts rise; a factor (4 = four times slower) "
                        "scales the sound search windows by it (the 1 s cap on an edge move stays)")
    c.add_argument("--out", default=None, help="output folder (default: <video stem>_inframe/<mode> next to the video)")
    pk = sub.add_parser("pick", help="open a local page to click the person to follow; writes target.json for `cut --target tap:`")
    pk.add_argument("video")
    pk.add_argument("--out", default=None, help="where to write target.json (default: <video stem>_inframe/target.json next to the video)")
    pk.add_argument("--photos", default=None, metavar="DIR|saved:NAME", help="rank the people in the video against the photos in DIR (or a saved target) and show the top 5 as \"Is this you?\"; one confirm is written into target.json")
    pk.add_argument("--embedder", choices=sorted(identity.SPECS), default=identity.DEFAULT_EMBEDDER, help="with --photos: the embedder")
    pk.add_argument("--class", dest="cls", default=None, metavar="NAME", help="pick an object of this COCO class (dog, car, ...) instead of a person; target.json records the class")
    pk.add_argument("--preset", choices=sorted(PRESETS), default="balanced", help="which detector finds the people: accurate = RF-DETR base (slower, finds smaller people)")
    pk.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: a free one)")
    pk.add_argument("--offline", action="store_true", help="never download a model; fail with a clear message when one is missing")
    pk.add_argument("--no-open", action="store_true", help="do not open the page in the browser; only print the address")
    sv = sub.add_parser("serve", help="open the review UI: a local web page (127.0.0.1 only) to open a video, pick the target, run and fix the clips")
    sv.add_argument("video", nargs="?", default=None, help="a video to open at start (default: the page asks)")
    sv.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: a free one)")
    sv.add_argument("--no-open", action="store_true", help="do not open the page in the browser; only print the one-use link")
    sv.add_argument("--offline", action="store_true", help="never download a model; fail with a clear message when one is missing")
    sv.add_argument("--keep-picked", action="store_true", help="keep the videos exported from Apple Photos in the temp folder when the server stops")
    m = sub.add_parser("fetch-models", help="download the RF-DETR checkpoints into the model cache (checksummed); needs a network connection")
    m.add_argument("names", nargs="*", help="file names (default: all); known: " + ", ".join(sorted(models.REGISTRY)))
    ph = sub.add_parser("photos", help="read the macOS Photos library (`list` needs Full Disk Access and `uv sync --extra photos`; `pick` needs neither)")
    phs = ph.add_subparsers(dest="photos_cmd", required=True)
    pp = phs.add_parser("pick", help="open Apple's photo picker, pick videos, and cut them (no Full Disk Access). "
                                     "Options of `inframe cut` go after a bare --, e.g. `inframe photos pick -- --target photos:DIR --reel`")
    pp.add_argument("--limit", type=int, default=0, help="most videos the picker accepts (default 0: no limit)")
    pp.add_argument("--export-dir", default=None, help="where the picked originals are exported (default: a temp folder, removed after the cuts)")
    pp.add_argument("--keep", action="store_true", help="keep the exported originals (with --export-dir they are always kept)")
    pp.add_argument("--no-cut", action="store_true", help="only pick and export, print the paths as JSON, do not cut (the exports are kept)")
    pl = phs.add_parser("list", help="list videos: uuid, date, slow-motion, missing (iCloud only), path of the original; nothing else is read")
    pl.add_argument("--since", default=None, help="only videos created on or after this date (YYYY-MM-DD or YYYY-MM-DDTHH:MM, local time)")
    pl.add_argument("--until", default=None, help="only videos created before this date")
    pl.add_argument("--slomo", action="store_true", help="only slow-motion videos")
    pl.add_argument("--json", action="store_true", help="print the records as JSON")
    pl.add_argument("--library", default=None, help="the .photoslibrary to read (default: the last library Photos opened)")
    x = sub.add_parser("export", help="build the reel and/or edit lists from a finished run's segments.json")
    x.add_argument("run", help="segments.json, or the run folder that holds it (the clips are read from that folder)")
    x.add_argument("--reel", action="store_true", help="join the clips into <stem>_reel.<ext>")
    x.add_argument("--edit-list", default=None, metavar="FORMATS", help=f"{', '.join(FORMATS)} (comma separated) or all")
    _add_music_options(x, needs_reel=True)
    x.add_argument("--allow-by", action="store_true", help="accepted and ignored: CC BY is allowed by default since 2026-10-16")
    x.add_argument("--out", default=None, help="folder for the outputs and export.json (default: the run folder)")
    x.add_argument("--to-photos", action="store_true", help="also add the run's clip files and the reel to the Photos library (macOS; add-only permission, "
                   "no album; the files stay where they are). Alone it saves just the clips. With iCloud Photos on, Apple uploads what is saved, so the footage "
                        "leaves this Mac")
    mu = sub.add_parser("music", help="free music for the reel: the songs that come with inframe (list), and more CC0 and CC BY tracks from Openverse (CC0 only with --cc0-only)")
    ms = mu.add_subparsers(dest="music_cmd", required=True)
    li_ = ms.add_parser("list", help="the songs that come with inframe (work offline): id, length, BPM, license, title; no network")
    li_.add_argument("--cc0-only", action="store_true", help="only CC0 songs")
    li_.add_argument("--json", action="store_true", help="print the list as JSON")
    s_ = ms.add_parser("search", help="list more tracks from Openverse (online) with their license")
    s_.add_argument("--mood", default=None, help="search words, e.g. upbeat, calm, energetic (default: upbeat)")
    s_.add_argument("--min-seconds", type=float, default=0.0, help="only tracks at least this long")
    s_.add_argument("--limit", type=int, default=10)
    g_ = ms.add_parser("get", help="download a track into the music cache with its license and attribution (JSON)")
    g_.add_argument("id")
    for q_ in (s_, g_):
        q_.add_argument("--cc0-only", action="store_true", help="CC0 tracks only (default: CC0 and CC BY; a CC BY track needs a credit line, written next to the reel)")
        q_.add_argument("--allow-by", action="store_true", help="accepted and ignored: CC BY is allowed by default since 2026-10-16")
    au = ms.add_parser("auto", help="list the songs auto music would pick for a finished run, best first (rank, fit, BPM, length, license), without building a reel; the songs that come with inframe are first and need no network")
    au.add_argument("run", help="segments.json, or the run folder that holds it (the clips are read from that folder)")
    au.add_argument("--mood", default=None, help="search word (default: calm, upbeat or energetic, from how fast the clips change)")
    au.add_argument("--rank", type=int, default=0, help="list up to this rank and mark it with * (0 to 63)")
    au.add_argument("--cc0-only", action="store_true", help="CC0 only; Openverse has no CC0 music, so the list is then inframe's built-in beat")
    au.add_argument("--json", action="store_true", help="print the list as JSON")
    t_ = sub.add_parser("target", help="saved targets: a person kept by name (crops and embeddings, no video), used as --target saved:NAME")
    ts = t_.add_subparsers(dest="target_cmd", required=True)
    ta = ts.add_parser("add", help="add crops of a person to target NAME (created when new); duplicates are dropped, the set is capped, outfits stay")
    ta.add_argument("name")
    ta.add_argument("--from", dest="sources", action="append", required=True, metavar="SRC",
                    help="photos:<folder> | tap:<target.json from `inframe pick` or list_people> | run:<segments.json of a run that kept one person>; repeat for several")
    ta.add_argument("--video", default=None, help="tap: the video the tap was made in, when target.json does not name it (or for tap:T,x,y)")
    ta.add_argument("--embedder", choices=sorted(identity.SPECS), default=identity.DEFAULT_EMBEDDER)
    ta.add_argument("--offline", action="store_true", help="never download a model; fail with a clear message when one is missing")
    tl = ts.add_parser("list", help="the saved targets")
    tl.add_argument("--json", action="store_true")
    tw = ts.add_parser("show", help="one target: its crops, where they came from, and which embedder versions are stored (stale = no embedder uses it now)")
    tw.add_argument("name")
    tw.add_argument("--json", action="store_true")
    tr = ts.add_parser("remove", help="delete target NAME (its crops and embeddings)")
    tr.add_argument("name")
    te = ts.add_parser("export", help="write target NAME to a zip (crops, embeddings, target.json) to move it to another machine")
    te.add_argument("name")
    te.add_argument("out", help="the zip file, or a folder (then NAME.inframe-target.zip)")
    ti = ts.add_parser("import", help="read a zip written by `target export`")
    ti.add_argument("archive")
    ti.add_argument("--name", default=None, help="save it under this name (default: the name inside)")
    ti.add_argument("--force", action="store_true", help="replace a target of that name")
    k = sub.add_parser("compare", help="compare two segments.json files; writes compare.json and compare.md into --out (default: A's parent folder); refuses two different videos")
    k.add_argument("a")
    k.add_argument("b")
    k.add_argument("--min-len", type=float, default=1.0, help="report disagreements at least this long (seconds)")
    k.add_argument("--out", default=None, help="folder for compare.json and compare.md (default: the parent of A's folder)")
    return p


def check_numbers(a: argparse.Namespace) -> str | None:
    """One-line message for the first bad number among the `cut` options, or None."""
    for flag, v, ok, need in (
        ("--pad", a.pad, a.pad >= 0, "0 or more"),
        ("--min-len", a.min_len, a.min_len >= 0, "0 or more"),
        ("--merge-gap", a.merge_gap, a.merge_gap >= 0, "0 or more"),
        ("--fps", a.fps, a.fps is None or a.fps > 0, "more than 0"),
    ):
        if not ok or (v is not None and not math.isfinite(v)):
            return f"{flag} must be {need} (got {v:g})"
    bad = check_ranges(a.enter, a.exit, a.min_box)
    if bad:
        return bad
    for flag, v in (("--activity-threshold", a.activity_threshold), ("--slowmo-factor", a.slowmo_factor)):
        if v is not None and not (math.isfinite(v) and v > 0):
            return f"{flag} must be more than 0 (got {v:g})"
    if a.activity != "any" and a.mode != "detect":
        return "--activity riding needs --mode detect"
    enter = a.enter if a.enter is not None else (DETECT_ENTER if a.mode == "detect" else None)
    if a.exit is not None and enter is not None and a.exit > enter:
        return f"--exit ({a.exit:g}) must not exceed --enter ({enter:g}{'' if a.enter is not None else ', the detect default'})"
    return None


def check_target(a: argparse.Namespace):
    """The parsed --target (None without one); ValueError or TapError says what is wrong. Cheap: no model is loaded."""
    if not a.target:
        return None
    if a.mode != "detect":
        raise ValueError("--target needs --mode detect")
    return identity.parse_targets(a.target, a.embedder, a.offline, identity.lazy_detect(PRESETS[a.preset or "balanced"].model))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cut_args: list[str] = []
    if argv[:2] == ["photos", "pick"] and "--" in argv:  # everything after a bare -- belongs to `inframe cut`
        i = argv.index("--")
        argv, cut_args = argv[:i], argv[i + 1:]
    a = build_parser().parse_args(argv)
    if a.cmd == "photos" and a.photos_cmd == "pick":
        return photos_pick_command(a, cut_args)
    if a.cmd == "fetch-models":
        try:
            for p in models.fetch_all(a.names or None, offline=False):
                print(p)
        except (ValueError, models.ModelsMissing, models.ChecksumError) as e:
            print(f"inframe: {e}", file=sys.stderr)
            return 2
        return 0
    if a.cmd == "serve":
        from .serve import run as serve_run

        try:
            return serve_run(a.video and str(Path(a.video).expanduser()), a.port, open_browser=not a.no_open, offline=a.offline,
                             keep_picked=a.keep_picked, announce=lambda m: print(m, flush=True))
        except OSError as e:
            print(f"inframe: {e}", file=sys.stderr)
            return 2
    if a.cmd == "pick":
        from .pick import serve

        video = Path(a.video).expanduser().resolve()
        if not video.is_file():
            print(f"inframe: no such file: {video}", file=sys.stderr)
            return 2
        out = Path(a.out).expanduser() if a.out else video.with_name(video.stem + "_inframe") / "target.json"
        models.set_offline(a.offline)
        try:
            cls = classes.class_id(a.cls) if a.cls else classes.PERSON
            code = serve(str(video), out, a.port, announce=lambda m: print(m, flush=True), photos=a.photos, embedder=a.embedder, offline=a.offline, cls=cls,
                         open_browser=not a.no_open, model=PRESETS[a.preset].model, preset_name=a.preset)
        except (ValueError, OSError, models.ModelsMissing, identity.EmbedderUnavailable) as e:
            print(f"inframe: {e}", file=sys.stderr)
            return 2
        print(f"wrote {out}" if code == 0 else "inframe: no tap was made; nothing written", file=sys.stdout if code == 0 else sys.stderr)
        return code
    if a.cmd == "compare":
        from .compare import compare_files

        for f in (a.a, a.b):
            if not Path(f).expanduser().is_file():
                print(f"inframe: no such file: {f}", file=sys.stderr)
                return 2
        if not math.isfinite(a.min_len) or a.min_len < 0:
            print(f"inframe: --min-len must be 0 or more (got {a.min_len:g})", file=sys.stderr)
            return 2
        try:
            d = compare_files(a.a, a.b, a.min_len, a.out)
        except ValueError as e:
            print(f"inframe: {e}", file=sys.stderr)
            return 2
        print(f"both {d['both_s']} s, only {d['a']['label']} {d['only_a_s']} s, only {d['b']['label']} {d['only_b_s']} s, "
              f"{d['n_disagreements']} disagreements of {a.min_len:g} s or more")
        print(f"wrote {d['compare_json']} and {d['compare_md']}")
        return 0
    if a.cmd == "export":
        return export_command(a)
    if a.cmd == "photos":
        return photos_command(a)
    if a.cmd == "music":
        return music_command(a)
    if a.cmd == "target":
        return target_command(a)
    try:
        formats = parse_formats(a.edit_list)
    except ValueError as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 2
    bad = check_numbers(a) or check_music_options(a)
    if bad:
        print(f"inframe: {bad}", file=sys.stderr)
        return 2
    if a.slowmo_audio and a.slowmo == "off":
        print("inframe: --slowmo-audio needs --slowmo auto or a factor", file=sys.stderr)
        return 2
    from . import photos

    rec = None
    if photos.is_photos_ref(a.video):
        try:
            rec = photos.resolve(photos.open_db(a.library), photos.parse_ref(a.video))
        except photos.PhotosError as e:
            print(f"inframe: {e}", file=sys.stderr)
            return 2
        video = Path(rec.path)
        if a.slowmo == "auto" and not rec.slow_mo:  # a plain 120 fps video in the library is not a slow-motion clip
            a.slowmo = "off"
            a.slowmo_audio = False
            print("inframe: --slowmo auto: Photos does not mark this video as slow motion, so the clips are not retimed", file=sys.stderr)
        out = Path(a.out).expanduser() if a.out else Path.cwd() / f"{rec.uuid}_inframe" / a.mode  # never inside the library
    else:
        video = Path(a.video).expanduser().resolve()
        if not video.is_file():
            print(f"inframe: no such file: {video}", file=sys.stderr)
            return 2
        out = Path(a.out).expanduser() if a.out else video.with_name(video.stem + "_inframe") / a.mode
    target = None
    from .tap import TapError, resolve_preset
    from .targets import StaleTarget

    try:
        a.preset = resolve_preset(a.target, a.preset)
    except TapError as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 2
    if a.target:
        try:
            target = check_target(a)
            if hasattr(target, "min_box"):
                target.min_box = a.min_box
        except StaleTarget as e:
            print(f"inframe: {e}", file=sys.stderr)
            return 3
        except (ValueError, TapError) as e:
            print(f"inframe: {e}", file=sys.stderr)
            return 2
    if a.add_crops and not any(x.startswith("saved:") for x in a.target or []):
        print("inframe: --add-crops needs --target saved:NAME", file=sys.stderr)
        return 2
    try:
        zone = Zone.parse(a.zone) if a.zone else zone_from_target_files(a.target)
        check_overlap(a.zone_overlap)
        if zone is None and a.zone_overlap is not None:
            raise ValueError("--zone-overlap needs --zone")
    except ValueError as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 2
    s = Settings(mode=a.mode, preset=a.preset, fps=a.fps, pad=a.pad, min_len=a.min_len, merge_gap=a.merge_gap,
                 enter=a.enter, exit=a.exit, min_box=a.min_box, tracker=a.tracker,
                 reel=a.reel and not a.music, edit_lists=tuple(formats),  # with --music the reel is built below, with the music
                 zone=tuple(zone.as_list()) if zone else None, zone_overlap=a.zone_overlap, activity=a.activity,
                 activity_threshold=a.activity_threshold, slowmo_factor=a.slowmo_factor,
                 slowmo=a.slowmo, slowmo_audio=a.slowmo_audio, sound=a.sound, markers=a.markers, sound_speed=a.sound_speed)
    models.set_offline(a.offline)
    try:
        doc = run_cut(str(video), str(out), s, target=target)
    except identity.Ambiguous as e:  # photo mode cannot decide alone: confirm with `inframe pick --photos`
        print(f"inframe: {e}", file=sys.stderr)
        return 3
    except (ValueError, models.ModelsMissing, identity.EmbedderUnavailable) as e:  # bad settings that only show once the thresholds are known; missing models
        print(f"inframe: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:  # a failed run: the message says what was saved
        print(f"inframe: {e}", file=sys.stderr)
        return 1
    factor = 1.0
    if rec is not None:
        from .probe import capture_fps

        photos.annotate(doc, rec, capture_fps(str(video)))
        (out / "segments.json").write_text(json.dumps(doc, indent=1))
        factor = doc["photos"]["time_map"]["factor"]
        print(f"photos:{rec.uuid} slow-motion={rec.slow_mo}; times below are source seconds"
              + (f", playback seconds = source x {factor:g}" if factor != 1 else ""))
    print(f"{len(doc['clips'])} clips in {out}")
    for c in doc["clips"]:
        print(f"  {c['file']}  {c['start']:.1f}-{c['end']:.1f} s  peak {c['peak_score']:.4g}"
              + (f"  (playback {c['playback_start']:.1f}-{c['playback_end']:.1f} s)" if factor != 1 else "")
              + (f"  slow motion x{c['slowmo']['factor']:g}: {c['slowmo']['duration_s']:.1f} s at {c['slowmo']['fps']:g} fps"
                 + (f", audio kept (A/V offset {c['slowmo']['av_offset_ms']:g} ms)" if c["slowmo"]["av_offset_ms"] is not None else ", no audio")
                 if c.get("slowmo") else ""))
    code = print_exports(doc.get("reel"), doc.get("edit_lists", []), out)
    if a.music:
        if doc["clips"]:
            code = _music_reel(doc, out, out, a, True, []) or code
        else:
            print("reel not built: no clips")
    saved = next((x[len("saved:"):] for x in a.target or [] if x.startswith("saved:")), None)
    if saved and doc["clips"]:
        code = offer_crops(saved, out, a) or code
    return code


def offer_crops(name: str, out: Path, a: argparse.Namespace) -> int:
    """After a run with saved:NAME: print the command that adds crops of the person it found, or (--add-crops) do it."""
    from . import targets

    run = f"run:{out / 'segments.json'}"
    if not a.add_crops:
        print(f"saved target {name}: to add crops of the person this run found, run `inframe target add {name} --from {run}` (or pass --add-crops)")
        return 0
    try:
        r = targets.add_crops(name, targets.crops_from_run(str(out / "segments.json")), "run", a.embedder, a.offline)
    except (targets.TargetError, identity.EmbedderUnavailable, ValueError) as e:
        print(f"inframe: crops were not added: {e}", file=sys.stderr)
        return 1
    print(f"saved target {name}: {r['added']} crops added ({r['dropped_duplicates']} duplicates dropped, {r['evicted']} replaced), {r['n_crops']} in all")
    return 0


def target_command(a: argparse.Namespace) -> int:
    from . import targets

    try:
        if a.target_cmd == "add":
            models.set_offline(a.offline)
            crops, labels = [], []
            for src in a.sources:
                c, label = targets.crops_from(src, a.video)
                crops += c
                labels.append(label)
            r = targets.add_crops(a.name, crops, ",".join(dict.fromkeys(labels)), a.embedder, a.offline)
            print(f"{'created' if r['created'] else 'updated'} saved target {r['name']}: {r['added']} of {r['offered']} crops added "
                  f"({r['dropped_duplicates']} duplicates dropped, {r['evicted']} older ones replaced{', ' + str(r['lost_thumbnails']) + ' missing ones dropped' if r['lost_thumbnails'] else ''}), {r['n_crops']} in all; embedder {r['embedder']}")
        elif a.target_cmd == "list":
            items = targets.list_targets()
            if a.json:
                print(json.dumps(items, indent=1))
            else:
                for it in items:
                    print(f"{it['name']}  {it['n_crops']} crops  updated {it.get('updated')}" if "error" not in it else f"{it['name']}  UNREADABLE: {it['error']}")
                print(f"{len(items)} saved targets in {targets.store_dir()}", file=sys.stderr)
        elif a.target_cmd == "show":
            s = targets.Saved(a.name)
            if a.json:
                print(json.dumps(dict(s.summary(), crops=s.crops), indent=1))
            else:
                sm = s.summary()
                print(f"{sm['name']}: {sm['n_crops']} crops, created {sm['created']}, updated {sm['updated']}\n  folder {sm['path']}")
                for c in s.crops:
                    print(f"  {c['file']}  from {c.get('source')}  added {c.get('added')}")
                for v, e in sm["embeddings"].items():
                    why = "" if e["current"] and e["valid"] else (" (STALE: no embedder uses this version now)" if not e["current"] else " (does not match the crops)")
                    print(f"  embeddings {v}{why}")
                if not sm["embeddings"]:
                    print("  no embeddings stored yet; the first run embeds the crops")
        elif a.target_cmd == "remove":
            targets.remove(a.name)
            print(f"removed saved target {a.name}")
        elif a.target_cmd == "export":
            print(targets.export_target(a.name, a.out))
        else:
            print(f"imported saved target {targets.import_target(a.archive, a.name, a.force)}")
    except targets.StaleTarget as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 3
    except (targets.TargetError, models.ModelsMissing, identity.EmbedderUnavailable, identity.PhotoError, ValueError, OSError) as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 2
    return 0


def photos_command(a: argparse.Namespace) -> int:
    from . import photos

    try:
        since = photos.parse_when(a.since) if a.since else None
        until = photos.parse_when(a.until) if a.until else None
        recs = photos.list_videos(photos.open_db(a.library), since, until, a.slomo)
    except photos.PhotosError as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps([r.to_dict() for r in recs], indent=1))
    else:
        for r in recs:
            print(f"{r.uuid}  {r.date}  {'slomo' if r.slow_mo else '-    '}  {'MISSING (iCloud only)' if r.ismissing else r.path}")
        print(f"{len(recs)} videos", file=sys.stderr)
    return 0


def photos_pick_command(a: argparse.Namespace, cut_args: list[str]) -> int:
    """Open the photo picker, then run the normal `inframe cut` on every exported original."""
    import shutil

    from . import picker
    from .tap import TapError

    if a.limit < 0:
        print("inframe: --limit must be 0 or more", file=sys.stderr)
        return 2
    try:  # check the cut options before the person has picked anything
        c = build_parser().parse_args(["cut", "placeholder.mov", *cut_args])
    except SystemExit:
        return 2
    if c.video != "placeholder.mov":
        print("inframe: put only `inframe cut` options after --, no video: the videos come from the picker", file=sys.stderr)
        return 2
    try:  # the same checks `cut` runs after its video is found, before a download of GBs from iCloud
        parse_formats(c.edit_list)
        bad = check_numbers(c)
        if bad:
            raise ValueError(bad)
        check_target(c)
    except (ValueError, TapError) as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 3 if type(e).__name__ == "StaleTarget" else 2
    keep = a.keep or a.no_cut or bool(a.export_dir)
    export = Path(a.export_dir).expanduser().resolve() if a.export_dir else picker.new_export_dir()
    try:  # from here on the export is removed on every way out (failure, Ctrl-C), unless it is kept
        try:
            res = picker.pick(export, a.limit, announce=lambda m: print(m, flush=True))
        except picker.PickerError as e:
            print(f"inframe: {e}", file=sys.stderr)
            return 2
        for e in res.errors:
            print(f"inframe: picker: {e}", file=sys.stderr)
        if res.cancelled or not res.files:
            print("inframe: nothing was picked", file=sys.stderr)
            return 1
        if a.no_cut:
            print(json.dumps({"files": [str(f) for f in res.files]}, indent=1))
            return 0
        worst = 0
        stems = [f.stem for f in res.files]
        for k, f in enumerate(res.files, 1):
            argv = ["cut", str(f), *cut_args]
            name = f.stem if stems.count(f.stem) == 1 else f"{f.stem}_{k}"  # equal names keep apart
            if c.out is None:  # the default output would sit next to the export, which is deleted afterwards
                argv += ["--out", str(Path.cwd() / f"{name}_inframe" / c.mode)]
            elif len(res.files) > 1:
                argv = _with_out(argv, str(Path(c.out).expanduser() / name))
            print(f"[{k}/{len(res.files)}] cutting {f.name}", flush=True)
            code = main(argv)
            worst = worst or code
        return worst
    finally:
        if not keep:
            shutil.rmtree(export, ignore_errors=True)
        elif export.exists():
            print(f"exported originals kept in {export}")


def _with_out(argv: list[str], out: str) -> list[str]:
    """argv with the value of --out replaced (handles `--out X` and `--out=X`)."""
    res, skip = [], False
    for i, x in enumerate(argv):
        if skip:
            skip = False
        elif x == "--out":
            skip = True
        elif not x.startswith("--out="):
            res.append(x)
    return res + ["--out", out]


def print_exports(reel: dict | None, edit_lists: list[dict], out: Path) -> int:
    """Print one line per reel / edit list; exit code 1 if one failed its check."""
    bad = False
    if reel:
        ck = reel.get("check") or {}
        mu = reel.get("music")
        if reel.get("file") and mu:
            ck = reel.get("check") or {}
            print(f"reel {out / reel['file']}: {'ok' if reel['ok'] else 'FAILED'}, {reel['n_clips']} clips, music only ({mu.get('file')}), "
                  f"{ck.get('lufs')} LUFS, {ck.get('video_s')} s"
                  + (f", beat sync: largest join error {ck.get('max_join_err_frames')} pictures, {mu.get('tempo')} BPM" if mu.get("beat_sync") else "")
                  + (f", credits in {out / mu['credits']}" if mu.get("credits") else ""))
            for pr in ck.get("problems", [])[:3]:
                print(f"  {pr}")
        elif reel.get("file"):
            print(f"reel {out / reel['file']}: {'ok' if reel['ok'] else 'FAILED'}, {reel['n_clips']} clips, {ck.get('duration_s', 0):.1f} s, "
                  f"largest A/V offset at a join {ck.get('max_av_offset_ms', 0):.1f} ms"
                  + (", joined with -avoid_negative_ts make_zero" if reel.get("fallback_used") else ""))
        else:
            print(f"reel not written: {reel.get('problem')}")
        bad |= not reel.get("ok")
    for r in edit_lists:
        if r.get("skipped"):
            print(f"{r['format']}: {r['skipped']}")
        elif r.get("file"):
            print(f"{r['format']} {out / r['file']}: {'ok' if r['ok'] else 'FAILED ' + str(r['problem'])}, "
                  f"largest cut error {r['max_err_frames']} frames")
        else:
            print(f"{r['format']}: {r.get('problem')}")
        bad |= r.get("ok") is False
    return 1 if bad else 0


def music_auto_command(a: argparse.Namespace) -> int:
    from . import automusic, music

    seg = Path(a.run).expanduser()
    seg = seg / "segments.json" if seg.is_dir() else seg
    if not 0 <= a.rank <= 63:
        print(f"inframe: --rank must be 0 to 63 (got {a.rank})", file=sys.stderr)
        return 2
    try:
        mood = automusic.parse_spec("auto:" + a.mood) if a.mood else ""
        doc = json.loads(seg.read_text())
        shape = automusic.shape_from_files(doc, seg.parent)
        r = automusic.rank_list(shape, mood, a.cc0_only, a.rank)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"inframe: {seg}: {e}", file=sys.stderr)
        return 2
    except (music.MusicError, RuntimeError) as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 1
    chosen = a.rank % len(r.entries)
    items = []
    for k, e in enumerate(r.entries):
        m = e.meta
        items.append(dict(rank=k, chosen=k == chosen, id=m.get("id"), bundled=bool(m.get("bundled")), score=e.score.get("score", e.score.get("J")), bpm=e.bpm,
                          duration_s=m.get("duration_s"), license=m.get("license"), title=m.get("title"), creator=m.get("creator"),
                          builtin=e.builtin))
    for w in r.warnings:
        print(f"[WARNING] {w}", file=sys.stderr)
    if a.json:
        print(json.dumps(dict(mood=r.mood, speed=r.speed, n=r.n, complete=r.complete, songs=items, warnings=r.warnings), indent=1))
        return 0
    for it in items:
        d = int(it["duration_s"] or 0)
        print(f"{'*' if it['chosen'] else ' '} {it['rank']:2d}  {it['score'] if it['score'] is not None else '-'}  "
              f"{'%.0f' % it['bpm'] if it['bpm'] else '-'} BPM  {d // 60}:{d % 60:02d}  {str(it['license']).upper()}  {it['title']} - {it['creator']}"
              f"{'  [included]' if it['bundled'] else ''}")
    print(f"{len(items)} songs (mood {r.mood}, {r.speed} reel{'' if r.complete else ', more not fetched yet'}). {music.OPENVERSE_NOTICE}")
    return 0


def music_list_command(a: argparse.Namespace) -> int:
    """The bundled songs (musiclib): offline, no request."""
    from . import musiclib

    items = musiclib.listing(a.cc0_only)
    if a.json:
        print(json.dumps(dict(songs=items, total_bytes=sum(i["bytes"] or 0 for i in items)), indent=1))
        return 0
    for i in items:
        d = int(i["duration_s"])
        lic = "CC0" if i["license"] == "cc0" else f"CC BY {i['license_version'] or ''}".strip()
        print(f"{i['id']}  {d // 60}:{d % 60:02d}  {i['bpm']:5.1f} BPM  {lic:<9} {i['title']} - {i['creator']}  [{', '.join(i['moods'])}]")
    print(f"{len(items)} songs come with inframe ({sum(i['bytes'] or 0 for i in items) / 1e6:.1f} MB, offline; credits in the package's music_library/CREDITS.md). "
          "Use one with: inframe export RUN --reel --music ID  (a CC BY song's credit line is written next to the reel). More songs online: inframe music search")
    return 0


def music_command(a: argparse.Namespace) -> int:
    from . import music

    if a.music_cmd == "auto":
        return music_auto_command(a)
    if a.music_cmd == "list":
        return music_list_command(a)
    try:
        if a.music_cmd == "search":
            if not math.isfinite(a.min_seconds) or a.min_seconds < 0 or a.limit < 1:
                print("inframe: --min-seconds must be 0 or more and --limit 1 or more", file=sys.stderr)
                return 2
            tracks = music.search(a.mood, a.min_seconds, a.limit, cc0_only=a.cc0_only)
            for t in tracks:
                print(f"{t.id}  {int(t.duration_s) // 60}:{int(t.duration_s) % 60:02d}  {t.license.upper()}{' ' + t.license_version if t.license_version else ''}  "
                      f"{t.title} - {t.creator}")
            print(f"{len(tracks)} tracks ({'CC0 only' if a.cc0_only else 'CC0 and CC BY'}). {music.OPENVERSE_NOTICE}")
            return 0
        path = music.get(a.id, cc0_only=a.cc0_only)
    except music.MusicError as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 1
    meta = music.load_meta(path) or {}
    print(path)
    print(f"license: CC {str(meta.get('license', '?')).upper()} {meta.get('license_version') or ''}; {meta.get('attribution') or ''}".rstrip())
    return 0


def export_command(a: argparse.Namespace) -> int:
    from .export import export_run

    try:
        formats = parse_formats(a.edit_list)
    except ValueError as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 2
    if not a.reel and not formats and not a.to_photos:
        print("inframe: nothing to do; pass --reel, --edit-list and/or --to-photos", file=sys.stderr)
        return 2
    if a.music and not a.reel:
        print("inframe: --music needs --reel", file=sys.stderr)
        return 2
    bad = check_music_options(a)
    if bad:
        print(f"inframe: {bad}", file=sys.stderr)
        return 2
    seg = Path(a.run).expanduser()
    if seg.is_dir():
        seg = seg / "segments.json"
    if not seg.is_file():
        print(f"inframe: no such file: {seg}", file=sys.stderr)
        return 2
    try:
        doc = json.loads(seg.read_text())
        doc["source"]["path"], doc["clips"]
    except (ValueError, KeyError, TypeError):
        print(f"inframe: {seg} is not a segments.json written by inframe", file=sys.stderr)
        return 2
    out = Path(a.out).expanduser() if a.out else seg.parent
    if a.music:
        code = _music_reel(doc, seg.parent, out, a, a.reel, formats)
    elif a.reel or formats:
        ex = export_run(doc, seg.parent, out, a.reel, formats, None, False)
        (out / "export.json").write_text(json.dumps({k: v for k, v in ex.items() if k != "ok"}, indent=1))
        for w in ex["warnings"]:
            print(f"[WARNING] {w}", file=sys.stderr)
        code = print_exports(ex["reel"], ex["edit_lists"], out)
    else:
        code = 0
    if a.to_photos and code != 2:  # 2 = the request was refused before anything was built
        code = max(code, _save_to_photos(doc, seg.parent, out, a.reel))
    return code


def _save_to_photos(doc: dict, clips_dir: Path, out: Path, reel: bool) -> int:
    """`--to-photos`: add the run's clip files (from `clips_dir`) and the reel just built (in `out`) to the Photos library.
    Prints one line and returns 0 when everything was saved, else 1 (denied permission, nothing to save, a failed file)."""
    from . import photosave

    files = photosave.videos_of_export(clips_dir, {"clips": doc.get("clips")})
    if reel:
        try:
            files += photosave.videos_of_export(out, {"reel": json.loads((out / "export.json").read_text()).get("reel")})
        except (OSError, ValueError, AttributeError):
            pass
    if not files:
        print("inframe: --to-photos: no clip or reel file to save", file=sys.stderr)
        return 1
    try:
        res = photosave.save(files, announce=lambda m: print(m, flush=True))
    except photosave.PhotosSaveError as e:
        print(f"inframe: --to-photos: {e}", file=sys.stderr)
        return 1
    print(photosave.describe(res), file=sys.stderr if not res.ok else sys.stdout)
    for p, e in res.failed:
        print(f"  not saved: {p.name}: {e}", file=sys.stderr)
    return 0 if res.ok else 1


def _music_reel(doc: dict, clips_dir: Path, out: Path, a: argparse.Namespace, reel: bool, formats: list[str]) -> int:
    """`--music` of `export` and `cut`: choose the music (a file, a track id, or auto), build the reel with it, write export.json and print.
    Returns the exit code."""
    from . import automusic, music
    from .export import export_run

    track, auto = None, None
    beat = a.beat_sync
    try:
        mood = automusic.parse_spec(a.music)
        if mood is None:
            track = music.resolve(a.music, cc0_only=a.cc0_only)
        else:
            beat = False if a.no_beat_sync else (a.beat_sync or automusic.can_analyse())
            good = [c for c in doc["clips"] if "error" not in c]
            if good and all(c.get("file") and (Path(clips_dir) / c["file"]).is_file() for c in good):  # otherwise export_run says why the reel was not built
                auto = automusic.choose(automusic.shape_from_files(doc, clips_dir), mood, a.cc0_only, a.music_rank or 0)
                track = (auto.audio, dict(auto.meta, auto=auto.summary()))
                for w in auto.warnings:
                    print(f"[WARNING] {w}", file=sys.stderr)
    except music.MusicError as e:
        print(f"inframe: {e}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as e:  # a clip file ffprobe cannot read
        print(f"inframe: could not choose the music: {e}", file=sys.stderr)
        return 1
    ex = export_run(doc, clips_dir, out, reel, formats, track, beat)
    (out / "export.json").write_text(json.dumps({k: v for k, v in ex.items() if k != "ok"}, indent=1))
    for w in ex["warnings"]:
        print(f"[WARNING] {w}", file=sys.stderr)
    code = print_exports(ex["reel"], ex["edit_lists"], out)
    if auto and (ex["reel"] or {}).get("file"):
        m, lic = auto.meta, str(auto.meta.get("license") or "?")
        fit = auto.score.get("score", auto.score.get("J"))
        label = "CC0" if lic == "cc0" else f"CC {lic.upper()}" + (f" {m['license_version']}" if m.get("license_version") else "")
        print(f'music: auto pick {auto.rank + 1} of {auto.n} ("{m.get("title")}" by {m.get("creator")}, {label}'
              + (f", {auto.bpm:.0f} BPM" if auto.bpm else "") + (f", fit {fit:.2f}" if fit is not None else "")
              + f"); another song: --music-rank {auto.rank + 1}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
