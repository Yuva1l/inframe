#!/usr/bin/env python3
"""Spliced test composites with composition truth, and scoring against them. No labelling, no looking at footage.

    uv run python tools/splice.py build [--out DIR] [--only NAME ...]   write composites + <name>.truth.json
    uv run python tools/splice.py score TRUTH.json SEGMENTS.json        time-weighted precision / recall of kept seconds
    uv run python tools/splice.py ids [--out DIR] [--fps 2 5 10]        ID switches of the tracker on the composites
    uv run python tools/splice.py tap [--out DIR] [--one-tap]           tap mode on the composites: precision, recall, ID switches
    uv run python tools/splice.py photos [--out DIR]                    "photos of the target": crops from target clips that no composite uses
    uv run python tools/splice.py photo [--out DIR] [--embedder E] [--runs N]
                                                                        photo mode on the composites: ranking, one confirm, and the no-UI pick;
                                                                        with --runs the median and the spread over N runs

Precision and recall of the tap and photo runs are tile-aware (score_tile): a kept second counts only when the run's box lies in
the target tile, so keeping a stranger while he is also in frame earns nothing. The ceilings are oracle runs (oracle_dets: every detection in
the target tile is him; oracle_tracks: every track that is mostly in the target tile is him), not "any person".

A composite is ffmpeg stacks (single, side by side, 2x2, picture in picture) of the approved clips of the target person
(read in place) with openly licensed clips of other people (docs/TESTDATA.md), cut together. The truth is the
composition itself: for every second, which tile holds the target and where. Composites are written outside the repo
(default ~/.cache/inframe/splice) and are never committed or uploaded.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("INFRAME_MODELS", str(ROOT / "bench/detect/out/models"))
from inframe import localcfg  # noqa: E402

TARGET_DIR = localcfg.get("splice_target_dir")  # the clips of the target person, read in place (inframe.localcfg)
OTHERS_DIR = localcfg.get("dev_samples")  # openly licensed, see docs/TESTDATA.md
TESTDATA_DIR = Path.home() / ".cache/inframe/testdata"  # openly licensed clips fetched for T1b (Wikimedia Commons), see docs/TESTDATA.md
DEFAULT_OUT = Path.home() / ".cache/inframe/splice"
PRESET = "balanced"  # the detector preset the tap and photo measurements run (--preset); taps are picked with the same detector
W, H, FPS = 1280, 720, 30

# short key -> file (target: the clip numbered by index inside TARGET_DIR, found as *_<index>_*.mov; others: path inside OTHERS_DIR)
TARGET_CLIPS = {"targetA": "01", "targetB": "03", "targetC": "05"}
# Where the target is inside each target clip file, in file seconds: (file duration, first, last second it is in frame); the rest of the
# file is padding. Read from target_windows.json inside TARGET_DIR (below), not kept in the repo: {"targetA": [duration, first, last], ...}.
TARGET_WINDOW: dict[str, tuple[float, float, float]] | None = None  # None until target_windows() has read the file
COMPOSITE_CLASS = {"dog_among_dogs": "dog", "dog_twins": "dog"}  # composites whose target is not the target person but one object of a COCO class
OTHERS = {
    "pair1": "jump/cmj_squatjump_front_2people_pexels_8857689_danilyuk.mp4",       # 2 people
    "pair2": "jump/cmj_squatjump_front_2people_pexels_8858442_danilyuk.mp4",       # 2 people
    "bystanders": "skateboard/kickflip/hippyjump_bystanders_slowmo_commons.webm",  # skater and bystanders
    "lone1": "skateboard/ollie_front_pexels_2791955_cottonbro.mp4",                # 1 person, vertical
    "lone2": "skateboard/kickflip/kickflip_front_pexels_4824363_podrez.mp4",       # 1 person
}


# more others, in TESTDATA_DIR (docs/TESTDATA.md lists author and license)
MORE_OTHERS = {
    "dogs1": "Goldendoodle_and_Black_Lab_puppies_playing_in_slow_motion.webm",  # two dogs, almost no person
    "dogs2": "Two_poodles_playing.ogv",                                         # two dogs, almost no person
    "cars1": "Bangkok_Traffic_During_Enduring_Partners_26.webm",                # street with cars, a few people
    "crowd1": "Calle_preciados.ogv",                                            # busy street, 9 people on average, static camera
    "pan1": "Dog_walking_in_the_Mission_in_San_Francisco.ogv",                  # camera follows a walker with a dog: a long pan
    "pan2": "Central_Park_Connaught_Place_175602.webm",                         # handheld pan over a crowd, 5 s
}


def T(src, role="other", ss=0.0, hue=None):
    return dict(src=src, role=role, ss=ss, hue=hue)


# layouts: tile rectangles (x, y, w, h) in the 1280x720 composite; later tiles are drawn over earlier ones
LAYOUTS = {
    "full": [(0, 0, W, H)],
    "side": [(0, 0, W // 2, H), (W // 2, 0, W // 2, H)],
    "quad": [(0, 0, W // 2, H // 2), (W // 2, 0, W // 2, H // 2), (0, H // 2, W // 2, H // 2), (W // 2, H // 2, W // 2, H // 2)],
    "pip": [(0, 0, W, H), (W - 340, H - 200, 320, 180)],
    "pip_big": [(0, 0, W, H), (W - 660, H - 380, 640, 360)],
}

# name -> list of (layout, seconds, tiles); the tile with role "target" is the target person
COMPOSITES = {
    "solo_gaps": [
        ("full", 10, [T("targetA", "target", 2)]), ("full", 8, [T("lone1")]), ("full", 10, [T("targetB", "target", 2)]),
        ("full", 8, [T("lone2")]), ("full", 10, [T("targetC", "target", 2)]),
    ],
    "two_side": [
        ("side", 12, [T("targetA", "target", 4), T("pair1")]), ("full", 6, [T("pair2")]),
        ("side", 12, [T("pair2"), T("targetB", "target", 4)]), ("full", 8, [T("targetC", "target", 4)]),
    ],
    "swap_sides": [
        ("side", 8, [T("targetA", "target", 2), T("pair1")]), ("side", 8, [T("pair1", ss=1), T("targetA", "target", 10)]),
        ("side", 8, [T("targetA", "target", 4), T("bystanders")]), ("side", 8, [T("bystanders", ss=1), T("targetA", "target", 12)]),
    ],
    "quad": [
        ("quad", 14, [T("targetB", "target", 2), T("pair1"), T("lone1"), T("bystanders")]),
        ("quad", 14, [T("pair2"), T("lone2"), T("bystanders", ss=1), T("targetB", "target", 6)]),
        ("quad", 8, [T("pair1"), T("pair2"), T("lone1"), T("lone2")]),
    ],
    "pip_small": [
        ("pip", 12, [T("bystanders"), T("targetC", "target", 2)]), ("full", 6, [T("targetC", "target", 8)]),
        ("pip", 12, [T("pair1"), T("targetC", "target", 6)]),
    ],
    "outfit_changed": [  # target clip colour-shifted (hue +140) as if the target wore something else than in a reference photo
        ("full", 8, [T("targetA", "target", 2)]), ("full", 6, [T("lone1")]),
        ("full", 10, [T("targetA", "target", 8, hue=140)]), ("side", 10, [T("targetB", "target", 4, hue=140), T("pair2")]),
    ],
    # T1b: dogs, cars, a crowd and pans (clips in TESTDATA_DIR); "tracked" tiles only feed the ID-switch count
    "dogs_cars": [
        ("side", 10, [T("targetA", "target", 3), T("dogs1")]), ("full", 6, [T("cars1")]),
        ("side", 10, [T("cars1", ss=4), T("targetB", "target", 3)]), ("full", 6, [T("dogs2")]),
        ("quad", 10, [T("dogs1", ss=8), T("targetC", "target", 3), T("cars1", ss=10), T("dogs2", ss=6)]),
    ],
    "crowd_quad": [
        ("quad", 12, [T("targetA", "target", 5), T("crowd1"), T("pan1"), T("pair1")]),
        ("quad", 12, [T("crowd1", ss=6), T("pan2"), T("targetB", "target", 6), T("bystanders")]),
        ("side", 10, [T("crowd1", ss=3), T("targetC", "target", 6)]),
    ],
    "pan_pip": [
        ("pip_big", 14, [T("pan1"), T("targetB", "target", 2)]), ("full", 4.9, [T("pan2")]), ("pip_big", 10, [T("pan1", ss=3), T("targetA", "target", 6)]),
    ],
    # T8a: one dog among other dogs. The target tile is pan1 (a walker with one dog on a leash, camera follows); the others are dogs1
    # (a goldendoodle and a black lab) and dogs2 (two poodles). Truth is the tile: the clip of the dog that gets tapped.
    "dog_among_dogs": [
        ("side", 12, [T("pan1", "target", 0), T("dogs1")]), ("full", 8, [T("dogs2")]),
        ("side", 12, [T("dogs2", ss=3), T("pan1", "target", 4)]), ("quad", 12, [T("dogs1", ss=6), T("dogs2", ss=8), T("pan1", "target", 2), T("cars1", ss=2)]),
    ],
    # the same dog twice side by side, the right one colour-shifted (hue +140) and seven seconds later: two instances of nearly one look
    "dog_twins": [
        ("side", 14, [T("pan1", "target", 0), T("pan1", ss=7, hue=140)]), ("side", 12, [T("pan1", ss=2, hue=140), T("pan1", "target", 3)]),
    ],
    "pan_follow": [  # no target: the panning clips alone, for the BoT-SORT ID-switch count
        ("full", 16.5, [T("pan1", "tracked")]), ("full", 4.5, [T("pan2", "tracked")]),
    ],
}


def ffprobe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def target_windows() -> dict[str, tuple[float, float, float]]:
    """{key: (file duration, first, last second the target is in frame)} of the approved target clips, from TARGET_DIR/target_windows.json."""
    global TARGET_WINDOW
    if TARGET_WINDOW is None:
        if TARGET_DIR is None:
            raise SystemExit("splice: no target clips configured (INFRAME_SPLICE_TARGET_DIR or splice_target_dir in ~/.config/inframe/local.json)")
        f = TARGET_DIR / "target_windows.json"
        try:
            raw = json.loads(f.read_text())
            TARGET_WINDOW = {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in raw.items()}
        except (OSError, ValueError, TypeError, IndexError, AttributeError):
            raise SystemExit(f'splice: {f} is missing or malformed; it holds {{"targetA": [file seconds, first, last second in frame], ...}}') from None
    return TARGET_WINDOW


def target_clip(index: str) -> Path:
    """The target clip numbered `index` ("01", "03", ...) inside TARGET_DIR: the one file named *_<index>_*.mov."""
    if TARGET_DIR is None:
        raise SystemExit("splice: no target clips configured (INFRAME_SPLICE_TARGET_DIR or splice_target_dir in ~/.config/inframe/local.json)")
    found = sorted(TARGET_DIR.glob(f"*_{index}_*.mov"))
    if len(found) != 1:
        raise SystemExit(f"splice: expected one target clip *_{index}_*.mov in {TARGET_DIR}, found {len(found)}")
    return found[0]


def sources_available() -> bool:
    """Whether every clip the composites use is on this machine (target clips, sample clips, fetched clips)."""
    needed = {t["src"] for recipe in COMPOSITES.values() for _, _, tiles in recipe for t in tiles}
    try:
        return all(source_path(k).is_file() for k in needed)
    except SystemExit:
        return False


def source_path(key: str) -> Path:
    if key in TARGET_CLIPS:
        return target_clip(TARGET_CLIPS[key])
    if key in MORE_OTHERS:
        return TESTDATA_DIR / MORE_OTHERS[key]
    if OTHERS_DIR is None:
        raise SystemExit("splice: no sample clips configured (INFRAME_DEV_SAMPLES or dev_samples in ~/.config/inframe/local.json; docs/TESTDATA.md)")
    return OTHERS_DIR / OTHERS[key]


def segment_command(layout: str, secs: float, tiles: list[dict], out: Path) -> list[str]:
    rects = LAYOUTS[layout]
    assert len(rects) == len(tiles), (layout, len(tiles))
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    for t in tiles:
        cmd += ["-stream_loop", "-1", "-ss", f"{t['ss']:.3f}", "-t", f"{secs:.3f}", "-i", str(source_path(t["src"]))]
    parts = []
    for i, (t, (_, _, w, h)) in enumerate(zip(tiles, rects)):
        hue = f",hue=h={t['hue']}" if t.get("hue") else ""
        parts.append(f"[{i}:v]fps={FPS},scale={w}:{h}:force_original_aspect_ratio=decrease,"
                     f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1{hue}[t{i}]")
    if layout == "full":
        parts.append("[t0]null[v]")
    elif layout == "side":
        parts.append("[t0][t1]hstack[v]")
    elif layout == "quad":
        parts += ["[t0][t1]hstack[a]", "[t2][t3]hstack[b]", "[a][b]vstack[v]"]
    elif layout in ("pip", "pip_big"):
        x, y = rects[1][:2]
        parts.append(f"[t0][t1]overlay={x}:{y}[v]")
    return cmd + ["-filter_complex", ";".join(parts), "-map", "[v]", "-t", f"{secs:.3f}", "-an",
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-g", str(FPS), "-pix_fmt", "yuv420p",
                  "-r", str(FPS), str(out)]


def merge(iv: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def present_in_shot(src: str, ss: float, secs: float) -> tuple[list[list[float]], list[float]]:
    """(seconds of the shot that show him, seconds where the clip restarts), both relative to the shot start.
    The tile plays the target clip from `ss`; `-stream_loop -1` restarts it at 0 when the file ends, which is a hard cut.
    The target is in frame only inside its target window; the padding at either end of the file does not count."""
    dur, a, b = target_windows()[src] if src in TARGET_CLIPS else (ffprobe_duration(source_path(src)), 0.0, ffprobe_duration(source_path(src)))
    plays, x, p = [], 0.0, ss  # (shot time, file time, length) of every pass over the file
    while x < secs - 1e-9:
        n = min(dur - p, secs - x)
        plays.append((x, p, n))
        x, p = x + (dur - p), 0.0
    iv = [[x0 + max(a, p0) - p0, x0 + min(b, p0 + n) - p0] for x0, p0, n in plays if min(b, p0 + n) > max(a, p0)]
    cuts = [x0 for x0, _, _ in plays[1:]]
    return [[round(u, 6), round(v, 6)] for u, v in iv], [round(c, 6) for c in cuts]


def shot_starts(truth: dict) -> list[float]:
    """Where the picture of a tile changes abruptly: the start of every shot and every restart of a looped clip.
    A tracker starts afresh at each; a real clip has no hard cuts."""
    return sorted({sg["t0"] for sg in truth["segments"]} | set(truth.get("cuts", [])))


def truth_for(name: str, recipe) -> dict:
    """The composition truth: segments with tile rectangles and roles, and the seconds in which he is in frame
    (not the padding of the target clips, nor the restart of a clip that is shorter than the shot)."""
    t, segs, target, cuts = 0.0, [], [], []
    for layout, secs, tiles in recipe:
        segs.append(dict(t0=t, t1=t + secs, layout=layout, tiles=[
            dict(id=i, src=tl["src"], role=tl["role"], rect=list(LAYOUTS[layout][i]), source_start=tl["ss"], hue_shift=tl.get("hue"))
            for i, tl in enumerate(tiles)]))
        for tl in tiles:
            if tl["role"] == "target":
                iv, c = present_in_shot(tl["src"], tl["ss"], secs)
                target += [[t + u, t + v] for u, v in iv]
                cuts += [round(t + x, 6) for x in c]
        t += secs
    return dict(name=name, video=f"{name}.mp4", fps=FPS, width=W, height=H, duration=t, segments=segs,
                target_intervals=merge(target), cuts=sorted(set(cuts)), **({"class": COMPOSITE_CLASS[name]} if name in COMPOSITE_CLASS else {}),
                note="Truth is the composition itself. Tiles with role 'target' hold the approved clips of the target person; "
                     "the target is in frame only inside the target window of each clip, and 'cuts' are the restarts of looped clips.")


def build(out: Path, only: list[str] | None = None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    needed = {t["src"] for recipe in COMPOSITES.values() for _, _, tiles in recipe for t in tiles}
    for key in [*TARGET_CLIPS, *OTHERS, *MORE_OTHERS]:
        if key in needed and not source_path(key).is_file():
            raise SystemExit(f"splice: missing source {key}: {source_path(key)} (docs/TESTDATA.md says where the new clips come from)")
    for key, (dur, _, _) in target_windows().items():
        if abs(ffprobe_duration(source_path(key)) - dur) > 0.1:
            raise SystemExit(f"splice: {key} is not {dur} s long; target_windows.json no longer matches the approved clip")
    made = []
    for name, recipe in COMPOSITES.items():
        if only and name not in only:
            continue
        if any(not source_path(t["src"]).is_file() for _, _, tiles in recipe for t in tiles):
            print(f"{name}: skipped, a source clip is missing (docs/TESTDATA.md)")
            continue
        parts = []
        for i, (layout, secs, tiles) in enumerate(recipe):
            for tl in tiles:
                need = tl["ss"] + 0.0
                if ffprobe_duration(source_path(tl["src"])) <= need:
                    raise SystemExit(f"splice: {tl['src']} is shorter than its start offset {need} s")
            seg = out / f"{name}.seg{i}.mp4"
            r = subprocess.run(segment_command(layout, secs, tiles, seg), capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"splice: ffmpeg failed for {name} segment {i}: {r.stderr.strip()[-300:]}")
            parts.append(seg)
        lst = out / f"{name}.concat.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
        mp4 = out / f"{name}.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(mp4)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"splice: concat failed for {name}: {r.stderr.strip()[-300:]}")
        for p in [*parts, lst]:
            p.unlink()
        truth = truth_for(name, recipe)
        truth["measured_duration"] = round(ffprobe_duration(mp4), 3)
        (out / f"{name}.truth.json").write_text(json.dumps(truth, indent=1))
        made.append(mp4)
        print(f"{name}: {truth['duration']:.0f} s, target {sum(e - s for s, e in truth['target_intervals']):.0f} s")
    return made


# ---------------------------------------------------------------- scoring
def overlap(a: list[list[float]], b: list[list[float]]) -> float:
    from inframe.compare import intersect, total

    return total(intersect([tuple(x) for x in a], [tuple(x) for x in b]))


def score_intervals(truth: dict, kept: list[list[float]]) -> dict:
    """Time-weighted precision and recall of the kept seconds against the seconds that hold him."""
    tgt = merge([list(x) for x in truth["target_intervals"]])
    kept = merge([list(x) for x in kept])
    hit = overlap(kept, tgt)
    kept_s, tgt_s = sum(e - s for s, e in kept), sum(e - s for s, e in tgt)
    p = hit / kept_s if kept_s else (1.0 if not tgt_s else 0.0)
    r = hit / tgt_s if tgt_s else 1.0
    return dict(name=truth["name"], precision=round(p, 4), recall=round(r, 4), kept_s=round(kept_s, 2),
                target_s=round(tgt_s, 2), correct_s=round(hit, 2))


def score_files(truth_path: Path, segments_path: Path, use: str = "clips") -> dict:
    truth = json.loads(Path(truth_path).read_text())
    d = json.loads(Path(segments_path).read_text())
    if use == "presence":
        kept = d["presence_intervals"]
    else:
        kept = [[c["start"], c["end"]] for c in d["clips"] if "error" not in c]
    return score_intervals(truth, kept)


def hit_samples(samples_csv: Path) -> list[tuple[float, tuple | None]]:
    """samples.csv of a run -> [(t, box or None)]: the box the run kept at every sample (None where it kept nothing)."""
    import csv

    return [(float(r["t"]), tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1")) if r["x0"] else None) for r in csv.DictReader(open(samples_csv))]


def in_target_tile(truth: dict, t: float, box) -> bool:
    """The centre of `box` (normalised) lies in a tile that holds him at time t."""
    _, tl = tile_at(truth, t, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return tl is not None and tl["role"] == "target"


def score_tile(truth: dict, presence: list, hits: list[tuple[float, tuple | None]], spans: list[list[float]] | None = None) -> dict:
    """Tile-aware precision and recall. A kept second counts only when he is in frame in it (as in score_intervals) AND the run's box
    in that sample lies in the target tile: a run that keeps a stranger while he is also in frame (in another tile) gets no credit for
    those seconds, neither in precision nor in recall. A sample without a box (a gap the segment logic bridged) is judged by time alone.
    `spans`: score only inside these intervals (the shots that got a tap)."""
    from inframe.compare import subtract
    from inframe.segments import sample_edges

    clip = (lambda iv: [[max(a, u), min(b, v)] for a, b in iv for u, v in spans if min(b, v) > max(a, u)]) if spans is not None else (lambda iv: [list(x) for x in iv])
    tgt = merge(clip(truth["target_intervals"]))
    kept = merge(clip(presence))
    ts = [t for t, _ in hits]
    edges = sample_edges(ts, truth["duration"])
    wrong = merge(clip([[edges[i], edges[i + 1]] for i, (t, box) in enumerate(hits) if box is not None and not in_target_tile(truth, t, box)]))
    right = [list(x) for x in subtract([tuple(x) for x in overlap_iv(kept, tgt)], [tuple(x) for x in wrong])]
    hit = sum(e - s for s, e in right)
    kept_s, tgt_s = sum(e - s for s, e in kept), sum(e - s for s, e in tgt)
    p = hit / kept_s if kept_s else (1.0 if not tgt_s else 0.0)
    r = hit / tgt_s if tgt_s else 1.0
    return dict(name=truth["name"], precision=round(p, 4), recall=round(r, 4), kept_s=round(kept_s, 2), target_s=round(tgt_s, 2),
                correct_s=round(hit, 2), wrong_tile_s=round(overlap(kept, wrong), 2))


def overlap_iv(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    from inframe.compare import intersect

    return [list(x) for x in intersect([tuple(x) for x in a], [tuple(x) for x in b])]


def oracle_hits(truth: dict, tgt, mode: str):
    """The ceiling of a target run: perfect identity, judged by the truth. `tgt` is a finished TapTarget (it holds the samples).
    mode "dets": every detection in the target tile is him, tracked or not. "tracks": every track that is mostly in the target tile is him,
    plus the bridge over the holes (what a perfect identity decision could reach with this tracker). -> [Hit]"""
    import numpy as np

    from inframe.target import Hit

    S = tgt.samples
    units, unit_of, have_ids = tgt._units()
    target_u = {u for u, ms in units.items() if np.mean([in_target_tile(truth, S[si].t, S[si].cands[ci].box) for si, ci in ms]) > 0.5}
    chosen = []
    for si, s in enumerate(S):
        best = None
        for ci, d in enumerate(s.cands):
            ok = (unit_of.get((si, ci)) in target_u) if mode == "tracks" else in_target_tile(truth, s.t, d.box)
            if ok and (best is None or d.score > best.score):
                best = d
        chosen.append(best)
    if mode == "tracks" and have_ids:
        tgt._bridge(chosen)
    return [Hit(d.score, d.box, d.track_id) if d else Hit(0.0, None) for d in chosen]


def oracle_ceiling(truth: dict, tgt, mode: str, spans: list[list[float]] | None = None) -> dict:
    """score_tile of the oracle run `mode` ("dets" or "tracks"), cut with the pipeline's segment rules (no second run: it reads the samples)."""
    from inframe.segments import build_segments

    hits = oracle_hits(truth, tgt, mode)
    ts = [s.t for s in tgt.samples]
    pres, _ = build_segments(ts, [h.score for h in hits], truth["duration"], 0.5, 0.3, 1.5, 1.0, 1.0)
    return score_tile(truth, [list(p) for p in pres], [(t, h.box) for t, h in zip(ts, hits)], spans)


def weighted_pr(rows: list[dict]) -> tuple[float, float]:
    """Time-weighted (precision, recall) over the composites of one run."""
    kt, tt = sum(r["kept_s"] for r in rows), sum(r["target_s"] for r in rows)
    return (sum(r["precision"] * r["kept_s"] for r in rows) / max(kt, 1e-9), sum(r["recall"] * r["target_s"] for r in rows) / max(tt, 1e-9))


def median_spread(values: list[float]) -> tuple[float, float]:
    """(median, max - min) of the repeated runs of one number."""
    v = sorted(values)
    return (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2), v[-1] - v[0]


def tile_at(truth: dict, t: float, cx: float, cy: float):
    """(segment index, tile) a detection centre (normalised) falls in at time t; the smallest tile that holds it.
    A sample at exactly a shot edge (t == t1 of one shot, 5.0 s on the 0.2 s grid) is the first frame of the next shot."""
    segs = truth["segments"]
    for si, seg in enumerate(segs):
        last = si == len(segs) - 1
        if seg["t0"] - 1e-6 <= t < seg["t1"] - 1e-6 or (last and t < seg["t1"] + 1e-9):
            px, py = cx * truth["width"], cy * truth["height"]
            best = None
            for tl in seg["tiles"]:
                x, y, w, h = tl["rect"]
                if x <= px < x + w and y <= py < y + h and (best is None or w * h < best["rect"][2] * best["rect"][3]):
                    best = tl
            return si, best
    return None, None


def _cover(sets: list[set], keep: float = 0.95) -> int:
    """Fewest track ids that between them appear in at least `keep` of the non-empty samples (greedy)."""
    left = [i for i, st in enumerate(sets) if st]
    need = len(left) - int(keep * len(left))  # samples that may stay uncovered
    n = 0
    while len(left) > need and left:
        counts: dict = {}
        for i in left:
            for x in sets[i]:
                counts[x] = counts.get(x, 0) + 1
        best = max(counts, key=lambda k: (counts[k], -k if isinstance(k, int) else 0))
        left = [i for i in left if best not in sets[i]]
        n += 1
    return n


def id_switches(truth: dict, samples: list[tuple[float, list[tuple[int | None, tuple, float]]]], min_score: float = 0.5) -> dict:
    """samples: [(t, [(track_id, box, score), ...]), ...] in time order, boxes normalised (x0, y0, x1, y1).
    The truth says which tile holds him in every shot. Without boxes in the truth, an ID switch is measured as
    fragmentation of the target tile: `id_switches` = (fewest track ids that cover 95 % of the tracked samples in the target tile) - 1,
    summed over shots, so a second person who is tracked steadily next to him costs nothing, and one person who gets
    a new id half way costs 1. `strongest_changes` is the stricter count (the strongest detection in the target tile changed
    id, MOTA-style); it also counts the strongest detection hopping between two people. `untracked` = samples where he
    had a detection but no id yet. `id_leaks` = ids used both in the target tile and in another tile of the same shot."""
    tiles: dict[tuple, list[set]] = {}
    last: dict[tuple, int] = {}
    in_target: set[tuple] = set()
    in_other: set[tuple] = set()
    changes = frames = untracked = missed = 0
    cuts = truth.get("cuts", [])
    for t, dets in samples:
        seg = next((i for i, sg in enumerate(truth["segments"]) if sg["t0"] <= t < sg["t1"] + 1e-9), None)
        if seg is None:
            continue
        si = (seg, sum(1 for c in cuts if truth["segments"][seg]["t0"] < c <= t + 1e-9))  # a clip restart starts a new shot
        sets: dict[tuple, set] = {}
        best: dict[tuple, tuple] = {}
        for tid, box, score in dets:
            _, tl = tile_at(truth, t, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if tl is None or score < min_score:
                continue
            key = (si, tl["id"])
            if tl["role"] in ("target", "tracked"):
                if tid is not None:
                    sets.setdefault(key, set()).add(tid)
                    in_target.add((si, tid))
                if key not in best or score > best[key][2]:
                    best[key] = (tid, box, score)
            elif tid is not None:
                in_other.add((si, tid))
        for tl in truth["segments"][seg]["tiles"]:
            if tl["role"] not in ("target", "tracked"):
                continue
            key = (si, tl["id"])
            tiles.setdefault(key, []).append(sets.get(key, set()))
            if key not in best:
                missed += 1
                continue
            frames += 1
            tid = best[key][0]
            if tid is None:
                untracked += 1
                continue
            if key in last and last[key] != tid:
                changes += 1
            last[key] = tid
    cover = {k: _cover(v) for k, v in tiles.items()}
    return dict(id_switches=sum(max(c - 1, 0) for c in cover.values()), strongest_changes=changes, target_samples=frames,
                untracked=untracked, missed=missed, id_leaks=len(in_target & in_other),
                ids_per_target=round(sum(cover.values()) / max(len(cover), 1), 2))


# ---------------------------------------------------------------- ID-switch measurement
def detect_composite(video: Path, fps: float = 10.0, sliced: bool = False, cls: int = 1) -> list[tuple[float, list]]:
    """Run the detector once over a composite at `fps`: [(t, [(box, score), ...])]. Cached next to the video.
    `sliced`: with the slices tap mode uses (inframe.detect.SlicedDetector, grid from INFRAME_SLICES)."""
    from inframe.detect import compute_units_name, for_tap, slice_grid

    units = compute_units_name()  # the GPU and CPU_ONLY scores differ: they get their own cache
    sliced = sliced and slice_grid() != (1, 1)
    from inframe.pipeline import PRESETS

    model = PRESETS[PRESET].model
    tag = (f".{units.lower()}" if units != "CPU_AND_GPU" else "") + (f".{model}" if model != "small" else "") + ("." + "x".join(map(str, slice_grid())) if sliced else "") \
        + (f".c{cls}" if cls != 1 else "")
    cache = video.with_suffix(f".dets{fps:g}{tag}.json")
    if cache.is_file() and cache.stat().st_mtime >= video.stat().st_mtime:
        return [(t, [(tuple(b), s) for b, s in d]) for t, d in json.loads(cache.read_text())]
    from inframe.decode import iter_samples
    from inframe.detect import BATCH, load_detector
    from inframe.probe import probe

    info = probe(str(video))
    det = load_detector(model)
    det = for_tap(det) if sliced else det
    out, buf = [], []

    def flush():
        if buf:
            for s, d in zip(buf, det.detect([b.rgb for b in buf], cls=cls)):
                out.append((s.t, [(x.box, x.score) for x in d]))
            buf.clear()

    for s in iter_samples(str(video), info, fps=fps):
        buf.append(s)
        if len(buf) == BATCH:
            flush()
    flush()
    cache.write_text(json.dumps([(t, [(list(b), sc) for b, sc in d]) for t, d in out]))
    return out


def frames_at(video: Path, times: list[float], fps: float) -> dict[int, "np.ndarray"]:
    """Frames of `video` sampled at `fps` by the pipeline's own decoder, keyed by round(t * 10) (the key of the 10 fps detections).
    BoT-SORT estimates camera motion from the picture, so it needs the real frames; OC-SORT does not."""
    from inframe.decode import iter_samples
    from inframe.probe import probe

    return {round(s.t * 10): s.rgb for s in iter_samples(str(video), probe(str(video)), fps=fps)}


def subsample_step(fps: float) -> int:
    """Every how-manyth detection (they are at 10 fps) feeds a tracker at `fps`; at 10 fps or more every one (never 0: a zero slice step raises)."""
    return max(1, round(10.0 / fps))


def work_dir(out: Path, kind: str, sub: str = "") -> Path:
    """Scratch folder of a measurement, next to the composites it runs on (so two --out folders never share one)."""
    return Path(out) / "work" / kind / sub


@contextlib.contextmanager
def temp_targets_store():
    """A throwaway INFRAME_TARGETS folder for the length of the block; the variable is put back as it was (or removed) afterwards."""
    import tempfile

    before = os.environ.get("INFRAME_TARGETS")
    with tempfile.TemporaryDirectory() as store:
        os.environ["INFRAME_TARGETS"] = store
        try:
            yield Path(store)
        finally:
            if before is None:
                os.environ.pop("INFRAME_TARGETS", None)
            else:
                os.environ["INFRAME_TARGETS"] = before


def run_tracker(dets, fps: float, kind: str, truth: dict | None = None, frame_shape=(360, 640, 3), video: Path | None = None, **params):
    """dets at 10 fps -> subsample to `fps` -> tracker; returns samples for id_switches. The composites are cut
    together from separate shots, and a real clip has no hard cuts, so the tracker starts afresh at every cut (shot starts and restarts of looped clips)."""
    import numpy as np

    from inframe.detect import Detection
    from inframe.tracks import Tracker

    step = subsample_step(fps)
    keep = dets[::step]
    frame = np.zeros(frame_shape, np.uint8)
    res = []
    starts = shot_starts(truth) if truth else []
    shot = -1
    real = frames_at(video, [t for t, _ in keep], fps) if kind == "botsort" and video is not None else {}
    with Tracker(kind, fps, allow_low_fps=True, **params) as tr:
        for t, d in keep:
            frame = real.get(round(t * 10), frame)
            now = sum(1 for x in starts if x <= t + 1e-9)
            if now != shot:
                shot = now
                tr.reset()
            out = tr.update(t, [Detection(s, tuple(b)) for b, s in d], frame)
            res.append((t, [(o.track_id, o.box, o.score) for o in out]))
    return res


def measure_ids(out: Path, fps_list: list[float], kinds: list[str], only: list[str] | None = None) -> list[dict]:
    rows = []
    for tp in sorted(out.glob("*.truth.json")):
        truth = json.loads(tp.read_text())
        if (only and truth["name"] not in only) or truth.get("class"):  # the ID-switch count is for people
            continue
        dets = detect_composite(out / truth["video"])
        for kind in kinds:
            for fps in fps_list:
                r = id_switches(truth, run_tracker(dets, fps, kind, truth, video=out / truth["video"]))
                rows.append(dict(composite=truth["name"], tracker=kind, fps=fps, **r))
                print(f"{truth['name']:15s} {kind:7s} {fps:4g} fps  switches {r['id_switches']:3d}  untracked {r['untracked']:3d}  "
                      f"of {r['target_samples']:4d}  strongest {r['strongest_changes']:3d}  ids/target {r['ids_per_target']}  leaks {r['id_leaks']}", flush=True)
    return rows


# ---------------------------------------------------------------- tap mode on the composites
def taps_for(truth: dict, dets, one: bool = False):
    """What a person would click: in every shot where he is in frame, the strongest detection inside the target tile, at the middle
    of the target's time in that shot (on the 0.2 s grid of the 5 fps run). `one`: only the first such shot. Returns (taps, untappable)."""
    from inframe.tap import Tap

    by_t = {round(t * 10): d for t, d in dets}
    taps, missed = [], []
    for si, seg in enumerate(truth["segments"]):
        mine = [[max(a, seg["t0"]), min(b, seg["t1"])] for a, b in truth["target_intervals"] if min(b, seg["t1"]) - max(a, seg["t0"]) > 0.5]
        if not mine:
            continue
        a, b = mine[0]
        mid = (a + b) / 2
        best = None  # (score, -distance from the middle, t, box): a person taps a frame where he is clearly boxed
        for k in range(int(a / 0.2) + 1, int(b / 0.2 - 1e-6) + 1):  # strictly inside the shot
            t = round(k * 0.2, 1)
            for box, score in by_t.get(round(t * 10), []):
                _, tl = tile_at(truth, t, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if tl is not None and tl["role"] == "target" and score >= 0.5:
                    key = (round(score, 1), -abs(t - mid), t, tuple(box))
                    if best is None or key > best:
                        best = key
        t = round(mid, 1)
        if best is None:
            missed.append(round(t, 1))
        else:
            taps.append(Tap(best[2], box=best[3]))
        if one and taps:
            break
    return taps, missed


def target_tile_ceiling(truth: dict, samples, spans: list[list[float]]) -> dict:
    """The most any tap run can recall in the tapped shots: every detection inside the target tile counts as him (perfect identity), the
    run is cut with the pipeline's segment rules. `samples`: objects with .t and .cands (box, score) as TapTarget records them.
    Scored like the tap run itself (inside `spans`)."""
    from inframe.segments import build_segments

    ts, sc = [s.t for s in samples], []
    for s in samples:
        best = 0.0
        for d in s.cands:
            _, tl = tile_at(truth, s.t, (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
            if tl is not None and tl["role"] == "target":
                best = max(best, d.score)
        sc.append(best)
    pres, _ = build_segments(ts, sc, truth["duration"], 0.5, 0.3, 1.5, 1.0, 1.0)
    inside = lambda iv: [[max(a, u), min(b, v)] for a, b in iv for u, v in spans if min(b, v) > max(a, u)]
    return score_intervals(dict(truth, target_intervals=inside(truth["target_intervals"])), inside([list(p) for p in pres]))


def tap_samples(samples_csv: Path):
    """samples.csv of a tap run -> [(t, [(track, box, score)])] holding only the target's hit, for id_switches."""
    import csv

    rows = []
    for r in csv.DictReader(open(samples_csv)):
        hit = [(int(r["track"]) if r.get("track") else None, (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])), float(r["score"]))] if r["x0"] else []
        rows.append((float(r["t"]), hit))
    return rows


def measure_taps(out: Path, one: bool, only: list[str] | None, work: Path, tracker: str = "ocsort", class_name: str | None = None) -> list[dict]:
    """`class_name`: the composites whose target is one object of that COCO class (COMPOSITE_CLASS); None: the person composites (the target person)."""
    from inframe import classes
    from inframe.pipeline import Settings, run_cut
    from inframe.tap import TapTarget

    cls = classes.class_id(class_name) if class_name else classes.PERSON
    rows = []
    for tp in sorted(out.glob("*.truth.json")):
        truth = json.loads(tp.read_text())
        if (only and truth["name"] not in only) or not truth["target_intervals"] or truth.get("class") != (class_name and classes.class_name(cls)):
            continue
        video = out / truth["video"]
        taps, missed = taps_for(truth, detect_composite(video, sliced=True, cls=cls), one)
        run = work / truth["name"]
        tg = TapTarget(taps, cls=cls)
        t_run = time.perf_counter()
        doc = run_cut(str(video), str(run), Settings(tracker=tracker, preset=PRESET), target=tg)
        wall = time.perf_counter() - t_run
        hits = hit_samples(run / "samples.csv")
        pr = score_tile(truth, doc["presence_intervals"], hits)  # a kept second counts only when the box is in the target tile
        spans = [[sg["t0"], sg["t1"]] for sg in truth["segments"] if any(sg["t0"] <= tp_.t < sg["t1"] for tp_ in taps)]
        pt = score_tile(truth, doc["presence_intervals"], hits, spans)  # only the shots that got a tap
        hc = target_tile_ceiling(truth, tg.samples, spans)  # what a perfect identity could recall from the detections in the target tile
        dets = {round(t * 10): d for t, d in detect_composite(video, sliced=True, cls=cls)}  # how much of the target the detector sees at all (score >= 0.5 in the target tile)
        grid = [round(k * 0.2, 1) for k in range(int(truth["duration"] / 0.2))]
        in_spans = lambda t: any(a <= t < b for a, b in spans)
        want = [t for t in grid if in_spans(t) and any(a <= t < b for a, b in truth["target_intervals"])]
        seen = [t for t in want if any(sc >= 0.5 and (tl := tile_at(truth, t, (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)[1]) is not None
                                       and tl["role"] == "target" for bx, sc in dets.get(round(t * 10), []))]
        detectable = len(seen) / max(len(want), 1)
        od, ot = oracle_ceiling(truth, tg, "dets"), oracle_ceiling(truth, tg, "tracks")  # perfect identity: the ceilings of this detector and tracker
        cl = score_intervals(truth, [[c["start"], c["end"]] for c in doc["clips"]])
        ids = id_switches(truth, tap_samples(run / "samples.csv"))
        info = doc["settings"].get("target_info", {})
        rows.append(dict(composite=truth["name"], taps=len(taps), tap_precision=pt["precision"], tap_recall=pt["recall"], tap_kept_s=pt["kept_s"], tap_target_s=pt["target_s"], wall_s=round(wall, 1), duration=truth["duration"], target_ceiling=hc["recall"], target_ceiling_precision=hc["precision"], tap_over_ceiling=round(pt["recall"] / hc["recall"], 3) if hc["recall"] else None, detectable=round(detectable, 3), oracle_dets=od["recall"], oracle_tracks=ot["recall"], oracle_tracks_precision=ot["precision"], untappable=missed, precision=pr["precision"], recall=pr["recall"],
                         clip_precision=cl["precision"], clip_recall=cl["recall"], kept_s=pr["kept_s"], target_s=pr["target_s"],
                         id_switches=ids["id_switches"], id_leaks=ids["id_leaks"], strongest_changes=ids["strongest_changes"],
                         tracks=info.get("tracks"), tracks_accepted=info.get("tracks_accepted")))
        r = rows[-1]
        print(f"{r['composite']:15s} taps {r['taps']}  precision {r['precision']:.3f} recall {r['recall']:.3f}  "
              f"| tapped shots only {r['tap_precision']:.3f} / {r['tap_recall']:.3f} (the target is detected in {r['detectable']:.2f} of that time) | target-tile ceiling {r['target_ceiling']:.3f}, tap/ceiling {r['tap_over_ceiling'] if r['tap_over_ceiling'] is None else format(r['tap_over_ceiling'], '.3f')} | [oracle dets {r['oracle_dets']:.3f}, oracle tracks {r['oracle_tracks']:.3f}]  (clips, padded: {r['clip_precision']:.3f} / {r['clip_recall']:.3f})  id switches {r['id_switches']}  leaks {r['id_leaks']}  "
              f"tracks accepted {r['tracks_accepted']} of {r['tracks']}" + (f"  untappable shots at {missed}" if missed else ""), flush=True)
        for f in run.glob("*_*-*.mp4"):  # the clips are not needed
            f.unlink()
    if rows:
        tk = sum(r["target_s"] for r in rows)
        kt, tt = sum(r["tap_kept_s"] for r in rows), sum(r["tap_target_s"] for r in rows)
        print(f"TAPPED SHOTS ONLY (time-weighted): precision {sum(r['tap_precision'] * r['tap_kept_s'] for r in rows) / max(kt, 1e-9):.3f} "
              f"recall {sum(r['tap_recall'] * r['tap_target_s'] for r in rows) / max(tt, 1e-9):.3f} over {tt:.0f} s of him")
        ok = [r for r in rows if r["target_ceiling"] >= 0.95]  # composites where the detector sees him nearly always: the tap must too
        ct = sum(r["tap_target_s"] for r in rows)
        print(f"TARGET-TILE CEILING (time-weighted): {sum(r['target_ceiling'] * r['tap_target_s'] for r in rows) / max(ct, 1e-9):.3f}; tap/ceiling "
              f"{sum(r['tap_recall'] * r['tap_target_s'] for r in rows) / max(sum(r['target_ceiling'] * r['tap_target_s'] for r in rows), 1e-9):.3f}; "
              f"on the {len(ok)} composites with ceiling >= 0.95: tap recall "
              f"{sum(r['tap_recall'] * r['tap_target_s'] for r in ok) / max(sum(r['tap_target_s'] for r in ok), 1e-9):.3f}, "
              f"untappable shots {sum(len(r['untappable']) for r in rows)}")
        print(f"TOTAL (time-weighted over {len(rows)} composites): precision {sum(r['precision'] * r['kept_s'] for r in rows) / max(sum(r['kept_s'] for r in rows), 1e-9):.3f} "
              f"recall {sum(r['recall'] * r['target_s'] for r in rows) / max(tk, 1e-9):.3f} (oracle dets {sum(r['oracle_dets'] * r['target_s'] for r in rows) / max(tk, 1e-9):.3f}, oracle tracks {sum(r['oracle_tracks'] * r['target_s'] for r in rows) / max(tk, 1e-9):.3f}); id switches {sum(r['id_switches'] for r in rows)}; "
              f"leaks {sum(r['id_leaks'] for r in rows)}")
    return rows


# ---------------------------------------------------------------- T1c: photo mode
PHOTO_CLIPS = {"07": "p7", "09": "p9", "11": "p11"}  # clip index in TARGET_DIR -> key; no composite uses these
PHOTOS_PER_CLIP = 3


def build_photos(out: Path) -> Path:
    """The "photos of the target": crops of the largest person in frames of three target clips that no composite uses (other
    moments of the same video, so the same clothes). Written as JPEGs next to the composites, never committed. Nothing
    is looked at: a frame counts when the detector's largest person is at least 12 % of the frame high (the person is small in these clips)."""
    import tempfile

    import numpy as np
    from PIL import Image

    from inframe.identity import lazy_detect, person_crop

    dest = out / "photos"
    dest.mkdir(parents=True, exist_ok=True)
    for f in dest.glob("*.jpg"):
        f.unlink()
    detect = lazy_detect("small")
    n = 0
    for name, key in PHOTO_CLIPS.items():  # name is the clip index
        src = target_clip(name)
        dur = ffprobe_duration(src)
        got = 0
        for frac in (0.3, 0.45, 0.6, 0.75, 0.5, 0.35, 0.65, 0.4, 0.55):
            if got >= PHOTOS_PER_CLIP:
                break
            with tempfile.TemporaryDirectory() as td:
                png = Path(td) / "f.png"
                subprocess.run(["ffmpeg", "-nostats", "-loglevel", "error", "-ss", f"{dur * frac:.2f}", "-i", str(src), "-frames:v", "1", str(png)], check=True)
                rgb = np.asarray(Image.open(png).convert("RGB"))
            dets = [d for d in detect([rgb])[0] if d.score >= 0.7 and d.box[3] - d.box[1] >= 0.12]
            if not dets:
                continue
            big = max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
            crop = person_crop(rgb, big.box)
            if crop is not None:
                Image.fromarray(crop).save(dest / f"{key}_{got}.jpg", quality=92)
                got += 1
                n += 1
    print(f"{n} photos in {dest}")
    return dest


def measure_photos(out: Path, only: list[str] | None, embedder: str, work: Path, auto: bool = True) -> list[dict]:
    """Photo mode on the composites. Per composite: (1) rank the tracks against the photos; (2) the "one confirm": the first of the
    top 5 that really is him (by the truth) is confirmed, and the run is scored; (3) the same without a confirm (auto-pick
    above the margin, else Ambiguous = exit 3). Precision and recall are tile-aware (score_tile): a kept second counts only when the
    run's box lies in the target tile. The ceilings are the oracle runs (perfect identity on the same detections and tracks)."""
    from inframe.identity import Ambiguous, PhotoTarget, lazy_detect
    from inframe.pipeline import Settings, run_cut
    from inframe.tap import Tap

    photos = out / "photos"
    if not list(photos.glob("*.jpg")):
        build_photos(out)
    from inframe.pipeline import PRESETS
    detect = lazy_detect(PRESETS[PRESET].model)
    rows = []
    for tp in sorted(out.glob("*.truth.json")):
        truth = json.loads(tp.read_text())
        if (only and truth["name"] not in only) or not truth["target_intervals"] or truth.get("class"):
            continue
        video = out / truth["video"]
        run = work / truth["name"]
        rk = PhotoTarget(str(photos), embedder=embedder, rank_only=True, detect=detect)
        run_cut(str(video), str(run / "rank"), Settings(preset=PRESET), target=rk)
        ranking = rk.info["ranking"]
        who = [tile_at(truth, r["t"] + 0.05, (r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2)[1] for r in ranking]
        pos = next((i + 1 for i, tl in enumerate(who) if tl is not None and tl["role"] == "target"), None)
        od, ot = oracle_ceiling(truth, rk, "dets"), oracle_ceiling(truth, rk, "tracks")  # perfect identity on this run's detections and tracks
        row = dict(composite=truth["name"], rank_of_him=pos, oracle_dets=od["recall"], oracle_tracks=ot["recall"], top_scores=[r["score"] for r in ranking], n_photos=rk.info["photos"])
        if pos is not None:
            c = ranking[pos - 1]
            tgt = PhotoTarget(str(photos), [Tap(c["t"], box=tuple(c["box"]))], embedder=embedder, detect=detect)
            doc = run_cut(str(video), str(run / "confirm"), Settings(preset=PRESET), target=tgt)
            pr = score_tile(truth, doc["presence_intervals"], hit_samples(run / "confirm" / "samples.csv"))
            ids = id_switches(truth, tap_samples(run / "confirm" / "samples.csv"))
            row.update(precision=pr["precision"], recall=pr["recall"], kept_s=pr["kept_s"], target_s=pr["target_s"], correct_s=pr["correct_s"],
                       id_switches=ids["id_switches"], id_leaks=ids["id_leaks"], tracks_accepted=tgt.info.get("tracks_accepted"),
                       accepted=[(a["id"], a["score"], a["t0"], a["t1"]) for a in tgt.info.get("accepted_tracks", [])],
                       ambiguous=tgt.info.get("ambiguous"))
        else:
            row.update(precision=0.0, recall=0.0, kept_s=0.0, correct_s=0.0, target_s=score_intervals(truth, [])["target_s"], id_switches=None, id_leaks=None)
        if auto:
            try:
                a = PhotoTarget(str(photos), embedder=embedder, detect=detect)
                d2 = run_cut(str(video), str(run / "auto"), Settings(preset=PRESET), target=a)
                pa = score_tile(truth, d2["presence_intervals"], hit_samples(run / "auto" / "samples.csv"))
                row.update(auto="picked", auto_precision=pa["precision"], auto_recall=pa["recall"])
            except Ambiguous as e:
                row.update(auto="exit 3", auto_why=str(e)[:160])
        rows.append(row)
        print(f"{row['composite']:15s} him at rank {row['rank_of_him']}  top scores {row['top_scores']}  precision {row['precision']:.3f} recall {row['recall']:.3f}  "
              f"[oracle dets {row['oracle_dets']:.3f}, oracle tracks {row['oracle_tracks']:.3f}] switches {row['id_switches']} leaks {row['id_leaks']}" + (f"  | no-UI: {row['auto']}" + (f" {row['auto_precision']:.3f}/{row['auto_recall']:.3f}" if row.get("auto") == "picked" else "") if auto else ""), flush=True)
        for f in [*(run / "confirm").glob("*_*-*.mp4"), *(run / "auto").glob("*_*-*.mp4")]:
            f.unlink()
    if rows:
        kt, tt = sum(r["kept_s"] for r in rows), sum(r["target_s"] for r in rows)
        print(f"ONE CONFIRM (time-weighted over {len(rows)} composites): precision {sum(r['precision'] * r['kept_s'] for r in rows) / max(kt, 1e-9):.3f} "
              f"recall {sum(r['recall'] * r['target_s'] for r in rows) / max(tt, 1e-9):.3f} over {tt:.0f} s of him; id switches {sum(r['id_switches'] or 0 for r in rows)}; "
              f"leaks {sum(r['id_leaks'] or 0 for r in rows)}; ceilings: oracle dets {sum(r['oracle_dets'] * r['target_s'] for r in rows) / max(tt, 1e-9):.3f}, oracle tracks {sum(r['oracle_tracks'] * r['target_s'] for r in rows) / max(tt, 1e-9):.3f}")
        got = [r for r in rows if r["rank_of_him"] is not None]
        print(f"him in the top 5 on {len(got)} of {len(rows)} composites")
    return rows


# ---------------------------------------------------------------- T2: saved targets
SAVED_PAIRS = [("swap_sides", "quad"), ("quad", "pip_small"), ("pip_small", "swap_sides")]  # A's clips of him and B's clips of him are different clips


def measure_saved(out: Path, pairs: list[tuple[str, str]], embedder: str, work: Path) -> list[dict]:
    """A target saved from composite A finds the target in composite B with no tap. Per pair: run A with taps on the target (what a person would click),
    save crops from that run (`inframe target add --from run:`), then run B with `saved:NAME` alone and score it tile-aware. The store
    is a temp folder, never the real one."""
    from inframe import identity, targets
    from inframe.identity import Ambiguous
    from inframe.pipeline import Settings, run_cut
    from inframe.tap import TapTarget

    rows = []
    with temp_targets_store():
        for an, bn in pairs:
            ta, tb = (json.loads((out / f"{n}.truth.json").read_text()) for n in (an, bn))
            srcs = [{x["src"] for s_ in t["segments"] for x in s_["tiles"] if x["role"] == "target"} for t in (ta, tb)]
            if srcs[0] & srcs[1]:
                print(f"{an} -> {bn}: skipped, both use {sorted(srcs[0] & srcs[1])} of him")
                continue
            va, vb = out / ta["video"], out / tb["video"]
            taps, _ = taps_for(ta, detect_composite(va, sliced=True))
            run_a = work / f"{an}_to_{bn}" / "A"
            run_cut(str(va), str(run_a), Settings(), target=TapTarget(taps))
            for f in run_a.glob("*_*-*.mp4"):
                f.unlink()
            name = f"him_{an}"
            t0 = time.perf_counter()
            r = targets.add_crops(name, targets.crops_from_run(str(run_a / "segments.json")), "run", embedder)
            tgt = identity.parse_targets([f"saved:{name}"], embedder)
            run_b = work / f"{an}_to_{bn}" / "B"
            row = dict(a=an, b=bn, crops=r["n_crops"], target_s=score_intervals(tb, [])["target_s"])
            try:
                doc = run_cut(str(vb), str(run_b), Settings(), target=tgt)
                pr = score_tile(tb, doc["presence_intervals"], hit_samples(run_b / "samples.csv"))
                row.update(result="picked", precision=pr["precision"], recall=pr["recall"], kept_s=pr["kept_s"], correct_s=pr["correct_s"])
            except Ambiguous as e:
                row.update(result="exit 3", why=str(e)[:160], precision=0.0, recall=0.0, kept_s=0.0, correct_s=0.0)
            row["wall_s"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
            print(f"{an} -> {bn}: {row['crops']} crops saved; {row['result']}  precision {row['precision']:.3f} recall {row['recall']:.3f} "
                  f"(kept {row['kept_s']} s of {row['target_s']} s of him)", flush=True)
            for f in run_b.glob("*_*-*.mp4"):
                f.unlink()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.add_argument("--only", nargs="*")
    s = sub.add_parser("score")
    s.add_argument("truth")
    s.add_argument("segments")
    s.add_argument("--use", choices=["clips", "presence"], default="clips")
    i = sub.add_parser("ids")
    i.add_argument("--out", default=str(DEFAULT_OUT))
    i.add_argument("--fps", nargs="*", type=float, default=[2, 5, 10])
    i.add_argument("--tracker", nargs="*", default=["ocsort"])
    i.add_argument("--only", nargs="*", help="composite names (default: all in --out)")
    i.add_argument("--json", help="write the rows here")
    t = sub.add_parser("tap")
    t.add_argument("--out", default=str(DEFAULT_OUT))
    t.add_argument("--one-tap", action="store_true", help="a single tap per composite, on the target's first appearance")
    t.add_argument("--only", nargs="*")
    t.add_argument("--tracker", default="ocsort")
    t.add_argument("--class", dest="cls", default=None, metavar="NAME", help="score the composites whose target is one object of this COCO class (dog_among_dogs, dog_twins) instead of the target person")
    t.add_argument("--slices", help="grid of detector slices for tap mode: 2x2, 3x2 (default: off)")
    t.add_argument("--preset", choices=["fast", "balanced", "accurate"], default="balanced", help="detector preset (accurate = RF-DETR base)")
    t.add_argument("--json", help="write the rows here")
    ph = sub.add_parser("photos")
    ph.add_argument("--out", default=str(DEFAULT_OUT))
    sv = sub.add_parser("saved")
    sv.add_argument("--out", default=str(DEFAULT_OUT))
    sv.add_argument("--embedder", default="dinov2")
    sv.add_argument("--pair", nargs="*", help="A:B pairs of composite names (default: " + ", ".join(f"{x}:{y}" for x, y in SAVED_PAIRS) + ")")
    sv.add_argument("--json", help="write the rows here")
    pm = sub.add_parser("photo")
    pm.add_argument("--out", default=str(DEFAULT_OUT))
    pm.add_argument("--embedder", default="dinov2")
    pm.add_argument("--only", nargs="*")
    pm.add_argument("--no-auto", action="store_true", help="skip the run without a confirm")
    pm.add_argument("--runs", type=int, default=1, help="repeat the whole measurement and report the median and the spread (max - min)")
    pm.add_argument("--preset", choices=["fast", "balanced", "accurate"], default="balanced", help="detector preset (accurate = RF-DETR base)")
    pm.add_argument("--json", help="write the rows here")
    a = ap.parse_args(argv)
    # The detector's default compute units (CPU_AND_GPU with the fp32-topk package) are deterministic, so the measurement runs what the product runs.
    global PRESET
    PRESET = getattr(a, "preset", "balanced")
    if a.cmd == "tap" and a.slices:
        os.environ["INFRAME_SLICES"] = a.slices
    if a.cmd == "saved":
        pairs = [tuple(p_.split(":")) for p_ in a.pair] if a.pair else SAVED_PAIRS
        rows = measure_saved(Path(a.out).expanduser(), pairs, a.embedder, work_dir(Path(a.out).expanduser(), "saved"))
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1))
        return 0
    if a.cmd == "photos":
        build_photos(Path(a.out).expanduser())
        return 0
    if a.cmd == "photo":
        runs = []
        for k in range(max(1, a.runs)):
            work = work_dir(Path(a.out).expanduser(), "photo", f"run{k}" if a.runs > 1 else "")
            runs.append(measure_photos(Path(a.out).expanduser(), a.only, a.embedder, work, not a.no_auto))
        if a.runs > 1:
            pr_, rc_ = zip(*[weighted_pr(r) for r in runs if r])
            (pm_, ps_), (rm_, rs_) = median_spread(list(pr_)), median_spread(list(rc_))
            print(f"OVER {len(runs)} RUNS: precision median {pm_:.3f} spread {ps_:.3f} ({', '.join(f'{x:.3f}' for x in pr_)}); "
                  f"recall median {rm_:.3f} spread {rs_:.3f} ({', '.join(f'{x:.3f}' for x in rc_)})")
        if a.json:
            Path(a.json).write_text(json.dumps(runs[0] if a.runs == 1 else runs, indent=1))
        return 0
    if a.cmd == "tap":
        rows = measure_taps(Path(a.out).expanduser(), a.one_tap, a.only, work_dir(Path(a.out).expanduser(), "tap"), a.tracker, a.cls)
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1))
        return 0
    if a.cmd == "build":
        build(Path(a.out).expanduser(), a.only)
    elif a.cmd == "score":
        r = score_files(Path(a.truth), Path(a.segments), a.use)
        print(f"{r['name']}: precision {r['precision']:.3f} recall {r['recall']:.3f} "
              f"(kept {r['kept_s']} s, target {r['target_s']} s, correct {r['correct_s']} s)")
    else:
        rows = measure_ids(Path(a.out).expanduser(), a.fps, a.tracker, a.only)
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
