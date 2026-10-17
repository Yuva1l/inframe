"""Tap mode: the user taps the person (or animal-free: any detected person) they want; the run follows that person.

A tap is a time and either a box (from `inframe pick`, which writes target.json) or a point (`--target tap:T,x,y`,
x and y as fractions of the frame, 0..1, upright). The tap is resolved against the detections of the run itself:
the detection that holds the point (the smallest one if several do) or overlaps the box, at the sample nearest to
the tap that has one, within WINDOW_S seconds. No such detection: TapError (the CLI exits 2).

The target is then the tracker's track of that detection (tracks.py), followed backwards and forwards in time, plus every
other track in the run that looks like the same person (a colour signature of the clothes, compared with the tapped
track; the "same-session crops" idea, with a plain colour histogram until identity.py brings the embedders in T1c)
and does not overlap in time with an accepted track (two people in one frame are not one person).
Several taps (one per appearance) all seed the same gallery.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classes import PERSON
from .detect import Detection
from .target import Hit

WINDOW_S = 1.0  # a tap needs a detection at a sample this close in time
MIN_IOU = 0.3  # a tapped box must overlap a detection at least this much
POINT_MARGIN = 0.01  # a tapped point may miss a box by this fraction of the frame
MIN_TRACK_SAMPLES = 3  # shorter tracks are never matched by appearance
ACCEPT = 0.70  # signature similarity (histogram intersection, 0..1) that makes another track "the same person"
ACCEPT_LONG = 0.65  # the same for a track of LONG_TRACK samples or more: a mean over that many frames is steadier
LONG_TRACK = 15
GROW = 0.80  # above this the accepted track's signature joins the gallery
GALLERY_MAX = 8
BRIDGE_IOU = 0.1  # a detection this close to the target's last box is the target (a person moves little between samples)
BRIDGE_INSIDE = 0.8  # ... or one that lies at least this far inside that box (frame exit: the box shrinks at the edge)
LOST_BUFFER_S = 2.0  # the bridge crosses a hole of this many seconds (10 samples at 5 fps) in the same shot, chaining from the last chosen box; further than this from a matched sample it takes only a box that looks like him (ACCEPT against the gallery)
NEAR_S = 1.0  # a detection that shows up out of an empty sample is him when it lies near his last box (centre distance) within this many seconds of the last chosen sample
NEAR_DIST = 0.25  # ... "near": the centres are this far apart at most, as a fraction of the frame (a walker does not cover more in a second)
NEAR_SIZE = 3.0  # ... and the box is no more than this many times larger or smaller (by height) than his last one
NEAR_MIN_SCORE = 0.3  # ... and scores at least the exit threshold of pipeline.DETECT_EXIT: under it the segment logic would not count it anyway
NEAR_LOOK = 0.5  # ... and its look does not contradict him: a box that only the near rule takes needs this similarity to the gallery (ACCEPT is for boxes far from the last match); no signature or no gallery = no evidence either way
STITCH_GAP_S = 2.0  # a short track starting this soon after an accepted one ended (or ending this soon before one starts) is stitched on
STITCH_IOU = 0.3  # ... if its first (last) box overlaps the accepted track's last (first) box this much, or lies BRIDGE_INSIDE inside it
OVERLAP_MAX = 2  # samples a candidate track may share with an accepted track


class TapError(ValueError):
    """The tap cannot be turned into a target. The message says what to do."""


@dataclass
class Tap:
    t: float
    point: tuple[float, float] | None = None  # (x, y) fractions of the frame
    box: tuple[float, float, float, float] | None = None
    track_id: int | None = None  # the id `inframe pick` saw (informational: ids differ between runs)


def is_point_spec(spec: str) -> bool:
    """True for a `T,x,y` tap: three comma-separated parts that are not the path of an existing file (a target.json whose
    path has exactly two commas is a file)."""
    spec = spec.strip()
    if len(spec.split(",")) != 3:
        return False
    try:
        return not Path(spec).expanduser().is_file()
    except OSError:
        return True


def parse_spec(spec: str) -> list[Tap]:
    """`T,x,y` (seconds, then x and y as fractions 0..1), or the path of a target.json written by `inframe pick`."""
    spec = spec.strip()
    parts = spec.split(",")
    if is_point_spec(spec):
        try:
            t, x, y = (float(v) for v in parts)
        except ValueError:
            raise TapError(f"tap:{spec} is not T,x,y with numbers (seconds, then x and y from 0 to 1)") from None
        if not (0 <= t < float("inf") and 0 <= x <= 1 and 0 <= y <= 1):
            raise TapError(f"tap:{spec}: T must be 0 or more and x, y fractions of the frame from 0 to 1")
        return [Tap(t, point=(x, y))]
    p = Path(spec).expanduser()
    if not p.is_file():
        raise TapError(f"tap:{spec}: not a file, and not T,x,y")
    try:
        d = json.loads(p.read_text())
        items = d["taps"] if "taps" in d else [d]
        taps = []
        for it in items:
            box = tuple(float(v) for v in it["box"]) if it.get("box") else None
            pt = tuple(float(v) for v in it["point"]) if it.get("point") else None
            if box is None and pt is None:
                raise KeyError("box")
            taps.append(Tap(float(it["time"]), pt, box, it.get("track_id")))
    except (ValueError, KeyError, TypeError):
        raise TapError(f"{p} is not a target.json written by `inframe pick` (needs time and box, or a list of taps)") from None
    if not taps:
        raise TapError(f"{p} holds no taps")
    return taps


def recorded_models(specs: list[str]) -> dict[str, str | None]:
    """{detector model: preset name} that the target.json files among the `--target` specs were picked with (`inframe pick`
    records them). Specs without a file, and files without a record (older, or written by list_people), say nothing."""
    out: dict[str, str | None] = {}
    for s in specs:
        kind, _, rest = s.partition(":")
        if kind != "tap" or is_point_spec(rest):
            continue
        try:
            d = json.loads(Path(rest.strip()).expanduser().read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("model"), str):
            out.setdefault(d["model"], d.get("preset") if isinstance(d.get("preset"), str) else None)
    return out


def resolve_preset(specs: list[str] | None, requested: str | None) -> str:
    """The preset `cut` runs with. A tap box was found by the detector of the preset `pick` used, and the cut must find it again
    with that same detector: another model may not see the person at all (or at another size), so `tap` would fail with
    "no person detected". No preset asked for = the one the target.json records (else balanced); a preset with another model
    than the recorded one is refused with a message that says so."""
    from .pipeline import PRESETS

    rec = recorded_models(list(specs or []))
    if len(rec) > 1:
        raise TapError(f"the target files were picked with different detectors ({', '.join(sorted(rec))}); pick again with one --preset")
    if not rec:
        return requested or "balanced"
    model, name = next(iter(rec.items()))
    if requested is not None:
        if requested in PRESETS and PRESETS[requested].model == model:
            return requested
        raise TapError(f"the target.json was picked with the {name or model} preset (detector {model}); --preset {requested} would look for the "
                       f"tapped person with another detector and may not find them. Use --preset {name or 'the same one'}, or pick again with `inframe pick --preset {requested}`")
    if name in PRESETS and PRESETS[name].model == model:
        return name
    return next((n for n, p in PRESETS.items() if p.model == model and p.detect_fps), "balanced")


def parse_targets(specs: list[str]) -> "TapTarget":
    """`--target` values: each `tap:<file or T,x,y>`. Several taps are several appearances of the same person."""
    taps: list[Tap] = []
    for s in specs:
        kind, _, rest = s.partition(":")
        if kind != "tap" or not rest:
            raise TapError(f"--target {s!r}: the only kind so far is tap:<target.json> or tap:T,x,y")
        taps += parse_spec(rest)
    return TapTarget(taps)


# ---------------------------------------------------------------- appearance
_GRAYS, _HUES = 4, 8
SIG_LEN = 2 * (_GRAYS + _HUES * 2)  # upper and lower body, each: grey levels, then hue x (dark, light)


def _region_hist(px: np.ndarray) -> np.ndarray:
    """Histogram of a (n, 3) uint8 block: low-saturation pixels by brightness, the rest by hue and dark/light."""
    x = px.astype(np.float32) / 255.0
    mx, mn = x.max(1), x.min(1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
    dn = np.maximum(d, 1e-6)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    h = np.where(mx == r, ((g - b) / dn) % 6, np.where(mx == g, (b - r) / dn + 2, (r - g) / dn + 4)) / 6.0
    grey = s < 0.2
    out = np.zeros(_GRAYS + _HUES * 2, np.float64)
    gi = np.minimum((mx[grey] * _GRAYS).astype(int), _GRAYS - 1)
    out[:_GRAYS] = np.bincount(gi, minlength=_GRAYS)
    hi = np.minimum((h[~grey] * _HUES).astype(int), _HUES - 1) * 2 + (mx[~grey] >= 0.5)
    out[_GRAYS:] = np.bincount(hi, minlength=_HUES * 2)
    return out / max(len(px), 1)


def crop_signature(frame: np.ndarray, box: tuple[float, float, float, float], person: bool = True) -> np.ndarray | None:
    """Colour signature of the clothes in a person box: upper and lower body, centre 70 % of the width. Any other class
    (`person=False`) has no clothes: the signature is the same two-band histogram over the centre 80 % of the box, so the head
    and the legs of an animal and the sides of a car fall in the two bands alike."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * w, (y1 - y0) * h
    if bw < 6 or bh < 12:
        return None
    trim, bands = (0.15, ((0.12, 0.5), (0.5, 0.9))) if person else (0.10, ((0.10, 0.5), (0.5, 0.90)))
    cx0, cx1 = int((x0 + trim * (x1 - x0)) * w), int((x1 - trim * (x1 - x0)) * w)
    parts = []
    for a, b in bands:
        ry0, ry1 = int((y0 + a * (y1 - y0)) * h), int((y0 + b * (y1 - y0)) * h)
        blk = frame[max(ry0, 0):max(ry1, 0), max(cx0, 0):max(cx1, 0)]
        if blk.size == 0:
            return None
        blk = blk[:: max(1, blk.shape[0] // 24), :: max(1, blk.shape[1] // 12)]
        parts.append(_region_hist(blk.reshape(-1, 3)))
    return np.concatenate(parts)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection of two signatures, 0 (nothing alike) to 1 (same)."""
    return float(np.minimum(a, b).sum() / 2)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _inside(inner, outer) -> float:
    """Fraction of the area of `inner` that lies inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / a if a > 0 else 0.0


def _continues(last, box) -> bool:
    return iou(last, box) >= BRIDGE_IOU or _inside(box, last) >= BRIDGE_INSIDE


def _near(last, box) -> bool:
    """Same size class and close by: `box` is plausibly `last` after a short hole, although the two hardly overlap."""
    cx, cy = (last[0] + last[2]) / 2 - (box[0] + box[2]) / 2, (last[1] + last[3]) / 2 - (box[1] + box[3]) / 2
    ha, hb = max(last[3] - last[1], 1e-6), max(box[3] - box[1], 1e-6)
    return (cx * cx + cy * cy) ** 0.5 <= NEAR_DIST and 1 / NEAR_SIZE <= hb / ha <= NEAR_SIZE


def _looks_like(sig, gallery) -> bool:
    return bool(gallery) and sig is not None and max(similarity(sig, g) for g in gallery) >= ACCEPT


def _coarse(sig: np.ndarray) -> np.ndarray:
    """The signature with the bins that a change of light moves between merged: dark and light of a hue are one bin, and the grey
    levels are smoothed into their neighbours, so a box 20 % darker (shade, exposure) still reads as the same clothes."""
    out = []
    for r in sig.reshape(2, -1):
        g = np.pad(r[:_GRAYS], 1, mode="edge")  # clamp at the ends: the extremes keep their mass
        g = 0.25 * g[:-2] + 0.5 * g[1:-1] + 0.25 * g[2:]
        out.append(np.concatenate([g, r[_GRAYS:].reshape(_HUES, 2).sum(1)]))
    return np.concatenate(out)


def _not_contradicting(sig, gallery, last_sig=None) -> bool:
    """True unless the box's look is clearly somebody else's: it is compared, after a light-tolerant coarsening (_coarse), with the
    gallery (the mean look of his tracks) and with the look of the box he was in just before the hole, which has the light and pose
    that the track mean has lost. Nothing to compare with, or no signature: no contradiction."""
    refs = [g for g in (list(gallery or []) + [last_sig]) if g is not None]
    if sig is None or not refs:
        return True
    c = _coarse(sig)
    return max(similarity(c, _coarse(g)) for g in refs) >= NEAR_LOOK


def _continuity(last, box) -> float:
    return iou(last, box) + _inside(box, last)


# ---------------------------------------------------------------- the target
@dataclass
class _Sample:
    t: float
    cands: list[Detection]
    sigs: list[np.ndarray | None]


@dataclass
class TapTarget:
    """Stateful target (target.py): records every sample, and `finalize()` turns the run into hits once the whole video
    has been seen, so a person is also found before the tap and again after a gap."""

    taps: list[Tap]
    name: str = "tap"
    cls: int = PERSON  # the COCO class of the thing followed (classes.py): `--target class:dog --target tap:...` follows one dog
    wants_tracks: bool = True
    wants_slices: bool = True  # the pipeline runs the detector on slices of the frame too (detect.SlicedDetector): small people
    samples: list[_Sample] = field(default_factory=list)
    anchors: dict[int, tuple[int, int]] = field(default_factory=dict)  # tap index -> (sample, candidate)
    cuts: list[int] = field(default_factory=list)  # index of the first sample of every shot after a hard cut
    info: dict = field(default_factory=dict)

    # -- per sample
    def on_cut(self) -> None:
        """The pipeline saw a hard cut before the sample it is about to pass in: nothing carries over from the earlier samples."""
        self.cuts.append(len(self.samples))

    def shot_of(self, si: int) -> int:
        return sum(1 for c in self.cuts if c <= si)

    def evaluate(self, t: float, frame: np.ndarray, candidates: list[Detection]) -> Hit:
        self.samples.append(_Sample(t, list(candidates), [crop_signature(frame, d.box) if self.cls == PERSON else crop_signature(frame, d.box, False) for d in candidates]))
        for i, tap in enumerate(self.taps):  # fail early: the run need not go on once a tap has no detection to land on
            if i not in self.anchors and t > tap.t + WINDOW_S + 1e-6:
                self.anchors[i] = self._resolve(tap)
        return Hit(0.0, None)  # provisional; finalize() decides

    def _resolve(self, tap: Tap) -> tuple[int, int]:
        near = sorted((abs(s.t - tap.t), si) for si, s in enumerate(self.samples) if abs(s.t - tap.t) <= WINDOW_S + 1e-6)
        if near:  # stay in the shot of the sample nearest the tap: a detection after a cut is somebody else
            shot = self.shot_of(near[0][1])
            near = [(dt, si) for dt, si in near if self.shot_of(si) == shot]
        matches = []
        for dt, si in near:
            hits = []
            for ci, d in enumerate(self.samples[si].cands):
                if tap.box is not None:
                    o = iou(tap.box, d.box)
                    if o >= MIN_IOU:
                        hits.append((-o, ci))
                else:
                    x, y = tap.point
                    m = POINT_MARGIN
                    if d.box[0] - m <= x <= d.box[2] + m and d.box[1] - m <= y <= d.box[3] + m:
                        hits.append(((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]), ci))  # the smallest box that holds the point
            if hits:
                matches.append((si, min(hits)[1]))
        if not matches:
            what = "box" if tap.box is not None else "point"
            from .classes import class_name

            raise TapError(f"no {class_name(self.cls)} detected within {WINDOW_S:g} s of the tap at {tap.t:.1f} s (nothing at that {what}); "
                           f"pick another moment or position with `inframe pick`")
        with_id = [m for m in matches if self.samples[m[0]].cands[m[1]].track_id is not None]
        return (with_id or matches)[0]

    def _units(self):
        """(units, unit_of, have_ids): a unit is one track id, or (no tracker) one detection; unit -> [(sample, candidate)]."""
        S = self.samples
        have_ids = any(d.track_id is not None for s in S for d in s.cands)
        units: dict = {}
        for si, s in enumerate(S):
            for ci, d in enumerate(s.cands):
                if d.track_id is not None:
                    units.setdefault(("id", d.track_id), []).append((si, ci))
                elif not have_ids:
                    units[("det", si, ci)] = [(si, ci)]
        return units, {m: u for u, ms in units.items() for m in ms}, have_ids

    def _bridge(self, chosen: list, gallery: list | None = None) -> None:
        """Fill holes in the target's presence from the detections that continue its last box. Walks forwards and backwards;
        from the last chosen box (a matched one, or one bridged earlier: the chain follows him) it takes, in each of the next
        samples of the same shot, the detection that overlaps it (BRIDGE_IOU) or lies mostly inside it (BRIDGE_INSIDE), for
        up to LOST_BUFFER_S since the last chosen sample and, so that a stranger who keeps overlapping is not followed for ever,
        up to LOST_BUFFER_S since the last matched (not bridged) sample, unless the box looks like him (signature against the gallery,
        ACCEPT: he is small or far away and has no track id, and his clothes tell him from a stranger who keeps overlapping). Only a detection that is not in an accepted track can be taken, and
        a track that is long enough to have been judged by its look (MIN_TRACK_SAMPLES) and was rejected is somebody else;
        the boxes of tracks shorter than that were never judged, and are taken when they continue his box."""
        S = self.samples
        gap = max(1, int(round(LOST_BUFFER_S / self._dt()))) if len(S) > 1 else 1
        near_gap = max(1, int(round(NEAR_S / self._dt()))) if len(S) > 1 else 1
        lens = Counter(d.track_id for s in S for d in s.cands if d.track_id is not None)
        real = [c is not None for c in chosen]  # the matched samples; a bridged box never counts as one
        for order in (range(len(S)), range(len(S) - 1, -1, -1)):
            last, li, ri, lsig = None, None, None, None  # last chosen box, its sample, the sample of the last matched one, its look
            sig_at = lambda si, d: next((S[si].sigs[ci] for ci, c in enumerate(S[si].cands) if c is d), None)
            step = 1 if order[0] == 0 else -1
            for si in order:
                if last is not None and (abs(si - li) > gap or self.shot_of(si) != self.shot_of(li)):
                    last = None
                if real[si]:
                    last, li, ri, lsig = chosen[si], si, si, sig_at(si, chosen[si])
                    continue
                if last is None or chosen[si] is not None:  # (the latter: bridged by the forward pass)
                    continue
                far = abs(si - ri) > gap  # beyond the cap from a matched sample: only a box that looks like him
                pv = si - step  # a detection that appears after an empty sample is a re-acquisition: it may have moved further than an overlap
                reacquired = abs(si - li) <= near_gap and not (0 <= pv < len(S) and any(c.score >= NEAR_MIN_SCORE for c in S[pv].cands))
                cands = [d for ci, d in enumerate(S[si].cands)
                         if (d.track_id is None or lens.get(d.track_id, 0) < MIN_TRACK_SAMPLES)
                         and (_continues(last.box, d.box)
                              or (reacquired and d.score >= NEAR_MIN_SCORE and _near(last.box, d.box) and _not_contradicting(S[si].sigs[ci], gallery, lsig)))
                         and (not far or _looks_like(S[si].sigs[ci], gallery))]
                if cands:
                    chosen[si] = max(cands, key=lambda d: _continuity(last.box, d.box))
                    last, li, lsig = chosen[si], si, sig_at(si, chosen[si])

    def _dt(self) -> float:
        d = sorted(b.t - a.t for a, b in zip(self.samples, self.samples[1:]) if b.t > a.t)
        return d[len(d) // 2] if d else 0.2

    def _stitch(self, units, accepted) -> None:
        """A track that is too short to be judged by its look (under MIN_TRACK_SAMPLES; a longer one that the look rejected is somebody
        else, whatever its position) but starts right after an accepted track ended, in the same shot, where that track's last box was, is the same person: accept it, whatever its length.
        Same for one that ends right before an accepted track starts. No time overlap with the accepted track. Repeats, so a
        chain of short tracks is followed."""
        S = self.samples
        gap = max(1, int(round(STITCH_GAP_S / self._dt())))
        spans = lambda u: (units[u][0][0], units[u][-1][0])
        covered = set()
        for u in accepted:
            covered |= {si for si, _ in units[u]}
        changed = True
        while changed:
            changed = False
            for u in sorted((u for u in units if u[0] == "id" and u not in accepted and len(units[u]) < MIN_TRACK_SAMPLES), key=lambda u: spans(u)[0]):
                a0, a1 = spans(u)
                if len({si for si, _ in units[u]} & covered):
                    continue
                for v in list(accepted):
                    if v[0] != "id":
                        continue
                    b0, b1 = spans(v)
                    if 0 < a0 - b1 <= gap and self.shot_of(a0) == self.shot_of(b1):  # u follows v
                        end, start = S[units[v][-1][0]].cands[units[v][-1][1]].box, S[units[u][0][0]].cands[units[u][0][1]].box
                    elif 0 < b0 - a1 <= gap and self.shot_of(b0) == self.shot_of(a1):  # u comes before v
                        end, start = S[units[u][-1][0]].cands[units[u][-1][1]].box, S[units[v][0][0]].cands[units[v][0][1]].box
                    else:
                        continue
                    if iou(end, start) >= STITCH_IOU or _inside(end, start) >= BRIDGE_INSIDE or _inside(start, end) >= BRIDGE_INSIDE:
                        accepted[u] = None
                        covered |= {si for si, _ in units[u]}
                        changed = True
                        break

    # -- after the last sample
    def finalize(self) -> list[Hit]:
        for i, tap in enumerate(self.taps):
            if i not in self.anchors:
                self.anchors[i] = self._resolve(tap)
        S = self.samples
        units, unit_of, have_ids = self._units()

        def sig_of(u):
            v = [S[si].sigs[ci] for si, ci in units[u] if S[si].sigs[ci] is not None]
            return np.mean(v, axis=0) if v else None

        accepted: dict = {}  # unit -> similarity to the gallery at the time it was accepted (None for a tap)
        gallery: list[np.ndarray] = []
        extra: set[tuple[int, int]] = set()  # tapped detections without a track (the track was not confirmed yet)
        for si, ci in self.anchors.values():
            u = unit_of.get((si, ci))
            if u is None:
                extra.add((si, ci))
                if S[si].sigs[ci] is not None:
                    gallery.append(S[si].sigs[ci])
            else:
                accepted[u] = None
                g = sig_of(u)
                if g is not None:
                    gallery.append(g)
        misses = []  # tracks that were close to being taken as him: reported so a threshold can be judged
        if have_ids:  # the same person again later (or earlier): tracks that look like the gallery and do not share a frame
            covered = set()
            for u in accepted:
                covered |= {si for si, _ in units[u]}
            covered |= {si for si, _ in extra}
            sigs = {u: sig_of(u) for u in units if u not in accepted and len(units[u]) >= MIN_TRACK_SAMPLES}
            sigs = {u: v for u, v in sigs.items() if v is not None}
            # best look-alike first, not earliest first: a stranger who starts sooner must not block the real target
            while gallery and sigs:
                best = None
                for u, sg in sigs.items():
                    sim = max(similarity(sg, g) for g in gallery)
                    need = ACCEPT_LONG if len(units[u]) >= LONG_TRACK else ACCEPT
                    shared = len({si for si, _ in units[u]} & covered)
                    if sim >= need and shared <= OVERLAP_MAX and (best is None or (sim, -units[u][0][0]) > best[:2]):
                        best = (sim, -units[u][0][0], u, sg)
                if best is None:
                    break
                sim, _, u, sg = best
                accepted[u] = sim
                del sigs[u]
                covered |= {si for si, _ in units[u]}
                if sim >= GROW and len(gallery) < GALLERY_MAX:
                    gallery.append(sg)
            for u, sg in sigs.items():
                if gallery:
                    misses.append((max(similarity(sg, g) for g in gallery), u[1], units[u][0][0], units[u][-1][0],
                                   len({si for si, _ in units[u]} & covered)))
            self._stitch(units, accepted)
        elif gallery:  # no tracker: chain by position and look, sample by sample, from the tapped detections
            self._chain(units, gallery, accepted, extra)
        chosen: list[Detection | None] = []
        for si, s in enumerate(S):
            best = None
            for ci, d in enumerate(s.cands):
                u = unit_of.get((si, ci))
                if (u in accepted and u is not None) or (si, ci) in extra:
                    if best is None or d.score > best.score:
                        best = d
            chosen.append(best)
        if have_ids:
            self._bridge(chosen, gallery)
        hits = [Hit(d.score, d.box, d.track_id) if d else Hit(0.0, None) for d in chosen]
        self.info = dict(taps=len(self.taps), tap_times=[round(t.t, 3) for t in self.taps],
                         tracks=sum(1 for u in units if u[0] == "id"), tracks_accepted=sum(1 for u in accepted if u[0] == "id"),
                         tracked=have_ids,
                         accepted_tracks=[dict(id=u[1], sim=None if v is None else round(v, 3), t0=round(S[units[u][0][0]].t, 2),
                                               t1=round(S[units[u][-1][0]].t, 2)) for u, v in accepted.items() if u[0] == "id"][:60],
                         near_misses=[dict(id=i, sim=round(m, 3), t0=round(S[a].t, 1), t1=round(S[b].t, 1), shared_samples=sh)
                                      for m, i, a, b, sh in sorted(misses, reverse=True)[:8]],
                         appearance_accept=ACCEPT, appearance_grow=GROW)
        return hits

    def _chain(self, units, gallery, accepted, extra) -> None:
        """Without track ids: a detection counts when it overlaps the last accepted box (IoU 0.3) and looks like the gallery
        (0.6 of ACCEPT), or looks very much like it (GROW)."""
        S = self.samples
        seeds = sorted({si for si, _ in extra} | {u[1] for u in accepted if u[0] == "det"})
        for direction in (1, -1):
            for start in seeds:
                last = None
                for si in range(start, len(S) if direction == 1 else -1, direction):
                    pick = None
                    for ci, d in enumerate(S[si].cands):
                        sg = S[si].sigs[ci]
                        if sg is None:
                            continue
                        sim = max(similarity(sg, g) for g in gallery)
                        near = last is not None and iou(last, d.box) >= MIN_IOU
                        if (near and sim >= 0.6 * ACCEPT) or sim >= GROW or (si == start and (si, ci) in extra):
                            if pick is None or sim > pick[0]:
                                pick = (sim, ci, d.box)
                    if pick is None:
                        continue
                    accepted[("det", si, pick[1])] = pick[0]
                    last = pick[2]

    def height_frac(self) -> float | None:
        """The smallest tapped box height as a fraction of the frame height (None when no tap carries a box): classes.model_for."""
        hs = [t.box[3] - t.box[1] for t in self.taps if t.box is not None]
        return min(hs) if hs else None

    def describe(self) -> dict:
        return dict(self.info)
