"""Identity by appearance: who is the person in the photos, and which tracks of the run are the same person.

`--target photos:DIR` (photo mode): the target is the person in the photos of DIR. The run is tracked (tracks.py), every
track is embedded (at its start, after a gap, and every EVERY_S seconds), and the tracks are ranked by how much their
crops look like the photos. Two ways to start:
  * one confirm (`inframe pick VIDEO --photos DIR` shows the top 5 as "Is this you?" and writes the confirmed tap into
    target.json): the confirmed track seeds the session gallery;
  * no UI: the best track is taken only when it clears the embedder's threshold and beats every track that shares a frame
    with it by the embedder's margin; otherwise `Ambiguous` (the CLI exits 3) with the ranking in the message.
From the seed the target grows like tap mode does, but with embeddings: the best look-alike track that does not share a
frame with an accepted one joins when it clears the threshold against photos plus session gallery; its confident crops
join the session gallery (deduplicated, capped, the most redundant crop dropped, so different outfits stay).

Each embedder has its own threshold and second-best margin (cosines of different models are not comparable). The numbers
come from sweeps on synthetic composites. The embedders run in a child process (embed_worker.py); "color" is a
no-model embedder (the clothes' colour signature of tap.py) for tests and installs without torch.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import classes
from .classes import PERSON
from .detect import Detection
from .tap import MIN_TRACK_SAMPLES, OVERLAP_MAX, Tap, TapError, TapTarget, crop_signature, is_point_spec, parse_spec, similarity
from .target import AnyOfClass, Hit

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EVERY_S = 1.0  # a track is embedded again after this many seconds (once it has EARLY_CROPS crops)
EARLY_CROPS = 3  # a track is embedded at every sample until it holds this many crops: a track of 3 to 6 samples is judged on them
DECODE_LONG_SIDE = 1280  # photo mode decodes at this long side (the detector squashes to 512 whatever it gets): a person 30 px high at 640 is 60 px here
GAP_S = 0.45  # ... and after it was missing from the samples for longer than this
MIN_CROP_H, MIN_CROP_W = 48, 16  # pixels; smaller boxes are not embedded
CROP_PAD = 0.04
MIN_EMBEDS = 2  # a track needs this many embedded crops to be ranked (1 without a tracker)
GALLERY_CAP = 24
PHOTOS_MAX = 40
TOP_N = 5  # the "Is this you?" list


class Ambiguous(Exception):
    """Photo mode cannot decide alone (the CLI exits 3). The message names the best candidates."""


class PhotoError(TapError):
    """The photo folder cannot be used."""


class EmbedderUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Spec:
    kind: str
    version: str  # stored with any embedding that is kept (T2 saved targets re-embed when it differs)
    threshold: float  # cosine a track must reach against the photos (median over its crops, of the best photo for each crop)
    session: float  # ... or against the session gallery (the confirmed track and what joined it): higher, because that gallery
    #                  is made of crops of the video (same blur, same light), so a stranger gets closer to it than to a photo
    margin: float  # the best must beat a track that shares a frame with it by this much
    grow: float  # a crop this close to the gallery joins the session gallery
    dedup: float  # a crop this close to a gallery member is not added again


# thresholds start from the 1 % false-positive points of a sweep on synthetic composites. dinov2 `threshold` (photos) and `session` (session gallery)
# come from a tile-aware sweep on the composites (t_p 0.52-0.62 x t_g 0.62-0.75): 0.62 / 0.66 gave tile precision 0.996 at the best recall (0.735);
# with t_p under 0.60 the tile precision fell under 0.95
SPECS = {
    "siglip2": Spec("siglip2", "siglip2-base-patch16-224/1", threshold=0.83, session=0.90, margin=0.02, grow=0.87, dedup=0.97),
    "dinov2": Spec("dinov2", "dinov2-small/1", threshold=0.62, session=0.66, margin=0.04, grow=0.62, dedup=0.95),
    "color": Spec("color", "color-signature/1", threshold=0.85, session=0.92, margin=0.05, grow=0.92, dedup=0.98),
}
DEFAULT_EMBEDDER = "dinov2"  # on the composites dinov2 separates the target from strangers; siglip2 cosines of strangers sit at 0.85-0.93


class Embedder:
    """embedder.embed(list of (H, W, 3) uint8 crops) -> (n, dim) float32, rows of length 1. Close it (or use `with`)."""

    def __init__(self, kind: str = DEFAULT_EMBEDDER, backend: str = "auto", offline: bool = False):
        if kind not in SPECS:
            raise ValueError(f"unknown embedder {kind!r}; known: {', '.join(SPECS)}")
        self.kind, self.spec, self.backend, self.proc = kind, SPECS[kind], "inprocess", None
        if kind == "color":
            return
        self.proc = subprocess.Popen([sys.executable, "-m", "inframe.embed_worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.backend = self._ask(("init", kind, backend, offline))["backend"]
        except Exception:
            self.close()
            raise

    def _ask(self, msg):
        try:
            pickle.dump(msg, self.proc.stdin, protocol=4)
            self.proc.stdin.flush()
            tag, val = pickle.load(self.proc.stdout)
        except (BrokenPipeError, EOFError, OSError):
            err = self.proc.stderr.read().decode(errors="replace")[-400:] if self.proc.poll() is not None else ""
            raise EmbedderUnavailable(f"the embedder process stopped ({err.strip() or 'no message'}); "
                                      "is the identity extra installed? (uv sync --extra identity)") from None
        if tag == "error":
            raise EmbedderUnavailable(f"embedder {self.kind}: {val}")
        return val

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 1), np.float32)
        if self.kind == "color":
            out = []
            for c in crops:
                sg = crop_signature(c, (0.0, 0.0, 1.0, 1.0))
                out.append(sg if sg is not None else np.zeros(1))
            n = max(len(v) for v in out)
            m = np.stack([np.pad(v, (0, n - len(v))) for v in out]).astype(np.float32)
            return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        return np.asarray(self._ask(("embed", crops)), np.float32)

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                pickle.dump(("close",), self.proc.stdin, protocol=4)
                self.proc.stdin.flush()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()  # our own child only
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                f.close()
            except OSError:
                pass
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- crops and photos
def person_crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """The box (fractions of the frame) padded a little, as a uint8 array; None when it is too small to say anything."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
    a, b = max(int((x0 - px) * w), 0), max(int((y0 - py) * h), 0)
    c, d = min(int((x1 + px) * w) + 1, w), min(int((y1 + py) * h) + 1, h)
    if d - b < MIN_CROP_H or c - a < MIN_CROP_W:
        return None
    return np.ascontiguousarray(frame[b:d, a:c])


VARIANT_HEIGHTS = (128, 80)  # the photo is also embedded this small: people in a busy video are 60 to 150 px high, photos 500+


def resolution_variants(crop: np.ndarray) -> list[np.ndarray]:
    """The crop, and copies of it reduced to VARIANT_HEIGHTS (when the crop is clearly taller), so a small person in the video is
    compared with a photo at the same blur."""
    out = [crop]
    h = crop.shape[0]
    for vh in VARIANT_HEIGHTS:
        if vh < h * 0.8:
            im = Image.fromarray(crop).resize((max(1, round(crop.shape[1] * vh / h)), vh), Image.LANCZOS)
            out.append(np.asarray(im))
    return out


def photo_files(folder: str | Path) -> list[Path]:
    p = Path(folder).expanduser()
    if not p.is_dir():
        raise PhotoError(f"photos:{folder}: not a folder")
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in PHOTO_EXT and not f.name.startswith("."))
    if not files:
        raise PhotoError(f"photos:{folder}: no photos in it ({', '.join(sorted(PHOTO_EXT))})")
    return files[:PHOTOS_MAX]


def load_photo_crops(folder: str | Path, detect=None, cls: int = PERSON) -> list[np.ndarray]:
    """One crop per photo: the largest object of class `cls` (default: person) the detector finds (`detect(list of rgb) -> list of list of Detection`), or the
    whole photo when there is no detector or it finds nobody (a photo that is already a crop of the person)."""
    out = []
    for f in photo_files(folder):
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except OSError as e:
            raise PhotoError(f"{f.name}: cannot be read as an image ({e})") from None
        arr = np.asarray(im)
        crop = None
        if detect is not None:
            small = np.asarray(im.copy().resize((max(1, round(im.width * 1280 / max(im.size))), max(1, round(im.height * 1280 / max(im.size)))))) \
                if max(im.size) > 1280 else arr
            dets = [d for d in (detect([small]) if cls == PERSON else detect([small], cls))[0] if d.score >= 0.5]
            if dets:
                big = max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                crop = person_crop(arr, big.box)
        if crop is None:
            crop = arr
        if max(crop.shape[:2]) > 768:  # no need for more than the embedders look at
            s = 768 / max(crop.shape[:2])
            crop = np.asarray(Image.fromarray(crop).resize((max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s)))))
        out.append(crop)
    return out


def lazy_detect(size: str = "small"):
    """A `detect(list of rgb, cls=PERSON) -> list of list of Detection` that loads the detector on its first call."""
    holder: list = []

    def run(imgs, cls: int = PERSON):
        if not holder:
            from .detect import load_detector

            holder.append(load_detector(size))
        return holder[0].detect(imgs, cls=cls)

    return run


# ---------------------------------------------------------------- gallery
class Gallery:
    """Embeddings kept apart enough to cover the person: a crop too close to a member is dropped (dedup), and at the cap the
    member most like another one goes (embedding and colour signature both count, so a second outfit is kept over a tenth
    crop of the first)."""

    def __init__(self, spec: Spec, cap: int = GALLERY_CAP):
        self.spec, self.cap = spec, cap
        self.v = np.zeros((0, 1), np.float32)
        self.sig: list[np.ndarray | None] = []
        self.tags: list = []  # what the caller attached to each member (saved targets: which crop it is)

    def __len__(self) -> int:
        return len(self.sig)

    def add(self, vec: np.ndarray, sig: np.ndarray | None = None, tag=None) -> bool:
        if len(self) and self.v.shape[1] == len(vec):
            near = self.v @ vec
            if float(near.max()) >= self.spec.dedup:
                j = int(near.argmax())
                if sig is None or self.sig[j] is None or similarity(sig, self.sig[j]) >= 0.9:
                    return False
        self.v = vec[None].astype(np.float32) if not len(self) else np.vstack([self.v, vec[None]])
        self.sig.append(sig)
        self.tags.append(tag)
        if len(self) > self.cap:
            self._evict()
        return True

    def _evict(self) -> None:
        n = len(self)
        e = self.v @ self.v.T
        s = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                s[i, j] = e[i, j] if self.sig[i] is None or self.sig[j] is None else 0.5 * e[i, j] + 0.5 * similarity(self.sig[i], self.sig[j])
        np.fill_diagonal(s, -1)
        k = int(s.max(1).argmax())
        self.v = np.delete(self.v, k, 0)
        del self.sig[k]
        del self.tags[k]


def unit_score(vecs: np.ndarray, gallery: np.ndarray) -> float:
    """Median over the track's crops of the best cosine to any gallery member."""
    if not len(vecs) or not len(gallery) or vecs.shape[1] != gallery.shape[1]:
        return 0.0
    return float(np.median((vecs @ gallery.T).max(1)))


# ---------------------------------------------------------------- the target
class PhotoTarget(TapTarget):
    """Stateful target for `--target photos:DIR`. `taps` are the confirm(s) of `inframe pick --photos` (they seed the session
    gallery); without them the best track is taken only above the margin (else Ambiguous). `rank_only`: decide nothing, just
    fill `info["ranking"]` (the top TOP_N tracks, for the pick page)."""

    decode_long_side = DECODE_LONG_SIDE  # read by pipeline.score_detect: the frames this target sees are decoded at this long side

    def __init__(self, photos: str, taps: list[Tap] | None = None, embedder: str = DEFAULT_EMBEDDER, backend: str = "auto",
                 offline: bool = False, rank_only: bool = False, detect=None, photo_crops: list[np.ndarray] | None = None,
                 embedder_obj: Embedder | None = None, every_s: float = EVERY_S, cls: int = PERSON):
        super().__init__(list(taps or []), cls=cls)
        self.name, self.photos, self.rank_only, self.every_s = "photos", str(photos), rank_only, every_s
        if embedder not in SPECS:
            raise ValueError(f"unknown embedder {embedder!r}; known: {', '.join(SPECS)}")
        self.spec = SPECS[embedder]
        self.backend_req, self.offline, self.detect = backend, offline, detect
        self.emb, self.own_emb = embedder_obj, embedder_obj is None
        self.photo_crops = photo_crops
        self.photo_vecs: np.ndarray | None = None
        self.vecs: dict[tuple[int, int], np.ndarray] = {}
        self.last_seen: dict[int, float] = {}
        self.last_embed: dict[int, float] = {}
        self.n_crops: dict[int, int] = {}
        self.embedded = 0

    def _start(self) -> None:
        if self.emb is None:
            self.emb = Embedder(self.spec.kind, self.backend_req, self.offline)
        if self.photo_vecs is None and self.photos.startswith("saved:"):
            from . import targets

            self.photo_vecs = targets.vectors_for(self.photos[len("saved:"):], self.emb)  # stored embeddings, or the thumbnails embedded now
        elif self.photo_vecs is None:
            crops = self.photo_crops if self.photo_crops is not None else load_photo_crops(self.photos, self.detect, self.cls)
            self.photo_vecs = self.emb.embed([v for c in crops for v in resolution_variants(c)])

    def on_cut(self) -> None:
        super().on_cut()
        self.last_seen.clear()
        self.last_embed.clear()

    def evaluate(self, t: float, frame: np.ndarray, candidates: list[Detection]) -> Hit:
        hit = super().evaluate(t, frame, candidates)
        if self.photo_vecs is None:
            self._start()
        si = len(self.samples) - 1
        todo, crops = [], []
        for ci, d in enumerate(candidates):
            tid = d.track_id
            if tid is not None:
                due = (tid not in self.last_embed or t - self.last_embed[tid] >= self.every_s - 1e-6 or t - self.last_seen.get(tid, t) > GAP_S
                       or self.n_crops.get(tid, 0) < EARLY_CROPS)
                self.last_seen[tid] = t
                if not due:
                    continue
            c = person_crop(frame, d.box)
            if c is not None:
                todo.append((si, ci))
                crops.append(c)
                if tid is not None:
                    self.last_embed[tid] = t
                    self.n_crops[tid] = self.n_crops.get(tid, 0) + 1
        if crops:
            for key, v in zip(todo, self.emb.embed(crops)):
                self.vecs[key] = v
            self.embedded += len(crops)
        return hit

    # -- after the last sample
    def finalize(self) -> list[Hit]:
        try:
            return self._decide()
        finally:
            if self.own_emb and self.emb is not None:
                self.emb.close()

    def _decide(self) -> list[Hit]:
        for i, tap in enumerate(self.taps):
            if i not in self.anchors:
                self.anchors[i] = self._resolve(tap)
        S, spec = self.samples, self.spec
        units, unit_of, have_ids = self._units()
        need = MIN_EMBEDS if have_ids else 1
        ovl = OVERLAP_MAX if have_ids else 0  # without a tracker a unit is one detection: one frame holds one of him
        uv = {u: np.stack([self.vecs[m] for m in ms if m in self.vecs]) for u, ms in units.items() if any(m in self.vecs for m in ms)}
        frames = {u: {si for si, _ in units[u]} for u in uv}
        seeds = {unit_of[m] for m in self.anchors.values() if m in unit_of}
        uv = {u: v for u, v in uv.items() if len(v) >= need or u in seeds}
        P = self.photo_vecs if self.photo_vecs is not None else np.zeros((0, 1), np.float32)
        base = {u: unit_score(v, P) for u, v in uv.items()}

        def together(a, b) -> bool:
            return len(frames[a] & frames[b]) > ovl

        def best_time(u):
            """(sample, candidate) of the crop of unit u that looks most like the photos."""
            keys = [m for m in units[u] if m in self.vecs]
            return max(keys, key=lambda m: float((P @ self.vecs[m]).max()) if len(P) and P.shape[1] == len(self.vecs[m]) else 0.0)

        def who(u) -> str:
            """'track 3', or for a unit without a tracker (one detection) 'the detection at 12.4 s'; never 'track None' or a sample index."""
            return f"track {u[1]}" if u[0] == "id" else f"the detection at {S[u[1]].t:.1f} s"

        order = sorted(uv, key=lambda u: (-base[u], units[u][0][0]))
        ranking = []
        for r, u in enumerate(order[:TOP_N], 1):
            si, ci = best_time(u)
            d = S[si].cands[ci]
            ranking.append(dict(rank=r, track_id=u[1] if u[0] == "id" else None, score=round(base[u], 4), n_crops=len(uv[u]),
                                t0=round(S[units[u][0][0]].t, 2), t1=round(S[units[u][-1][0]].t, 2), t=round(S[si].t, 3),
                                box=[round(v, 4) for v in d.box]))
        self.info = dict(embedder=spec.kind, embedder_version=spec.version, threshold=spec.threshold, session_threshold=spec.session, margin=spec.margin,
                         photos=len(P), crops_embedded=self.embedded, tracks=len(units) if have_ids else None, tracked=have_ids,
                         ranking=ranking,
                         scores=[dict(id=u[1] if u[0] == "id" else None, score=round(base[u], 4), n=len(uv[u]), t0=round(S[units[u][0][0]].t, 2),
                                      t1=round(S[units[u][-1][0]].t, 2), box=[round(v, 3) for v in S[units[u][len(units[u]) // 2][0]].cands[units[u][len(units[u]) // 2][1]].box],
                                      tm=round(S[units[u][len(units[u]) // 2][0]].t, 2)) for u in order[:80]])
        if self.rank_only:
            return [Hit(0.0, None) for _ in S]

        accepted: dict = {}
        sess = Gallery(spec)
        covered: set[int] = set()
        extra: set[tuple[int, int]] = set()

        def take(u, score, confident_only: bool):
            accepted[u] = score
            covered.update(frames[u])
            for k, m in enumerate(units[u]):
                if m not in self.vecs:
                    continue
                v = self.vecs[m]
                if confident_only and not (len(P) and float((P @ v).max()) >= spec.grow or len(sess) and float((sess.v @ v).max()) >= spec.grow):
                    continue
                sess.add(v, S[m[0]].sigs[m[1]])

        if self.taps:  # the confirm
            for m in self.anchors.values():
                u = unit_of.get(m)
                if u is not None and u in uv:
                    take(u, None, False)
                else:
                    extra.add(m)
                    if m in self.vecs:
                        sess.add(self.vecs[m], S[m[0]].sigs[m[1]])
                    covered.add(m[0])
        else:  # no UI: only above the margin
            if not order or base[order[0]] < spec.threshold:
                top = ", ".join(f"{'track ' + str(r['track_id']) if r['track_id'] is not None else 'a detection'} {r['t0']:.0f}-{r['t1']:.0f} s ({r['score']:.2f})" for r in ranking[:3]) or "none"
                raise Ambiguous(f"nobody in the video looks like the photos well enough ({spec.kind} needs {spec.threshold:g}; best: {top}). "
                                f"Run `inframe pick VIDEO --photos {self.photos}` and confirm the person, or use --target tap:...")
            u0 = order[0]
            rivals = [w for w in order[1:] if together(u0, w) and base[w] >= base[u0] - spec.margin]
            if rivals:
                raise Ambiguous(f"{who(u0)} ({base[u0]:.2f}) and {who(rivals[0])} ({base[rivals[0]]:.2f}) share a frame and are within "
                                f"the {spec.kind} margin of {spec.margin:g}. Run `inframe pick VIDEO --photos {self.photos}` and confirm the person.")
            take(u0, base[u0], False)

        ambiguous = []
        # The growth loop. A unit is him when the median over its crops of the best cosine to a PHOTO reaches spec.threshold, or the
        # same median against the SESSION gallery reaches spec.session (higher: crops of the video look alike to a stranger sooner than
        # a photo does). z = the better margin over the two thresholds orders the units and decides the rivals. Each pass used to
        # rescore every unit; now the per-crop best is kept and only the gallery rows that arrived since the last pass are folded
        # in; a full recompute happens only when the gallery evicted a row.
        rem = [u for u in uv if u not in accepted]
        dims = Counter(uv[u].shape[1] for u in rem)
        D = dims.most_common(1)[0][0] if dims else 0
        rem = [u for u in rem if uv[u].shape[1] == D]  # another width never matches the gallery (unit_score gives 0)
        if rem:
            n = len(rem)
            allV = np.vstack([uv[u] for u in rem])
            counts = np.array([len(uv[u]) for u in rem])
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            single, multi = counts == 1, np.nonzero(counts > 1)[0]
            first = np.array([units[u][0][0] for u in rem])
            by_frame: dict[int, list[int]] = {}
            for i, u in enumerate(rem):
                for si in frames[u]:
                    by_frame.setdefault(si, []).append(i)
            ovl_cnt = np.array([len(frames[u] & covered) for u in rem])  # frames each unit shares with the accepted ones
            alive = np.ones(n, bool)

            def per_unit(cropmax: np.ndarray) -> np.ndarray:
                sc = np.zeros(n)
                sc[single] = cropmax[starts[single]]
                for i in multi:
                    sc[i] = np.median(cropmax[starts[i]:starts[i] + counts[i]])
                sc[~np.isfinite(sc)] = -np.inf  # empty gallery
                return sc

            from_photos = (allV @ P.T).max(1) if len(P) and P.shape[1] == D else np.full(len(allV), -np.inf, np.float32)
            pm = per_unit(from_photos)  # the photos never change
            scrop, folded = np.full(len(allV), -np.inf, np.float32), np.zeros((0, D), np.float32)
            while True:
                cur = sess.v if len(sess) and sess.v.shape[1] == D else folded[:0]
                if len(cur) >= len(folded) and np.array_equal(cur[:len(folded)], folded):
                    if len(cur) > len(folded):
                        scrop = np.maximum(scrop, (allV @ cur[len(folded):].T).max(1))
                else:  # the gallery evicted a row: its crops' best may have dropped
                    scrop = (allV @ cur.T).max(1) if len(cur) else np.full(len(allV), -np.inf, np.float32)
                folded = cur.copy()
                sm = per_unit(scrop)
                z = np.maximum(pm - spec.threshold, sm - spec.session)
                sc = np.maximum(pm, sm)  # the cosine reported for a unit
                ci = np.nonzero(alive & (z >= 0) & (ovl_cnt <= ovl))[0]
                if not len(ci):
                    break
                k = int(ci[np.lexsort((first[ci], -z[ci]))[0]])
                u = rem[k]
                shared = Counter(j for si in frames[u] for j in by_frame[si])
                # a rival must be able to be him: a unit that shares frames with an accepted one never is, so it cannot veto
                rivals = sorted(j for j, c in shared.items() if j != k and c > ovl and alive[j] and ovl_cnt[j] <= ovl
                                and z[j] >= 0 and z[k] - z[j] < spec.margin)
                alive[k] = False
                if rivals:  # two look-alikes in one frame and no way to tell: neither is taken
                    w = rem[rivals[0]]
                    ambiguous.append(dict(track_id=u[1] if u[0] == "id" else None, score=round(float(sc[k]), 3),
                                          other=w[1] if w[0] == "id" else None, other_score=round(float(sc[rivals[0]]), 3)))
                    alive[rivals] = False
                    continue
                for si in frames[u] - covered:
                    ovl_cnt[by_frame[si]] += 1
                take(u, float(sc[k]), True)

        chosen: list[Detection | None] = []
        for si, s in enumerate(S):
            best = None
            for ci, d in enumerate(s.cands):
                u = unit_of.get((si, ci))
                if (u is not None and u in accepted) or (si, ci) in extra:
                    if best is None or d.score > best.score:
                        best = d
            chosen.append(best)
        if have_ids:
            self._bridge(chosen)
        self.info.update(tracks_accepted=sum(1 for u in accepted if u[0] == "id"), session_gallery=len(sess), ambiguous=ambiguous[:8],
                         accepted_tracks=[dict(id=u[1], score=None if v is None else round(v, 3), t0=round(S[units[u][0][0]].t, 2),
                                               t1=round(S[units[u][-1][0]].t, 2)) for u, v in accepted.items() if u[0] == "id"][:60])
        return [Hit(d.score, d.box, d.track_id) if d else Hit(0.0, None) for d in chosen]


# ---------------------------------------------------------------- --target parsing
def parse_targets(specs: list[str], embedder: str = DEFAULT_EMBEDDER, offline: bool = False, detect=None):
    """`--target` values: tap:<target.json | T,x,y>, photos:<folder> and class:<name>. A target.json written by `inframe pick --photos` names
    its folder, so `--target tap:target.json` alone is photo mode with the confirm; one written by `inframe pick --class dog` names its class.
    `class:dog` alone keeps every dog (AnyOfClass); with a tap it follows the one dog that was tapped (tap mode for any class)."""
    taps: list[Tap] = []
    photos: str | None = None
    cls: int | None = None

    def set_class(name: str) -> None:
        nonlocal cls
        try:
            c = classes.class_id(name)
        except classes.UnknownClass as e:
            raise TapError(str(e)) from None
        if cls is not None and cls != c:
            raise TapError(f"--target names two classes ({classes.class_name(cls)} and {classes.class_name(c)}); one run follows one class")
        cls = c

    for s in specs:
        kind, _, rest = s.partition(":")
        if kind == "photos" and rest:
            if photos is not None and photos.startswith("saved:"):
                raise TapError("--target: give one photos:<folder> or saved:<name>, not several")
            photos = rest
        elif kind == "saved" and rest:
            from . import targets

            if photos is not None:
                raise TapError("--target: give one photos:<folder> or saved:<name>, not several")
            targets.check_usable(rest)  # fail before the run when the target is missing or stale
            photos = f"saved:{rest}"
        elif kind == "class" and rest:
            set_class(rest)
        elif kind == "tap" and rest:
            taps += parse_spec(rest)
            f = Path(rest.strip()).expanduser()
            if not is_point_spec(rest) and f.is_file():
                try:
                    d = json.loads(f.read_text())
                    pfolder, pcls = d.get("photos"), d.get("class")
                except (ValueError, AttributeError):
                    pfolder = pcls = None
                if pfolder and photos is None and str(pfolder).startswith("saved:"):
                    from . import targets

                    targets.check_usable(str(pfolder)[len("saved:"):])  # outside the try: a missing or stale saved target must reach the caller
                photos = photos or pfolder
                if pcls:
                    set_class(str(pcls))  # outside the try: TapError is a ValueError and must reach the caller
        else:
            raise TapError(f"--target {s!r}: use tap:<target.json> or tap:T,x,y, photos:<folder of photos of the person>, saved:<name> (see `inframe target`), "
                           "or class:<name> (a COCO class such as dog or car)")
    c = PERSON if cls is None else cls
    if photos is None:
        if not taps and cls is not None:
            return AnyOfClass(c)
        return TapTarget(taps, cls=c)
    if not photos.startswith("saved:"):
        photo_files(photos)  # fail before the run when the folder is unusable
    return PhotoTarget(photos, taps, embedder, offline=offline, detect=detect, cls=c)
