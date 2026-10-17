"""Saved targets (T2): a person kept by name, so `--target saved:NAME` finds them in any video with no tap.

A saved target is a folder in the targets store (`$INFRAME_TARGETS`, else ~/Library/Application Support/inframe/targets on macOS,
$XDG_DATA_HOME/inframe/targets elsewhere):

    NAME/target.json         format, name, dates, the crops (file, where they came from) and the stored embeddings
    NAME/crops/c0001.jpg     thumbnails of the person (longest side <= THUMB_SIDE); never video, never a whole frame of a video
    NAME/emb/<version>.npz   the embeddings of the crops (and their small copies), one file per embedder version

The embeddings are keyed by the embedder's version string (identity.SPECS[kind].version). A target saved with one embedder
is used with another by embedding the thumbnails again (the result is stored next to the old one); only thumbnails that are
missing or unreadable make a target stale (StaleTarget, the CLI exits 3), and then it has to be added to again.

Crops come from `photos:DIR` (one person per photo), `tap:<target.json>` (the frame of the tap, read from the video it names) or
`run:<segments.json>` (a finished run that kept one person: crops spread over the time it was in frame). Every addition is
deduplicated against what is stored, capped at TARGET_CAP, and chosen so that different outfits stay (identity.Gallery).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import fcntl
except ImportError:  # not on Windows; there the store is not locked
    fcntl = None

from . import identity
from .identity import Embedder, Gallery, SPECS, person_crop, resolution_variants
from .tap import TapError, crop_signature, is_point_spec, parse_spec

FORMAT = 1
TARGET_CAP = 32  # crops kept per target
THUMB_SIDE = 512  # longest side of a stored crop, pixels
RUN_CANDIDATES = 48  # crops looked at when a run is the source
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ZIP_MAX_BYTES = 64 * 1024 * 1024
ZIP_MAX_FILES = 200
_ZIP_OK = re.compile(r"^(target\.json|crops/c\d{4}\.jpg|emb/[A-Za-z0-9._-]{1,80}\.npz)$")


class TargetError(TapError):
    """A saved target cannot be used or changed. The message says what to do. (A TapError, so the CLI and MCP report it as a bad --target.)"""


class StaleTarget(TargetError):
    """The stored crops are missing or unreadable, so no embedding can be made from them (the CLI exits 3)."""


# ---------------------------------------------------------------- the store
def store_dir() -> Path:
    base = os.environ.get("INFRAME_TARGETS")
    if base:
        return Path(base).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "inframe" / "targets"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "inframe" / "targets"


def check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise TargetError(f"target name {name!r}: use 1 to 64 letters, digits, - or _, starting with a letter or digit")
    return name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vfile(version: str) -> str:
    return "emb/" + re.sub(r"[^A-Za-z0-9._-]", "_", version) + ".npz"


@contextlib.contextmanager
def _locked(name: str):
    """Hold the store's lock for target `name` (a file next to its folder, so removing the folder does not touch it) while its
    target.json is read, changed and written back; CLI runs and MCP calls on the same store wait for each other."""
    if fcntl is None:
        yield
        return
    root = store_dir()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f".{name}.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Saved:
    """One saved target on disk."""

    def __init__(self, name: str, root: Path | None = None):
        self.name = check_name(name)
        self.dir = (root or store_dir()) / name
        f = self.dir / "target.json"
        if not f.is_file():
            raise TargetError(f"no saved target named {name!r} (see `inframe target list`)")
        try:
            self.doc = json.loads(f.read_text())
            assert self.doc["format"] == FORMAT and isinstance(self.doc["crops"], list) and isinstance(self.doc.get("embeddings", {}), dict)
            assert all(isinstance(c, dict) and isinstance(c["file"], str) and isinstance(c.get("source", ""), str) for c in self.doc["crops"])
            assert all(isinstance(e, dict) and isinstance(e["file"], str) for e in self.doc.get("embeddings", {}).values())
        except (ValueError, KeyError, TypeError, AssertionError, AttributeError):
            raise TargetError(f"{f} is not a saved target of this version of inframe") from None
        self.doc.setdefault("embeddings", {})

    @classmethod
    def blank(cls, name: str, d: Path, doc: dict) -> "Saved":
        """A Saved for a target whose target.json is about to be written."""
        s = cls.__new__(cls)
        s.name, s.dir, s.doc = name, d, doc
        return s

    @property
    def crops(self) -> list[dict]:
        return self.doc["crops"]

    def crop_paths(self) -> list[Path]:
        out = []
        for c in self.crops:
            p = (self.dir / c["file"]).resolve()
            if (self.dir.resolve() not in p.parents) or not p.is_file():
                raise StaleTarget(f"saved target {self.name!r}: the thumbnail {c['file']} is missing, so it cannot be embedded again; "
                                  f"add the person again with `inframe target add {self.name} --from ...`")
            out.append(p)
        return out

    def readable_crops(self) -> tuple[list[dict], list[np.ndarray]]:
        """(entries, images) of the thumbnails that are still there and can be read; the others are left out."""
        entries, imgs = [], []
        root = self.dir.resolve()
        for c in self.crops:
            p = (self.dir / c["file"]).resolve()
            if root not in p.parents or not p.is_file():
                continue
            try:
                with Image.open(p) as im:
                    imgs.append(np.asarray(im.convert("RGB")))
            except OSError:
                continue
            entries.append(c)
        return entries, imgs

    def load_crops(self) -> list[np.ndarray]:
        out = []
        for p in self.crop_paths():
            try:
                with Image.open(p) as im:
                    out.append(np.asarray(im.convert("RGB")))
            except OSError as e:
                raise StaleTarget(f"saved target {self.name!r}: {p.name} cannot be read ({e}); add the person again with "
                                  f"`inframe target add {self.name} --from ...`") from None
        if not out:
            raise StaleTarget(f"saved target {self.name!r} holds no crops; add some with `inframe target add {self.name} --from ...`")
        return out

    def vectors(self, version: str) -> np.ndarray | None:
        """The stored embeddings for `version` (rows of the crops' variants), or None when there are none that match the crops."""
        e = self.doc["embeddings"].get(version)
        if not e or e.get("n_crops") != len(self.crops):
            return None
        try:
            with np.load(self.dir / e["file"]) as z:
                return z["vecs"].astype(np.float32)
        except (OSError, ValueError, KeyError):
            return None

    def stale_versions(self) -> list[str]:
        """Stored versions no embedder uses any more."""
        current = {s.version for s in SPECS.values()}
        return sorted(v for v in self.doc["embeddings"] if v not in current)

    def summary(self) -> dict:
        cur = {s.version: k for k, s in SPECS.items()}
        emb = {v: dict(embedder=cur.get(v), current=v in cur, valid=e.get("n_crops") == len(self.crops)) for v, e in self.doc["embeddings"].items()}
        return dict(name=self.name, n_crops=len(self.crops), created=self.doc.get("created"), updated=self.doc.get("updated"),
                    sources=sorted({c.get("source", "?").split(":")[0] for c in self.crops}), embeddings=emb, path=str(self.dir))

    def _write(self) -> None:
        self.doc["updated"] = _now()
        tmp = self.dir / "target.json.tmp"
        tmp.write_text(json.dumps(self.doc, indent=1))
        os.replace(tmp, self.dir / "target.json")

    def store_vectors(self, version: str, vecs: np.ndarray) -> None:
        (self.dir / "emb").mkdir(exist_ok=True)
        f = _vfile(version)
        np.savez_compressed(self.dir / f, vecs=vecs.astype(np.float32))
        self.doc["embeddings"][version] = dict(file=f, n_crops=len(self.crops), dim=int(vecs.shape[1]), stored=_now())
        self._write()


def names() -> list[str]:
    d = store_dir()
    return sorted(p.name for p in d.iterdir() if p.is_dir() and NAME_RE.match(p.name) and (p / "target.json").is_file()) if d.is_dir() else []


def list_targets() -> list[dict]:
    out = []
    for n in names():
        try:
            out.append(Saved(n).summary())
        except TargetError as e:
            out.append(dict(name=n, error=str(e)))
    return out


def remove(name: str) -> None:
    with _locked(check_name(name)):
        s = Saved(name)  # raises when there is no such target; only a folder that holds our target.json is ever deleted
        shutil.rmtree(s.dir)


# ---------------------------------------------------------------- use
def vectors_for(name: str, emb: Embedder) -> np.ndarray:
    """The target's embeddings for `emb`: the stored ones, or the thumbnails embedded now (and stored for next time)."""
    s = Saved(name)
    version = emb.spec.version
    v = s.vectors(version)
    if v is not None:
        return v
    crops = s.load_crops()
    files = [c["file"] for c in s.crops]
    v = emb.embed([x for c in crops for x in resolution_variants(c)])
    try:
        with _locked(name):
            # embedding took seconds: the target may have been added to meanwhile, so write into what is there now,
            # and only when it still holds the crops these vectors belong to
            fresh = Saved(name)
            if [c["file"] for c in fresh.crops] == files:
                fresh.store_vectors(version, v)
    except (OSError, TargetError):
        pass  # a read-only store still works, it just embeds again next time
    return v


def check_usable(name: str) -> Saved:
    """Fail before a run when the target is missing or stale."""
    s = Saved(name)
    s.crop_paths()
    if not s.crops:
        raise StaleTarget(f"saved target {name!r} holds no crops; add some with `inframe target add {name} --from ...`")
    return s


# ---------------------------------------------------------------- add
def _thumb(crop: np.ndarray) -> np.ndarray:
    """The crop as it will be stored: reduced to THUMB_SIDE and through a JPEG, so the embedding of the stored file is the one kept."""
    im = Image.fromarray(np.ascontiguousarray(crop)).convert("RGB")
    im.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def add_crops(name: str, crops: list[np.ndarray], source: str, embedder: str = identity.DEFAULT_EMBEDDER, offline: bool = False,
              cap: int = TARGET_CAP, emb: Embedder | None = None) -> dict:
    """Add `crops` (uint8 RGB arrays of the person) to target `name`, which is created when it does not exist. A crop too like one
    already kept is dropped; above `cap` the most redundant crop leaves (so different outfits stay). Thumbnails of the target that
    are missing or unreadable are dropped from it (that is how a stale target is repaired). One add at a time per target. Returns
    what happened."""
    check_name(name)
    if not crops:
        raise TargetError("no usable crops of the person were found in that source")
    with _locked(name):
        return _add_crops(name, crops, source, embedder, offline, cap, emb)


def _add_crops(name, crops, source, embedder, offline, cap, emb) -> dict:
    spec = SPECS[embedder]
    root = store_dir()
    d = root / name
    fresh = not (d / "target.json").is_file()
    if fresh:
        if d.exists():
            raise TargetError(f"{d} exists and is not a saved target")
        old: list[np.ndarray] = []
        was = []
        doc = dict(format=FORMAT, name=name, created=_now(), updated=_now(), crops=[], embeddings={})
    else:
        s = Saved(name)
        was = list(s.crops)
        kept, old = s.readable_crops()
        doc = s.doc
        doc["crops"] = kept
    new = [_thumb(c) for c in crops]
    own = emb is None
    emb = emb or Embedder(embedder, offline=offline)
    try:
        allc = old + new
        flat = [x for c in allc for x in resolution_variants(c)]
        vecs = emb.embed(flat)
        starts = np.cumsum([0] + [len(resolution_variants(c)) for c in allc])[:-1]
    finally:
        if own:
            emb.close()
    g = Gallery(spec, cap=cap)
    for i, c in enumerate(allc):
        g.add(vecs[starts[i]], crop_signature(c, (0.0, 0.0, 1.0, 1.0)), tag=i)
    keep = sorted(g.tags)
    kept_old = [i for i in keep if i < len(old)]
    kept_new = [i for i in keep if i >= len(old)]
    # write: the new thumbnails first, then target.json (the file that makes them part of the target), then remove what left
    existed = d.exists()
    written: list[Path] = []
    try:
        (d / "crops").mkdir(parents=True, exist_ok=True)
        n = 0
        for c in was or doc["crops"]:
            n = max(n, int(re.sub(r"\D", "", c["file"]) or 0))
        by_old = {i: doc["crops"][i] for i in range(len(old))}
        new_entries = {}
        for i in kept_new:
            n += 1
            fn = f"crops/c{n:04d}.jpg"
            written.append(d / fn)
            Image.fromarray(allc[i]).save(d / fn, "JPEG", quality=95)
            new_entries[i] = dict(file=fn, source=source, added=_now())
        entries = [by_old[i] if i < len(old) else new_entries[i] for i in keep]
        rows = np.vstack([vecs[starts[i]:starts[i] + len(resolution_variants(allc[i]))] for i in keep])
        doc["crops"], doc["embeddings"] = entries, {}
        s = Saved.blank(name, d, doc)
        s.store_vectors(spec.version, rows)
    except BaseException:
        # a failed add leaves no half target behind: a first add removes its folder, a later one its new thumbnails
        if not existed:
            shutil.rmtree(d, ignore_errors=True)
        else:
            for f in written:
                f.unlink(missing_ok=True)
        raise
    for f in {c["file"] for c in was} - {c["file"] for c in entries}:
        (d / f).unlink(missing_ok=True)
    for f in (d / "emb").glob("*.npz"):
        if f.name != Path(_vfile(spec.version)).name:
            f.unlink(missing_ok=True)  # embeddings of the old crop set no longer match
    return dict(name=name, created=fresh, added=len(kept_new), offered=len(new), dropped_duplicates=len(new) - len(kept_new),
                evicted=len(old) - len(kept_old), lost_thumbnails=len(was) - len(old), n_crops=len(entries), embedder=embedder, version=spec.version)


# ---------------------------------------------------------------- sources
def _video_crop(video: str, t: float, box, detect=None, point=None) -> np.ndarray:
    from .pick import grab_frame
    from .probe import probe

    info = probe(video)
    _, frame = grab_frame(video, info, t, long_side=identity.DECODE_LONG_SIDE)
    if box is None:  # a tap by point: the smallest detected box that holds it
        dets = (detect or identity.lazy_detect())([frame])[0]
        x, y = point
        hits = [d for d in dets if d.box[0] <= x <= d.box[2] and d.box[1] <= y <= d.box[3]]
        if not hits:
            raise TargetError(f"no person detected at ({x:g}, {y:g}) at {t:.1f} s in {video}; use `inframe pick` to tap with a box")
        box = min(hits, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])).box
    c = person_crop(frame, tuple(box))
    if c is None:
        raise TargetError(f"the person at {t:.1f} s in {video} is too small to keep a crop of")
    return c


def crops_from_taps(spec: str, video: str | None = None, detect=None) -> list[np.ndarray]:
    """`tap:` source: one crop per tap of a target.json (or one `T,x,y`). The video is named in target.json, or by `video`."""
    taps = parse_spec(spec)
    vid = video
    f = Path(spec.strip()).expanduser()
    if vid is None and not is_point_spec(spec) and f.is_file():
        try:
            vid = json.loads(f.read_text()).get("video")
        except (ValueError, AttributeError):
            vid = None
    if not vid or not Path(vid).expanduser().is_file():
        raise TargetError(f"tap:{spec}: the video is {'not named in it' if not vid else 'not found: ' + str(vid)}; pass --video FILE")
    return [_video_crop(str(Path(vid).expanduser()), t.t, t.box, detect, t.point) for t in taps]


def crops_from_run(seg: str, n: int = RUN_CANDIDATES) -> list[np.ndarray]:
    """`run:` source: crops of the person from a finished run, spread over the samples where the run found them (samples.csv).
    Refuses a run that kept every person or only motion."""
    import csv

    p = Path(seg).expanduser()
    p = p / "segments.json" if p.is_dir() else p
    if not p.is_file():
        raise TargetError(f"run:{seg}: no such segments.json")
    try:
        doc = json.loads(p.read_text())
        video, enter, kind = doc["source"]["path"], float(doc["settings"]["enter"]), doc["settings"]["target"]
    except (ValueError, KeyError, TypeError):
        raise TargetError(f"{p} is not a segments.json written by inframe") from None
    if doc.get("mode") != "detect" or kind in ("any-person", "any-motion"):
        raise TargetError(f"{p} kept {'every person' if kind == 'any-person' else 'whatever moved'}, not one person; "
                          "use a run made with --target (a tap, photos or a saved target)")
    if not Path(video).is_file():
        raise TargetError(f"the video of that run is gone: {video}")
    csvf = p.with_name("samples.csv")
    if not csvf.is_file():
        raise TargetError(f"{csvf} is missing; the run's samples say where the person was")
    rows = []
    with open(csvf, newline="") as f:
        for r in csv.DictReader(f):
            try:
                if float(r["score"]) >= max(enter, 0.5) and r["x0"] != "":
                    rows.append((float(r["t"]), (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))))
            except (ValueError, KeyError):
                continue
    if not rows:
        raise TargetError(f"{p}: the run found the person in no sample (nothing to take crops from)")
    idx = sorted({int(round(i)) for i in np.linspace(0, len(rows) - 1, min(n, len(rows)))})
    out = []
    for i in idx:
        t, box = rows[i]
        try:
            out.append(_video_crop(video, t, box))
        except (TargetError, ValueError):
            continue
    return out


def crops_from(spec: str, video: str | None = None, detect=None) -> tuple[list[np.ndarray], str]:
    """`--from` value -> (crops, a short source label for target.json)."""
    kind, _, rest = spec.partition(":")
    if kind == "photos" and rest:
        return identity.load_photo_crops(rest, detect or identity.lazy_detect()), f"photos:{Path(rest).expanduser().name}"
    if kind == "tap" and rest:
        return crops_from_taps(rest, video, detect), "tap"
    if kind == "run" and rest:
        return crops_from_run(rest), "run"
    raise TargetError(f"--from {spec!r}: use photos:<folder>, tap:<target.json> or run:<segments.json>")


# ---------------------------------------------------------------- export and import
def export_target(name: str, out: str | Path) -> Path:
    s = Saved(name)
    out = Path(out).expanduser()
    if out.is_dir():
        out = out / f"{name}.inframe-target.zip"
    s.crop_paths()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(s.dir / "target.json", "target.json")
        for c in s.crops:
            z.write(s.dir / c["file"], c["file"])
        for e in s.doc["embeddings"].values():
            if (s.dir / e["file"]).is_file():
                z.write(s.dir / e["file"], e["file"])
    return out


def import_target(archive: str | Path, name: str | None = None, force: bool = False) -> str:
    """Unpack an exported target into the store, as `name` (default: the name inside). Refuses a name that exists unless `force`."""
    a = Path(archive).expanduser()
    if not a.is_file() or not zipfile.is_zipfile(a):
        raise TargetError(f"{archive}: not an exported target (a zip written by `inframe target export`)")
    try:
        return _import_zip(a, archive, name, force)
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, NotImplementedError, RuntimeError) as e:
        raise TargetError(f"{archive}: damaged or unreadable zip ({e.__class__.__name__}); export the target again") from None


def _import_zip(a: Path, archive: str | Path, name: str | None, force: bool) -> str:
    with zipfile.ZipFile(a) as z:
        infos = z.infolist()
        if len(infos) > ZIP_MAX_FILES or sum(i.file_size for i in infos) > ZIP_MAX_BYTES:
            raise TargetError(f"{archive}: too many or too large files for a saved target")
        bad = [i.filename for i in infos if not _ZIP_OK.match(i.filename)]
        if bad:
            raise TargetError(f"{archive}: unexpected file {bad[0]!r}; not a target exported by inframe")
        try:
            doc = json.loads(z.read("target.json"))
            assert doc["format"] == FORMAT
            files = {c["file"] for c in doc["crops"]} | {e["file"] for e in doc.get("embeddings", {}).values()}
            assert files <= {i.filename for i in infos}
            assert all(isinstance(c["file"], str) for c in doc["crops"])
        except (KeyError, ValueError, TypeError, AssertionError, AttributeError):
            raise TargetError(f"{archive}: target.json is missing or not of this version of inframe") from None
        name = check_name(name or doc.get("name", ""))
        dest = store_dir() / name
        if dest.exists() and not force:
            raise TargetError(f"a saved target named {name!r} already exists; pass --name NEW or --force to replace it")
        if dest.exists() and (dest.is_symlink() or not (dest / "target.json").is_file()):
            raise TargetError(f"{dest} exists but is not a saved target; --force only replaces saved targets (move that folder away or use --name NEW)")
        doc["name"] = name
        store_dir().mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".import-", dir=store_dir()))
        try:
            for i in infos:
                if i.filename != "target.json":
                    (tmp / i.filename).parent.mkdir(parents=True, exist_ok=True)
                    (tmp / i.filename).write_bytes(z.read(i))
            (tmp / "target.json").write_text(json.dumps(doc, indent=1))
            if dest.exists():
                shutil.rmtree(dest)
            os.replace(tmp, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return name
