"""Model files: where they live, first-run download with a checksum and progress, and --offline.

Cache folder: ~/Library/Caches/inframe/models on macOS (~/.cache/inframe/models elsewhere); $INFRAME_CACHE replaces
the part before /models. Downloads go to a .part file, are checked against the registry checksum and only then renamed,
so a half-finished or corrupt download is never used.

What can be downloaded: the RF-DETR checkpoints (Apache-2.0, published by Roboflow). The CoreML packages the detector
runs are exported from them (bench/detect/setup.sh) and are not published anywhere, so for those the message says
how to build or copy them. Nothing here talks to the network unless a download is asked for; `--offline` makes any
missing file an error instead.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

G = "https://storage.googleapis.com/rfdetr"


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    checksum: str  # "md5:<hex>" or "sha256:<hex>"
    note: str = ""


# md5 values are the ones bench/detect/fetch_weights.py has been checking since the detector benchmark
REGISTRY = {m.name: m for m in (
    ModelFile("rf-detr-nano.pth", f"{G}/nano_coco/checkpoint_best_regular.pth", "md5:fb6504cce7fbdc783f7a46991f07639f", "RF-DETR nano, Apache-2.0"),
    ModelFile("rf-detr-small.pth", f"{G}/small_coco/checkpoint_best_regular.pth", "md5:fb37061c1af7bace359c91b723a8d5c1", "RF-DETR small, Apache-2.0"),
    ModelFile("rf-detr-base.pth", f"{G}/rf-detr-base-coco.pth", "md5:b4d3ce46099eaed50626ede388caf979", "RF-DETR base, Apache-2.0"),
)}

_offline = False


class ModelsMissing(FileNotFoundError):
    """A model file is not on disk (and was not downloaded)."""


class OfflineError(ModelsMissing):
    """A download was needed but --offline is set, or the network could not be reached."""


class ChecksumError(RuntimeError):
    pass


def set_offline(value: bool) -> None:
    global _offline
    _offline = bool(value)


def is_offline() -> bool:
    return _offline


def cache_dir() -> Path:
    base = os.environ.get("INFRAME_CACHE")
    if base:
        return Path(base).expanduser() / "models"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "inframe" / "models"
    return Path.home() / ".cache" / "inframe" / "models"


def missing_message(what: str, searched: Path | str | None = None) -> str:
    where = f" under {searched}" if searched else ""
    return (f"{what} not found{where}. Put them in {cache_dir()} or set INFRAME_MODELS to their folder. "
            "The CoreML packages are exported from the RF-DETR checkpoints with bench/detect/setup.sh "
            "(`inframe fetch-models` downloads the checkpoints into the cache first; it needs a network connection)")


def file_checksum(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checksum_ok(path: Path, checksum: str) -> bool:
    algo, _, want = checksum.partition(":")
    return path.is_file() and file_checksum(path, algo) == want.lower()


def _bar(done: int, total: int | None, name: str) -> None:
    if not sys.stderr.isatty():
        return
    if total:
        sys.stderr.write(f"\r[models] {name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({100 * done / total:3.0f} %)")
    else:
        sys.stderr.write(f"\r[models] {name}: {done / 1e6:7.1f} MB")
    sys.stderr.flush()


def fetch(spec: ModelFile, dest_dir: Path | None = None, offline: bool | None = None,
          progress: Callable[[int, int | None, str], None] | None = _bar, timeout: float = 30.0) -> Path:
    """The path of `spec` in `dest_dir` (default: the cache), downloading it first when it is missing or fails its checksum.
    Raises OfflineError when a download is needed but offline is set or the host cannot be reached, and
    ChecksumError when the download does not match (nothing is kept)."""
    dest_dir = Path(dest_dir) if dest_dir else cache_dir()
    dest = dest_dir / spec.name
    if checksum_ok(dest, spec.checksum):
        return dest
    offline = _offline if offline is None else offline
    if offline:
        raise OfflineError(f"{spec.name} is not in {dest_dir} (or fails its checksum) and --offline is set; "
                           f"run `inframe fetch-models` once with a network connection, or copy the file there")
    if not spec.url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise ValueError(f"refusing to download from {spec.url}: https only")
    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(spec.url, timeout=timeout) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or None
            done = 0
            while chunk := r.read(1 << 18):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total, spec.name)
        if progress and sys.stderr.isatty():
            sys.stderr.write("\n")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        part.unlink(missing_ok=True)
        raise OfflineError(f"could not download {spec.name} from {spec.url} ({type(e).__name__}: {e}); "
                           f"check the network connection, or copy the file into {dest_dir} and use --offline") from None
    if not checksum_ok(part, spec.checksum):
        got = file_checksum(part, spec.checksum.partition(":")[0])
        part.unlink(missing_ok=True)
        raise ChecksumError(f"{spec.name}: checksum mismatch (want {spec.checksum}, got {got}); the download was discarded")
    os.replace(part, dest)
    return dest


def torch_checkpoint(size: str) -> Path:
    """The checkpoint the torch fallback (rfdetr) loads: the checked file in the inframe cache, where fetch-models puts it.
    It is downloaded there first when missing, unless --offline is set (then OfflineError). The path is passed to rfdetr as
    pretrain_weights: rfdetr's own default is a path relative to the current directory, which it fills by downloading."""
    name = f"rf-detr-{size}.pth"
    if name not in REGISTRY:
        raise ValueError(f"no RF-DETR checkpoint known for size {size!r}; known: {sorted(REGISTRY)}")
    try:
        return fetch(REGISTRY[name])
    except OfflineError as e:
        if _offline:
            raise OfflineError(f"the torch fallback needs {name} in {cache_dir()}, which is not there, and --offline is set; "
                               f"run `inframe fetch-models {name}` with a network connection") from e
        raise


def fetch_all(names: list[str] | None = None, offline: bool | None = None) -> list[Path]:
    unknown = [n for n in (names or []) if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown model file(s) {unknown}; known: {sorted(REGISTRY)}")
    return [fetch(REGISTRY[n], offline=offline) for n in (names or list(REGISTRY))]
