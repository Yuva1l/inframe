"""Path rules for `inframe serve` (docs/T9A_DESIGN.md section 3.4 "Path safety", 2.3 state folders, 2.8 target files).

The server takes a path only from the CLI argument, a native dialog, the file browser or a paste; every one goes through
`check_read` / `check_video` / `check_audio` (reads) or `check_write` (writes) before anything opens it. A text file (HLS,
ffconcat, SDP) named .mp4 never reaches ffprobe, ffmpeg or PyAV, because the magic bytes are checked first.
Failures are `PathError` (an `HttpError`), so a route handler can let them propagate.
"""
from __future__ import annotations

import os
import re
import shutil
import stat as _stat
import sys
import tempfile
import time
from pathlib import Path

from .serve_http import HttpError

VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".mkv", ".webm", ".ogv")
AUDIO_EXTS = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus")
PICKED_PREFIX = "inframe-picked-"
PICKED_MAX_AGE_S = 24 * 3600
_ISO_BOXES = (b"ftyp", b"moov", b"mdat", b"wide", b"free")


class PathError(HttpError):
    """A path the server refuses: 404 (missing, not a file), 403 (hidden, under ~/Library, not writable), 415 (not a video/audio)."""


# ---------------------------------------------------------------- state folders (2.3)
def cache_base() -> Path:
    """The inframe cache folder: $INFRAME_CACHE, else ~/Library/Caches/inframe (macOS) / ~/.cache/inframe (the models.cache_dir rule)."""
    base = os.environ.get("INFRAME_CACHE")
    if base:
        return Path(base).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "inframe"
    return Path.home() / ".cache" / "inframe"


def cache_root() -> Path:
    """Per-media caches (filmstrip, proxy) live under <cache_root>/<media_id>/."""
    return cache_base() / "serve"


def state_root() -> Path:
    """The serve state folder (recent.json): $INFRAME_SERVE_STATE, else the targets.store_dir rule with `serve` instead of `targets`."""
    base = os.environ.get("INFRAME_SERVE_STATE")
    if base:
        return Path(base).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "inframe" / "serve"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "inframe" / "serve"


def default_out_parent() -> Path:
    """Where runs go when the video's own folder cannot hold them (and for Photos-picked copies): ~/Movies/inframe (macOS), ~/inframe elsewhere."""
    return Path.home() / "Movies" / "inframe" if sys.platform == "darwin" else Path.home() / "inframe"


# ---------------------------------------------------------------- reads
def _library() -> Path:
    return Path.home() / "Library"


def _under(p: Path, base: Path) -> bool:
    """Is `p` inside `base`, or `base` itself? Compared by identity (the device and inode of each existing ancestor of `p`), because
    resolve() keeps the letter case it was given and APFS ignores case: ~/LIBRARY and ~/Library are one folder."""
    if p.is_relative_to(base):
        return True
    try:
        want = os.stat(base)
    except OSError:
        return False
    for a in (p, *p.parents):
        try:
            if os.path.samestat(os.stat(a), want):
                return True
        except OSError:
            continue
    return False


def _is_hidden(p: Path) -> bool:
    return any(part.startswith(".") and part not in (".", "..") for part in p.parts)


def _in_library_exception(p: Path) -> bool:
    """Inside ~/Library, only the inframe cache and the picker temp folders may be read."""
    try:
        if p.is_relative_to(cache_base().resolve()):
            return True
    except OSError:
        pass
    tmp = Path(tempfile.gettempdir()).resolve()
    return any(parent.parent == tmp and parent.name.startswith(PICKED_PREFIX) for parent in [p, *p.parents])


def check_read(path, kind: str = "file") -> Path:
    """`path` resolved (strict), as a regular file (`kind="file"`) or a folder (`kind="dir"`). Refused: a missing path (404), a
    component starting with `.` or anything inside ~/Library except the inframe cache and the picker temp folders (403), the
    wrong kind (404)."""
    if kind not in ("file", "dir"):
        raise ValueError(f"kind must be 'file' or 'dir', not {kind!r}")
    if not isinstance(path, (str, os.PathLike)) or not str(path) or "\0" in str(path):
        raise PathError(400, "bad path")
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise PathError(400, f"not an absolute path: {path}")
    try:
        p = raw.resolve(strict=True)
    except (OSError, RuntimeError):
        raise PathError(404, f"No such {'folder' if kind == 'dir' else 'file'}: {path}") from None
    if _is_hidden(p):
        raise PathError(403, "inframe serve does not open files there.")
    lib = _library().resolve()
    if _under(p, lib) and not _in_library_exception(p):
        raise PathError(403, "inframe serve does not open files there.")
    try:
        st = os.stat(p)
    except OSError:
        raise PathError(404, f"No such file: {path}") from None
    if kind == "file" and not _stat.S_ISREG(st.st_mode):
        raise PathError(404, f"Not a file: {path}")
    if kind == "dir" and not _stat.S_ISDIR(st.st_mode):
        raise PathError(404, f"Not a folder: {path}")
    return p


def _head(p: Path, n: int = 64) -> bytes:
    try:
        with open(p, "rb") as f:
            return f.read(n)
    except OSError:
        raise PathError(404, f"Cannot read: {p}") from None


def video_magic_ok(ext: str, head: bytes) -> bool:
    if ext in (".mp4", ".m4v", ".mov"):
        return len(head) >= 8 and head[4:8] in _ISO_BOXES
    if ext in (".mkv", ".webm"):
        return head[:4] == b"\x1a\x45\xdf\xa3"
    if ext == ".ogv":
        return head[:4] == b"OggS"
    return False


def audio_magic_ok(ext: str, head: bytes) -> bool:
    id3 = head[:3] == b"ID3"
    mpeg = len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0
    if ext == ".mp3":
        return id3 or mpeg
    if ext == ".m4a":
        return len(head) >= 8 and head[4:8] == b"ftyp"
    if ext == ".aac":
        return id3 or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xF0 == 0xF0)
    if ext == ".wav":
        return len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if ext == ".flac":
        return head[:4] == b"fLaC" or id3
    if ext in (".ogg", ".opus"):
        return head[:4] == b"OggS"
    return False


def check_video(path) -> Path:
    """`check_read`, then the extension allowlist and the magic bytes (415 when either fails)."""
    p = check_read(path, "file")
    ext = p.suffix.lower()
    if ext not in VIDEO_EXTS:
        raise PathError(415, f"Not a video file type: {p.name}")
    if not video_magic_ok(ext, _head(p)):
        raise PathError(415, f"{p.name} does not look like a {ext[1:]} video.")
    return p


def check_audio(path) -> Path:
    p = check_read(path, "file")
    ext = p.suffix.lower()
    if ext not in AUDIO_EXTS:
        raise PathError(415, f"Not an audio file type: {p.name}")
    if not audio_magic_ok(ext, _head(p)):
        raise PathError(415, f"{p.name} does not look like {ext[1:]} audio.")
    return p


# ---------------------------------------------------------------- writes
def check_write(folder) -> Path:
    """`folder` resolved (it need not exist yet) for a write: not hidden, not under ~/Library (the inframe cache and the serve
    state folder excepted), and writable (`os.access`, checked on the nearest existing parent). Else 403."""
    raw = Path(folder).expanduser()
    if not raw.is_absolute() or "\0" in str(folder):
        raise PathError(400, f"not an absolute path: {folder}")
    p = raw.resolve(strict=False)
    if _is_hidden(p):
        raise PathError(403, "inframe serve does not write there.")
    lib = _library().resolve()
    if _under(p, lib):
        ok = [cache_base().resolve(), state_root().resolve()]
        if not any(p.is_relative_to(o) for o in ok):
            raise PathError(403, "inframe serve does not write there.")
    probe = p
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir() or not os.access(probe, os.W_OK | os.X_OK):
        raise PathError(403, f"Cannot write to {folder}")
    return p


# ---------------------------------------------------------------- shared names
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")[:64]


def target_file(out_root, cls: str, target_key: str) -> Path:
    """`<out_root>/targets/target-<cls>[-<target_key slug>].json`: the file of one ServePicker (A3 writes it, B1 reads it).
    `target_key` is "" for plain taps, "photos:<photos_id>" or "saved:<name>"."""
    c = _slug(str(cls)) or "person"
    k = _slug(str(target_key))
    return Path(out_root) / "targets" / (f"target-{c}-{k}.json" if k else f"target-{c}.json")


# ---------------------------------------------------------------- picker temp folders
def new_picked_dir() -> Path:
    """A fresh picker export folder (mode 0700, `picker.new_export_dir`) marked with this process as its owner."""
    from .picker import new_export_dir

    d = new_export_dir()
    (d / "owner.pid").write_text(str(os.getpid()))
    return d


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _owner(d: Path) -> int | None:
    try:
        return int((d / "owner.pid").read_text().strip())
    except (OSError, ValueError):
        return None


def sweep_picked(tmp: Path | None = None, max_age_s: float = PICKED_MAX_AGE_S, now: float | None = None) -> list[Path]:
    """Remove `inframe-picked-*` folders in the temp dir that are older than 24 h and that no running server owns (`owner.pid`
    names a live process). Only real folders of this user are touched. Returns what was removed."""
    tmp = Path(tmp) if tmp else Path(tempfile.gettempdir())
    now = time.time() if now is None else now
    removed: list[Path] = []
    try:
        entries = list(os.scandir(tmp))
    except OSError:
        return removed
    for e in entries:
        if not e.name.startswith(PICKED_PREFIX):
            continue
        try:
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        if not _stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or now - st.st_mtime < max_age_s:
            continue
        pid = _owner(Path(e.path))
        if pid is not None and _pid_alive(pid):
            continue
        shutil.rmtree(e.path, ignore_errors=True)
        removed.append(Path(e.path))
    return removed


def cleanup_own_picked(tmp: Path | None = None) -> list[Path]:
    """At stop: remove the picker folders this process owns (unless --keep-picked)."""
    tmp = Path(tmp) if tmp else Path(tempfile.gettempdir())
    removed: list[Path] = []
    try:
        entries = list(os.scandir(tmp))
    except OSError:
        return removed
    for e in entries:
        if e.name.startswith(PICKED_PREFIX) and e.is_dir(follow_symlinks=False) and _owner(Path(e.path)) == os.getpid():
            shutil.rmtree(e.path, ignore_errors=True)
            removed.append(Path(e.path))
    return removed
