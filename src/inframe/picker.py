"""T3p: pick videos with Apple's photo picker (PHPicker), no Full Disk Access.

A small Swift helper (src/inframe/swift/photo_picker.swift, built with swiftc, no Xcode project) opens PHPicker on the
shared Photos library: videos only, multiple selection, preferred representation `current`. It exports the picked
originals into a folder and prints their paths as JSON. macOS hands the helper only the items the person picked, so
neither Full Disk Access nor library-wide Photos permission is needed.

The helper is built on first use into ~/Library/Caches/inframe/bin (override: INFRAME_BIN_DIR) as
inframe-photo-picker-<hash of the source>, so a changed source rebuilds. INFRAME_PICKER_HELPER names a different
executable to run instead (the tests use a fake one); it gets `--out DIR [--limit N] [--images]` and prints the same JSON.
`pick(kind="images")` passes `--images`: the picker then shows photos only (the target photos of `inframe serve`).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SOURCE = Path(__file__).with_name("swift") / "photo_picker.swift"
HELPER_ENV = "INFRAME_PICKER_HELPER"
BIN_ENV = "INFRAME_BIN_DIR"
BUILD_TIMEOUT_S = 300
KINDS = ("videos", "images")


class PickerError(RuntimeError):
    """A problem the person can fix or report; the message says what happened."""


@dataclass
class PickResult:
    files: list[Path] = field(default_factory=list)
    cancelled: bool = False
    errors: list[str] = field(default_factory=list)


def bin_dir() -> Path:
    return Path(os.environ.get(BIN_ENV) or "~/Library/Caches/inframe/bin").expanduser()


def helper_binary() -> Path:
    """Where the compiled helper lives: the name carries a hash of the source, so editing the source rebuilds."""
    h = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:10]
    return bin_dir() / f"inframe-photo-picker-{h}"


def ensure_helper(announce=lambda m: None) -> Path:
    """The helper to run: INFRAME_PICKER_HELPER if set, else the cached build (compiled now when missing)."""
    over = os.environ.get(HELPER_ENV)
    if over:
        p = Path(over).expanduser()
        if not os.access(p, os.X_OK):
            raise PickerError(f"{HELPER_ENV}={over} is not an executable file")
        return p
    if sys.platform != "darwin":
        raise PickerError("the photo picker is macOS only (PHPicker)")
    out = helper_binary()
    if out.is_file() and os.access(out, os.X_OK):
        return out
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise PickerError("swiftc not found: install the Xcode command line tools (`xcode-select --install`) and run again")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    announce(f"building the photo picker helper (once) into {out.parent} ...")
    try:
        r = subprocess.run([swiftc, "-O", "-swift-version", "5", str(SOURCE), "-o", str(tmp)],
                           capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        tmp.unlink(missing_ok=True)
        raise PickerError(f"swiftc did not finish within {BUILD_TIMEOUT_S} s") from e
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise PickerError("swiftc failed to build the photo picker helper:\n" + "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:]))
    os.replace(tmp, out)
    for old in out.parent.glob("inframe-photo-picker-*"):  # earlier builds of other source versions
        if old != out and ".tmp" not in old.name:
            old.unlink(missing_ok=True)
    return out


def parse_output(stdout: str) -> PickResult:
    """The helper's JSON is the last non-empty line of stdout."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        raise PickerError("the photo picker helper printed nothing")
    try:
        d = json.loads(lines[-1])
        files = [Path(f["path"]) for f in d.get("files", [])]
        return PickResult(files=files, cancelled=bool(d.get("cancelled")), errors=[str(e) for e in d.get("errors", [])])
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise PickerError(f"the photo picker helper printed something that is not its JSON: {lines[-1][:200]!r}") from e


def pick(out_dir: Path, limit: int = 0, announce=lambda m: None, kind: str = "videos") -> PickResult:
    """Open the picker, wait for the person, return the exported files (each checked to exist).

    kind is "videos" (default) or "images"; "images" passes --images to the helper. Any other kind is a ValueError.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    helper = ensure_helper(announce)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(helper), "--out", str(out_dir)] + (["--limit", str(limit)] if limit > 0 else []) + (["--images"] if kind == "images" else [])
    announce(f"opening the Photos picker: choose {'photos' if kind == 'images' else 'videos'}, then click Add (close the window to cancel) ...")
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)  # its progress lines go to our stderr
    except OSError as e:
        raise PickerError(f"could not start {helper}: {e}") from e
    if r.returncode != 0 and not r.stdout.strip():
        raise PickerError(f"the photo picker helper failed (exit {r.returncode})")
    res = parse_output(r.stdout)
    if r.returncode != 0 and not res.files:
        raise PickerError(f"the photo picker helper failed (exit {r.returncode}): " + "; ".join(res.errors))
    missing = [f for f in res.files if not f.is_file()]
    if missing:
        raise PickerError(f"the helper listed files that are not there: {missing[0]}")
    return res


def new_export_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="inframe-picked-"))
