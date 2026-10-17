"""Save exported clips and reels to the Photos library (roadmap T3b), library only, with the add-only permission.

A small Swift helper (src/inframe/swift/photo_save.swift, built with swiftc, no Xcode project) asks macOS for
PHAccessLevel.addOnly and adds each video with PHAssetCreationRequest. Add-only means inframe can put videos into the library
and can never read it: no Full Disk Access, no library-wide permission, no album. The helper is built on first use into
~/Library/Caches/inframe/bin (override: INFRAME_BIN_DIR) as inframe-photo-save-<hash of source and Info.plist>; the plist
(NSPhotoLibraryAddUsageDescription) is linked into the binary because macOS refuses to ask without it.
INFRAME_PHOTOS_SAVE_HELPER names a different executable to run instead (the tests use a fake one); it gets the file paths as
arguments and prints the JSON described in photo_save.swift. No real Photos library is touched by the tests.

The helper prints one {"row": {...}} line per file as soon as that file is done and the summary JSON last, so a timeout or a crash
partway through still tells which videos were already added.

Result: `SaveResult.status` is "saved" (all), "partial" (some), "denied" (no permission: `message` says where to allow it),
"failed" (permission fine, nothing saved), "interrupted" (the helper timed out or died partway: the files without a confirmed row may
or may not be in the library, so the person must check Photos before exporting again, which would duplicate them) or "skipped"
(there was nothing to save). `describe()` turns a result into one line.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).with_name("swift")
SOURCE = HERE / "photo_save.swift"
PLIST = HERE / "photo_save_Info.plist"
HELPER_ENV = "INFRAME_PHOTOS_SAVE_HELPER"
BIN_ENV = "INFRAME_BIN_DIR"
BUILD_TIMEOUT_S = 300
RUN_TIMEOUT_S = 900  # the system permission dialog waits for the person; a big reel takes a while to copy in
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")
DENIED_HELP = ("Allow it in System Settings > Privacy & Security > Photos (the add-only choice is enough) "
               "for the app that runs inframe (Terminal, your editor), then run again.")


class PhotosSaveError(RuntimeError):
    """The helper could not be built or run; the message says what happened."""


@dataclass
class SaveResult:
    status: str  # saved | partial | denied | failed | interrupted | skipped
    authorization: str = "unknown"
    saved: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("saved", "skipped")

    def to_json(self) -> dict:
        return {"status": self.status, "authorization": self.authorization, "message": describe(self),
                "saved": [p.name for p in self.saved], "failed": [{"name": p.name, "error": e} for p, e in self.failed]}


def available() -> bool:
    """True when a save can be tried here: macOS, or a helper named by INFRAME_PHOTOS_SAVE_HELPER."""
    return sys.platform == "darwin" or bool(os.environ.get(HELPER_ENV))


def bin_dir() -> Path:
    return Path(os.environ.get(BIN_ENV) or "~/Library/Caches/inframe/bin").expanduser()


def helper_binary() -> Path:
    h = hashlib.sha256(SOURCE.read_bytes() + PLIST.read_bytes()).hexdigest()[:10]
    return bin_dir() / f"inframe-photo-save-{h}"


def ensure_helper(announce=lambda m: None) -> Path:
    """The helper to run: INFRAME_PHOTOS_SAVE_HELPER if set, else the cached build (compiled now when missing)."""
    over = os.environ.get(HELPER_ENV)
    if over:
        p = Path(over).expanduser()
        if not os.access(p, os.X_OK):
            raise PhotosSaveError(f"{HELPER_ENV}={over} is not an executable file")
        return p
    if sys.platform != "darwin":
        raise PhotosSaveError("saving to Photos is macOS only")
    out = helper_binary()
    if out.is_file() and os.access(out, os.X_OK):
        return out
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise PhotosSaveError("swiftc not found: install the Xcode command line tools (`xcode-select --install`) and run again")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    announce(f"building the Photos save helper (once) into {out.parent} ...")
    cmd = [swiftc, "-O", "-swift-version", "5", str(SOURCE), "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist",
           "-Xlinker", str(PLIST), "-o", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        tmp.unlink(missing_ok=True)
        raise PhotosSaveError(f"swiftc did not finish within {BUILD_TIMEOUT_S} s") from e
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise PhotosSaveError("swiftc failed to build the Photos save helper:\n" + "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:]))
    os.replace(tmp, out)
    for old in out.parent.glob("inframe-photo-save-*"):  # earlier builds of other source versions
        if old != out and ".tmp" not in old.name:
            old.unlink(missing_ok=True)
    return out


def _json_lines(stdout: str) -> list:
    out = []
    for ln in stdout.splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except ValueError:
                out.append(None)
    return out


def _interrupted(files: list[Path], streamed: dict[str, dict], why: str) -> SaveResult:
    saved = [f for f in files if (streamed.get(str(f)) or {}).get("saved") is True]
    failed = []
    for f in files:
        row = streamed.get(str(f))
        if row and row.get("saved") is True:
            continue
        failed.append((f, str(row.get("error") or "not saved") if row else "not confirmed: it may or may not be in Photos"))
    unsure = sum(1 for f, e in failed if e.startswith("not confirmed"))
    msg = (f"The Photos save stopped before it finished ({why}). {len(saved)} of {len(files)} video(s) were added to Photos"
           + (f"; {unsure} more may or may not be there. Check Photos before exporting again, or they will be duplicated." if unsure
              else "."))
    return SaveResult(status="interrupted", saved=saved, failed=failed, message=msg)


def parse_output(stdout: str, files: list[Path], interrupted: str = "") -> SaveResult:
    """The helper's summary JSON is the last non-empty line of stdout; before it come one {"row": ...} line per finished file.
    A file the helper did not mention counts as failed. Without a summary (`interrupted` says why the helper stopped) the rows
    that did arrive give an "interrupted" result: the files without a row may or may not be in the library."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    docs = _json_lines(stdout)
    streamed = {str(d["row"]["path"]): d["row"] for d in docs if isinstance(d, dict) and isinstance(d.get("row"), dict) and "path" in d["row"]}
    if interrupted and not (docs and isinstance(docs[-1], dict) and "status" in docs[-1]):
        return _interrupted(files, streamed, interrupted)
    if not lines:
        raise PhotosSaveError("the Photos save helper printed nothing")
    try:
        d = docs[-1]
        status, auth, message = str(d["status"]), str(d.get("authorization", "unknown")), str(d.get("message") or "")
        rows = {str(r["path"]): r for r in d.get("files", [])}
        if status not in ("saved", "partial", "denied", "failed"):
            raise ValueError(status)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        if streamed:
            return _interrupted(files, streamed, "it printed no summary")
        raise PhotosSaveError(f"the Photos save helper printed something that is not its JSON: {lines[-1][:200]!r}") from e
    saved: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for f in files:
        row = rows.get(str(f))
        if row and row.get("saved") is True:
            saved.append(f)
        else:
            failed.append((f, str((row or {}).get("error") or "the helper did not report this file")))
    if status in ("saved", "partial") and (failed or not saved):  # trust the per-file rows over the summary
        status = "partial" if saved else "failed"
    return SaveResult(status=status, authorization=auth, saved=saved, failed=failed, message=message)


def save(files: list[Path], announce=lambda m: None) -> SaveResult:
    """Add `files` (videos) to the Photos library. Files that are not there are reported as failed without being sent to the helper.
    Raises PhotosSaveError when the helper cannot be built or started; a denied permission is a result, not an exception."""
    files = [Path(f) for f in files]
    if not files:
        return SaveResult(status="skipped", message="nothing to save to Photos")
    absent = [(f, "no such file") for f in files if not f.is_file()]
    present = [f for f in files if f.is_file()]
    if not present:
        return SaveResult(status="failed", failed=absent, message="none of the files exist")
    helper = ensure_helper(announce)
    announce(f"saving {len(present)} video(s) to Photos (add-only; if macOS asks, choose Allow) ...")
    try:
        proc = subprocess.Popen([str(helper)] + [str(f) for f in present], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as e:
        raise PhotosSaveError(f"could not start {helper}: {e}") from e
    try:
        out, err = proc.communicate(timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        res = parse_output(out or "", present, interrupted=f"the helper did not finish within {RUN_TIMEOUT_S} s; is a permission dialog waiting?")
        res.failed = absent + res.failed
        return res
    r = subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
    if not r.stdout.strip():
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise PhotosSaveError(f"the Photos save helper failed (exit {r.returncode})" + (": " + " | ".join(tail) if tail else ""))
    res = parse_output(r.stdout, present, interrupted=f"the helper stopped with exit {r.returncode}" if r.returncode not in (0, 1, 3) else "")
    res.failed = absent + res.failed
    if absent and res.status == "saved":
        res.status = "partial"
    return res


def describe(res: SaveResult) -> str:
    """One plain sentence for the person (CLI, MCP and web all show it)."""
    n = len(res.saved) + len(res.failed)
    if res.status == "skipped":
        return res.message or "nothing to save to Photos"
    if res.status == "denied":
        if res.authorization == "restricted":
            return "Photos access is restricted on this Mac (parental controls or a device policy), so nothing was saved to Photos."
        return f"Photos permission was denied, so nothing was saved to Photos. {DENIED_HELP}"
    if res.status == "interrupted":
        return res.message
    if res.status == "saved":
        return f"Saved {len(res.saved)} video(s) to the Photos library."
    first = f" First problem: {res.failed[0][0].name}: {res.failed[0][1]}." if res.failed else ""
    if res.status == "partial":
        return f"Saved {len(res.saved)} of {n} videos to the Photos library; {len(res.failed)} failed.{first}"
    return f"Nothing was saved to Photos.{first}"


def videos_of_export(out_dir: Path, doc: dict) -> list[Path]:
    """The video files an export wrote, from its export.json-shaped document: the clips that were cut (not failed, not removed after
    a reel-only export) in order, then the reel when it was built. Names are joined to `out_dir`; only plain files are returned."""
    names: list[str] = []
    for c in doc.get("clips") or []:
        if isinstance(c, dict) and c.get("file") and not c.get("error") and c.get("kept") is not False:
            names.append(str(c["file"]))
    reel = doc.get("reel")
    if isinstance(reel, dict) and reel.get("file") and reel.get("ok") is not False:
        names.append(str(reel["file"]))
    out: list[Path] = []
    for n in names:
        p = out_dir / n
        if os.path.basename(n) == n and p.suffix.lower() in VIDEO_SUFFIXES and p.is_file() and not p.is_symlink():
            out.append(p)
    return out
