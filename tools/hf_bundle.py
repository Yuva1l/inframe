"""Stage the folder that is uploaded to the Hugging Face model repo (docs/T9C_DEPLOY.md, docs/T9C_DESIGN.md section 1).

    uv run python tools/hf_bundle.py --out ~/inframe-hf [--models $INFRAME_MODELS/onnx]

It puts the three model files named in web/pages/manifest.json (the manifest the page trusts) into --out by hardlink, so nothing is copied and no
disk is used twice, plus manifest.json (a byte-for-byte copy), LICENSE-APACHE, LICENSE-MIT, NOTICE, THIRD_PARTY_NOTICES (tools/hf_files/) and a generated README.md model
card. A file whose size or sha256 differs from the manifest is refused and nothing is staged. It never uploads and never contacts a host: the
upload is `hf upload`, run by the maintainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES_DIR = ROOT / "tools" / "hf_files"
MANIFEST = ROOT / "web" / "pages" / "manifest.json"
EXTRAS = ("LICENSE-APACHE", "LICENSE-MIT", "NOTICE", "THIRD_PARTY_NOTICES")  # copied from FILES_DIR, must exist and be non-empty
STAGED_EXTRA = ("manifest.json", "README.md") + EXTRAS


class BundleError(Exception):
    """Something that must stop the staging: nothing has been written when this is raised."""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    try:
        m = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise BundleError(f"{path}: cannot read the manifest ({e})") from e
    files = m.get("files") if isinstance(m, dict) else None
    if not isinstance(m, dict) or m.get("format") != 1 or not isinstance(files, dict) or not files:
        raise BundleError(f"{path}: not a model manifest (format 1 with files)")
    for name, meta in files.items():
        if "/" in name or name in STAGED_EXTRA or name.startswith("."):
            raise BundleError(f"{path}: file name {name!r} is not allowed in the bundle")
        if not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str) or len(meta["sha256"]) != 64 or not isinstance(meta.get("bytes"), int):
            raise BundleError(f"{path}: {name} has no bytes and sha256")
    return m


def verify(models: Path, manifest: dict) -> None:
    """Every manifest file is in `models` with the manifest's size and sha256; raises BundleError listing all that are not."""
    bad = []
    for name, meta in manifest["files"].items():
        p = models / name
        if not p.is_file():
            bad.append(f"{name}: not found in {models}")
        elif p.stat().st_size != meta["bytes"]:
            bad.append(f"{name}: {p.stat().st_size} bytes, the manifest says {meta['bytes']}")
        elif sha256(p) != meta["sha256"]:
            bad.append(f"{name}: sha256 {sha256(p)[:16]}... differs from the manifest's {meta['sha256'][:16]}...")
    if bad:
        raise BundleError("refusing to stage; the files do not match web/pages/manifest.json:\n  " + "\n  ".join(bad))


def model_card(manifest: dict) -> str:
    rows = "\n".join(
        f"| `{n}` | {m.get('role', '')} | {m['bytes']:,} | `{m['sha256']}` | {m.get('license', '')} | {m.get('source', '')} |"
        for n, m in sorted(manifest["files"].items())
    )
    return f"""---
license: apache-2.0
library_name: onnx
tags:
  - onnx
  - object-detection
  - image-feature-extraction
  - webgpu
---

# inframe browser models

The three files the [inframe](https://github.com/OWNER/inframe) web page downloads to cut a video into the clips where a chosen target is in frame,
entirely in the visitor's browser. The video never leaves the browser; only these files are fetched, once per browser.

| file | role | bytes | sha256 | license | origin |
|---|---|---|---|---|---|
{rows}

`manifest.json` holds the same digests. The page carries its own copy of the manifest and checks every file against it before using it, so a
changed file is dropped.

## Licenses

`rfdetr_small_v1.onnx` and `dinov2_s_v1.onnx` are Apache-2.0 (`LICENSE-APACHE`); `ort-wasm-simd-threaded.asyncify.wasm` is MIT (`LICENSE-MIT`).
`NOTICE` says where each file comes from and what was changed (a format conversion only, no retraining). The card header says apache-2.0 because
the two weight files are; the runtime file keeps its own MIT license. The wasm runtime is a statically linked binary that also contains
abseil-cpp and onnx (Apache-2.0), protobuf (BSD-3-Clause), Eigen (MPL-2.0) and a few smaller libraries; their notices and license texts are in
`THIRD_PARTY_NOTICES`.

## How they were made

`tools/export_onnx.py` in the inframe repository (RF-DETR small with rfdetr 1.5.2, DINOv2-S from facebook/dinov2-small at opset 17, fp32, and a copy of
the onnxruntime-web 1.30.0 asyncify wasm):

```
INFRAME_MODELS=... python tools/export_onnx.py
```

## Use

Fetch a file from a fixed commit (`/resolve/<40-hex commit>/<file>`), never from `main`, and compare its sha256 with `manifest.json`.
"""


def stage(models: Path, out: Path, manifest_path: Path = MANIFEST, files_dir: Path = FILES_DIR) -> list[str]:
    """Verify, then hardlink the model files into `out` and write the rest. Returns the staged names."""
    manifest = load_manifest(manifest_path)
    for n in EXTRAS:
        f = files_dir / n
        if not f.is_file() or f.stat().st_size == 0:
            raise BundleError(f"{f}: {n} is missing or empty; Apache-2.0 requires the license and NOTICE to travel with the weights")
    verify(models, manifest)
    allowed = set(manifest["files"]) | set(STAGED_EXTRA)
    if out.exists():
        if not out.is_dir():
            raise BundleError(f"{out} is not a folder")
        stray = sorted(p.name for p in out.iterdir() if p.name not in allowed)
        if stray:
            raise BundleError(f"{out} holds files that are not part of the bundle ({', '.join(stray)}); use an empty folder")
    out.mkdir(parents=True, exist_ok=True)
    staged = []
    for name in sorted(manifest["files"]):
        src, dst = models / name, out / name
        if dst.exists() or dst.is_symlink():
            if dst.exists() and os.path.samefile(src, dst):
                staged.append(name)
                continue
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError as e:  # another volume: fall back to a copy rather than fail
            print(f"hardlink failed for {name} ({e}); copying instead", file=sys.stderr)
            shutil.copyfile(src, dst)
        staged.append(name)
    shutil.copyfile(manifest_path, out / "manifest.json")
    for n in EXTRAS:
        shutil.copyfile(files_dir / n, out / n)
    (out / "README.md").write_text(model_card(manifest))
    staged += ["manifest.json", *EXTRAS, "README.md"]
    # the staged copy is what gets uploaded: check it again, so a link to a file that changed in between cannot slip through
    verify(out, manifest)
    return staged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="folder to stage into (created; must be empty or an earlier staging)")
    ap.add_argument("--models", default=None, help="folder with the model files (default $INFRAME_MODELS/onnx)")
    ap.add_argument("--manifest", default=str(MANIFEST), help="the manifest the page trusts (default web/pages/manifest.json)")
    a = ap.parse_args(argv)
    models = a.models or (os.path.join(os.environ["INFRAME_MODELS"], "onnx") if os.environ.get("INFRAME_MODELS") else None)
    if not models:
        print("set INFRAME_MODELS or pass --models", file=sys.stderr)
        return 2
    try:
        staged = stage(Path(models).expanduser(), Path(a.out).expanduser(), Path(a.manifest))
    except BundleError as e:
        print(f"hf_bundle: {e}", file=sys.stderr)
        return 1
    out = Path(a.out).expanduser()
    for n in staged:
        print(f"{(out / n).stat().st_size:>12}  {n}")
    print(f"staged {len(staged)} files in {out}; upload with: hf upload <user>/inframe-models {out} .  (see docs/T9C_DEPLOY.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
