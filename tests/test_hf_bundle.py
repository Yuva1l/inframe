"""tools/hf_bundle.py (docs/T9C_DEPLOY.md): stages the model-repo folder by hardlink and refuses a file that differs from the manifest. Tiny fake files in
tmp folders; no model, no network, nothing is uploaded."""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hf_bundle  # noqa: E402

NAMES = ["dinov2_s_v1.onnx", "ort-wasm-simd-threaded.asyncify.wasm", "rfdetr_small_v1.onnx"]


def make_models(tmp: Path, sizes=(11, 13, 17)) -> tuple[Path, Path]:
    models = tmp / "models"
    models.mkdir()
    files = {}
    for n, k in zip(NAMES, sizes):
        data = (n.encode() * k)[:k * 3]
        (models / n).write_bytes(data)
        files[n] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "license": "MIT" if n.endswith(".wasm") else "Apache-2.0",
                    "role": "x", "source": "https://example.invalid/" + n}
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"format": 1, "files": files}, indent=1, sort_keys=True) + "\n")
    return models, manifest


def test_stage_links_the_models_and_adds_manifest_licenses_notice_and_card(tmp_path):
    models, manifest = make_models(tmp_path)
    out = tmp_path / "out"
    staged = hf_bundle.stage(models, out, manifest)
    assert sorted(p.name for p in out.iterdir()) == sorted(NAMES + ["LICENSE-APACHE", "LICENSE-MIT", "NOTICE", "THIRD_PARTY_NOTICES", "README.md", "manifest.json"])
    assert sorted(staged) == sorted(p.name for p in out.iterdir())
    for n in NAMES:  # hardlinks: the same file, not a second copy
        assert os.path.samefile(models / n, out / n)
        assert (out / n).stat().st_nlink >= 2
    assert (out / "manifest.json").read_bytes() == manifest.read_bytes()
    for n in ("LICENSE-APACHE", "LICENSE-MIT", "NOTICE", "THIRD_PARTY_NOTICES"):
        assert (out / n).stat().st_size > 0
    assert "Apache License" in (out / "LICENSE-APACHE").read_text() and "Permission is hereby granted" in (out / "LICENSE-MIT").read_text()
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: apache-2.0") and all(n in card for n in NAMES)
    assert all(json.loads(manifest.read_text())["files"][n]["sha256"] in card for n in NAMES)
    assert "Apache License 2.0" in (out / "NOTICE").read_text() and "MIT License" in (out / "NOTICE").read_text()


def test_staging_twice_is_fine(tmp_path):
    models, manifest = make_models(tmp_path)
    hf_bundle.stage(models, tmp_path / "out", manifest)
    hf_bundle.stage(models, tmp_path / "out", manifest)


def test_a_wrong_digest_is_refused_and_nothing_is_staged(tmp_path):
    models, manifest = make_models(tmp_path)
    p = models / "rfdetr_small_v1.onnx"
    data = bytearray(p.read_bytes())
    data[0] ^= 1  # same size, different content
    p.write_bytes(bytes(data))
    with pytest.raises(hf_bundle.BundleError, match=r"rfdetr_small_v1\.onnx: sha256 .* differs from the manifest"):
        hf_bundle.stage(models, tmp_path / "out", manifest)
    assert not (tmp_path / "out").exists()


def test_a_wrong_size_and_a_missing_file_are_refused(tmp_path):
    models, manifest = make_models(tmp_path)
    (models / "dinov2_s_v1.onnx").write_bytes(b"short")
    (models / "ort-wasm-simd-threaded.asyncify.wasm").unlink()
    with pytest.raises(hf_bundle.BundleError) as e:
        hf_bundle.stage(models, tmp_path / "out", manifest)
    assert "dinov2_s_v1.onnx: 5 bytes" in str(e.value) and "ort-wasm-simd-threaded.asyncify.wasm: not found" in str(e.value)
    assert not (tmp_path / "out").exists()


def test_a_digest_edited_in_the_manifest_is_refused_too(tmp_path):
    models, manifest = make_models(tmp_path)
    m = json.loads(manifest.read_text())
    m["files"]["dinov2_s_v1.onnx"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(m))
    with pytest.raises(hf_bundle.BundleError, match="differs from the manifest"):
        hf_bundle.stage(models, tmp_path / "out", manifest)


@pytest.mark.parametrize("missing", ["LICENSE-APACHE", "LICENSE-MIT", "NOTICE", "THIRD_PARTY_NOTICES"])
def test_a_missing_license_or_notice_stops_the_staging(tmp_path, missing):
    models, manifest = make_models(tmp_path)
    files = tmp_path / "files"
    files.mkdir()
    for n in hf_bundle.EXTRAS:
        if n != missing:
            (files / n).write_text("text")
    with pytest.raises(hf_bundle.BundleError, match=missing):
        hf_bundle.stage(models, tmp_path / "out", manifest, files)
    assert not (tmp_path / "out").exists()
    (files / missing).write_text("")  # present but empty is as bad
    with pytest.raises(hf_bundle.BundleError, match=missing):
        hf_bundle.stage(models, tmp_path / "out", manifest, files)


def test_a_folder_with_other_files_is_refused(tmp_path):
    models, manifest = make_models(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("not for upload")
    with pytest.raises(hf_bundle.BundleError, match="notes.txt"):
        hf_bundle.stage(models, out, manifest)


@pytest.mark.parametrize("bad", ["../x.onnx", "NOTICE", "README.md"])
def test_a_manifest_naming_a_path_or_one_of_the_extra_files_is_refused(tmp_path, bad):
    models, manifest = make_models(tmp_path)
    m = json.loads(manifest.read_text())
    m["files"][bad] = m["files"].pop("dinov2_s_v1.onnx")
    manifest.write_text(json.dumps(m))
    with pytest.raises(hf_bundle.BundleError, match="not allowed"):
        hf_bundle.stage(models, tmp_path / "out", manifest)


def test_main_exits_1_on_a_mismatch_and_0_on_a_match(tmp_path, capsys):
    models, manifest = make_models(tmp_path)
    args = ["--out", str(tmp_path / "out"), "--models", str(models), "--manifest", str(manifest)]
    assert hf_bundle.main(args) == 0
    assert "hf upload" in capsys.readouterr().out
    (models / "dinov2_s_v1.onnx").write_bytes(b"x" * (models / "dinov2_s_v1.onnx").stat().st_size)
    # the link shares the inode, so the staged copy changed too; a fresh folder shows the refusal
    assert hf_bundle.main(["--out", str(tmp_path / "out2"), *args[2:]]) == 1
    assert "refusing to stage" in capsys.readouterr().err


def test_the_committed_files_are_in_place():
    for n in hf_bundle.EXTRAS:
        assert (hf_bundle.FILES_DIR / n).stat().st_size > 0
    assert "Apache License" in (hf_bundle.FILES_DIR / "LICENSE-APACHE").read_text()
    m = hf_bundle.load_manifest(hf_bundle.MANIFEST)  # the manifest the page trusts: the three files, no path, no extra name
    assert sorted(m["files"]) == NAMES
    notice = (hf_bundle.FILES_DIR / "NOTICE").read_text()
    assert all(n in notice for n in NAMES)


def test_the_wasm_runtime_ships_with_the_notices_of_the_libraries_linked_into_it():
    """The ORT wasm statically contains abseil-cpp and onnx (Apache-2.0), protobuf (BSD-3-Clause: binary redistribution must reproduce its copyright and
    conditions) and Eigen (MPL-2.0); NOTICE must not call the file plain MIT, and the bundle must carry their texts."""
    assert "THIRD_PARTY_NOTICES" in hf_bundle.EXTRAS
    notices = (hf_bundle.FILES_DIR / "THIRD_PARTY_NOTICES").read_text()
    assert "Copyright 2008 Google Inc." in notices and "Redistributions in binary form must reproduce the above" in notices  # protobuf, BSD-3
    assert "Copyright (c) Facebook, Inc. and Microsoft Corporation" in notices  # onnx
    assert "abseil-cpp" in notices and "Mozilla Public License Version 2.0" in notices  # abseil, Eigen
    assert notices.count("Apache License") >= 2
    notice = (hf_bundle.FILES_DIR / "NOTICE").read_text()
    for lib in ("protobuf", "abseil", "onnx", "Eigen"):
        assert lib in notice, lib
    assert "THIRD_PARTY_NOTICES" in notice and "BSD-3-Clause" in notice and "MPL-2.0" in notice
    card = hf_bundle.model_card({"files": {}})
    assert "THIRD_PARTY_NOTICES" in card
