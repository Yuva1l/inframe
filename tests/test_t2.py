"""T2: saved targets. The colour embedder needs no model; the detector is replaced by one that finds nobody (a photo is then
used whole), so these tests run without torch, RF-DETR or any sample clips."""
import argparse
import asyncio
import csv
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inframe import cli, identity, mcp_server, targets
from inframe.cli import main
from inframe.detect import Detection
from inframe.identity import PhotoTarget, SPECS
from inframe.targets import Saved, StaleTarget, TargetError

from test_t1c import BLUE, GREEN, RED, person, photos_dir, run


@pytest.fixture(autouse=True)
def no_detector(monkeypatch):
    monkeypatch.setattr(identity, "lazy_detect", lambda size="small": (lambda imgs: [[] for _ in imgs]))


def shirts_dir(tmp_path, shirts, name="shirts"):
    d = tmp_path / name
    d.mkdir()
    for i, c in enumerate(shirts):
        Image.fromarray(person(c, 70 + 3 * i, 180)).save(d / f"p{i}.jpg", quality=95)
    return d


def add(name, folder, **kw):
    crops, label = targets.crops_from(f"photos:{folder}")
    return targets.add_crops(name, crops, label, "color", **kw)


# ---------------------------------------------------------------- the store
def test_add_creates_the_target_with_thumbnails_and_embeddings_keyed_by_version(tmp_path):
    r = add("alice", photos_dir(tmp_path, RED, 3))
    assert r["created"] and r["n_crops"] >= 1 and r["version"] == SPECS["color"].version
    s = Saved("alice")
    assert s.dir.parent == targets.store_dir()
    assert all((s.dir / c["file"]).suffix == ".jpg" and (s.dir / c["file"]).is_file() for c in s.crops)
    assert not [f for f in s.dir.rglob("*") if f.suffix in (".mov", ".mp4")]  # never video
    assert list(s.doc["embeddings"]) == [SPECS["color"].version]
    v = s.vectors(SPECS["color"].version)
    assert v is not None and len(v) >= len(s.crops)
    assert s.vectors("dinov2-small/1") is None


def test_the_same_photos_again_add_nothing_and_a_different_outfit_is_added(tmp_path):
    add("a", photos_dir(tmp_path, RED, 3))
    n = len(Saved("a").crops)
    (tmp_path / "x").mkdir()
    again = add("a", photos_dir(tmp_path / "x", RED, 3))
    assert again["added"] == 0 and again["dropped_duplicates"] == again["offered"] and len(Saved("a").crops) == n
    blue = add("a", shirts_dir(tmp_path, [BLUE], "blue"))
    assert blue["added"] == 1 and len(Saved("a").crops) == n + 1
    assert Saved("a").vectors(SPECS["color"].version) is not None  # re-embedded for the new set


def test_the_cap_keeps_different_outfits(tmp_path):
    d = shirts_dir(tmp_path, [RED, (205, 35, 30), BLUE, GREEN, (210, 28, 33)])
    crops, label = targets.crops_from(f"photos:{d}")
    r = targets.add_crops("capped", crops, label, "color", cap=3)
    assert r["n_crops"] == 3
    s = Saved("capped")
    means = [np.asarray(Image.open(s.dir / c["file"]).convert("RGB"))[:60].reshape(-1, 3).mean(0) for c in s.crops]
    dominant = {int(np.argmax(m)) for m in means}
    assert dominant == {0, 1, 2}  # a red, a blue and a green one stayed; the near-copies of red went


def test_a_version_nobody_uses_is_flagged_and_the_thumbnails_are_embedded_again(tmp_path):
    add("y", photos_dir(tmp_path, RED, 2))
    s = Saved("y")
    old = s.doc["embeddings"][SPECS["color"].version]
    s.doc["embeddings"] = {"color-signature/0": old}
    s._write()
    s = Saved("y")
    assert s.stale_versions() == ["color-signature/0"] and s.vectors(SPECS["color"].version) is None
    emb = identity.Embedder("color")
    v = targets.vectors_for("y", emb)
    assert len(v) >= len(s.crops) and Saved("y").vectors(SPECS["color"].version) is not None  # stored for next time


def test_missing_thumbnails_make_a_target_stale_and_the_cli_exits_3(tmp_path, capsys):
    add("z", photos_dir(tmp_path, RED, 2))
    s = Saved("z")
    (s.dir / s.crops[0]["file"]).unlink()
    with pytest.raises(StaleTarget, match="inframe target add z"):
        targets.check_usable("z")
    with pytest.raises(StaleTarget):
        identity.parse_targets(["saved:z"], "color")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert main(["cut", str(video), "--target", "saved:z"]) == 3
    assert "thumbnail" in capsys.readouterr().err


def test_names_are_checked_and_unknown_targets_are_plain_errors(tmp_path, capsys):
    for bad in ("../x", "a b", "", ".hidden", "x" * 80):
        with pytest.raises(TargetError):
            targets.check_name(bad)
    with pytest.raises(TargetError, match="no saved target named 'nobody'"):
        identity.parse_targets(["saved:nobody"], "color")
    assert main(["target", "show", "nobody"]) == 2
    assert main(["target", "remove", "nobody"]) == 2
    assert "nobody" in capsys.readouterr().err


def test_remove_deletes_only_that_target(tmp_path):
    add("one", photos_dir(tmp_path, RED, 2))
    add("two", shirts_dir(tmp_path, [BLUE]))
    targets.remove("one")
    assert targets.names() == ["two"]


def test_export_and_import_round_trip(tmp_path, monkeypatch):
    add("me", photos_dir(tmp_path, RED, 3))
    z = targets.export_target("me", tmp_path)
    assert z.name == "me.inframe-target.zip"
    names = zipfile.ZipFile(z).namelist()
    assert "target.json" in names and any(n.startswith("crops/") for n in names) and any(n.startswith("emb/") for n in names)
    with pytest.raises(TargetError, match="already exists"):
        targets.import_target(z)
    assert targets.import_target(z, name="copy") == "copy"
    a, b = Saved("me"), Saved("copy")
    assert len(a.crops) == len(b.crops) and b.doc["name"] == "copy"
    assert np.allclose(a.vectors(SPECS["color"].version), b.vectors(SPECS["color"].version))
    assert targets.import_target(z, force=True) == "me"
    monkeypatch.setenv("INFRAME_TARGETS", str(tmp_path / "other_machine"))  # a clean store
    assert targets.import_target(z) == "me" and Saved("me").vectors(SPECS["color"].version) is not None


def test_import_refuses_files_outside_the_layout(tmp_path):
    add("me", photos_dir(tmp_path, RED, 1))
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("target.json", (Saved("me").dir / "target.json").read_text())
        f.writestr("../escape.txt", "x")
    with pytest.raises(TargetError, match="unexpected file"):
        targets.import_target(z, name="evil")
    not_zip = tmp_path / "x.zip"
    not_zip.write_text("no")
    with pytest.raises(TargetError, match="not an exported target"):
        targets.import_target(not_zip)
    assert "evil" not in targets.names() and not (tmp_path / "escape.txt").exists()


# ---------------------------------------------------------------- sources
def _video(tmp_path):
    p = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=10:duration=4",
                    "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_crops_from_a_tap_come_from_the_video_the_target_json_names(tmp_path):
    v = _video(tmp_path)
    t = tmp_path / "target.json"
    t.write_text(json.dumps(dict(video=str(v), taps=[dict(time=1.0, box=[0.2, 0.1, 0.5, 0.9]), dict(time=2.5, box=[0.4, 0.1, 0.7, 0.9])])))
    crops, label = targets.crops_from(f"tap:{t}")
    assert label == "tap" and len(crops) == 2 and crops[0].shape[0] > 200
    t.write_text(json.dumps(dict(time=1.0, box=[0.2, 0.1, 0.5, 0.9])))  # a person_NNN.json of an old scan: no video named
    with pytest.raises(TargetError, match="--video"):
        targets.crops_from(f"tap:{t}")
    assert len(targets.crops_from(f"tap:{t}", video=str(v))[0]) == 1
    t.write_text(json.dumps(dict(time=1.0, box=[0.2, 0.1, 0.21, 0.12])))
    with pytest.raises(TargetError, match="too small"):
        targets.crops_from(f"tap:{t}", video=str(v))


def _run_folder(tmp_path, v, target="photos", n=30):
    d = tmp_path / "run"
    d.mkdir()
    (d / "segments.json").write_text(json.dumps(dict(mode="detect", source=dict(path=str(v)), settings=dict(enter=0.5, target=target))))
    with open(d / "samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "score", "x0", "y0", "x1", "y1"])
        for i in range(n):
            t = i * 0.1
            w.writerow([f"{t:.3f}", "0.9" if i % 3 else "0.1", "0.2", "0.1", "0.5", "0.9"] if i % 3 else [f"{t:.3f}", "0.1", "", "", "", ""])
    return d


def test_crops_from_a_run_spread_over_the_samples_where_the_person_was(tmp_path):
    v = _video(tmp_path)
    d = _run_folder(tmp_path, v)
    crops = targets.crops_from_run(str(d), n=8)
    assert 4 <= len(crops) <= 8
    r = targets.add_crops("fromrun", crops, "run", "color")
    assert r["created"] and 1 <= r["n_crops"] <= len(crops)


def test_a_run_that_kept_every_person_is_not_a_target(tmp_path):
    d = _run_folder(tmp_path, _video(tmp_path), target="any-person")
    with pytest.raises(TargetError, match="every person"):
        targets.crops_from_run(str(d / "segments.json"))
    with pytest.raises(TargetError, match="photos:<folder>, tap:<target.json> or run"):
        targets.crops_from("face:3")


# ---------------------------------------------------------------- use: a target saved from clip A finds him in clip B, no tap
def test_a_target_saved_from_clip_a_finds_the_same_person_in_clip_b_with_no_tap(tmp_path):
    # clip A: he wears red and walks beside a stranger in blue; the run is confirmed and its crops are saved
    A = tmp_path / "a"
    A.mkdir()
    Image.fromarray(person(RED, 70, 180)).save(A / "p0.jpg", quality=95)
    add("him", A)
    # clip B: other positions, other track ids, a stranger in blue and one in green; no tap, no photos
    box_b1, box_b2, box_b3 = (0.55, 0.25, 0.85, 0.9), (0.05, 0.3, 0.3, 0.9), (0.35, 0.1, 0.5, 0.5)
    tgt = identity.parse_targets(["saved:him"], "color")
    assert isinstance(tgt, PhotoTarget) and tgt.photos == "saved:him" and not tgt.taps
    hits = run(tgt, 12, [box_b1, box_b2, box_b3], [RED, BLUE, GREEN], [11, 12, 13])
    assert all(h.box == box_b1 for h in hits)  # the red one, found by what he looks like
    assert tgt.info["photos"] >= 1 and tgt.info["embedder_version"] == SPECS["color"].version


def test_a_stranger_alone_is_not_taken_for_him(tmp_path):
    add("him", photos_dir(tmp_path, RED, 2))
    tgt = identity.parse_targets(["saved:him"], "color")
    with pytest.raises(identity.Ambiguous):
        run(tgt, 8, [(0.1, 0.2, 0.4, 0.9)], [BLUE], [1])


def test_pick_accepts_a_saved_target_and_writes_it_into_target_json(tmp_path):
    from inframe.pick import Picker

    add("him", photos_dir(tmp_path, RED, 2))
    v = _video(tmp_path)
    pk = Picker(str(v), tmp_path / "t.json", photos="saved:him", embedder="color")
    pk.taps = [dict(time=1.0, box=[0.1, 0.1, 0.4, 0.9], track_id=None, index=None)]
    pk.save()
    assert json.loads((tmp_path / "t.json").read_text())["photos"] == "saved:him"
    tgt = identity.parse_targets([f"tap:{tmp_path / 't.json'}"], "color")
    assert isinstance(tgt, PhotoTarget) and tgt.photos == "saved:him" and len(tgt.taps) == 1


def test_two_photo_sources_are_refused(tmp_path):
    add("him", photos_dir(tmp_path, RED, 1))
    (tmp_path / "q").mkdir()
    with pytest.raises(identity.TapError, match="one photos"):
        identity.parse_targets([f"photos:{photos_dir(tmp_path / 'q', RED, 1)}", "saved:him"], "color")


def test_saved_then_photos_is_refused_not_silently_dropped(tmp_path):
    add("him", photos_dir(tmp_path, RED, 1))
    (tmp_path / "q").mkdir()
    with pytest.raises(identity.TapError, match="one photos"):
        identity.parse_targets(["saved:him", f"photos:{photos_dir(tmp_path / 'q', RED, 1)}"], "color")


# ---------------------------------------------------------------- the command line
def test_cli_add_list_show_export_import_remove(tmp_path, capsys):
    d = photos_dir(tmp_path, RED, 3)
    assert main(["target", "add", "cli", "--from", f"photos:{d}", "--embedder", "color"]) == 0
    assert "created saved target cli" in capsys.readouterr().out
    assert main(["target", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["name"] == "cli"
    assert main(["target", "show", "cli"]) == 0
    out = capsys.readouterr().out
    assert "embeddings color-signature/1" in out and "STALE" not in out
    z = tmp_path / "cli.zip"
    assert main(["target", "export", "cli", str(z)]) == 0 and z.is_file()
    assert main(["target", "import", str(z)]) == 2  # exists
    assert main(["target", "import", str(z), "--name", "cli2"]) == 0
    assert main(["target", "remove", "cli"]) == 0 and targets.names() == ["cli2"]
    capsys.readouterr()
    assert main(["target", "add", "bad", "--from", "face:1", "--embedder", "color"]) == 2
    assert main(["target", "add", "../bad", "--from", f"photos:{d}", "--embedder", "color"]) == 2


def test_cli_add_crops_needs_a_saved_target(static_box, capsys):
    assert main(["cut", static_box, "--add-crops"]) == 2
    assert "--add-crops needs --target saved:NAME" in capsys.readouterr().err


def test_offer_crops_prints_the_command_and_add_crops_does_it(tmp_path, capsys):
    v = _video(tmp_path)
    d = _run_folder(tmp_path, v)
    ns = argparse.Namespace(add_crops=False, embedder="color", offline=False)
    assert cli.offer_crops("him", d, ns) == 0
    assert f"inframe target add him --from run:{d / 'segments.json'}" in capsys.readouterr().out
    ns.add_crops = True
    assert cli.offer_crops("him", d, ns) == 0
    assert "crops added" in capsys.readouterr().out and "him" in targets.names()
    (d / "samples.csv").unlink()
    assert cli.offer_crops("him", d, ns) == 1


# ---------------------------------------------------------------- MCP
def test_mcp_list_targets_and_saved_target_in_cut_video(tmp_path, static_box, monkeypatch):
    add("mcp1", photos_dir(tmp_path, RED, 2))
    got = asyncio.run(mcp_server.list_targets())
    assert [g["name"] for g in got] == ["mcp1"] and got[0]["n_crops"] >= 1
    with pytest.raises(mcp_server.ToolError, match="no saved target named 'nobody'"):
        asyncio.run(mcp_server.cut_video(static_box, target="saved:nobody", out=str(tmp_path / "o")))
    assert not (tmp_path / "o").exists()
    seen = {}

    def fake(video, out, settings, target, progress=None):
        seen["target"] = target
        raise RuntimeError("stop here")

    monkeypatch.setattr(mcp_server, "run_cut", fake)
    with pytest.raises(mcp_server.ToolError, match="stop here"):
        asyncio.run(mcp_server.cut_video(static_box, target="saved:mcp1", out=str(tmp_path / "o")))
    assert seen["target"].photos == "saved:mcp1"


def test_mcp_add_target_crops(tmp_path):
    v = _video(tmp_path)
    d = _run_folder(tmp_path, v)
    r = asyncio.run(mcp_server.add_target_crops("grown", str(d), embedder="color"))
    assert r["created"] and r["n_crops"] >= 1
    with pytest.raises(mcp_server.ToolError, match="embedder"):
        asyncio.run(mcp_server.add_target_crops("grown", str(d), embedder="nope"))
    with pytest.raises(mcp_server.ToolError, match="no such file"):
        asyncio.run(mcp_server.add_target_crops("grown", str(tmp_path / "nowhere")))


# ---------------------------------------------------------------- stale repair, concurrent writers, failed first add
def test_adding_again_repairs_a_stale_target_instead_of_failing_with_the_same_error(tmp_path, capsys):
    add("st", photos_dir(tmp_path, RED, 2))
    other = shirts_dir(tmp_path, [BLUE], "other")
    s = Saved("st")
    (s.dir / s.crops[0]["file"]).unlink()
    with pytest.raises(StaleTarget):
        targets.check_usable("st")
    assert main(["target", "add", "st", "--from", f"photos:{other}", "--embedder", "color"]) == 0
    capsys.readouterr()
    s = Saved("st")
    assert targets.check_usable("st") is not None and len(s.crops) >= 1
    assert all((s.dir / c["file"]).is_file() for c in s.crops)
    assert s.vectors(SPECS["color"].version) is not None


def test_a_run_that_stores_embeddings_does_not_put_back_an_old_target_json(tmp_path, monkeypatch):
    d = shirts_dir(tmp_path, [RED, BLUE], "two")
    crops, label = targets.crops_from(f"photos:{d}")
    targets.add_crops("race", crops, label, "color", cap=2)
    s = Saved("race")
    s.doc["embeddings"] = {}  # nothing stored for this embedder: vectors_for has to embed
    s._write()
    green = shirts_dir(tmp_path, [GREEN], "green")
    gcrops, glabel = targets.crops_from(f"photos:{green}")
    emb = identity.Embedder("color")
    real = emb.embed
    fired = []

    def embed_and_meanwhile_add(imgs):
        out = real(imgs)
        if not fired:  # another process commits an add while this run is embedding
            fired.append(1)
            r = targets.add_crops("race", gcrops, glabel, "color", cap=2)
            assert r["evicted"] == 1
        return out

    monkeypatch.setattr(emb, "embed", embed_and_meanwhile_add)
    targets.vectors_for("race", emb)
    assert fired
    assert targets.check_usable("race") is not None  # target.json still lists only files that exist
    s = Saved("race")
    assert all((s.dir / c["file"]).is_file() for c in s.crops)
    assert sorted(p.name for p in (s.dir / "crops").glob("*.jpg")) == sorted(Path(c["file"]).name for c in s.crops)


def test_a_first_add_that_fails_leaves_no_folder_that_blocks_add_or_remove(tmp_path, monkeypatch):
    folder = photos_dir(tmp_path, RED, 2)
    real = Saved.store_vectors

    def full_disk(self, version, vecs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Saved, "store_vectors", full_disk)
    with pytest.raises(OSError):
        add("half", folder)
    assert not (targets.store_dir() / "half").exists()
    monkeypatch.setattr(Saved, "store_vectors", real)
    assert add("half", folder)["created"]
    targets.remove("half")


def test_a_later_add_that_fails_keeps_the_target_and_removes_its_new_thumbnails(tmp_path, monkeypatch):
    add("keep", photos_dir(tmp_path, RED, 2))
    before = sorted(p.name for p in (Saved("keep").dir / "crops").glob("*.jpg"))
    monkeypatch.setattr(Saved, "store_vectors", lambda self, v, x: (_ for _ in ()).throw(OSError(28, "full")))
    blue = shirts_dir(tmp_path, [BLUE], "blue")
    with pytest.raises(OSError):
        add("keep", blue)
    assert sorted(p.name for p in (Saved("keep").dir / "crops").glob("*.jpg")) == before


# ---------------------------------------------------------------- lows: import and list robustness
def test_import_corrupt_zip_is_a_target_error(tmp_path):
    add("me", photos_dir(tmp_path, RED, 1))
    z = targets.export_target("me", tmp_path)
    data = z.read_bytes()
    trunc = tmp_path / "trunc.zip"  # central directory intact, member data damaged
    bad = bytearray(data)
    i = bad.find(b"target.json") + len(b"target.json") + 2
    for k in range(i, i + 12):
        bad[k] ^= 0xFF
    trunc.write_bytes(bytes(bad))
    with pytest.raises(TargetError):
        targets.import_target(trunc, name="broken")
    cut = tmp_path / "cut.zip"  # tail cut off: not a zip at all
    cut.write_bytes(data[: len(data) // 2])
    with pytest.raises(TargetError):
        targets.import_target(cut, name="cut")
    assert "broken" not in targets.names() and "cut" not in targets.names()


def test_import_force_keeps_a_folder_that_is_not_a_target(tmp_path):
    add("me", photos_dir(tmp_path, RED, 1))
    z = targets.export_target("me", tmp_path)
    other = targets.store_dir() / "notes"
    other.mkdir()
    (other / "keep.txt").write_text("mine")
    with pytest.raises(TargetError, match="not a saved target"):
        targets.import_target(z, name="notes", force=True)
    assert (other / "keep.txt").read_text() == "mine"


@pytest.mark.parametrize("doc", [
    {"format": 1, "crops": [{"file": "crops/c0001.jpg", "source": 5}], "embeddings": {}},
    {"format": 1, "crops": [], "embeddings": {"v": "oops"}},
    {"format": 1, "crops": [{"file": "crops/c0001.jpg"}], "embeddings": {"v": {"file": 3}}},
    [1, 2],
])
def test_list_targets_survives_a_malformed_target_json(tmp_path, doc):
    add("good", photos_dir(tmp_path, RED, 1))
    d = targets.store_dir() / "bad"
    d.mkdir()
    (d / "target.json").write_text(json.dumps(doc))
    out = {t["name"]: t for t in targets.list_targets()}
    assert "n_crops" in out["good"] and "error" in out["bad"]
