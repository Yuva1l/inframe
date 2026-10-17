"""T3a: Photos library reading. The library is faked at osxphotos' PhotosDB.photos() interface; one opt-in test uses a
real osxphotos test library when INFRAME_OSXPHOTOS_TEST_LIB points at one (never your own library)."""
import datetime as dt
import json
import os
from types import SimpleNamespace

import pytest

from inframe import cli, photos


def fake_photo(uuid, date, slow=False, missing=False, path="/lib/o.mov", secret="Paris"):
    # `secret` stands for every field we must not read: it is present on the object and must never come out
    return SimpleNamespace(uuid=uuid, date=date, slow_mo=slow, ismissing=missing, path=None if missing else path,
                           location=(48.85, 2.35), place=secret, original_filename="IMG_0001.MOV")


class FakeDB:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def photos(self, uuid=None, images=True, movies=True, from_date=None, to_date=None, **kw):
        self.calls.append(dict(uuid=uuid, images=images, movies=movies, from_date=from_date, to_date=to_date))
        out = [p for p in self.items if (uuid is None or p.uuid in uuid)]
        if from_date:
            out = [p for p in out if p.date.replace(tzinfo=None) >= from_date]
        if to_date:
            out = [p for p in out if p.date.replace(tzinfo=None) < to_date]
        return out


D = lambda d: dt.datetime(2026, 10, d, 12, 0, tzinfo=dt.timezone.utc)


def test_record_has_only_the_five_fields():
    r = photos.record_of(fake_photo("U1", D(1), slow=True))
    assert set(r.to_dict()) == {"uuid", "date", "slow_mo", "ismissing", "path"}
    assert "Paris" not in json.dumps(r.to_dict()) and "48.85" not in json.dumps(r.to_dict())


def test_list_filters_and_orders():
    db = FakeDB([fake_photo("B", D(5), slow=True), fake_photo("A", D(2)), fake_photo("C", D(9), slow=True)])
    assert [r.uuid for r in photos.list_videos(db)] == ["A", "B", "C"]
    assert [r.uuid for r in photos.list_videos(db, slomo_only=True)] == ["B", "C"]
    assert [r.uuid for r in photos.list_videos(db, since=dt.datetime(2026, 10, 3))] == ["B", "C"]
    assert [r.uuid for r in photos.list_videos(db, until=dt.datetime(2026, 10, 5))] == ["A"]
    assert all(c["images"] is False and c["movies"] is True for c in db.calls)  # videos only


def test_resolve_errors(tmp_path):
    f = tmp_path / "o.mov"
    f.write_bytes(b"x")
    db = FakeDB([fake_photo("OK", D(1), path=str(f)), fake_photo("ICL", D(2), missing=True), fake_photo("GONE", D(3), path=str(tmp_path / "no.mov"))])
    assert photos.resolve(db, "OK").path == str(f)
    with pytest.raises(photos.PhotosError, match="no video with uuid NOPE"):
        photos.resolve(db, "NOPE")
    with pytest.raises(photos.ICloudOnlyError, match="Download Original"):
        photos.resolve(db, "ICL")
    with pytest.raises(photos.ICloudOnlyError, match="not on disk"):
        photos.resolve(db, "GONE")


def test_parse_ref_and_when():
    assert photos.parse_ref("photos:ab-12") == "AB-12"
    for bad in ("photos:", "photos:a/b", "photos:a b"):
        with pytest.raises(photos.PhotosError):
            photos.parse_ref(bad)
    assert photos.parse_when("2026-10-01") == dt.datetime(2026, 10, 1)
    with pytest.raises(photos.PhotosError, match="not a date"):
        photos.parse_when("last week")


def test_no_full_disk_access_is_a_clear_error(monkeypatch, tmp_path):
    import osxphotos
    lib = tmp_path / "x.photoslibrary"
    (lib / "database").mkdir(parents=True)

    def boom(*a, **k):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(osxphotos, "PhotosDB", boom)
    with pytest.raises(photos.FullDiskAccessError, match="Full Disk Access"):
        photos.open_db(str(lib))
    import sqlite3

    def locked(*a, **k):
        raise sqlite3.OperationalError("unable to open database file")
    monkeypatch.setattr(osxphotos, "PhotosDB", locked)
    with pytest.raises(photos.FullDiskAccessError):
        photos.open_db(str(lib))
    # the default library cannot even be named without access
    import osxphotos.utils as u
    monkeypatch.setattr(u, "get_last_library_path", lambda: None)
    monkeypatch.setattr(u, "get_system_library_path", lambda: None)
    with pytest.raises(photos.FullDiskAccessError, match="Full Disk Access"):
        photos.open_db(None)
    with pytest.raises(photos.PhotosError, match="no such Photos library"):
        photos.open_db(str(tmp_path / "missing.photoslibrary"))


def test_slowmo_factor():
    assert photos.slowmo_factor(240) == 8 and photos.slowmo_factor(120) == 4
    assert photos.slowmo_factor(30) == 1 and photos.slowmo_factor(60) == 2 and photos.slowmo_factor(0) == 1
    assert photos.time_map(False, 240)["factor"] == 1.0  # not flagged slow motion: no mapping


def test_cli_list_and_errors(monkeypatch, capsys, tmp_path):
    db = FakeDB([fake_photo("B", D(5), slow=True, path="/lib/b.mov"), fake_photo("A", D(2), missing=True)])
    monkeypatch.setattr(photos, "open_db", lambda library=None: db)
    assert cli.main(["photos", "list", "--slomo", "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert [g["uuid"] for g in got] == ["B"] and got[0]["slow_mo"] is True
    assert cli.main(["photos", "list"]) == 0
    assert "MISSING (iCloud only)" in capsys.readouterr().out
    assert cli.main(["photos", "list", "--since", "soon"]) == 2

    def denied(library=None):
        raise photos.FullDiskAccessError(photos.FDA_HELP)
    monkeypatch.setattr(photos, "open_db", denied)
    assert cli.main(["photos", "list"]) == 2
    assert "Full Disk Access" in capsys.readouterr().err
    assert cli.main(["cut", "photos:ABC"]) == 2
    assert "Full Disk Access" in capsys.readouterr().err


def test_cut_photos_uuid_end_to_end(monkeypatch, capsys, tmp_path, static_box):
    """photos:<uuid> cuts the original in place, writes outside the library, and adds the time map."""
    db = FakeDB([fake_photo("ABC-1", D(1), slow=True, path=static_box)])
    monkeypatch.setattr(photos, "open_db", lambda library=None: db)
    out = tmp_path / "run"
    assert cli.main(["cut", "photos:abc-1", "--mode", "motion", "--out", str(out)]) == 0
    doc = json.loads((out / "segments.json").read_text())
    assert doc["source"]["path"] == static_box
    assert doc["photos"]["uuid"] == "ABC-1" and doc["photos"]["slow_mo"] is True
    assert set(doc["photos"]) == {"uuid", "date", "slow_mo", "time_map"}  # no location, name, album, ...
    f = doc["photos"]["time_map"]["factor"]
    assert f == 1.0  # the 10 fps fixture is not a high-frame-rate original
    c = doc["clips"][0]
    assert c["playback_start"] == pytest.approx(c["start"] * f, abs=1e-3)
    # default output is under the current directory, never next to the original
    monkeypatch.chdir(tmp_path)
    assert cli.main(["cut", "photos:abc-1", "--mode", "motion"]) == 0
    assert (tmp_path / "ABC-1_inframe" / "motion" / "segments.json").is_file()


def doc_of(intervals, fps):
    return dict(source=dict(fps=fps), presence_intervals=[list(x) for x in intervals])


def test_compare_tries_maps_times_through_the_factor():
    orig = doc_of([(10.0, 14.0), (30.0, 33.0), (50.0, 52.5)], 240)
    export = doc_of([(80.5, 112.0), (240.0, 264.8), (400.0, 420.0)], 30)
    r = photos.compare_tries(orig, export)
    assert r["factor"] == 8 and r["ok"] and r["same_count"] and len(r["tries"]) == 3
    assert r["tolerance_playback_s"] == pytest.approx(2.7)
    assert r["max_err_playback_s"] == pytest.approx(0.8, abs=0.01)


def test_compare_tries_catches_count_order_and_drift():
    orig = doc_of([(10, 14), (30, 33)], 240)
    good = [(80, 112), (240, 264)]
    assert photos.compare_tries(orig, doc_of(good, 30))["ok"]
    assert not photos.compare_tries(orig, doc_of(good + [(300, 310)], 30))["ok"]  # one extra try
    assert not photos.compare_tries(orig, doc_of(good[:1], 30))["ok"]  # one missing
    assert not photos.compare_tries(orig, doc_of(good[::-1], 30))["ok"]  # order
    assert not photos.compare_tries(orig, doc_of([(80, 112), (250, 274)], 30))["ok"]  # drift beyond the tolerance
    assert not photos.compare_tries(orig, doc_of(good, 30), factor=1.0)["ok"]  # unmapped times do not match


def test_annotate_writes_playback_times():
    doc = dict(source=dict(fps=240.0), presence_intervals=[[1.0, 2.0]], clips=[dict(start=0.5, end=2.5)])
    photos.annotate(doc, photos.PhotoRecord("U", "2026-10-01T00:00:00", True, False, "/x"))
    assert doc["clips"][0]["playback_start"] == 4.0 and doc["clips"][0]["playback_end"] == 20.0
    assert doc["presence_playback"] == [[8.0, 16.0]]


@pytest.mark.skipif(not os.environ.get("INFRAME_OSXPHOTOS_TEST_LIB"), reason="set INFRAME_OSXPHOTOS_TEST_LIB to an osxphotos test .photoslibrary")
def test_real_osxphotos_test_library():
    db = photos.open_db(os.environ["INFRAME_OSXPHOTOS_TEST_LIB"])
    recs = photos.list_videos(db)
    assert all(set(r.to_dict()) == {"uuid", "date", "slow_mo", "ismissing", "path"} for r in recs)


# ---- fixes: ramp/trim in compare_tries, capture rate, --library errors, licenses ----

def test_compare_tries_with_a_real_time_head_and_a_trim():
    """Photos plays the first second at 1x and the rest at 8x: export = 1 + (s - 1) * 8."""
    ramp = lambda s: 1 + (s - 1) * 8
    orig = doc_of([(2, 4), (6, 7)], 240)
    export = doc_of([(ramp(2), ramp(4)), (ramp(6), ramp(7))], 30)
    assert not photos.compare_tries(orig, export)["ok"]  # one constant factor cannot match (max_err 7.0 > 2.7)
    r = photos.compare_tries(orig, export, head_s=1.0)
    assert r["ok"] and r["max_err_playback_s"] == pytest.approx(0.0, abs=1e-6)
    # a user trim of 0.5 s off the start of the export shifts every boundary
    trimmed = doc_of([(s - 0.5, e - 0.5) for s, e in [(ramp(2), ramp(4)), (ramp(6), ramp(7))]], 30)
    assert photos.compare_tries(orig, trimmed, head_s=1.0, offset_s=0.5)["ok"]
    assert photos.compare_tries(orig, trimmed, head_s=1.0)["max_err_playback_s"] == pytest.approx(0.5)
    # a tail back at 1x from source second 6.5: later tries land earlier on the export
    tail = lambda s: ramp(s) if s <= 6.5 else ramp(6.5) + (s - 6.5)
    e2 = doc_of([(tail(2), tail(4)), (tail(6), tail(7))], 30)
    assert photos.compare_tries(orig, e2, head_s=1.0, tail_from_s=6.5)["ok"]
    assert not photos.compare_tries(orig, e2, head_s=1.0)["ok"]
    assert photos.playback_time(10, 8) == 80 and photos.playback_time(0.5, 8, head_s=1) == 0.5


def _vfr_file(tmp_path):
    """17 frames at 30 fps (a pre-roll) followed by 343 frames at 240 fps: ffprobe's average rate is 175.25."""
    import subprocess
    p = tmp_path / "vfr.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=240",
                    "-frames:v", "360", "-vf", "setpts='if(lt(N,17),N/30,17/30+(N-17)/240)/TB'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return str(p)


def test_factor_uses_the_capture_rate_not_the_average(tmp_path):
    from inframe import probe
    f = _vfr_file(tmp_path)
    info = probe.probe(f, with_keyframes=False)
    assert info.fps == pytest.approx(175.25, abs=0.1)
    assert photos.slowmo_factor(info.fps) == 6  # what the average alone would give
    assert probe.capture_fps(f) == 240
    doc = dict(source=dict(fps=info.fps), presence_intervals=[[1.0, 2.0]], clips=[dict(start=0.5, end=2.5)])
    photos.annotate(doc, photos.PhotoRecord("U", "2026-10-01T00:00:00", True, False, f), probe.capture_fps(f))
    assert doc["photos"]["time_map"]["factor"] == 8 and doc["presence_playback"] == [[8.0, 16.0]]
    assert doc["photos"]["time_map"]["source_fps"] == 240


def test_library_folder_that_is_not_a_photos_library(tmp_path, monkeypatch, capsys):
    pytest.importorskip("osxphotos")
    empty = tmp_path / "not_a_library"
    empty.mkdir()
    with pytest.raises(photos.PhotosError, match="not a Photos library"):
        photos.open_db(str(empty))
    assert cli.main(["photos", "list", "--library", str(empty)]) == 2
    assert "not a Photos library" in capsys.readouterr().err
    assert cli.main(["cut", "photos:ABC", "--library", str(empty)]) == 2
    assert "not a Photos library" in capsys.readouterr().err


def test_readme_logs_the_license_of_every_photos_dependency():
    """Every package osxphotos brings into uv.lock is named in the README's Licenses section."""
    import re
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    lock = tomllib.loads((root / "uv.lock").read_text())
    pkgs = {p["name"]: [d["name"] for d in p.get("dependencies", [])] for p in lock["package"]}

    def closure(starts):
        seen, todo = set(), list(starts)
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo += pkgs.get(n, [])
        return seen

    me = next(p for p in lock["package"] if p["name"] == "inframe")
    others = [d["name"] for d in me.get("dependencies", [])]
    for extra, deps in me.get("optional-dependencies", {}).items():
        others += [d["name"] for d in deps if extra != "photos"]
    for grp, deps in me.get("dev-dependencies", {}).items():
        others += [d["name"] for d in deps if d["name"] != "osxphotos"]
    seen = closure(["osxphotos"]) - closure(others)  # what the photos extra adds
    text = (root / "README.md").read_text().split("## Licenses")[1].lower()
    missing = sorted(n for n in seen if not re.search(r"(?<![\w-])" + re.escape(n.lower()) + r"(?![\w-])", text)
                     and not (n.startswith("pyobjc") and "pyobjc" in text))
    assert not missing, f"not logged in README Licenses: {missing}"
