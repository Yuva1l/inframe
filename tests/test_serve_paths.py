"""inframe serve path rules (docs/T9A_DESIGN.md 3.4 Path safety, 2.3 state folders, 2.8 target files)."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from inframe import serve_paths as sp
from inframe.serve_http import HttpError


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("INFRAME_CACHE", raising=False)
    return h.resolve()


def code(fn, *a):
    with pytest.raises(HttpError) as e:
        fn(*a)
    return e.value.code


MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------- check_read
def test_check_read_file_and_dir(tmp_path):
    f = write(tmp_path / "a.txt", b"x")
    assert sp.check_read(f) == f.resolve()
    assert sp.check_read(tmp_path, "dir") == tmp_path.resolve()
    assert code(sp.check_read, tmp_path) == 404  # a folder is not a file
    assert code(sp.check_read, f, "dir") == 404
    assert code(sp.check_read, tmp_path / "missing") == 404
    assert code(sp.check_read, "relative/a.txt") == 400
    assert code(sp.check_read, "") == 400
    assert code(sp.check_read, str(f) + "\0") == 400
    with pytest.raises(ValueError):
        sp.check_read(f, "socket")


def test_check_read_refuses_hidden_components_after_resolving(tmp_path):
    f = write(tmp_path / ".secret" / "a.txt", b"x")
    assert code(sp.check_read, f) == 403
    g = write(tmp_path / "ok" / ".dot.txt", b"x")
    assert code(sp.check_read, g) == 403
    link = tmp_path / "plain-name.txt"
    os.symlink(f, link)  # a harmless name that points at a hidden file
    assert code(sp.check_read, link) == 403
    ok = write(tmp_path / "fine" / "a.txt", b"x")
    link2 = tmp_path / "link2.txt"
    os.symlink(ok, link2)
    assert sp.check_read(link2) == ok.resolve()


def test_check_read_refuses_library_except_inframe_cache(home, tmp_path, monkeypatch):
    f = write(home / "Library" / "Mobile Documents" / "a.txt", b"x")
    assert code(sp.check_read, f) == 403
    cache = home / "Library" / "Caches" / "inframe"
    c = write(cache / "serve" / "abc" / "proxy.mp4", b"x")
    monkeypatch.setenv("INFRAME_CACHE", str(cache))
    assert sp.check_read(c) == c.resolve()
    other = write(home / "Library" / "Caches" / "other" / "x", b"x")
    assert code(sp.check_read, other) == 403
    assert sp.check_read(write(home / "Movies" / "a.txt", b"x")).name == "a.txt"


def test_library_rule_ignores_letter_case(home, tmp_path):
    """APFS ignores case but resolve() keeps it: ~/LIBRARY/... must be refused like ~/Library/..."""
    f = write(home / "Library" / "Messages" / "secret.mp4", MP4)
    if not (home / "LIBRARY" / "Messages" / "secret.mp4").exists():
        pytest.skip("case-sensitive file system: ~/LIBRARY is another folder")
    assert code(sp.check_read, home / "Library" / "Messages" / "secret.mp4") == 403
    assert code(sp.check_read, home / "LIBRARY" / "Messages" / "secret.mp4") == 403
    assert code(sp.check_video, home / "LIBRARY" / "Messages" / "secret.mp4") == 403
    assert code(sp.check_read, home / "library" / "Messages", "dir") == 403
    assert code(sp.check_write, home / "library" / "LaunchAgents") == 403
    assert code(sp.check_write, home / "LIBRARY" / "Messages" / "new" / "deeper") == 403
    assert f.exists() and sp.check_read(write(home / "Movies" / "a.txt", b"x")).name == "a.txt"


def test_picker_temp_folder_is_readable_even_inside_library(home, monkeypatch):
    tmp = home / "Library" / "Tmp"
    tmp.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    d = tmp / "inframe-picked-abc"
    f = write(d / "IMG.mov", MP4)
    assert sp.check_read(f) == f.resolve()
    assert code(sp.check_read, write(tmp / "other-dir" / "IMG.mov", MP4)) == 403


# ---------------------------------------------------------------- check_video / check_audio
@pytest.mark.parametrize("name,data,ok", [
    ("a.mp4", MP4, True),
    ("a.m4v", MP4, True),
    ("a.MOV", b"\x00\x00\x00\x08wide\x00\x00\x00\x00", True),
    ("a.mov", b"\x00\x00\x10\x00mdat0000", True),
    ("a.mov", b"\x00\x00\x10\x00moov0000", True),
    ("a.mp4", b"\x00\x00\x00\x08free0000", True),
    ("a.mp4", b"#EXTM3U\n#EXT-X-VERSION:3\n", False),
    ("a.mp4", b"ffconcat version 1.0\nfile x\n", False),
    ("a.mp4", b"v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n", False),
    ("a.mp4", b"abc", False),
    ("a.mp4", b"", False),
    ("a.mp4", b"\x1a\x45\xdf\xa3....", False),
    ("a.mkv", b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81", True),
    ("a.webm", b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81", True),
    ("a.webm", MP4, False),
    ("a.ogv", b"OggS\x00\x02", True),
    ("a.ogv", MP4, False),
    ("a.avi", b"RIFF\x00\x00\x00\x00AVI ", False),
    ("a.txt", MP4, False),
    ("a", MP4, False),
])
def test_check_video_extension_and_magic(tmp_path, name, data, ok):
    f = write(tmp_path / name, data)
    if ok:
        assert sp.check_video(f) == f.resolve()
    else:
        assert code(sp.check_video, f) == 415


@pytest.mark.parametrize("name,data,ok", [
    ("a.mp3", b"ID3\x04\x00\x00", True),
    ("a.mp3", b"\xff\xfb\x90\x00", True),
    ("a.mp3", b"\xff\x00\x00", False),
    ("a.m4a", MP4, True),
    ("a.aac", b"\xff\xf1\x50\x80", True),
    ("a.aac", b"ID3\x03", True),
    ("a.wav", b"RIFF\x24\x00\x00\x00WAVEfmt ", True),
    ("a.wav", b"RIFF\x24\x00\x00\x00AVI LIST", False),
    ("a.flac", b"fLaC\x00\x00", True),
    ("a.ogg", b"OggS\x00", True),
    ("a.opus", b"OggS\x00", True),
    ("a.mp3", b"#EXTM3U\n", False),
    ("a.m4a", b"#EXTM3U\n", False),
    ("a.mp4", MP4, False),
    ("a.txt", b"ID3", False),
])
def test_check_audio_extension_and_magic(tmp_path, name, data, ok):
    f = write(tmp_path / name, data)
    if ok:
        assert sp.check_audio(f) == f.resolve()
    else:
        assert code(sp.check_audio, f) == 415


# ---------------------------------------------------------------- check_write
def test_check_write(home, tmp_path):
    new = tmp_path / "out" / "deeper"
    assert sp.check_write(new) == new.resolve()  # need not exist yet
    assert not new.exists()
    assert code(sp.check_write, tmp_path / ".hidden" / "x") == 403
    assert code(sp.check_write, home / "Library" / "Mobile Documents" / "x") == 403
    assert code(sp.check_write, "relative") == 400
    f = write(tmp_path / "afile", b"x")
    assert code(sp.check_write, f / "sub") == 403  # the nearest parent is a file
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        if not os.access(ro, os.W_OK):
            assert code(sp.check_write, ro / "new") == 403
            assert code(sp.check_write, ro) == 403
    finally:
        ro.chmod(0o755)


def test_check_write_allows_cache_and_state_inside_library(home, monkeypatch):
    cache = home / "Library" / "Caches" / "inframe"
    monkeypatch.setenv("INFRAME_CACHE", str(cache))
    monkeypatch.setenv("INFRAME_SERVE_STATE", str(home / "Library" / "Application Support" / "inframe" / "serve"))
    (home / "Library" / "Caches").mkdir(parents=True)
    (home / "Library" / "Application Support").mkdir(parents=True)
    assert sp.check_write(cache / "serve" / "abc") == (cache / "serve" / "abc").resolve()
    assert sp.check_write(sp.state_root()) == sp.state_root().resolve()


# ---------------------------------------------------------------- state folders, target_file
def test_state_folders(home, monkeypatch):
    monkeypatch.delenv("INFRAME_SERVE_STATE", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    if sys.platform == "darwin":
        assert sp.cache_root() == home / "Library" / "Caches" / "inframe" / "serve"
        assert sp.state_root() == home / "Library" / "Application Support" / "inframe" / "serve"
        assert sp.default_out_parent() == home / "Movies" / "inframe"
    else:
        assert sp.cache_root() == home / ".cache" / "inframe" / "serve"
        assert sp.state_root() == home / ".local" / "share" / "inframe" / "serve"
        assert sp.default_out_parent() == home / "inframe"
    monkeypatch.setenv("INFRAME_CACHE", str(home / "c"))
    monkeypatch.setenv("INFRAME_SERVE_STATE", str(home / "s"))
    assert sp.cache_root() == home / "c" / "serve" and sp.state_root() == home / "s"
    # the cache root follows the models.cache_dir rule
    from inframe import models

    assert models.cache_dir().parent == sp.cache_base()


def test_target_file(tmp_path):
    out = tmp_path / "IMG_0001_inframe"
    assert sp.target_file(out, "person", "") == out / "targets" / "target-person.json"
    assert sp.target_file(out, "dog", "") == out / "targets" / "target-dog.json"
    assert sp.target_file(out, "person", "photos:p1") == out / "targets" / "target-person-photos-p1.json"
    assert sp.target_file(out, "person", "saved:alice") == out / "targets" / "target-person-saved-alice.json"
    names = {sp.target_file(out, c, k).name for c in ("person", "dog") for k in ("", "photos:p1", "saved:a")}
    assert len(names) == 6  # one file per (class, target key): a picker never touches another's file
    evil = sp.target_file(out, "../../etc", "saved:../../x")
    assert evil.parent == out / "targets" and "/" not in evil.name and ".." not in evil.name


# ---------------------------------------------------------------- picker temp folders
def old(path: Path, age_s: float):
    t = time.time() - age_s
    os.utime(path, (t, t))


def test_sweep_picked(tmp_path):
    day = 24 * 3600
    stale = tmp_path / "inframe-picked-stale"
    write(stale / "a.mov", b"x")
    old(stale, 2 * day)
    fresh = tmp_path / "inframe-picked-fresh"
    write(fresh / "a.mov", b"x")
    mine = tmp_path / "inframe-picked-mine"
    write(mine / "owner.pid", str(os.getpid()).encode())
    old(mine, 2 * day)
    dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_proc.wait()
    dead = tmp_path / "inframe-picked-dead"
    write(dead / "owner.pid", str(dead_proc.pid).encode())
    old(dead, 2 * day)
    junk = tmp_path / "inframe-picked-junk"
    write(junk / "owner.pid", b"not a number")
    old(junk, 2 * day)
    other = tmp_path / "something-else"
    write(other / "a", b"x")
    old(other, 2 * day)
    target = tmp_path / "precious"
    write(target / "keep", b"x")
    link = tmp_path / "inframe-picked-link"
    os.symlink(target, link)
    old(target, 2 * day)
    removed = {p.name for p in sp.sweep_picked(tmp_path)}
    assert removed == {"inframe-picked-stale", "inframe-picked-dead", "inframe-picked-junk"}
    assert fresh.exists() and mine.exists() and other.exists() and (target / "keep").exists() and link.is_symlink()
    assert not stale.exists() and not dead.exists()


def test_new_picked_dir_is_owned_and_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    d = sp.new_picked_dir()
    assert d.parent == tmp_path and d.name.startswith("inframe-picked-")
    assert (d / "owner.pid").read_text() == str(os.getpid()) and (d.stat().st_mode & 0o777) == 0o700
    foreign = tmp_path / "inframe-picked-foreign"
    write(foreign / "owner.pid", b"1")
    write(d / "IMG.mov", b"x")
    assert [p.name for p in sp.cleanup_own_picked(tmp_path)] == [d.name]
    assert not d.exists() and foreign.exists()
