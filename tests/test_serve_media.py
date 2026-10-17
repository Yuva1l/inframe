"""inframe serve, media routes (docs/T9A_DESIGN.md sections 2.4, 2.5, 2.9 and 3.2, rows 4-18): open, dialogs, Photos picker, locate,
file browser, recent, Range, proxy, filmstrip and frames. Synthetic clips only; no model, no network beyond 127.0.0.1."""
import io
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from PIL import Image
from serve_helpers import call, start_server

from inframe import picker, serve_media, serve_paths
from inframe.serve_media import Task


@pytest.fixture
def srv(tmp_path):
    base, app, stop = start_server(tmp_path)
    yield app
    stop()


def open_video(app, path) -> dict:
    r = call(app, "POST", "/api/media/open", {"path": str(path)})
    assert r.status == 200, r.text
    return r.json()


def wait_for(fn, what: str, timeout: float = 60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def film_done(app, mid) -> dict:
    return wait_for(lambda: (d := call(app, "GET", f"/api/media/{mid}/filmstrip").json())["state"] != "running" and d, "filmstrip")


def proxy_done(app, mid) -> dict:
    return wait_for(lambda: (d := call(app, "GET", f"/api/media/{mid}/proxy").json())["state"] != "running" and d, "proxy")


# ---------------------------------------------------------------- row 4, 11, 12: open, info, the original
def test_open_info_and_video_range(srv, synth, late, tmp_path):
    info = open_video(srv, synth["plain"])
    assert info["name"] == "plain.mp4" and info["codec"] == "h264" and info["duration"] == pytest.approx(3.0, abs=0.1)
    assert info["proxy"]["state"] == "none" and info["filmstrip_state"] in ("running", "done")
    assert info["video_url"] == f"api/media/{info['id']}/video"
    assert call(srv, "GET", f"/api/media/{info['id']}").json()["id"] == info["id"]
    assert call(srv, "GET", "/api/media/000000000000").status == 404

    size = os.path.getsize(synth["plain"])
    r = call(srv, "GET", f"/api/media/{info['id']}/video", headers={"Range": "bytes=10-19"}, key="media")
    assert r.status == 206 and len(r.body) == 10 and r.headers["content-type"] == "video/mp4"
    assert r.headers["content-range"] == f"bytes 10-19/{size}"
    assert call(srv, "GET", f"/api/media/{info['id']}/video", key=None).status == 403  # neither token nor media key
    assert call(srv, "HEAD", f"/api/media/{info['id']}/video", key="media").status == 200

    mov = open_video(srv, late)
    assert call(srv, "GET", f"/api/media/{mov['id']}/video", headers={"Range": "bytes=0-3"}, key="media").headers["content-type"] == "video/quicktime"
    r = call(srv, "GET", f"/api/media/{mov['id']}/video?as=mp4", headers={"Range": "bytes=0-3"}, key="media")
    assert r.headers["content-type"] == "video/mp4"
    assert call(srv, "GET", f"/api/media/{mov['id']}/video?as=avi", key="media").status == 400


def test_open_refusals(srv, tmp_path):
    assert call(srv, "POST", "/api/media/open", {"path": str(tmp_path / "nope.mp4")}).status == 404
    txt = tmp_path / "list.mp4"
    txt.write_text("#EXTM3U\n#EXT-X-VERSION:3\n")
    assert call(srv, "POST", "/api/media/open", {"path": str(txt)}).status == 415
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    assert call(srv, "POST", "/api/media/open", {"path": str(hidden / "a.mp4")}).status in (403, 404)
    assert call(srv, "POST", "/api/media/open", {}).status == 400
    assert call(srv, "POST", "/api/media/open", {"path": 5}).status == 400


# ---------------------------------------------------------------- row 5: the open dialog
def test_open_dialog(srv, synth, monkeypatch):
    monkeypatch.setattr(serve_media, "_darwin", lambda: True)
    seen = []

    def fake(script):
        seen.append(script)
        return subprocess.CompletedProcess(["osascript"], 1, "", "execution error: User canceled. (-128)")

    monkeypatch.setattr(serve_media, "run_osascript", fake)
    r = call(srv, "POST", "/api/media/open-dialog", {"kind": "video"})
    assert r.status == 200 and r.json() == {"cancelled": True}
    assert "choose file" in seen[0] and "public.movie" in seen[0]

    monkeypatch.setattr(serve_media, "run_osascript",
                        lambda s: subprocess.CompletedProcess(["osascript"], 0, synth["plain"] + "\n", ""))
    r = call(srv, "POST", "/api/media/open-dialog", {"kind": "video"})
    assert r.status == 200 and r.json()["name"] == "plain.mp4"

    monkeypatch.setattr(serve_media, "run_osascript", lambda s: subprocess.CompletedProcess(["osascript"], 1, "", "boom"))
    assert call(srv, "POST", "/api/media/open-dialog", {}).status == 500
    assert call(srv, "POST", "/api/media/open-dialog", {"kind": "folder"}).status == 400


def test_dialogs_are_501_off_macos(srv, monkeypatch):
    monkeypatch.setattr(serve_media, "_darwin", lambda: False)
    monkeypatch.setattr(serve_media, "run_osascript", lambda s: pytest.fail("osascript must not run off macOS"))
    for path in ("/api/media/open-dialog", "/api/media/photos-pick"):
        r = call(srv, "POST", path, {})
        assert r.status == 501 and "error" in r.json()


# ---------------------------------------------------------------- row 6: Photos
def test_photos_pick(srv, synth, monkeypatch):
    monkeypatch.setattr(serve_media, "_darwin", lambda: True)
    calls = []

    def fake_pick(out, limit=0, announce=lambda m: None, kind="videos"):
        calls.append((Path(out), limit, kind))
        if len(calls) == 1:
            return picker.PickResult(cancelled=True)
        f = Path(out) / "IMG_0001.mp4"
        shutil.copy(synth["plain"], f)
        return picker.PickResult(files=[f])

    monkeypatch.setattr(picker, "pick", fake_pick)
    assert call(srv, "POST", "/api/media/photos-pick", {}).json() == {"cancelled": True}
    assert not calls[0][0].exists()  # the empty export folder is removed
    r = call(srv, "POST", "/api/media/photos-pick", {})
    assert r.status == 200 and r.json()["picked_copy"] is True and calls[1][1:] == (1, "videos")
    assert call(srv, "GET", "/api/recent").json() == {"recent": []}  # a temp copy is not "recent"

    def broken(*a, **k):
        raise picker.PickerError("needs Xcode tools")

    monkeypatch.setattr(picker, "pick", broken)
    r = call(srv, "POST", "/api/media/photos-pick", {})
    assert r.status == 500 and r.json()["error"] == "needs Xcode tools"


# ---------------------------------------------------------------- row 7: locate
def test_locate(srv, synth, tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b" / "deep"
    a.mkdir()
    b.mkdir(parents=True)
    f1 = a / "drop.mp4"
    shutil.copy(synth["plain"], f1)
    os.utime(f1, (1_700_000_000, 1_700_000_000.123))
    size, ms = f1.stat().st_size, f1.stat().st_mtime_ns / 1e6
    monkeypatch.setattr(serve_media, "locate_roots", lambda app: [a, tmp_path / "b", tmp_path / "missing"])
    body = {"name": "drop.mp4", "size": size, "last_modified_ms": int(ms)}  # a browser reports whole milliseconds
    r = call(srv, "POST", "/api/media/locate", body)
    assert r.status == 200 and r.json()["path"] == str(f1.resolve())

    f2 = b / "drop.mp4"  # a second identical file two levels down
    shutil.copy(f1, f2)
    os.utime(f2, (1_700_000_000, 1_700_000_000.123))
    r = call(srv, "POST", "/api/media/locate", body)
    assert r.status == 404 and r.json()["matches"] == 2 and "Open file" in r.json()["error"]

    r = call(srv, "POST", "/api/media/locate", {**body, "size": size + 1})
    assert r.status == 404 and r.json()["matches"] == 0
    r = call(srv, "POST", "/api/media/locate", {**body, "last_modified_ms": int(ms) + 5000})
    assert r.json()["matches"] == 0
    for bad in ({"name": "../x.mp4", "size": 1, "last_modified_ms": 1}, {"name": "x.mp4", "size": -1, "last_modified_ms": 1},
                {"name": "x.mp4", "size": 1}, {"name": "", "size": 1, "last_modified_ms": 1}):
        assert call(srv, "POST", "/api/media/locate", bad).status == 400

    # depth: a match three levels below a root is not looked for; hidden folders are skipped
    g = tmp_path / "c" / "x" / "y" / "z"
    g.mkdir(parents=True)
    shutil.copy2(f1, g / "drop.mp4")
    h = tmp_path / "d" / ".hid"
    h.mkdir(parents=True)
    shutil.copy2(f1, h / "drop.mp4")
    assert serve_media.locate_matches([tmp_path / "c"], "drop.mp4", size, ms) == []
    assert serve_media.locate_matches([tmp_path / "d"], "drop.mp4", size, ms) == []
    assert len(serve_media.locate_matches([tmp_path / "c" / "x"], "drop.mp4", size, ms)) == 1


def test_locate_default_roots_include_recent_folders(srv, synth):
    open_video(srv, synth["plain"])
    roots = serve_media.locate_roots(srv)
    assert Path(synth["plain"]).parent in roots and Path.home() / "Movies" in roots


# ---------------------------------------------------------------- row 8: file browser
def test_fs_list(srv, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Zeta").mkdir(parents=True)
    (home / "alpha").mkdir()
    (home / ".secret").mkdir()
    (home / "Library").mkdir()
    (home / "clip.mp4").write_bytes(b"x" * 10)
    (home / "Beta.MOV").write_bytes(b"y" * 20)
    (home / "song.mp3").write_bytes(b"z")
    (home / "notes.txt").write_text("hi")
    (home / ".hidden.mp4").write_bytes(b"h")
    (home / "alpha" / "inner.mkv").write_bytes(b"i")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(serve_media, "fs_roots", lambda: [{"name": "Home", "path": str(home.resolve())}])

    r = call(srv, "GET", "/api/fs/list?mode=video")
    assert r.status == 200
    d = r.json()
    assert d["dir"] == str(home.resolve()) and d["parent"] is None and d["roots"][0]["name"] == "Home"
    assert [(e["name"], e["kind"]) for e in d["entries"]] == [("alpha", "dir"), ("Zeta", "dir"), ("Beta.MOV", "video"), ("clip.mp4", "video")]
    assert next(e for e in d["entries"] if e["name"] == "clip.mp4")["size"] == 10
    assert all(isinstance(e["mtime"], float) for e in d["entries"])

    d = call(srv, "GET", f"/api/fs/list?mode=audio&dir={home}").json()
    assert [e["name"] for e in d["entries"] if e["kind"] != "dir"] == ["song.mp3"]
    d = call(srv, "GET", f"/api/fs/list?mode=folder&dir={home}").json()
    assert [e["name"] for e in d["entries"]] == ["alpha", "Zeta"]

    sub = call(srv, "GET", f"/api/fs/list?dir={home / 'alpha'}").json()
    assert sub["parent"] == str(home.resolve()) and [e["name"] for e in sub["entries"]] == ["inner.mkv"]

    # refused: outside the roots (including by ".."), hidden, ~/Library, a file, a missing folder, a relative path, a bad mode
    assert call(srv, "GET", f"/api/fs/list?dir={home / 'alpha' / '..' / '..'}").status == 403
    assert call(srv, "GET", "/api/fs/list?dir=/").status == 403
    assert call(srv, "GET", f"/api/fs/list?dir={home / '.secret'}").status == 403
    assert call(srv, "GET", f"/api/fs/list?dir={home / 'Library'}").status == 403
    assert call(srv, "GET", f"/api/fs/list?dir={home / 'clip.mp4'}").status == 404
    assert call(srv, "GET", f"/api/fs/list?dir={home / 'nope'}").status == 404
    assert call(srv, "GET", "/api/fs/list?dir=alpha").status == 400
    assert call(srv, "GET", "/api/fs/list?mode=image").status == 400


def test_fs_roots_are_home_and_volumes():
    roots = serve_media.fs_roots()
    assert roots[0] == {"name": "Home", "path": str(Path.home().resolve())}
    assert all(r["path"] != "/" for r in roots)


# ---------------------------------------------------------------- rows 9, 10: recent
def test_recent(srv, synth, late, tmp_path):
    assert call(srv, "GET", "/api/recent").json() == {"recent": []}
    open_video(srv, synth["plain"])
    open_video(srv, late)
    open_video(srv, synth["plain"])  # opened again: moves to the front, once
    items = call(srv, "GET", "/api/recent").json()["recent"]
    assert [i["name"] for i in items] == ["plain.mp4", "late.mov"] and items[0]["path"] == str(Path(synth["plain"]).resolve())
    assert items[0]["opened"] >= items[1]["opened"] > 1e9
    assert (Path(srv.state_root) / "recent.json").is_file()

    gone = tmp_path / "gone.mp4"
    shutil.copy(synth["plain"], gone)
    open_video(srv, gone)
    gone.unlink()
    assert "gone.mp4" not in [i["name"] for i in call(srv, "GET", "/api/recent").json()["recent"]]  # missing files are dropped

    assert call(srv, "DELETE", "/api/recent").json() == {"recent": []}
    assert call(srv, "GET", "/api/recent").json() == {"recent": []}


def test_recent_keeps_20_and_survives_a_bad_file(srv, synth, tmp_path):
    (Path(srv.state_root)).mkdir(parents=True, exist_ok=True)
    (Path(srv.state_root) / "recent.json").write_text("{not json")
    assert call(srv, "GET", "/api/recent").json() == {"recent": []}
    f = tmp_path / "v.mp4"
    shutil.copy(synth["plain"], f)
    for i in range(23):
        m = type("M", (), {"picked_copy": False, "path": tmp_path / f"v{i}.mp4"})()
        serve_media.recent_add(srv, m)
    saved = serve_media.recent_load(srv)
    assert len(saved) == 20 and saved[0]["name"] == "v22.mp4"


# ---------------------------------------------------------------- rows 16, 17: filmstrip
@pytest.mark.parametrize("which", ["plain", "open_gop"])
def test_filmstrip(srv, synth, open_gop, which):
    path = synth["plain"] if which == "plain" else open_gop
    info = open_video(srv, path)
    doc = film_done(srv, info["id"])
    assert doc["state"] == "done" and doc["done"] is True and "error" not in doc
    assert (doc["cols"], doc["rows"], doc["version"]) == (10, 10, 1)
    cw, ch = doc["cell_w"], doc["cell_h"]
    dw, dh = info["display_w"], info["display_h"]
    assert ch == 108 and cw % 2 == 0 and cw <= 192 and abs(cw - 108 * dw / dh) <= 2
    cells = doc["cells"]
    assert cells and [c["t"] for c in cells] == sorted(c["t"] for c in cells)
    assert len({c["t"] for c in cells}) == len(cells)
    kf = info["keyframes"]
    for c in cells:  # every cell is a keyframe, to the millisecond
        assert min(abs(c["t"] - k) for k in kf) <= 0.001, c
    # one cell per slot (1 s here): the first keyframe of each second
    assert len(cells) == pytest.approx(info["duration"], abs=1.01)
    assert [(c["sheet"], c["i"]) for c in cells] == [(0, i) for i in range(len(cells))]
    assert doc["sheets"] == 1
    r = call(srv, "GET", f"/api/media/{info['id']}/filmstrip/0.jpg", key="media")
    assert r.status == 200 and r.headers["content-type"] == "image/jpeg"
    im = Image.open(io.BytesIO(r.body))
    assert im.format == "JPEG" and im.size == (10 * cw, 10 * ch)
    assert call(srv, "GET", f"/api/media/{info['id']}/filmstrip/1.jpg", key="media").status == 404
    assert call(srv, "GET", f"/api/media/{info['id']}/filmstrip/0.jpg", key=None).status == 403
    assert call(srv, "GET", f"/api/media/{info['id']}").json()["filmstrip_state"] == "done"


def test_filmstrip_slots_and_sheets(srv, tmp_path):
    """A 130 s clip with a keyframe every second: one cell per second, so two sheets (100 + 30)."""
    f = tmp_path / "long.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=5:duration=130",
                    "-c:v", "libx264", "-g", "5", "-keyint_min", "5", "-sc_threshold", "0", "-pix_fmt", "yuv420p", str(f)], check=True)
    info = open_video(srv, f)
    doc = film_done(srv, info["id"])
    assert doc["state"] == "done" and len(doc["cells"]) == 130 and doc["sheets"] == 2
    assert (doc["cells"][99]["sheet"], doc["cells"][99]["i"], doc["cells"][100]["sheet"], doc["cells"][100]["i"]) == (0, 99, 1, 0)
    assert doc["cell_w"] == 108  # square picture: width follows the aspect
    for n in (0, 1):
        assert Image.open(io.BytesIO(call(srv, "GET", f"/api/media/{info['id']}/filmstrip/{n}.jpg", key="media").body)).size == (1080, 1080)


def test_filmstrip_cached_after_restart_and_failure(srv, synth, tmp_path):
    info = open_video(srv, synth["plain"])
    film_done(srv, info["id"])
    # a second app with the same cache folder reads the finished filmstrip from disk and does not decode again
    base, app2, stop2 = start_server(tmp_path / "second")
    try:
        app2.cache_root = srv.cache_root
        m = app2.open_media(synth["plain"])
        assert m.id == info["id"] and serve_media.filmstrip_state(app2, m) == "done"
        d = call(app2, "GET", f"/api/media/{m.id}/filmstrip").json()
        assert d["state"] == "done" and ("filmstrip", m.id) not in app2.media_tasks
    finally:
        stop2()


def test_filmstrip_failure_is_reported(srv, synth, monkeypatch):
    from inframe import decode

    def boom(*a, **k):
        raise RuntimeError("decoder exploded")
        yield  # pragma: no cover

    monkeypatch.setattr(decode, "iter_samples", boom)
    info = open_video(srv, synth["plain"])
    doc = film_done(srv, info["id"])
    assert doc["state"] == "failed" and "decoder exploded" in doc["error"]
    assert call(srv, "GET", f"/api/media/{info['id']}").json()["filmstrip_state"] == "failed"


# ---------------------------------------------------------------- rows 13-15: proxy
def test_proxy_encoder_choice(monkeypatch):
    monkeypatch.delenv("INFRAME_PROXY_ENCODER", raising=False)
    for have, want in (({"h264_videotoolbox", "libx264", "mpeg4"}, "h264_videotoolbox"), ({"libx264", "mpeg4"}, "libx264"),
                       ({"mpeg4"}, "mpeg4"), (set(), "mpeg4")):
        monkeypatch.setattr(serve_media, "_ffmpeg_encoders", lambda have=have: frozenset(have))
        assert serve_media.proxy_encoder() == want
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "libx264")
    assert serve_media.proxy_encoder() == "libx264"
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "rm -rf")
    with pytest.raises(ValueError):
        serve_media.proxy_encoder()


def test_ffmpeg_encoders_parsed_from_the_real_listing():
    serve_media._ffmpeg_encoders.cache_clear()
    enc = serve_media._ffmpeg_encoders()
    assert "mpeg4" in enc  # in every ffmpeg build
    serve_media._ffmpeg_encoders.cache_clear()


def test_proxy_command_is_a_safe_argument_list():
    cmd = serve_media.proxy_command("/v/a b.mov", "/c/proxy.mp4.part", "mpeg4")
    assert cmd[0] == "ffmpeg" and cmd[cmd.index("-protocol_whitelist") + 1] == "file,pipe"
    assert cmd[cmd.index("-i") + 1] == "/v/a b.mov" and cmd[cmd.index("-f") + 1] == "mp4" and cmd[-1] == "/c/proxy.mp4.part"
    assert "-nostdin" in cmd and "-nostats" in cmd and cmd[cmd.index("-loglevel") + 1] == "error"
    assert cmd[cmd.index("-progress") + 1] == "pipe:1" and cmd[cmd.index("-g") + 1] == "15"
    assert serve_media.proxy_command("a", "b", "libx264")[serve_media.proxy_command("a", "b", "libx264").index("-crf") + 1] == "26"
    vt = serve_media.proxy_command("a", "b", "h264_videotoolbox")
    assert vt[vt.index("-b:v") + 1] == "4M"


def check_proxy(app, info, expect_encoder=None):
    r = call(app, "POST", f"/api/media/{info['id']}/proxy", {})
    assert r.status == 200 and r.json()["state"] in ("running", "done"), r.text
    st = proxy_done(app, info["id"])
    assert st["state"] == "done", st
    assert st["url"] == f"api/media/{info['id']}/proxy.mp4" and st["error"] is None and st["fraction"] == 1.0
    # duration within 0.1 s; first_frame_file_time recorded and equal to where the source's video starts (file time)
    assert isinstance(st["first_frame_file_time"], float)
    assert st["first_frame_file_time"] == pytest.approx(info["offset"], abs=0.05)
    mp4 = Path(app.cache_root) / info["id"] / "proxy.mp4"
    from inframe import probe

    pi = probe.probe(str(mp4), with_keyframes=False)
    assert pi.duration == pytest.approx(info["duration"], abs=0.1)
    if expect_encoder:
        assert pi.codec == expect_encoder
    assert max(pi.display_size) <= max(info["display_w"], info["display_h"]) and min(pi.display_size) <= 720
    assert not mp4.with_name("proxy.mp4.part").exists()
    size = mp4.stat().st_size
    r = call(app, "GET", f"/api/media/{info['id']}/proxy.mp4", headers={"Range": "bytes=0-99"}, key="media")
    assert r.status == 206 and len(r.body) == 100 and r.headers["content-range"] == f"bytes 0-99/{size}"
    assert call(app, "GET", f"/api/media/{info['id']}/proxy.mp4", key=None).status == 403
    assert call(app, "POST", f"/api/media/{info['id']}/proxy", {}).json() == {"state": "done", "fraction": 1.0}  # idempotent
    assert call(app, "GET", f"/api/media/{info['id']}").json()["proxy"]["state"] == "done"
    return pi


def test_proxy_mpeg4_late_clip(srv, late, monkeypatch):
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "mpeg4")
    info = open_video(srv, late)
    assert info["offset"] == pytest.approx(2.0, abs=0.05)  # the video starts 2 s into the file
    assert call(srv, "GET", f"/api/media/{info['id']}/proxy").json()["state"] == "none"
    assert call(srv, "GET", f"/api/media/{info['id']}/proxy.mp4", key="media").status == 404
    pi = check_proxy(srv, info, expect_encoder="mpeg4")
    assert (pi.width, pi.height) == (160, 90)  # never upscaled
    doc = Path(srv.cache_root) / info["id"] / "proxy.json"
    assert '"encoder": "mpeg4"' in doc.read_text()


def test_proxy_rotated_is_upright_and_720_short_side(srv, tmp_path, monkeypatch):
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "mpeg4")
    big = tmp_path / "big.mp4"
    rot = tmp_path / "bigrot.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=10:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(big)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-display_rotation", "-90", "-i", str(big), "-c", "copy", str(rot)], check=True)
    info = open_video(srv, rot)
    assert (info["display_w"], info["display_h"]) == (1080, 1920)
    pi = check_proxy(srv, info)
    assert (pi.width, pi.height, pi.rotation) == (720, 1280, 0)  # rotation baked in, short side 720


def test_proxy_real_h264_encoder_when_present(srv, synth, monkeypatch):
    monkeypatch.delenv("INFRAME_PROXY_ENCODER", raising=False)
    serve_media._ffmpeg_encoders.cache_clear()
    enc = serve_media.proxy_encoder()
    if enc == "mpeg4":
        pytest.skip("no h264 encoder (h264_videotoolbox or libx264) in this ffmpeg build; the mpeg4 case above covers the code path")
    print(f"proxy encoder under test: {enc}")
    check_proxy(srv, open_video(srv, synth["plain"]), expect_encoder="h264")


def test_proxy_failure_and_busy(srv, synth, tmp_path, monkeypatch):
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "mpeg4")
    info = open_video(srv, synth["plain"])
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "bogus")
    assert call(srv, "POST", f"/api/media/{info['id']}/proxy", {}).status == 200
    st = proxy_done(srv, info["id"])
    assert st["state"] == "failed" and "INFRAME_PROXY_ENCODER" in st["error"] and st["url"] is None
    assert call(srv, "GET", f"/api/media/{info['id']}").json()["proxy"]["state"] == "failed"

    # another media's proxy is running: 409; the same media's running proxy is idempotent
    srv.media_tasks[("proxy", "aaaaaaaaaaaa")] = Task("proxy")
    r = call(srv, "POST", f"/api/media/{info['id']}/proxy", {})
    assert r.status == 409 and "error" in r.json()
    srv.media_tasks[("proxy", "aaaaaaaaaaaa")].state = "done"
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "mpeg4")
    assert call(srv, "POST", f"/api/media/{info['id']}/proxy", {}).status == 200  # a failed proxy is retried
    assert proxy_done(srv, info["id"])["state"] == "done"


def test_prune_proxies_keeps_the_three_newest(tmp_path):
    root = tmp_path / "serve"
    for i in range(5):
        d = root / f"m{i}"
        d.mkdir(parents=True)
        (d / "proxy.mp4").write_bytes(b"x")
        (d / "proxy.json").write_text("{}")
        os.utime(d / "proxy.mp4", (1_700_000_000 + i, 1_700_000_000 + i))
    removed = serve_media.prune_proxies(root)
    assert sorted(p.parent.name for p in removed) == ["m0", "m1"]
    assert [(root / f"m{i}" / "proxy.mp4").exists() for i in range(5)] == [False, False, True, True, True]
    assert not (root / "m0" / "proxy.json").exists() and (root / "m4" / "proxy.json").exists()


def test_prune_proxies_spares_a_protected_media(tmp_path):
    root = tmp_path / "serve"
    for i in range(5):
        d = root / f"m{i}"
        d.mkdir(parents=True)
        (d / "proxy.mp4").write_bytes(b"x")
        (d / "proxy.json").write_text("{}")
        os.utime(d / "proxy.mp4", (1_700_000_000 + i, 1_700_000_000 + i))
    removed = serve_media.prune_proxies(root, protect={"m0"})
    assert sorted(p.parent.name for p in removed) == ["m1"]  # m0 is oldest but a player reads it; the 3 newest of the rest stay
    assert (root / "m0" / "proxy.mp4").exists() and (root / "m0" / "proxy.json").exists()


def test_a_proxy_a_player_fetched_survives_the_prune_after_a_new_proxy(srv, synth, monkeypatch):
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "mpeg4")
    info = open_video(srv, synth["plain"])
    mid = info["id"]
    assert call(srv, "POST", f"/api/media/{mid}/proxy", {}).status == 200
    assert proxy_done(srv, mid)["state"] == "done"
    mp4 = Path(srv.cache_root) / mid / "proxy.mp4"
    os.utime(mp4, (1_600_000_000, 1_600_000_000))  # the oldest of all
    assert call(srv, "GET", f"/api/media/{mid}/proxy.mp4", key="media").status in (200, 206)  # a tab plays it
    for i in range(4):
        d = Path(srv.cache_root) / f"n{i}"
        d.mkdir()
        (d / "proxy.mp4").write_bytes(b"x")
    serve_media.prune_proxies_for(srv)  # what a new proxy's thread runs
    assert mp4.is_file()


def test_prune_filmstrips_keeps_the_newest_and_protected(tmp_path):
    root = tmp_path / "serve"
    for i in range(5):
        d = root / f"m{i}"
        (d / "filmstrip").mkdir(parents=True)
        (d / "filmstrip" / "sheet_000.jpg").write_bytes(b"x")
        (d / "filmstrip.json").write_text('{"done": true}')
        os.utime(d / "filmstrip.json", (1_700_000_000 + i, 1_700_000_000 + i))
    (root / "m5").mkdir()
    (root / "m5" / "proxy.mp4").write_bytes(b"x")  # no filmstrip: untouched
    removed = serve_media.prune_filmstrips(root, keep=2, protect={"m0"})
    assert sorted(p.parent.name for p in removed) == ["m1", "m2"]
    assert [(root / f"m{i}" / "filmstrip").exists() for i in range(5)] == [True, False, False, True, True]
    assert not (root / "m1" / "filmstrip.json").exists() and (root / "m0" / "filmstrip.json").exists()
    assert (root / "m5" / "proxy.mp4").exists()


def test_a_finished_filmstrip_prunes_the_old_ones(srv, synth, monkeypatch):
    monkeypatch.setattr(serve_media, "FILM_KEEP", 0)  # the open media is protected, so none of the others stay
    old = Path(srv.cache_root) / "aaaaaaaaaaaa"
    (old / "filmstrip").mkdir(parents=True)
    (old / "filmstrip" / "sheet_000.jpg").write_bytes(b"x")
    (old / "filmstrip.json").write_text('{"done": true}')
    os.utime(old / "filmstrip.json", (1_600_000_000, 1_600_000_000))
    info = open_video(srv, synth["plain"])
    assert film_done(srv, info["id"])["state"] == "done"
    wait_for(lambda: not (old / "filmstrip").exists(), "the old filmstrip is pruned")
    assert (Path(srv.cache_root) / info["id"] / "filmstrip" / "sheet_000.jpg").is_file()


def test_close_stops_a_running_proxy(srv, synth, monkeypatch):
    monkeypatch.setenv("INFRAME_PROXY_ENCODER", "mpeg4")
    info = open_video(srv, synth["plain"])
    monkeypatch.setattr(serve_media, "proxy_command", lambda src, out, enc: ["sleep", "30"])
    call(srv, "POST", f"/api/media/{info['id']}/proxy", {})
    t = wait_for(lambda: srv.media_tasks.get(("proxy", info["id"])) if getattr(srv.media_tasks.get(("proxy", info["id"])), "proc", None) else None,
                 "ffmpeg child")
    t0 = time.time()
    srv.close()
    assert time.time() - t0 < 10 and t.proc.poll() is not None and t.state == "failed"


# ---------------------------------------------------------------- row 18: frames
def frame(app, mid, t=None, w=None):
    q = "&".join(f"{k}={v}" for k, v in (("t", t), ("w", w)) if v is not None)
    return call(app, "GET", f"/api/media/{mid}/frame.jpg" + (f"?{q}" if q else ""), key="media")


def test_frame_jpeg_sizes_and_file_time(srv, late, synth, monkeypatch):
    info = open_video(srv, synth["plain"])  # 320x180
    r = frame(srv, info["id"], t=1.0)
    assert r.status == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.body)).size == (320, 180)  # never upscaled, even though w defaults to 960
    assert Image.open(io.BytesIO(frame(srv, info["id"], t=1.0, w=160).body)).size == (160, 90)
    assert Image.open(io.BytesIO(frame(srv, info["id"], t=1.0, w=99999).body)).size == (320, 180)  # w is capped at 1280
    assert frame(srv, info["id"]).status == 200  # t defaults to the start
    assert frame(srv, info["id"], t=999).status == 200  # clamped to the end
    assert frame(srv, info["id"], t="abc").status == 400 and frame(srv, info["id"], w="x").status == 400
    assert frame(srv, info["id"], t="nan").status == 400
    assert call(srv, "GET", f"/api/media/{info['id']}/frame.jpg?t=1", key=None).status == 403
    assert call(srv, "GET", "/api/media/000000000000/frame.jpg?t=1", key="media").status == 404

    # file time: `late` starts at 2.0 s and flashes white at file time 7.0-7.4 s
    mid = open_video(srv, late)["id"]
    lum = lambda t: Image.open(io.BytesIO(frame(srv, mid, t=t).body)).convert("L").resize((1, 1)).getpixel((0, 0))
    assert lum(7.2) > 200 and lum(6.8) < 40 and lum(7.6) < 40


def test_frame_cache_is_an_lru_of_32(srv, synth, monkeypatch):
    from inframe import pick

    n = []
    real = pick.grab_frame

    def counting(*a, **k):
        n.append(a[2])
        return real(*a, **k)

    monkeypatch.setattr(pick, "grab_frame", counting)
    mid = open_video(srv, synth["plain"])["id"]
    a = frame(srv, mid, t=1.0).body
    assert frame(srv, mid, t=1.0).body == a and len(n) == 1  # second request: no decode
    assert len(frame(srv, mid, t=1.0, w=160).body) != len(a) and len(n) == 2  # another width is another entry
    for i in range(40):
        frame(srv, mid, t=i / 100)
    assert len(srv.media_tasks["frames"]) == 32
    before = len(n)
    frame(srv, mid, t=1.0)
    assert len(n) == before + 1  # evicted long ago
