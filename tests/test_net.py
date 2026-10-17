"""net.fetch and the music download path (docs/T9A_DESIGN.md section 4, "Outbound network"). No network: `net._open` and
`net._resolve` are replaced, and the playlist case goes through music.get with a fake ffprobe that must not run."""
import email.message
import socket
import urllib.error

import pytest

from inframe import music, net

OV = "https://api.openverse.org/v1/audio/"
TRACK = "0f0f0f0f-1111-2222-3333-444444444444"
FILE = "https://cdn.example.org/song.mp3"


class FakeResp:
    """A response with headers and a body that is read in pieces; `size` makes a body of that many zero bytes without holding it."""

    def __init__(self, body=b"", ctype="audio/mpeg", size=None, length=True):
        self.headers = email.message.Message()
        if ctype is not None:
            self.headers["Content-Type"] = ctype
        self.left = size if size is not None else len(body)
        self.body = body
        if length:
            self.headers["Content-Length"] = str(self.left)
        self.reads = 0
        self.closed = False

    def read(self, n=-1):
        self.reads += 1
        n = self.left if n < 0 else min(n, self.left)
        self.left -= n
        if self.body:
            out, self.body = self.body[:n], self.body[n:]
            return out
        return b"\0" * n

    def close(self):
        self.closed = True


def redirect(code, location):
    h = email.message.Message()
    if location is not None:
        h["Location"] = location
    return urllib.error.HTTPError("https://x", code, "moved", h, None)


@pytest.fixture
def web(monkeypatch):
    """`web.hosts` maps host -> addresses (default: a public one), `web.answers` is a list of FakeResp / exceptions served in order."""
    class W:
        hosts: dict = {}
        answers: list = []
        asked: list = []

    w = W()
    w.hosts, w.answers, w.asked = {}, [], []

    def resolve(host, port):
        return w.hosts.get(host, ["93.184.216.34"])

    def opener(url, kind):
        w.asked.append(url)
        a = w.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(net, "_resolve", resolve)
    monkeypatch.setattr(net, "_open", opener)
    return w


def test_http_is_refused_before_any_request(web):
    with pytest.raises(net.NetError, match="non-https"):
        net.fetch("http://api.openverse.org/v1/audio/", kind="api")
    with pytest.raises(net.NetError, match="non-https"):
        net.fetch("file:///etc/passwd", kind="audio")
    assert web.asked == []


def test_api_kind_allows_only_openverse(web):
    web.answers = [FakeResp(b"{}", "application/json")]
    assert net.fetch(OV + "?q=calm", kind="api") == b"{}"
    for url in ("https://example.org/v1/audio/", "https://api.openverse.org.evil.example/x", "https://evil.example/api.openverse.org"):
        with pytest.raises(net.NetError, match="only api.openverse.org"):
            net.fetch(url, kind="api")
    assert web.asked == [OV + "?q=calm"]


def test_user_name_and_unknown_kind_are_refused(web):
    for kind in ("audio", "api"):
        with pytest.raises(net.NetError, match="user name"):
            net.fetch("https://api.openverse.org@evil.example/x", kind=kind)
        with pytest.raises(net.NetError, match="user name"):
            net.fetch("https://user:pw@api.openverse.org/x", kind=kind)
    with pytest.raises(ValueError):
        net.fetch(FILE, kind="other")


@pytest.mark.parametrize("addr", ["127.0.0.1", "10.0.0.5", "169.254.1.1", "::1", "192.168.1.9", "100.64.0.1", "::ffff:127.0.0.1", "fe80::1%en0",
                                  "64:ff9b::a00:1", "64:ff9b::7f00:1", "64:ff9b::c0a8:109", "64:ff9b:1::a00:1", "::a00:1", "::7f00:1",
                                  "2002:a00:1::1", "2002:7f00:1::1", "::ffff:0:a00:1", "::ffff:0:7f00:1", "::ffff:0:808:808", "::ffff:0:10.0.0.1", "2001:0:4136:e378:8000:63bf:3fff:fdd2"])
def test_audio_host_resolving_to_a_non_global_address_is_refused(web, addr):
    web.hosts["cdn.example.org"] = [addr]
    with pytest.raises(net.NetError, match="non-public"):
        net.fetch(FILE, kind="audio")
    assert web.asked == []


def test_nat64_to_a_public_address_is_allowed(web):
    assert net._is_global("64:ff9b::808:808") and net._is_global("2606:4700::1111") and net._is_global("2002:808:808::1")


def test_one_bad_address_among_good_ones_refuses(web):
    web.hosts["cdn.example.org"] = ["93.184.216.34", "10.0.0.5"]
    with pytest.raises(net.NetError, match="non-public"):
        net.fetch(FILE, kind="audio")


def test_ip_literal_hosts_are_checked_too(web):
    web.hosts = {"10.0.0.5": ["10.0.0.5"], "[::1]": ["::1"], "::1": ["::1"]}
    for url in ("https://10.0.0.5/a.mp3", "https://[::1]/a.mp3"):
        with pytest.raises(net.NetError, match="non-public"):
            net.fetch(url, kind="audio")


def test_resolution_uses_getaddrinfo(monkeypatch):
    real = socket.getaddrinfo

    def fake(host, port, *a, **kw):
        if host == "rebind.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]
        return real(host, port, *a, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    with pytest.raises(net.NetError, match="non-public"):
        net.check_url("https://rebind.example/a.mp3", "audio")

    def gone(host, port, *a, **kw):
        raise socket.gaierror(8, "nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", gone)
    with pytest.raises(net.NetError, match="could not resolve"):
        net.check_url("https://nowhere.example/a.mp3", "audio")


def test_a_redirect_on_the_same_host_is_followed(web):
    web.answers = [redirect(302, "/other/song.mp3"), redirect(301, FILE.replace("song", "moved")), FakeResp(b"ID3....")]
    assert net.fetch(FILE, kind="audio") == b"ID3...."
    assert web.asked == [FILE, "https://cdn.example.org/other/song.mp3", "https://cdn.example.org/moved.mp3"]


@pytest.mark.parametrize("target", ["https://evil.example/song.mp3", "http://cdn.example.org/song.mp3", "https://cdn.example.org:8443/x",
                                    "https://127.0.0.1/song.mp3", "//evil.example/song.mp3"])
def test_a_redirect_to_another_host_or_http_is_refused(web, target):
    web.answers = [redirect(302, target)]
    with pytest.raises(net.NetError, match="redirect"):
        net.fetch(FILE, kind="audio")
    assert web.asked == [FILE]


def test_a_redirect_is_checked_for_its_addresses_again(web, monkeypatch):
    # the same name, but now it points inside the network (a DNS answer that changed between the hops)
    answers = iter([["93.184.216.34"], ["10.0.0.5"]])
    monkeypatch.setattr(net, "_resolve", lambda host, port: next(answers))
    web.answers = [redirect(302, "/b.mp3")]
    with pytest.raises(net.NetError, match="non-public"):
        net.fetch(FILE, kind="audio")
    assert web.asked == [FILE]


def test_redirect_loops_and_missing_locations_stop(web):
    web.answers = [redirect(302, "/a.mp3")] * 20
    with pytest.raises(net.NetError, match="too many redirects"):
        net.fetch(FILE, kind="audio")
    assert len(web.asked) == net.MAX_REDIRECTS + 1
    web.answers, web.asked = [redirect(302, None)], []
    with pytest.raises(net.NetError, match="without a location"):
        net.fetch(FILE, kind="audio")


def test_the_cap_is_50_mb(web):
    cap = 50 * 2**20
    assert net.MAX_AUDIO_BYTES == cap
    web.answers = [FakeResp(size=cap + 1, length=False)]  # no Content-Length: stopped while reading
    with pytest.raises(net.NetError, match="larger than 50 MB"):
        net.fetch(FILE, kind="audio")
    r = FakeResp(size=cap + 1)  # a Content-Length that is too large is refused before reading anything
    web.answers = [r]
    with pytest.raises(net.NetError, match="larger than 50 MB"):
        net.fetch(FILE, kind="audio")
    assert r.reads == 0 and r.closed
    web.answers = [FakeResp(size=cap, length=False)]  # exactly the cap is fine
    assert len(net.fetch(FILE, kind="audio")) == cap


def test_a_given_max_bytes_wins(web):
    web.answers = [FakeResp(b"x" * 11, "application/json")]
    with pytest.raises(net.NetError, match="larger than"):
        net.fetch(OV, kind="api", max_bytes=10)


@pytest.mark.parametrize("ctype", ["text/html", "text/html; charset=utf-8", "application/json", "application/vnd.apple.mpegurl", "video/mp4", None])
def test_a_non_audio_content_type_is_refused(web, ctype):
    web.answers = [FakeResp(b"#EXTM3U\n", ctype)]
    with pytest.raises(net.NetError, match="not audio"):
        net.fetch(FILE, kind="audio")


@pytest.mark.parametrize("ctype", ["audio/mpeg", "audio/x-wav", "Audio/OGG; codecs=opus", "application/octet-stream"])
def test_audio_and_octet_stream_are_accepted(web, ctype):
    web.answers = [FakeResp(b"ID3\x03", ctype)]
    assert net.fetch(FILE, kind="audio") == b"ID3\x03"


def test_http_errors_carry_their_status(web):
    web.answers = [urllib.error.HTTPError(OV, 429, "Too Many", email.message.Message(), None)]
    with pytest.raises(net.NetError) as e:
        net.fetch(OV, kind="api")
    assert e.value.status == 429
    web.answers = [urllib.error.URLError("connection refused")]
    with pytest.raises(net.NetError, match="could not reach api.openverse.org") as e:
        net.fetch(OV, kind="api")
    assert e.value.status is None


# ---------------------------------------------------------------- music through net.fetch
def _ov_record(**kw):
    import json
    r = dict(id=TRACK, title="Song", creator="Artist", creator_url=None, license="cc0", license_version="1.0", license_url=None, url=FILE,
             foreign_landing_url=None, duration=95000, attribution=None, source="jamendo", filetype="mp3")
    r.update(kw)
    return json.dumps(r).encode()


def test_music_get_goes_through_net_fetch_with_the_right_kinds(web, monkeypatch, tmp_path):
    kinds = []
    real = net.fetch
    monkeypatch.setattr(music.net, "fetch", lambda url, *, kind, max_bytes=None: (kinds.append((url, kind, max_bytes)), real(url, kind=kind, max_bytes=max_bytes))[1])
    monkeypatch.setattr(music, "_check_audio", lambda p: None)
    web.answers = [FakeResp(_ov_record(), "application/json"), FakeResp(b"ID3\x03\0\0\0\0\0\0rest", "audio/mpeg")]
    audio = music.get(TRACK, tmp_path / "m")
    assert audio.read_bytes().startswith(b"ID3") and audio.suffix == ".mp3"
    assert kinds == [(OV + TRACK + "/", "api", None), (FILE, "audio", music.MAX_BYTES)]
    assert music.MAX_BYTES == 50 * 2**20


def test_an_hls_playlist_is_refused_before_ffprobe_looks_at_it(web, monkeypatch, tmp_path):
    import subprocess

    def boom(*a, **kw):
        raise AssertionError("ffprobe must not run")

    monkeypatch.setattr(subprocess, "run", boom)
    hls = b"#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:10,\nfile:///etc/passwd\n"
    for ctype in ("audio/mpeg", "application/octet-stream"):  # a host can claim any audio type for a text playlist
        web.answers = [FakeResp(_ov_record(), "application/json"), FakeResp(hls, ctype)]
        with pytest.raises(music.MusicError, match="not a playable audio file"):
            music.get(TRACK, tmp_path / "m")
    assert not list((tmp_path / "m").glob("*"))  # no .part left behind


def test_music_errors_keep_their_messages(web):
    web.answers = [urllib.error.HTTPError(OV, 429, "Too Many", email.message.Message(), None)]
    with pytest.raises(music.MusicError, match="rate-limiting"):
        music._get(OV + "?q=x")
    web.answers = [urllib.error.HTTPError(OV, 503, "Down", email.message.Message(), None)]
    with pytest.raises(music.MusicError, match="Openverse answered HTTP 503"):
        music._get(OV + "?q=x")
    web.answers = [FakeResp(b"<html>", "text/html")]
    with pytest.raises(music.MusicError, match="not JSON"):
        music._get(OV + "?q=x")
    web.answers = [FakeResp(b"[]", "text/html")]  # the API's type is not checked, its JSON is
    assert music._get(OV) == []
    web.hosts["cdn.example.org"] = ["10.0.0.5"]
    with pytest.raises(music.MusicError, match="non-public"):
        music._get(FILE, binary=True, limit=music.MAX_BYTES)


def test_ffprobe_gets_a_protocol_whitelist(monkeypatch, tmp_path):
    import subprocess
    seen = []

    class R:
        stdout = "12.0\n"

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: (seen.append(cmd), R())[1])
    f = tmp_path / "a.mp3"
    f.write_bytes(b"ID3")
    music._check_audio(f)
    cmd = seen[0]
    i = cmd.index("-protocol_whitelist")
    assert cmd[i + 1] == "file,pipe" and i < len(cmd) - 1 and cmd[-1] == str(f.resolve())
