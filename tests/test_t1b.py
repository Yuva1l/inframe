"""T1b: tap mode (tap.py), the pick page (pick.py), hard-cut detection and unique track ids across resets."""
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from inframe import models, pick, tap, tracks
from inframe.cli import main as cli_main
from inframe.detect import Detection
from inframe.tap import Tap, TapError, TapTarget

BOX = (0.4, 0.2, 0.6, 0.8)


def frame_of(rgb_upper, rgb_lower, h=360, w=640, box=BOX):
    f = np.full((h, w, 3), 128, np.uint8)
    y0, y1, x0, x1 = int(box[1] * h), int(box[3] * h), int(box[0] * w), int(box[2] * w)
    ym = (y0 + y1) // 2
    f[y0:ym, x0:x1], f[ym:y1, x0:x1] = rgb_upper, rgb_lower
    return f


RED, BLUE, GREEN = (200, 30, 30), (30, 30, 200), (30, 200, 30)


def run(target, shots):
    """shots: list of lists of per-sample candidate lists [(box, track_id, upper, lower)]; every list starts a new shot."""
    t = 0.0
    for si, shot in enumerate(shots):
        if si:
            target.on_cut()
        for cands in shot:
            frame = np.full((360, 640, 3), 128, np.uint8)
            dets = []
            for box, tid, up, lo in cands:
                frame = np.where(frame_of(up, lo, box=box) != 128, frame_of(up, lo, box=box), frame)
                dets.append(Detection(0.9, box, tid))
            target.evaluate(round(t, 1), frame, dets)
            t += 0.2
    return target.finalize()


def test_parse_spec(tmp_path):
    assert tap.parse_spec("12.5,0.4,0.6")[0].point == (0.4, 0.6)
    for bad in ("1,2", "a,b,c", "1,1.5,0.5", "-1,0.5,0.5", str(tmp_path / "missing.json")):
        with pytest.raises(TapError):
            tap.parse_spec(bad)
    f = tmp_path / "target.json"
    f.write_text(json.dumps(dict(time=3.0, box=[0.1, 0.1, 0.3, 0.9], track_id=4)))
    assert tap.parse_spec(str(f))[0].box == (0.1, 0.1, 0.3, 0.9)
    f.write_text(json.dumps(dict(taps=[dict(time=1, box=[0, 0, 1, 1]), dict(time=9, box=[0, 0, 1, 1])])))
    assert [t.t for t in tap.parse_spec(str(f))] == [1.0, 9.0]
    f.write_text("{}")
    with pytest.raises(TapError):
        tap.parse_spec(str(f))
    with pytest.raises(TapError):
        tap.parse_targets(["photos:dir"])


def test_signature_tells_clothes_apart():
    a = tap.crop_signature(frame_of(RED, BLUE), BOX)
    assert tap.similarity(a, tap.crop_signature(frame_of(RED, BLUE), BOX)) > 0.99
    assert tap.similarity(a, tap.crop_signature(frame_of(GREEN, GREEN), BOX)) < 0.3


def test_tap_follows_track_backwards_and_through_a_gap():
    other = (0.05, 0.2, 0.25, 0.8)
    him = [(BOX, 1, RED, BLUE), (other, 2, GREEN, GREEN)]
    shot = [him] * 10 + [[(other, 2, GREEN, GREEN)]] * 3 + [[(BOX, 7, RED, BLUE), (other, 2, GREEN, GREEN)]] * 6
    tg = TapTarget([Tap(1.0, point=(0.5, 0.5))])  # sample 5: inside his box only
    hits = run(tg, [shot])
    assert [h.score > 0 for h in hits] == [True] * 10 + [False] * 3 + [True] * 6  # before the tap, after the gap (new id, same clothes)
    assert all(h.box == BOX for h in hits if h.score)
    assert tg.describe()["tracks_accepted"] == 2


def test_two_people_at_once_are_not_one_person():
    lookalike = (0.05, 0.2, 0.25, 0.8)
    shot = [[(BOX, 1, RED, BLUE), (lookalike, 2, RED, BLUE)]] * 6
    hits = run(TapTarget([Tap(0.4, point=(0.5, 0.5))]), [shot])
    assert all(h.box == BOX for h in hits)


def test_tap_does_not_cross_a_cut():
    a = [[(BOX, 1, RED, BLUE)]] * 5
    b = [[(BOX, 2, GREEN, GREEN), ((0.05, 0.2, 0.25, 0.8), 3, RED, BLUE)]] * 5  # new shot, somebody else where he stood
    tg = TapTarget([Tap(0.8, box=BOX)])
    hits = run(tg, [a, b])
    assert [h.score > 0 for h in hits[:5]] == [True] * 5
    assert tg.shot_of(5) == 1
    assert all(h.box != BOX for h in hits[5:] if h.score)  # the look-alike (3) may be re-acquired, the stranger (2) not


def test_tap_with_nothing_there_fails_early():
    tg = TapTarget([Tap(0.4, point=(0.05, 0.05))])
    with pytest.raises(TapError, match="no person detected within 1 s of the tap at 0.4 s"):
        run(tg, [[[(BOX, 1, RED, BLUE)]] * 12])
    with pytest.raises(TapError):  # the video ends before the window closes
        run(TapTarget([Tap(90.0, point=(0.5, 0.5))]), [[[(BOX, 1, RED, BLUE)]] * 3])


def test_without_a_tracker_it_matches_per_sample():
    other = (0.05, 0.2, 0.25, 0.8)
    shot = [[(BOX, None, RED, BLUE), (other, None, GREEN, GREEN)]] * 8
    hits = run(TapTarget([Tap(0.6, point=(0.5, 0.5))]), [shot])
    assert all(h.box == BOX for h in hits)


def test_frame_distance_and_cut_threshold():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:360, 0:640]
    a = np.stack([xx * 255 // 640, yy * 255 // 360, (xx + yy) * 255 // 1000], -1).astype(np.uint8)  # a smooth picture; a pan moves it a little
    shifted = np.roll(a, 12, axis=1)
    b = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) // 3
    sa = tracks.frame_signature(a)
    assert tracks.frame_distance(sa, tracks.frame_signature(a)) == 0
    assert tracks.frame_distance(sa, tracks.frame_signature(shifted)) < tracks.CUT_THRESHOLD
    assert tracks.frame_distance(sa, tracks.frame_signature(b)) > tracks.CUT_THRESHOLD


def test_track_ids_stay_unique_across_resets():
    pytest.importorskip("trackers")
    frame = np.zeros((360, 640, 3), np.uint8)
    det = [Detection(0.9, (0.4, 0.2, 0.6, 0.8))]
    with tracks.Tracker("ocsort", 5.0) as tr:
        seen = []
        for _ in range(4):
            seen += [d.track_id for d in tr.update(0, det, frame) if d.track_id is not None]
        tr.reset()
        for _ in range(4):
            seen_after = [d.track_id for d in tr.update(0, det, frame) if d.track_id is not None]
    assert seen and seen_after and not set(seen) & set(seen_after)


def test_cli_target_errors(tmp_path, synth, capsys):
    assert cli_main(["cut", synth["plain"], "--target", "tap:1,2", "--out", str(tmp_path)]) == 2
    assert "not a file" in capsys.readouterr().err
    assert cli_main(["cut", synth["plain"], "--mode", "motion", "--target", "tap:1,0.5,0.5", "--out", str(tmp_path)]) == 2
    assert "--mode detect" in capsys.readouterr().err
    assert cli_main(["pick", str(tmp_path / "none.mp4")]) == 2


class FakeDetector:
    backend = "fake"

    def detect(self, frames):
        return [[Detection(0.9, (0.1, 0.1, 0.5, 0.9)), Detection(0.8, (0.6, 0.2, 0.9, 0.9))] for _ in frames]


def start_pick(video, out, monkeypatch, detector=None):
    monkeypatch.setattr(pick, "load_detector", lambda size, units=None: detector or FakeDetector())
    monkeypatch.setattr(pick.tracks, "Tracker", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no tracker here")))
    url, code = [], []
    th = threading.Thread(target=lambda: code.append(pick.serve(video, out, announce=url.append, open_browser=False)))
    th.start()
    for _ in range(100):
        if url:
            break
        threading.Event().wait(0.05)
    base = "http://" + url[0].split("http://")[1].split("/")[0]
    token = url[0].split("?k=")[1].split()[0]
    return base, token, th, code


def raw(base, path, method="GET", headers=None, data=None):
    """(status, body) without raising on 4xx and 5xx."""
    import urllib.error
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_pick_page_writes_target_json(tmp_path, synth, monkeypatch):
    out = tmp_path / "target.json"
    base, k, th, code = start_pick(synth["plain"], out, monkeypatch)
    get = lambda p: urllib.request.urlopen(f"{base}{p}{'&' if '?' in p else '?'}k={k}").read()
    assert b"inframe pick" in get("/")
    assert json.loads(get("/info"))["duration"] > 2
    boxes = json.loads(get("/boxes?t=1.0"))
    assert [b["i"] for b in boxes["boxes"]] == [1, 2]
    assert get(f"/frame.jpg?t={boxes['t']}")[:2] == b"\xff\xd8"
    post = lambda p, d: json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{base}{p}?k={k}", data=json.dumps(d).encode(), method="POST", headers={"Origin": base})).read())
    assert post("/tap", dict(t=boxes["t"], box=boxes["boxes"][0]["box"], index=1))["taps"] == 1
    doc = json.loads(out.read_text())
    assert doc["time"] == boxes["t"] and doc["box"] == boxes["boxes"][0]["box"] and "track_id" in doc and len(doc["taps"]) == 1
    assert tap.parse_spec(str(out))[0].box == tuple(doc["box"])
    post("/done", {})
    th.join(10)
    assert code == [0]


def test_pick_refuses_foreign_hosts_origins_and_missing_token(tmp_path, synth, monkeypatch):
    out = tmp_path / "target.json"
    base, k, th, code = start_pick(synth["plain"], out, monkeypatch)
    port = base.rsplit(":", 1)[1]
    try:
        assert raw(base, f"/frame.jpg?t=1.0&k={k}")[0] == 200
        assert raw(base, f"/frame.jpg?t=1.0&k={k}", headers={"Host": "attacker.example", "Origin": "http://attacker.example"})[0] == 403
        assert raw(base, f"/frame.jpg?t=1.0&k={k}", headers={"Host": f"attacker.example:{port}"})[0] == 403
        assert raw(base, "/frame.jpg?t=1.0")[0] == 403  # no token
        assert raw(base, "/frame.jpg?t=1.0&k=wrong")[0] == 403
        assert raw(base, "/")[0] == 403
        body = json.dumps(dict(t=1.0, box=[0, 0, 1, 1])).encode()
        for h in ({"Content-Type": "text/plain", "Origin": "http://attacker.example"}, {"Content-Type": "text/plain"}):
            assert raw(base, f"/tap?k={k}", "POST", h, body)[0] == 403  # a simple cross-origin POST
            assert raw(base, "/tap", "POST", {**h, "Origin": base}, body)[0] == 403  # right page, no token
        assert raw(base, f"/done?k={k}", "POST", {"Origin": "http://attacker.example"}, b"")[0] == 403
        assert not out.exists() and th.is_alive()
    finally:
        raw(base, f"/done?k={k}", "POST", {"Origin": base}, b"")
        th.join(10)


def test_pick_reports_a_missing_model_instead_of_dropping_the_connection(tmp_path, synth, monkeypatch):
    class Broken:
        def detect(self, frames):
            raise models.ModelsMissing("rf-detr-small missing (offline)")

    base, k, th, code = start_pick(synth["plain"], tmp_path / "target.json", monkeypatch, Broken())
    try:
        status, body = raw(base, f"/boxes?t=1.0&k={k}")
        assert status == 500 and "rf-detr-small missing" in json.loads(body)["error"]
        assert raw(base, f"/info?k={k}")[0] == 200  # and the server goes on
    finally:
        raw(base, f"/done?k={k}", "POST", {"Origin": base}, b"")
        th.join(10)


# ---------------------------------------------------------------- review fixes
def test_bridge_does_not_follow_a_rejected_track():
    near = (0.41, 0.2, 0.61, 0.8)  # IoU about 0.9 with BOX: a stranger standing where he stood
    after = [[(BOX, 1, RED, BLUE)]] * 5 + [[(near, 2, GREEN, GREEN)]] * 10
    hits = run(TapTarget([Tap(0.4, point=(0.5, 0.5))]), [after])
    assert [h.score > 0 for h in hits] == [True] * 5 + [False] * 10
    before = [[(near, 2, GREEN, GREEN)]] * 10 + [[(BOX, 1, RED, BLUE)]] * 5
    hits = run(TapTarget([Tap(2.4, point=(0.5, 0.5))]), [before])
    assert [h.score > 0 for h in hits] == [False] * 10 + [True] * 5


def test_bridge_hole_is_capped():
    """A box that continues his is taken only while the hole since the last chosen sample is within LOST_BUFFER_S."""
    near = (0.41, 0.2, 0.61, 0.8)
    n = int(round(tap.LOST_BUFFER_S / 0.2))  # samples at 5 fps
    gap = [[]] * (n + 3)  # nobody detected for longer than the buffer
    shot = [[(BOX, 1, RED, BLUE)]] * 5 + gap + [[(near, None, GREEN, GREEN)]] * 3
    hits = run(TapTarget([Tap(0.4, point=(0.5, 0.5))]), [shot])
    assert [h.score > 0 for h in hits] == [True] * 5 + [False] * len(gap) + [False] * 3
    shot = [[(near, None, GREEN, GREEN)]] * 3 + gap + [[(BOX, 1, RED, BLUE)]] * 5
    hits = run(TapTarget([Tap(len(shot) * 0.2 - 0.4, point=(0.5, 0.5))]), [shot])
    assert [h.score > 0 for h in hits] == [False] * 3 + [False] * len(gap) + [True] * 5


def test_best_look_alike_wins_not_the_earliest(monkeypatch):
    b1, b5, b7 = (0.1, 0.2, 0.2, 0.8), (0.4, 0.2, 0.5, 0.8), (0.7, 0.2, 0.8, 0.8)
    def unit(lower_first):
        v = np.zeros(tap.SIG_LEN)
        n = tap._GRAYS + tap._HUES * 2
        v[0] = 1.0
        v[n], v[n + 1] = (1.0, 0.0) if lower_first == 1.0 else (0.75, 0.25)
        return v
    sigs = {b1: unit(1.0), b5: unit(0.75), b7: unit(1.0)}  # stranger 5 looks 0.875 like him, track 7 (him again) 1.0
    monkeypatch.setattr(tap, "crop_signature", lambda frame, box: sigs[tuple(box)])
    assert tap.similarity(sigs[b1], sigs[b5]) == pytest.approx(0.875)
    shot = ([[(b1, 1, RED, BLUE)]] * 5 + [[]] * 3 + [[(b5, 5, GREEN, GREEN)]] * 4
            + [[(b5, 5, GREEN, GREEN), (b7, 7, RED, BLUE)]] * 10)
    tg = TapTarget([Tap(0.4, point=(0.15, 0.5))])
    hits = run(tg, [shot])
    assert [h.box for h in hits[:5]] == [b1] * 5
    assert [h.box for h in hits[12:]] == [b7] * 10 and all(h.box is None for h in hits[5:12])
    assert [a["id"] for a in tg.describe()["accepted_tracks"]] == [1, 7]
