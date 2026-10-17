"""A whole run in Chrome on synthetic HEVC (ffmpeg testsrc2, no person in it): the real BrowserEngine, Worker, WebCodecs and WebGPU. The run
must end with the expected number of samples and no clips, cancel must stop it, and every request the browser made must be a GET to the page's
own origin (docs/T9B_DESIGN.md section 8)."""
from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from conftest import open_page


def run(page, **opts):
    return page.evaluate("(o) => window.__t.run(o)", opts)


@pytest.fixture
def loaded(chrome, site, models_cached, hevc20):
    page = open_page(chrome, site)
    page.set_input_files("#file", str(hevc20))
    yield page
    page.close()


def test_balanced_any_person_run_ends_with_the_expected_samples(loaded, chrome, capsys):
    chrome.problems.clear()
    r = run(loaded)
    with capsys.disabled():
        print(f"\n[run] {r['state']} in {r['ms'] / 1000:.1f} s: {r.get('nSamples')} samples of {r['duration']:.1f} s ({r['codec']}), {r.get('nClips')} clips, max score {r.get('maxScore')}, "
              f"timing {r.get('timing')}, detector {r.get('settings', {}).get('detector')}")
    assert r["error"] is None and r["state"] == "done", r
    assert abs(r["nSamples"] - 5 * 20) <= 1, r["nSamples"]
    assert r["nClips"] == 0 and r["maxScore"] < 0.5
    assert r["firstT"] == 0 and 19.5 <= r["lastT"] <= 20
    assert r["codec"] == "hevc" and r["settings"]["detector"] == "onnxruntime-web webgpu"
    assert r["runs"] == 1 and r["stillRunning"] is False
    assert chrome.problems == [], chrome.problems


def test_cancel_stops_the_run_mid_way_and_keeps_nothing(loaded, chrome):
    chrome.problems.clear()
    r = run(loaded, cancelAtFraction=0.15)
    assert r["state"] == "cancelled" and r["error"] is None, r
    assert r["cancelledAt"] is not None and r["cancelledAt"] < 0.9
    assert r["stillRunning"] is False
    assert r["runs"] == 0 or r["runs"] == 1  # a run of an earlier test with the same id may be there; a cancelled one never replaces it
    assert "nSamples" not in r  # no run document came out of the cancelled job
    assert chrome.problems == [], chrome.problems
    again = run(loaded)  # the Worker is still good after a cancel
    assert again["state"] == "done" and abs(again["nSamples"] - 100) <= 1, again


def test_boxes_for_one_picture(loaded):
    b = loaded.evaluate("window.__t.boxes(7.5)")
    assert abs(b["t"] - 7.5) < 0.05 and b["boxes"] == []  # testsrc2 has no person


def test_every_request_is_a_get_to_the_page_origin(chrome, site, models_cached):
    assert chrome.requests, "no requests were seen"
    origin = site.base
    seen = [(m, u) for m, u in chrome.requests if not u.startswith(("blob:", "data:"))]
    bad = [(m, u) for m, u in seen if m != "GET" or not u.startswith(origin + "/")]
    assert bad == [], bad
    paths = {urlsplit(u).path for _, u in seen}
    assert "/models/manifest.json" in paths and any(p.startswith("/models/rfdetr") for p in paths)
    assert not any(urlsplit(u).hostname not in ("127.0.0.1",) for _, u in seen)
    # and what the server answered, by its own count
    assert site.log and all(m == "GET" for m, _ in site.log), [x for x in site.log if x[0] != "GET"]
    assert all(urlsplit(p).path.startswith(("/", "/models/", "/assets/", "/api/selftest/data/")) for _, p in site.log)


def test_open_gop_leading_pictures_all_come_back(chrome, site, models_cached, open_gop6, capsys):
    """Pictures shown before their keyframe are dropped by a decoder that starts at that keyframe (open-GOP leading pictures): the plan must start their run a keyframe earlier."""
    _, pics = open_gop6
    # the leading pictures: shown before the keyframe their group starts with (decode order)
    key_pts, leading = None, 0
    for pts, key in pics:
        if key:
            key_pts = pts
        elif key_pts is not None and pts < key_pts:
            leading += 1
    assert leading >= 10, f"the synthetic stream has only {leading} leading pictures"
    # decode.iter_samples at 30 fps: the first picture at or after each k / 30 slot
    planned, k = 0, 0
    for t in sorted(p for p, _ in pics):
        if t + 1e-4 < k / 30:
            continue
        k = max(k + 1, int(t / (1 / 30) + 1e-4) + 1)
        planned += 1
    assert planned == len(pics) == 180
    page = open_page(chrome, site)
    try:
        page.set_input_files("#file", str(open_gop6[0]))
        chrome.problems.clear()
        r = run(page, advanced={"fps": 30})
        with capsys.disabled():
            print(f"\n[open gop] {r['state']} in {r['ms'] / 1000:.1f} s: {r.get('nSamples')} samples planned {planned}, {leading} leading pictures")
        assert r["error"] is None and r["state"] == "done", r
        assert r["nSamples"] == planned, (r["nSamples"], planned)
        assert chrome.problems == [], chrome.problems
    finally:
        page.close()


def test_open_gop_boxes_and_preview_on_a_leading_picture(chrome, site, models_cached, open_gop6):
    """Picking a person (boxes) and the frame preview decode one picture: a leading one of an open GOP needs its run to start a keyframe earlier too."""
    _, pics = open_gop6
    key_pts, lead_times, plain_times = None, [], []
    for pts, key in pics:
        if key:
            key_pts = pts
            if pts >= 1.0:
                plain_times.append(pts)
        elif key_pts is not None and pts < key_pts and key_pts >= 1.0:
            lead_times.append(pts)
    assert len(lead_times) >= 4, lead_times
    page = open_page(chrome, site)
    try:
        page.set_input_files("#file", str(open_gop6[0]))
        for t in plain_times[:2] + lead_times:
            b = page.evaluate(f"window.__t.boxes({t})")
            assert abs(b["t"] - t) < 0.02 and b["boxes"] == [], (t, b)  # testsrc2 has no person; the picture came back
    finally:
        page.close()
