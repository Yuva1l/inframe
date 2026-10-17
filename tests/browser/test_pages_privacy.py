"""Privacy check of the Cloudflare Pages build (docs/T9C_DESIGN.md section 3), on two local origins, in Chrome.

The built pages-dist is served by one local host (its generated _headers applied), the model files by two more local hosts that behave like Hugging Face: a
first-hop host (GET /resolve/<commit>/<name>) that answers 302 to a signed /cdn/ path on a third origin (CORS, Range, HEAD). The page is built with the
shape of web/pages/model-host.json: a path-restricted connect-src entry for the first hop plus the bare redirect origin, so Chrome enforces that shape and
meets a cross-origin redirect (Origin: null on the second hop). The real UI is driven: Download models, open a clip that carries a canary (a random token in
its metadata and in a free box, plus a random file name) and shows a person, run "Anyone", run a tap target on the person's box (found from the detector's
numbers, so the DINOv2 crop and embedding path really runs), export as a download. Everything the browser asks for is recorded twice, by Chrome (every page,
worker and redirect) and by the three hosts, and tests/browser/privacy.py must find nothing wrong: only the three origins, GET (HEAD on the model side) with
no body, only the headers a browser adds, only manifest files on the model side, the signed URL exactly as issued, no secret anywhere, no WebSocket, no
service worker, no sendBeacon, and a second run that never asks the model side again. What this cannot show: the real hosts' names, which the part-3
preflight (docs/T9C_DEPLOY.md) checks against huggingface.co.

The pictures are an openly licensed Pexels shot (docs/TESTDATA.md) and no picture is looked at: the test reads text, counts and logs.
"""
from __future__ import annotations

import json
import re
import zipfile
from urllib.parse import urlsplit

import pytest

from privacy import Seen, check
from static_hosts import Entry

INIT = """
(() => {
  window.__spy = { beacons: [] };
  const sb = navigator.sendBeacon && navigator.sendBeacon.bind(navigator);
  navigator.sendBeacon = (...a) => { window.__spy.beacons.push(String(a[0])); return sb ? sb(...a) : false; };
  // the real Chrome would open a native folder dialog: without the pickers the page offers the zip as a download
  delete window.showDirectoryPicker;
  delete window.showSaveFilePicker;
})();
"""
SLOW = 240_000  # ms: the model download and a run on this Mac


class Recorder:
    """Everything one Chrome context does: its requests (headers and bodies read after the fact), WebSockets, service workers and page errors."""

    def __init__(self, ctx):
        self.requests: list = []
        self._seen: list[Seen] = []
        self._done = 0
        self.sockets: list[str] = []
        self.workers: list[str] = []
        self.problems: list[str] = []
        ctx.on("request", lambda r: self.requests.append(r))
        ctx.on("serviceworker", lambda w: self.workers.append(w.url))

    def watch(self, page):
        page.on("websocket", lambda ws: self.sockets.append(ws.url))
        page.on("pageerror", lambda e: self.problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: self.problems.append(f"console.error: {m.text[:300]}") if m.type == "error" and not re.search(r"\[[VIWEF]:onnxruntime", m.text) else None)

    def flush(self) -> None:
        """Read the headers and bodies of the requests made so far (Chrome can only be asked while the page is open)."""
        for r in self.requests[self._done :]:
            try:
                h = r.all_headers()
            except Exception:  # a request that was cancelled before it went out has no raw headers
                h = r.headers
            self._seen.append(Seen(r.method, r.url, {k.lower(): v for k, v in h.items()}, r.post_data_buffer, "browser"))
        self._done = len(self.requests)

    def seen(self) -> list[Seen]:
        self.flush()
        return list(self._seen)


def host_seen(site) -> list[Seen]:
    """What the three hosts logged, as the check reads it."""
    out = []
    for host, origin in ((site.page, site.page.origin), (site.model, site.model.origin), (site.cdn, site.cdn.origin)):
        for e in host.log.snapshot():
            out.append(Seen(e.method, origin + e.path + (f"?{e.query}" if e.query else ""), e.headers, e.body or None, f"{host.kind}-host"))
    return out


def issued_locations(site) -> set[str]:
    """The signed URLs the first-hop host really put in a Location header."""
    return {e.location for e in site.model.log.snapshot() if e.location}


def manifest_sizes(site) -> dict[str, int]:
    files = json.loads((site.dist / "models" / "manifest.json").read_text())["files"]
    return {n: int(f["bytes"]) for n, f in files.items()}


def run_check(site, seen, canary, **kw):
    return check(seen, page_origin=site.page.origin, model_origin=site.model.origin, redirect_origin=site.cdn.origin, model_paths=site.model_paths, page_files=site.page_files,
                 secrets=canary.secrets, issued=issued_locations(site), sizes=manifest_sizes(site), **kw)


# ---------------------------------------------------------------- driving the real UI
def open_app(ctx, rec, site):
    page = ctx.new_page()
    rec.watch(page)
    resp = page.goto(site.page.origin + "/")
    page.main_headers = resp.all_headers()  # the page document's own response headers (the CSP Chrome enforces)
    page.get_by_role("heading", name="Open a video").wait_for(timeout=30_000)
    return page


def open_clip(page, canary):
    page.locator("[data-testid=file-input]").set_input_files(str(canary.path))
    page.locator("video.video").wait_for(timeout=60_000)
    page.locator("header.top").get_by_text(canary.name).wait_for(timeout=60_000)  # the editor shows the file's name


def run_and_wait(page):
    """Click Run, see the job start, and wait until it ends (a "Done" line or an error). Returns the text of the page then."""
    page.evaluate(
        """() => {
          window.__sawJob = false;
          if (window.__jobWatch) window.__jobWatch.disconnect();
          const look = () => { if (document.querySelector('[data-testid=job-progress]')) window.__sawJob = true; };
          window.__jobWatch = new MutationObserver(look);
          window.__jobWatch.observe(document.body, { childList: true, subtree: true });
          look();
        }"""
    )
    page.locator("button.runbtn").click()
    # a video whose run was edited (the first visit added a clip) asks before it runs again
    page.wait_for_function("() => window.__sawJob || [...document.querySelectorAll('button')].some((b) => b.textContent === 'Run again')", timeout=60_000)
    again = page.get_by_role("button", name="Run again", exact=True)
    if again.count():
        again.click()
    page.wait_for_function("() => window.__sawJob", timeout=60_000)
    page.wait_for_function("() => !document.querySelector('[data-testid=job-progress]') && document.querySelector('.okline, .errpanel')", timeout=SLOW)
    return page.locator("body").inner_text()


def collect(page, rec):
    """The things only the page knows: service worker registrations and the sendBeacon spy."""
    regs = page.evaluate("navigator.serviceWorker ? navigator.serviceWorker.getRegistrations().then(r => r.map(x => x.scope)) : []")
    beacons = page.evaluate("window.__spy ? window.__spy.beacons : null")
    assert beacons is not None, "the init script did not run in this page"
    return regs, beacons


@pytest.fixture(scope="module")
def session(chrome_browser, pages_site, canary_clip, tmp_path_factory):
    """The whole first visit and a second one in the same browser context (a fresh profile, so an empty Cache Storage), with the evidence kept."""
    ctx = chrome_browser.new_context(accept_downloads=True)
    ctx.add_init_script(INIT)
    rec = Recorder(ctx)
    log: dict = {"zips": []}
    page = open_app(ctx, rec, pages_site)

    # 1. download the models (the only large transfer)
    page.locator("[data-testid=models-privacy]").wait_for(timeout=30_000)
    log["privacy_line"] = page.locator("[data-testid=models-privacy]").inner_text()
    page.get_by_role("button", name=re.compile(r"Download models")).click()
    page.get_by_role("button", name="Remove downloaded models").wait_for(timeout=SLOW)
    log["after_download"] = pages_site.model_requests()

    # 2. open the canary clip and run "Anyone"
    open_clip(page, canary_clip)
    log["any_text"] = run_and_wait(page)
    log["after_any"] = pages_site.model_requests()

    # 3. a tap target on the person: the page keeps taps in IndexedDB, so the tap is written the way a click on a numbered box writes it, with the box the
    #    detector's numbers gave (conftest._person_box); the run then crops that person and DINOv2 embeds them. Then the Tap tab and Run
    page.evaluate(
        """async ([time, box]) => {
          const db = await new Promise((ok, no) => { const r = indexedDB.open('inframe'); r.onsuccess = () => ok(r.result); r.onerror = () => no(r.error); });
          const media = await new Promise((ok, no) => { const r = db.transaction('media').objectStore('media').getAll(); r.onsuccess = () => ok(r.result); r.onerror = () => no(r.error); });
          const id = media[0].id;
          const pick = { class: 'person', target_key: '', taps: [{ time, box, track_id: null, index: 1 }], zone: null };
          await new Promise((ok, no) => { const tx = db.transaction('pick', 'readwrite'); tx.objectStore('pick').put(pick, id + '|person'); tx.oncomplete = ok; tx.onerror = () => no(tx.error); });
          db.close();
        }""",
        [canary_clip.tap_time, list(canary_clip.tap_box)],
    )
    page.get_by_role("tab", name="Tap").click()
    page.locator("[data-testid=tap-count]").get_by_text("1 tap").wait_for(timeout=10_000)
    log["tap_text"] = run_and_wait(page)
    log["after_tap"] = pages_site.model_requests()

    # 4. Export -> a download of the clips the tap run found
    page.get_by_role("button", name="Export", exact=True).first.click()
    with page.expect_download(timeout=SLOW) as dl:
        page.get_by_role("dialog").get_by_role("button", name="Export", exact=True).click()
    d = dl.value
    z = tmp_path_factory.mktemp("zip") / d.suggested_filename
    d.save_as(z)
    log["zip"] = z
    page.locator("[data-testid=export-outputs]").wait_for(timeout=60_000)
    log["export_text"] = page.locator("body").inner_text()
    log["after_export"] = pages_site.model_requests()
    log["main_headers"] = page.main_headers
    log["regs1"], log["beacons1"] = collect(page, rec)
    log["page1_problems"] = list(rec.problems)
    rec.flush()
    page.close()

    # 5. the second run: a new page in the same context (the models are in Cache Storage), the same clip, "Anyone" again
    before = pages_site.model_requests()
    page2 = open_app(ctx, rec, pages_site)
    log["second_download_button"] = page2.get_by_role("button", name=re.compile(r"Download models")).count()
    open_clip(page2, canary_clip)
    log["second_text"] = run_and_wait(page2)
    log["second_model_requests"] = pages_site.model_requests() - before
    log["regs2"], log["beacons2"] = collect(page2, rec)
    rec.flush()
    page2.close()

    log["seen"] = rec.seen()
    log["rec"] = rec
    log["workers_in_ctx"] = list(ctx.service_workers)
    yield log
    ctx.close()


# ---------------------------------------------------------------- the checks
def test_the_run_and_the_export_really_happened(session, canary_clip):
    """If the flow did not run, the privacy results below would be empty ones."""
    assert "Your video never leaves this browser" in session["privacy_line"] and "from 127.0.0.1" in session["privacy_line"], session["privacy_line"]
    assert canary_clip.name in session["any_text"]
    # the clip shows a person: "Anyone" finds clips in it (on both visits), and the tap run resolves the tap on a detection that was embedded (DINOv2 ran on the
    # crop), so it ends in clips and not in one of the TapError messages of web/src/browser/targets.ts
    for key in ("any_text", "tap_text", "second_text"):
        assert re.search(r"Done: [1-9]\d* clips?\.", session[key]), (key, session[key][-300:])
    assert not re.search(r"no person detected|too small to recognise|Tap the person", session["tap_text"]), session["tap_text"][-300:]
    z = zipfile.ZipFile(session["zip"])
    assert z.testzip() is None
    names = z.namelist()
    assert any(n.lower().endswith(".mp4") for n in names), names
    assert session["second_download_button"] == 0
    assert session["page1_problems"] == [], session["page1_problems"]


def test_only_the_three_origins_are_contacted(session, pages_site):
    origins = {f"{urlsplit(s.url).scheme}://{urlsplit(s.url).netloc}" for s in session["seen"] if urlsplit(s.url).scheme not in ("blob", "data", "about")}
    assert origins == {pages_site.page.origin, pages_site.model.origin, pages_site.cdn.origin}
    host_origins = {s.url.split("/", 3)[0] + "//" + s.url.split("/", 3)[2] for s in host_seen(pages_site)}
    assert host_origins == origins  # the hosts' own logs agree with Chrome's list


def test_the_browsers_list_passes_the_privacy_check(session, pages_site, canary_clip):
    assert run_check(pages_site, session["seen"], canary_clip) == []


def test_the_hosts_logs_pass_the_privacy_check(session, pages_site, canary_clip):
    """What the servers received (their own count of methods, paths, headers and bodies), including anything Chrome did not report."""
    hosts = host_seen(pages_site)
    assert hosts, "the hosts logged nothing"
    assert run_check(pages_site, hosts, canary_clip) == []
    bad = [e for h in (pages_site.page, pages_site.model, pages_site.cdn) for e in h.log.snapshot() if e.method not in ("GET", "HEAD") or e.body]
    assert bad == [], bad


def test_chromes_list_and_the_hosts_logs_agree_on_what_reached_them(session, pages_site):
    """Every request Chrome made to a host is in that host's log (a cached answer is the only difference, so the hosts may log fewer)."""
    chrome_paths = {(s.method, urlsplit(s.url).path) for s in session["seen"] if s.url.startswith((pages_site.page.origin, pages_site.model.origin, pages_site.cdn.origin))}
    logged = {(s.method, urlsplit(s.url).path) for s in host_seen(pages_site)}
    assert logged <= chrome_paths | {("GET", "/favicon.ico")}, logged - chrome_paths


def test_the_model_host_serves_only_manifest_files_and_the_page_its_own_manifest(session, pages_site):
    first, signed = pages_site.model.log.snapshot(), pages_site.cdn.log.snapshot()
    model = first + signed
    assert {e.path.split("/")[1] for e in first} == {"resolve"} and {e.path.split("/")[1] for e in signed} == {"cdn"}  # each hop on its own origin
    paths = {e.path for e in model}
    assert paths <= pages_site.model_paths, paths - pages_site.model_paths
    names = {p.rsplit("/", 1)[1] for p in paths}
    assert names == set(pages_site.names)  # all three files, no manifest.json
    assert not any(e.path.endswith("manifest.json") for e in model)
    assert any(e.path == "/models/manifest.json" for e in pages_site.page.log.snapshot())  # the page read the pinned manifest from its own site
    assert {e.status for e in first} == {302} and 200 in {e.status for e in signed}  # the redirect to the other origin was followed
    # the redirect crossed origins: Chrome sends Origin: null on the second hop and the cdn host answered with CORS all the same
    assert any(e.headers.get("origin") == "null" for e in signed), [e.headers.get("origin") for e in signed]
    assert {e.location for e in first} == {pages_site.cdn.origin + p for p in {e.path + "?" + e.query for e in signed}}  # only the issued URLs were followed


def test_every_page_request_is_for_a_file_that_exists_and_has_no_query(session, pages_site):
    page = pages_site.page.log.snapshot()
    assert page and all(e.query == "" and e.status == 200 for e in page), [(e.path, e.query, e.status) for e in page if e.query or e.status != 200]
    assert {e.path for e in page} <= pages_site.page_files


def test_no_secret_is_in_any_request(session, pages_site, canary_clip):
    """The token, the file name and three 64-byte runs of the clip, plain, percent-encoded, hex or base64, in every URL, header name, header value and body."""
    seen = session["seen"] + host_seen(pages_site)
    assert len(seen) > 20
    problems = [p for p in run_check(pages_site, seen, canary_clip) if "secret" in p]
    assert problems == []
    assert all(len(s.url) <= 512 for s in seen)
    # the token reached the page (it is in the file's metadata) and the name is what the editor shows: the test would catch a leak of either
    assert canary_clip.name in session["any_text"] and canary_clip.token not in "".join(s.url for s in seen)


def test_no_websocket_no_service_worker_no_beacon(session):
    rec = session["rec"]
    assert rec.sockets == []
    assert rec.workers == [] and session["workers_in_ctx"] == []
    assert session["regs1"] == [] and session["regs2"] == []
    assert session["beacons1"] == [] and session["beacons2"] == []


def test_the_download_is_the_only_big_transfer_and_stays_in_the_browser(session, pages_site):
    """The models were requested during the download step and never again after it (the run, the tap run and the export used the cache)."""
    assert session["after_download"] > 0
    assert session["after_export"] == session["after_download"] == session["after_any"] == session["after_tap"]


def test_a_second_run_makes_no_model_host_request(session):
    assert session["second_model_requests"] == 0
    assert re.search(r"Done: [1-9]\d* clips?\.", session["second_text"])  # and the run did happen


def test_the_csp_header_is_enforced_as_generated(session, pages_site):
    """The page host sent the _headers CSP with the shape of the shipped one (a pinned path on the first hop, the bare redirect origin), and Chrome raised
    no CSP violation on the redirect it followed."""
    hdr = session["main_headers"]
    csp = hdr["content-security-policy"]
    assert f"connect-src 'self' {pages_site.model.origin}/resolve/{pages_site.commit}/ {pages_site.cdn.origin};" in csp
    assert hdr["referrer-policy"] == "no-referrer" and hdr["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert not any("Content Security Policy" in p or "Refused to" in p for p in session["rec"].problems), session["rec"].problems


def test_summary(session, pages_site, capsys):
    """One line of numbers for the record."""
    page, model, cdn = pages_site.page.log.snapshot(), pages_site.model.log.snapshot(), pages_site.cdn.log.snapshot()
    with capsys.disabled():
        print(f"\n[pages privacy] chrome saw {len(session['seen'])} requests ({sum(1 for s in session['seen'] if urlsplit(s.url).scheme in ('blob', 'data'))} blob/data); "
              f"page host {len(page)} ({sorted({e.method for e in page})}), model host {len(model)} ({sorted({e.method for e in model})}, statuses {sorted({e.status for e in model})}), cdn host {len(cdn)} ({sorted({e.method for e in cdn})}, statuses {sorted({e.status for e in cdn})}); "
              f"second run: {session['second_model_requests']} model requests")
    assert len(page) > 3 and len(model) == 3 and len(cdn) == 3  # three files, one request to each hop
