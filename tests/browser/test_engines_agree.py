"""Both engines on one composite whose truth is known (docs/T9B_DESIGN.md section 8). The composite is built with tools/splice.py from openly licensed
clips only (Pexels people and Wikimedia Commons dogs, docs/TESTDATA.md): four stretches with a person, three gaps without one. The Mac engine (`inframe cut`,
CoreML RF-DETR small) and the browser engine (Chrome, WebCodecs, onnxruntime-web WebGPU) each run the balanced any-person preset. Each engine's presence
intervals (the unpadded stretches, as tools/regress.py compares them) must score at least 0.9 time-weighted precision and recall against the truth
(splice.score_intervals), and the two engines' interval boundaries must agree within 0.5 s. Nothing is looked at: only interval times are compared."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT, open_page

MIN_SCORE = 0.9
BOUNDARY_TOL = 0.5  # seconds


@pytest.fixture(scope="module")
def mac_run(gaps_composite, models_dir, tmp_path_factory):
    """`inframe cut` on the composite in a subprocess (the Mac engine); returns its segments.json."""
    mp4, _ = gaps_composite
    out = tmp_path_factory.mktemp("mac")
    env = dict(os.environ, INFRAME_MODELS=str(models_dir.parent))
    r = subprocess.run([sys.executable, "-m", "inframe.cli", "cut", str(mp4), "--out", str(out), "--offline"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    return json.loads((out / "segments.json").read_text())


@pytest.fixture(scope="module")
def browser_run(chrome, site, models_cached, gaps_composite):
    mp4, _ = gaps_composite
    chrome.problems.clear()
    page = open_page(chrome, site)
    try:
        page.set_input_files("#file", str(mp4))
        res = page.evaluate("(o) => window.__t.run(o)", {})
    finally:
        page.close()
    assert res["error"] is None and res["state"] == "done", res
    assert res["settings"]["detector"] == "onnxruntime-web webgpu", res["settings"]
    assert chrome.problems == [], chrome.problems
    return res["segments"]


def test_truth_has_gaps(gaps_composite):
    _, truth = gaps_composite
    tgt = truth["target_intervals"]
    assert len(tgt) >= 4 and abs(truth["measured_duration"] - truth["duration"]) < 0.5
    assert sum(e - s for s, e in tgt) < 0.8 * truth["duration"]  # there is something to cut away


@pytest.mark.parametrize("engine", ["mac", "browser"])
def test_each_engine_scores_against_the_truth(engine, gaps_composite, mac_run, browser_run, capsys):
    import splice

    _, truth = gaps_composite
    seg = mac_run if engine == "mac" else browser_run
    sc = splice.score_intervals(truth, seg["presence_intervals"])
    with capsys.disabled():
        print(f"\n[engines] {engine}: precision {sc['precision']:.3f}, recall {sc['recall']:.3f} (kept {sc['kept_s']} s, truth {sc['target_s']} s, correct {sc['correct_s']} s); "
              f"{len(seg['presence_intervals'])} intervals vs {len(truth['target_intervals'])} true, detector {seg.get('settings', {}).get('detector')}")
    assert sc["precision"] >= MIN_SCORE and sc["recall"] >= MIN_SCORE, (engine, sc, seg["presence_intervals"], truth["target_intervals"])


def test_the_engines_agree_on_the_boundaries(mac_run, browser_run, capsys):
    a, b = mac_run["presence_intervals"], browser_run["presence_intervals"]
    assert len(a) == len(b), f"interval counts differ: mac {a} vs browser {b}"
    gaps = [abs(x - y) for (s0, e0), (s1, e1) in zip(a, b) for x, y in ((s0, s1), (e0, e1))]
    worst = max(gaps)
    with capsys.disabled():
        print(f"\n[engines] {len(a)} intervals, worst boundary gap between the engines {worst:.2f} s (limit {BOUNDARY_TOL} s)")
    assert worst <= BOUNDARY_TOL, (worst, a, b)
