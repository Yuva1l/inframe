"""Regression tests for the review findings: motion thresholds and background, cut failures, stream offset, reused folders."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from inframe import cli, pipeline
from inframe.cut import cut_clip
from inframe.decode import iter_samples
from inframe.motion import auto_thresholds
from inframe.pipeline import Settings, run_cut
from inframe.probe import probe
from inframe.segments import build_segments


# 1. auto threshold when the subject is in frame most of the time
def test_auto_thresholds_when_subject_is_in_frame_most_of_the_time():
    rng = np.random.default_rng(1)
    t = np.arange(600) * 0.2
    s = np.abs(rng.normal(0, 3e-4, 600))
    busy = (t >= 10) & (t < 100)  # 75% of the samples
    s[busy] = np.clip(rng.normal(0.065, 0.012, busy.sum()), 0.02, None)
    enter, exit_, st = auto_thresholds(s)
    assert (s[~busy] >= enter).mean() < 0.01 and exit_ < enter < s[busy].min()
    presence, _ = build_segments(t.tolist(), s.tolist(), 120.0, enter, exit_, 1.5, 1.0, 1.0)
    assert len(presence) == 1 and presence[0][0] == pytest.approx(10, abs=0.3) and presence[0][1] == pytest.approx(100, abs=0.3)


def test_auto_thresholds_unchanged_when_quiet_samples_are_the_majority():
    rng = np.random.default_rng(0)
    quiet = np.abs(rng.normal(0, 3e-4, 900))
    old_enter, _, _ = auto_thresholds(np.concatenate([quiet, np.full(100, 0.02)]))
    assert 0.0005 < old_enter < 0.005  # noise-level threshold, as before
    noise_only = np.abs(rng.normal(0, 3e-4, 1000))
    assert (noise_only >= auto_thresholds(noise_only)[0]).mean() < 0.01


def test_motion_moving_subject_most_of_the_video(moving_box_most_of_video, tmp_path):
    doc = run_cut(moving_box_most_of_video, str(tmp_path / "o"), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5))
    assert len(doc["presence_intervals"]) == 1
    s, e = doc["presence_intervals"][0]
    assert 8 < s < 12 and 98 < e < 102
    assert len(doc["clips"]) == 1


# 2. a still subject must not be absorbed into the background
def test_motion_still_subject_is_not_absorbed(still_box_most_of_video, tmp_path):
    doc = run_cut(still_box_most_of_video, str(tmp_path / "o"), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5))
    assert len(doc["presence_intervals"]) == 1, doc["presence_intervals"]
    s, e = doc["presence_intervals"][0]
    assert 8 < s < 12 and 98 < e < 102


# 3. ffmpeg preflight and keeping the metadata when a cut fails
def test_missing_ffmpeg_fails_before_any_work(synth, tmp_path, monkeypatch):
    only_ffprobe = tmp_path / "bin"
    only_ffprobe.mkdir()
    os.symlink(shutil.which("ffprobe"), only_ffprobe / "ffprobe")
    monkeypatch.setenv("PATH", str(only_ffprobe))

    def boom(*a, **k):
        raise AssertionError("scoring ran although ffmpeg is missing")

    monkeypatch.setattr(pipeline, "score_motion", boom)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        run_cut(synth["plain"], str(tmp_path / "o"), Settings(mode="motion"))


def test_cut_failure_keeps_samples_and_segments(static_box, tmp_path, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("ffmpeg cut failed (disk full)")

    monkeypatch.setattr(pipeline, "cut_clip", fail)
    out = tmp_path / "o"
    with pytest.raises(RuntimeError, match="failed"):
        run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5))
    assert (out / "samples.csv").is_file()
    doc = json.loads((out / "segments.json").read_text())
    assert len(doc["presence_intervals"]) == 1
    assert "disk full" in doc["clips"][0]["error"]


# 4. analysis time and ffmpeg -ss use the same timeline when the video starts after the file start
def test_video_starting_late_cuts_the_right_frames(late, tmp_path):
    info = probe(late)
    assert info.keyframes[0] == pytest.approx(2.0, abs=1e-3)  # same timeline as -ss
    flash = [s.t for s in iter_samples(late, info, fps=10.0) if s.rgb.mean() > 200]
    assert flash and flash[0] == pytest.approx(7.0, abs=0.11)
    clip = tmp_path / "c.mp4"
    cut_clip(late, clip, flash[0] - 0.2, flash[0] + 0.5)
    ci = probe(str(clip))
    assert sum(1 for s in iter_samples(str(clip), ci, fps=10.0) if s.rgb.mean() > 200) >= 3


# 5. a reused output folder holds only the latest run
def test_reused_output_folder_has_no_stale_clips(static_box, tmp_path):
    out = tmp_path / "o"
    run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=0.5, min_len=0.5))
    first = sorted(p.name for p in out.glob("static_box_*.mp4"))
    doc = run_cut(static_box, str(out), Settings(mode="motion", fps=5.0, pad=1.5, min_len=0.5))
    now = sorted(p.name for p in out.glob("static_box_*.mp4"))
    assert now == sorted(c["file"] for c in doc["clips"])
    assert set(first) != set(now)


def test_default_output_folder_is_per_mode(static_box, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "run_cut", lambda video, out, s, **k: seen.append(Path(out)) or dict(clips=[]))
    v = tmp_path / "clip.mp4"
    shutil.copy(static_box, v)
    cli.main(["cut", str(v), "--mode", "motion"])
    cli.main(["cut", str(v), "--mode", "detect"])
    assert seen[0] != seen[1] and seen[0].parent == seen[1].parent == tmp_path.resolve() / "clip_inframe"
