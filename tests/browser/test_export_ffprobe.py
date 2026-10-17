"""Lossless clip export in Chrome (docs/T9B_DESIGN.md sections 6 and 8): the real BrowserEngine, the real mux and zip code, a real file, the File System
Access API (the page's private folder, a streaming FileSystemWritableFileStream) and the Blob download used where there is no picker (Safari).

Each clip of the zip is checked with ffprobe and ffmpeg and against the Python `cut` of the same stretch: the first shown picture at 0 (one frame of
slack), an edit list, the rotation of the source, hvc1 with hvcC (avc1 with avcC for H.264), frame count and duration within a frame of the plan and of
the Python cut, audio within one AAC frame of the picture, nothing printed by `ffmpeg -v error -i clip -f null -`. The zip must pass `unzip -t`, the
sidecars tools/validate_sidecars.py and the CSV must equal cut.write_losslesscut_csv. Synthetic media only (ffmpeg testsrc2 and a sine); nothing is looked at.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, _ff, open_page
from inframe.cut import cut_clip, write_losslesscut_csv

CLIPS = [[0.0, 3.0], [3.5, 6.1], [8.2, 11.9]]  # in time order, as the engine writes them


def _x265_params() -> str:
    return "open-gop=1:keyint=30:min-keyint=30:bframes=3:log-level=error"


@pytest.fixture(scope="session")
def sources(tmp_path_factory) -> dict[str, Path]:
    """12 s, 30 fps, 640x360: HEVC with B-frames, an open GOP, display rotation 90 and AAC; and an H.264 .mov with B-frames and AAC."""
    d = tmp_path_factory.mktemp("export")
    v, a = "testsrc2=size=640x360:rate=30:duration=12", "sine=frequency=440:duration=12"
    raw = d / "raw.mp4"
    _ff("-f", "lavfi", "-i", v, "-f", "lavfi", "-i", a, "-c:v", "libx265", "-tag:v", "hvc1", "-preset", "ultrafast", "-g", "30", "-x265-params", _x265_params(), "-c:a", "aac", "-shortest", str(raw))
    hevc = d / "hevc_open_gop_rot90.mp4"
    _ff("-display_rotation", "90", "-i", str(raw), "-c", "copy", "-tag:v", "hvc1", str(hevc))
    mov = d / "h264_bframes.mov"
    _ff("-f", "lavfi", "-i", v, "-f", "lavfi", "-i", a, "-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-bf", "2", "-c:a", "aac", "-shortest", str(mov))
    return {"hevc": hevc, "h264": mov}


def facts(path: Path) -> dict:
    j = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", "-count_frames", "-show_entries",
         "stream=codec_type,codec_tag_string,start_time,duration,r_frame_rate,nb_read_frames,sample_rate:stream_side_data=rotation:format=duration", str(path)],
        capture_output=True, text=True, check=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    fr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-select_streams", "v:0", "-show_entries", "frame=pts_time", str(path)],
                                   capture_output=True, text=True, check=True).stdout)["frames"]
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    rot = next((x["rotation"] for x in v.get("side_data_list", []) if "rotation" in x), 0)
    return dict(tag=v["codec_tag_string"], rotation=round(float(rot)), fps=num / den, frames=int(v["nb_read_frames"]), first_shown=min(float(f["pts_time"]) for f in fr),
                duration=float(v.get("duration") or j["format"]["duration"]), v_start=float(v["start_time"]), a_start=None if a is None else float(a["start_time"]),
                a_rate=0 if a is None else int(a["sample_rate"]))


def has_box(path: Path, *names: bytes) -> bool:
    data = path.read_bytes()
    return all(n in data for n in names)


def has_elst(path: Path) -> int:
    """How many tracks of the clip carry an edit list (the elst box sits inside edts inside each trak)."""
    return path.read_bytes().count(b"elst")


def run_export(page, sink: str, tmp: Path):
    r = page.evaluate("(o) => window.__t.exportZip(o)", {"clips": CLIPS, "sink": sink})
    assert r["error"] is None and r["state"] == "done", r
    zp = tmp / r["zipName"]
    zp.write_bytes(base64.b64decode(r["zipBase64"]))
    return r, zp


@pytest.mark.parametrize("kind,sink", [("hevc", "folder"), ("hevc", "download"), ("h264", "folder")])
def test_zip_of_lossless_clips(chrome, site, kind, sink, sources, tmp_path, capsys):
    src = sources[kind]
    page = open_page(chrome, site)
    start = len(chrome.requests)
    chrome.problems.clear()
    try:
        page.set_input_files("#file", str(src))
        r, zp = run_export(page, sink, tmp_path)
    finally:
        page.close()
    with capsys.disabled():
        print(f"\n[export:{kind}/{sink}] {r['ms']:.0f} ms, zip {zp.stat().st_size} bytes, {len(r['result']['clips'])} clips")
    assert r["monotonic"] and r["result"]["n_clips"] == 3 and r["result"]["warnings"] == []
    assert r["result"]["outputs"][0]["kind"] == "zip"
    if sink == "folder":
        assert r["opfsSize"] == zp.stat().st_size == r["result"]["outputs"][0]["size"]

    # the zip unpacks
    t = subprocess.run(["unzip", "-t", str(zp)], capture_output=True, text=True)
    assert t.returncode == 0 and "No errors detected" in t.stdout, t.stdout + t.stderr
    out = tmp_path / "unzipped"
    subprocess.run(["unzip", "-q", str(zp), "-d", str(out)], check=True)
    stem, ext = src.stem, src.suffix
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"{stem}_{i:02d}_{int(s) // 60:02d}{int(s) % 60:02d}-{int(e) // 60:02d}{int(e) % 60:02d}{ext}" for i, (s, e) in enumerate(CLIPS, 1)]
        + [f"{stem}_{i:02d}_{int(s) // 60:02d}{int(s) % 60:02d}-{int(e) // 60:02d}{int(e) % 60:02d}.inframe.json" for i, (s, e) in enumerate(CLIPS, 1)]
        + [f"{stem}_losslesscut.csv"])

    # sidecars pass the validator; the CSV equals the Python one
    v = subprocess.run([sys.executable, str(ROOT / "tools/validate_sidecars.py"), str(out)], capture_output=True, text=True)
    assert v.returncode == 0 and "PASS validate_sidecars: 3 of 3 sidecars valid" in v.stdout, v.stdout + v.stderr
    clips = sorted(p for p in out.iterdir() if p.suffix == ext)
    ref_csv = tmp_path / "ref.csv"
    write_losslesscut_csv(ref_csv, [(s, e, c.name) for (s, e), c in zip(CLIPS, clips)])
    assert (out / f"{stem}_losslesscut.csv").read_bytes() == ref_csv.read_bytes()

    sf = facts(src)
    frame = 1 / sf["fps"]
    hevc = kind == "hevc"
    for (s, e), clip in zip(CLIPS, clips):
        f = facts(clip)
        assert abs(f["first_shown"]) <= frame + 1e-6, f
        assert abs(f["v_start"]) <= frame + 1e-6, f
        assert has_elst(clip) >= 2, "an edit list in each track"
        assert f["tag"] == ("hvc1" if hevc else "avc1") and has_box(clip, b"hvcC" if hevc else b"avcC")
        assert f["rotation"] == sf["rotation"] == (90 if hevc else 0)
        ref = tmp_path / f"ref_{clip.name}"
        cut_clip(str(src), ref, s, e)
        r_ = facts(ref)
        assert abs(f["frames"] - r_["frames"]) <= 1, (f["frames"], r_["frames"])
        assert abs(f["duration"] - r_["duration"]) <= frame + 1e-3, (f["duration"], r_["duration"])
        assert f["a_start"] is not None and abs(f["a_start"] - f["v_start"]) <= 1024 / f["a_rate"] + 1e-6
        d = subprocess.run(["ffmpeg", "-nostats", "-v", "error", "-i", str(clip), "-f", "null", "-"], capture_output=True, text=True)
        assert d.stdout + d.stderr == ""
    # nothing but GETs to the page's own origin
    new = chrome.requests[start:]
    assert all(m == "GET" and (u.startswith(site.base + "/") or u.startswith(("blob:", "data:"))) for m, u in new), [x for x in new if x[0] != "GET"]
    # mp4box prints a line through console.error for each box it does not know (ffmpeg's udta in a .mov); those are logs, not errors
    real = [x for x in chrome.problems if "[BoxParser] Invalid box type" not in x]
    assert real == [], real


def test_two_exports_in_a_row(chrome, site, sources, tmp_path):
    """The second export of the same page writes over the first one's zip and works as well."""
    page = open_page(chrome, site)
    try:
        page.set_input_files("#file", str(sources["h264"]))
        r = page.evaluate("(o) => window.__t.exportZip(o)", {"clips": [[1.0, 3.0]], "sink": "folder"})
        assert r["state"] == "done" and r["result"]["n_clips"] == 1
        again = page.evaluate("(o) => window.__t.exportZip(o)", {"clips": [[1.0, 3.0]], "sink": "folder"})
        assert again["state"] == "done"
    finally:
        page.close()
    shutil.rmtree(tmp_path, ignore_errors=True)
