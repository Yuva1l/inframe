"""T3c: slow-motion playback of the clips (setts after the align pass, picture never re-encoded)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from inframe import cli, pipeline
from inframe.cut import atempo_chain, cut_clip, parse_slowmo, resolve_slowmo, retimed_facts
from inframe.pipeline import Settings, run_cut
from inframe.probe import probe

from inframe.detect import Detection

ROOT = Path(__file__).resolve().parents[1]


class FakeDetector:
    """A person in samples 5..9 only (1.0 s to 2.0 s at 5 samples a second)."""
    backend, path = "fake", "fake.mlpackage"

    def __init__(self):
        self.n = 0

    def detect(self, frames):
        out = []
        for _ in frames:
            out.append([Detection(0.9, (0.1, 0.1, 0.3, 0.5))] if 5 <= self.n <= 9 else [])
            self.n += 1
        return out


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostats", "-nostdin", *args], check=True)


def picture_md5(path: str) -> str:
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostats", "-nostdin", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "md5", "-"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


@pytest.fixture(scope="session")
def syn240(tmp_path_factory):
    """8 s, 240 fps HEVC (open GOP, B-frames, one keyframe per second), 160x90, 48 kHz AAC, display rotation -90."""
    d = tmp_path_factory.mktemp("syn240")
    plain, rot = d / "plain.mov", d / "syn240.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=240:duration=8", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
        "-c:v", "libx265", "-x265-params", "keyint=240:min-keyint=240:bframes=3:open-gop=1:log-level=error", "-tag:v", "hvc1",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(plain))
    _ff("-display_rotation", "-90", "-i", str(plain), "-c", "copy", str(rot))
    return str(rot)


def test_parse_and_resolve():
    assert [parse_slowmo(v) for v in ("auto", "OFF", "8", "4.0", "2.5", "1")] == ["auto", "off", "8", "4", "2.5", "off"]
    for bad in ("0.5", "0", "-4", "fast", "nan", "inf", "1000"):
        with pytest.raises(ValueError):
            parse_slowmo(bad)
    assert resolve_slowmo("auto", 240.0) == 8.0 and resolve_slowmo("auto", 120.0) == 4.0
    assert resolve_slowmo("auto", 30.0) == 1.0 and resolve_slowmo("auto", 0.0) == 1.0  # a normal file is left alone
    assert resolve_slowmo("off", 240.0) == 1.0 and resolve_slowmo("4", 240.0) == 4.0


def test_atempo_chain():
    assert atempo_chain(8) == "atempo=0.5,atempo=0.5,atempo=0.5"
    assert atempo_chain(4) == "atempo=0.5,atempo=0.5"
    assert atempo_chain(2) == "atempo=0.5"
    assert atempo_chain(1.25) == "atempo=0.8"
    assert atempo_chain(3) == "atempo=0.5,atempo=0.66666667"


def test_retime_video_only_keeps_the_picture(syn240, tmp_path):
    base, slow = tmp_path / "base.mov", tmp_path / "slow.mov"
    cut_clip(syn240, base, 2.0, 5.0)
    cut_clip(syn240, slow, 2.0, 5.0, 8.0)
    b, s = retimed_facts(str(base)), retimed_facts(str(slow))
    assert b["fps"] == 240.0 and s["fps"] == 30.0
    assert s["duration_s"] == pytest.approx(b["duration_s"] * 8, abs=0.1)  # the video stream is what lasts: 8 x ~3.0 s
    assert not s["has_audio"] and s["av_offset_ms"] is None
    assert picture_md5(str(slow)) == picture_md5(str(base))  # same packets, only the timestamps moved
    si = probe(str(slow))
    assert si.rotation == -90 and si.codec == "hevc"
    assert si.offset == pytest.approx(0.0, abs=1e-3)  # the first picture sits at zero, not at 8 x its old offset
    assert not list(tmp_path.glob("*.part*"))


def test_retime_with_audio_offset_within_one_aac_frame(syn240, tmp_path):
    slow = tmp_path / "slow_a.mov"
    cut_clip(syn240, slow, 2.0, 5.0, 8.0, slowmo_audio=True)
    f = retimed_facts(str(slow))
    assert f["has_audio"] and abs(f["av_offset_ms"]) <= 11.0  # half an AAC frame, well inside the 21 ms limit
    r = subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_entries", "stream=codec_type,codec_name,duration", str(slow)],
                       capture_output=True, text=True, check=True)
    st = {x["codec_type"]: x for x in json.loads(r.stdout)["streams"]}
    assert st["audio"]["codec_name"] == "aac"
    assert float(st["audio"]["duration"]) == pytest.approx(float(st["video"]["duration"]), abs=0.3)  # stretched to the picture's length


def test_factor_4_on_240_fps_gives_60_fps(syn240, tmp_path):
    out = tmp_path / "s4.mov"
    cut_clip(syn240, out, 2.0, 4.0, 4.0)
    f = retimed_facts(str(out))
    assert f["fps"] == 60.0 and f["duration_s"] == pytest.approx(4.0 * (f["duration_s"] / 4.0), abs=1e-9)
    base = tmp_path / "b.mov"
    cut_clip(syn240, base, 2.0, 4.0)
    assert f["duration_s"] == pytest.approx(retimed_facts(str(base))["duration_s"] * 4, abs=0.1)


def test_run_cut_records_map_and_sidecar(syn240, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    out = tmp_path / "o"
    doc = run_cut(syn240, str(out), Settings(pad=0.5, min_len=0.5, slowmo="auto", slowmo_audio=True, reel=True), None)
    assert doc["slowmo"]["factor"] == 8.0 and doc["slowmo"]["capture_fps"] == 240.0 and doc["slowmo"]["audio"] == "atempo"
    c = doc["clips"][0]
    sm = c["slowmo"]
    assert sm["factor"] == 8.0 and sm["source_t0"] == c["expected_copy_start"] and sm["clip_start_s"] == 0.0 and sm["fps"] == 30.0
    assert sm["duration_s"] == pytest.approx((c["end"] - c["expected_copy_start"]) * 8, abs=0.2)
    assert abs(sm["av_offset_ms"]) <= 21.0
    on_disk = json.loads((out / "segments.json").read_text())
    assert on_disk["clips"][0]["slowmo"] == sm and on_disk["settings"]["slowmo"] == "auto"
    assert c["start"] < 2.0 < c["end"] and doc["presence_intervals"][0][0] < 1.1  # times stay in source seconds
    sc = json.loads((out / c["sidecar"]).read_text())
    assert sc["schema_version"] == "1.2" and sc["slowmo"]["factor"] == 8.0 and sc["slowmo"]["played_duration_s"] == sm["duration_s"]
    assert sc["samples"][0]["clip_time"] == pytest.approx(sc["samples"][0]["source_time"] - c["expected_copy_start"], abs=1e-3)
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "validate_sidecars.py"), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert doc["reel"]["ok"], doc["reel"]  # the retimed clips still join cleanly


def test_run_cut_slowmo_off_is_unchanged_and_audio_dropped_by_default(syn240, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    off = run_cut(syn240, str(tmp_path / "off"), Settings(pad=0.5, min_len=0.5), None)
    assert "slowmo" not in off and "slowmo" not in off["clips"][0]
    assert retimed_facts(str(tmp_path / "off" / off["clips"][0]["file"]))["fps"] == 240.0
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    on = run_cut(syn240, str(tmp_path / "on"), Settings(pad=0.5, min_len=0.5, slowmo="4"), None)
    sm = on["clips"][0]["slowmo"]
    assert on["slowmo"]["audio"] == "dropped" and not sm["has_audio"] and sm["factor"] == 4.0 and sm["fps"] == 60.0
    assert not probe(str(tmp_path / "on" / on["clips"][0]["file"])).has_audio


def test_auto_on_a_normal_speed_file_changes_nothing(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_detector", lambda size: FakeDetector())
    doc = run_cut(synth["plain"], str(tmp_path / "o"), Settings(pad=0.5, min_len=0.5, slowmo="auto"), None)
    assert doc["slowmo"]["factor"] == 1.0 and "slowmo" not in doc["clips"][0]
    assert any("not retimed" in w for w in doc["warnings"])
    assert probe(str(tmp_path / "o" / doc["clips"][0]["file"])).has_audio


def test_bad_settings_are_refused(synth, tmp_path):
    with pytest.raises(ValueError, match="--slowmo"):
        run_cut(synth["plain"], str(tmp_path / "a"), Settings(slowmo="0.5"))
    with pytest.raises(ValueError, match="--slowmo-audio needs"):
        run_cut(synth["plain"], str(tmp_path / "b"), Settings(slowmo_audio=True))


def test_cli_flags(synth, capsys):
    p = cli.build_parser()
    assert p.parse_args(["cut", "x.mov"]).slowmo == "off"
    assert p.parse_args(["cut", "x.mov", "--slowmo", "8", "--slowmo-audio"]).slowmo == "8"
    assert p.parse_args(["cut", "x.mov", "--slowmo", "AUTO"]).slowmo == "auto"
    with pytest.raises(SystemExit) as e:
        p.parse_args(["cut", "x.mov", "--slowmo", "0.5"])
    assert e.value.code == 2
    assert cli.main(["cut", synth["plain"], "--slowmo-audio"]) == 2
    assert "--slowmo-audio needs" in capsys.readouterr().err


@pytest.fixture(scope="session")
def vfr30(tmp_path_factory):
    """A normal-speed 30 fps file whose every 7th packet is 400 ticks (of 1/90000) off: ffprobe says r_frame_rate 450/1, avg 30/1."""
    d = tmp_path_factory.mktemp("vfr30")
    plain, vfr = d / "cfr30.mov", d / "vfr30.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30:duration=4", "-c:v", "libx264", "-bf", "0", "-g", "30", "-pix_fmt", "yuv420p",
        "-video_track_timescale", "90000", str(plain))
    _ff("-i", str(plain), "-c", "copy", "-video_track_timescale", "90000",
        "-bsf:v", "setts=ts='if(eq(mod(N,7),0),TS+400,TS)':duration=3000", str(vfr))
    return str(vfr)


def test_auto_does_not_trust_a_jittery_r_frame_rate(vfr30):
    from inframe.probe import capture_fps
    d = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate",
                                   "-of", "json", vfr30], capture_output=True, text=True, check=True).stdout)["streams"][0]
    assert d["avg_frame_rate"] == "30/1" and float(__import__("fractions").Fraction(d["r_frame_rate"])) > 100  # the premise: r_frame_rate is wrong here
    cap = capture_fps(vfr30)
    assert cap == pytest.approx(30.0, abs=0.5)
    assert resolve_slowmo("auto", cap) == 1.0


def test_auto_is_clamped_and_leaves_ordinary_high_frame_rates_alone():
    assert resolve_slowmo("auto", 5000.0) <= 100.0
    assert resolve_slowmo("auto", 60.0) == 1.0 and resolve_slowmo("auto", 59.94) == 1.0 and resolve_slowmo("auto", 90.0) == 1.0
    assert resolve_slowmo("auto", 120.0) == 4.0 and resolve_slowmo("4", 60.0) == 4.0  # an explicit factor is still honoured
