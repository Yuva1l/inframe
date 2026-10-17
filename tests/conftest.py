import subprocess

import pytest


@pytest.fixture(autouse=True)
def no_bundled_songs(monkeypatch):
    """The songs that ship in the package would be the first pick of every auto-music test; those tests look at the online part, so the
    bundled library is off unless a test (tests/test_music_library.py) turns it on."""
    monkeypatch.setenv("INFRAME_MUSIC_LIBRARY", "off")


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """3 s, 30 fps, 320x180 testsrc2 + sine audio, keyframe every 15 frames (0.5 s), display rotation -90."""
    d = tmp_path_factory.mktemp("synth")
    plain, rot = d / "plain.mp4", d / "rot.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=3", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-g", "15", "-keyint_min", "15", "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(plain))
    _ff("-display_rotation", "-90", "-i", str(plain), "-c", "copy", str(rot))
    return dict(plain=str(plain), rot=str(rot))


@pytest.fixture(scope="session")
def static_box(tmp_path_factory):
    """8 s, 10 fps, 320x180: static grey scene with sensor noise; a white box is in frame from 2 s to 4 s."""
    d = tmp_path_factory.mktemp("static")
    p = d / "static_box.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x404040:s=320x180:r=10:d=8,drawgrid=w=40:h=40:t=2:c=white@0.4,noise=alls=4:allf=t,drawbox=x=120:y=60:w=50:h=70:color=white:t=fill:enable='between(t,2,4)'",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=8", "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p))
    return str(p)


@pytest.fixture(scope="session")
def shaky(tmp_path_factory):
    """Same scene as static_box but the whole picture jumps by +-6 px every frame (a camera that is not static)."""
    d = tmp_path_factory.mktemp("shaky")
    p = d / "shaky.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x404040:s=360x220:r=10:d=8,drawgrid=w=40:h=40:t=2:c=white@0.4,noise=alls=4:allf=t",
        "-vf", "crop=320:180:x='20+6*sin(n*2.3)':y='20+6*cos(n*1.7)':exact=1,format=yuv420p",
        "-c:v", "libx264", "-g", "10", str(p))
    return str(p)


def _long_scene(path, box_filter):
    """120 s, 5 fps, 160x90 noisy grey scene with a grid; a 25x35 white box (6.1% of the frame) is in frame 10 s to 100 s."""
    _ff("-f", "lavfi", "-i", "color=c=0x404040:s=160x90:r=5:d=120,drawgrid=w=20:h=20:t=1:c=white@0.4,noise=alls=4:allf=t",
        "-f", "lavfi", "-i", "color=c=white:s=25x35:r=5:d=120",
        "-filter_complex", f"[0][1]overlay={box_filter}:enable='between(t,10,100)',format=yuv420p",
        "-c:v", "libx264", "-g", "10", str(path))
    return str(path)


@pytest.fixture(scope="session")
def moving_box_most_of_video(tmp_path_factory):
    return _long_scene(tmp_path_factory.mktemp("mov") / "moving.mp4", "x='60+40*sin(t*2)':y=25")


@pytest.fixture(scope="session")
def still_box_most_of_video(tmp_path_factory):
    return _long_scene(tmp_path_factory.mktemp("still") / "still.mp4", "x=60:y=25")


@pytest.fixture(scope="session")
def late(tmp_path_factory):
    """8 s of video that starts at 2.0 s in the file, audio from 0.0 s. Black, with a white flash at video time 5.0-5.4 s
    (file time 7.0-7.4 s). Keyframe every 0.5 s."""
    d = tmp_path_factory.mktemp("late")
    plain, p = d / "late_plain.mp4", d / "late.mov"
    _ff("-f", "lavfi", "-i", "color=c=black:s=160x90:r=10:d=8,drawbox=x=0:y=0:w=160:h=90:color=white:t=fill:enable='between(t,5,5.4)'",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=10", "-c:v", "libx264", "-g", "5", "-pix_fmt", "yuv420p", "-c:a", "aac", str(plain))
    _ff("-itsoffset", "2", "-i", str(plain), "-itsoffset", "0", "-i", str(plain), "-map", "0:v", "-map", "1:a", "-c", "copy", str(p))
    return str(p)


@pytest.fixture(scope="session")
def open_gop(tmp_path_factory):
    """6 s, 30 fps, 320x180 HEVC with open GOPs and B-frames (CRA keyframe every 30 frames, so the pictures
    right after a keyframe in decode order show before it), 44.1 kHz AAC audio, display rotation -90."""
    d = tmp_path_factory.mktemp("opengop")
    plain, rot = d / "og.mp4", d / "og_rot.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=6", "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx265", "-x265-params", "open-gop=1:keyint=30:min-keyint=30:bframes=4:scenecut=0:log-level=error",
        "-tag:v", "hvc1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-shortest", str(plain))
    _ff("-display_rotation", "-90", "-i", str(plain), "-c", "copy", str(rot))
    return str(rot)


@pytest.fixture(autouse=True)
def _no_slices_by_default(monkeypatch):
    """Tap mode runs the detector on slices of the frame (detect.SlicedDetector). Tests with a fake detector, or a real one on
    synthetic clips, want the plain detector; the tests of the slicing itself set INFRAME_SLICES."""
    monkeypatch.setenv("INFRAME_SLICES", "off")


@pytest.fixture(autouse=True)
def _own_targets_store(tmp_path_factory, monkeypatch):
    """Saved targets (T2) never touch the real store in ~/Library/Application Support/inframe/targets."""
    monkeypatch.setenv("INFRAME_TARGETS", str(tmp_path_factory.mktemp("targets_store")))


# Where the target is inside each target clip (file seconds: duration, first, last second in frame). Real values are read from
# target_windows.json next to the clips (tools/splice.py); the unit tests use these made-up ones.
SYNTH_TARGET_WINDOW = {"targetA": (17.5, 1.25, 16.5), "targetB": (17.0, 1.0, 15.9), "targetC": (16.5, 2.0, 15.5)}


@pytest.fixture(autouse=True)
def synthetic_target_windows(request, monkeypatch):
    import sys

    splice = sys.modules.get("splice")  # only tests that import tools/splice.py need it
    if splice is not None and "browser" not in request.node.path.parts:
        monkeypatch.setattr(splice, "TARGET_WINDOW", dict(SYNTH_TARGET_WINDOW))
    yield
