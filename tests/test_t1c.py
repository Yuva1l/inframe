"""T1c: identity (embeddings, gallery) and photo mode. Uses the colour embedder, so no model and no torch are needed;
the real embedders run in a child process and are covered by one test that skips when torch is not installed."""
import json

import numpy as np
import pytest
from PIL import Image

from inframe import cli, identity
from inframe.detect import Detection
from inframe.identity import Ambiguous, Gallery, PhotoError, PhotoTarget, SPECS, person_crop, unit_score
from inframe.tap import Tap, TapError

RED, BLUE, GREEN = (200, 30, 30), (30, 40, 200), (30, 180, 40)


def person(shirt, w=60, h=160, pants=(25, 25, 30)):
    a = np.zeros((h, w, 3), np.uint8)
    a[: h // 2] = shirt
    a[h // 2:] = pants
    return a


def frame_with(people, W=400, H=600):
    """people: list of (colour, box); a grey frame with a coloured two-part figure in every box."""
    f = np.full((H, W, 3), 120, np.uint8)
    for shirt, (x0, y0, x1, y1) in people:
        a, b, c, d = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
        f[b:d, a:c] = person(shirt, c - a, d - b)
    return f


def photos_dir(tmp_path, shirt=RED, n=3):
    d = tmp_path / "photos"
    d.mkdir()
    for i in range(n):
        Image.fromarray(person(shirt, 70 + 4 * i, 180)).save(d / f"p{i}.jpg", quality=95)
    return d


def run(target, steps, boxes, shirts, tracks):
    """steps samples, 0.2 s apart. boxes[k], shirts[k], tracks[k] describe candidate k in every sample."""
    for i in range(steps):
        cands = [Detection(0.9, b, tr) for b, tr in zip(boxes, tracks)]
        frame = frame_with(list(zip(shirts, boxes)))
        target.evaluate(round(i * 0.2, 1), frame, cands)
    return target.finalize()


def test_person_crop_refuses_tiny_boxes():
    f = np.zeros((600, 400, 3), np.uint8)
    assert person_crop(f, (0.1, 0.1, 0.3, 0.5)).shape[0] >= 240
    assert person_crop(f, (0.1, 0.1, 0.12, 0.12)) is None  # 12 px high: nothing to say about it


def test_gallery_dedups_and_keeps_the_diverse_ones():
    spec = SPECS["color"]
    g = Gallery(spec, cap=3)
    v = lambda x: np.asarray(x, np.float32) / np.linalg.norm(x)
    assert g.add(v([1, 0, 0]))
    assert not g.add(v([1, 0.01, 0]))  # a duplicate
    assert g.add(v([0, 1, 0]))
    assert g.add(v([0.9, 0.3, 0]))  # close to the first, not a duplicate
    assert len(g) == 3
    assert g.add(v([0, 0, 1]))  # over the cap: the most redundant member leaves, the new one is diverse and stays
    assert len(g) == 3
    assert float((g.v @ v([0, 0, 1])).max()) > 0.99 and float((g.v @ v([0, 1, 0])).max()) > 0.99


def test_unit_score_is_the_median_of_the_best_matches():
    G = np.eye(3, dtype=np.float32)
    crops = np.asarray([[1, 0, 0], [0.6, 0.8, 0], [0.5, 0.4, 0.3]], np.float32)
    assert unit_score(crops, G) == pytest.approx(0.8)
    assert unit_score(np.zeros((0, 3), np.float32), G) == 0.0


def test_photo_folder_errors_are_plain(tmp_path):
    with pytest.raises(PhotoError, match="not a folder"):
        identity.photo_files(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(PhotoError, match="no photos"):
        identity.photo_files(tmp_path / "empty")
    assert issubclass(PhotoError, TapError)  # the CLI exits 2 on it


def test_photos_become_crops_and_small_variants(tmp_path):
    d = photos_dir(tmp_path, n=2)
    crops = identity.load_photo_crops(d)
    assert len(crops) == 2
    vs = identity.resolution_variants(np.zeros((500, 200, 3), np.uint8))
    assert [v.shape[0] for v in vs] == [500, 128, 80]
    assert len(identity.resolution_variants(np.zeros((120, 50, 3), np.uint8))) == 2  # only the 80 px copy is clearly smaller


def test_parse_targets_photos_and_confirm_file(tmp_path):
    d = photos_dir(tmp_path)
    t = identity.parse_targets([f"photos:{d}", "tap:3.0,0.2,0.5"], "color")
    assert isinstance(t, PhotoTarget) and len(t.taps) == 1
    f = tmp_path / "target.json"
    f.write_text(json.dumps(dict(time=2.0, box=[0.1, 0.1, 0.3, 0.9], photos=str(d), taps=[dict(time=2.0, box=[0.1, 0.1, 0.3, 0.9])])))
    assert isinstance(identity.parse_targets([f"tap:{f}"], "color"), PhotoTarget)  # the confirm file names its photos
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(dict(time=2.0, box=[0.1, 0.1, 0.3, 0.9])))
    assert not isinstance(identity.parse_targets([f"tap:{plain}"], "color"), PhotoTarget)  # tap mode is unchanged
    with pytest.raises(TapError, match="photos"):
        identity.parse_targets(["dog:1"], "color")
    with pytest.raises(PhotoError):
        identity.parse_targets([f"photos:{tmp_path / 'missing'}"], "color")


def test_auto_pick_takes_the_one_that_looks_like_the_photos(tmp_path):
    d = photos_dir(tmp_path, RED)
    t = PhotoTarget(str(d), embedder="color")
    boxes = [(0.05, 0.1, 0.30, 0.9), (0.55, 0.1, 0.80, 0.9)]
    hits = run(t, 15, boxes, [RED, BLUE], [1, 2])
    assert all(h.box == boxes[0] for h in hits) and all(h.track_id == 1 for h in hits)
    assert t.info["ranking"][0]["track_id"] == 1 and t.info["ranking"][0]["score"] > t.info["ranking"][1]["score"]
    assert t.info["tracks_accepted"] == 1


def test_auto_pick_refuses_two_look_alikes(tmp_path):
    d = photos_dir(tmp_path, RED)
    t = PhotoTarget(str(d), embedder="color")
    boxes = [(0.05, 0.1, 0.30, 0.9), (0.55, 0.1, 0.80, 0.9)]
    with pytest.raises(Ambiguous, match="share a frame"):
        run(t, 15, boxes, [RED, RED], [1, 2])


def test_auto_pick_refuses_when_nobody_looks_like_the_photos(tmp_path):
    d = photos_dir(tmp_path, RED)
    t = PhotoTarget(str(d), embedder="color")
    with pytest.raises(Ambiguous, match="inframe pick"):
        run(t, 15, [(0.05, 0.1, 0.30, 0.9)], [GREEN], [1])


def test_confirm_seeds_the_session_gallery_and_finds_him_again(tmp_path):
    """He is in the photos in red. In the video he first wears red (confirmed), later green, as a different track. The confirm's
    crops are the only link: with the green crops of a second track that is NOT similar enough to the photos he stays out."""
    d = photos_dir(tmp_path, RED)
    t = PhotoTarget(str(d), [Tap(0.4, box=(0.05, 0.1, 0.30, 0.9))], embedder="color")
    box = (0.05, 0.1, 0.30, 0.9)
    for i in range(30):
        shirt, tr = (RED, 1) if i < 15 else (GREEN, 2)
        t.evaluate(round(i * 0.2, 1), frame_with([(shirt, box)]), [Detection(0.9, box, tr)])
    hits = t.finalize()
    assert all(h.box == box for h in hits[:15])  # the confirmed track, with the confirm only
    assert all(h.box is None for h in hits[20:])  # green is not red: nothing joins him by chance
    assert t.info["session_gallery"] >= 1


def test_a_second_track_in_the_same_frame_is_never_him(tmp_path):
    d = photos_dir(tmp_path, RED)
    t = PhotoTarget(str(d), [Tap(0.4, box=(0.05, 0.1, 0.30, 0.9))], embedder="color")
    boxes = [(0.05, 0.1, 0.30, 0.9), (0.55, 0.1, 0.80, 0.9)]
    hits = run(t, 15, boxes, [RED, RED], [1, 2])  # a twin in red next to him
    assert all(h.box == boxes[0] for h in hits)


def test_rank_only_decides_nothing_and_lists_the_top(tmp_path):
    d = photos_dir(tmp_path, RED)
    t = PhotoTarget(str(d), embedder="color", rank_only=True)
    hits = run(t, 15, [(0.05, 0.1, 0.30, 0.9), (0.55, 0.1, 0.80, 0.9)], [RED, BLUE], [1, 2])
    assert all(h.box is None for h in hits) and [r["track_id"] for r in t.info["ranking"]] == [1, 2]


def test_cli_exits_3_when_photo_mode_cannot_decide(tmp_path, monkeypatch, capsys):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    monkeypatch.setattr(cli, "run_cut", lambda *a, **k: (_ for _ in ()).throw(Ambiguous("two people look like the photos")))
    monkeypatch.setattr(identity, "parse_targets", lambda *a, **k: object())
    assert cli.main(["cut", str(v), "--target", "photos:/x"]) == 3
    assert "two people look like the photos" in capsys.readouterr().err


def test_cli_exits_2_on_a_bad_photo_folder(tmp_path, capsys):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    assert cli.main(["cut", str(v), "--target", f"photos:{tmp_path / 'none'}"]) == 2
    assert "not a folder" in capsys.readouterr().err


def test_real_embedder_child_process():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (200, 100, 3), dtype=np.uint8) for _ in range(3)]
    try:
        with identity.Embedder("dinov2", "cpu", offline=True) as e:
            v = e.embed(crops)
    except identity.EmbedderUnavailable as e:
        pytest.skip(f"model not in the cache: {e}")
    assert v.shape == (3, 384) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)
