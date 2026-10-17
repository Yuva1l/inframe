"""A source named with a control character (a line break) gets one sanitized stem in every name this tool writes and in every
pattern that finds those names again, so a re-export or a re-run replaces the old clips instead of piling up or failing."""
from pathlib import Path

from inframe import cut, editlist, export, pipeline, serve_export, sidecar

STEM = "my\nclip"


def test_every_name_and_pattern_use_the_same_stem(tmp_path):
    clip = cut.clip_name(STEM, 1, 0, 4, ".mp4")
    assert clip == "my_clip_01_0000-0004.mp4"
    side = sidecar.sidecar_name(clip)
    pats = serve_export.name_patterns(STEM)
    assert serve_export.classify(clip, pats) == "clip"
    assert serve_export.classify(side, pats) == "sidecar"
    assert serve_export.classify(export.reel_name(STEM, ".mp4"), pats) == "reel"
    assert serve_export.classify(export.credits_name(STEM), pats) == "credits"
    assert serve_export.classify(f"{cut.safe_stem(STEM)}_losslesscut.csv", pats) == "csv"
    for fmt in editlist.FORMATS:
        assert serve_export.classify(editlist.edit_list_name(STEM, fmt), pats) == "editlist"
    assert "\n" not in export.reel_name(STEM) and "\n" not in editlist.edit_list_name(STEM, editlist.FORMATS[0])


def test_cleanup_removes_the_files_a_run_wrote(tmp_path):
    names = [cut.clip_name(STEM, 1, 0, 4, ".mp4"), sidecar.sidecar_name(cut.clip_name(STEM, 1, 0, 4, ".mp4")),
             export.reel_name(STEM, ".mp4"), export.credits_name(STEM)]
    for n in names:
        (tmp_path / n).write_text("x")
    assert pipeline.clear_old_clips(tmp_path, STEM) == [names[0]]
    sidecar.clear_old_sidecars(tmp_path, STEM)
    export.clear_old_exports(tmp_path, STEM)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_second_export_of_a_line_break_source_replaces_the_clips(tmp_path, static_box):
    import shutil
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from serve_helpers import call, start_server
    from test_serve_export import ADV, wait_job

    vids = tmp_path / "vids"
    vids.mkdir()
    video = vids / "my\nclip.mp4"
    shutil.copy(static_box, video)
    base, app, stop = start_server(tmp_path)
    media = app.open_media(str(video))
    r = call(app, "POST", "/api/runs", {"media_id": media.id, "target": {"kind": "motion"}, "advanced": ADV})
    assert r.status == 200, r.text
    assert wait_job(app, r.json()["job"]["id"])["state"] == "done"
    rid = r.json()["run_id"]
    try:
        for k in range(2):
            e = call(app, "POST", f"/api/runs/{rid}/export", {"clips": True, "reel": False, "edit_lists": [], "replace": True})
            assert e.status == 200, e.text
            res = wait_job(app, e.json()["job"]["id"])
            assert res["state"] == "done", res
            out = res["result"]
            assert [c.get("error") for c in out.get("clips", [])] == [None] * len(out.get("clips", [])), out
            kinds = [o["kind"] for o in out["outputs"]]
            assert kinds.count("clip") == kinds.count("sidecar") >= 1, (k, kinds)
    finally:
        stop()


def test_export_run_names_and_messages_use_the_sanitised_stem(tmp_path, monkeypatch):
    """The reel, credits and LosslessCut names and every warning of `export_run` carry no line break from the source's name."""
    from inframe import export as export_mod

    made = []

    def fake_reel(paths, out):
        made.append(Path(out).name)
        return dict(file=Path(out).name, ok=False, fallback_used=False, check=dict(problems=["x"]))

    monkeypatch.setattr(export_mod, "build_reel", fake_reel)
    (tmp_path / "c.mp4").write_text("x")
    doc = dict(source=dict(path=f"/v/{STEM}.mp4", fps=10, duration=5), presence_intervals=[],
               clips=[dict(start=0, end=2, file="c.mp4")])
    res = export_mod.export_run(doc, tmp_path, tmp_path, True, [])
    assert made == ["my_clip_reel.mp4"]
    assert not any("\n" in x for x in made + res["warnings"] + [p.name for p in tmp_path.iterdir()])
    assert export_mod.write_credits(tmp_path, STEM, {"license": "by", "title": "t", "creator": "c"}, []) == "my_clip_credits.txt"
