"""Reel and edit lists for a finished run: `inframe cut --reel --edit-list ...` and `inframe export`."""
from __future__ import annotations

import re
from pathlib import Path

from .names import safe_stem
from .editlist import FORMATS, clear_old_edit_lists, write_edit_lists
from .reel import build_reel



def credits_name(stem: str) -> str:
    """One credits file per source video, so two videos exported into one folder do not overwrite each other's."""
    return f"{safe_stem(stem)}_credits.txt"


def reel_name(stem: str, suffix: str = ".mov") -> str:
    return f"{safe_stem(stem)}_reel{suffix}"


def music_reel_name(stem: str, suffix: str = ".mov") -> str:
    """The reel with music is always re-encoded H.264/AAC, so it keeps the source's container only when that holds them (.mov, .mp4,
    .m4v); a .ogv, .ogg, .webm or any other container gets .mp4."""
    return reel_name(stem, suffix if suffix.lower() in (".mov", ".mp4", ".m4v") else ".mp4")


def clear_old_exports(out: Path, stem: str) -> None:
    """Remove the reel and edit lists an earlier run left in `out` (their names are the ones this tool writes), and the
    export.json that `inframe export` wrote about them: a new cut run replaces segments.json, so it no longer matches."""
    pat = re.compile(rf"^{re.escape(safe_stem(stem))}_reel\.\w+$")
    for f in out.iterdir():
        if f.is_file() and pat.match(f.name):
            f.unlink()
    (out / credits_name(stem)).unlink(missing_ok=True)  # the credit of an earlier reel's music
    clear_old_edit_lists(out, stem)
    (out / "export.json").unlink(missing_ok=True)


def tail_slack(doc: dict, clip: dict) -> float:
    """Seconds at the end of `clip` that are padding, not presence: the clip's end minus the end of the last presence interval
    inside it (the run's `pad` when the intervals are not in the document). A beat-synced reel may drop this much of the tail."""
    k = float((clip.get("slowmo") or {}).get("factor", 1.0))  # a retimed clip plays `factor` times longer than its source stretch
    ends = [e for s_, e in doc.get("presence_intervals", []) if s_ < clip["end"] and e <= clip["end"] + 1e-6 and e > clip["start"]]
    if ends:
        return max(0.0, clip["end"] - max(ends)) * k
    return float(doc.get("settings", {}).get("pad", 0.0)) * k


def write_credits(out: Path, stem: str, meta: dict | None, warnings: list[str]) -> str | None:
    """<stem>_credits.txt in `out` when the music's license asks for a credit; returns its name, or None."""
    from .music import credits_text, needs_credit

    if meta is None:
        warnings.append("the music has no license record (no <name>.json beside it), so no credits file was written; check its license yourself")
        return None
    if not str(meta.get("license") or "").strip():
        warnings.append("the music's license record has no license field, so no credits file was written; check its license yourself")
        return None
    if not needs_credit(meta.get("license")):
        return None
    (out / credits_name(stem)).write_text(credits_text(meta))
    return credits_name(stem)


def export_run(doc: dict, clips_dir: str | Path, out_dir: str | Path, reel: bool, formats: list[str],
               music: tuple[str | Path, dict | None] | None = None, beat_sync: bool = False) -> dict:
    """Build the reel from the run's clips (in `clips_dir`) and the edit lists from its segments, into `out_dir`.
    With `music` = (audio file, license metadata or None) the reel's only sound is that music (trimmed or looped, faded, -14 LUFS);
    with `beat_sync` every clip change also lands on a beat (the clips are re-encoded), and `<stem>_credits.txt` is written when the
    license asks for a credit. Returns {"reel": dict | None, "edit_lists": [...], "warnings": [...], "ok": bool}. A failure of one output
    is reported, never raised."""
    clips_dir, out = Path(clips_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(doc["source"]["path"]).stem
    warnings: list[str] = []
    result: dict = dict(reel=None, edit_lists=[])
    good = [c for c in doc["clips"] if "error" not in c]
    if reel:
        # a clip without "file" was cut but its file was not kept (a reel-only export): it is as missing as a deleted one
        missing = [c.get("file") or f"clip {c.get('index', '?')}" for c in good if not c.get("file") or not (clips_dir / c["file"]).is_file()]
        if len(good) < len(doc["clips"]):
            warnings.append(f"the reel lacks {len(doc['clips']) - len(good)} clip(s) that could not be cut")
        if missing:
            warnings.append(f"reel not built: clip file(s) missing from {clips_dir}: {', '.join(missing[:3])}")
            result["reel"] = dict(file=None, ok=False, problem=warnings[-1])
        elif not good:
            warnings.append("reel not built: no clips")
            result["reel"] = dict(file=None, ok=False, problem=warnings[-1])
        else:
            suffix = Path(good[0]["file"]).suffix or ".mov"
            try:
                paths = [str(clips_dir / c["file"]) for c in good]
                if music is None:
                    r = build_reel(paths, out / reel_name(stem, suffix))
                    (out / credits_name(stem)).unlink(missing_ok=True)  # a new reel never keeps the credit of the music of an earlier one
                else:
                    from .musicreel import build_music_reel

                    from .music import credits_text, needs_credit

                    credit = (credits_text(music[1]).strip()
                              if music[1] and str(music[1].get("license") or "").strip() and needs_credit(music[1].get("license")) else None)
                    r = build_music_reel(paths, out / music_reel_name(stem, suffix), str(music[0]), beat_sync, [tail_slack(doc, c) for c in good],
                                         credit=credit)
                    (out / credits_name(stem)).unlink(missing_ok=True)  # the new reel is in place: drop the credit of the earlier one's music (a failed build keeps both)
                    warnings += [f"beat sync: {w}" for w in r.pop("warnings", [])]
                    r["music"]["file"] = Path(music[0]).name
                    r["music"]["license"] = (music[1] or {}).get("license")
                    r["music"]["title"] = (music[1] or {}).get("title")
                    r["music"]["creator"] = (music[1] or {}).get("creator")
                    r["music"]["auto"] = (music[1] or {}).get("auto")  # set by auto music: {rank, n, mood, speed, score, bpm, builtin, wrapped}
                    r["music"]["credits"] = write_credits(out, stem, music[1], warnings)
                    r["music"]["credit_text"] = credits_text(music[1]).strip() if r["music"]["credits"] else None  # what the UI shows
            except Exception as e:
                warnings.append(f"reel failed: {type(e).__name__}: {str(e)[:200]}")
                r = dict(file=None, ok=False, problem=warnings[-1])
            else:
                if r["fallback_used"]:
                    warnings.append("the reel needed -avoid_negative_ts make_zero to line up its joins")
                if not r["ok"]:
                    warnings.append("the reel failed its join check: " + "; ".join(r["check"]["problems"][:3]))
            result["reel"] = r
    if formats:
        results, w = write_edit_lists(doc, out, formats)
        result["edit_lists"] = results
        warnings += w
    ok = (result["reel"] is None or bool(result["reel"].get("ok"))) and all(r.get("ok") in (True, None) for r in result["edit_lists"])
    return dict(result, warnings=warnings, ok=ok)


__all__ = ["export_run", "clear_old_exports", "reel_name", "music_reel_name", "FORMATS"]
