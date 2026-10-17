"""Compare two segments.json files: seconds kept by both, only A, only B, and every stretch where
they disagree for 1 s or more. Works on the clips (the padded stretches that were kept)."""
from __future__ import annotations

import json
from pathlib import Path

Interval = tuple[float, float]


def _norm(iv: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    a, b = _norm(a), _norm(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Parts of `a` not covered by `b`."""
    out: list[Interval] = []
    b = _norm(b)
    for s, e in _norm(a):
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
        if cur < e:
            out.append((cur, e))
    return out


def total(iv: list[Interval]) -> float:
    return sum(e - s for s, e in iv)


def compare_intervals(a: list[Interval], b: list[Interval], min_len: float = 1.0) -> dict:
    both, only_a, only_b = intersect(a, b), subtract(a, b), subtract(b, a)
    dis = sorted([("A", s, e) for s, e in only_a if e - s >= min_len] + [("B", s, e) for s, e in only_b if e - s >= min_len], key=lambda x: x[1])
    return dict(both_s=total(both), only_a_s=total(only_a), only_b_s=total(only_b),
                disagreements=[dict(only_in=w, start=s, end=e, length=e - s) for w, s, e in dis])


def _mmss(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _load(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{p} is not a readable segments.json ({type(e).__name__}: {e})") from e
    if not isinstance(d, dict) or not isinstance(d.get("clips"), list) or not isinstance(d.get("source"), dict):
        raise ValueError(f"{p} is not a segments.json from inframe (needs 'source' and 'clips')")
    return d


def same_source_problem(sa: dict, sb: dict) -> str | None:
    """None when both runs describe the same video, else one line naming what differs."""
    diffs = []
    na, nb = Path(str(sa.get("path", ""))).name, Path(str(sb.get("path", ""))).name
    if na and nb and na != nb:
        diffs.append(f"file name {na!r} vs {nb!r}")
    for k in ("width", "height"):
        if sa.get(k) is not None and sb.get(k) is not None and sa.get(k) != sb.get(k):
            diffs.append(f"{k} {sa.get(k)} vs {sb.get(k)}")
    for k, tol in (("duration", 0.05), ("fps", 0.01)):
        x, y = sa.get(k), sb.get(k)
        if x is None or y is None:
            continue
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or abs(x - y) > tol:
            diffs.append(f"{k} {x} vs {y}")
    return "; ".join(diffs) or None


def compare_files(path_a: str | Path, path_b: str | Path, min_len: float = 1.0, out: str | Path | None = None) -> dict:
    """Compare two segments.json files of the same source video. Raises ValueError when a file is not a
    segments.json or the two come from different videos.
    Writes compare.json and compare.md into `out` (created if missing); by default that is the parent of
    A's folder (<stem>_inframe/ for the default layout <stem>_inframe/<mode>/segments.json)."""
    pa, pb = Path(path_a).expanduser().resolve(), Path(path_b).expanduser().resolve()
    da, db = _load(pa), _load(pb)
    bad = same_source_problem(da["source"], db["source"])
    if bad:
        raise ValueError(f"A and B come from different videos ({bad}); refusing to compare")
    la, lb = da.get("mode", "A"), db.get("mode", "B")
    if la == lb:
        la, lb = f"A ({la})", f"B ({lb})"
    # a clip that failed to cut ("error" in its entry) has no file, so it was not kept
    ia = [(c["start"], c["end"]) for c in da["clips"] if "error" not in c]
    ib = [(c["start"], c["end"]) for c in db["clips"] if "error" not in c]
    r = compare_intervals(ia, ib, min_len)
    for d in r["disagreements"]:
        d["only_in"] = la if d["only_in"] == "A" else lb
    dur = da["source"]["duration"]
    doc = dict(
        a=dict(file=str(pa), label=la, kept_s=round(total(_norm(ia)), 2), n_clips=len(ia)),
        b=dict(file=str(pb), label=lb, kept_s=round(total(_norm(ib)), 2), n_clips=len(ib)),
        video_duration_s=dur, min_disagreement_s=min_len,
        both_s=round(r["both_s"], 2), only_a_s=round(r["only_a_s"], 2), only_b_s=round(r["only_b_s"], 2),
        n_disagreements=len(r["disagreements"]),
        disagreements=[dict(d, start=round(d["start"], 2), end=round(d["end"], 2), length=round(d["length"], 2)) for d in r["disagreements"]],
    )
    out = Path(out).expanduser().resolve() if out else pa.parent.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(doc, indent=1))
    lines = [
        f"# Compare {la} and {lb}", "",
        f"Video {dur:.1f} s. Kept seconds are the padded clips in each segments.json.", "",
        "| | clips | kept seconds |", "|---|---|---|",
        f"| {la} | {len(ia)} | {doc['a']['kept_s']} |", f"| {lb} | {len(ib)} | {doc['b']['kept_s']} |", "",
        f"- kept by both: {doc['both_s']} s",
        f"- only {la}: {doc['only_a_s']} s",
        f"- only {lb}: {doc['only_b_s']} s", "",
        f"## Disagreement stretches of {min_len:g} s or more ({doc['n_disagreements']})", "",
    ]
    if doc["disagreements"]:
        lines += ["| only in | start | end | length s |", "|---|---|---|---|"]
        lines += [f"| {d['only_in']} | {_mmss(d['start'])} | {_mmss(d['end'])} | {d['length']} |" for d in doc["disagreements"]]
    else:
        lines.append("None.")
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    return dict(doc, compare_json=str(out / "compare.json"), compare_md=str(out / "compare.md"))
