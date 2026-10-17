#!/usr/bin/env python3
"""Validate inframe clip sidecars against docs/sidecar.schema.json.

    uv run python tools/validate_sidecars.py RUN_DIR_OR_FILE...

Checks the schema, then what a schema cannot say: the sidecar's clip file sits next to it, the samples lie inside the clip and are in
time order, clip_time = source_time - copy_start_s, boxes are ordered and inside 0..1, and try_number is
within tries_total. Prints one line per sidecar and a last PASS or FAIL line; exit 0 only when every sidecar passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "docs" / "sidecar.schema.json"


def extra_checks(d: dict, folder: Path) -> list[str]:
    bad = []
    c = d["clip"]
    if not (folder / c["file"]).is_file():
        bad.append(f"clip file {c['file']} not next to the sidecar")
    if d["try_number"] > d["tries_total"]:
        bad.append("try_number above tries_total")
    if not c["copy_start_s"] <= c["in_source_s"] + 1e-6 <= c["out_source_s"]:
        bad.append("copy_start_s <= in_source_s <= out_source_s does not hold")
    prev = None
    for i, s in enumerate(d["samples"]):
        if abs(s["clip_time"] - (s["source_time"] - c["copy_start_s"])) > 0.002:
            bad.append(f"sample {i}: clip_time is not source_time - copy_start_s")
        if not c["copy_start_s"] - 0.002 <= s["source_time"] <= c["out_source_s"] + 0.002:
            bad.append(f"sample {i}: source_time {s['source_time']} lies outside the clip ({c['copy_start_s']}..{c['out_source_s']})")
        if prev is not None and s["source_time"] < prev:
            bad.append(f"sample {i}: not in time order")
        prev = s["source_time"]
        b = s["box"]
        if b is not None and not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            bad.append(f"sample {i}: box {b} is not an ordered box inside 0..1")
    if d["sampling"]["n_samples"] != len(d["samples"]):
        bad.append("sampling.n_samples differs from the sample count")
    if len(d["presence_in_clip"]["source_s"]) != len(d["presence_in_clip"]["clip_s"]):
        bad.append("presence_in_clip lists differ in length")
    return bad[:6]


def main(argv: list[str]) -> int:
    from jsonschema import Draft202012Validator

    if not argv:
        print(__doc__)
        return 2
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    files: list[Path] = []
    for a in argv:
        p = Path(a)
        files += sorted(p.glob("*.inframe.json")) if p.is_dir() else [p]
    if not files:
        print("FAIL validate_sidecars: no sidecars found")
        return 1
    failed = 0
    for f in files:
        try:
            d = json.loads(f.read_text())
            errs = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:120]}" for e in validator.iter_errors(d)][:6]
            errs += extra_checks(d, f.parent) if not errs else []
        except Exception as e:  # unreadable or malformed JSON
            errs = [f"{type(e).__name__}: {e}"]
        print(f"{'ok  ' if not errs else 'FAIL'} {f.name}" + ("" if not errs else "  " + "; ".join(errs)))
        failed += bool(errs)
    print(f"{'FAIL' if failed else 'PASS'} validate_sidecars: {len(files) - failed} of {len(files)} sidecars valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
