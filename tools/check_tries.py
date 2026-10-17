"""Do the tries of a slow-motion original match the same session's 30 fps export? (T3a, checked by script)

    uv run python tools/check_tries.py <original segments.json> <export segments.json> [--factor 8] [--tol 0.3] [--head 1] [--tail-from 20] [--offset 0.5]

Reads two segments.json files (the original cut with `inframe cut photos:<uuid>`, the export cut as a plain file). Prints one
PASS/FAIL line plus one line per try, and exits 0 on PASS. Only numbers are read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inframe import photos  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("original")
    p.add_argument("export")
    p.add_argument("--factor", type=float, default=None, help="slow-down factor (default: the one inframe wrote into the original's segments.json, else fps/30)")
    p.add_argument("--tol", type=float, default=0.3, help="seconds; allowed error is tol * (1 + factor) playback seconds")
    p.add_argument("--head", type=float, default=0.0, help="source seconds at the start that Photos plays at 1x (the ramp head)")
    p.add_argument("--tail-from", type=float, default=None, help="source second from which Photos plays at 1x again (the ramp tail)")
    p.add_argument("--offset", type=float, default=0.0, help="playback seconds a trim cut off the start of the export")
    a = p.parse_args(argv)
    r = photos.compare_tries(*(json.loads(Path(f).read_text()) for f in (a.original, a.export)), factor=a.factor, tol=a.tol,
                             head_s=a.head, tail_from_s=a.tail_from, offset_s=a.offset)
    for t in r["tries"]:
        print(f"try {t['try_']}: original x{r['factor']:g} {t['original_playback']}  export {t['export']}  start err {t['start_err_s']} s  end err {t['end_err_s']} s")
    print(f"{'PASS' if r['ok'] else 'FAIL'}: {r['n_original']} tries in the original, {r['n_export']} in the export, order kept, "
          f"largest boundary error {r['max_err_playback_s']} playback s (allowed {r['tolerance_playback_s']})")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
