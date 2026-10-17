#!/usr/bin/env python3
"""Synthetic music for tests and number checks: no real recording, so no license and no footage involved.

    uv run python tools/synth_music.py OUT.wav --bpm 124 --seconds 90 [--offset 0.8] [--kind click|beat]

`click`: one short 1 kHz click on every beat (the truth is exact). `beat`: kick on every beat (accented every 4th), a hat on
the off-beats and a soft chord, so the tracker sees something closer to music. Prints the beat times' first/last values.
"""
from __future__ import annotations

import argparse

from inframe.synthbeat import SR, render, write_wav  # noqa: F401  (re-exported for the tests)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--kind", choices=("click", "beat"), default="beat")
    a = ap.parse_args()
    y, beats = render(a.bpm, a.seconds, a.offset, a.kind)
    write_wav(a.out, y)
    print(f"{a.out}: {a.seconds:g} s, {a.bpm:g} BPM, {len(beats)} beats, first {beats[0]:.3f} s, last {beats[-1]:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
