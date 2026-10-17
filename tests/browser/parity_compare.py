"""Pure-numpy side of the browser parity check (tests/browser/test_models_parity.py), kept free of onnxruntime, Pillow and a browser so
tests/test_parity_compare.py can run it everywhere.

The browser sends every output value (not a sum or a prefix): a sum of 27300 logits with a 1e-3 tolerance lets one detector query move by the
whole tolerance, and the first six values only cover query 0. Each element is compared on its own, and for a detector's class logits
(last dimension 91) the sigmoid scores, which are what the clip logic thresholds, are compared per element and by their top-k."""
from __future__ import annotations

import numpy as np

REL = 1e-3  # per element, against max(|reference|, the output's mean |value|)
SCORE_ABS = 2e-3  # sigmoid scores live in 0..1
TOPK = 20
N_CLASSES = 91


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -60, 60)))


def compare(ref: dict, got: dict) -> float:
    """Assert agreement element by element and return the worst relative error. `ref[name]["values"]` and `got["outputs"][name]["values"]`
    are the flat outputs; an element is compared against max(|reference|, the output's mean |value|), so a value near zero does not turn
    rounding noise into a failure."""
    assert set(ref) == set(got["outputs"]), (sorted(ref), sorted(got["outputs"]))
    worst = 0.0
    for name, r in ref.items():
        g = got["outputs"][name]
        a = np.asarray(g["values"], np.float64)
        b = np.asarray(r["values"], np.float64)
        assert a.size == b.size == r["n"] == g["n"], (name, a.size, b.size, r["n"], g["n"])
        assert np.all(np.isfinite(a)), f"{name}: the browser returned non-finite values"
        scale = float(np.abs(b).mean())
        err = np.abs(a - b) / np.maximum(np.abs(b), scale)
        i = int(err.argmax())
        worst = max(worst, float(err[i]))
        assert err[i] <= REL, f"{name} element {i}: browser {a[i]} vs reference {b[i]} (relative {err[i]:.2e})"
        if tuple(g["dims"])[-1:] == (N_CLASSES,):  # class logits: the scores that decide clips
            sa, sb = sigmoid(a), sigmoid(b)
            d = np.abs(sa - sb)
            j = int(d.argmax())
            assert d[j] <= SCORE_ABS, f"{name} score {j}: browser {sa[j]:.5f} vs reference {sb[j]:.5f}"
            ta, tb = np.sort(sa)[::-1][:TOPK], np.sort(sb)[::-1][:TOPK]
            assert np.abs(ta - tb).max() <= SCORE_ABS, f"{name} top-{TOPK} scores {ta[:5]} vs {tb[:5]}"
    return worst
