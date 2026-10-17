"""The browser-vs-python parity comparison (tests/browser/parity_compare.py) must catch a wrong detector score. It is pure numpy, so this
runs without Chrome or the models."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "browser"))
from parity_compare import compare  # noqa: E402

DIMS = (1, 300, 91)


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.normal(-6.0, 3.0, 300 * 91)
    ref[5 * 91 + 1] = -3.645  # a query whose person logit is low (p = 0.026)
    return {"logits": {"values": ref, "n": ref.size}}, ref


def _got(values, dims=DIMS):
    return {"outputs": {"logits": {"values": list(values), "n": len(values), "dims": dims}}}


def test_identical_and_rounding_noise_pass():
    ref, v = _pair()
    assert compare(ref, _got(v)) == 0.0
    assert compare(ref, _got(v * (1 + 2e-4))) < 1e-3


def test_one_query_score_moved_by_the_old_sum_tolerance_fails():
    """The old check summed all values with a tolerance of 1e-3 of the sum of |values| (about 166 here), so one logit could move to p = 1."""
    ref, v = _pair()
    bad = v.copy()
    bad[5 * 91 + 1] = 160.0
    assert abs(bad.sum() - v.sum()) < 1e-3 * np.abs(v).sum()  # what the sum comparison let through
    with pytest.raises(AssertionError):
        compare(ref, _got(bad))


def test_a_change_outside_the_first_six_values_fails():
    ref, v = _pair()
    bad = v.copy()
    bad[12345] += 0.5
    with pytest.raises(AssertionError):
        compare(ref, _got(bad))


def test_a_reordered_top_k_fails_even_when_the_sum_is_unchanged():
    ref, v = _pair()
    bad = v.copy()
    bad[[0, 91 * 7]] = bad[[91 * 7, 0]]
    assert abs(bad.sum() - v.sum()) < 1e-9
    with pytest.raises(AssertionError):
        compare(ref, _got(bad))


def test_non_finite_or_missing_output_fails():
    ref, v = _pair()
    bad = v.copy()
    bad[3] = float("nan")
    with pytest.raises(AssertionError):
        compare(ref, _got(bad))
    with pytest.raises(AssertionError):
        compare(ref, {"outputs": {}})
