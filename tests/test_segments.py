import pytest

from inframe.segments import build_segments, drop_short, hysteresis, merge_gaps, pad, sample_edges


def test_edges_are_midpoints_and_clamped():
    assert sample_edges([0.0, 1.0, 2.0], 2.5) == [0.0, 0.5, 1.5, 2.5]
    assert sample_edges([0.2, 0.4], 0.5) == pytest.approx([0.1, 0.3, 0.5])
    assert sample_edges([], 5) == []


def test_hysteresis_basic_and_boundaries():
    t = [0, 1, 2, 3, 4, 5]
    s = [0, 0.6, 0.9, 0.1, 0, 0]
    assert hysteresis(t, s, 0.5, 0.3, 5.5) == [(0.5, 2.5)]


def test_hysteresis_holds_between_exit_and_enter():
    t = list(range(6))
    s = [0.0, 0.6, 0.4, 0.4, 0.6, 0.0]  # dips to 0.4: above exit, so no split
    assert hysteresis(t, s, 0.5, 0.3, 5.5) == [(0.5, 4.5)]
    s2 = [0.0, 0.6, 0.2, 0.4, 0.6, 0.0]  # drops below exit: split, and 0.4 alone cannot re-enter
    assert hysteresis(t, s2, 0.5, 0.3, 5.5) == [(0.5, 1.5), (3.5, 4.5)]


def test_hysteresis_open_at_start_and_end():
    t = [0.0, 1.0, 2.0]
    assert hysteresis(t, [0.9, 0.9, 0.9], 0.5, 0.3, 2.4) == [(0.0, 2.4)]
    assert hysteresis(t, [0.1, 0.1, 0.1], 0.5, 0.3, 2.4) == []


def test_hysteresis_validates():
    with pytest.raises(ValueError):
        hysteresis([0, 1], [0.5], 0.5, 0.3, 2)
    with pytest.raises(ValueError):
        hysteresis([0, 1], [0.5, 0.5], 0.3, 0.5, 2)


def test_merge_gaps():
    assert merge_gaps([(0, 1), (2, 3), (10, 11)], 1.0) == [(0, 3), (10, 11)]
    assert merge_gaps([(0, 1), (2, 3)], 0.9) == [(0, 1), (2, 3)]
    assert merge_gaps([(5, 6), (0, 4)], 1.0) == [(0, 6)]  # unsorted input, touching within gap
    assert merge_gaps([(0, 10), (2, 3)], 0) == [(0, 10)]  # contained


def test_drop_short():
    assert drop_short([(0, 0.5), (1, 3)], 1.0) == [(1, 3)]
    assert drop_short([(0, 1.0)], 1.0) == [(0, 1.0)]


def test_pad_clamps_and_merges_overlaps():
    assert pad([(0.5, 2), (9, 10)], 1, 1, 10.5) == [(0, 3), (8, 10.5)]
    assert pad([(2, 3), (4.5, 6)], 1, 1, 100) == [(1, 7)]  # padded ranges touch/overlap
    assert pad([], 1, 1, 10) == []


def test_build_segments_end_to_end():
    t = [i * 0.2 for i in range(100)]  # 20 s at 5 fps
    s = [0.0] * 100
    for i in range(10, 20): s[i] = 0.9      # 2 s stretch
    for i in range(24, 30): s[i] = 0.8      # 1.2 s, 0.8 s after the first: merged (gap <= 1.5)
    s[60] = 0.9                             # single sample: 0.2 s, dropped by min_len
    presence, clips = build_segments(t, s, 20.0)
    assert presence == [(pytest.approx(1.9), pytest.approx(5.9))]
    assert clips == [(pytest.approx(0.9), pytest.approx(6.9))]


def test_build_segments_nothing_found():
    assert build_segments([0, 1, 2], [0, 0, 0], 3.0) == ([], [])


def test_clamp_to_the_first_picture_of_a_video_that_starts_late():
    t = [2.0, 2.5, 3.0, 3.5, 4.0]
    assert sample_edges(t, 6.0, 2.0)[0] == 2.0 and sample_edges(t, 6.0)[0] == 1.75
    assert sample_edges([2.1], 6.0, 2.0)[0] == 2.0
    assert pad([(2.2, 3)], 1, 1, 10, 2.0) == [(2.0, 4)]
    presence, clips = build_segments(t, [0.9] * 5, 6.0, lo=2.0)
    assert presence[0][0] == 2.0 and clips[0][0] == 2.0
