import pytest

from slp.evaluation import evaluate
from slp.evaluation.metrics import accuracy, macro_f1


def test_accuracy():
    assert accuracy([0, 1, 1, 2], [0, 1, 2, 2]) == pytest.approx(0.75)


def test_macro_f1_perfect_and_known():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    # class0: tp=1 fp=0 fn=1 -> 2/3 ; class1: tp=1 fp=1 fn=0 -> 2/3
    assert macro_f1([0, 1, 1], [0, 0, 1]) == pytest.approx(2 / 3)


def test_evaluate_length_mismatch():
    with pytest.raises(ValueError):
        evaluate([0, 1], [0], ["accuracy"])


# --------------------------------------------------------------- SLP metrics
import numpy as np  # noqa: E402

from slp.evaluation.metrics import (  # noqa: E402
    _pairwise_jpe, dtw_cost, dtw_jpe, dtw_jpe_norm, sibleu4, sibleu_unigram)


def _seq(T, J=4, seed=0):
    return np.random.default_rng(seed).normal(size=(T, J, 3))


def test_dtw_identical_is_zero():
    x = _seq(10)
    assert dtw_jpe([x], [x]) == pytest.approx(0.0)
    assert dtw_jpe_norm([x], [x]) == pytest.approx(0.0)


def test_dtw_root_translation_invariant():
    x = _seq(8)
    assert dtw_jpe([x + np.array([1.0, -2.0, 3.0])], [x]) == pytest.approx(0.0, abs=1e-9)


def test_dtw_cost_known_example():
    # optimal path (0,0)->(1,1)->(2,1): 0 + 0 + 0 = 0 with 3 steps
    cost = np.array([[0.0, 5.0], [5.0, 0.0], [9.0, 0.0]])
    acc, n = dtw_cost(cost)
    assert acc == pytest.approx(0.0) and n == 3
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    acc, n = dtw_cost(cost)  # diagonal 1+4=5 (vs 1+2+4, 1+3+4)
    assert acc == pytest.approx(5.0) and n == 2


def test_dtw_norm_equals_accumulated_over_path():
    x, y = _seq(6, seed=1), _seq(9, seed=2)
    acc, n = dtw_cost(_pairwise_jpe(x, y))
    assert dtw_jpe([x], [y]) == pytest.approx(acc)
    assert dtw_jpe_norm([x], [y]) == pytest.approx(acc / n)
    assert max(len(x), len(y)) <= n <= len(x) + len(y) - 1


def test_dtw_accumulated_scales_with_length_norm_does_not():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 4, 3))
    off = np.zeros((1, 4, 3)); off[0, 1:, 0] = 0.1  # constant 7.5 cm offset on 3 of 4 joints
    short_t, long_t = np.repeat(base, 10, 0), np.repeat(base, 40, 0)
    short_p, long_p = short_t + off, long_t + off
    assert dtw_jpe([long_p], [long_t]) == pytest.approx(4 * dtw_jpe([short_p], [short_t]))
    assert dtw_jpe_norm([long_p], [long_t]) == pytest.approx(dtw_jpe_norm([short_p], [short_t]))


def test_sibleu_unigram_matches_madis_formula_and_is_order_free():
    ref = [[1, 2, 3, 4]]
    assert sibleu_unigram([[1, 2, 3, 4]], ref) == pytest.approx(100.0)
    assert sibleu_unigram([[4, 3, 2, 1]], ref) == pytest.approx(100.0)  # order ignored
    # hyp shorter: overlap 2 / hyp 2, BP = exp(1 - 4/2)
    assert sibleu_unigram([[1, 9]], ref) == pytest.approx(100 * np.exp(-1) * 0.5)


def test_sibleu4_is_order_sensitive():
    ref = [[1, 2, 3, 4, 5, 6]]
    assert sibleu4([[1, 2, 3, 4, 5, 6]], ref) == pytest.approx(100.0)
    assert sibleu4([[6, 5, 4, 3, 2, 1]], ref) == pytest.approx(0.0)
