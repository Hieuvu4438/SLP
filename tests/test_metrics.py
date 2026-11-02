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
