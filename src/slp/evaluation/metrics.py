"""Metric implementations. ONE implementation per metric, shared by all
methods. Each metric: fn(predictions, targets) -> float.
Placeholder metrics below - replace/extend for the actual domain, and add a
unit test for each new metric in tests/test_metrics.py."""

from __future__ import annotations

from typing import Callable

import numpy as np


def accuracy(preds, targets) -> float:
    preds, targets = np.asarray(preds), np.asarray(targets)
    return float((preds == targets).mean())


def macro_f1(preds, targets) -> float:
    preds, targets = np.asarray(preds), np.asarray(targets)
    f1s = []
    for c in np.unique(targets):
        tp = np.sum((preds == c) & (targets == c))
        fp = np.sum((preds == c) & (targets != c))
        fn = np.sum((preds != c) & (targets == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


METRICS: dict[str, Callable] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
}
