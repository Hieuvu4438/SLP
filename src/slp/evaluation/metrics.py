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


# ---------------------------------------------------------------------------
# Sign Language Production metrics (SOKE / MaDiS protocol + calibrated variants)
#
# Predictions/targets are lists of per-sequence joint arrays [T, J, 3] (already
# forward-kinematised and restricted to one body part). Lengths may differ
# between prediction and target. Joint 0 of each array is the alignment root
# (pelvis for body, wrist for hands), as in SOKE/MaDiS.
# ---------------------------------------------------------------------------


def _pairwise_jpe(gen: np.ndarray, ref: np.ndarray, align_idx: int = 0) -> np.ndarray:
    """[Tg, Tr] mean joint position error after root-translation alignment."""
    gen = np.asarray(gen, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    g = gen - gen[:, align_idx:align_idx + 1]
    r = ref - ref[:, align_idx:align_idx + 1]
    diff = g[:, None] - r[None]
    return np.sqrt((diff ** 2).sum(-1)).mean(-1)


def dtw_cost(cost: np.ndarray) -> tuple[float, int]:
    """Exact DTW on a cost matrix. Returns (accumulated cost, warping-path length)."""
    cost = np.asarray(cost, dtype=np.float64)
    r, c = cost.shape
    if r == 0 or c == 0:
        raise ValueError("DTW inputs must contain at least one frame")
    acc = np.full((r + 1, c + 1), np.inf)
    ln = np.zeros((r + 1, c + 1))
    acc[0, 0] = 0.0
    for d in range(2, r + c + 1):
        i = np.arange(max(1, d - c), min(r, d - 1) + 1)
        j = d - i
        cand = np.stack([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        lens = np.stack([ln[i - 1, j - 1], ln[i - 1, j], ln[i, j - 1]])
        best = cand.argmin(0)
        acc[i, j] = cost[i - 1, j - 1] + cand[best, np.arange(len(i))]
        ln[i, j] = lens[best, np.arange(len(i))] + 1
    return float(acc[r, c]), int(ln[r, c])


def dtw_jpe(preds, targets) -> float:
    """SOKE/MaDiS DTW-JPE: accumulated DTW cost, averaged over sequences (paper-compatible)."""
    return float(np.mean([dtw_cost(_pairwise_jpe(p, t))[0] for p, t in zip(preds, targets)]))


def dtw_jpe_norm(preds, targets) -> float:
    """Calibrated DTW-JPE: accumulated cost divided by warping-path length (per-step error, m)."""
    vals = []
    for p, t in zip(preds, targets):
        a, n = dtw_cost(_pairwise_jpe(p, t))
        vals.append(a / n)
    return float(np.mean(vals))


def _ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def sibleu_unigram(preds, targets) -> float:
    """MaDiS released-code SiBLEU: corpus unigram clipped precision x brevity penalty
    over sign-token IDs (order-insensitive). preds/targets: lists of token-id lists."""
    from collections import Counter
    overlap = hyp = ref = 0
    for p, t in zip(preds, targets):
        overlap += sum((Counter(p) & Counter(t)).values())
        hyp += len(p)
        ref += len(t)
    if hyp == 0:
        return 0.0
    bp = 1.0 if hyp > ref else float(np.exp(1 - ref / hyp))
    return 100.0 * bp * overlap / hyp


def sibleu4(preds, targets) -> float:
    """Order-sensitive corpus BLEU-4 over sign-token IDs (uniform weights, no smoothing)."""
    from collections import Counter
    match = np.zeros(4)
    total = np.zeros(4)
    hyp = ref = 0
    for p, t in zip(preds, targets):
        hyp += len(p)
        ref += len(t)
        for n in range(1, 5):
            pc, tc = Counter(_ngrams(p, n)), Counter(_ngrams(t, n))
            match[n - 1] += sum((pc & tc).values())
            total[n - 1] += max(len(p) - n + 1, 0)
    if hyp == 0 or np.any(match == 0):
        return 0.0
    bp = 1.0 if hyp > ref else float(np.exp(1 - ref / hyp))
    return 100.0 * bp * float(np.exp(np.mean(np.log(match / total))))


METRICS: dict[str, Callable] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "dtw_jpe": dtw_jpe,
    "dtw_jpe_norm": dtw_jpe_norm,
    "sibleu_unigram": sibleu_unigram,
    "sibleu4": sibleu4,
}
