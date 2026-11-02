"""Synthetic toy dataset - ONLY for smoke tests / checking the pipeline.
Replace with real loaders for the research domain."""

from __future__ import annotations

import numpy as np

from slp.data.base import DatasetSplits, Split


def load_toy(cfg: dict) -> DatasetSplits:
    d = cfg["data"]
    rng = np.random.default_rng(d.get("split_seed", 0))  # fixed, independent of run seed
    n, dim, k = d.get("n_samples", 600), d.get("dim", 8), d.get("n_classes", 3)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    w = rng.normal(size=(dim, k))
    y = (x @ w).argmax(axis=1)
    idx = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    parts = np.split(idx, [n_tr, n_tr + n_va])
    tr, va, te = (Split(x[p], y[p]) for p in parts)
    return DatasetSplits(tr, va, te, info={"n_classes": k, "dim": dim})
