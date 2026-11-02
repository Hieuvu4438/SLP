"""Single evaluation entry point used by every run."""

from __future__ import annotations

from slp.evaluation.metrics import METRICS


def evaluate(preds, targets, metric_names: list[str]) -> dict[str, float]:
    unknown = set(metric_names) - set(METRICS)
    if unknown:
        raise KeyError(f"Unknown metrics {sorted(unknown)}. Available: {sorted(METRICS)}")
    if len(preds) != len(targets):
        raise ValueError(f"len(preds)={len(preds)} != len(targets)={len(targets)}")
    return {m: METRICS[m](preds, targets) for m in metric_names}


__all__ = ["METRICS", "evaluate"]
