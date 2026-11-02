"""The interface every method must implement.

Contract (this is what keeps comparisons honest):
  * ``fit`` receives train and val splits (with targets).
  * ``predict`` receives ONLY inputs - a method never sees evaluation targets.
  * Metrics are computed by ``slp.evaluation``, never inside a method.

If your domain needs a different shape (e.g. generative models, RL), change
this interface ONCE here, so all methods stay comparable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from slp.data.base import Split
from slp.tracking.run import RunContext


class BaseMethod(ABC):
    def __init__(self, cfg: dict, ctx: RunContext):
        self.cfg = cfg                       # full resolved config
        self.hparams = cfg.get("method", {})  # this method's hyper-parameters
        self.ctx = ctx                       # run dir, seed, logger, metric logging

    @abstractmethod
    def fit(self, train: Split, val: Split) -> None:
        """Train / fit the method. Use ``val`` for model selection only."""

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Return predictions for ``inputs`` (no targets available here)."""

    # Optional hooks -------------------------------------------------------
    def save(self) -> None:
        """Persist checkpoints into ``self.ctx.ckpt_dir`` if needed."""

    def load(self, ckpt_dir) -> None:
        """Restore from ``ckpt_dir`` (used by evaluate-only runs)."""
