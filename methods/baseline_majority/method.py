"""Majority-class baseline. Sanity check: every real method must beat this."""

from __future__ import annotations

import numpy as np

from slp.data import Split
from slp.method import BaseMethod


class Method(BaseMethod):
    def fit(self, train: Split, val: Split) -> None:
        values, counts = np.unique(np.asarray(train.targets), return_counts=True)
        self.majority = values[counts.argmax()]
        self.ctx.logger.info(f"majority class = {self.majority}")

    def predict(self, inputs):
        return np.full(len(inputs), self.majority)
