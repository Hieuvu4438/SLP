"""d2c_sign - one-line description of the idea."""

from __future__ import annotations

from slp.data import Split
from slp.method import BaseMethod


class Method(BaseMethod):
    def fit(self, train: Split, val: Split) -> None:
        # self.hparams -> method hyper-parameters from config.yaml / CLI
        # self.ctx.log_metrics({...}, step=epoch) -> training curves
        # self.ctx.ckpt_dir -> where to save checkpoints
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError
