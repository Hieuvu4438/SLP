"""text_retrieval - trivial SLP baselines for calibrating DTW metrics (docs/proposal7 E0.4).

mode: retrieval  -> the motion of the most similar *training* sentence (character 1-3-gram TF-IDF,
                    cosine similarity; ties between identical training sentences are broken at random
                    with the run seed). A text-aware model must beat this floor.
mode: mean_pose  -> the mean pose (0 in the normalised SOKE feature space) for every frame. Any
                    model that does not clearly beat it on DTW is regressing to the mean.

Output lengths: the retrieved motion's own length, or the mean training length for mean_pose;
``lengths_file`` (JSON {name: frames}, e.g. from methods/signmflow_plus/tools/lengths.py) fixes them
for both modes (the retrieved motion is then resampled linearly in time).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from slp.data import Split
from slp.method import BaseMethod

ROOT = Path(__file__).resolve().parents[2]


def resample(seq: np.ndarray, length: int) -> np.ndarray:
    """Linear time resampling of [T,D] to [length,D]."""
    if len(seq) == length:
        return seq
    src = np.linspace(0, len(seq) - 1, length)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(seq) - 1)
    w = (src - i0)[:, None]
    return ((1 - w) * seq[i0] + w * seq[i1]).astype(seq.dtype)


class Method(BaseMethod):
    def fit(self, train: Split, val: Split) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.mode = self.hparams.get("mode", "retrieval")
        if self.mode not in ("retrieval", "mean_pose"):
            raise ValueError(f"mode={self.mode!r}: expected retrieval | mean_pose")
        self.train = train
        self.texts = [x["text"] for x in train.inputs]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True)
        self.X = self.vec.fit_transform(self.texts)
        self.mean_len = int(round(np.mean([len(t) for t in train.targets]))) if self.mode == "mean_pose" else None
        self.ctx.logger.info(f"text_retrieval mode={self.mode}: {len(self.texts)} training sentences")

    def _lengths(self, inputs):
        p = self.hparams.get("lengths_file")
        if not p:
            return None
        p = Path(os.path.expandvars(str(p)))
        table = json.load(open(p if p.is_absolute() else ROOT / p))
        return [int(table.get(x["name"], table.get(x["name"].replace("/", "__")))) for x in inputs]

    def predict(self, inputs):
        rng = np.random.default_rng(int(self.ctx.seed))
        fixed = self._lengths(inputs)
        out = []
        if self.mode == "mean_pose":
            dim = np.asarray(self.train.targets[0]).shape[-1]
            for i in range(len(inputs)):
                out.append(np.zeros((fixed[i] if fixed else self.mean_len, dim), np.float32))
            return out
        sims = (self.vec.transform([x["text"] for x in inputs]) @ self.X.T).toarray()
        for i, row in enumerate(sims):
            best = np.flatnonzero(row >= row.max() - 1e-9)
            j = int(rng.choice(best))
            seq = np.asarray(self.train.targets[j], np.float32)
            out.append(resample(seq, fixed[i]) if fixed else seq)
        return out
