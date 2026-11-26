"""Train-only text retrieval for the SignRAE memory (PLAN_V3 C4).

Keys are TF-IDF vectors of character n-grams of the normalised sentence (works for German,
Chinese and English without a tokenizer, and does not depend on the fine-tuned text encoder,
whose features drift during training). The score of a candidate is

    cos(q, c) - length_weight * |log(1 + n_q) - log(1 + n_c)|

with n = number of characters of the normalised sentence: a *text* length term (never the GT
motion length, which is unknown at test time). During training a query never retrieves itself
or any clip whose normalised sentence is identical (CSL repeats each sentence ~2.8 times while
dev/test sentences are almost always new), so training sees the same conditions as test.
Everything is stored as tensors / lists so that it survives ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np
import torch
from scipy import sparse

_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Case-folded NFKC text with punctuation removed and whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(ch if not unicodedata.category(ch).startswith("P") else " " for ch in text)
    return _SPACE.sub(" ", text).strip()


def char_ngrams(text: str, low: int, high: int) -> list[str]:
    padded = f" {text} "
    return [padded[i:i + n] for n in range(low, high + 1) for i in range(len(padded) - n + 1)]


class TextIndex:
    """Sublinear-TF x smoothed-IDF character n-gram index over the train sentences."""

    def __init__(self, texts: list[str], ngram: tuple[int, int] = (2, 4),
                 length_weight: float = 0.1, min_df: int = 1):
        if not texts:
            raise ValueError("TextIndex needs at least one train sentence")
        self.low, self.high = ngram
        if not 1 <= self.low <= self.high:
            raise ValueError("ngram must satisfy 1 <= low <= high")
        self.length_weight = float(length_weight)
        self.norm = [normalise(t) for t in texts]
        counts = [Counter(char_ngrams(t, self.low, self.high)) for t in self.norm]
        df = Counter(g for c in counts for g in c)
        self.vocab = sorted(g for g, n in df.items() if n >= min_df)
        self.lookup = {g: i for i, g in enumerate(self.vocab)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[g])) + 1.0 for g in self.vocab],
                            dtype=np.float32)
        self.matrix = self._vectorise_counts(counts)
        self.lengths = np.log1p([len(t) for t in self.norm]).astype(np.float32)
        # Group id per normalised sentence: queries never retrieve their own group in training.
        groups: dict[str, int] = {}
        self.group = np.array([groups.setdefault(t, len(groups)) for t in self.norm])
        self.group_of = groups

    def _vectorise_counts(self, counts: list[Counter]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for r, c in enumerate(counts):
            for g, k in c.items():
                j = self.lookup.get(g)
                if j is not None:
                    rows.append(r)
                    cols.append(j)
                    vals.append((1.0 + math.log(k)) * self.idf[j])
        m = sparse.csr_matrix((vals, (rows, cols)), shape=(len(counts), len(self.vocab)),
                              dtype=np.float32)
        norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
        norms[norms == 0] = 1.0
        return sparse.diags(1.0 / norms) @ m

    def vectorise(self, texts: list[str]) -> sparse.csr_matrix:
        return self._vectorise_counts([Counter(char_ngrams(normalise(t), self.low, self.high))
                                       for t in texts])

    def scores(self, texts: list[str]) -> np.ndarray:
        """[Q, N] retrieval scores of query sentences against every train sentence."""
        q = self.vectorise(texts)
        cos = (q @ self.matrix.T).toarray()
        qlen = np.log1p([len(normalise(t)) for t in texts]).astype(np.float32)
        return cos - self.length_weight * np.abs(qlen[:, None] - self.lengths[None, :])

    def query(self, texts: list[str], k: int, exclude_groups: bool = False,
              query_ids: list[int] | None = None, chunk: int = 512) -> np.ndarray:
        """[Q, k] indices of the best train clips (-1 if fewer candidates remain).

        ``query_ids`` = train indices of TRAIN queries: a train query never retrieves itself, and
        with ``exclude_groups`` also none of its verbatim repeats (its sentence group). Dev / test
        queries pass ``query_ids=None``."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if exclude_groups and query_ids is None:
            raise ValueError("exclude_groups needs the train indices of the queries")
        out = np.full((len(texts), k), -1, dtype=np.int64)
        for start in range(0, len(texts), chunk):
            s = self.scores(texts[start:start + chunk])
            if query_ids is not None:
                ids = np.asarray(query_ids[start:start + chunk])
                s[np.arange(len(ids)), ids] = -np.inf  # never the query clip itself
                if exclude_groups:
                    own = self.group[ids]
                    s[self.group[None, :] == own[:, None]] = -np.inf
            # Stable tie-break by index: argsort of (-score, index).
            order = np.argsort(-s, axis=1, kind="stable")[:, :k]
            picked = np.take_along_axis(s, order, axis=1)
            order[~np.isfinite(picked)] = -1
            out[start:start + len(s), :order.shape[1]] = order
        return out

    def duplicate_of_train(self, texts: list[str]) -> np.ndarray:
        """Boolean per query: its normalised sentence occurs verbatim in train (report subsets)."""
        return np.array([normalise(t) in self.group_of for t in texts])

    # ---------------------------------------------------------------- persistence
    def state(self) -> dict:
        m = self.matrix.tocoo()
        return {"ngram": [self.low, self.high], "length_weight": self.length_weight,
                "vocab": list(self.vocab), "idf": torch.from_numpy(self.idf),
                "rows": torch.from_numpy(m.row.astype(np.int64)),
                "cols": torch.from_numpy(m.col.astype(np.int64)),
                "vals": torch.from_numpy(m.data.astype(np.float32)),
                "shape": list(m.shape), "lengths": torch.from_numpy(self.lengths),
                "norm": list(self.norm)}

    @classmethod
    def from_state(cls, state: dict) -> TextIndex:
        self = cls.__new__(cls)
        self.low, self.high = state["ngram"]
        self.length_weight = float(state["length_weight"])
        self.vocab = list(state["vocab"])
        self.lookup = {g: i for i, g in enumerate(self.vocab)}
        self.idf = state["idf"].numpy()
        self.matrix = sparse.csr_matrix(
            (state["vals"].numpy(), (state["rows"].numpy(), state["cols"].numpy())),
            shape=tuple(state["shape"]))
        self.lengths = state["lengths"].numpy()
        self.norm = list(state["norm"])
        groups: dict[str, int] = {}
        self.group = np.array([groups.setdefault(t, len(groups)) for t in self.norm])
        self.group_of = groups
        return self


def gather_memory(bank: list[torch.Tensor], indices: torch.Tensor, max_frames: int,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieved latents [B, k, L, d] and valid mask [B, k, L] from a list of train latents.
    Index -1 gives an empty (all-invalid) slot. Clips longer than ``max_frames`` are truncated."""
    b, k = indices.shape
    d = bank[0].shape[-1]
    lengths = [[0 if int(i) < 0 else min(len(bank[int(i)]), max_frames) for i in row]
               for row in indices.tolist()]
    length = max(1, max(max(row) for row in lengths))
    out = torch.zeros(b, k, length, d)
    valid = torch.zeros(b, k, length, dtype=torch.bool)
    for r, row in enumerate(indices.tolist()):
        for s, i in enumerate(row):
            if i >= 0:
                n = lengths[r][s]
                out[r, s, :n] = bank[i][:n].float()
                valid[r, s, :n] = True
    return out.to(device), valid.to(device)
