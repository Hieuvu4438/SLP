"""Deterministic logical batches: corpus first, bucket by included clip counts."""

import numpy as np


class BalancedBatchSampler:
    def __init__(
        self,
        rows,
        batch_size,
        seed=17,
        datasets=None,
        probabilities=None,
        boundaries=(128, 256, 512, 1024),
    ):
        self.rows, self.batch_size, self.seed = rows, batch_size, seed
        self.datasets = list(datasets or sorted({r["dataset_id"] for r in rows}))
        self.probabilities = probabilities or [1 / len(self.datasets)] * len(self.datasets)
        self.buckets = {}
        for d in self.datasets:
            bins = {}
            for i, r in enumerate(rows):
                if r["dataset_id"] == d:
                    bucket = next((b for b in boundaries if r["frames"] <= b), None)
                    if bucket is None:
                        raise ValueError("Long train clip must be in explicit exclusion ledger")
                    bins.setdefault(bucket, []).append(i)
            if not bins:
                raise ValueError(f"No included train clips in {d}")
            self.buckets[d] = list(bins.values())
        self.cursor = 0

    def plan(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        dataset = rng.choice(self.datasets, p=self.probabilities)
        buckets = self.buckets[dataset]
        sizes = np.array([len(x) for x in buckets])
        indices = buckets[rng.choice(len(buckets), p=sizes / sizes.sum())]
        result = rng.choice(
            indices, self.batch_size, replace=len(indices) < self.batch_size
        ).tolist()
        return result

    def batch(self, step=None):
        step = self.cursor if step is None else step
        result = self.plan(step)
        self.cursor = step + 1
        return result

    def state_dict(self):
        return {"cursor": self.cursor, "seed": self.seed, "batch_size": self.batch_size}

    def load_state_dict(self, state):
        if state["seed"] != self.seed or state["batch_size"] != self.batch_size:
            raise ValueError("Sampler resume mismatch")
        self.cursor = state["cursor"]

    @staticmethod
    def shard(indices, rank, world_size):
        if len(indices) % world_size:
            raise ValueError("Global batch must divide world size")
        return indices[rank::world_size]
