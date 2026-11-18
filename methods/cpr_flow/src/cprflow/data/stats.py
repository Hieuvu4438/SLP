"""Train-only equal-clip, equal-dataset population moments."""

from pathlib import Path

import numpy as np
import torch

from cprflow.data.cache import digest, save_numeric, file_hash


def balanced_moments(records, datasets, std_floor=0.01, split="train"):
    if split != "train":
        raise ValueError("Statistics may only be fitted on train")
    accum = {d: [None, None, 0] for d in datasets}
    for row, values in records:
        if row["split"] != "train":
            raise ValueError(f"Statistics leakage: {row['uid']}")
        if row["dataset_id"] not in accum:
            raise ValueError("Unexpected dataset in joint statistics")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim < 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Invalid statistics input")
        state = accum[row["dataset_id"]]
        mean, second = x.mean(0), (x * x).mean(0)
        if state[0] is None:
            state[0], state[1] = mean, second
        else:
            state[0] += mean
            state[1] += second
        state[2] += 1
    if any(state[2] == 0 for state in accum.values()):
        raise ValueError("Every configured dataset needs included train clips")
    mean = np.mean([state[0] / state[2] for state in accum.values()], 0)
    second = np.mean([state[1] / state[2] for state in accum.values()], 0)
    return (
        mean.astype("float32"),
        np.sqrt(np.maximum(second - mean * mean, 0)).clip(std_floor).astype("float32"),
        {d: state[2] for d, state in accum.items()},
    )


class Normalizer(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError("Invalid normalization statistics")

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["mean"], data["std"])


def save_stats(path, mean, std, provenance):
    import json

    save_numeric(path, mean=mean, std=std)
    Path(str(path) + ".json").write_text(
        json.dumps(
            {**provenance, "stats_key": digest(provenance), "array_sha256": file_hash(path)},
            indent=2,
        )
    )
