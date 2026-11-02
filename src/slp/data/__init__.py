"""Dataset registry. Add a loader: ``DATASETS["name"] = fn(cfg) -> DatasetSplits``
and a matching ``configs/data/<name>.yaml``."""

from __future__ import annotations

from typing import Callable

from slp.data.base import DatasetSplits, Split
from slp.data.toy import load_toy

DATASETS: dict[str, Callable[[dict], DatasetSplits]] = {
    "toy": load_toy,
}


def load_dataset(cfg: dict) -> DatasetSplits:
    name = cfg["data"]["name"]
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Registered: {sorted(DATASETS)}")
    return DATASETS[name](cfg)


__all__ = ["DATASETS", "DatasetSplits", "Split", "load_dataset"]
