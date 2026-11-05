"""Dataset registry. Add a loader: ``DATASETS["name"] = fn(cfg) -> DatasetSplits``
and a matching ``configs/data/<name>.yaml``."""

from __future__ import annotations

from typing import Callable

from slp.data.base import DatasetSplits, Split
from slp.data.sign_smplx import load_sign_smplx
from slp.data.toy import load_toy

DATASETS: dict[str, Callable[[dict], DatasetSplits]] = {
    "toy": load_toy,
    "phoenix_smplx": load_sign_smplx,
    "csl_smplx": load_sign_smplx,
    "how2sign_smplx": load_sign_smplx,
}


def load_dataset(cfg: dict) -> DatasetSplits:
    name = cfg["data"]["name"]
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Registered: {sorted(DATASETS)}")
    return DATASETS[name](cfg)


__all__ = ["DATASETS", "DatasetSplits", "Split", "load_dataset"]
