"""Explicit execution-only migration; scientific config/dependencies remain immutable."""

import copy
import datetime
import os
from pathlib import Path

import torch

from cprflow.config import validate
from cprflow.data.cache import file_hash
from cprflow.jobs.schema import atomic_json
from cprflow.training.checkpoint import atomic_save, load_checkpoint

ALLOWED = {
    "ae.execution",
    "train.microbatch_per_rank",
    "train.accumulate",
    "geometry.fk_chunk_frames",
}


def flattened(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}".strip(".")
        if isinstance(item, dict):
            result.update(flattened(item, name))
        else:
            result[name] = item
    return result


def config_changes(old, new):
    old = copy.deepcopy(old)
    old["ae"].setdefault("execution", "per_clip")
    a, b = flattened(old), flattened(new)
    changes = {
        k: {"old": a.get(k), "new": b.get(k)} for k in set(a) | set(b) if a.get(k) != b.get(k)
    }
    if set(changes) - ALLOWED:
        raise ValueError(
            "Not an execution-only migration: " + ", ".join(sorted(set(changes) - ALLOWED))
        )
    validate(new, 1)
    return changes


def tensor_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensor_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tensor_tree_equal(x, y) for x, y in zip(a, b))
    try:
        import numpy as np

        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
    except ImportError:
        pass
    return a == b


def migrate_checkpoint(path, new_config, expected_provenance, receipt):
    path = Path(path)
    state = load_checkpoint(path, world_size=1)
    changes = config_changes(state["config"], new_config)
    if any(
        state["provenance"].get(k) != v for k, v in expected_provenance.items() if k != "config"
    ):
        raise ValueError("Frozen data/assets/stats/model dependencies changed")
    backup = Path(receipt).parent / "before.pt"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise FileExistsError("Migration backup already exists; do not overwrite")
    os.link(path, backup)
    prior_sha = file_hash(path)
    state["runtime_migration"] = {
        "source_checkpoint_sha256": prior_sha,
        "changes": changes,
        "optimizer_reset": False,
        "successful_optimizer_step_preserved": state["successful_optimizer_step"],
        "bitwise_training_trajectory_claim": False,
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    preserved = {
        key: state[key]
        for key in ("model", "optimizer", "scheduler", "scaler", "ema", "sampler", "rank_rng")
    }
    state["config"] = new_config
    state["provenance"] = expected_provenance
    atomic_save(state, path)
    loaded = load_checkpoint(path, expected_provenance, 1)
    if not all(tensor_tree_equal(value, loaded[key]) for key, value in preserved.items()):
        raise AssertionError("Migration state preservation failed; original remains in before.pt")
    output = {
        **state["runtime_migration"],
        "destination_checkpoint_sha256": file_hash(path),
        "backup": str(backup),
        "state_preservation_verified": True,
    }
    atomic_json(receipt, output)
    return output
