"""Atomic trusted training checkpoints with strict provenance and per-rank RNG."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    from cprflow.jobs.storage import check_free_space

    check_free_space(path.parent)
    torch.save(state, tmp)
    os.replace(tmp, path)
    import json

    metadata = {
        key: state.get(key)
        for key in ("successful_optimizer_step", "parent_global_step", "world_size")
    }
    metadata["config_hash"] = state.get("provenance", {}).get("config")
    metadata["size"] = path.stat().st_size
    metadata["mtime_ns"] = path.stat().st_mtime_ns
    sidecar = Path(str(path) + ".json")
    temporary = sidecar.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, sort_keys=True))
    os.replace(temporary, sidecar)


def load_checkpoint(path, expected=None, world_size=None):
    # These are local trusted training artifacts, never public untrusted request files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if expected is not None and state["provenance"] != expected:
        differences = sorted(
            k
            for k in set(expected) | set(state["provenance"])
            if expected.get(k) != state["provenance"].get(k)
        )
        raise ValueError(f"Strict resume provenance mismatch: {differences}")
    if world_size is not None and state["world_size"] != world_size:
        raise ValueError("Strict resume requires the same world size")
    return state
