#!/usr/bin/env python3
"""Select Phoenix snapshots by generated dev pose on a fixed, broadcast-disjoint subset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from slp.config import load_yaml  # noqa: E402
from slp.data import load_dataset  # noqa: E402
from slp.evaluation import evaluate  # noqa: E402
from slp.evaluation.metrics import soke_kinematics, soke_protocol_scores  # noqa: E402
from slp.registry import get_method  # noqa: E402
from slp.tracking.run import RunContext  # noqa: E402
from slp.utils.seed import set_seed  # noqa: E402

from methods.motionhiflow_slp.method import atomic_write  # noqa: E402

GEOMETRY = {
    "soke_dtw_jpe_body": ("jpe_body",),
    "soke_dtw_jpe_hand": ("jpe_lhand", "jpe_rhand"),
    "soke_dtw_pa_jpe_body": ("pa_body",),
    "soke_dtw_pa_jpe_body_ua": ("pa_body_ua",),
    "soke_dtw_pa_jpe_hand": ("pa_lhand", "pa_rhand"),
}
PRIMARY = "soke_dtw_jpe_hand"


def split_indices(inputs: list[dict]) -> tuple[list[int], list[int]]:
    """Select about 64 clips using whole broadcasts and a fixed random ordering."""
    groups = {}
    for i, item in enumerate(inputs):
        broadcast = item["name"].rsplit("-", 1)[0]
        groups.setdefault(broadcast, []).append(i)
    rng = np.random.default_rng(20261123)
    tune = []
    for i in rng.permutation(len(groups)):
        tune.extend(list(groups.values())[i])
        if len(tune) >= 64:
            break
    tune = set(tune)
    return sorted(tune), sorted(set(range(len(inputs))) - tune)


def summarize(scores: dict, indices) -> dict[str, float]:
    return {
        name: float(np.stack([scores[part][indices] for part in parts]).mean())
        for name, parts in GEOMETRY.items()
    }


def run(path: Path) -> dict:
    cfg = load_yaml(path / "config.yaml")
    if cfg["data"]["name"] != "phoenix_smplx" or cfg.get("eval_split") != "val":
        raise ValueError("This experiment only scores Phoenix dev")
    cfg["method"]["export_predictions"] = False
    seed = cfg["seed"]
    set_seed(seed, cfg.get("train", {}).get("deterministic", True))
    dataset = load_dataset(cfg)
    dev = dataset.val
    tune, gate = split_indices(dev.inputs)
    if not tune or not gate:
        raise ValueError("Checkpoint selection needs both tuning and reporting broadcasts")
    targets = [dev.targets[i] for i in range(len(dev))]
    tune_inputs = [dev.inputs[i] for i in tune]
    tune_targets = [targets[i] for i in tune]
    model = get_method(cfg["method_name"])(
        cfg, RunContext(path, seed, logging.getLogger(__name__))
    )

    candidates = {}
    files = sorted((path / "checkpoints").glob("model_step*.pt"))
    expected = {f"model_step{step:06d}.pt" for step in cfg["method"]["flow"]["snapshot_steps"]}
    if not expected or {checkpoint.name for checkpoint in files} != expected:
        raise FileNotFoundError(f"Incomplete snapshot checkpoints under {path}")
    for checkpoint in files:
        model.load_checkpoint(checkpoint)
        tick = time.monotonic()
        scores = evaluate(model.predict(tune_inputs), tune_targets, list(GEOMETRY))
        candidates[checkpoint.name] = scores
        print(checkpoint.name, f"{time.monotonic() - tick:.2f}s", scores, flush=True)
    selected = min(candidates, key=lambda name: candidates[name][PRIMARY])
    model.load_checkpoint(path / "checkpoints" / selected)
    predictions = model.predict(dev.inputs)
    scores = soke_protocol_scores(predictions, targets)
    target_kinematics = soke_kinematics(targets)
    model_kinematics = soke_kinematics(predictions)
    ratios = {
        field: float(values.mean() / target_kinematics[field].mean())
        for field, values in model_kinematics.items()
    }
    summary = {
        "run": str(path),
        "seed": seed,
        "checkpoint": selected,
        "selection_metric": PRIMARY,
        "snapshot_selection": candidates,
        "tune_clip_ids": [dev.inputs[i]["name"] for i in tune],
        "gate_clip_ids": [dev.inputs[i]["name"] for i in gate],
        "full_dev": summarize(scores, slice(None)),
        "gate_dev": summarize(scores, gate),
        "full_dev_kinematic_ratios": ratios,
    }
    keys = [item["name"].replace("/", "__") for item in dev.inputs]
    arrays = dict(zip(keys, predictions, strict=True))
    atomic_write(path / "predictions_selected.npz", lambda f: np.savez_compressed(f, **arrays))
    encoded = (json.dumps(summary, indent=2) + "\n").encode()
    atomic_write(path / "selection.json", lambda f: f.write(encoded))
    print("Selected", selected, summary["full_dev"], flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run(parser.parse_args().run)
