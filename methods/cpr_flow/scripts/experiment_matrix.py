#!/usr/bin/env python
"""Emit planned reproducible controls. This file launches no jobs and contains no results."""

import argparse
import json
from pathlib import Path


def matrix(seed):
    common = f"--set seed={seed}"
    parent = f"runs/cfm_joint3_parent_s{seed}/checkpoints/step_200000_ema.pt"
    c0 = f"runs/relation_joint3_s{seed}/checkpoints/best_ema.pt"

    def train(run, name, source=None):
        command = f"python -m cprflow.cli.train --config configs/runs/{run}.yaml {common} --set run.id={name}"
        return command + (f" --warm-start {source}" if source else "")

    def evaluation(checkpoint, extra=""):
        return [
            f"python -m cprflow.cli.evaluate --checkpoint {checkpoint} --dataset {d} --split dev {extra}".strip()
            for d in ("csl_daily", "how2sign", "phoenix2014t")
        ]

    rows = {
        "B1-pre": {
            "updates": 200000,
            "train": train("baseline_joint3", f"cfm_joint3_parent_s{seed}"),
        },
        "B1-cont": {
            "updates": 100000,
            "parent": parent,
            "train": train("baseline_cont_joint3", f"cfm_joint3_s{seed}", parent),
        },
        "B2": {
            "updates": 100000,
            "parent": parent,
            "train": train("auxonly_joint3", f"auxonly_joint3_s{seed}", parent),
        },
        "B3": {
            "updates": 100000,
            "parent": parent,
            "train": train("capacity_joint3", f"capacity_joint3_s{seed}", parent),
        },
        "C0": {
            "updates": 100000,
            "parent": parent,
            "train": train("relation_joint3", f"relation_joint3_s{seed}", parent),
        },
        "B0-pre": {
            "updates": 200000,
            "train": train("diffusion_joint3", f"diffusion_joint3_parent_s{seed}"),
        },
        "B0-cont": {
            "updates": 100000,
            "train": train(
                "diffusion_cont_joint3",
                f"diffusion_joint3_s{seed}",
                f"runs/diffusion_joint3_parent_s{seed}/checkpoints/step_200000_ema.pt",
            ),
        },
        "C1": {
            "evaluate": evaluation(c0, "--guard-mode unrestricted"),
            "label": "geometry-only correction control",
        },
        "C2": {
            "evaluate": evaluation(
                c0,
                "--guard-mode feature_only --teacher-checkpoint runs/teacher_joint3_s17/checkpoints/best_ema.pt --teacher-calibration teacher_calibration.json",
            ),
            "requires": "genuine approved dev calibration; never synthesized",
        },
        "C3": {
            "updates": 100000,
            "parent": c0,
            "train": train("full_joint3", f"full_joint3_s{seed}", c0)
            + " --set paths.teacher_checkpoint=runs/teacher_joint3_s17/checkpoints/best_ema.pt --set paths.teacher_calibration=teacher_calibration.json",
            "requires": "approved teacher calibration; compare to full-no-semantic control with equal added updates",
        },
        "C3-no-semantic": {
            "updates": 100000,
            "parent": c0,
            "train": train("full_control_joint3", f"full_control_joint3_s{seed}", c0),
        },
        "OracleT": {
            "evaluate": evaluation(c0, "--duration-source oracle --diagnostic-only"),
            "main_comparison": False,
        },
        "OracleR": {
            "evaluate": evaluation(c0, "--relation-source oracle --diagnostic-only"),
            "main_comparison": False,
        },
    }
    for short, dataset in [
        ("csl", "csl_daily"),
        ("how2sign", "how2sign"),
        ("phoenix", "phoenix2014t"),
    ]:
        rows[f"Mono-{dataset}"] = {
            "updates": 300000,
            "shared_joint3_AE": True,
            "train": train(f"mono_{short}", f"cfm_{dataset}_sharedAE_s{seed}"),
        }
    return {key: {"status": "planned_not_executed", **value} for key, value in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="17,42,123")
    args = parser.parse_args()
    artifact = {
        "results": False,
        "shared_AE_text_stats": "train-fit fixed s17 artifacts",
        "runs": {seed: matrix(int(seed)) for seed in args.seeds.split(",")},
    }
    Path(args.output).write_text(json.dumps(artifact, indent=2))
    print(f"Wrote planned matrix: {args.output}")


if __name__ == "__main__":
    main()
