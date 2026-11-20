"""Runnable dev ablations through the standard SLP runner."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from slp.paths import PROJECT_ROOT

METHOD = "signmflow_kinematic"
DATASETS = ("csl_smplx", "phoenix_smplx", "how2sign_smplx_soke")
S3 = {"flow.path": "geodesic", "flow.input_rotation": "canonical", "flow.endpoint_loss": "canonical"}
SO3 = {**S3, "flow.path": "so3"}
CHORDAL = {**SO3, "flow.endpoint_loss": "chordal"}
E2 = {"generator.epochs": 30, "generator.lr": 0.00002, "generator.text_lr": 0.00002}
POSITION = {**E2, "flow.position_weight": 1.0}
VELOCITY = {**POSITION, "flow.velocity_weight": 0.01}
E4 = {**CHORDAL, "flow.path": "so3_corrected", "correction.frame_chunk": 16}
PRESETS = {
    "b0": {},
    "e1_uniform": {},
    "e1_late": {"sampler.time_grid": "[0,0.25,0.5,0.7,0.9,1]"},
    "e1_anchor_expr": {"sampler.expression_update": "anchored"},
    "e1_one": {"sampler.steps": 1},
    "e1_ten": {"sampler.steps": 10},
    "e2_continue": E2,
    "e2_position": POSITION,
    "e2_palm": {**POSITION, "supervision.groups.palm": 1.0},
    "e2_velocity": VELOCITY,
    "e2_position_scratch": {"flow.position_weight": 1.0},
    "e2_velocity_scratch": {"flow.position_weight": 1.0, "flow.velocity_weight": 0.01},
    "e2_jaw_expression": {**E2, "flow.w_jaw": 0.1, "flow.w_expr": 0.2},
    "e3_s3": S3,
    "e3_so3": SO3,
    "e3_chordal": CHORDAL,
    "e3_chordal_with_loss": {**CHORDAL, **{k: VELOCITY[k] for k in ("flow.position_weight", "flow.velocity_weight")}},
    "e3_invariant_input": {**CHORDAL, "flow.input_rotation": "6d"},
    "e3_projected_6d": {**CHORDAL, "flow.path": "projected_6d"},
    "e3_euclidean_6d": {"flow.representation": "6d", "flow.path": "linear"},
    "e4_diagonal": {**E4, "measure.kind": "fk_diagonal"},
    "e4_joint_block": {**E4, "measure.kind": "fk_joint_block"},
    "e4_full": {**E4, "measure.kind": "fk_full"},
    "e4_density": {**E4, "measure.kind": "density"},
    "e4_diagonal_with_loss": {**E4, **{k: VELOCITY[k] for k in ("flow.position_weight", "flow.velocity_weight")},
                              "measure.kind": "fk_diagonal"},
    "e4_full_with_loss": {**E4, **{k: VELOCITY[k] for k in ("flow.position_weight", "flow.velocity_weight")},
                          "measure.kind": "fk_full"},
    "e5_ar": {"flow.source_ar": 0.5},
    "e5_length": {"generator.length_mode": "quantile"},
    "e5_coarse": {"generator.coarse_stride": 4},
    "e5_matched_ot": {"flow.coupling": "matched_ot"},
}


def command(preset, dataset, seeds, name=None, checkpoint=None, legacy=False, extra=()):
    values = PRESETS[preset].copy()
    has_reuse = preset.startswith(("e1_", "e2_")) and not preset.endswith("_scratch")
    if has_reuse != (checkpoint is not None):
        raise ValueError("Evaluation and warm-start presets require --checkpoint-dir. Scratch presets train independently.")
    if legacy and not checkpoint:
        raise ValueError("--legacy-signmflow requires --checkpoint-dir")
    if checkpoint:
        if len(seeds.split(",")) != 1:
            raise ValueError("Pass one seed per checkpoint. Repeat with each matched baseline checkpoint.")
        corr = checkpoint / "correction_final.pt"
        gen = checkpoint / "generator.pt"
        for path in (corr, gen):
            if not path.is_file():
                raise FileNotFoundError(path)
        values["init.correction"] = str(corr.resolve())
        values["init.generator" if preset.startswith("e1_") else "init.warm_start"] = str(gen.resolve())
        if legacy:
            values["init.legacy_signmflow"] = "true"
    return [sys.executable, "scripts/run.py", "--method", METHOD, "--data", dataset, "--name",
            name or f"{METHOD}_{preset}_{dataset}", "--seeds", seeds, "--split", "val",
            *(f"method.{key}={value}" for key, value in values.items()), *extra]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("--data", choices=DATASETS, default="csl_smplx")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--name")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--legacy-signmflow", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Config override, repeatable")
    parser.add_argument("--run", action="store_true", help="Run training or scoring. Default prints the command.")
    a = parser.parse_args(argv)
    cmd = command(a.preset, a.data, a.seeds, a.name, a.checkpoint_dir, a.legacy_signmflow, a.set)
    print(shlex.join(cmd), flush=True)
    if a.run:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
