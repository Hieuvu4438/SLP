"""Select EMA snapshots on PHOENIX dev broadcasts and export controlled comparisons."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from configs.experiments.motionhiflow_slp.select_dev import split_indices  # noqa: E402
from slp.config import load_yaml  # noqa: E402
from slp.data import load_dataset  # noqa: E402
from slp.evaluation import evaluate  # noqa: E402
from slp.evaluation.metrics import articulatory_geometry_scores  # noqa: E402
from slp.registry import get_method  # noqa: E402
from slp.tracking.run import RunContext  # noqa: E402
from slp.utils.seed import set_seed  # noqa: E402

from methods.articulatory_hierarchical_flow_matching.method import atomic_write  # noqa: E402
from methods.articulatory_hierarchical_flow_matching.subspace import (  # noqa: E402
    DescriptorExtractor,
    checksum,
)

NAME = "articulatory_hierarchical_flow_matching"
PRIMARY = "soke_dtw_jpe_hand"


def save_predictions(directory, name, ids, predictions):
    atomic_write(directory / f"{name}.npz",
                 lambda file: np.savez_compressed(file, **dict(zip(ids, predictions, strict=True))))


def swapped_inputs(inputs):
    # Pair different texts of similar word count using a stable ordering.
    order = sorted(range(len(inputs)), key=lambda i: (len(inputs[i]["text"].split()), i))
    swapped = []
    for i in range(0, len(order) - 1, 2):
        left, right = order[i:i + 2]
        a, b = inputs[left], inputs[right]
        if a["text"] != b["text"]:
            swapped.extend(((left, {**a, "text": b["text"]}), (right, {**b, "text": a["text"]})))
    return sorted(swapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--lengths-from", type=Path, help="Common text-predicted dev durations")
    parser.add_argument("--descriptor-from", type=Path,
                        help="Articulatory checkpoint supplying shared train descriptor scales")
    parser.add_argument("--diagnostic-clips", type=int, default=0,
                        help="Number of held-out dev clips to use for component and text interventions")
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = load_yaml(run / "config.yaml")
    if (cfg["method_name"] != NAME or cfg["data"]["name"] != "phoenix_smplx"
            or cfg["eval_split"] != "val"):
        parser.error("Expected an articulatory hierarchical flow run on PHOENIX dev")
    if args.descriptor_from and not args.diagnostic_clips:
        parser.error("--descriptor-from requires --diagnostic-clips")
    if args.diagnostic_clips < 0:
        parser.error("--diagnostic-clips must be nonnegative")
    seed = cfg["seed"]
    set_seed(seed, cfg["train"].get("deterministic", True))
    model = get_method(NAME)(cfg, RunContext(run, seed, logging.getLogger(__name__)))
    dev = load_dataset(cfg).val
    tune, held = split_indices(dev.inputs)
    if not tune or not held:
        raise ValueError("Need separate tune and reporting broadcasts")
    tune_inputs = [dev.inputs[i] for i in tune]
    tune_targets = [dev.targets[i] for i in tune]
    held_inputs = [dev.inputs[i] for i in held]
    held_targets = [dev.targets[i] for i in held]
    if args.lengths_from:
        model.sampler["lengths_from_npz"] = str(args.lengths_from.resolve())
    length_source = model.sampler["lengths_from_npz"]
    tune_lengths = model._length_archive(model.prediction_keys(tune_inputs))
    held_ids = model.prediction_keys(held_inputs)
    held_lengths = model._length_archive(held_ids)
    candidates = sorted((run / "checkpoints").glob("model_step*.pt"))
    if not candidates:
        candidates = [run / "checkpoints" / "model.pt"]
    scores = {}
    for checkpoint in candidates:
        model.load_checkpoint(checkpoint)
        poses = model.generate(tune_inputs, lengths=tune_lengths)
        scores[checkpoint] = evaluate(poses, tune_targets, [PRIMARY])[PRIMARY]
    best = min(scores, key=scores.get)
    model.load_checkpoint(best)
    generated = model.generate(held_inputs, lengths=held_lengths)
    report = {"checkpoint": best.name, "checkpoint_sha256": checksum(best),
              "selection_scores": {path.name: value for path, value in scores.items()},
              "scorer_sha256": checksum(ROOT / "src/slp/evaluation/metrics.py"),
              "length_archive_sha256": checksum(ROOT / Path(length_source)) if length_source else None,
              "tune_ids": model.prediction_keys(tune_inputs), "held_out_ids": held_ids,
              "held_out": evaluate(generated, held_targets, cfg["eval"]["metrics"])}
    report["held_out"]["duration_mae"] = float(np.mean([
        abs(len(x) - len(y)) for x, y in zip(generated, held_targets, strict=True)
    ]))
    directory = run / "dev_selection"
    directory.mkdir(exist_ok=True)
    save_predictions(directory, "held_out", held_ids, generated)
    n = min(args.diagnostic_clips, len(held_inputs))
    if n:
        inputs, targets, ids = held_inputs[:n], held_targets[:n], held_ids[:n]
        true_lengths = [len(y) for y in targets]
        predicted_lengths = [len(y) for y in generated[:n]]
        systems = {
            "text": generated[:n],
            "oracle_duration": model.generate(inputs, lengths=true_lengths),
            "new_seed": model.generate(inputs, lengths=predicted_lengths, noise_seed=seed + 1),
        }
        if model.subspace["mode"] != "single" and not model.subspace["unconstrained"]:
            for part in ("coarse", "residual"):
                systems[f"oracle_{part}"] = model.generate(
                    inputs, oracle_component=part, oracle_targets=targets,
                )
        stats = ROOT / cfg["data"]["stats"]
        descriptor = DescriptorExtractor(stats, model.device, model.subspace["fk_chunk"])

        def observe(rows):
            return [descriptor(torch.from_numpy(row)[None]).squeeze(0).cpu().numpy() for row in rows]

        scales = None
        if model.calibration["rig"]:
            scales = model.representation.observation_std.cpu().numpy()
        if args.descriptor_from:
            source = torch.load(args.descriptor_from, weights_only=True, map_location="cpu")
            provenance = source["calibration"]
            if (source["signature"]["mode"] != "articulatory"
                    or provenance["manifest_sha256"] != model.calibration["manifest_sha256"]
                    or provenance["rig"] != descriptor.provenance):
                raise ValueError("Descriptor scales must come from the same train manifest and rig")
            scales = source["representation"]["observation_std"].numpy()
            report["descriptor_checkpoint_sha256"] = checksum(args.descriptor_from)
        ref = observe(targets) if scales is not None else [None] * n

        def measurements(poses, refs, ref_descriptors):
            values = evaluate(poses, refs, [PRIMARY])
            if scales is not None:
                rows = [articulatory_geometry_scores(a, b, scales)
                        for a, b in zip(observe(poses), ref_descriptors, strict=True)]
                values.update({k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
            return values

        report["diagnostics"] = {}
        for name, poses in systems.items():
            report["diagnostics"][name] = measurements(poses, targets, ref)
            save_predictions(directory, name, ids, poses)
        swapped = swapped_inputs(inputs)
        if swapped:
            idx, other_inputs = zip(*swapped)
            other = model.generate(other_inputs, lengths=[predicted_lengths[i] for i in idx])
            compare = [targets[i] for i in idx]
            ref_cmp = [ref[i] for i in idx]
            report["diagnostics"]["swapped"] = measurements(other, compare, ref_cmp)
            report["diagnostics"]["matched_subset"] = measurements(
                [generated[i] for i in idx], compare, ref_cmp,
            )
            save_predictions(directory, "swapped", [ids[i] for i in idx], other)
    atomic_write(directory / "scores.json", lambda f: f.write(
        (json.dumps(report, indent=2) + "\n").encode(),
    ))


if __name__ == "__main__":
    main()
