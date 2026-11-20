"""Dev-only checks on saved runs and paired comparisons using the shared evaluator."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from slp.config import load_yaml
from slp.data import load_dataset
from slp.evaluation.metrics import soke_protocol_scores
from slp.tracking.run import RunContext
from slp.utils.logging import get_logger

from . import geometry as G
from .checkpoints import write_json
from .method import Method


def dev_context(run):
    cfg = load_yaml(run / "config.yaml")
    metadata = json.loads((run / "meta.json").read_text())
    if metadata["eval_split"] != "val" or cfg.get("eval_split") != "val":
        raise ValueError(f"Expected a development run at {run}")
    return cfg, metadata


def development(run, limit=None):
    cfg, info = dev_context(run)
    data = load_dataset(cfg)
    n = len(data.val) if limit is None else min(limit, len(data.val))
    if n < 1:
        raise ValueError("Empty development set")
    inputs = data.val.inputs[:n]
    targets = [np.asarray(data.val.targets[i]) for i in range(n)]
    return cfg, info, inputs, targets, data.train


def archive_values(path, inputs):
    with np.load(path, allow_pickle=False) as archive:
        return [archive[x["name"].replace("/", "__")] for x in inputs]


def save_archive(path, inputs, poses):
    np.savez_compressed(path, **{x["name"].replace("/", "__"): pose for x, pose in zip(inputs, poses, strict=True)})


def swap_indices(inputs, lengths):
    """Deterministic nearest-length different sentence within the same language."""
    buckets = defaultdict(list)
    for i, item in enumerate(inputs):
        buckets[item["src"]].append(i)
    swaps = {}
    for idx in buckets.values():
        ordered = sorted(idx, key=lambda j: (lengths[j], inputs[j]["name"]))
        left, right = {}, {}
        for order, previous in ((ordered, left), (ordered[::-1], right)):
            anchor = None
            for item in order:
                if anchor is not None:
                    previous[item] = (anchor if inputs[anchor]["text"] != inputs[item]["text"]
                                      else previous.get(anchor))
                anchor = item
        for j in ordered:
            choices = [k for k in (left.get(j), right.get(j)) if k is not None]
            if choices:
                swaps[j] = min(choices, key=lambda k: (abs(lengths[j] - lengths[k]), inputs[k]["name"]))
    return swaps


def generate_controls(run, output, limit, inference, seed, device):
    cfg, info, inputs, targets, train = development(run, limit)
    output.mkdir(parents=True, exist_ok=True)
    length = [len(p) for p in targets]
    swaps = swap_indices(inputs, length)
    if len(swaps) != len(inputs):
        raise ValueError("Control set lacks a different sentence in at least one language")
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(train), min(256, len(train)), replace=False)
    static = np.concatenate([np.asarray(train.targets[i]) for i in subset]).mean(0, keepdims=True)
    # Reordering blocks preserves per-frame poses and much of the short-range motion.
    temporal = [np.concatenate(np.array_split(a, min(4, len(a)))[::-1]) for a in targets]
    controls = {"ground_truth": targets, "wrong_motion": [targets[swaps[j]] for j in range(len(inputs))],
                "reordered_motion": temporal, "static": [np.repeat(static, n, axis=0) for n in length]}
    if inference:
        if info["method"] != "signmflow_kinematic":
            raise ValueError("Counterfactual inference needs a run with the new method manifest")
        if device:
            cfg["method"]["device"] = device
        cfg["method"]["sampler"]["noise"] = "clip"
        context = RunContext(output, info["seed"], get_logger(f"diagnostics.{output}", output / "diagnostics.log"))
        method = Method(cfg, context)
        method.load(run / "checkpoints")
        base, _ = method.infer(inputs, sampling_seed=seed)
        locks = [len(a) for a in base]
        replacement = [dict(x, text=inputs[swaps[j]]["text"]) for j, x in enumerate(inputs)]
        controls.update({"same_text_same_noise": base,
                         "different_text_same_noise": method.infer(replacement, lengths=locks, sampling_seed=seed)[0],
                         "same_text_new_noise": method.infer(inputs, lengths=locks, sampling_seed=seed + 1)[0],
                         "oracle_length": method.infer(inputs, lengths=length, sampling_seed=seed)[0]})
        write_json(output / "temporal_summary.json", {
            key: motion_summary(method.codec, sequences, targets) for key, sequences in controls.items()
        })
    for key, poses in controls.items():
        save_archive(output / f"{key}.npz", inputs, poses)
    write_json(output / "manifest.json", {
        "run": str(run.resolve()), "split": "val", "seed": seed, "count": len(inputs),
        "id_to_swap_id": {x["name"]: inputs[swaps[j]]["name"] for j, x in enumerate(inputs)},
        "mean_swap_length_gap_frames": float(np.mean([abs(length[j] - length[k]) for j, k in swaps.items()])),
        "controls": list(controls),
    })


def motion_summary(codec, predictions, targets):
    speeds = [[], []]
    for slot, clips in zip(speeds, (targets, predictions)):
        for clip in clips:
            q, _ = codec.encode(torch.as_tensor(clip))
            if codec.rep == "6d":
                q = G.matrix_to_quat(G.sixd_to_matrix(q))
            angles = G.relative_rotvec(q[:-1, 10:40], q[1:, 10:40]).norm(dim=-1).rad2deg()
            slot.append(angles.flatten().numpy())
    p95 = [float(np.quantile(x, 0.95)) if len(x) else None for x in map(np.concatenate, speeds)]
    return {"length_mae_frames": float(np.mean([abs(len(a) - len(b)) for a, b in zip(predictions, targets)])),
            "hand_angle_per_frame_p95_target_then_prediction": p95}


def metrics_values(predictions, targets):
    rows = soke_protocol_scores(predictions, targets)
    return {**rows, "jpe_hand": (rows["jpe_lhand"] + rows["jpe_rhand"]) / 2,
            "pa_hand": (rows["pa_lhand"] + rows["pa_rhand"]) / 2}


def bootstrap_mean(values, labels, samples, seed):
    values = np.asarray(values)
    if len(values) != len(labels) or len(values) < 1 or samples < 1:
        raise ValueError("Invalid paired bootstrap arguments")
    _, index = np.unique(labels, return_inverse=True)
    n = int(index.max()) + 1
    sums = np.bincount(index, weights=values)
    count = np.bincount(index)
    rng = np.random.default_rng(seed)
    means = []
    for size in range(0, samples, 128):
        draws = rng.integers(n, size=(min(128, samples - size), n))
        means.extend(sums[draws].sum(1) / count[draws].sum(1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def compare(pairs, output, limit, clusters, bootstrap, seed):
    cfg, _, inputs, targets, _ = development(pairs[0][0], limit)
    ids = [row["name"] for row in inputs]
    labels = ids
    if clusters:
        groups = json.loads(clusters.read_text())
        labels = [groups[name] for name in ids]
    required_metrics = ("jpe_body", "jpe_hand", "pa_body", "pa_body_ua", "pa_hand")
    deltas, per_seed, paired_runs = defaultdict(list), {}, []
    for baseline, candidate in pairs:
        base_cfg, base_meta = dev_context(baseline)
        cand_cfg, cand_meta = dev_context(candidate)
        if not base_cfg["data"] == cand_cfg["data"] == cfg["data"]:
            raise ValueError("Comparisons require the same resolved dataset and preprocessing")
        trial = base_meta["seed"]
        if trial != cand_meta["seed"] or trial in per_seed:
            raise ValueError("Pair equal, unique training seeds")
        x, y = (archive_values(p / "predictions.npz", inputs) for p in (baseline, candidate))
        before, after = metrics_values(x, targets), metrics_values(y, targets)
        per_seed[trial] = {}
        for key in required_metrics:
            delta = after[key] - before[key]
            deltas[key].append(delta)
            per_seed[trial][key] = {"baseline": float(before[key].mean()), "candidate": float(after[key].mean()),
                                    "candidate_minus_baseline": float(delta.mean())}
        paired_runs.append({"seed": trial, "baseline": str(baseline.resolve()), "candidate": str(candidate.resolve())})
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate = {}
    for key, delta in deltas.items():
        array = np.stack(delta)
        aggregate[key] = {"candidate_minus_baseline": float(array.mean()),
                          "seed_delta_std": float(array.mean(1).std(ddof=1)) if len(array) > 1 else None,
                          "ci_95_fixed_seeds": bootstrap_mean(array.mean(0), labels, bootstrap, seed)}
    write_json(output, {"split": "val", "comparison": paired_runs, "n_clips": len(ids),
                        "n_clusters": len(set(labels)), "bootstrap_draws": bootstrap,
                        "per_training_seed": per_seed, "paired_metrics": aggregate})


def score_controls(run, control_dir, output, limit):
    _, _, inputs, targets, _ = development(run, limit)
    manifest = json.loads((control_dir / "manifest.json").read_text())
    if manifest["run"] != str(run.resolve()) or manifest["count"] != len(inputs):
        raise ValueError("Controls and dataset IDs must match")
    result = {}
    for control in manifest["controls"]:
        poses = archive_values(control_dir / f"{control}.npz", inputs)
        result[control] = {key: float(value.mean()) for key, value in metrics_values(poses, targets).items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)


def latency(run, output, limit, device, warmup):
    cfg, info = dev_context(run)
    if not (warmup >= 0 and info["method"] == "signmflow_kinematic"):
        raise ValueError("Expected the new method and nonnegative warmup")
    inputs = load_dataset(cfg).val.inputs[:limit]
    if not inputs:
        raise ValueError("No clips to benchmark")
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg["method"]["sampler"].update(batch_size=1, diagnostics=False)
    if device:
        cfg["method"]["device"] = device
    ctx = RunContext(output.parent, info["seed"], get_logger(f"latency.{output}", output.with_suffix(".log")))
    method = Method(cfg, ctx)
    method.load(run / "checkpoints")
    cuda = method.device.type == "cuda"
    for i in range(warmup):
        method.infer(inputs[i % len(inputs): i % len(inputs) + 1])
    if cuda:
        torch.cuda.synchronize(method.device)
        torch.cuda.reset_peak_memory_stats(method.device)
    elapsed = []
    lengths = []
    for item in inputs:
        begin = time.perf_counter()
        predictions, _ = method.infer([item])
        if cuda:
            torch.cuda.synchronize(method.device)
        elapsed.append(time.perf_counter() - begin)
        lengths.append(len(predictions[0]))
    write_json(output, {
        "split": "val", "batch_size": 1, "device": str(method.device), "warmup": warmup,
        "torch": str(torch.__version__), "run": str(run.resolve()),
        "device_name": torch.cuda.get_device_name(method.device) if cuda else "cpu",
        "nfe": len(method.grid) - 1, "coarse_passes": int(method.gen.coarse is not None),
        "p50_seconds": float(np.median(elapsed)), "p95_seconds": float(np.quantile(elapsed, 0.95)),
        "peak_cuda_mib": torch.cuda.max_memory_allocated(method.device) / 2**20 if cuda else None,
        "items": [{"id": item["name"], "seconds": t, "length": n}
                  for item, t, n in zip(inputs, elapsed, lengths, strict=True)],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    c = modes.add_parser("controls")
    c.add_argument("run", type=Path)
    c.add_argument("output", type=Path)
    c.add_argument("--inference", action="store_true")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--device")
    s = modes.add_parser("score-controls")
    s.add_argument("run", type=Path)
    s.add_argument("controls", type=Path)
    s.add_argument("output", type=Path)
    p = modes.add_parser("compare")
    p.add_argument("--pair", nargs=2, action="append", required=True, type=Path, metavar=("BASELINE", "CANDIDATE"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clusters", type=Path)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    b = modes.add_parser("latency")
    b.add_argument("run", type=Path)
    b.add_argument("output", type=Path)
    b.add_argument("--device")
    b.add_argument("--warmup", type=int, default=5)
    for mode in (c, s, p, b):
        mode.add_argument("--limit", type=int, help="Use matching dev prefix for verification runs")
    a = parser.parse_args(argv)
    if a.mode == "controls":
        generate_controls(a.run, a.output, a.limit, a.inference, a.seed, a.device)
    elif a.mode == "score-controls":
        score_controls(a.run, a.controls, a.output, a.limit)
    elif a.mode == "latency":
        latency(a.run, a.output, a.limit, a.device, a.warmup)
    else:
        compare(a.pair, a.output, a.limit, a.clusters, a.bootstrap, a.seed)


if __name__ == "__main__":
    main()
