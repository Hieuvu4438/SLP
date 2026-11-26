"""Fit and compare PHOENIX roadmap arms with a sealed dev checkpoint selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from slp.config import compose, load_yaml  # noqa: E402
from slp.data import Split  # noqa: E402
from slp.data.sign_smplx import load_sign_smplx_split  # noqa: E402
from slp.evaluation.metrics import (  # noqa: E402
    cluster_bootstrap_ci,
    soke_kinematics,
    soke_protocol_scores,
)
from slp.paths import TEST_ACCESS_LOG  # noqa: E402
from slp.registry import get_method  # noqa: E402
from slp.tracking import create_run  # noqa: E402
from slp.tracking.run import RunContext  # noqa: E402
from slp.utils.git import git_info  # noqa: E402
from slp.utils.seed import set_seed  # noqa: E402

from configs.experiments.articulatory_hierarchical_flow_matching.backtranslation import (  # noqa: E402
    judge_blind_bundle,
    score_archives,
)
from configs.experiments.articulatory_hierarchical_flow_matching.minimal_pairs import (  # noqa: E402
    export_pairs,
    score_pairs,
    write_json,
)
from configs.experiments.motionhiflow_slp.select_dev import split_indices  # noqa: E402
from methods.articulatory_hierarchical_flow_matching.data import broadcast_id  # noqa: E402
from methods.articulatory_hierarchical_flow_matching.method import atomic_write  # noqa: E402
from methods.articulatory_hierarchical_flow_matching.subspace import (  # noqa: E402
    checksum,
    rig_assets,
    rig_checksums,
)

NAME = "articulatory_hierarchical_flow_matching"
DATA = "phoenix_smplx"
FIT_ARMS = {
    "baseline": (),
    "temporal": ("subspace.mode=temporal",),
    "single": ("subspace.mode=single",),
    "sentence": ("semantic.mode=sentence",),
    "word": ("semantic.mode=word",),
    "finetune": ("text.finetune=true",),
    "plan-uniform": ("planner.mode=uniform",),
    "plan-predicted": ("planner.mode=predicted",),
    "plan-shuffled": ("planner.mode=shuffled",),
    "plan-eos": ("planner.mode=predicted", "planner.duration_source=planner"),
    "fine-fk": ("flow.endpoint_fk_weight=0.02",),
    "coarse-fk": ("flow.coarse_fk_weight=0.02",),
    "coarse-rotation": ("flow.coarse_rotation_weight=0.02",),
    "tangent": ("subspace.rotation_chart=tangent",),
    "unconstrained": ("subspace.unconstrained=true",),
    "jitter-reference": ("refinement.reference_train_only=true",),
    "jitter": ("refinement.jitter_scale=0.15",),
    "retrieve-nearest": ("retrieval.mode=nearest",),
    "retrieve-random": ("retrieval.mode=random",),
}
GEOMETRY = {
    "body": ("jpe_body",),
    "hand": ("jpe_lhand", "jpe_rhand"),
    "pa_body": ("pa_body",),
    "pa_body_ua": ("pa_body_ua",),
    "pa_hand": ("pa_lhand", "pa_rhand"),
}


def read(path):
    return json.loads(path.read_text())


def subset(split, indices):
    return ([split.inputs[i] for i in indices], [split.targets[i] for i in indices])


def dev_phase(cfg, held):
    val = load_sign_smplx_split(cfg, "val")
    tune, holdout = split_indices(val.inputs)
    if not tune or not holdout:
        raise ValueError("Dev must have both tuning and held-out broadcasts")
    return subset(val, holdout if held else tune)


def fit(args):
    overrides = [f"method.{entry}" for entry in FIT_ARMS[args.arm]]
    cfg = compose(NAME, DATA, overrides + args.overrides + ["method.workers=0"])
    set_seed(args.seed, cfg["train"].get("deterministic", True))
    train = load_sign_smplx_split(cfg, "train")
    inputs, targets = dev_phase(cfg, False)
    ctx = create_run(cfg, args.seed, args.name or f"articulatory_{args.arm}", "val")
    cfg.update({"method_name": NAME, "eval_split": "val"})
    model = get_method(NAME)(cfg, ctx)
    model.fit(train, Split(inputs, targets))
    model.save()
    model.ctx.logger.info("Fit complete. Run roadmap tune against the saved snapshots.")


def load_run(run):
    cfg = load_yaml(run / "config.yaml")
    if (cfg.get("method_name"), cfg["data"].get("name"), cfg.get("eval_split")) != (NAME, DATA, "val"):
        raise ValueError("Expected an articulatory PHOENIX dev run")
    set_seed(cfg["seed"], cfg["train"].get("deterministic", True))
    ctx = RunContext(run, cfg["seed"], logging.getLogger(NAME))
    return cfg, get_method(NAME)(cfg, ctx)


def geometry(predictions, targets):
    scores = soke_protocol_scores(predictions, targets, path_normalized=True)
    result = {f"{key}{suffix}": np.stack([scores[f"{part}{suffix}"] for part in parts]).mean(0)
              for key, parts in GEOMETRY.items() for suffix in ("", "_per_step")}
    gen, ref = soke_kinematics(predictions), soke_kinematics(targets)
    for key in ("hand_speed", "hand_accel"):
        result[key] = gen[key]
        result[f"ref_{key}"] = ref[key]
    result["generated_lengths"] = np.array(list(map(len, predictions)))
    result["target_lengths"] = np.array(list(map(len, targets)))
    result["duration_error"] = np.abs(result["generated_lengths"] - result["target_lengths"])
    return {k: v.tolist() for k, v in result.items()}


def summary(rows):
    result = {k: float(np.mean(rows[k])) for k in (
        *GEOMETRY, *(f"{name}_per_step" for name in GEOMETRY), "duration_error",
    )}
    for stat in ("hand_speed", "hand_accel"):
        result[f"{stat}_ratio"] = float(np.mean(rows[stat]) /
                                       max(float(np.mean(rows[f"ref_{stat}"])), 1e-12))
    lengths = np.array(rows["target_lengths"])
    error = np.array(rows["generated_lengths"]) - lengths
    result["duration_bias"] = float(error.mean())
    long = lengths >= 200
    result["long_clips"] = int(long.sum())
    result["long_duration_bias"] = float(error[long].mean()) if long.any() else None
    return result


def save_poses(path, keys, rows):
    atomic_write(path, lambda file: np.savez_compressed(file, **dict(zip(keys, rows, strict=True))))


def make_policy(model, length_path):
    source = length_path or model.sampler["lengths_from_npz"]
    source = (ROOT / Path(source)).resolve() if source else None
    model.sampler["lengths_from_npz"] = str(source) if source else None
    return {"lengths_from": str(source) if source else None,
            "lengths_sha256": checksum(source) if source else None,
            "seed": model.ctx.seed, "steps": sum(model.sampler["steps"]),
            "cfg_scale": model.sampler["cfg_scale"],
            "precision": model.precision, "device": model.device.type,
            "batch_size": model.sampler["batch_size"],
            "frame_range": [model.sampler["min_frames"], model.sampler["max_frames"]],
            "metric_rig": rig_checksums(*rig_assets()),
            "metrics_sha256": checksum(ROOT / "src/slp/evaluation/metrics.py")}


def target_hash(keys, targets):
    digest = hashlib.sha256()
    for key, pose in zip(keys, targets, strict=True):
        pose = np.ascontiguousarray(pose, dtype="<f4")
        digest.update(json.dumps([key, pose.shape]).encode())
        digest.update(pose.tobytes())
    return digest.hexdigest()


def pair_lengths(model, keys, path):
    current = model.sampler["lengths_from_npz"]
    if path is not None:
        model.sampler["lengths_from_npz"] = str(path.resolve())
    try:
        result = model._length_archive(keys)
    finally:
        model.sampler["lengths_from_npz"] = current
    if result is None:
        raise ValueError("Provide common text-predicted lengths through --pair-lengths-from")
    return result


def tune(args):
    cfg, model = load_run(args.run)
    inputs, targets = dev_phase(cfg, False)
    directory = args.run / "roadmap"
    directory.mkdir(exist_ok=True)
    if (directory / "selection.json").exists():
        raise ValueError("Selection is already sealed for this run")
    policy = make_policy(model, args.lengths_from)
    keys = model.prediction_keys(inputs)
    lengths = model._length_archive(keys)
    candidates = sorted((args.run / "checkpoints").glob("model_step*.pt"))
    if not candidates:
        raise ValueError("The roadmap uses its declared EMA snapshots")
    scheduled = {f"model_step{step:06d}.pt" for step in cfg["method"]["flow"]["snapshot_steps"]}
    if set(path.name for path in candidates) != scheduled:
        raise ValueError("The run does not have the complete scheduled set of EMA snapshots")
    result = {"phase": "tune", "ids": keys, "policy": policy, "snapshots": {},
              "targets_sha256": target_hash(keys, targets)}
    for path in candidates:
        model.load_checkpoint(path)
        poses = model.generate(inputs, lengths=lengths)
        values = geometry(poses, targets)
        meta = model.calibration.get("semantic", {})
        result["snapshots"][path.name] = {
            "checkpoint_sha256": checksum(path), "summary": summary(values), "geometry": values,
            "teacher_sha256": meta.get("teacher_sha256"),
        }
    write_json(directory / "tune.json", result)
    if args.pairs:
        shared = pair_lengths(model, keys, args.pair_lengths_from)
        export_pairs(model, args.pairs, "tune", inputs, targets, candidates, shared,
                     directory / "tune_pairs")


def durations(args):
    cfg, model = load_run(args.run)
    candidates = sorted((args.run / "checkpoints").glob("model_step*.pt"))
    if not candidates:
        raise ValueError("Duration generation needs a declared EMA snapshot")
    checkpoints = {path.name: path for path in candidates}
    snapshot = checkpoints[args.snapshot] if args.snapshot else candidates[-1]
    model.load_checkpoint(snapshot)
    inputs = load_sign_smplx_split(cfg, "val").inputs
    keys = model.prediction_keys(inputs)
    lengths = [np.int32(n) for n in model.predict_durations(inputs)]
    save_poses(args.output, keys, lengths)
    write_json(args.output.with_suffix(".json"), {
        "phase": "val", "source": "native_text_only", "ids": keys,
        "checkpoint_sha256": checksum(snapshot), "archive_sha256": checksum(args.output),
    })


def matching_policy(a, b):
    if any(a[k] != b[k] for k in ("policy", "phase", "ids", "targets_sha256")):
        raise ValueError("The comparison requires identical IDs, noise seed, sampling and evaluation")


def same_pair_contract(a, b, phase):
    for key in ("manifest_sha256", "lengths", "scorer", "seed"):
        if a[key] != b[key]:
            raise ValueError(f"Semantic judgments disagree on {key}")
    if a["phase"] != phase or b["phase"] != phase or a["pairs"] != b["pairs"]:
        raise ValueError("Semantic comparisons need the same reviewed pairs and phase")


def semantic_values(report, snapshot, component):
    teacher = snapshot.get("teacher_sha256")
    if teacher and report["scorer"].get("checkpoint_sha256") == teacher:
        raise ValueError("The independent evaluation must use a different checkpoint")
    return report["scores"][snapshot["checkpoint_sha256"]][component]


def healthy_motion(current, reference):
    return all(abs(current[f"{k}_ratio"] - 1) <= abs(reference[f"{k}_ratio"] - 1) + 0.05
               for k in ("hand_speed", "hand_accel"))


def choose_snapshot(rows, reference=None, semantics=None):
    permitted = [(name, row) for name, row in rows.items() if (
        reference is None or healthy_motion(row["summary"], reference)
    ) and (semantics is None or semantics[name])]
    if not permitted:
        raise ValueError("No EMA snapshot passes the specified dynamics and semantics gates")
    return min(permitted, key=lambda item: (item[1]["summary"]["hand"], item[0]))


def select(args):
    directory = args.run / "roadmap"
    path = directory / "tune.json"
    result = read(path)
    ref = gate = evidence = None
    if not args.geometry_only and args.held_manifest is None:
        raise ValueError("Seal the independent held-pair manifest before selecting an EMA")
    if args.baseline:
        comparison = args.baseline / "roadmap"
        base_path = comparison / "tune.json"
        base = read(base_path)
        matching_policy(result, base)
        base_selection_path = comparison / "selection.json"
        base_selection = read(base_selection_path)
        if base_selection["tune_sha256"] != checksum(base_path):
            raise ValueError("Baseline tune has changed since checkpoint selection")
        ref = base["snapshots"][base_selection["file"]]["summary"]
        evidence = {"baseline_selection_sha256": checksum(base_selection_path)}
        if not args.geometry_only:
            a_path = comparison / "tune_semantic.json"
            b_path = directory / "tune_semantic.json"
            a, b = read(a_path), read(b_path)
            same_pair_contract(a, b, "tune")
            baseline = semantic_values(a, base_selection["row"], "full")["accuracy"]
            gate = {k: semantic_values(b, row, "full")["accuracy"] >= baseline
                    for k, row in result["snapshots"].items()}
            evidence.update({"semantic_baseline_sha256": checksum(a_path),
                             "semantic_candidate_sha256": checksum(b_path)})
    elif not args.geometry_only:
        raise ValueError("Select a baseline, or explicitly declare baseline geometry-only selection")
    name, row = choose_snapshot(result["snapshots"], ref, gate)
    output = directory / "selection.json"
    selection = {"file": name, "row": row, "tune_sha256": checksum(path),
                 "geometry_only": args.geometry_only, "evidence": evidence,
                 "policy": result["policy"],
                 "held_manifest_sha256": checksum(args.held_manifest) if args.held_manifest else None}
    # Exclusive creation prevents accidentally resealing a different choice after held-out use.
    if output.exists():
        raise ValueError("Selection is already sealed for this run")
    write_json(output, selection)


def load_selection(run, model):
    directory = run / "roadmap"
    path = directory / "selection.json"
    selection = read(path)
    if selection["tune_sha256"] != checksum(directory / "tune.json"):
        raise ValueError("Tune evidence changed after checkpoint selection")
    snapshot = run / "checkpoints" / selection["file"]
    if checksum(snapshot) != selection["row"]["checkpoint_sha256"]:
        raise ValueError("The sealed checkpoint changed")
    policy = make_policy(model, selection["policy"]["lengths_from"])
    if policy != selection["policy"]:
        raise ValueError("The sealed sampling policy changed")
    model.load_checkpoint(snapshot)
    return path, selection, snapshot


def score_locked(args, phase):
    run = args.run
    cfg, model = load_run(run)
    path, selected, snapshot = load_selection(run, model)
    directory = run / "roadmap"
    result_path = directory / f"{phase}.json"
    if result_path.exists():
        raise ValueError(f"Results already exist at {result_path}")
    if phase == "held":
        manifest = selected["held_manifest_sha256"]
        if (checksum(args.pairs) if args.pairs else None) != manifest:
            raise ValueError("The held-pair manifest differs from the sealed selection")
        inputs, targets = dev_phase(cfg, True)
    else:
        git = git_info()
        meta = read(run / "meta.json")["git"]
        if git["dirty"] or meta["dirty"] or git["commit"] != meta["commit"]:
            raise ValueError("Test requires the clean training commit")
        if selected["policy"]["lengths_from"]:
            raise ValueError("Use a native-duration run sealed on dev for the test track")
        confirmation = read(args.confirmation)
        held_path = run / "roadmap" / "held.json"
        if (confirmation.get("candidate_evidence_sha256") != checksum(held_path)
                or not confirmation.get("passes")):
            raise ValueError("Test requires a passing matched held-out comparison")
        append_test(run, snapshot, "started", git)
    try:
        if phase == "test":
            split = load_sign_smplx_split(cfg, "test")
            inputs, targets = split.inputs, split.targets
        score_partition(args, phase, model, inputs, targets, path, selected, snapshot, result_path)
    except (Exception, KeyboardInterrupt):
        if phase == "test":
            append_test(run, snapshot, "failed", git)
        raise
    if phase == "test":
        append_test(run, snapshot, "completed", git, read(result_path)["summary"])


def score_partition(args, phase, model, inputs, targets, path, selected, snapshot, result_path):
    keys = model.prediction_keys(inputs)
    lengths = model._length_archive(keys)
    directory = result_path.parent
    shared = pair_lengths(model, keys, args.pair_lengths_from) if phase == "held" and args.pairs else None
    poses = model.generate(inputs, lengths=lengths)
    coarse = model.generate(inputs, lengths=list(map(len, poses)), component="coarse")
    values = geometry(poses, targets)
    comparisons = geometry(coarse, targets)
    result = {"phase": phase, "ids": keys, "broadcasts": [broadcast_id(i) for i in inputs],
              "selection_sha256": checksum(path), "snapshot": selected["row"],
              "targets_sha256": target_hash(keys, targets), "subspace": model.subspace["mode"],
              "policy": selected["policy"], "summary": summary(values), "geometry": values,
              "coarse_summary": summary(comparisons)}
    if model.use_plan:
        result["planner"] = {
            "stop_by_id": dict(zip(keys, (model.plan_stops[item["text"], 0] for item in inputs),
                                   strict=True)),
            "segments_by_id": dict(zip(keys, (len(model.plan_cache[item["text"], 0][1])
                                              for item in inputs), strict=True)),
        }
    save_poses(directory / f"{phase}.npz", keys, poses)
    save_poses(directory / f"{phase}_coarse.npz", keys, coarse)
    if shared is not None:
        export_pairs(model, args.pairs, phase, inputs, targets, [snapshot], shared,
                     directory / f"{phase}_pairs")
    write_json(result_path, result)


def append_test(run, snapshot, event, git, metrics=None):
    row = {"event": f"articulatory_roadmap_test_{event}", "method": NAME,
           "commit": git["commit"], "run_dir": str(run), "checkpoint_sha256": checksum(snapshot),
           "time": datetime.now(timezone.utc).isoformat(), "metrics": metrics}
    TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TEST_ACCESS_LOG.open("a") as file:
        file.write(json.dumps(row, allow_nan=False) + "\n")


def ci_delta(candidate, reference, clusters):
    if len(candidate) != len(reference) or len(candidate) != len(clusters):
        raise ValueError("Unaligned paired observations")
    if len(set(clusters)) < 2:
        raise ValueError("A broadcast-clustered comparison needs at least two broadcasts")
    center, lo, hi, _ = cluster_bootstrap_ci(
        np.array(candidate) - np.array(reference), clusters, n_boot=10000, seed=20261126,
    )
    return {"delta": center, "ci95": [lo, hi]}


def compare(args):
    baseline, candidate, temporal = (p / "roadmap" for p in (
        args.baseline, args.candidate, args.temporal,
    ))
    a, b, t = (read(folder / "held.json") for folder in (baseline, candidate, temporal))
    if t["subspace"] != "temporal":
        raise ValueError("The temporal comparison must use the matched-capacity temporal control")
    matching_policy(a, b)
    matching_policy(t, b)
    if a["broadcasts"] != b["broadcasts"] or t["broadcasts"] != b["broadcasts"]:
        raise ValueError("Held-out evidence disagrees on broadcast membership")
    clusters = b["broadcasts"]
    hand, other = (ci_delta(b["geometry"]["hand"], row["geometry"]["hand"], clusters)
                   for row in (a, t))
    m, n = (read(folder / "held_semantic.json") for folder in (baseline, candidate))
    same_pair_contract(m, n, "held")
    old = semantic_values(m, a["snapshot"], "full")["per_pair"]
    new = semantic_values(n, b["snapshot"], "full")["per_pair"]
    if set(old) != set(new):
        raise ValueError("Semantic comparison is missing paired observations")
    meaning = ci_delta([new[i] for i in old], list(old.values()),
                       [m["pairs"][i]["broadcast"] for i in old])
    healthy = healthy_motion(b["summary"], a["summary"])
    target = min(6.975, 0.97 * a["summary"]["hand"])
    passed = (b["summary"]["hand"] <= target and hand["ci95"][1] < 0 and other["ci95"][1] < 0
              and healthy and meaning["delta"] >= 0.05 - 1e-12 and meaning["ci95"][0] > 0)
    write_json(args.output, {"passes": bool(passed), "hand_target": target, "hand_vs_base": hand,
                             "hand_vs_temporal": other, "semantic_vs_base": meaning,
                             "dynamics_guard": healthy, "candidate": b["summary"],
                             "candidate_evidence_sha256": checksum(candidate / "held.json"),
                             "baseline_evidence_sha256": checksum(baseline / "held.json"),
                             "temporal_evidence_sha256": checksum(temporal / "held.json"),
                             "semantic_evidence_sha256": [checksum(p / "held_semantic.json")
                                                         for p in (baseline, candidate)]})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    train = commands.add_parser("fit")
    train.add_argument("arm", choices=FIT_ARMS)
    train.add_argument("--seed", default=0, type=int)
    train.add_argument("--name")
    train.add_argument("overrides", nargs="*")
    for action in ("tune", "select", "held", "test", "pairs"):
        p = commands.add_parser(action)
        p.add_argument("run", type=Path)
        if action in ("tune", "held"):
            p.add_argument("--pairs", type=Path)
            p.add_argument("--pair-lengths-from", type=Path)
        if action == "tune":
            p.add_argument("--lengths-from", type=Path)
        if action == "select":
            p.add_argument("--baseline", type=Path)
            p.add_argument("--geometry-only", action="store_true")
            p.add_argument("--held-manifest", type=Path)
        if action == "pairs":
            p.add_argument("phase", choices=("tune", "held"))
            p.add_argument("judgments", type=Path)
        if action == "test":
            p.add_argument("--confirmation", required=True, type=Path)
    collate = commands.add_parser("compare")
    for name in ("baseline", "candidate", "temporal", "output"):
        collate.add_argument(name, type=Path)
    lengths = commands.add_parser("lengths")
    lengths.add_argument("run", type=Path)
    lengths.add_argument("output", type=Path)
    lengths.add_argument("--snapshot", help="Fixed snapshot name, defaults to the latest scheduled EMA")
    bt = commands.add_parser("bt")
    bt.add_argument("run", type=Path)
    bt.add_argument("arch", choices=("mbart", "qwen"))
    bt.add_argument("checkpoint", type=Path)
    bt.add_argument("output", type=Path)
    bt.add_argument("archives", nargs="+", metavar="LABEL=PATH")
    bt.add_argument("--device", default="auto")
    judge = commands.add_parser("bt-pairs")
    judge.add_argument("run", type=Path)
    judge.add_argument("phase", choices=("tune", "held"))
    judge.add_argument("arch", choices=("mbart", "qwen"))
    judge.add_argument("checkpoint", type=Path)
    judge.add_argument("output", type=Path)
    judge.add_argument("--device", default="auto")
    judge.add_argument("--margin", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.action == "fit":
        fit(args)
    elif args.action == "tune":
        tune(args)
    elif args.action == "select":
        select(args)
    elif args.action in ("held", "test"):
        score_locked(args, args.action)
    elif args.action == "pairs":
        directory = args.run / "roadmap"
        key = read(directory / f"{args.phase}_pairs" / "key.json")
        write_json(directory / f"{args.phase}_semantic.json", score_pairs(key, read(args.judgments)))
    elif args.action == "compare":
        compare(args)
    elif args.action == "lengths":
        durations(args)
    elif args.action == "bt":
        directory = args.run / "roadmap"
        held = read(directory / "held.json")
        if held["selection_sha256"] != checksum(directory / "selection.json"):
            raise ValueError("Back-translation requires a sealed held-out result")
        cfg = load_yaml(args.run / "config.yaml")
        inputs, targets = dev_phase(cfg, True)
        result = score_archives(inputs, targets, args.archives, args.checkpoint,
                                args.arch, args.device)
        write_json(args.output, result)
    elif args.action == "bt-pairs":
        bundle = args.run / "roadmap" / f"{args.phase}_pairs"
        judgments = judge_blind_bundle(bundle, args.checkpoint, args.arch, args.device, args.margin)
        write_json(args.output, judgments)


if __name__ == "__main__":
    main()
