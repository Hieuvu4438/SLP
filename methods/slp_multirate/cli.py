"""Command line for preparation, stage training, inference and pilot review."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import PILOTS, STAGES, VARIANTS, resolve
from .data import (
    SPLITS, CanonicalDataset, audit_manifest, create_synthetic, digest, file_hash,
    make_cohort, make_signer_diagnostic, prepare_data, read_jsonl, save_npz,
    write_json, write_jsonl,
)
from .evaluation import compare, fit_evaluator, run_evaluation
from .geometry import JOINTS, Rig, matrix_to_rot6d
from .pipeline import (
    PROTOCOLS, Cascade, Statistics, TextStore, inference_dependencies, load_stage, load_torch,
    required_dependencies, tensor_state_hash, training_statistics,
)
from .training import (
    fit_confidence, git_info, runtime_versions, split_hash, tiny_overfit, train_stage,
)

CHECKPOINTS = (*STAGES, "confidence")


def _config(args):
    overrides = list(args.set)
    if getattr(args, "seed", None) is not None:
        overrides.append(f"experiment.seed={args.seed}")
    if getattr(args, "run_dir", None):
        overrides.append(f"experiment.run_dir={json.dumps(str(args.run_dir))}")
    return resolve(args.pilot, args.variant, args.config, overrides)


def _run_dir(cfg):
    exp = cfg["experiment"]
    return Path(exp["run_dir"]) / exp["pilot"] / exp["variant"] / str(exp["seed"])


def _paths(args, cfg):
    manifest = Path(args.manifest or cfg["data"]["manifest"])
    rig = args.rig
    if rig is None:
        rig = manifest.parent / "rig.json" if args.manifest else cfg["data"]["rig"]
    return manifest, Path(rig)


def _device(args):
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Select an available CPU or CUDA device")
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
    return device


def _bundle(path):
    if path is None:
        return None
    path = Path(path).resolve()
    payload = json.loads(path.read_bytes())
    files = payload.get("checkpoints")
    if payload.get("schema") != 1 or not isinstance(files, dict) or set(files) - set(CHECKPOINTS):
        raise ValueError("Invalid checkpoint bundle schema")
    results = {}
    for key, item in files.items():
        file = path.parent / item["path"]
        if file_hash(file) != item["sha256"]:
            raise ValueError(f"Bundle hash mismatch for {key}")
        results[key] = file
    return results


def _checkpoint_paths(args, keys):
    paths = _bundle(getattr(args, "bundle", None)) or {}
    for key in CHECKPOINTS:
        path = getattr(args, f"{key}_ckpt", None)
        if path is not None:
            if key in paths:
                raise ValueError(f"{key} supplied in both bundle and explicit path")
            paths[key] = path
    missing = set(keys) - set(paths)
    if missing:
        raise ValueError(f"Missing explicit checkpoint(s) for {', '.join(sorted(missing))}")
    return {k: paths[k] for k in keys}


def create_bundle(paths, out):
    import os

    out = Path(out).resolve()
    write_json(out, {"schema": 1, "checkpoints": {key: {
        "path": os.path.relpath(Path(path).resolve(), out.parent), "sha256": file_hash(path)
    } for key, path in paths.items()}})
    return out


def create_cascade(cfg, rig, device, paths=None, split_key=None, text_cache=None,
                   condition_cache=None, require_fine_dependencies=False):
    text = TextStore(cfg["text"], device, text_cache)
    frozen = {stage: load_stage(path, stage, cfg, rig, text, device, split_key)
              for stage, path in (paths or {}).items() if stage != "confidence"}
    confidence = None
    confidence_hash = None
    if paths and "confidence" in paths:
        path = paths["confidence"]
        calibration = load_torch(path)
        if (calibration["planner_sha256"] != frozen["anchors"].sha256 or
                calibration["rig_sha256"] != rig.sha256):
            raise ValueError("Confidence calibration belongs to a different planner or rig")
        confidence = torch.nn.Linear(cfg["anchors"]["width"], 2).to(device)
        confidence.load_state_dict(calibration["ema"])
        confidence.requires_grad_(False).eval()
        confidence_hash = file_hash(path)
    cascade = Cascade(cfg, rig, text, device, frozen,
                      condition_root=Path(condition_cache) if condition_cache else None,
                      confidence=confidence, confidence_hash=confidence_hash)
    if "fine" in frozen:
        state = frozen["fine"].state
        cascade.stats = Statistics.load(state["statistics"])
        frozen_split = state["split_hash"]
        for stage, item in frozen.items():
            if item.state["split_hash"] != frozen_split:
                raise ValueError("Checkpoint bundle contains different training splits/cohorts")
            if stage in ("body", "anchors"):
                key = "body" if stage == "body" else "anchor"
                matching = Statistics.load(item.state["statistics"])
                if tensor_state_hash(getattr(matching, key).state()) != tensor_state_hash(
                        getattr(cascade.stats, key).state()):
                    raise ValueError(f"Fine stage has incompatible {key} normalization")
            expected = state["dependencies"].get(stage)
            if require_fine_dependencies and expected is not None and item.sha256 != expected:
                raise ValueError(f"Fine stage was trained against a different {stage} checkpoint")
        # Uniform and calibrated confidence can use the same frozen planner.
        # An exact confidence dependency is preserved for a fine model trained with learned q.
        expected = state["dependencies"].get("confidence")
        if require_fine_dependencies and expected and confidence_hash and confidence_hash != expected:
            raise ValueError("Fine stage was trained against a different confidence checkpoint")
    return cascade


def _run_train(args):
    cfg = _config(args)
    device = _device(args)
    manifest, rig_path = _paths(args, cfg)
    rig = Rig.load(rig_path)
    if args.stage == "anchors" and cfg["anchors"]["policy"] == "none":
        raise ValueError("No-anchor variants do not have an anchor training stage")
    train = CanonicalDataset(manifest, rig, ("train",), args.cohort, cfg["data"]["fps"])
    val = CanonicalDataset(manifest, rig, ("dev_tune",), args.cohort, cfg["data"]["fps"])
    paths = _checkpoint_paths(args, required_dependencies(args.stage, cfg))
    run = _run_dir(cfg)
    directory = args.stage_dir or run / "stages" / args.stage
    cache_root = run.parent.parent / f"text_cache_{cfg['experiment']['seed']}"
    cascade = create_cascade(cfg, rig, device, paths, split_hash(manifest, args.cohort),
                             args.text_cache or cache_root,
                             args.condition_cache or run / "conditions")
    try:
        ckpt = train_stage(args.stage, cascade, train, val, directory, args.cohort,
                           args.resume, args.stop_after)
    except Exception as error:
        write_json(Path(directory) / "failure.json", {"stage": args.stage,
                   "type": type(error).__name__, "message": str(error)})
        raise
    return {"checkpoint": str(ckpt), "sha256": file_hash(ckpt)}


def _prompts(args):
    if args.text is not None:
        return [{"clip_id": args.clip_id, "text": args.text}]
    prompts = read_jsonl(args.texts)
    if not prompts or any(set(row) != {"clip_id", "text"} or
                          any(not isinstance(row[k], str) or not row[k].strip()
                              for k in ("clip_id", "text")) for row in prompts):
        raise ValueError("Prompt JSONL rows require exactly clip_id and text")
    if len({p["clip_id"] for p in prompts}) != len(prompts):
        raise ValueError("Duplicate clip_id in prompts")
    return prompts


def _run_infer(args):
    cfg = _config(args)
    device = _device(args)
    rig = Rig.load(args.rig)
    paths = _checkpoint_paths(args, inference_dependencies("text_only", cfg))
    cascade = create_cascade(cfg, rig, device, paths, require_fine_dependencies=True)
    out = Path(args.out)
    prompts = _prompts(args)
    rows = []
    for prompt in prompts:
        cid, text = prompt["clip_id"], prompt["text"]
        for idx in range(cfg["sample"]["samples_per_text"]):
            result = cascade.generate([text], [cid], sample_index=idx)
            stem = digest([cid, idx])[:20]
            pose = result["pose"]
            save_npz(out / f"{stem}.npz", timestamps_s=result["times"][0].cpu().numpy(),
                     local_rot6d=matrix_to_rot6d(pose.local[0]).cpu().numpy(),
                     joint_names=JOINTS, joints_m=pose.joints[0].cpu().numpy(),
                     tips_m=pose.tips[0].cpu().numpy())
            rows.append(dict(clip_id=cid, sample_index=idx, path=f"{stem}.npz",
                             frames=int(result["lengths"][0]), length_clamped=bool(
                                 result["length_clamped"][0]), text_truncated=bool(
                                 result["truncated"][0]), nfe=result["nfe"],
                             latency_ms=result["latency_ms"]))
    write_jsonl(out / "predictions.jsonl", rows)
    write_json(out / "run.json", {"schema": 1, "protocol": "text_only",
                                "rig_sha256": rig.sha256, "seed": cascade.seed, "config": cfg,
                                "checkpoints": {name: file_hash(p) for name, p in paths.items()},
                                "versions": runtime_versions(device),
                                "scope": "manual_signing_head_neck_rotations"})
    return {"output": str(out), "samples": len(rows)}


def _lock_contract(args, cfg, manifest, rig, commit):
    return {"schema": 1, "commit": commit, "split": "test", "protocol": "text_only",
            "config_sha256": digest(cfg), "manifest_sha256": file_hash(manifest),
            "cohort_sha256": file_hash(args.cohort) if args.cohort else None,
            "rig_sha256": rig.sha256, "bundle_sha256": file_hash(args.bundle),
            "calibration_sha256": file_hash(args.calibration)}


def _lock_test(args):
    cfg = _config(args)
    if cfg["experiment"]["pilot"] != "p3":
        raise ValueError("Final test locks are for P3")
    manifest, rig_path = _paths(args, cfg)
    rig = Rig.load(rig_path)
    code = git_info()
    if code["dirty_except_test_log"] or not code["commit"]:
        raise ValueError("Test lock requires a clean source tree at a recorded commit")
    device = _device(args)
    paths = _checkpoint_paths(args, inference_dependencies("text_only", cfg))
    cascade = create_cascade(cfg, rig, device, paths, split_hash(manifest, args.cohort),
                             require_fine_dependencies=True)
    from .evaluation import validate_calibration

    calibration = json.loads(args.calibration.read_bytes())
    validate_calibration(cascade, manifest, calibration)
    value = _lock_contract(args, cfg, manifest, rig, code["commit"])
    write_json(args.out, value)
    return value


def _run_eval(args):
    cfg = _config(args)
    device = _device(args)
    manifest, rig_path = _paths(args, cfg)
    rig = Rig.load(rig_path)
    paths = _checkpoint_paths(args, inference_dependencies(args.protocol, cfg))
    cascade = create_cascade(cfg, rig, device, paths, split_hash(manifest, args.cohort),
                             args.text_cache, require_fine_dependencies=True)
    # Cohorts select training/development lists. The canonical test split is fixed.
    if args.split == "test" and args.eval_cohort:
        raise ValueError("Final test uses the full canonical test split")
    evaluation_cohort = (args.eval_cohort or args.cohort) if args.split != "test" else None
    data = CanonicalDataset(manifest, rig, (args.split,), evaluation_cohort, cfg["data"]["fps"])
    out = args.out or _run_dir(cfg) / "eval" / args.protocol / args.split
    if Path(out, "protocol.json").exists():
        raise ValueError("Evaluation output already exists. Select a separate output directory")
    calibration = json.loads(Path(args.calibration).read_bytes()) if args.calibration else None
    if args.split == "test":
        if (args.protocol != "text_only" or not args.bundle or not args.calibration or
                not args.test_lock):
            raise ValueError("Final test requires --test-lock, --bundle and --calibration")
        from slp.paths import TEST_ACCESS_LOG
        import fcntl

        code = git_info()
        if code["dirty_except_test_log"] or not code["commit"]:
            raise ValueError("Final test requires a clean source tree at a recorded commit")
        expected = _lock_contract(args, cfg, manifest, rig, code["commit"])
        if json.loads(args.test_lock.read_bytes()) != expected:
            raise ValueError("Test lock does not match the final evaluation contract")
        lock_sha = digest(expected)
        TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TEST_ACCESS_LOG.open("a+", encoding="utf-8") as log:
            fcntl.flock(log, fcntl.LOCK_EX)
            log.seek(0)
            if any(json.loads(line).get("lock_sha256") == lock_sha
                   for line in log if line.strip()):
                raise ValueError("This test lock already has an access record")
            log.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                  "experiment": "slp_multirate", "split": "test",
                                  "seed": cfg["experiment"]["seed"], "commit": code["commit"],
                                  "bundle_sha256": file_hash(args.bundle),
                                  "calibration_sha256": file_hash(args.calibration),
                                  "lock_sha256": lock_sha, "run_dir": str(out)}) + "\n")
            log.flush()
    return run_evaluation(cascade, data, args.protocol, out, cfg["experiment"]["seed"],
                          calibration, file_hash(evaluation_cohort) if evaluation_cohort else None)


def _populate_text(dataset, cascade, batch_size=16):
    if cascade.text.root is None:
        raise ValueError("Enable text.cache_hidden and supply --text-cache")
    counts = {}
    for split in sorted({row["split"] for row in dataset.rows}):
        texts = [row["text"] for row in dataset.rows if row["split"] == split]
        n = 0
        for i in range(0, len(texts), batch_size):
            n += int(cascade.text(texts[i:i + batch_size], split).truncated.sum())
        counts[split] = {"clips": len(texts), "truncated": n}
    return counts


def _verify_stage(stage_dir, rig=None, manifest=None):
    stage_dir = Path(stage_dir)
    info = json.loads((stage_dir / "run.json").read_bytes())
    state = load_torch(stage_dir / "latest.pt")
    if any(state.get(k) != value for k, value in info["contract"].items()):
        raise ValueError("Stage latest and declared run contract mismatch")
    body = (stage_dir / "best.pt")
    if body.exists():
        best = load_torch(body)
        if any(best.get(k) != value for k, value in info["contract"].items()) or (
            best["update"] > state["update"]
        ):
            raise ValueError("Stage best checkpoint has invalid contract/update")
    if rig and state["stage_contract"]["rig_sha256"] != Rig.load(rig).sha256:
        raise ValueError("Verification rig hash mismatch")
    manifest = manifest or stage_dir / "dataset.manifest.jsonl"
    cohort = stage_dir / "cohort.json"
    if split_hash(manifest, cohort if cohort.exists() else None) != state["split_hash"]:
        raise ValueError("Verification split/cohort hash mismatch")
    resolved = resolve(path=stage_dir / "config.resolved.yaml")
    if digest(resolved) != state["config_hash"]:
        raise ValueError("Verification resolved config hash mismatch")
    metrics = read_jsonl(stage_dir / "metrics.jsonl")
    updates = [r["update"] for r in metrics if r["split"] == "train"]
    if updates != list(range(1, state["update"] + 1)):
        raise ValueError("Logged updates differ from checkpoint")
    return {"verified": True, "stage": state["stage_contract"]["stage"], "updates": state["update"],
            "split_sha256": state["split_hash"], "latest_sha256": file_hash(stage_dir / "latest.pt")}


def _smoke(args):
    cfg = _config(args)
    if cfg["text"]["encoder"] != "mock":
        raise ValueError("Synthetic smoke requires text.encoder=mock")
    device = _device(args)
    if args.threads is None and device.type == "cpu":
        torch.set_num_threads(1)
    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="slp-multirate-smoke-"))
    raw, rig_path = create_synthetic(root / "raw", cfg["experiment"]["seed"])
    manifest = prepare_data(raw, rig_path, "synthetic", root / "canonical", cfg["data"])
    rig = Rig.load(rig_path)
    cohort = root / "cohort.json"
    make_cohort(manifest, rig, "p0", cfg["experiment"]["seed"], cohort)
    train = CanonicalDataset(manifest, rig, ("train",), cohort)
    dev = CanonicalDataset(manifest, rig, ("dev_tune",), cohort)
    gate = CanonicalDataset(manifest, rig, ("dev_gate",), cohort)
    key = split_hash(manifest, cohort)
    files, verified = {}, {}
    for stage in STAGES:
        if stage == "anchors" and cfg["anchors"]["policy"] == "none":
            continue
        directory = root / "stages" / stage
        files_needed = {k: files[k] for k in required_dependencies(stage, cfg)}
        cascade = create_cascade(cfg, rig, device, files_needed, key, root / "text",
                                 root / "conditions")
        updates = cfg["train"]["stages"][stage]["total_updates"]
        if updates > 1:
            train_stage(stage, cascade, train, dev, directory, cohort, stop_after=updates // 2)
        files[stage] = train_stage(stage, cascade, train, dev, directory, cohort, resume=updates > 1)
        verified[stage] = _verify_stage(directory, rig_path, manifest)
    overfit = None
    if args.tiny_overfit:
        tiny_cfg = resolve("p0", cfg["experiment"]["variant"], overrides=[
            f"experiment.seed={cfg['experiment']['seed']}",
            "train.condition_prob.predicted=0.0", "train.condition_prob.oracle_jitter=1.0",
            "train.condition_prob.no_anchor=0.0",
        ])
        toy = create_cascade(tiny_cfg, rig, device, text_cache=root / "text")
        overfit = tiny_overfit(toy, train, root / "tiny_overfit.json")
    bundle = create_bundle(files, root / "bundle.json")
    cascade = create_cascade(cfg, rig, device, files, key)
    result = cascade.generate(["toy sign 1 length 13"], ["synthetic-text-only"])
    if not result["pose"].joints.isfinite().all():
        raise FloatingPointError("Smoke generation emitted invalid joints")
    for protocol in ("oracle", "text_only"):
        run_evaluation(cascade, gate, protocol, root / "eval" / protocol,
                       cfg["experiment"]["seed"], cohort_hash=file_hash(cohort))
    return {"output": str(root), "bundle": str(bundle), "verified": verified, "tiny_overfit": overfit, "updates": {
        stage: cfg["train"]["stages"][stage]["total_updates"] for stage in files}}


def _common(parser, data=True):
    parser.add_argument("--pilot", choices=PILOTS)
    parser.add_argument("--variant", choices=(*VARIANTS, "E", "F"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    if data:
        parser.add_argument("--manifest", type=Path)
        parser.add_argument("--rig", type=Path)
        parser.add_argument("--cohort", type=Path)
        parser.add_argument("--text-cache", type=Path)
        parser.add_argument("--condition-cache", type=Path)


def _checkpoint_args(parser, bundle=True):
    if bundle:
        parser.add_argument("--bundle", type=Path)
    for stage in CHECKPOINTS:
        parser.add_argument(f"--{stage}-ckpt", type=Path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("synthetic", help="Write a synthetic rig and golden pose fixture")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", default=17, type=int)

    p = subs.add_parser("prepare")
    _common(p)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--adapter", required=True, choices=("synthetic", "named_npz", "signspark_lmdb"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split-seed", default=17, type=int)

    for command in ("audit", "make-cohort", "cache-text", "cache-conditions", "calibrate-confidence",
                    "calibrate-eval", "train", "evaluate"):
        p = subs.add_parser(command)
        _common(p)
        p.add_argument("--out", type=Path)
        if command in ("cache-conditions", "calibrate-confidence", "calibrate-eval", "train", "evaluate"):
            _checkpoint_args(p)
        if command in ("audit", "cache-text"):
            p.add_argument("--split", default="all" if command == "audit" else "train,dev_tune,dev_cal")
        if command == "train":
            p.add_argument("--stage", required=True, choices=STAGES)
            p.add_argument("--stage-dir", type=Path)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--stop-after", type=int)
        if command == "calibrate-eval":
            p.add_argument("--contact-audit", type=Path)
        if command == "calibrate-confidence":
            p.add_argument("--updates", type=int, default=300)
        if command == "evaluate":
            p.add_argument("--split", choices=("dev_tune", "dev_gate", "test"), default="dev_tune")
            p.add_argument("--protocol", choices=PROTOCOLS, required=True)
            p.add_argument("--calibration", type=Path)
            p.add_argument("--test-lock", type=Path)
            p.add_argument("--eval-cohort", type=Path)

    p = subs.add_parser("lock-test")
    _common(p)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = subs.add_parser("smoke")
    _common(p, data=False)
    p.set_defaults(pilot="p0")
    p.add_argument("--out", type=Path)
    p.add_argument("--tiny-overfit", action="store_true")

    p = subs.add_parser("infer")
    _common(p, data=False)
    _checkpoint_args(p)
    p.add_argument("--rig", required=True, type=Path)
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--text")
    prompts.add_argument("--texts", type=Path)
    p.add_argument("--clip-id", default="prompt-0")
    p.add_argument("--out", required=True, type=Path)

    p = subs.add_parser("profile")
    _common(p, data=False)
    _checkpoint_args(p)
    p.add_argument("--rig", required=True, type=Path)
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--text")
    prompts.add_argument("--texts", type=Path)
    p.add_argument("--clip-id", default="profile-0")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--warmup", default=10, type=int)
    p.add_argument("--repeats", default=50, type=int)
    p.add_argument("--no-flops", action="store_true")

    p = subs.add_parser("bundle")
    _checkpoint_args(p, bundle=False)
    p.add_argument("--out", required=True, type=Path)

    p = subs.add_parser("compare")
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--baseline-text", type=Path)
    p.add_argument("--candidate-text", type=Path)
    p.add_argument("--gate", choices=("p1", "p2"))
    p.add_argument("--out", type=Path)

    p = subs.add_parser("compare-compute")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metric", choices=("total_p50_ms", "flops"), default="total_p50_ms")

    p = subs.add_parser("signer-diagnostic")
    _common(p)
    p.add_argument("--private-source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    p = subs.add_parser("p0-readiness")
    _common(p)
    p.add_argument("--geometry-review", type=Path)
    p.add_argument("--out", type=Path, required=True)

    p = subs.add_parser("review-geometry")
    _common(p)
    p.add_argument("--clip-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--render", action="store_true")

    p = subs.add_parser("pilot-plan")
    p.add_argument("--pilot", choices=("p1", "p2", "p3"), required=True)
    p.add_argument("--out", type=Path)

    p = subs.add_parser("compare-matrix")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = subs.add_parser("verify-run")
    check = p.add_mutually_exclusive_group(required=True)
    check.add_argument("--stage-dir", type=Path)
    check.add_argument("--bundle", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--rig", type=Path)
    return parser


def dispatch(args):
    cmd = args.command
    handlers = {"train": _run_train, "evaluate": _run_eval, "infer": _run_infer, "smoke": _smoke,
                "lock-test": _lock_test}
    if cmd in handlers:
        return handlers[cmd](args)
    if cmd == "compare-compute":
        from .profiling import compare_compute

        return compare_compute(args.baseline, args.candidate, args.out, args.metric)
    if cmd == "profile":
        from .profiling import profile_text

        cfg = _config(args)
        device, rig = _device(args), Rig.load(args.rig)
        cascade = create_cascade(cfg, rig, device, _checkpoint_paths(
            args, inference_dependencies("text_only", cfg)), require_fine_dependencies=True)
        return profile_text(cascade, _prompts(args), args.out, args.warmup, args.repeats,
                            not args.no_flops)
    if cmd in ("pilot-plan", "compare-matrix"):
        from .pilots import compare_matrix, plan

        if cmd == "compare-matrix":
            return compare_matrix(args.manifest, args.out)
        result = plan(args.pilot)
        if args.out:
            write_json(args.out, result)
        return result
    if cmd == "synthetic":
        paths = create_synthetic(args.out, args.seed)
        return {"manifest": str(paths[0]), "rig": str(paths[1])}
    if cmd == "bundle":
        paths = {k: getattr(args, k + "_ckpt") for k in CHECKPOINTS if getattr(args, k + "_ckpt")}
        if not paths:
            raise ValueError("Bundle needs at least one checkpoint")
        return {"bundle": str(create_bundle(paths, args.out))}
    if cmd == "compare":
        if bool(args.baseline_text) != bool(args.candidate_text):
            raise ValueError("Provide both text-only run paths")
        return compare(args.baseline, args.candidate, args.gate,
                       args.out or args.candidate / "compare", args.baseline_text, args.candidate_text)
    if cmd == "verify-run":
        if args.stage_dir:
            return _verify_stage(args.stage_dir, args.rig, args.manifest)
        return {"verified": True, "checkpoints": {k: str(p) for k, p in _bundle(args.bundle).items()}}

    cfg = _config(args)
    manifest, rig_path = _paths(args, cfg)
    if cmd == "prepare":
        if args.rig is None and args.adapter == "synthetic":
            rig_path = args.source.parent / "rig.json"
        if not args.out:
            raise ValueError("Prepare requires --out")
        return {"manifest": str(prepare_data(args.source, rig_path, args.adapter, args.out,
                                            cfg["data"], args.split_seed))}
    rig = Rig.load(rig_path)
    run = _run_dir(cfg)
    if cmd == "audit":
        value = audit_manifest(manifest, rig, args.split)
        if args.out:
            write_json(args.out / "audit.json", value)
        return {k: v for k, v in value.items() if k != "clips"}
    if cmd == "signer-diagnostic":
        return make_signer_diagnostic(args.private_source, manifest, rig, args.out)
    if cmd == "review-geometry":
        from .review_geometry import review_geometry

        return review_geometry(manifest, rig, args.clip_id, args.out, cfg["body"]["stride"],
                               args.render)
    if cmd == "p0-readiness":
        from .evaluation import p0_readiness

        training = CanonicalDataset(manifest, rig, ("train",), args.cohort, cfg["data"]["fps"])
        gate = CanonicalDataset(manifest, rig, ("dev_gate",), args.cohort, cfg["data"]["fps"])
        review = json.loads(args.geometry_review.read_bytes()) if args.geometry_review else None
        result = p0_readiness(gate, training, rig, cfg, review)
        write_json(args.out, result)
        return result
    if cmd == "make-cohort":
        counts = make_cohort(manifest, rig, cfg["experiment"]["pilot"],
                             cfg["experiment"]["seed"], args.out or run / "cohort.json")
        return {key: len(ids) for key, ids in counts.items()}
    device = _device(args)
    key = split_hash(manifest, args.cohort)
    if cmd == "cache-text":
        splits = tuple(args.split.split(","))
        if not set(splits) <= (set(SPLITS) - {"test"}):
            raise ValueError("Text cache split must be train/dev_tune/dev_gate/dev_cal")
        dataset = CanonicalDataset(manifest, rig, splits, args.cohort, cfg["data"]["fps"])
        root = args.text_cache or args.out or run.parent.parent / (
            f"text_cache_{cfg['experiment']['seed']}"
        )
        cascade = create_cascade(cfg, rig, device, text_cache=root)
        return _populate_text(dataset, cascade)
    if cmd == "cache-conditions":
        deps = required_dependencies("fine", cfg)
        if not deps:
            raise ValueError("Predicted-conditions cache requires a predicted-body training mixture")
        paths = _checkpoint_paths(args, deps)
        data = CanonicalDataset(manifest, rig, ("train",), args.cohort, cfg["data"]["fps"])
        root = args.condition_cache or args.out or run / "conditions"
        cascade = create_cascade(cfg, rig, device, paths, key, args.text_cache, root)
        training_statistics(cascade, data)
        for clip in data:
            cascade.cached_prediction(clip, bool(cfg["train"]["condition_prob"]["predicted"]))
        return {"clips": len(data), "out": str(root)}
    if cmd == "calibrate-confidence":
        if args.updates < 1:
            raise ValueError("Calibration needs positive updates")
        paths = _checkpoint_paths(args, ("anchors",))
        data = CanonicalDataset(manifest, rig, ("dev_cal",), args.cohort, cfg["data"]["fps"])
        cascade = create_cascade(cfg, rig, device, paths, key, args.text_cache)
        return fit_confidence(cascade, data, args.out or run / "confidence.pt", args.updates)
    if cmd == "calibrate-eval":
        paths = _checkpoint_paths(args, ("fine",))
        data = CanonicalDataset(manifest, rig, ("dev_tune",), args.cohort, cfg["data"]["fps"])
        cascade = create_cascade(cfg, rig, device, paths, key, args.text_cache)
        audit = json.loads(args.contact_audit.read_bytes()) if args.contact_audit else None
        result = fit_evaluator(data, rig, cascade.stats.event_scales, cfg, audit)
        write_json(args.out or run / "evaluation_calibration.json", result)
        return result
    raise ValueError(f"Unsupported subcommand {cmd}")


def main(argv=None):
    parser = build_parser()
    try:
        result = dispatch(parser.parse_args(argv))
    except (ValueError, FileNotFoundError, FloatingPointError) as error:
        parser.exit(2, f"slp.cli: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
