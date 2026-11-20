"""Independent stage training with deterministic optimizer-boundary resume."""

from __future__ import annotations

import copy
import fcntl
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .anchors import confidence_labels, reliability
from .data import atomic_bytes, collate_clips, digest, file_hash, stable_seed, write_json, write_jsonl
from .flow import anchor_loss, body_objective, fine_objective, length_loss, linear_path
from .geometry import build_fine_target, matrix_to_rot6d, rot6d_to_matrix
from .models import Condition, make_model
from .pipeline import (
    FrozenStage, Statistics, atomic_torch, load_torch, stage_contract, tensor_state_hash,
    training_statistics,
)


def split_hash(manifest, cohort):
    return digest([file_hash(manifest), file_hash(cohort) if cohort is not None else None])


def runtime_versions(device):
    deps = {}
    for name in ("torch", "numpy", "PyYAML", "transformers", "sentencepiece", "lmdb"):
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    return {"python": platform.python_version(), "packages": deps, "device": str(device),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def git_info():
    root = Path(__file__).resolve().parents[2]

    def command(*args):
        result = subprocess.run(["git", "-C", str(root), *args],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": command("rev-parse", "HEAD"), "dirty": bool(command("status", "--porcelain")),
            "dirty_except_test_log": bool(command("status", "--porcelain", "--", ".",
                                                  ":(exclude)results/test_access_log.jsonl")),
            "tracked_diff_sha256": digest(command("diff", "HEAD", "--binary")),
            "implementation_sha256": digest([
                (str(f.relative_to(root)), file_hash(f))
                for f in sorted((*Path(__file__).parent.rglob("*.py"),
                                 *Path(__file__).parent.rglob("*.yaml"),
                                 *Path(__file__).parent.rglob("*.txt"),
                                 root / "src/slp/evaluation/multirate.py",
                                 root / "src/slp/geometry.py",
                                 root / "src/slp/cli.py"))
                if f.is_file()
            ])}


def collect_rng():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate()}


def restore_rng(value):
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])
    random.setstate(value["python"])


class BucketStream:
    """Repeat shuffled length-bucketed batches, resumable by epoch and batch offset."""

    def __init__(self, dataset, cfg, seed):
        self.dataset, self.cfg, self.seed = dataset, cfg, seed
        self.epoch = self.offset = 0
        self.batches = None
        self.iterator = None
        if not len(dataset):
            raise ValueError("Empty training split")
        self.bucketed = defaultdict(list)
        for i, row in enumerate(dataset.rows):
            edge = next((b for b in cfg["data"]["bucket_boundaries"] if row["frames"] <= b), None)
            if edge is None or row["frames"] < cfg["data"]["min_frames"]:
                raise ValueError("Canonical/cohort lengths incompatible with training config")
            self.bucketed[edge].append(i)

    def state(self):
        return {"epoch": self.epoch, "offset": self.offset}

    def restore(self, state):
        self.epoch, self.offset = state["epoch"], state["offset"]
        self.batches = self.iterator = None

    def _start(self):
        rng = random.Random(stable_seed(self.seed, "epoch", self.epoch))
        batches = []
        size = self.cfg["train"]["microbatch"]
        for edge in sorted(self.bucketed):
            ids = self.bucketed[edge].copy()
            rng.shuffle(ids)
            batches.extend(ids[i:i + size] for i in range(0, len(ids), size))
        rng.shuffle(batches)
        if not 0 <= self.offset < len(batches):
            raise ValueError("Invalid saved batch offset")
        self.batches = batches
        self.iterator = iter(DataLoader(self.dataset, batch_sampler=batches[self.offset:],
                             collate_fn=collate_clips, num_workers=self.cfg["data"]["num_workers"],
                             generator=torch.Generator().manual_seed(stable_seed(
                                 self.seed, "workers", self.epoch))))

    def __next__(self):
        if self.iterator is None:
            self._start()
        value = next(self.iterator)
        self.offset += 1
        if self.offset == len(self.batches):
            self.epoch, self.offset = self.epoch + 1, 0
            self.batches = self.iterator = None
        return value


def _optimizer(model, cfg, lr):
    decay, no_decay = [], []
    for param in model.parameters():
        (decay if param.ndim >= 2 else no_decay).append(param)
    return torch.optim.AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.}],
                             lr=lr, betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])


def lr_multiplier(index, total, warmup_ratio, minimum):
    warmup = max(1, round(total * warmup_ratio))
    update = index + 1
    if update <= warmup:
        return update / warmup
    frac = min((update - warmup) / max(1, total - warmup), 1)
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * frac))


def _validation_steps(total, every):
    return {max(1, math.ceil(total * .5)), max(1, math.ceil(total * .75)), total} | (
        set(range(every, total, every)) if every else set()
    )


def _precision(cascade):
    return (cascade.device.type == "cuda" and cascade.cfg["train"]["precision"] == "bf16"
            and torch.cuda.is_bf16_supported())


def stage_loss(stage, model, cascade, batch, rng=None, oracle=False):
    cfg = cascade.cfg
    batch = cascade.move_batch(batch)
    split = batch["clips"][0]["row"]["split"]
    text = cascade.text(batch["text"], split)
    calibrated = dict(truncated=text.truncated.sum())
    if stage == "length":
        value, terms = length_loss(model, text, batch["length"])
        if oracle:
            prediction, clamped = model.predict(
                text, cfg["data"]["min_frames"], cfg["data"]["max_frames"]
            )
            terms["relative_errors"] = (prediction - batch["length"]).abs() / batch["length"]
            terms["length_clamped"] = clamped.sum()
    elif stage == "anchors":
        value, terms = anchor_loss(model(text, batch["length"]), cascade.oracle(batch),
                                   batch["length"])
    elif stage == "body":
        grid = cascade.oracle_grid(batch)
        state = matrix_to_rot6d(grid.rotations).flatten(-2)
        endpoint = cascade.stats.body.normalize(state, grid.observation.repeat_interleave(6, -1))
        noisy, tau = linear_path(endpoint, grid.mask, rng)
        cond = Condition(text, grid.timestamps, grid.mask)
        pred = model(noisy, tau, cond)
        value, terms = body_objective(pred, endpoint, grid, cascade.stats.body, tau, cfg["loss"])
        if oracle:
            from .geometry import fk, identity6, merge_local, upsample_body
            # Wrist positions depend on body and wrist rest offsets, even without hand rotation.
            raw = cascade.stats.body.denormalize(pred).unflatten(-1, (12, 6))
            full, body_obs = upsample_body(rot6d_to_matrix(raw), grid.timestamps, grid.mask,
                                          batch["timestamps_s"], batch["frame_mask"], grid.observation)
            hand = rot6d_to_matrix(identity6(*full.shape[:2], 2, 16, device=full.device))
            p = fk(merge_local(full, hand), cascade.rig)
            gt_body = rot6d_to_matrix(batch["body_rot6d"])
            q = fk(merge_local(gt_body, hand), cascade.rig)
            valid = (body_obs & batch["body_obs"]).all(-1) & batch["frame_mask"]
            if valid.any():
                terms["wrist_proxy_mm"] = ((p.wrist - q.wrist).norm(dim=-1)[valid].mean()
                                           * cascade.rig.shoulder_width * 1000)
    else:
        parts = ((*cascade.oracle_body(batch), cascade.oracle(batch)) if oracle else
                 cascade.fine_training_conditions(batch, rng))
        body, seen = parts[3:5]
        cap = cfg["fine"]["correction_max_rad"]
        enabled = cfg["fine"]["correction_enabled"]
        state, obs, capped, cap_error = build_fine_target(
            rot6d_to_matrix(batch["body_rot6d"]), batch["hands_rot6d"], body, batch["body_obs"],
            batch["hands_obs"], seen, cap, enabled)
        endpoint = cascade.stats.fine.normalize(state, obs)
        noisy, tau = linear_path(endpoint, batch["frame_mask"], rng)
        cond = cascade.fine_condition(text, batch["timestamps_s"], batch["frame_mask"], parts)
        pred = model(noisy, tau, cond)
        value, terms = fine_objective(pred, endpoint, obs, batch, body, seen, cascade.stats.fine,
                                      cascade.rig, tau, cfg["loss"], cfg["fine"])
        terms.update(correction_clamped=capped.sum(),
                     correction_valid=obs[..., 192:].sum() // 3,
                     correction_cap_excess_rad=cap_error.sum() / capped.sum().clamp_min(1))
    return value, dict(terms, **calibrated)


@torch.no_grad()
def validate_stage(stage, model, cascade, dataset):
    from .evaluation import evaluate_clip

    values, rel = defaultdict(list), []
    model.eval()
    if stage == "fine":
        cascade.frozen["fine"] = FrozenStage(model, {}, "validation")
    try:
        for clip in dataset:
            rng = torch.Generator(device=cascade.device).manual_seed(stable_seed(
                cascade.seed, "dev_tune_loss", stage, clip["row"]["clip_id"]))
            value, terms = stage_loss(stage, model, cascade, collate_clips([clip]), rng, oracle=True)
            values["loss"].append(float(value))
            for k, v in terms.items():
                if k == "relative_errors":
                    rel.extend(v.tolist())
                else:
                    if k + "_n" not in terms or int(terms[k + "_n"]):
                        values[k].append(float(v))
            if stage == "fine":
                cfg = cascade.cfg
                protocol = ("predicted_gt_length" if cfg["experiment"]["pilot"] in ("p2", "p3")
                            and "body" in cascade.frozen else "oracle")
                evaluation = evaluate_clip(cascade, clip, protocol, cascade.seed, 0)
                for key in sorted({cfg["eval"]["primary_metric"], "wrist_mm", "tip_mm",
                                   "outside_tip_mm", "event_tip_mm", "duration_relative"}):
                    if evaluation.get(key) is not None and key not in terms:
                        values[key].append(float(evaluation[key]))
    finally:
        if stage == "fine":
            cascade.frozen.pop("fine")
    report = {k: float(np.mean(v)) for k, v in values.items()}
    if rel:
        report["median_duration_relative"] = float(np.median(rel))
    key = "median_duration_relative" if stage == "length" else (
        cascade.cfg["eval"]["primary_metric"] if stage == "fine" else "loss"
    )
    if key not in report:
        if stage != "fine" or cascade.cfg["experiment"]["pilot"] != "p0":
            raise ValueError(f"dev_tune has no valid samples for {key}")
        key = "tip_mm"
    if key not in report or not math.isfinite(report[key]):
        raise ValueError("Non-finite or missing validation score")
    report["selection_metric"] = key
    return report, report[key]


def _stage_log(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def _snapshot_data(directory, dataset, cohort):
    for src, dest in ((dataset.manifest, "dataset.manifest.jsonl"), (cohort, "cohort.json"),
                      (dataset.manifest.parent / "audit.json", "dataset.audit.json"),
                      (dataset.manifest.parent / "prepare.json", "dataset.prepare.json")):
        if src is not None and Path(src).is_file():
            atomic_bytes(directory / dest, Path(src).read_bytes())


def train_stage(stage, cascade, train, val, directory, cohort=None, resume=False, stop_after=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another training process holds {directory}") from error
        return _train_stage(stage, cascade, train, val, directory, cohort, resume, stop_after)


def _train_stage(stage, cascade, train, val, directory, cohort, resume, stop_after):
    if not len(val) or any(r["split"] != "dev_tune" for r in val.rows):
        raise ValueError("Checkpoint selection requires a nonempty dev_tune split")
    if any(row["split"] != "train" for row in train.rows):
        raise ValueError("Training can only read the train split")
    cfg = cascade.cfg
    settings = cfg["train"]
    stage_cfg = settings["stages"][stage]
    latest, best_path = (directory / file for file in ("latest.pt", "best.pt"))
    if resume != latest.exists():
        raise ValueError("Use --resume with an existing latest.pt or choose a new stage directory")
    seed = int(cascade.seed)
    all_rng = stable_seed(seed, stage)
    torch.manual_seed(all_rng)
    random.seed(all_rng)
    source_versions = runtime_versions(cascade.device)
    source_git = git_info()
    split_key = split_hash(train.manifest, cohort)
    dependencies = {name: dep.sha256 for name, dep in cascade.frozen.items()}
    if cascade.confidence_hash is not None:
        dependencies["confidence"] = cascade.confidence_hash
    contract = {"stage_contract": stage_contract(stage, cfg, cascade.rig, cascade.text),
                "config_hash": digest(cfg), "split_hash": split_key, "seed": seed,
                "dependencies": dependencies, "versions": source_versions,
                "code_hash": source_git["implementation_sha256"]}
    model = make_model(stage, cascade.text.dim, cfg).to(cascade.device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = _optimizer(model, settings, stage_cfg["lr"])
    total = stage_cfg["total_updates"]

    def schedule(i):
        return lr_multiplier(i, total, settings["warmup_ratio"], settings["min_lr_ratio"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    stream = BucketStream(train, cfg, all_rng)
    update, best, best_update = 0, float("inf"), 0
    if resume:
        state = load_torch(latest)
        mismatches = [k for k, v in contract.items() if state.get(k) != v]
        if mismatches:
            raise ValueError(f"Resume contract mismatch: {', '.join(mismatches)}")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stream.restore(state["sampler"])
        update, best, best_update = state["update"], state["best"], state["best_update"]
        # Metric appends can be ahead of the last published optimizer boundary.
        log = directory / "metrics.jsonl"
        metrics = []
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                break
            if item["update"] <= update:
                metrics.append(item)
        if [r["update"] for r in metrics if r["split"] == "train"] != list(range(1, update + 1)):
            raise ValueError("Committed checkpoint metrics are missing or inconsistent")
        write_jsonl(log, metrics)
        if best_update == update:
            atomic_torch(best_path, state)
        elif best_update and (not best_path.exists() or load_torch(best_path)["update"] != best_update):
            raise ValueError("Saved best checkpoint does not match the optimizer boundary")
        if state["statistics"] is not None:
            cascade.stats = Statistics.load(state["statistics"])
        restore_rng(state["rng"])
    else:
        if stage != "length":
            training_statistics(cascade, train, need_fine=stage == "fine")
        atomic_bytes(directory / "config.resolved.yaml", yaml.safe_dump(cfg, sort_keys=False).encode())
        write_json(directory / "versions.json", dict(source_versions, git=source_git,
                   command=sys.argv, precision="bf16" if _precision(cascade) else "fp32"))
        write_json(directory / "stage_condition_hashes.json", dependencies)
        write_json(directory / "run.json", {
            "contract": contract, "seed_policy": "SHA256(run,stage,sample_index,clip_id)",
            "stats_sha256": None if cascade.stats is None else tensor_state_hash(cascade.stats.state())})
        _snapshot_data(directory, train, cohort)
        write_jsonl(directory / "metrics.jsonl", [])
    finish = min(total, stop_after) if stop_after is not None else total
    if finish < update or finish < 1:
        raise ValueError("Invalid optimizer update boundary")
    val_steps = _validation_steps(total, settings["val_every_updates"])
    precision = _precision(cascade)
    accumulation = settings["gradient_accumulation"]
    while update < finish:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logged = defaultdict(list)
        for _ in range(accumulation):
            batch = next(stream)
            with torch.autocast(device_type=cascade.device.type, dtype=torch.bfloat16,
                                enabled=precision):
                value, terms = stage_loss(stage, model, cascade, batch)
            if not value.isfinite():
                raise FloatingPointError(f"{stage} loss is non-finite before update {update + 1}")
            (value / accumulation).backward()
            logged["loss"].append(float(value.detach()))
            for key, item in terms.items():
                if item.ndim == 0:
                    logged[key].append(float(item))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"],
                                              error_if_nonfinite=True)
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        with torch.no_grad():
            for average, param in zip(ema.parameters(), model.parameters()):
                average.lerp_(param, 1 - settings["ema_decay"])
        update += 1
        record = {"update": update, "grad_norm": float(norm), "lr": lr, "split": "train"}
        _stage_log(directory / "metrics.jsonl", dict(record, **{
            k: float(np.mean(v)) for k, v in logged.items()
        }))
        improved = False
        if update in val_steps:
            values, score = validate_stage(stage, ema, cascade, val)
            _stage_log(directory / "metrics.jsonl", dict(values, update=update, split="dev_tune"))
            if score < best:
                best, best_update, improved = score, update, True
        periodic = update % settings["checkpoint_every_updates"] == 0
        if improved or periodic or update == finish:
            full = dict(contract, update=update, best=best, best_update=best_update, model=model.state_dict(),
                        ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=None, rng=collect_rng(),
                        sampler=stream.state(), statistics=None if cascade.stats is None
                        else cascade.stats.state())
            atomic_torch(latest, full)
            if improved:
                atomic_torch(best_path, full)
            if periodic:
                atomic_torch(directory / f"update_{update:08d}.pt", full)
                keep = settings["keep_last"]
                for path in sorted(directory.glob("update_*.pt"))[:-keep]:
                    path.unlink()
    return best_path if best_path.exists() else latest


def fit_confidence(cascade, data, out, updates=300):
    if not len(data) or any(row["split"] != "dev_cal" for row in data.rows):
        raise ValueError("Calibration requires dev_cal, distinct from dev_tune and dev_gate")
    stage = cascade.frozen["anchors"]
    cascade.stats = Statistics.load(stage.state["statistics"])
    videos = sorted({row["origin_video_id"] for row in data.rows},
                    key=lambda name: stable_seed(cascade.seed, "calibration", name))
    if len(videos) < 2:
        raise ValueError("Calibration requires at least two dev_cal video groups")
    train_vids = set(videos[:max(1, len(videos) * 2 // 3)])
    folds = [{k: [] for k in ("x", "y", "mask")} for _ in range(2)]
    with torch.no_grad():
        for clip in data:
            row = clip["row"]
            fold = folds[int(row["origin_video_id"] not in train_vids)]
            batch = cascade.move_batch(collate_clips([clip]))
            target = cascade.oracle(batch)
            text = cascade.text(batch["text"], "dev_cal")
            pred = stage.model(text, batch["length"])
            label, mask = confidence_labels(pred, target, batch["length"],
                                            cascade.stats.anchor.denormalize)
            for key, x in (("x", pred.hidden[0]), ("y", label[0]), ("mask", mask[0])):
                fold[key].append(x.detach())
    fit, check = ({k: torch.cat(v) for k, v in fold.items()} for fold in folds)
    if not fit["mask"].any() or not check["mask"].any():
        raise ValueError("Both confidence folds must have observed hand labels")
    torch.manual_seed(stable_seed(cascade.seed, "confidence"))
    head = torch.nn.Linear(cascade.cfg["anchors"]["width"], 2).to(cascade.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    for _ in range(updates):
        logits = head(fit["x"])[fit["mask"]]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, fit["y"][fit["mask"]].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prob = head(check["x"]).sigmoid()[check["mask"]]
        labels = check["y"][check["mask"]]
        metrics = reliability(prob, labels)
        metrics["uniform_baseline"] = reliability(torch.ones_like(prob), labels)
    atomic_torch(out, {"ema": head.state_dict(), "planner_sha256": stage.sha256,
                       "rig_sha256": cascade.rig.sha256, "metrics": metrics,
                       "split": "dev_cal", "groups_fit": len(train_vids),
                       "groups_check": len(videos) - len(train_vids)})
    return metrics


def tiny_overfit(cascade, dataset, out, updates=300):
    """Memorize fixed synthetic clips and fixed paths as an optimizer/gradient test."""
    if updates < 40 or len(dataset) < 8:
        raise ValueError("Tiny-overfit requires 8 synthetic clips and at least 40 updates")
    if any(row["provenance"] != "synthetic-fixture-v1" for row in dataset.rows):
        raise ValueError("Tiny-overfit is restricted to the synthetic fixture")
    if cascade.cfg["train"]["condition_prob"]["oracle_jitter"] != 1:
        raise ValueError("Tiny-overfit uses deterministic oracle conditioning")
    training_statistics(cascade, dataset, need_fine=True)
    batch = collate_clips([dataset[i] for i in range(8)])
    seed = stable_seed(cascade.seed, "tiny-overfit")
    torch.manual_seed(seed)
    model = make_model("fine", cascade.text.dim, cascade.cfg).to(cascade.device)
    optimizer = _optimizer(model, cascade.cfg["train"], .001)
    model.train()
    losses = []
    for _ in range(updates):
        generator = torch.Generator(device=cascade.device).manual_seed(seed)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = stage_loss("fine", model, cascade, batch, generator, oracle=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    initial, final = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
    result = {"synthetic": True, "stage": "fine", "updates": updates, "clips": 8,
              "seed": seed, "initial_20_mean": initial, "final_20_mean": final,
              "relative_reduction": 1 - final / max(initial, 1e-8), "pass": final <= .5 * initial,
              "fixed_paths": True, "condition": "oracle"}
    write_json(out, result)
    if not result["pass"]:
        raise FloatingPointError(f"Synthetic overfit reduction below 50%. See {out}")
    return result
