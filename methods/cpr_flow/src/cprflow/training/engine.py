"""Logical-batch optimizer engine, exact resume, overflow replay and DDP reductions."""

import contextlib
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from cprflow.training.checkpoint import atomic_save, load_checkpoint, restore_rng, rng_state
from cprflow.training.distributed import all_sum, coordinated_error, gather_objects
from cprflow.training.ema import EMA
from cprflow.training.logging import JsonLogger
from cprflow.training.precision import Precision


def seed_everything(seed, deterministic=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def make_optimizer(model, lr, weight_decay):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (no_decay if param.ndim < 2 or name.endswith("bias") else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def make_scheduler(optimizer, warmup, total, floor=0.1):
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    def __init__(self, model, config, provenance, device="cpu", rank=0, world_size=1):
        self.config, self.provenance = config, provenance
        self.device = torch.device(device)
        self.rank, self.world_size = rank, world_size
        self.raw = model.to(device)
        self.model = (
            DistributedDataParallel(
                self.raw,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=config["distributed"]["find_unused_parameters"],
            )
            if world_size > 1
            else self.raw
        )
        t = config["train"]
        self.optimizer = make_optimizer(self.raw, t["lr"], t["weight_decay"])
        self.scheduler = make_scheduler(
            self.optimizer, t["warmup_updates"], t["max_updates"], t["lr_floor_ratio"]
        )
        self.precision = Precision(device, t["precision"])
        self.ema = EMA(self.raw, t["ema_decay"])
        self.step, self.total_batches_seen, self.skipped = 0, 0, 0
        self.parent_global_step = 0
        self.stop_requested = False
        self.best = None
        self.bad_validations = 0
        self.directory = Path(config["run"]["out_dir"]) / config["run"]["id"]
        self.logger = JsonLogger(self.directory / "train.jsonl")
        self.dev_logger = JsonLogger(self.directory / "dev.jsonl")

    def save(self, path, sampler):
        rank_rng = gather_objects(rng_state())
        if self.rank == 0:
            atomic_save(
                dict(
                    format_version=1,
                    model=self.raw.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict(),
                    scaler=self.precision.scaler.state_dict(),
                    ema=self.ema.state_dict(),
                    successful_optimizer_step=self.step,
                    continuation_step=self.step,
                    parent_global_step=self.parent_global_step,
                    total_batches_seen=self.total_batches_seen,
                    skipped=self.skipped,
                    sampler=sampler.state_dict(),
                    rank_rng=rank_rng,
                    world_size=self.world_size,
                    provenance=self.provenance,
                    config=self.config,
                    best=self.best,
                    bad_validations=self.bad_validations,
                ),
                path,
            )
            from cprflow.training.retention import prune_checkpoints

            prune_checkpoints(
                Path(path).parent,
                self.config["train"]["keep_steps"],
                self.config["train"].get("checkpoint_keep_recent", 1),
            )
        if dist.is_initialized():
            dist.barrier()

    def resume(self, path, sampler):
        s = load_checkpoint(path, self.provenance, self.world_size)
        self.raw.load_state_dict(s["model"])
        self.optimizer.load_state_dict(s["optimizer"])
        self.scheduler.load_state_dict(s["scheduler"])
        self.precision.scaler.load_state_dict(s["scaler"])
        self.ema.load_state_dict(s["ema"])
        # EMA checkpoint loaded on CPU; keep shadow on the model device.
        self.ema.shadow = {k: v.to(self.device) for k, v in self.ema.shadow.items()}
        self.step = s["successful_optimizer_step"]
        self.parent_global_step = s["parent_global_step"]
        self.total_batches_seen, self.skipped = s["total_batches_seen"], s["skipped"]
        self.best, self.bad_validations = s.get("best"), s.get("bad_validations", 0)
        sampler.load_state_dict(s["sampler"])
        restore_rng(s["rank_rng"][self.rank])

    def warm_start(self, path):
        s = load_checkpoint(path)
        # Compatibility hashes must agree for all controlled forks.
        for key in (
            "motion_stats",
            "latent_stats",
            "relation_stats",
            "ae",
            "text_revision",
            "asset",
            "manifest",
            "included_train_hash",
        ):
            if self.provenance.get(key) != s["provenance"].get(key):
                raise ValueError(f"Warm start dependency mismatch: {key}")
        weights = s["ema"]["shadow"]
        if getattr(self.raw, "relational", False):
            required = "inputs.4.weight" not in weights
            if self.config["relation"]["warmup_required"] != required:
                raise ValueError("Relation key warm-up must be enabled only for a P4-to-P5 fork")
        if hasattr(self.raw, "warm_start"):
            self.raw.warm_start(weights)
        else:
            self.raw.load_state_dict(weights)
        self.ema = EMA(self.raw, self.config["train"]["ema_decay"])
        self.parent_global_step = s["parent_global_step"] + s["successful_optimizer_step"]

    def run(self, dataset, sampler, objective, validate=None, until=None, logical_objective=False):
        t = self.config["train"]
        target = t["max_updates"] if until is None else min(until, t["max_updates"])
        start = time.perf_counter()
        self.raw.train()
        from cprflow.data.loader import LogicalBatchLoader

        loader = LogicalBatchLoader(
            dataset,
            sampler,
            self.step,
            target,
            self.rank,
            self.world_size,
            self.config["data"]["workers_per_rank"],
            self.config["data"]["pin_memory"] and self.device.type == "cuda",
            self.config["seed"],
        )
        while self.step < target:
            before_rng = rng_state()
            cursor = sampler.state_dict()
            indices = sampler.batch(self.step)
            rows = None
            error = None
            wait_start = time.perf_counter()
            try:
                rows = loader.get(self.step)
                logical_rows = [dataset.rows[i] for i in indices]
                if hasattr(objective, "preflight"):
                    objective.preflight(rows, logical_rows)
            except Exception as exc:
                error = exc
            coordinated_error(error)
            data_wait = time.perf_counter() - wait_start
            self.total_batches_seen += 1
            self.optimizer.zero_grad(set_to_none=True)
            pieces = (
                [rows]
                if logical_objective
                else [
                    rows[i : i + t["microbatch_per_rank"]]
                    for i in range(0, len(rows), t["microbatch_per_rank"])
                ]
            )
            logs = {}
            finite_losses = []
            for i, micro in enumerate(pieces):
                sync = (
                    contextlib.nullcontext()
                    if self.world_size == 1 or i == len(pieces) - 1
                    else self.model.no_sync()
                )
                with sync, self.precision.autocast():
                    vector, terms = objective(self.model, micro, self.step, logical_rows)
                    loss = self.world_size * vector.float().sum() / t["effective_batch"]
                if self.world_size > 1:
                    coordinated_error(
                        None if torch.isfinite(loss) else f"Nonfinite loss at step {self.step}"
                    )
                else:
                    finite_losses.append(torch.isfinite(loss).detach())
                self.precision.scaler.scale(loss).backward()
                for key, value in {"loss": vector, **terms}.items():
                    v = torch.as_tensor(value, device=self.device).detach().float()
                    old_sum, old_count = logs.get(key, (v.new_zeros(()), 0))
                    logs[key] = (old_sum + v.sum(), old_count + v.numel())
            if finite_losses and not torch.stack(finite_losses).all():
                raise FloatingPointError(f"Nonfinite logical-batch loss at step {self.step}")
            self.precision.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), t["grad_clip"])
            if not self.precision.scaler.is_enabled():
                coordinated_error(None if torch.isfinite(norm) else "Nonfinite FP32/BF16 gradient")
            old_scale = self.precision.scaler.get_scale()
            self.precision.scaler.step(self.optimizer)
            self.precision.scaler.update()
            successful = self.precision.scaler.get_scale() >= old_scale
            if not successful:
                self.skipped += 1
                restore_rng(before_rng)
                sampler.load_state_dict(cursor)
                if self.rank == 0:
                    self.logger.log(step=self.step, skipped=self.skipped, overflow=True)
                continue
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite FP32/BF16 gradient norm")
            self.scheduler.step()
            self.ema.update(self.raw)
            self.step += 1
            keys = sorted(logs)
            packed = torch.stack([logs[key][0] for key in keys])
            counts = packed.new_tensor([logs[key][1] for key in keys])
            summed = all_sum(torch.stack((packed, counts), -1)).cpu().tolist()
            reduced = {key: value[0] / max(1, value[1]) for key, value in zip(keys, summed)}
            if self.rank == 0:
                self.logger.log(
                    step=self.step,
                    parent_global_step=self.parent_global_step,
                    elapsed=time.perf_counter() - start,
                    lr=self.optimizer.param_groups[0]["lr"],
                    gradnorm=norm,
                    skipped=self.skipped,
                    total_batches_seen=self.total_batches_seen,
                    data_wait=data_wait,
                    dataset=rows[0]["dataset_id"],
                    duplicate_ids=len(indices) - len(set(indices)),
                    corpus_samples={
                        d: sum(dataset.rows[j]["dataset_id"] == d for j in indices)
                        for d in sampler.datasets
                    },
                    corpus_frames={
                        d: sum(
                            dataset.rows[j]["frames"]
                            for j in indices
                            if dataset.rows[j]["dataset_id"] == d
                        )
                        for d in sampler.datasets
                    },
                    allocated_vram=torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else 0,
                    reserved_vram=torch.cuda.max_memory_reserved(self.device)
                    if self.device.type == "cuda"
                    else 0,
                    **reduced,
                )
            if validate and self.step % t["dev_loss_every"] == 0:
                state = rng_state()
                try:
                    score, details = validate(self)
                except InterruptedError:
                    restore_rng(state)
                    self.raw.train()
                    break
                restore_rng(state)
                self.raw.train()
                improved = self.best is None or score < self.best
                self.best = score if improved else self.best
                self.bad_validations = 0 if improved else self.bad_validations + 1
                if self.rank == 0:
                    self.dev_logger.log(
                        step=self.step, selection_metric=t["selection"], score=score, **details
                    )
                if improved:
                    self.save(self.directory / "checkpoints/best_ema.pt", sampler)
                    if self.config["run"]["task"] == "duration":
                        self.save(self.directory / "checkpoints/best.pt", sampler)
                if (
                    self.config["run"]["task"] == "duration"
                    and self.bad_validations >= t["early_stop_patience"]
                ):
                    break
            if self.stop_requested:
                break
            if self.step % t["checkpoint_every"] == 0 or self.step in t["keep_steps"]:
                self.save(self.directory / f"checkpoints/step_{self.step}_ema.pt", sampler)
                self.save(self.directory / "checkpoints/last.pt", sampler)
        loader.close()
        self.save(self.directory / "checkpoints/last.pt", sampler)
        return self
