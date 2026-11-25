"""Update-based training loop shared by the three stages (warmup + cosine, microbatches, bf16)."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from slp.tracking.run import RunContext

from .data import Batch, collate_microbatches


def make_loader(data: Dataset, cfg: dict, workers: int, device: torch.device, seed: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(
        data, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=workers,
        collate_fn=partial(collate_microbatches, microbatch=cfg["microbatch"]),
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )


def precision_context(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU cannot use BF16; set method.precision=fp32")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    """Linear warmup, then cosine from 1 to ``floor`` (as a fraction of the base rate)."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(params, cfg: dict, extra: tuple[list, float] | None = None
                   ) -> torch.optim.AdamW:
    """AdamW; biases, norm gains and other 1-D parameters are not decayed (ViT / I-JEPA).
    ``extra`` = (parameters, lr scale): a second set trained at ``scale * lr`` (e.g. a fine-tuned
    pretrained text encoder)."""
    def split(ps, lr):
        ps = [p for p in ps if p.requires_grad]
        return [{"params": [p for p in ps if p.ndim >= 2], "weight_decay": cfg["weight_decay"],
                 "lr": lr},
                {"params": [p for p in ps if p.ndim < 2], "weight_decay": 0.0, "lr": lr}]

    params = list({id(p): p for p in params if p.requires_grad}.values())
    groups = split(params, cfg["lr"])
    if extra is not None:
        seen = {id(p) for p in params}
        slow = list({id(p): p for p in extra[0] if id(p) not in seen}.values())
        groups += split(slow, cfg["lr"] * extra[1])
    if not any(g["params"] for g in groups):
        raise ValueError("No trainable parameters")
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=cfg["lr"],
                             betas=tuple(cfg["betas"]))


def batch_losses(batch: list[Batch], criterion: Callable[[Batch], dict], device: torch.device,
                 precision: str, backward: bool) -> dict[str, Tensor]:
    count = sum(len(item.lengths) for item in batch)
    averaged: dict[str, Tensor] = {}
    for item in batch:
        weight = len(item.lengths) / count
        with precision_context(device, precision):
            losses = criterion(item.to(device))
        if backward:
            (weight * losses["loss"]).backward()
        for key, value in losses.items():
            averaged[key] = averaged.get(key, 0) + weight * value.detach().float()
    return averaged


@torch.no_grad()
def evaluate_loader(loader: DataLoader, criterion: Callable[[Batch], dict], device: torch.device,
                    precision: str) -> dict[str, float]:
    totals: dict[str, Tensor] = {}
    count = 0
    for batch in loader:
        items = sum(len(item.lengths) for item in batch)
        for key, value in batch_losses(batch, criterion, device, precision, False).items():
            totals[key] = totals.get(key, 0) + items * value
        count += items
    return {key: (total / count).item() for key, total in totals.items()}


def train_stage(name: str, train: DataLoader, cfg: dict, optimizer: torch.optim.Optimizer,
                models: tuple[torch.nn.Module, ...], loss: Callable[[Batch], dict],
                device: torch.device, precision: str, ctx: RunContext, *,
                validate: Callable[[], dict[str, float]] | None = None,
                select: str | None = None, save: Callable[[], None] | None = None,
                after_step: Callable[[int], None] | None = None,
                snapshot: Callable[[int], None] | None = None) -> dict:
    """Run ``cfg['updates']`` optimiser updates.

    ``select``: a validation key to minimise (the best state is saved), or None to save the
    final state (MoRAE keeps the last EMA generator and the last JEPA teacher)."""
    total = cfg["updates"]
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    params = [p for group in optimizer.param_groups for p in group["params"]]
    snapshots = set(cfg.get("snapshot_steps") or ())
    best, best_step = math.inf, None
    running: dict[str, Tensor] = {}
    since = 0
    iterator = iter(train)
    for step in range(total):
        for model in models:
            model.train()
        batch = next(iterator, None)
        if batch is None:
            iterator = iter(train)
            batch = next(iterator)
        factor = lr_factor(step, cfg["warmup_updates"], total, cfg["min_lr_ratio"])
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * factor
        optimizer.zero_grad(set_to_none=True)
        losses = batch_losses(batch, loss, device, precision, backward=True)
        grad_norm = clip_grad_norm_(params, cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if after_step is not None:
            after_step(step + 1)
        for key, value in losses.items():
            running[key] = running.get(key, 0) + value
        running["grad_norm"] = running.get("grad_norm", 0) + grad_norm.detach().float()
        since += 1
        if snapshot is not None and step + 1 in snapshots:
            snapshot(step + 1)
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == total:
            record = {f"{name}/train_{k}": (v / since).item() for k, v in running.items()}
            record[f"{name}/lr"] = optimizer.param_groups[0]["lr"]
            if validate is not None:
                for model in models:
                    model.eval()
                scores = validate()
                record.update({f"{name}/val_{k}": v for k, v in scores.items()})
                if select is not None:
                    score = scores[select]
                    if not math.isfinite(score):
                        raise RuntimeError(f"Non-finite {name} validation {select} "
                                           f"at update {step + 1}")
                    if score < best:
                        best, best_step = score, step + 1
                        if save is not None:
                            save()
            if select is None and save is not None:
                save()
            ctx.log_metrics(record, step=step + 1)
            running.clear()
            since = 0
    return {"best": best, "best_step": best_step}
