"""Gradient accumulation and deterministic validation for variable-length torch batches."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from slp.tracking.run import RunContext


def precision_context(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU cannot use BF16. Set method.precision=fp32.")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def batch_losses(
    batch: list, criterion: Callable, device: torch.device, precision: str,
    backward: bool,
) -> dict[str, Tensor]:
    count = sum(len(item.lengths) for item in batch)
    averaged: dict[str, Tensor] = {}
    for item in batch:
        weight = len(item.lengths) / count
        with precision_context(device, precision):
            losses = criterion(item.to(device))
        if backward:
            (weight * losses["loss"]).backward()
        for key, val in losses.items():
            averaged[key] = averaged.get(key, 0) + weight * val.detach()
    return averaged


@torch.no_grad()
def validation(
    loader: DataLoader, criterion: Callable, device: torch.device, precision: str,
) -> dict[str, float]:
    totals: dict[str, Tensor] = {}
    count = 0
    for batch in loader:
        items = sum(len(item.lengths) for item in batch)
        losses = batch_losses(batch, criterion, device, precision, backward=False)
        for key, val in losses.items():
            totals[key] = totals.get(key, 0) + items * val
        count += items
    return {key: (total / count).item() for key, total in totals.items()}


def train_stage(
    name: str, train: DataLoader, val: DataLoader, cfg: dict, optimizer: torch.optim.Optimizer,
    models: tuple[torch.nn.Module, ...], train_loss: Callable, val_loss: Callable,
    save_best: Callable, device: torch.device, precision: str, ctx: RunContext, *,
    select: str = "loss", after_step: Callable | None = None,
    save_snapshot: Callable[[int], None] | None = None,
    best_from: int = 1,
) -> None:
    total_steps = cfg["updates"]
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    params = [p for group in optimizer.param_groups for p in group["params"]]
    best = math.inf
    loader = iter(train)
    running: dict[str, Tensor] = {}
    since_eval = 0
    snapshot_steps = set(cfg.get("snapshot_steps", ()))
    for step in range(total_steps):
        for model in models:
            model.train()
        batch = next(loader, None)
        if batch is None:
            loader = iter(train)
            batch = next(loader)
        factor = lr_factor(step, cfg["warmup_updates"], total_steps, cfg["min_lr_ratio"])
        for group, lr in zip(optimizer.param_groups, initial_lrs):
            group["lr"] = lr * factor
        optimizer.zero_grad(set_to_none=True)
        losses = batch_losses(batch, train_loss, device, precision, backward=True)
        clip_grad_norm_(params, cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if after_step is not None:
            after_step()
        for key, loss in losses.items():
            running[key] = running.get(key, 0) + loss
        since_eval += 1

        if step + 1 in snapshot_steps and save_snapshot is not None:
            save_snapshot(step + 1)

        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == total_steps:
            for model in models:
                model.eval()
            scores = validation(val, val_loss(), device, precision)
            score = scores[select]
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite {name} validation {select} at step {step + 1}")
            if step + 1 >= best_from and score < best:
                best = score
                save_best()
            ctx.log_metrics({
                **{f"{name}/train_{k}": (v / since_eval).item() for k, v in running.items()},
                **{f"{name}/val_{k}": v for k, v in scores.items()},
                f"{name}/lr": optimizer.param_groups[0]["lr"],
            }, step=step + 1)
            running.clear()
            since_eval = 0
