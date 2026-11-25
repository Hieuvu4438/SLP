"""Optimizer-step training and validation for the codec and frozen-codec flow."""

from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader

from slp.utils.torch_training import (
    batch_losses as batch_losses,
    lr_factor as lr_factor,
    precision_context as precision_context,
    train_stage as train_stage,
    validation as validation,
)

from .codec import MotionVAE
from .data import MotionDataset, collate_microbatches, valid_frames


def make_loader(
    data: MotionDataset, cfg: dict, workers: int, device: torch.device, seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        data, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=workers,
        collate_fn=partial(collate_microbatches, microbatch=cfg["microbatch"]),
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def calibrate(
    codec: MotionVAE, loader: DataLoader, device: torch.device, precision: str,
) -> float:
    """Estimate latent scale and duration prior from train clips only."""
    codec.eval()
    sums = torch.zeros_like(codec.latent_mean, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    n = torch.zeros((), device=device, dtype=torch.int64)
    duration_sum = torch.zeros((), device=device, dtype=torch.float64)
    clips = 0
    for batch in loader:
        for item in batch:
            item = item.to(device)
            with precision_context(device, precision):
                means, _, lengths = codec.encode(item.poses, item.lengths)
            mask = valid_frames(lengths, means.shape[1])
            values = means[mask].double()
            sums += values.sum(0)
            squares += values.square().sum(0)
            n += lengths.sum()
            duration_sum += item.lengths.double().log().sum()
            clips += len(item.lengths)
    center = sums / n
    var = squares / n - center.square()
    codec.latent_mean.copy_(center)
    codec.latent_std.copy_(var.clamp_min(0.05 ** 2).sqrt())
    return (duration_sum / clips).item()
