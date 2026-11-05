"""Length/mask contracts shared by this method (True always means valid)."""
from __future__ import annotations

import torch
from torch import Tensor


def valid_mask(lengths: Tensor, size: int | None = None) -> Tensor:
    if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be a one-dimensional integer tensor")
    if lengths.numel() == 0 or bool((lengths <= 0).any()):
        raise ValueError("empty sequences/batches are not supported")
    size = int(lengths.max()) if size is None else size
    if size < int(lengths.max()):
        raise ValueError("a sequence length exceeds the padded extent")
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


def zero_padding(x: Tensor, lengths: Tensor) -> Tensor:
    if x.ndim < 2 or x.shape[0] != lengths.numel():
        raise ValueError("expected batch/time leading axes")
    mask = valid_mask(lengths, x.shape[1])
    mask = mask.reshape(*mask.shape, *([1] * (x.ndim - 2)))
    # Multiplication would propagate NaNs from padding.
    return torch.where(mask, x, torch.zeros((), device=x.device, dtype=x.dtype))


def check_sequence(x: Tensor, lengths: Tensor, width: int | None = None) -> None:
    if x.ndim < 3 or x.shape[0] != lengths.numel() or x.device != lengths.device:
        raise ValueError("expected [batch,time,...] and lengths on the same device")
    if width is not None and x.shape[-1] != width:
        raise ValueError(f"expected feature width {width}, got {x.shape[-1]}")
    if not x.is_floating_point():
        raise ValueError("motion/features must be floating point")
    valid = x[valid_mask(lengths, x.shape[1])]
    if not bool(torch.isfinite(valid).all()):
        raise ValueError("non-finite values in valid sequence frames")


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Average across all valid scalar entries, including feature dimensions."""
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    count = mask.sum()
    if int(count) == 0:
        # A graph-connected zero, including single-frame velocity losses.
        return torch.where(mask, value, torch.zeros_like(value)).sum()
    return torch.where(mask, value, torch.zeros_like(value)).sum() / count


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.requires_grad_(False)
    module.eval()
    return module
