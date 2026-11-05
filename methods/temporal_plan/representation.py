"""Checkpointed SMPL-X hand-mean convention, separate from statistical normalization."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .geometry import learning256_to_soke133, soke133_to_learning256
from .tensors import check_sequence, zero_padding


class MotionRepresentation(nn.Module):
    def __init__(self, rotation_offsets: Tensor):
        super().__init__()
        if rotation_offsets.shape != (41, 3) or not torch.isfinite(rotation_offsets).all():
            raise ValueError("explicit finite SMPL-X pose mean [41,3] required")
        self.register_buffer("rotation_offsets", rotation_offsets.detach().clone())

    def encode(self, raw133: Tensor, lengths: Tensor) -> Tensor:
        check_sequence(raw133, lengths, 133)
        raw133 = zero_padding(raw133, lengths)
        return zero_padding(soke133_to_learning256(raw133, self.rotation_offsets), lengths)

    def decode(self, learning256: Tensor, lengths: Tensor) -> Tensor:
        check_sequence(learning256, lengths, 256)
        learning256 = zero_padding(learning256, lengths)
        return zero_padding(learning256_to_soke133(learning256, self.rotation_offsets), lengths)
