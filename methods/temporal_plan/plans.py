"""Validated plan containers: content vectors, positive durations, explicit padding."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .tensors import check_sequence, valid_mask, zero_padding


@dataclass
class PlanBatch:
    contents: Tensor       # [B,K,C], normalized content space
    durations: Tensor      # [B,K], latent steps, zero ONLY in padding
    lengths: Tensor        # [B], number of units, not motion duration

    def validate(self, max_duration: int | None = None) -> None:
        check_sequence(self.contents, self.lengths)
        if self.contents.ndim != 3 or self.durations.shape != self.contents.shape[:2]:
            raise ValueError("expected plan contents [B,K,C] and durations [B,K]")
        if self.durations.dtype != torch.long or self.durations.device != self.contents.device:
            raise ValueError("durations must be int64 on the content device")
        valid = valid_mask(self.lengths, self.contents.shape[1])
        if bool((self.durations[valid] < 1).any()):
            raise ValueError("every valid unit needs a positive duration")
        if bool((self.durations[~valid] != 0).any()):
            raise ValueError("padded durations must be zero")
        if max_duration is not None and bool((self.durations[valid] > max_duration).any()):
            raise ValueError("plan duration exceeds model support")

    @property
    def latent_lengths(self) -> Tensor:
        self.validate()
        return self.durations.sum(-1)

    def clean_contents(self) -> Tensor:
        return zero_padding(self.contents, self.lengths)

    def to(self, device) -> "PlanBatch":
        return PlanBatch(self.contents.to(device), self.durations.to(device), self.lengths.to(device))

    def select(self, indices: Tensor) -> "PlanBatch":
        return PlanBatch(self.contents[indices], self.durations[indices], self.lengths[indices])

    @classmethod
    def from_sequences(cls, contents: list[Tensor], durations: list[list[int]]) -> "PlanBatch":
        if not contents or len(contents) != len(durations):
            raise ValueError("empty or inconsistent plan list")
        lengths = torch.tensor([len(c) for c in contents], device=contents[0].device)
        if any(len(c) != len(d) for c, d in zip(contents, durations)):
            raise ValueError("content/duration counts disagree")
        ds = [torch.tensor(d, device=contents[0].device, dtype=torch.long) for d in durations]
        result = cls(pad_sequence(contents, batch_first=True),
                     pad_sequence(ds, batch_first=True), lengths)
        result.validate()
        return result
