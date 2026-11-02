"""Dataset containers. Splits are defined ONCE here (fixed, seeded, saved)
and shared by every method - never re-split inside a method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Split:
    inputs: Any
    targets: Any
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.inputs)


@dataclass
class DatasetSplits:
    train: Split
    val: Split
    test: Split
    info: dict = field(default_factory=dict)

    def get(self, name: str) -> Split:
        if name not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {name!r}")
        return getattr(self, name)
