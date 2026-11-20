"""Variable-length batches of normalized SMPL-X sentence clips."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from slp.data import Split

# Layout returned by slp.data.sign_smplx.feats133.
PART_NAMES = ("body", "left_hand", "right_hand", "face")
PART_DIMS = (30, 45, 45, 13)
POSE_DIM = sum(PART_DIMS)
LANGUAGES = ("de_DGS", "zh_CSL", "en_ASL")
SOURCES = {"phoenix": "de_DGS", "csl": "zh_CSL", "how2sign": "en_ASL"}


def text_input(item: dict) -> tuple[str, int]:
    lang = item.get("lang") or SOURCES.get(item.get("src"))
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown signing language {lang!r} for {item.get('name')!r}")
    text = item["text"]
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    return text, LANGUAGES.index(lang)


def valid_frames(lengths: Tensor, size: int) -> Tensor:
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def ceil_div(lengths: Tensor, stride: int) -> Tensor:
    return (lengths + stride - 1) // stride


class MotionDataset(Dataset):
    def __init__(self, split: Split, limit: int | None = None):
        self.split = split
        count = len(split)
        self.indices = range(count)
        if limit is not None and limit < count:
            self.indices = np.linspace(0, count - 1, limit, dtype=int).tolist()
        if not self.indices:
            raise ValueError("MotionHiFlow SLP needs non-empty train and validation splits")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        index = self.indices[i]
        item = self.split.inputs[index]
        pose = np.asarray(self.split.targets[index], dtype=np.float32)
        if pose.ndim != 2 or pose.shape[1] != POSE_DIM or len(pose) == 0:
            raise ValueError(f"{item.get('name')}: expected a non-empty [T,{POSE_DIM}] clip")
        if not np.isfinite(pose).all():
            raise ValueError(f"{item.get('name')}: found non-finite poses")
        return torch.from_numpy(pose), *text_input(item)


class MotionBatch(NamedTuple):
    poses: Tensor
    lengths: Tensor
    texts: list[str]
    languages: Tensor

    def to(self, device: torch.device) -> MotionBatch:
        return MotionBatch(
            self.poses.to(device, non_blocking=True),
            self.lengths.to(device, non_blocking=True),
            self.texts,
            self.languages.to(device, non_blocking=True),
        )


def collate_microbatches(items: list, microbatch: int) -> list[MotionBatch]:
    """Sort inside each shuffled optimizer batch to reduce padding."""
    from torch.nn.utils.rnn import pad_sequence

    items = sorted(items, key=lambda item: len(item[0]))
    batches = []
    for start in range(0, len(items), microbatch):
        poses, texts, languages = zip(*items[start:start + microbatch])
        lengths = torch.tensor([len(pose) for pose in poses])
        batches.append(MotionBatch(
            pad_sequence(poses, batch_first=True), lengths, list(texts), torch.tensor(languages)
        ))
    return batches
