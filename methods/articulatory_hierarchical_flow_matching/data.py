"""Padded batches of normalized SOKE poses."""

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from slp.data import Split
from slp.sign_text import MBART_LANGUAGES

POSE_DIM = 133
LANGUAGES = ("de_DGS", "zh_CSL", "en_ASL")
SOURCES = dict(zip(("phoenix", "csl", "how2sign"), LANGUAGES))
assert len(LANGUAGES) == len(MBART_LANGUAGES)


def text_input(item: dict) -> tuple[str, int]:
    lang = item.get("lang") or SOURCES.get(item.get("src"))
    text = item.get("text")
    if lang not in LANGUAGES or not isinstance(text, str):
        raise ValueError(f"Invalid text or signing language for {item.get('name')!r}: {lang!r}")
    return text, LANGUAGES.index(lang)


class MotionDataset(Dataset):
    def __init__(self, split: Split, limit: int | None):
        if not len(split) or len(split) != len(split.targets):
            raise ValueError("Expected a non-empty split with one pose per input")
        self.split = split
        self.indices = range(len(split))
        if limit is not None:
            self.indices = np.linspace(0, len(split) - 1, min(limit, len(split)), dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        item = self.split.inputs[index]
        pose = np.asarray(self.split.targets[index], dtype=np.float32)
        if pose.ndim != 2 or pose.shape[1] != POSE_DIM or not len(pose) or not np.isfinite(pose).all():
            raise ValueError(f"{item.get('name')}: expected finite, non-empty [T,{POSE_DIM}] poses")
        return torch.from_numpy(pose), *text_input(item)


class MotionBatch(NamedTuple):
    poses: Tensor
    lengths: Tensor
    texts: list[str]
    languages: Tensor

    def to(self, device: torch.device):
        return MotionBatch(self.poses.to(device), self.lengths.to(device), self.texts,
                           self.languages.to(device))


def collate(items: list, microbatch: int) -> list[MotionBatch]:
    items = sorted(items, key=lambda row: len(row[0]))
    batches = []
    for start in range(0, len(items), microbatch):
        poses, texts, languages = zip(*items[start:start + microbatch], strict=True)
        batches.append(MotionBatch(pad_sequence(poses, batch_first=True),
                                   torch.tensor([len(p) for p in poses]), list(texts),
                                   torch.tensor(languages)))
    return batches
