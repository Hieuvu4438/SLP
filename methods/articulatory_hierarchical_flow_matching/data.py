"""Padded batches of normalized SOKE poses."""

import hashlib
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


def broadcast_id(item: dict, index: int = 0) -> str:
    return str(item.get("name", index)).rsplit("-", 1)[0]


def train_monitor_indices(dataset, fraction: float) -> tuple[list[int], list[int]]:
    """Hold out whole train broadcasts in a fixed order, independent of the run seed."""
    groups: dict[str, list[int]] = {}
    for i, index in enumerate(dataset.indices):
        groups.setdefault(broadcast_id(dataset.split.inputs[index], index), []).append(i)
    if len(groups) < 2:
        raise ValueError("An auxiliary model needs at least two training broadcasts")
    names = sorted(groups, key=lambda name: hashlib.sha256(f"train-monitor/{name}".encode()).digest())
    target = max(1, round(len(dataset) * fraction))
    held = []
    for name in names[:-1]:
        held.extend(groups[name])
        if len(held) >= target:
            break
    kept = sorted(set(range(len(dataset))) - set(held))
    return kept, sorted(held)


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
        self.include_indices = False

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        item = self.split.inputs[index]
        pose = np.asarray(self.split.targets[index], dtype=np.float32)
        if pose.ndim != 2 or pose.shape[1] != POSE_DIM or not len(pose) or not np.isfinite(pose).all():
            raise ValueError(f"{item.get('name')}: expected finite, non-empty [T,{POSE_DIM}] poses")
        result = (torch.from_numpy(pose), *text_input(item))
        return (*result, int(i)) if self.include_indices else result


class MotionBatch(NamedTuple):
    poses: Tensor
    lengths: Tensor
    texts: list[str]
    languages: Tensor
    indices: Tensor | None = None

    def to(self, device: torch.device):
        return MotionBatch(self.poses.to(device), self.lengths.to(device), self.texts,
                           self.languages.to(device), self.indices)


def collate(items: list, microbatch: int) -> list[MotionBatch]:
    items = sorted(items, key=lambda row: len(row[0]))
    batches = []
    for start in range(0, len(items), microbatch):
        poses, texts, languages, *indices = zip(*items[start:start + microbatch], strict=True)
        batches.append(MotionBatch(pad_sequence(poses, batch_first=True),
                                   torch.tensor([len(p) for p in poses]), list(texts),
                                   torch.tensor(languages),
                                   torch.tensor(indices[0]) if indices else None))
    return batches
