"""Explicit raw-motion data contracts. No cropping, length oracle or implicit normalization."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .text import LANGUAGES


@dataclass(frozen=True)
class MotionExample:
    name: str
    text: str
    lang: str
    source_id: str
    fps: float
    raw133: Tensor
    motion_key: str


def validate_description(row: dict) -> None:
    for key in ("name", "text", "source_id"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"nonempty {key!r} is required")
    if row.get("lang") not in LANGUAGES:
        raise ValueError(f"unknown target language {row.get('lang')!r}")
    if isinstance(row.get("fps"), bool):
        raise ValueError("fps must be a positive number")
    try:
        fps = float(row.get("fps", float("nan")))
    except (TypeError, ValueError) as error:
        raise ValueError("fps must be a positive number") from error
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")


def raw_tensor(array) -> Tensor:
    # Copy read-only/mmap arrays; never mutate caller data or expose unsafe buffers.
    if isinstance(array, Tensor):
        if array.device.type != "cpu":
            raise ValueError("dataset motion storage must be CPU-side")
        array = array.detach().numpy()
    x = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    if x.ndim != 2 or x.shape[1] != 133 or len(x) == 0:
        raise ValueError("each raw motion must have shape [T>0,133]")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("valid raw motion contains non-finite values")
    return x


class MotionDataset(Dataset):
    split: str

    def __init__(self, records: Sequence[dict], motions: Sequence, *, split: str,
                 normalized_stats: tuple[np.ndarray, np.ndarray] | None = None):
        if split not in ("train", "val"):
            raise ValueError("method training datasets may only contain train/val")
        if len(records) != len(motions) or not records:
            raise ValueError("empty dataset or inconsistent inputs/targets")
        self.records = [dict(r) for r in records]
        self.motions, self.split, self.normalized_stats = motions, split, normalized_stats
        if normalized_stats is not None:
            mean, std = (np.asarray(x, dtype=np.float32) for x in normalized_stats)
            if (mean.shape != (133,) or std.shape != (133,) or
                    not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
                raise ValueError("normalized133 requires finite 133-D mean and positive std")
            self.normalized_stats = mean.copy(), std.copy()
        names = set()
        for record in self.records:
            validate_description(record)
            record["fps"] = float(record["fps"])
            if record.get("split", split) != split:
                raise ValueError("manifest contains a different split")
            key = (record["lang"], record["name"])
            if key in names:
                raise ValueError("duplicate sample name within a language")
            names.add(key)

    def __len__(self):
        return len(self.records)

    def _motion(self, index):
        value = self.motions[index]
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.suffix != ".npy":
                raise ValueError("manifest motions must be non-pickled .npy arrays")
            return np.load(path, mmap_mode="r", allow_pickle=False)
        return value

    def frame_count(self, index: int) -> int:
        # Inspect shape only: useful for deriving caps without materializing a corpus.
        x = self._motion(index)
        if x.ndim != 2 or x.shape[1] != 133 or x.shape[0] < 1:
            raise ValueError("each motion must have shape [T>0,133]")
        return int(x.shape[0])

    def __getitem__(self, index):
        row = self.records[index]
        raw = raw_tensor(self._motion(index))
        if self.normalized_stats is not None:
            mean, std = self.normalized_stats
            raw = raw * torch.from_numpy(std) + torch.from_numpy(mean)
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("inverse-normalized raw motion is not finite")
        key = str(self.motions[index]) if isinstance(self.motions[index], (str, Path)) else (
            str(row.get("motion_key", f"{self.split}:{row['lang']}:{row['name']}"))
        )
        return MotionExample(row["name"], row["text"], row["lang"], row["source_id"],
                              float(row["fps"]), raw, key)

    @property
    def max_latents(self):
        return max((self.frame_count(i)+1)//2 for i in range(len(self)))

    @property
    def fps_by_language(self) -> dict[str, float]:
        fps = {}
        for record in self.records:
            lang, value = record["lang"], float(record["fps"])
            if lang in fps and not math.isclose(fps[lang], value, rel_tol=0, abs_tol=1e-6):
                raise ValueError("resample to a declared common FPS per language before training")
            fps[lang] = value
        return fps

    @property
    def fingerprint(self):
        payload = {"split": self.split, "records": self.records,
                   "shapes": [self.frame_count(i) for i in range(len(self))]}
        if self.normalized_stats is not None:
            payload["normalization_sha256"] = [
                hashlib.sha256(x.tobytes()).hexdigest() for x in self.normalized_stats
            ]
        # A provenance digest of IDs/config/lengths, NOT a claimed hash of every motion byte.
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_manifest(path: str | Path, *, split: str) -> MotionDataset:
    path = Path(path).resolve()
    records, motions = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split or row.get("format") != "raw133":
            raise ValueError("manifest must explicitly declare split and format='raw133'")
        motion = Path(row["motion"])
        motion = (path.parent / motion).resolve() if not motion.is_absolute() else motion.resolve()
        records.append({**row, "motion": str(motion)})
        motions.append(motion)
    return MotionDataset(records, motions, split=split)


def validate_train_val(train: MotionDataset, val: MotionDataset | None, *,
                       allow_shared_source_groups: bool = False) -> None:
    if train.split != "train" or (val is not None and val.split != "val"):
        raise ValueError("train and validation roles must be explicit")
    train_fps = train.fps_by_language
    if val is None:
        return
    val_fps = val.fps_by_language
    if any(lang not in train_fps or not math.isclose(fps, train_fps[lang], abs_tol=1e-6)
           for lang, fps in val_fps.items()):
        raise ValueError("validation language/FPS must match training")
    for key in ("name", "source_id"):
        a = {(r["lang"], r[key]) for r in train.records}
        b = {(r["lang"], r[key]) for r in val.records}
        if a & b and not (key == "source_id" and allow_shared_source_groups):
            raise ValueError(f"train/val overlap in {key}")
    paths_a = {str(Path(p).resolve()) for p in train.motions if isinstance(p, (str, Path))}
    paths_b = {str(Path(p).resolve()) for p in val.motions if isinstance(p, (str, Path))}
    if paths_a & paths_b:
        raise ValueError("train/val reuse the same motion file")


@dataclass
class MotionBatch:
    raw: Tensor
    lengths: Tensor
    texts: tuple[str, ...]
    language_ids: Tensor
    names: tuple[str, ...]

    def to(self, device) -> "MotionBatch":
        return MotionBatch(self.raw.to(device), self.lengths.to(device), self.texts,
                           self.language_ids.to(device), self.names)


def collate_motion(examples: list[MotionExample]) -> MotionBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    return MotionBatch(pad_sequence([e.raw133 for e in examples], batch_first=True),
                       torch.tensor([len(e.raw133) for e in examples], dtype=torch.long),
                       tuple(e.text for e in examples),
                       torch.tensor([LANGUAGES[e.lang] for e in examples], dtype=torch.long),
                       tuple(e.name for e in examples))


def tokenize_texts(tokenizer, texts: Sequence[str], *, max_length: int, device):
    if max_length < 2 or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("nonempty texts and max_length>=2 are required")
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length,
                         return_tensors="pt")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
