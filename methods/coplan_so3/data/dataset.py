"""Variable-length batches, explicit timestamps, and streaming train moments."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from ..geometry.so3 import assert_valid


def length_mask(lengths, width=None):
    if lengths.ndim != 1 or (lengths < 1).any():
        raise ValueError("Lengths must be a positive vector")
    return (
        torch.arange(int(lengths.max()) if width is None else width, device=lengths.device)[None]
        < lengths[:, None]
    )


@dataclass
class MotionBatch:
    rotations: torch.Tensor
    expression: torch.Tensor
    mask: torch.Tensor
    timestamps: torch.Tensor
    texts: list[str]
    languages: torch.Tensor

    @property
    def lengths(self):
        return self.mask.sum(1)

    def to(self, device):
        return MotionBatch(
            self.rotations.to(device),
            self.expression.to(device),
            self.mask.to(device),
            self.timestamps.to(device),
            self.texts,
            self.languages.to(device),
        )


def collate(items):
    """Each item has physical rotations, normalized expression, timestamps and language ID."""
    lengths = torch.tensor([len(x["rotations"]) for x in items])
    mask = length_mask(lengths)
    b, t = mask.shape
    r0 = items[0]["rotations"]
    r = torch.eye(3, dtype=r0.dtype).expand(b, t, r0.shape[-3], 3, 3).clone()
    e = torch.zeros(b, t, items[0]["expression"].shape[-1], dtype=r0.dtype)
    times = torch.zeros(b, t, dtype=r0.dtype)
    for i, x in enumerate(items):
        n = lengths[i]
        assert_valid(x["rotations"])
        if x["timestamps"].shape != (n,) or not torch.isfinite(x["timestamps"]).all():
            raise ValueError("One finite timestamp in seconds per frame required")
        if n > 1 and not (x["timestamps"].diff() > 0).all():
            raise ValueError("Physical timestamps must be strictly increasing")
        if not torch.isfinite(x["expression"]).all():
            raise ValueError("Nonfinite expression")
        r[i, :n], e[i, :n], times[i, :n] = x["rotations"], x["expression"], x["timestamps"]
    return MotionBatch(
        r, e, mask, times, [x["text"] for x in items], torch.tensor([x["language"] for x in items])
    )


class TrainMoments(nn.Module):
    """Parallel-Welford moments; each row is one valid observation, never padding.

    fit/finalize require split='train'. Buffers travel with checkpoints.
    """

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("std", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("ready", torch.tensor(False))

    @torch.no_grad()
    def update(self, x, mask, *, split):
        if split != "train" or self.ready:
            raise ValueError("Statistics may only be fitted once on train")
        x = x[mask].detach().to(self.mean)
        if not len(x):
            return
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite training observations")
        n, mu = len(x), x.mean(0)
        delta, total = mu - self.mean, self.count + len(x)
        self.m2.add_(((x - mu) ** 2).sum(0) + delta.square() * self.count * n / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    @torch.no_grad()
    def finalize(self, *, split):
        if split != "train" or self.count < 1 or self.ready:
            raise ValueError("Nonempty, unfrozen training statistics required")
        self.std.copy_((self.m2 / self.count).clamp_min(1e-8).sqrt())
        self.ready.fill_(True)

    def normalize(self, x):
        if not self.ready:
            raise RuntimeError("Fit train statistics before normalization")
        return (x - self.mean.to(x)) / self.std.to(x)

    def inverse(self, x):
        if not self.ready:
            raise RuntimeError("Fit train statistics before decoding normalized latents")
        return x * self.std.to(x) + self.mean.to(x)


class CanonicalMotionDataset(torch.utils.data.Dataset):
    """Auditable NPZ + JSONL cache, without frame resampling or index guessing.

    JSONL records identify clip/parent video/dataset/split, language, text,
    extraction and annotation versions, the schema digest, and an NPZ checksum.
    NPZ stores physical rotations, normalized expression, timestamps in seconds.
    """

    def __init__(self, index_path, manifest, *, split):
        import json
        from pathlib import Path

        if split not in ("train", "val", "dev"):
            raise ValueError("Method dataset accepts training/validation only")
        self.root = Path(index_path).resolve().parent
        self.manifest = manifest
        self.records = [
            json.loads(line) for line in Path(index_path).read_text().splitlines() if line.strip()
        ]
        required = {
            "clip_id",
            "parent_video_id",
            "dataset_id",
            "split",
            "text",
            "sign_language",
            "extraction_version",
            "annotation_version",
            "manifest_sha256",
            "path",
            "sha256",
            "expression_coordinates",
        }
        ids = set()
        for rec in self.records:
            if not required.issubset(rec) or any(not rec[k] for k in required - {"text"}):
                raise ValueError("Cache provenance fields are missing")
            if rec["split"] != split or rec["manifest_sha256"] != manifest.digest:
                raise ValueError("Cache split/schema mismatch")
            if rec["expression_coordinates"] != "normalized":
                raise ValueError("Cache expression must use train-normalized coordinates")
            key = (rec["dataset_id"], rec["clip_id"])
            if key in ids:
                raise ValueError("Duplicate cache clip ID")
            ids.add(key)
            if rec["sign_language"] not in ("DGS", "CSL", "ASL"):
                raise ValueError("Unsupported sign language")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import numpy as np
        from .manifest import file_sha256

        rec = self.records[i]
        path = self.root / rec["path"]
        if file_sha256(path) != rec["sha256"]:
            raise ValueError("Motion cache checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            r = torch.from_numpy(data["rotations"].copy()).float()
            e = torch.from_numpy(data["expression"].copy()).float()
            times = torch.from_numpy(data["timestamps"].copy()).float()
        n = len(r)
        if (
            r.shape != (n, self.manifest.joints, 3, 3)
            or e.shape != (n, self.manifest.expression_dim)
            or times.shape != (n,)
            or n < 1
        ):
            raise ValueError("Motion cache dimensions do not match the schema")
        return {
            "rotations": r,
            "expression": e,
            "timestamps": times,
            "text": rec["text"],
            "language": {"DGS": 0, "CSL": 1, "ASL": 2}[rec["sign_language"]],
        }
