"""Data plumbing: Split -> cached normalised 133-D frames -> padded batches.

The slp Split targets are lazily-loaded normalised [T,133] arrays (MaDiS crop rule,
How2Sign at 24 fps). They are read once (in parallel) and cached as one float32 array of
concatenated frames + offsets, keyed by dataset/split/count. Rotations are produced on the
GPU from the cached features by ``MotionCodec.encode``, so one cache serves every variant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class _Reader(Dataset):
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return np.asarray(self.targets[i], dtype=np.float32)


def build_cache(split, cache_dir: str | Path, tag: str, workers: int = 8, log=None) -> dict:
    """Read every target sequence of ``split`` once; returns {'frames','offsets','inputs'}."""
    names = [str(x.get("name", i)) for i, x in enumerate(split.inputs)]
    key = hashlib.sha1(("\n".join(names)).encode()).hexdigest()[:12]
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    fr, meta = root / f"{tag}_{key}.npy", root / f"{tag}_{key}.json"
    if fr.exists() and meta.exists():
        m = json.load(open(meta))
        return {"frames": np.load(fr, mmap_mode="r"), "offsets": np.asarray(m["offsets"]),
                "inputs": m["inputs"]}
    loader = DataLoader(_Reader(split.targets), batch_size=1, num_workers=workers,
                        collate_fn=lambda b: b[0])
    seqs = []
    for i, s in enumerate(loader):
        seqs.append(s)
        if log and (i + 1) % 2000 == 0:
            log(f"cache {tag}: {i + 1}/{len(split)}")
    offsets = np.cumsum([0] + [len(s) for s in seqs])
    frames = np.concatenate(seqs, 0).astype(np.float32)
    tmp = fr.with_suffix(".tmp.npy")
    np.save(tmp, frames)
    tmp.rename(fr)
    inputs = [{"name": str(x.get("name", i)), "text": x["text"], "src": x["src"]}
              for i, x in enumerate(split.inputs)]
    json.dump({"offsets": offsets.tolist(), "inputs": inputs}, open(meta, "w"))
    return {"frames": np.load(fr, mmap_mode="r"), "offsets": offsets, "inputs": inputs}


class MotionTextDataset(Dataset):
    """Concatenation of one or more caches. Items: (feats [T,133], text, src)."""

    def __init__(self, caches: list[dict]):
        self.items = []
        for c in caches:
            for i, inp in enumerate(c["inputs"]):
                self.items.append((c, int(c["offsets"][i]), int(c["offsets"][i + 1]), inp))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c, a, b, inp = self.items[i]
        return torch.from_numpy(np.array(c["frames"][a:b])), inp["text"], inp["src"]

    def lengths(self):
        return [b - a for _, a, b, _ in self.items]


def collate(batch):
    feats, texts, srcs = zip(*batch)
    T = max(len(f) for f in feats)
    out = torch.zeros(len(feats), T, feats[0].shape[-1])
    mask = torch.zeros(len(feats), T, dtype=torch.bool)
    for i, f in enumerate(feats):
        out[i, :len(f)] = f
        mask[i, :len(f)] = True
    return out, mask, list(texts), list(srcs)


def all_frames(caches: list[dict]) -> np.ndarray:
    return np.concatenate([np.asarray(c["frames"]) for c in caches], 0)
