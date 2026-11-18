"""Data plumbing: Split -> cached normalised 133-D frames (+ optional anchors) -> padded batches.

A cache is one float32 array of concatenated frames + offsets, keyed by split content. Anchor
caches (Stage A predictions, flat [N_frames, 174] = quaternions | expression) share the frame
layout of their motion cache, so a dataset item is a pair of slices.
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
    """Read every target sequence of ``split`` once; returns {'frames','offsets','inputs','key'}."""
    names = [str(x.get("name", i)) for i, x in enumerate(split.inputs)]
    key = hashlib.sha1(("\n".join(names)).encode()).hexdigest()[:12]
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    fr, meta = root / f"{tag}_{key}.npy", root / f"{tag}_{key}.json"
    if not (fr.exists() and meta.exists()):
        loader = DataLoader(_Reader(split.targets), batch_size=1, num_workers=workers,
                            collate_fn=lambda b: b[0])
        seqs = []
        for i, s in enumerate(loader):
            seqs.append(s)
            if log and (i + 1) % 2000 == 0:
                log(f"cache {tag}: {i + 1}/{len(split)}")
        offsets = np.cumsum([0] + [len(s) for s in seqs])
        tmp = fr.with_suffix(".tmp.npy")
        np.save(tmp, np.concatenate(seqs, 0).astype(np.float32))
        tmp.rename(fr)
        inputs = [{"name": str(x.get("name", i)), "text": x["text"], "src": x["src"]}
                  for i, x in enumerate(split.inputs)]
        json.dump({"offsets": offsets.tolist(), "inputs": inputs}, open(meta, "w"))
    m = json.load(open(meta))
    return {"frames": np.load(fr, mmap_mode="r"), "offsets": np.asarray(m["offsets"]),
            "inputs": m["inputs"], "key": f"{tag}_{key}"}


class MotionTextDataset(Dataset):
    """Items: (feats [T,133], text, src, anchor [T,174] or None)."""

    def __init__(self, cache: dict, anchors: np.ndarray | None = None, indices=None):
        self.cache, self.anchors = cache, anchors
        n = len(cache["inputs"])
        self.indices = list(range(n)) if indices is None else list(indices)
        if anchors is not None and len(anchors) != int(cache["offsets"][-1]):
            raise ValueError("anchor cache does not match the frame layout")

    def __len__(self):
        return len(self.indices)

    def span(self, i):
        k = self.indices[i]
        return int(self.cache["offsets"][k]), int(self.cache["offsets"][k + 1])

    def __getitem__(self, i):
        a, b = self.span(i)
        inp = self.cache["inputs"][self.indices[i]]
        anc = None if self.anchors is None else torch.from_numpy(np.array(self.anchors[a:b], np.float32))
        return torch.from_numpy(np.array(self.cache["frames"][a:b])), inp["text"], inp["src"], anc

    def subset(self, idx) -> "MotionTextDataset":
        return MotionTextDataset(self.cache, self.anchors, [self.indices[i] for i in idx])

    def with_anchors(self, anchors) -> "MotionTextDataset":
        return MotionTextDataset(self.cache, anchors, self.indices)


def collate(batch):
    feats, texts, srcs, ancs = zip(*batch)
    T = max(len(f) for f in feats)
    out = torch.zeros(len(feats), T, feats[0].shape[-1])
    mask = torch.zeros(len(feats), T, dtype=torch.bool)
    anc = None if ancs[0] is None else torch.zeros(len(feats), T, ancs[0].shape[-1])
    for i, f in enumerate(feats):
        out[i, :len(f)] = f
        mask[i, :len(f)] = True
        if anc is not None:
            anc[i, :len(f)] = ancs[i]
    return out, mask, list(texts), list(srcs), anc


def all_frames(cache: dict) -> np.ndarray:
    return np.asarray(cache["frames"])
