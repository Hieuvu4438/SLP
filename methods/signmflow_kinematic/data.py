"""Fixed-split pose/time cache with fingerprints tied to the actual packed inputs."""

import errno
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from slp.data.sign_smplx import LazyPoses, feats133, fps24_indices, madis_crop

NOMINAL_FPS = {"phoenix": 25., "csl": 30., "how2sign": 24.}
CACHE_VERSION = 1


def cache_fingerprint(split, count, nominal):
    targets = split.targets
    digest = hashlib.sha256()
    header = [CACHE_VERSION, split.meta, count, nominal]
    if isinstance(targets, LazyPoses):
        header.extend((targets.crop, targets.subsample, targets.mean.tolist(), targets.std.tolist()))
    digest.update(json.dumps(header, sort_keys=True).encode())
    for i, info in enumerate(split.inputs[:count]):
        digest.update(json.dumps(info, sort_keys=True).encode())
        if isinstance(targets, LazyPoses):
            file = Path(targets.files[i])
            st = file.stat()
            fps = None if targets.fps is None else targets.fps[i]
            digest.update(str((file.resolve(), st.st_size, st.st_mtime_ns, fps)).encode())
        else:
            digest.update(np.ascontiguousarray(targets[i], dtype=np.float32).tobytes())
    return digest.hexdigest()


class PackedReader(Dataset):
    def __init__(self, split, count, nominal):
        self.split = split
        self.count = count
        self.nominal = nominal

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        import pickle

        targets = self.split.targets
        source = self.split.inputs[i]["src"]
        fps = self.nominal[source]
        if isinstance(targets, LazyPoses):
            with open(targets.files[i], "rb") as handle:
                pose = feats133(pickle.load(handle))
            idx = np.arange(len(pose))
            if targets.fps is not None:
                fps = targets.fps[i]
                if fps > 24:
                    idx = fps24_indices(len(pose), fps, targets.subsample)
            if targets.crop:
                idx = madis_crop(idx)
            pose = (pose[idx] - targets.mean) / targets.std
            time = idx / fps
        else:
            pose = np.asarray(targets[i], np.float32)
            time = np.arange(len(pose)) / fps
        if pose.ndim != 2 or pose.shape[1] != 133 or not len(pose) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid packed pose under {self.split.inputs[i]['name']!r}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid motion frame rate")
        block = np.column_stack((pose, time)).astype(np.float32)
        return block


def build_cache(split, root, nominal, limit=None, workers=0):
    if split.meta.get("split") == "test":
        raise ValueError("Training cache only accepts train/validation splits")
    count = len(split) if limit is None else min(len(split), int(limit))
    if count < 1:
        raise ValueError("Empty training or validation cache")
    key = cache_fingerprint(split, count, nominal)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / key
    if not directory.exists():
        reader = PackedReader(split, count, nominal)
        loader = DataLoader(reader, batch_size=None, num_workers=workers)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            work = Path(temp) / "cache"
            work.mkdir()
            offsets = [0]
            with (work / "frames.bin").open("wb") as output:
                for block in loader:
                    block.numpy().tofile(output)
                    offsets.append(offsets[-1] + len(block))
            metadata = {"offsets": offsets, "inputs": split.inputs[:count], "key": key}
            (work / "index.json").write_text(json.dumps(metadata, ensure_ascii=False))
            try:
                work.rename(directory)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
    return MotionDataset(directory)


class MotionDataset(Dataset):
    def __init__(self, directory):
        meta = json.loads((directory / "index.json").read_text())
        self.offsets = meta["offsets"]
        self.inputs = meta["inputs"]
        self.key = meta["key"]
        file = directory / "frames.bin"
        if file.stat().st_size != self.offsets[-1] * 134 * np.dtype(np.float32).itemsize:
            raise ValueError(f"Truncated cache {directory}")
        self.frames = np.memmap(file, dtype=np.float32, mode="r", shape=(self.offsets[-1], 134))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        a, b = self.offsets[i:i + 2]
        feat = torch.from_numpy(self.frames[a:b].copy())
        info = self.inputs[i]
        return feat[:, :133], feat[:, 133], info["text"], info["src"]


def collate(batch):
    feats, times, texts, sources = zip(*batch)
    lengths = [len(f) for f in feats]
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    times = torch.nn.utils.rnn.pad_sequence(times, batch_first=True)
    dt = times[:, 1:] - times[:, :-1]
    pair_valid = mask[:, 1:] & mask[:, :-1] & (dt > 0)
    inverse_dt = torch.where(pair_valid, dt.clamp_min(1e-6).reciprocal(), 0.)
    return feats, mask, list(texts), list(sources), inverse_dt
