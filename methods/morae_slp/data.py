"""Signing data for SignRAE: joint groups, masking units, datasets and batches.

The repository's 133-D SMPL-X layout (``slp.data.sign_smplx.feats133``) is

    upper body 0:30 | left hand 30:75 | right hand 75:120 | jaw 120:123 | expression 123:133

Each of the 41 rotated joints owns three axis-angle channels, and the expression coefficients
form one more group, so there are 42 masking groups. MoRAE masks whole joint trajectories
(every channel of a joint in every frame); here a "unit" can also be a kinematic chain or a hand.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from slp.data import Split

POSE_DIM = 133
LANGUAGES = ("de_DGS", "zh_CSL", "en_ASL")
SOURCES = {"phoenix": "de_DGS", "csl": "zh_CSL", "how2sign": "en_ASL"}
PART_SLICES = {"body": slice(0, 30), "lhand": slice(30, 75), "rhand": slice(75, 120),
               "face": slice(120, 133)}

# Upper-body joints are SMPL-X 12..21 in this order.
BODY_JOINTS = ("neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
               "left_elbow", "right_elbow", "left_wrist", "right_wrist")
# Hand joints are SMPL-X 25..39 (left) and 40..54 (right): three joints per finger in this order.
FINGERS = ("index", "middle", "pinky", "ring", "thumb")
N_JOINTS = 41
JAW_GROUP = 40
EXPR_GROUP = 41
N_GROUPS = 42


def _joint_channels(k: int) -> list[int]:
    if k < N_JOINTS:
        return list(range(3 * k, 3 * k + 3))  # body 0..9, lhand 10..24, rhand 25..39, jaw 40
    return list(range(123, 133))


GROUPS: tuple[tuple[int, ...], ...] = tuple(tuple(_joint_channels(k)) for k in range(N_GROUPS))
LHAND_GROUPS = tuple(range(10, 25))
RHAND_GROUPS = tuple(range(25, 40))


def _fingers(first: int) -> list[list[int]]:
    return [list(range(first + 3 * f, first + 3 * f + 3)) for f in range(len(FINGERS))]


def _singletons(groups) -> list[list[int]]:
    return [[g] for g in groups]


_BODY = {name: i for i, name in enumerate(BODY_JOINTS)}
UNITS: dict[str, tuple[tuple[int, ...], ...]] = {
    "joint": tuple((g,) for g in range(N_GROUPS)),
    "chain": tuple(tuple(u) for u in (
        [_BODY["left_collar"], _BODY["left_shoulder"], _BODY["left_elbow"], _BODY["left_wrist"]],
        [_BODY["right_collar"], _BODY["right_shoulder"], _BODY["right_elbow"], _BODY["right_wrist"]],
        [_BODY["neck"], _BODY["head"], JAW_GROUP],
        [EXPR_GROUP],
        *_fingers(LHAND_GROUPS[0]), *_fingers(RHAND_GROUPS[0]),
    )),
    "hand": tuple(tuple(u) for u in (
        list(LHAND_GROUPS), list(RHAND_GROUPS),
        *_singletons(range(10)), [JAW_GROUP], [EXPR_GROUP],
    )),
}


def group_channel_matrix(velocity: bool) -> Tensor:
    """[G, C] membership of input channels in masking groups (C = 133, or 266 with velocities)."""
    channels = POSE_DIM * (2 if velocity else 1)
    member = torch.zeros(N_GROUPS, channels, dtype=torch.bool)
    for g, idx in enumerate(GROUPS):
        member[g, list(idx)] = True
        if velocity:
            member[g, [POSE_DIM + i for i in idx]] = True
    return member


def sample_group_mask(batch: int, ratio: float, unit_probs: dict[str, float],
                      generator: torch.Generator | None = None) -> Tensor:
    """Boolean [batch, 42] mask. Per sample: choose a granularity, then mask random units of it
    until at least round(ratio * 42) groups are hidden. Every granularity partitions the groups,
    so a mask never hides part of a joint."""
    names = [n for n, p in unit_probs.items() if p > 0]
    if not names or any(n not in UNITS for n in names):
        raise ValueError(f"unit_probs must use units from {sorted(UNITS)}")
    probs = torch.tensor([float(unit_probs[n]) for n in names])
    target = max(1, min(N_GROUPS - 1, round(ratio * N_GROUPS)))
    mask = torch.zeros(batch, N_GROUPS, dtype=torch.bool)
    choice = torch.multinomial(probs / probs.sum(), batch, replacement=True, generator=generator)
    for b in range(batch):
        units = UNITS[names[int(choice[b])]]
        for u in torch.randperm(len(units), generator=generator).tolist():
            mask[b, list(units[u])] = True
            if int(mask[b].sum()) >= target:
                break
        if bool(mask[b].all()):  # never hide the whole body; keep one random unit visible
            keep = units[int(torch.randint(len(units), (1,), generator=generator))]
            mask[b, list(keep)] = False
    return mask


def valid_frames(lengths: Tensor, size: int) -> Tensor:
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def with_velocity(poses: Tensor, lengths: Tensor) -> Tensor:
    """[B,T,133] -> [B,T,266]: poses and first differences (0 at frame 0 and on padding)."""
    valid = valid_frames(lengths, poses.shape[1]).unsqueeze(-1)
    poses = poses * valid
    delta = torch.zeros_like(poses)
    delta[:, 1:] = poses[:, 1:] - poses[:, :-1]
    # A valid frame t >= 1 always has a valid predecessor; the first padded frame would see
    # 0 - x[len-1], which the valid mask removes.
    return torch.cat([poses, delta * valid], dim=-1)


def text_input(item: dict) -> tuple[str, int]:
    lang = item.get("lang") or SOURCES.get(item.get("src"))
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown signing language {lang!r} for {item.get('name')!r}")
    text = item.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{item.get('name')!r}: text must be a string")
    return text, LANGUAGES.index(lang)


class ConcatSplit:
    """Read-only union of several splits (multi-dataset self-supervised pretraining)."""

    def __init__(self, splits: list[Split]):
        if not splits:
            raise ValueError("ConcatSplit needs at least one split")
        self.splits = splits
        self.offsets = np.cumsum([0] + [len(s) for s in splits])
        self.inputs = _ConcatView(self, "inputs")
        self.targets = _ConcatView(self, "targets")

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, i: int) -> tuple[Split, int]:
        if not 0 <= i < len(self):
            raise IndexError(i)
        s = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.splits[s], i - int(self.offsets[s])


class _ConcatView:
    def __init__(self, owner: ConcatSplit, attr: str):
        self.owner, self.attr = owner, attr

    def __len__(self) -> int:
        return len(self.owner)

    def __getitem__(self, i: int):
        split, j = self.owner.locate(i)
        return getattr(split, self.attr)[j]


def subset_indices(count: int, limit: int | None) -> list[int]:
    if limit is not None and limit < count:
        return np.linspace(0, count - 1, limit, dtype=int).tolist()
    return list(range(count))


def load_pose(split, index: int) -> np.ndarray:
    pose = np.asarray(split.targets[index], dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != POSE_DIM or len(pose) == 0 or not np.isfinite(pose).all():
        item = split.inputs[index]
        name = item.get("name", index) if isinstance(item, dict) else index
    if pose.ndim != 2 or pose.shape[1] != POSE_DIM or len(pose) == 0:
        raise ValueError(f"{name}: expected a non-empty [T,{POSE_DIM}] clip, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name}: non-finite poses")
    return pose


class PoseDataset(Dataset):
    """Clips for the self-supervised stages (no text). ``crop=(lo, hi)`` takes a random window
    of U[lo, hi] frames (clips shorter than the drawn length are used whole)."""

    def __init__(self, split, limit: int | None = None, crop: tuple[int, int] | None = None):
        self.split = split
        self.indices = subset_indices(len(split), limit)
        if not self.indices:
            raise ValueError("Empty split")
        if crop is not None and not 1 <= crop[0] <= crop[1]:
            raise ValueError(f"Invalid crop range {crop}")
        self.crop = crop

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        pose = load_pose(self.split, self.indices[i])
        if self.crop is not None:
            size = int(torch.randint(self.crop[0], self.crop[1] + 1, (1,)))
            if len(pose) > size:
                start = int(torch.randint(0, len(pose) - size + 1, (1,)))
                pose = pose[start:start + size]
        return (torch.from_numpy(np.ascontiguousarray(pose)), "", 0)


class LatentDataset(Dataset):
    """Cached flow targets (standardised latents) with their text."""

    def __init__(self, latents: list[Tensor], items: list[dict]):
        if len(latents) != len(items) or not latents:
            raise ValueError("LatentDataset needs one latent per text item")
        self.latents = latents
        self.texts = [text_input(item) for item in items]

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, i: int):
        return (self.latents[i].float(), *self.texts[i])


class Batch(NamedTuple):
    values: Tensor      # [B, T, C] poses or latents, zero-padded
    lengths: Tensor     # [B]
    texts: list[str]
    languages: Tensor   # [B]

    def to(self, device: torch.device) -> Batch:
        return Batch(self.values.to(device, non_blocking=True),
                     self.lengths.to(device, non_blocking=True), self.texts,
                     self.languages.to(device, non_blocking=True))


def collate_microbatches(items: list, microbatch: int) -> list[Batch]:
    """Sort each shuffled optimizer batch by length and cut it into microbatches (less padding)."""
    from torch.nn.utils.rnn import pad_sequence

    items = sorted(items, key=lambda item: len(item[0]))
    out = []
    for start in range(0, len(items), microbatch):
        values, texts, languages = zip(*items[start:start + microbatch])
        out.append(Batch(pad_sequence(values, batch_first=True),
                         torch.tensor([len(v) for v in values]), list(texts),
                         torch.tensor(languages)))
    return out
