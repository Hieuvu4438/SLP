"""Versioned manifest, SOKE-133 codec and phase/timebase resampling."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from slp.evaluation.smplx_fk import soke_stats
from slp.geometry import quat_to_aa

from .rotations import canon, identity_like, slerp_short, so3_exp

LANGUAGES = {
    "csl_daily": ("zh_CN", "CSL"),
    "phoenix_2014t": ("de_DE", "DGS"),
    "how2sign": ("en_XX", "ASL"),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def unpack133(pose, layout="soke133_physical", stats=None):
    if pose.shape[-1] != 133:
        raise ValueError("Expected 133 SOKE features")
    if layout == "soke133_norm":
        mean, std = soke_stats(str(stats) if stats else None)
        if mean.shape != (133,) or std.shape != (133,) or not (std > 0).all():
            raise ValueError("Invalid SOKE statistics")
        pose = pose * pose.new_tensor(std) + pose.new_tensor(mean)
    elif layout != "soke133_physical":
        raise ValueError(f"Unknown SOKE layout {layout!r}")
    return canon(so3_exp(pose[..., :123].unflatten(-1, (41, 3)))), pose[..., 123:]


def pack133(q, expression):
    if q.shape[-2:] != (41, 4) or expression.shape != q.shape[:-2] + (10,):
        raise ValueError("Expected [...,41,4] and [...,10]")
    return torch.cat((quat_to_aa(canon(q)).flatten(-2), expression), -1)


def named_nsa(arrays, mapping):
    """Map declared NSA parameter names and channel slices into stored SOKE coordinates."""
    fields = {
        "body_pose": 30,
        "left_hand_pose": 45,
        "right_hand_pose": 45,
        "jaw_pose": 3,
        "expression": 10,
    }
    parts = []
    for field, width in fields.items():
        spec = mapping[field]
        values = np.asarray(arrays[spec["key"]])
        values = values.reshape(values.shape[0], -1)
        indices = spec.get("indices", list(range(width)))
        if len(indices) != width:
            raise ValueError(f"Invalid width for {field}")
        parts.append(values[:, indices])
    return np.concatenate(parts, axis=1)


@dataclass(frozen=True)
class Request:
    text: str
    source_language: str
    target_sign_language: str
    sample_id: str
    seed: int = 17


@dataclass
class Clip:
    row: dict
    q: torch.Tensor
    e: torch.Tensor
    q_valid: torch.Tensor
    expr_valid: torch.Tensor
    fps: float
    confidence: torch.Tensor | None = None
    unrepresented: dict | None = None

    @property
    def id(self):
        return self.row["clip_id"]

    def request(self, seed):
        return Request(
            self.row["text"],
            self.row["source_language"],
            self.row["target_sign_language"],
            self.id,
            seed,
        )


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, groups, pose_splits = set(), {}, {}
    for r in rows:
        clip = r["clip_id"]
        pair = (r["source_language"], r["target_sign_language"])
        if (
            not isinstance(clip, str)
            or not clip
            or clip in ids
            or r["split"] not in {"train", "dev", "test"}
            or LANGUAGES.get(r["corpus"]) != pair
            or not isinstance(r["text"], str)
        ):
            raise ValueError(f"Invalid or duplicate manifest row {clip}")
        ids.add(clip)
        group = (r["corpus"], r.get("video_id") or clip)
        if groups.setdefault(group, r["split"]) != r["split"]:
            raise ValueError(f"Video shared between splits: {group}")
        if r.get("nsa_reference_path"):
            r["nsa_reference_path"] = str((path.parent / r["nsa_reference_path"]).resolve())
        if r.get("pose_path"):
            asset = str((path.parent / r["pose_path"]).resolve())
            if pose_splits.setdefault(asset, r["split"]) != r["split"]:
                raise ValueError(f"Pose file shared between splits: {asset}")
            r["pose_path"] = asset
        elif r["split"] != "test" or not r.get("missing_reason"):
            raise ValueError(f"Missing pose and reference status: {clip}")
        if (
            not math.isfinite(r["fps"])
            or r["fps"] <= 0
            or not r.get("timebase_source")
            or not r.get("pose_fit_id")
            or int(r["num_frames"]) != r["num_frames"]
            or r["num_frames"] < 1
        ):
            raise ValueError(f"Invalid fitting or timebase: {clip}")
    return sorted(rows, key=lambda r: r["clip_id"])


def phase_grid(length, stride=1, *, device=None, dtype=None):
    if length < 1 or stride < 1:
        raise ValueError("Length and stride must be positive")
    count = 1 + (length - 1 + stride - 1) // stride
    return torch.linspace(0, 1, count, device=device, dtype=dtype)


def bracket(grid, positions):
    hi = torch.searchsorted(grid.contiguous(), positions.contiguous()).clamp(max=len(grid) - 1)
    lo = (hi - 1).clamp_min(0)
    lo = torch.where(grid[hi] == positions, hi, lo)
    delta = grid[hi] - grid[lo]
    ratio = ((positions - grid[lo]) / delta.clamp_min(1e-12)).clamp(0, 1)
    return lo, hi, ratio


def resample(values, positions, *, quaternion=False, valid=None, grid=None):
    if grid is None:
        grid = phase_grid(len(values), device=values.device, dtype=values.dtype)
    lo, hi, ratio = bracket(grid, positions.to(grid))
    ratio = ratio.reshape(-1, *([1] * (values.ndim - 1)))
    interpolated = (
        slerp_short(values[lo], values[hi], ratio)
        if quaternion
        else torch.lerp(values[lo], values[hi], ratio)
    )
    return interpolated if valid is None else (interpolated, valid[lo] & valid[hi])


def fill_missing(values, valid, quaternion=False):
    out = identity_like(values) if quaternion else torch.zeros_like(values)
    for j in range(valid.shape[-1]):
        ix = valid[:, j].nonzero().flatten()
        if len(ix):
            grid = ix.to(values)
            out[:, j] = resample(
                values[ix, j],
                torch.arange(len(values), device=grid.device),
                grid=grid,
                quaternion=quaternion,
            )
    return out


def load_clip(row, data_cfg):
    path = row["pose_path"]
    if row.get("sha256") and sha256(path) != row["sha256"]:
        raise ValueError(f"Pose checksum mismatch: {row['clip_id']}")
    with np.load(path, allow_pickle=False) as arrays:
        layout = row["pose_layout"]
        unrepresented = None
        if layout == "nsa_named":
            mapping = row.get("nsa_mapping") or data_cfg["nsa_mapping"]
            pose = named_nsa(arrays, mapping)
            channels = {
                "body_pose": 30,
                "left_hand_pose": 45,
                "right_hand_pose": 45,
                "jaw_pose": 3,
                "expression": 10,
            }
            used = {}
            for key, width in channels.items():
                field = mapping[key]
                used.setdefault(field["key"], set()).update(field.get("indices", range(width)))
            metadata = {"timestamps", "q_valid", "expr_valid", "r_confidence"}
            unrepresented = {}
            for key in set(arrays.files) - metadata:
                values = arrays[key]
                if key in used:
                    n = values.reshape(row["num_frames"], -1).shape[1]
                    unrepresented[key] = sorted(set(range(n)) - used[key])
                else:
                    unrepresented[key] = {"omitted_shape": list(values.shape)}
            layout = "soke133_physical"
        else:
            pose = arrays["pose133"]
        if pose.shape != (row["num_frames"], 133):
            raise ValueError(f"Manifest shape mismatch: {row['clip_id']}")
        q, e = unpack133(
            torch.as_tensor(pose, dtype=torch.float32), layout, data_cfg.get("soke_stats")
        )
        q_valid = q.isfinite().all(-1)
        expr_valid = e.isfinite()
        for name, mask in (("q_valid", q_valid), ("expr_valid", expr_valid)):
            if name in arrays:
                if arrays[name].shape != tuple(mask.shape):
                    raise ValueError(f"Invalid {name} in {path}")
                mask &= torch.as_tensor(arrays[name], dtype=torch.bool)
        if not q_valid.any() and not expr_valid.any():
            raise ValueError(f"Empty reference: {row['clip_id']}")
        if row["split"] != "train" and not (q_valid.all() and expr_valid.all()):
            raise ValueError(f"Incomplete evaluation reference: {row['clip_id']}")
        confidence = None
        if "r_confidence" in arrays:
            confidence = torch.as_tensor(arrays["r_confidence"], dtype=q.dtype)
            if confidence.shape != (len(q), 39) or not confidence.isfinite().all():
                raise ValueError(f"Invalid relation confidence in {path}")
            confidence = confidence.clamp(0, 1)
        time = (
            torch.as_tensor(arrays["timestamps"], dtype=torch.float64)
            if "timestamps" in arrays
            else torch.arange(len(q), dtype=torch.float64) / row["fps"]
        )
    if time.shape != (len(q),) or not time.isfinite().all() or not (time.diff() > 0).all():
        raise ValueError(f"Bad timestamps: {path}")
    time = time - time[0]
    if not q_valid.all():
        q = fill_missing(q, q_valid, True)
    if not expr_valid.all():
        e = fill_missing(e, expr_valid)
    fps = data_cfg.get("fps", {}).get(row["corpus"], row["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid configured FPS for {row['corpus']}")
    ticks = torch.arange(1 + round(float(time[-1]) * fps), dtype=time.dtype) / fps
    ticks = ticks.clamp(max=time[-1])
    # Avoid treating float representation noise at a native frame as fractional sampling.
    if len(ticks) != len(time) or not torch.allclose(ticks, time, rtol=0, atol=1e-7):
        if confidence is not None:
            confidence = resample(confidence, ticks, grid=time)
        q, q_valid = resample(q, ticks, quaternion=True, valid=q_valid, grid=time)
        e, expr_valid = resample(e, ticks, valid=expr_valid, grid=time)
    return Clip(row, canon(q), e, q_valid, expr_valid, fps, confidence, unrepresented)


def load_split(rows, split, data_cfg):
    return [
        load_clip(row, data_cfg) for row in rows if row["split"] == split and row.get("pose_path")
    ]


def collate(clips, device):
    batch = {
        name: pad_sequence([getattr(c, name) for c in clips], batch_first=True).to(device)
        for name in ("q", "e", "q_valid", "expr_valid")
    }
    lengths = torch.tensor([len(c.q) for c in clips], device=device)
    frame_valid = torch.arange(int(lengths.max()), device=device)[None] < lengths[:, None]
    batch["q"] = torch.where(frame_valid[..., None, None], batch["q"], identity_like(batch["q"]))
    batch.update(lengths=lengths, frame_valid=frame_valid)
    return batch
