"""FP32 rotation, canonical rig, FK and variable-length temporal grids."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from slp.geometry import aa_to_matrix, align_sign, geodesic, quat_to_aa, quat_to_matrix

BODY = (
    "pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_collar",
    "right_collar", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("left", "right")
HANDS = tuple(name for side in SIDES for name in (
    f"{side}_wrist", *(f"{side}_{finger}{joint}" for finger in FINGERS for joint in (1, 2, 3))
))
JOINTS = BODY + HANDS
TIP_JOINTS = tuple(f"{side}_{finger}3" for side in SIDES for finger in FINGERS)
WRISTS = (12, 28)
CORRECTIONS = (8, 9, 10, 11)
PARENTS = (
    None, "pelvis", "spine1", "spine2", "spine3", "neck", "spine3", "spine3",
    "left_collar", "right_collar", "left_shoulder", "right_shoulder",
) + tuple(parent for side in SIDES for parent in (
    f"{side}_elbow", *(f"{side}_wrist" if j == 1 else f"{side}_{finger}{j - 1}"
                      for finger in FINGERS for j in (1, 2, 3))
))


def identity6(*shape, device=None):
    return torch.tensor([1., 0., 0., 0., 1., 0.], device=device).expand(*shape, 6)


def rot6d_to_matrix(x, *, convention="rows", return_fallback=False):
    """Decode two rows or two concatenated columns, with a deterministic basis fallback."""
    if convention not in ("rows", "columns") or x.shape[-1] != 6:
        raise ValueError("Expected two rows or two concatenated columns")
    a, b = x.float().split(3, dim=-1)
    finite = x.isfinite().all(-1, keepdim=True)
    a, b = a.masked_fill(~finite, 0), b.masked_fill(~finite, 0)
    eps = 1e-6
    valid_a = a.norm(dim=-1, keepdim=True) >= eps
    axis_x = a.new_tensor([1., 0., 0.])
    u = F.normalize(torch.where(valid_a, a, axis_x), dim=-1)
    v = b - (b * u).sum(-1, keepdim=True) * u
    valid_b = v.norm(dim=-1, keepdim=True) >= eps
    # A second relative tolerance covers roundoff on large collinear vectors.
    valid_b = valid_b & (v.norm(dim=-1, keepdim=True) >= eps * b.norm(dim=-1, keepdim=True))
    axis = F.one_hot(u.abs().argmin(-1), 3).to(u.dtype)
    alternate = axis - (axis * u).sum(-1, keepdim=True) * u
    v = F.normalize(torch.where(valid_b, v, alternate), dim=-1)
    # Re-orthogonalize even inputs just above the degeneracy threshold.
    w = F.normalize(torch.cross(u, v, dim=-1), dim=-1)
    v = torch.cross(w, u, dim=-1)
    matrix = torch.stack((u, v, w), dim=-2 if convention == "rows" else -1)
    return (matrix, (~(finite & valid_a & valid_b)).squeeze(-1)) if return_fallback else matrix


def matrix_to_rot6d(matrix):
    return matrix.float()[..., :2, :].flatten(-2)


def matrix_to_quat(matrix):
    """Largest-component extraction, stable at 180 degrees and at identity."""
    m = matrix.float().flatten(-2)
    a, b, c, d, e, f, g, h, i = m.unbind(-1)
    components = torch.stack((1+a+e+i, 1+a-e-i, 1-a+e-i, 1-a-e+i), -1)
    numerators = torch.stack((
        torch.stack((components[..., 0], h-f, c-g, d-b), -1),
        torch.stack((h-f, components[..., 1], b+d, c+g), -1),
        torch.stack((c-g, b+d, components[..., 2], f+h), -1),
        torch.stack((d-b, c+g, f+h, components[..., 3]), -1),
    ), -2)
    row = components.argmax(-1, keepdim=True)
    chosen = numerators.gather(-2, row[..., None].expand(*row.shape, 4)).squeeze(-2)
    return F.normalize(chosen, dim=-1)


def so3_log(matrix):
    return quat_to_aa(matrix_to_quat(matrix))


def so3_exp(vector):
    return aa_to_matrix(vector.float())


def clipnorm(vector, cap):
    return vector * (cap / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)


@dataclass(frozen=True)
class Rig:
    offsets: torch.Tensor
    tip_offsets: torch.Tensor
    palm_offsets: torch.Tensor
    parents: tuple[int, ...]
    order: tuple[int, ...]
    shoulder_width: float
    sha256: str
    local_axes_id: str

    @classmethod
    def load(cls, filename):
        raw = Path(filename).read_bytes()
        spec = json.loads(raw)
        if (spec.get("version") != "1.0" or spec.get("unit") != "m" or
                spec.get("handedness") != "right" or
                spec.get("palm_convention") != "palm_to_wrist" or
                not isinstance(spec.get("local_axes_id"), str) or not spec["local_axes_id"]):
            raise ValueError("Rig convention must explicitly declare v1.0, metres, right-handed "
                             "coordinates, local axes ID and palm_to_wrist")
        rows = spec["joints"]
        table = {row["name"]: row for row in rows}
        if len(rows) != len(JOINTS) or set(table) != set(JOINTS):
            raise ValueError("Rig must have exactly the 44 unique canonical joints")
        if any(table[name]["parent"] != parent for name, parent in zip(JOINTS, PARENTS)):
            raise ValueError("Rig parent tree differs from the canonical skeleton")
        parents = tuple(JOINTS.index(p) if p else -1 for p in PARENTS)
        order = tuple(range(len(JOINTS)))
        offsets = torch.tensor([table[name]["offset"] for name in JOINTS], dtype=torch.float32)
        tips = torch.tensor([spec["tip_offsets"][name] for name in TIP_JOINTS], dtype=torch.float32)
        palms = torch.tensor([spec["palm_offsets"][s] for s in SIDES], dtype=torch.float32)
        if (offsets.shape != (44, 3) or tips.shape != (10, 3) or palms.shape != (2, 3, 3) or
                not all(x.isfinite().all() for x in (offsets, tips, palms)) or offsets[0].any() or
                not torch.allclose(palms.transpose(-2, -1) @ palms, torch.eye(3).expand_as(palms),
                                   atol=1e-5) or not torch.allclose(palms.det(), torch.ones(2),
                                                                   atol=1e-5)):
            raise ValueError("Invalid rig offsets, zero root translation or palm rotation")
        rest = []
        for j, p in enumerate(parents):
            rest.append(offsets[j] if p < 0 else rest[p] + offsets[j])
        h = (rest[8] - rest[9]).norm().item()
        if not h > 0:
            raise ValueError("Rest shoulder width must be positive")
        return cls(offsets, tips, palms, parents, order, h, hashlib.sha256(raw).hexdigest(),
                   spec["local_axes_id"])


@dataclass(frozen=True)
class Pose:
    local: torch.Tensor
    rotation: torch.Tensor
    joints: torch.Tensor
    tips: torch.Tensor
    joint_mask: torch.Tensor
    tip_mask: torch.Tensor
    wrist: torch.Tensor
    palm: torch.Tensor
    local_tips: torch.Tensor
    hand_mask: torch.Tensor


def fk(local, rig, observation=None):
    """Canonical FK in metres. Relative wrist/tip coordinates use shoulder widths."""
    if local.shape[-3:] != (44, 3, 3):
        raise ValueError("FK expects [..., 44, 3, 3] local rotations")
    with torch.autocast(device_type=local.device.type, enabled=False):
        return _fk(local.float(), rig, observation)


def _fk(local, rig, observation):
    offsets, tip_offsets, palm_offsets = (x.to(local) for x in (
        rig.offsets, rig.tip_offsets, rig.palm_offsets
    ))
    seen = observation if observation is not None else torch.ones(
        local.shape[:-2], device=local.device, dtype=torch.bool
    )
    if seen.shape != local.shape[:-2]:
        raise ValueError("FK observation mask shape differs from joint layout")
    rotations, points, valid = [], [], []
    for j in rig.order:
        p = rig.parents[j]
        if p < 0:
            rotations.append(local[..., j, :, :])
            points.append(torch.zeros_like(local[..., j, :, 0]))
            valid.append(seen[..., j])
            continue
        rotations.append(rotations[p] @ local[..., j, :, :])
        points.append(points[p] + (rotations[p] @ offsets[j]).squeeze(-1))
        valid.append(valid[p] & seen[..., j])
    tip_indices = [JOINTS.index(n) for n in TIP_JOINTS]
    rot, joints = torch.stack(rotations, -3), torch.stack(points, -2)
    joint_mask = torch.stack(valid, -1)
    return _pose_features(local, rot, joints, tip_offsets, palm_offsets, joint_mask,
                          tip_indices, rig.shoulder_width)


def _pose_features(local, rot, joints, tip_offsets, palm_offsets, joint_mask, tip_indices, h):
    tips = joints[..., tip_indices, :] + (
        rot[..., tip_indices, :, :] @ tip_offsets[..., None]
    ).squeeze(-1)
    tips = tips.unflatten(-2, (2, 5))
    wrist_points = joints[..., list(WRISTS), :]
    wrist_rot = rot[..., list(WRISTS), :, :]
    pelvis_inv = rot[..., :1, :, :].transpose(-1, -2)
    wrist = (pelvis_inv @ (wrist_points - joints[..., :1, :])[..., None]).squeeze(-1) / h
    palm = pelvis_inv @ wrist_rot @ palm_offsets
    local_tips = (
        wrist_rot.transpose(-1, -2).unsqueeze(-3) @ (tips - wrist_points.unsqueeze(-2))[..., None]
    ).squeeze(-1) / h
    return Pose(local, rot, joints, tips, joint_mask, joint_mask[..., tip_indices].unflatten(-1,
                (2, 5)), wrist, palm, local_tips, joint_mask[..., list(WRISTS)])


def merge_local(body, hands):
    return torch.cat((body, hands.flatten(-4, -3)), dim=-3)


def build_fine_target(body_gt, hands6, body_condition, body_obs, hands_obs, condition_obs,
                      cap=0.35, enabled=True):
    """Build a raw 204D endpoint and observation mask against the actual condition."""
    with torch.autocast(device_type=body_gt.device.type, enabled=False):
        gt = body_gt.float()[..., list(CORRECTIONS), :, :]
        base = body_condition.detach().float()[..., list(CORRECTIONS), :, :]
        delta = so3_log(base.transpose(-1, -2) @ gt)
        correction_mask = body_obs[..., list(CORRECTIONS)] & condition_obs[..., list(CORRECTIONS)]
        correction_mask = correction_mask & enabled
        capped = clipnorm(delta, cap)
        norm = delta.norm(dim=-1)
        clamped = (norm > cap) & correction_mask
        mask = torch.cat((hands_obs.repeat_interleave(6, -1).flatten(-2),
                          correction_mask.repeat_interleave(3, -1)), -1)
        state = torch.cat((hands6.flatten(-3), capped.masked_fill(
            ~correction_mask[..., None], 0).flatten(-2)), -1)
        return state, mask, clamped, (norm - cap).clamp_min(0).masked_fill(~correction_mask, 0)


def decode_pose(body, state, rig, cap=0.35, enabled=True, observation=None):
    """Decode physical endpoints with the fine stage as the sole wrist rotation source."""
    with torch.autocast(device_type=body.device.type, enabled=False):
        body, state = body.float(), state.float()
        corrected = body.clone()
        if enabled:
            delta = clipnorm(state[..., 192:].unflatten(-1, (4, 3)), cap)
            corrected[..., list(CORRECTIONS), :, :] = (
                body[..., list(CORRECTIONS), :, :] @ so3_exp(delta)
            )
        hands = rot6d_to_matrix(state[..., :192].unflatten(-1, (2, 16, 6)))
        return fk(merge_local(corrected, hands), rig, observation)


@dataclass(frozen=True)
class BodyGrid:
    rotations: torch.Tensor
    timestamps: torch.Tensor
    mask: torch.Tensor
    observation: torch.Tensor
    fallback_count: int


def _nearest_fill(values, source_time, observed):
    """Jointwise nearest observed value, identity for a joint missing throughout the clip."""
    distance = (source_time[:, None] - source_time[None, :]).abs()
    closest = distance[..., None].masked_fill(~observed[None], torch.inf).argmin(dim=1)
    joints = torch.arange(values.shape[1], device=values.device)
    return torch.where(observed.any(0)[..., None, None], values[closest, joints],
                       torch.eye(3, device=values.device))


def grid_timestamps(timestamps, frame_mask, stride):
    if stride < 1:
        raise ValueError("Stride must be positive")
    counts = (frame_mask.sum(-1) + stride - 1) // stride
    if (counts == 0).any():
        raise ValueError("Empty motion sequence")
    max_count = int(counts.max())
    coarse = timestamps.new_zeros((len(counts), max_count), dtype=torch.float32)
    mask = torch.arange(max_count, device=counts.device) < counts[:, None]
    for i in range(len(counts)):
        for k, chunk in enumerate(timestamps[i, frame_mask[i]].float().split(stride)):
            coarse[i, k] = chunk.mean()
    return coarse, mask


def build_body_grid(body6, timestamps, frame_mask, observation, stride):
    """Average sign-aligned unit quaternions within real-frame bins."""
    time, mask = grid_timestamps(timestamps, frame_mask, stride)
    clip_rot, clip_obs = [], []
    fallback_count = 0
    for i in range(len(body6)):
        frame = frame_mask[i]
        obs = observation[i, frame]
        valid_rot = rot6d_to_matrix(body6[i, frame].masked_fill(~obs[..., None], 0))
        real_time = timestamps[i, frame]
        rotations, masks = [], []
        for k, (bin_rot, bin_obs, bin_time) in enumerate(zip(
                valid_rot.split(stride), obs.split(stride), real_time.split(stride))):
            seen = bin_obs.any(0)
            masks.append(seen)
            if stride == 1:
                rotations.append(bin_rot[0])
                continue
            q = matrix_to_quat(bin_rot)
            joints = torch.arange(q.shape[1], device=q.device)
            first = bin_obs.long().argmax(0)
            mean = align_sign(q, q[first, joints]) * bin_obs[..., None]
            mean = mean.sum(0)
            fallback = seen & (mean.norm(dim=-1) < 1e-6)
            nearest = (bin_time - time[i, k]).abs()[:, None].expand_as(bin_obs)
            nearest = nearest.masked_fill(~bin_obs, torch.inf).argmin(0)
            rotations.append(torch.where(fallback[..., None, None], bin_rot[nearest, joints],
                                         quat_to_matrix(mean)))
            fallback_count += int(fallback.sum())
        values, seen = torch.stack(rotations), torch.stack(masks)
        filled = _nearest_fill(values, time[i, mask[i]], seen)
        pad = time.shape[1] - len(values)
        clip_rot.append(torch.cat((filled, torch.eye(3, device=values.device).expand(pad, 12, 3, 3))))
        clip_obs.append(F.pad(seen, (0, 0, 0, pad)))
    return BodyGrid(torch.stack(clip_rot), time, mask, torch.stack(clip_obs), fallback_count)


def interpolation_indices(source_time, target_time):
    if not len(source_time):
        raise ValueError("Cannot interpolate empty sequence")
    if len(source_time) == 1:
        idx = torch.zeros_like(target_time, dtype=torch.long)
        return idx, idx, torch.zeros_like(target_time)
    right = torch.searchsorted(source_time.contiguous(), target_time.contiguous())
    right = right.clamp(1, len(source_time) - 1)
    left = right - 1
    w = ((target_time - source_time[left]) / (source_time[right] - source_time[left])).clamp(0, 1)
    return left, right, w


def interpolate_rotations(rotations, source_time, target_time):
    left, right, w = interpolation_indices(source_time, target_time)
    q = matrix_to_quat(rotations)
    t = w.view(len(w), *([1] * (q.ndim - 1)))
    a = q[left]
    out = quat_to_matrix(geodesic(a, align_sign(q[right], a), t))
    t = t.unsqueeze(-1)
    return torch.where(t == 0, rotations[left], torch.where(t == 1, rotations[right], out))


def upsample_body(rotations, timestamps, coarse_mask, target_time, frame_mask, observation=None):
    """Interpolate one clip at a time, excluding all padded coarse timestamps and states."""
    out = torch.eye(3, device=rotations.device).expand(
        *target_time.shape, 12, 3, 3
    ).clone()
    valid = torch.zeros((*target_time.shape, 12), device=out.device, dtype=torch.bool)
    for i, cm in enumerate(coarse_mask):
        fm = frame_mask[i]
        source, target = timestamps[i, cm], target_time[i, fm]
        out[i, fm] = interpolate_rotations(rotations[i, cm], source, target)
        if observation is None:
            valid[i, fm] = True
        else:
            left, right, w = interpolation_indices(source, target)
            m = observation[i, cm]
            valid[i, fm] = ((m[left] | (w == 1)[:, None]) & (m[right] | (w == 0)[:, None]))
    return out, valid
