"""Differentiable rotations and the exact retained SMPL-X channel layout.

6D rotations store the first TWO ROWS of a rotation matrix, in row-major order.
No external geometry implementation is copied. All conversions act on raw,
unnormalized parameters; apply inverse feature normalization before calling.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

PART_SLICES = (slice(0, 60), slice(60, 150), slice(150, 240), slice(240, 256))
PART_NAMES = ("body", "left_hand", "right_hand", "face")
BODY_POSE_INDICES = tuple(range(11, 21))  # SMPL-X body-pose indices, excluding root
BODY_JOINT_NAMES = (
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)


def axis_angle_to_matrix(v: Tensor) -> Tensor:
    if v.shape[-1] != 3:
        raise ValueError("axis-angle must end in three components")
    theta = torch.linalg.vector_norm(v, dim=-1)
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    k = torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    # Stable Taylor limits without divisions by theta at identity.
    a = torch.sinc(theta / math.pi)[..., None, None]
    b = (0.5 * torch.sinc(theta / (2 * math.pi)).square())[..., None, None]
    return eye + a * k + b * (k @ k)


def matrix_to_axis_angle(m: Tensor) -> Tensor:
    """SO(3) logarithm via the best-conditioned quaternion branch, including pi."""
    if m.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    a, b, c = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    squared = torch.stack((1 + a + b + c, 1 + a - b - c,
                           1 - a + b - c, 1 - a - b + c), -1)
    # Non-selected branches at zero must not contribute sqrt(0) gradients.
    qabs = squared.clamp_min(torch.finfo(m.dtype).eps).sqrt()
    candidates = torch.stack((
        torch.stack((qabs[..., 0].square(), m[..., 2, 1] - m[..., 1, 2],
                     m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), -1),
        torch.stack((m[..., 2, 1] - m[..., 1, 2], qabs[..., 1].square(),
                     m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]), -1),
        torch.stack((m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0],
                     qabs[..., 2].square(), m[..., 1, 2] + m[..., 2, 1]), -1),
        torch.stack((m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0],
                     m[..., 1, 2] + m[..., 2, 1], qabs[..., 3].square()), -1),
    ), -2) / (2 * qabs[..., :, None])
    index = qabs.argmax(-1)
    q = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    angle = 2 * torch.atan2(torch.linalg.vector_norm(q[..., 1:], dim=-1), q[..., 0])
    return q[..., 1:] * (2 / torch.sinc(angle / (2 * math.pi)))[..., None]


def matrix_to_rotation6d(m: Tensor) -> Tensor:
    if m.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    return m[..., :2, :].reshape(*m.shape[:-2], 6)


def rotation6d_to_matrix(d: Tensor, eps: float = 1e-6,
                         return_fallback: bool = False):
    if d.shape[-1] != 6:
        raise ValueError("6D rotation must end in six components")
    a, b = d[..., :3], d[..., 3:]
    bad_a = torch.linalg.vector_norm(a, dim=-1, keepdim=True) < eps
    e1 = torch.zeros_like(a)
    e1[..., 0] = 1
    u = F.normalize(torch.where(bad_a, e1, a), dim=-1, eps=eps)
    orth = b - (u * b).sum(-1, keepdim=True) * u
    bad_b = torch.linalg.vector_norm(orth, dim=-1, keepdim=True) < eps
    axis = F.one_hot(u.abs().argmin(-1), 3).to(u)
    fallback = axis - (axis * u).sum(-1, keepdim=True) * u
    v = F.normalize(torch.where(bad_b, fallback, orth), dim=-1, eps=eps)
    m = torch.stack((u, v, torch.linalg.cross(u, v, dim=-1)), -2)
    return (m, (bad_a | bad_b).squeeze(-1)) if return_fallback else m


def soke133_to_learning256(raw: Tensor, rotation_offsets: Tensor | None = None) -> Tensor:
    if raw.shape[-1] != 133:
        raise ValueError("expected raw, unnormalized SOKE 133-D parameters")
    rotations = raw[..., :123].reshape(*raw.shape[:-1], 41, 3)
    if rotation_offsets is not None:
        if rotation_offsets.shape != (41, 3):
            raise ValueError("retained pose-mean offsets must have shape [41,3]")
        rotations = rotations + rotation_offsets.to(raw)
    d6 = matrix_to_rotation6d(axis_angle_to_matrix(rotations))
    return torch.cat((d6.flatten(-2), raw[..., 123:]), -1)


def learning256_to_soke133(features: Tensor, rotation_offsets: Tensor | None = None) -> Tensor:
    if features.shape[-1] != 256:
        raise ValueError("expected 256-D learning representation")
    matrices = rotation6d_to_matrix(features[..., :246].reshape(*features.shape[:-1], 41, 6))
    rotations = matrix_to_axis_angle(matrices)
    if rotation_offsets is not None:
        if rotation_offsets.shape != (41, 3):
            raise ValueError("retained pose-mean offsets must have shape [41,3]")
        rotations = rotations - rotation_offsets.to(features)
    return torch.cat((rotations.flatten(-2), features[..., 246:]), -1)


def rotation_distance(a: Tensor, b: Tensor) -> Tensor:
    """Geodesic distance in radians; finite subgradient at equal rotations."""
    return torch.linalg.vector_norm(matrix_to_axis_angle(a.transpose(-1, -2) @ b), dim=-1)


def unpack_smplx(raw: Tensor) -> dict[str, Tensor]:
    """Zero root/lower-body; retain body-pose indices 11..20, hands, jaw, expression."""
    if raw.ndim != 2 or raw.shape[-1] != 133:
        raise ValueError("SMPL-X inputs must be flattened valid frames [N,133]")
    body = torch.cat((raw.new_zeros(raw.shape[0], 33), raw[:, :30]), -1)
    return dict(global_orient=raw.new_zeros(raw.shape[0], 3), body_pose=body,
                left_hand_pose=raw[:, 30:75], right_hand_pose=raw[:, 75:120],
                jaw_pose=raw[:, 120:123], expression=raw[:, 123:],
                leye_pose=raw.new_zeros(raw.shape[0], 3),
                reye_pose=raw.new_zeros(raw.shape[0], 3),
                transl=raw.new_zeros(raw.shape[0], 3))


def forward_kinematics(local_rotations: Tensor, offsets: Tensor,
                       parents: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    """Analytic skeletal FK for a supplied rig; NOT an SMPL-X mesh substitute."""
    if local_rotations.shape[-3:] != (len(parents), 3, 3):
        raise ValueError("rotation count must match the supplied parent tree")
    if offsets.shape != (len(parents), 3) or parents[0] != -1:
        raise ValueError("invalid root or offsets")
    rotations, positions = [], []
    for j, parent in enumerate(parents):
        if j and not 0 <= parent < j:
            raise ValueError("parents must precede children")
        if parent == -1:
            rotations.append(local_rotations[..., j, :, :])
            positions.append(offsets[j].expand(*local_rotations.shape[:-3], 3))
        else:
            positions.append(positions[parent] +
                             (rotations[parent] @ offsets[j, :, None]).squeeze(-1))
            rotations.append(rotations[parent] @ local_rotations[..., j, :, :])
    return torch.stack(positions, -2), torch.stack(rotations, -3)
