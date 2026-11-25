"""Separate the signed sphere used by the bridge from physical rotations."""

import math

import torch
from slp.geometry import aa_to_quat as so3_exp
from slp.geometry import align_sign, inner, project, quat_to_aa
from slp.geometry import sign_convention as canon

__all__ = [
    "canon",
    "so3_exp",
    "so3_log_short",
    "sphere_exp",
    "sphere_log_signed",
    "slerp_short",
    "multiply",
    "normalize",
    "physical_angle",
    "rotation6d",
]


def identity_like(q):
    return torch.cat((torch.ones_like(q[..., :1]), torch.zeros_like(q[..., 1:])), -1)


def normalize(q):
    size = q.square().sum(-1, keepdim=True)
    return torch.where(size < 1e-16, identity_like(q), q / size.clamp_min(1e-16).sqrt())


def multiply(q, p):
    w, u = q[..., :1], q[..., 1:]
    s, v = p[..., :1], p[..., 1:]
    u, v = torch.broadcast_tensors(u, v)
    return torch.cat((w * s - inner(u, v), w * v + s * u + torch.cross(u, v, dim=-1)), -1)


def so3_log_short(q, p):
    inverse = torch.cat((q[..., :1], -q[..., 1:]), -1)
    return quat_to_aa(canon(multiply(inverse, p)))


def physical_angle(q, p):
    return 2 * inner(q, p).abs().clamp(0, 1).acos().squeeze(-1)


def sphere_exp(q, tangent):
    square = tangent.square().sum(-1, keepdim=True)
    angle = square.clamp_min(1e-12).sqrt()
    cosine = torch.where(square < 1e-8, 1 - square / 2, angle.cos())
    sinc = torch.where(square < 1e-8, 1 - square / 6, angle.sin() / angle)
    return cosine * q + sinc * tangent


def basis_least_parallel(q):
    return torch.nn.functional.one_hot(q.abs().argmin(-1), q.shape[-1]).to(q)


def sphere_log_signed(q, p):
    dot = inner(q, p).clamp(-1, 1)
    tangent = p - dot * q
    square = tangent.square().sum(-1, keepdim=True)
    # Exact equality also handles roundoff in the dot of identical unit vectors.
    equal = (p == q).all(-1, keepdim=True)
    opposite = (p == -q).all(-1, keepdim=True) | ((square == 0) & (dot < 0))
    size = square.clamp_min(torch.finfo(q.dtype).tiny).sqrt()
    factor = torch.where(
        (square < 1e-8) & (dot >= 0), 1 + square / 6, torch.atan2(size, dot) / size
    )
    axis = project(q, basis_least_parallel(q))
    anti = math.pi * axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(
        equal, torch.zeros_like(tangent), torch.where(opposite, anti, factor * tangent)
    )


def slerp_short(q, p, fraction):
    return normalize(sphere_exp(q, fraction * sphere_log_signed(q, align_sign(p, q))))


def axes(first, second):
    length = first.norm(dim=-1, keepdim=True)
    fixed = torch.zeros_like(first)
    fixed[..., 0] = 1
    u = torch.where(length < 1e-6, fixed, first / length.clamp_min(1e-6))
    v = project(u, second)
    length_v = v.norm(dim=-1, keepdim=True)
    fallback = project(u, basis_least_parallel(u))
    fallback = fallback / fallback.norm(dim=-1, keepdim=True)
    return u, torch.where(length_v < 1e-6, fallback, v / length_v.clamp_min(1e-6))


def rotation6d(values):
    """Convert two columns to a quaternion using the best-conditioned diagonal."""
    u, v = axes(values[..., :3], values[..., 3:])
    w = torch.cross(u, v, dim=-1)
    a, d, g = u.unbind(-1)
    b, e, h = v.unbind(-1)
    c, f, i = w.unbind(-1)
    diagonal = torch.stack((1 + a + e + i, 1 + a - e - i, 1 - a + e - i, 1 - a - e + i), -1)
    candidates = torch.stack(
        [
            torch.stack((diagonal[..., 0], h - f, c - g, d - b), -1),
            torch.stack((h - f, diagonal[..., 1], b + d, c + g), -1),
            torch.stack((c - g, b + d, diagonal[..., 2], f + h), -1),
            torch.stack((d - b, c + g, f + h, diagonal[..., 3]), -1),
        ],
        -2,
    )
    index = diagonal.argmax(-1)[..., None, None].expand(*diagonal.shape[:-1], 1, 4)
    return canon(normalize(candidates.gather(-2, index).squeeze(-2)))
