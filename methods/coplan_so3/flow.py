"""The same source and posterior laws are used in training and inference."""

from dataclasses import dataclass
import math

import torch

from .geometry import so3


@dataclass(frozen=True)
class SourceConfig:
    sigma_body: float = 0.2
    sigma_hand: float = 0.35
    sigma_expression: float = 0.5
    ar1: float = 0.7

    def __post_init__(self):
        values = (self.ar1, self.sigma_body, self.sigma_hand, self.sigma_expression)
        if (
            not all(math.isfinite(x) for x in values)
            or abs(self.ar1) >= 1
            or min(self.sigma_body, self.sigma_hand, self.sigma_expression) <= 0
        ):
            raise ValueError("Full rank source requires |a|<1 and positive noise scales")


def ar1_noise(shape, a, like, generator=None):
    if abs(a) >= 1:
        raise ValueError("AR(1) requires |a|<1")
    eps = torch.randn(shape, dtype=like.dtype, device=like.device, generator=generator)
    out = [eps[:, 0]]
    for t in range(1, shape[1]):
        out.append(a * out[-1] + (1 - a * a) ** 0.5 * eps[:, t])
    return torch.stack(out, 1)


def source(rbar, ebar, mask, manifest, spec, generator=None):
    sigma = rbar.new_full((manifest.joints, 1), spec.sigma_body)
    sigma[list(manifest.left_hand + manifest.right_hand)] = spec.sigma_hand
    xi = ar1_noise((*rbar.shape[:-2], 3), spec.ar1, rbar, generator)
    r = rbar @ so3.exp(xi * sigma)
    e = ebar + spec.sigma_expression * ar1_noise(ebar.shape, spec.ar1, ebar, generator)
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    return torch.where(mask[..., None, None, None], r, eye), e * mask[..., None]


def interpolate(r0, e0, r1, e1, s):
    if s.shape != (len(r0),):
        raise ValueError("One flow time per batch element required")
    omega = so3.log(r0.transpose(-1, -2) @ r1)
    r = r0 @ so3.exp(s[:, None, None, None] * omega)
    e = e0 + s[:, None, None] * (e1 - e0)
    return r, e, omega, e1 - e0


def plan_fm_loss(prior, z1, zmask, text, text_mask, language, lengths, generator=None):
    z0 = torch.randn(z1.shape, device=z1.device, dtype=z1.dtype, generator=generator)
    s = torch.rand(len(z1), device=z1.device, generator=generator)
    zs = ((1 - s[:, None, None]) * z0 + s[:, None, None] * z1) * zmask[..., None]
    pred = prior(zs, s, text, text_mask, language, lengths, zmask)
    error = ((pred - (z1 - z0)).square() * zmask[..., None]).sum((1, 2))
    return (error / (zmask.sum(1) * z1.shape[-1])).mean()


def exact_path_velocity(path, s, *, create_graph=False):
    """§10.6 ablation hook: differentiate a caller-supplied curved SO(3) path.

    path(s) returns (rotations, expression), with one independent flow time per
    batch example. Endpoint shortcuts would be incorrect for curved paths.
    This function does not import or claim reproduction of a baseline model.
    """
    state, derivative = torch.autograd.functional.jvp(
        path, s, torch.ones_like(s), create_graph=create_graph
    )
    r, e = state
    dr, de = derivative
    body = r.transpose(-1, -2) @ dr
    omega = so3.vee(0.5 * (body - body.transpose(-1, -2)))
    return r, e, omega, de
