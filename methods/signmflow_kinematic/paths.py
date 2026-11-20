"""Lift and quotient paths used by the matched experiments."""

import torch
from torch import nn

from . import geometry as G

PATHS = {"linear", "geodesic", "naturalness", "so3", "so3_corrected", "projected_6d"}


class CorrectionNet(nn.Module):
    def __init__(self, n_joints=41, hidden=512, layers=3, quotient=False, max_angle=None):
        super().__init__()
        if layers < 1 or (max_angle is not None and max_angle <= 0):
            raise ValueError("Invalid correction layers or maximum angular correction")
        input_dim, output_dim = (6, 3) if quotient else (4, 4)
        dims = [1 + 2 * n_joints * input_dim] + [hidden] * (layers - 1) + [n_joints * output_dim]
        modules = []
        for i in range(layers):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i + 1 < layers:
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n_joints = n_joints
        self.quotient = quotient
        self.max_angle = max_angle

    def forward(self, t, x0, x1):
        leading = x0.shape[:-2]
        t = t.expand(*leading, 1, 1).reshape(*leading, 1)
        if self.quotient:
            a, b = (G.matrix_to_6d(G.quat_to_matrix(q)) for q in (x0, x1))
            raw = self.net(torch.cat((t, a.flatten(-2), b.flatten(-2)), -1))
            correction = raw.reshape(*leading, self.n_joints, 3)
            if self.max_angle is not None:
                size = correction.norm(dim=-1, keepdim=True)
                correction = correction * (self.max_angle / size.clamp_min(self.max_angle))
            return correction
        raw = self.net(torch.cat((t, x0.flatten(-2), x1.flatten(-2)), -1)).reshape_as(x0)
        return G.project(x0, raw)


def path_point(kind, x0, x1, t, correction=None):
    if kind == "linear":
        return torch.lerp(x0, x1, t)
    if kind == "projected_6d":
        return G.projected_6d_path(x0, x1, t)
    if kind in {"so3", "so3_corrected"}:
        displacement = t * G.relative_rotvec(x0, x1)
        if kind == "so3_corrected":
            displacement = displacement + t * (1 - t) * correction(t, x0, x1)
        return G.rotate_by(x0, displacement)
    if kind in {"geodesic", "naturalness"}:
        displacement = t * G.logmap(x0, x1)
        if kind == "naturalness":
            displacement = displacement + t * (1 - t) * correction(t, x0, x1)
        return G.expmap(x0, displacement)
    raise ValueError(f"Unknown transport path {kind!r}")


def path_and_velocity(kind, x0, x1, t, correction=None):
    return torch.func.jvp(lambda tau: path_point(kind, x0, x1, tau, correction),
                          (t,), (torch.ones_like(t),))
