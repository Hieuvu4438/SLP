"""A train-fitted SO(3) chart with the original SOKE axes at its boundaries."""

import torch
from torch import Tensor, nn

from slp.geometry import aa_to_quat, quat_to_aa, sign_convention

ROTATION_DIM = 123
JOINTS = ROTATION_DIM // 3


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def multiply(a, b):
    aw, bw = a[..., :1], b[..., :1]
    av, bv = torch.broadcast_tensors(a[..., 1:], b[..., 1:])
    scalar = aw * bw - (av * bv).sum(-1, keepdim=True)
    vector = aw * bv + bw * av + torch.linalg.cross(av, bv)
    return torch.cat((scalar, vector), -1)


class RotationChart(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("raw_mean", torch.zeros(ROTATION_DIM))
        self.register_buffer("raw_std", torch.ones(ROTATION_DIM))
        self.register_buffer("rest", torch.zeros(ROTATION_DIM))
        self.register_buffer("reference", torch.zeros(JOINTS, 4))
        self.reference[:, 0] = 1

    def rotations(self, poses: Tensor):
        raw = poses[..., :ROTATION_DIM] * self.raw_std + self.raw_mean + self.rest
        return aa_to_quat(raw.unflatten(-1, (JOINTS, 3)))

    def to_tangent(self, poses: Tensor):
        delta = multiply(conjugate(self.reference), self.rotations(poses))
        vectors = quat_to_aa(sign_convention(delta)).flatten(-2)
        return torch.cat((vectors, poses[..., ROTATION_DIM:]), dim=-1)

    def from_tangent(self, coordinates: Tensor):
        delta = aa_to_quat(coordinates[..., :ROTATION_DIM].unflatten(-1, (JOINTS, 3)))
        raw = quat_to_aa(sign_convention(multiply(self.reference, delta))).flatten(-2)
        normalized = (raw - self.rest - self.raw_mean) / self.raw_std
        return torch.cat((normalized, coordinates[..., ROTATION_DIM:]), dim=-1)

    @torch.no_grad()
    def fit(self, rig, batches):
        self.raw_mean.copy_(rig.mean.reshape(-1)[:ROTATION_DIM])
        self.raw_std.copy_(rig.std.reshape(-1)[:ROTATION_DIM])
        # Stored order is body, fingers, jaw. The physical SMPL-X rig stores jaw before hands.
        rest = rig.pose_mean.reshape(-1)
        self.rest.copy_(torch.cat((rest[36:66], rest[75:165], rest[66:69])))
        # Sum quaternion outer products so antipodal annotations contribute identically.
        scatter = torch.zeros(JOINTS, 4, 4, dtype=torch.float64)
        for packed in batches:
            for batch in packed:
                valid = torch.arange(batch.poses.shape[1])[None] < batch.lengths[:, None]
                q = self.rotations(batch.poses[valid].to(self.reference)).cpu().double()
                scatter += q.permute(1, 2, 0) @ q.transpose(0, 1)
        self.reference.copy_(sign_convention(torch.linalg.eigh(scatter).eigenvectors[..., -1]))
