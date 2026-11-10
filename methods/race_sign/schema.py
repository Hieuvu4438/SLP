"""SOKE raw schema: body30, LH45, RH45, jaw3, expression10.
The first 36 coordinates of the full SOKE SMPL-X pose are omitted (root + 11 body joints).
"""

import torch
from torch import nn
from .rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
)

PART_SLICES = (slice(0, 60), slice(60, 150), slice(150, 240))
BODY_FULL_JOINTS = tuple(range(12, 22))
LEFT_FULL_JOINTS = tuple(range(25, 40))
RIGHT_FULL_JOINTS = tuple(range(40, 55))


def manual_to_6d(pose, manual_mean=None):
    if pose.shape[-1] != 133:
        raise ValueError("Raw SOKE poses must have 133 dimensions, in radians")
    angles = pose[..., :120]
    if manual_mean is not None:
        angles = angles + manual_mean.to(pose)
    rot = axis_angle_to_matrix(angles.reshape(*pose.shape[:-1], 40, 3))
    return matrix_to_rotation_6d(rot).flatten(-2)


def restore_pose(manual, baseline, manual_mean=None):
    if manual.shape[:-1] != baseline.shape[:-1] or manual.shape[-1] != 240:
        raise ValueError("Manual features and baseline timeline must agree")
    rot = rotation_6d_to_matrix(manual.reshape(*manual.shape[:-1], 40, 6))
    angles = matrix_to_axis_angle(rot).flatten(-2)
    if manual_mean is not None:
        angles = angles - manual_mean.to(angles)
    return torch.cat((angles, baseline[..., 120:133]), -1)


class MotionNormalizer(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        mean = torch.zeros(240) if mean is None else torch.as_tensor(mean).float()
        std = torch.ones(240) if std is None else torch.as_tensor(std).float()
        if (
            mean.shape != (240,)
            or std.shape != (240,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("Expected finite 240D mean and positive std")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    @torch.no_grad()
    def fit(self, records, manual_mean=None):
        count = 0
        total = torch.zeros(240, dtype=torch.float64, device=self.mean.device)
        square = total.clone()
        for pose in records:
            x = manual_to_6d(pose.to(self.mean.device), manual_mean).reshape(-1, 240).double()
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite training pose")
            count += len(x)
            total += x.sum(0)
            square += x.square().sum(0)
        if count < 2:
            raise ValueError("At least two training frames required")
        mean = total / count
        self.mean.copy_(mean)
        self.std.copy_((square / count - mean.square()).clamp_min(1e-8).sqrt())
