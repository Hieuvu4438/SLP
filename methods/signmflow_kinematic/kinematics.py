"""Differentiable SMPL-X skeleton and ten skinned fingertip vertices."""

from pathlib import Path

import torch
from torch import nn

from slp.evaluation.smplx_fk import SOKE_SHAPE

from . import geometry as G

POSE_JOINTS = (*range(12, 22), *range(25, 55), 22)
TIP_VERTICES = (5361, 4933, 5058, 5169, 5286, 8079, 7669, 7794, 7905, 8022)
WRISTS = (20, 21)
GROUPS = {"body": slice(0, 30), "wrist": slice(30, 36),
          "fingers": slice(36, 66), "palm": slice(66, 78)}


class SigningKinematics(nn.Module):
    """Compute signing observables in the fixed SOKE shape, with a neutral face.

    Inputs are physical local rotations ordered as in MotionCodec. No hand-mean
    offset is added here. The fingertip skinning uses the same pose correctives
    and blend weights as the evaluator. Body features use skeletal joints.
    """

    @classmethod
    def from_smplx(cls, model_dir: Path):
        import smplx

        layer = smplx.create(str(model_dir), "smplx", gender="NEUTRAL",
                             use_pca=False, num_betas=10)
        shape = torch.tensor(SOKE_SHAPE, dtype=layer.v_template.dtype)
        vertices = layer.v_template + torch.einsum("vck,k->vc", layer.shapedirs[..., :10], shape)
        joints = layer.J_regressor @ vertices
        tips = list(TIP_VERTICES)
        pose_dirs = layer.posedirs.reshape(layer.posedirs.shape[0], -1, 3)[:, tips].flatten(1)
        return cls(joints, tuple(layer.parents.tolist()), vertices[tips],
                   pose_dirs, layer.lbs_weights[tips])

    def __init__(self, joints, parents, tips, pose_dirs, weights):
        super().__init__()
        if len(parents) != 55 or parents[0] != -1 or any(not 0 <= p < i for i, p in enumerate(parents[1:], 1)):
            raise ValueError("Expected the topologically ordered 55-joint SMPL-X tree")
        self.parents = parents
        offsets = torch.cat((joints[:1], joints[1:] - joints[list(parents[1:])]))
        self.register_buffer("joints", joints)
        self.register_buffer("offsets", offsets)
        self.register_buffer("tips", tips)
        self.register_buffer("pose_dirs", pose_dirs)
        self.register_buffer("weights", weights)
        self.register_buffer("shoulder_width", (joints[16] - joints[17]).norm().clamp_min(1e-5))
        self.register_buffer("palm_lengths", (joints[[28, 43]] - joints[list(WRISTS)]).norm(dim=-1).clamp_min(1e-5))
        self.register_buffer("identity", torch.eye(3, dtype=joints.dtype))

    def points(self, local):
        """Physical rotations [...,41,3,3] -> skeleton, global rotations, tips."""
        lead = local.shape[:-3]
        rotations = self.identity.expand(*lead, 55, 3, 3)
        indices = torch.tensor(POSE_JOINTS, device=local.device)
        rotations = torch.index_copy(rotations, -3, indices, local)
        world_r = [rotations[..., 0, :, :]]
        world_p = [self.offsets[0].expand(*lead, 3)]
        for j, parent in enumerate(self.parents[1:], 1):
            world_p.append(world_p[parent] + (world_r[parent] @ self.offsets[j]))
            world_r.append(world_r[parent] @ rotations[..., j, :, :])
        positions = torch.stack(world_p, dim=-2)
        orientation = torch.stack(world_r, dim=-3)

        offsets = ((rotations[..., 1:, :, :] - self.identity).flatten(-3) @ self.pose_dirs)
        posed_tips = self.tips + offsets.reshape(*lead, 10, 3)
        translations = positions - (orientation @ self.joints[..., None]).squeeze(-1)
        tip_rotations = torch.einsum("vj,...jcd->...vcd", self.weights, orientation)
        tip_translations = torch.einsum("vj,...jc->...vc", self.weights, translations)
        tips = (tip_rotations @ posed_tips[..., None]).squeeze(-1) + tip_translations
        return positions, orientation, tips

    def forward(self, local):
        xyz, axes, tips = self.points(local)
        torso = axes[..., 9, :, :].transpose(-1, -2)
        body = (xyz[..., 12:22, :] - xyz[..., :1, :]) / self.shoulder_width
        wrist_xyz, wrist_axes = xyz[..., list(WRISTS), :], axes[..., list(WRISTS), :, :]
        wrists = ((wrist_xyz - xyz[..., 9:10, :]) @ torso.transpose(-1, -2)) / self.shoulder_width
        hands = (tips.unflatten(-2, (2, 5)) - wrist_xyz[..., :, None, :]) @ wrist_axes
        hands = hands / self.palm_lengths[..., None, None]
        palm_axes = (torso[..., None, :, :] @ wrist_axes)[..., :2]
        return torch.cat((body.flatten(-2), wrists.flatten(-2), hands.flatten(-3),
                          palm_axes.flatten(-3)), dim=-1)


def tangent_pushforward(fk, rotations, velocity):
    """Differential dF(R)[v] for right-multiplied, body-frame angular velocities."""
    return torch.func.jvp(lambda delta: fk(rotations + rotations @ G.infinitesimal_rotation(delta)),
                          (torch.zeros_like(velocity),), (velocity,))[1]


def axis_jacobian(fk, rotations, chunk=8):
    """Jacobian in the same 123 body-angular axes as tangent_pushforward."""
    def local_jacobian(rotation):
        delta = rotation.new_zeros(rotation.shape[0], 3)
        return torch.func.jacrev(lambda tangent: fk(rotation + rotation @ G.infinitesimal_rotation(tangent)),
                                 chunk_size=chunk)(delta)

    return torch.vmap(local_jacobian)(rotations)
