"""Differentiable SOKE wrist and fingertip positions for endpoint supervision."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from slp.evaluation.smplx_fk import SOKE_SHAPE, SMPLXJoints


class SigningEndpointLoss(nn.Module):
    """Use the evaluation rig, skinning only the ten evaluated fingertip vertices."""

    def __init__(self, stats):
        super().__init__()
        ref = SMPLXJoints(device="cpu", stats=stats)
        layer = ref.layer
        tips = torch.cat((ref.lreg[4::4], ref.rreg[4::4])).argmax(1)
        neutral = layer.v_template + torch.einsum(
            "vcb,b->vc", layer.shapedirs, torch.as_tensor(SOKE_SHAPE)
        )
        self.register_buffer("point_base", neutral[tips].contiguous())
        self.register_buffer("point_expression", layer.expr_dirs[tips].contiguous())
        self.register_buffer("point_pose", layer.posedirs.reshape(
            layer.posedirs.shape[0], -1, 3
        )[:, tips].flatten(1).contiguous())
        self.register_buffer("point_skinning", layer.lbs_weights[tips].contiguous())
        self.register_buffer("joints", layer.J_regressor @ neutral)
        self.register_buffer("joint_expression", torch.einsum(
            "jv,vce->jce", layer.J_regressor, layer.expr_dirs
        ))
        self.register_buffer("mean", ref.mean)
        self.register_buffer("std", ref.std)
        self.register_buffer("pose_mean", layer.pose_mean.clone())

        # SMPL-X ancestors are stored before descendants. Batch the transforms at each depth.
        parents = layer.parents.tolist()
        self.register_buffer("parents", layer.parents[1:].clone())
        depths = [0]
        for parent in parents[1:]:
            depths.append(depths[parent] + 1)
        order = [0]
        self.levels = max(depths)
        for level in range(1, self.levels + 1):
            children = [i for i, d in enumerate(depths) if d == level]
            self.register_buffer(f"level_{level}", torch.tensor(children))
            self.register_buffer(f"ancestors_{level}", torch.tensor(
                [order.index(parents[i]) for i in children]
            ))
            order.extend(children)
        self.register_buffer("restore_order", torch.argsort(torch.tensor(order)))

    def positions(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return wrist-to-pelvis and fingertip-to-wrist positions in centimetres."""
        from smplx.lbs import batch_rodrigues

        raw = features.float().reshape(-1, features.shape[-1]) * self.std + self.mean
        n = len(raw)
        rotations = torch.cat((
            raw.new_zeros(n, 36), raw[:, :30], raw[:, 120:123],
            raw.new_zeros(n, 6), raw[:, 30:120],
        ), dim=-1)
        rotations = rotations + self.pose_mean
        local_r = batch_rodrigues(rotations.reshape(-1, 3)).reshape(n, -1, 3, 3)
        expr = raw[:, 123:]
        rest = self.joints + torch.einsum("jce,ne->njc", self.joint_expression, expr)
        local_t = rest[:, 1:] - rest.index_select(1, self.parents)
        local_t = torch.cat((rest[:, :1], local_t), dim=1)

        world_r, world_t = local_r[:, :1], local_t[:, :1]
        for level in range(1, self.levels + 1):
            child = getattr(self, f"level_{level}")
            parent = getattr(self, f"ancestors_{level}")
            parent_r = world_r.index_select(1, parent)
            next_r = parent_r @ local_r.index_select(1, child)
            next_t = (parent_r @ local_t.index_select(1, child).unsqueeze(-1)).squeeze(-1)
            next_t = next_t + world_t.index_select(1, parent)
            world_r = torch.cat((world_r, next_r), dim=1)
            world_t = torch.cat((world_t, next_t), dim=1)
        world_r = world_r.index_select(1, self.restore_order)
        world_t = world_t.index_select(1, self.restore_order)

        eye = torch.eye(3, device=raw.device, dtype=raw.dtype)
        pose_feature = (local_r[:, 1:] - eye).flatten(1)
        tip_v = self.point_base + torch.einsum("vce,ne->nvc", self.point_expression, expr)
        tip_v = tip_v + (pose_feature @ self.point_pose).reshape(n, -1, 3)
        offsets = world_t - (world_r @ rest.unsqueeze(-1)).squeeze(-1)
        tip_r = torch.einsum("vj,njxy->nvxy", self.point_skinning, world_r)
        tip_t = torch.einsum("vj,njx->nvx", self.point_skinning, offsets)
        tips = ((tip_r @ tip_v.unsqueeze(-1)).squeeze(-1) + tip_t).reshape(n, 2, 5, 3)
        wrists = world_t[:, 20:22]
        return 100 * (wrists - world_t[:, :1]), 100 * (tips - wrists.unsqueeze(-2))

    def forward(
        self, predicted: Tensor, target: Tensor, lengths: Tensor, samples: int,
    ) -> tuple[Tensor, Tensor]:
        # Uniform midpoint sampling per clip keeps this auxiliary loss bounded for long sequences.
        samples = min(samples, predicted.shape[1])
        grid = 2 * torch.arange(samples, device=predicted.device) + 1
        frames = grid[None] * lengths[:, None] // (2 * samples)
        items = torch.arange(len(predicted), device=predicted.device)[:, None]
        with torch.autocast(predicted.device.type, enabled=False):
            wrist, hand = self.positions(predicted[items, frames])
            with torch.no_grad():
                ref_wrist, ref_hand = self.positions(target[items, frames])
            return (
                F.smooth_l1_loss(wrist, ref_wrist, beta=2.0),
                F.smooth_l1_loss(hand, ref_hand, beta=2.0),
            )
