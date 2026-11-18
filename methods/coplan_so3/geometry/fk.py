"""Differentiable tree FK and an exact adapter to the shared SOKE evaluator landmarks."""

from __future__ import annotations

import torch
from torch import nn


class TreeFK(nn.Module):
    """Local active rotations. Explicit full skeleton includes all fixed joints.

    landmarks: body/left/right lists of full-skeleton indices; hand entry 0
    is the wrist, the remaining entries are explicitly selected fingertips.
    """

    def __init__(self, parents, offsets, mapping, body, left, right, torso):
        super().__init__()
        self.parents, self.mapping = tuple(parents), tuple(mapping)
        if (
            parents[0] != -1
            or any(p < 0 or p >= i for i, p in enumerate(parents[1:], 1))
            or len(set(mapping)) != len(mapping)
        ):
            raise ValueError("Topologically ordered tree and unique mapping required")
        if any(i < 0 or i >= len(parents) for i in (*mapping, *body, *left, *right, torso)):
            raise ValueError("Skeleton index out of range")
        if len(left) < 2 or len(right) != len(left):
            raise ValueError("Matching wrist and fingertip landmarks required")
        self.register_buffer("offsets", torch.as_tensor(offsets).float())
        if self.offsets.shape != (len(parents), 3):
            raise ValueError("One rest offset per skeleton joint required")
        self.body, self.left, self.right, self.torso = body, left, right, torso

    def forward(self, r, e):
        eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(*r.shape[:-3], 3, 3)
        local = [eye for _ in self.parents]
        for k, i in enumerate(self.mapping):
            local[i] = r[..., k, :, :]
        global_r, pos = [local[0]], [self.offsets[0].to(r).expand(*r.shape[:-3], 3)]
        for i, p in enumerate(self.parents[1:], 1):
            global_r.append(global_r[p] @ local[i])
            pos.append(pos[p] + (global_r[p] @ self.offsets[i].to(r)[..., None])[..., 0])
        p = torch.stack(pos, -2)
        return {
            "body": p[..., self.body, :],
            "left": p[..., self.left, :],
            "right": p[..., self.right, :],
            "torso": p[..., self.torso, :],
            "left_rotation": global_r[self.left[0]],
            "right_rotation": global_r[self.right[0]],
        }


class SOKEFK(nn.Module):
    """Uses the same skinning, shape, regressor and fixed joints as slp.evaluation.

    The shared sparse implementation includes fingertip mesh landmarks and
    expression blend shapes; no arbitrary skeleton replacement is used.
    """

    def __init__(self, manifest, codec, model_dir=None, stats=None, fk=None):
        super().__init__()
        from slp.skeleton_fk import FEAT_TO_SMPLX, SkeletonFK

        if tuple(FEAT_TO_SMPLX) != manifest.smplx_joint_indices:
            raise ValueError("Manifest and SOKE FK mapping differ")
        self.fk = fk if fk is not None else SkeletonFK(model_dir=model_dir, stats=stats)
        if not torch.allclose(self.fk.hand_mean, codec.hand_mean):
            raise ValueError("Codec and FK hand means differ")
        self.register_buffer("expr_mean", codec.mean[123:].clone())
        self.register_buffer("expr_std", codec.std[123:].clone())
        self.mapping = tuple(FEAT_TO_SMPLX)
        self.parents = self.fk.parents

    def forward(self, r, e):
        points = self.fk(r, e * self.expr_std + self.expr_mean)
        eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(*r.shape[:-3], 3, 3)
        local = [eye for _ in self.parents]
        for k, i in enumerate(self.mapping):
            local[i] = r[..., k, :, :]
        global_r = [local[0]]
        for i, p in enumerate(self.parents[1:], 1):
            global_r.append(global_r[p] @ local[i])
        # Hand regressor order: wrist, thumb 1..tip, index 1..tip, ...
        tips = [0, 4, 8, 12, 16, 20]
        return {
            "body": points["body"],
            "left": points["lhand"][..., tips, :],
            "right": points["rhand"][..., tips, :],
            "torso": points["body"][..., 0, :],
            "left_rotation": global_r[20],
            "right_rotation": global_r[21],
        }
