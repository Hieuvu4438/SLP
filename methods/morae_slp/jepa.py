"""Sign-JEPA: MoRAE's Motion-JEPA (paper Sec. 3.1) on the 42 SMPL-X signing groups.

An online encoder reads poses whose masked groups are replaced, in every frame, by learned mask
values; a predictor regresses the layer-normalised frame representations that an EMA teacher
computes from the clean clip (smooth-L1 on every valid frame). After pre-training the teacher
provides H = LN(sum of its last four block outputs) for the compact-latent stage.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import N_GROUPS, POSE_DIM, group_channel_matrix, valid_frames, with_velocity
from .layers import Transformer, init_weights, masked_mean


class FrameEncoder(nn.Module):
    """One token per frame -> RoPE Transformer. Returns (final normed output, block outputs)."""

    def __init__(self, in_dim: int, width: int, depth: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, width)
        self.transformer = Transformer(width, depth, heads, mlp_ratio, rope=True)
        self.norm = nn.LayerNorm(width)
        self.width = width

    def forward(self, u: Tensor, valid: Tensor) -> tuple[Tensor, list[Tensor]]:
        hidden = self.transformer(self.embed(u), valid)
        return self.norm(hidden[-1]), hidden


def frame_features(hidden: list[Tensor], layers: int) -> Tensor:
    """MoRAE's generative features: LN(sum of the last ``layers`` block outputs), no affine."""
    total = torch.stack([h.float() for h in hidden[-layers:]]).sum(0)  # sum in FP32
    return F.layer_norm(total, (total.shape[-1],))


class Predictor(nn.Module):
    def __init__(self, width: int, pred_width: int, depth: int, heads: int,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.inp = nn.Linear(width, pred_width)
        self.transformer = Transformer(pred_width, depth, heads, mlp_ratio, rope=True)
        self.norm = nn.LayerNorm(pred_width)
        self.out = nn.Linear(pred_width, width)

    def forward(self, h: Tensor, valid: Tensor) -> Tensor:
        return self.out(self.norm(self.transformer(self.inp(h), valid)[-1]))


ARCH_KEYS = ("width", "depth", "heads", "pred_width", "pred_depth", "pred_heads", "mlp_ratio",
             "input_velocity", "feature_layers")


class SignJEPA(nn.Module):
    def __init__(self, width: int = 768, depth: int = 8, heads: int = 12, pred_width: int = 384,
                 pred_depth: int = 6, pred_heads: int = 6, mlp_ratio: float = 4.0,
                 input_velocity: bool = True, feature_layers: int = 4):
        super().__init__()
        if not 1 <= feature_layers <= depth:
            raise ValueError("feature_layers must be between 1 and the encoder depth")
        self.input_velocity = input_velocity
        self.feature_layers = feature_layers
        self.in_dim = POSE_DIM * (2 if input_velocity else 1)
        self.online = FrameEncoder(self.in_dim, width, depth, heads, mlp_ratio)
        self.predictor = Predictor(width, pred_width, pred_depth, pred_heads, mlp_ratio)
        init_weights(self.online)
        init_weights(self.predictor)
        self.mask_value = nn.Parameter(torch.zeros(self.in_dim))
        nn.init.trunc_normal_(self.mask_value, std=0.02)
        self.teacher = copy.deepcopy(self.online).requires_grad_(False)
        self.register_buffer("members", group_channel_matrix(input_velocity), persistent=False)

    def inputs(self, poses: Tensor, lengths: Tensor) -> Tensor:
        if self.input_velocity:
            return with_velocity(poses, lengths)
        return poses * valid_frames(lengths, poses.shape[1]).unsqueeze(-1)

    def channel_mask(self, group_mask: Tensor) -> Tensor:
        """[B, 42] group mask -> [B, C] channel mask."""
        if group_mask.shape[-1] != N_GROUPS:
            raise ValueError(f"group mask must have {N_GROUPS} columns")
        return (group_mask.to(self.members.device).float() @ self.members.float()) > 0

    def masked_inputs(self, u: Tensor, group_mask: Tensor) -> Tensor:
        cm = self.channel_mask(group_mask)[:, None, :]
        return torch.where(cm, self.mask_value.to(u.dtype), u)

    def loss(self, poses: Tensor, lengths: Tensor, group_mask: Tensor) -> dict[str, Tensor]:
        valid = valid_frames(lengths, poses.shape[1])
        u = self.inputs(poses, lengths)
        h, _ = self.online(self.masked_inputs(u, group_mask), valid)
        predicted = self.predictor(h, valid)
        with torch.no_grad():
            self.teacher.eval()
            target, _ = self.teacher(u, valid)
            target = F.layer_norm(target.float(), (target.shape[-1],))
        error = F.smooth_l1_loss(predicted.float(), target, reduction="none")
        loss = masked_mean(error, valid)
        # Collapse monitor: per-channel std of the target over valid frames (about 1 if healthy).
        values = target[valid]
        spread = values.std(0).mean() if len(values) > 1 else target.new_zeros(())
        return {"loss": loss, "target_std": spread.detach(),
                "masked_groups": group_mask.float().sum(1).mean().detach()}

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        online = [p.detach() for p in self.online.parameters()]
        teacher = list(self.teacher.parameters())
        torch._foreach_mul_(teacher, momentum)
        torch._foreach_add_(teacher, online, alpha=1.0 - momentum)

    @torch.no_grad()
    def features(self, poses: Tensor, lengths: Tensor) -> Tensor:
        """Frozen teacher features H [B, T, width] for the compact-latent stage."""
        valid = valid_frames(lengths, poses.shape[1])
        self.teacher.eval()
        _, hidden = self.teacher(self.inputs(poses, lengths), valid)
        return frame_features(hidden, self.feature_layers)


def teacher_momentum(step: int, total: int, start: float, end: float = 1.0) -> float:
    """Linear schedule from ``start`` to ``end`` over the updates (I-JEPA)."""
    return start + (end - start) * min(1.0, step / max(1, total))
