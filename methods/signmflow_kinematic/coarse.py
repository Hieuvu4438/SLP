"""Text-conditioned coarse motion with no target anchors at inference."""

import math

import torch
from torch import nn

from . import geometry as G


def coarse_targets(pose, expression, mask, stride):
    rotations = G.matrix_to_6d(G.quat_to_matrix(pose)) if pose.shape[-1] == 4 else pose
    features = torch.cat((rotations.flatten(-2), expression), -1)
    b, t, c = features.shape
    bins = math.ceil(t / stride)
    pad = bins * stride - t
    weights = torch.nn.functional.pad(mask, (0, pad)).reshape(b, bins, stride)
    values = torch.nn.functional.pad(features, (0, 0, 0, pad)).reshape(b, bins, stride, c)
    sums = (values * weights[..., None]).sum(2)
    return sums / weights.sum(2, keepdim=True).clamp_min(1), weights.any(2)


class CoarsePlan(nn.Module):
    def __init__(self, latent, text_dim, heads, features, max_frames, stride):
        super().__init__()
        if stride < 2:
            raise ValueError("Coarse planning needs a stride of at least two")
        self.stride = stride
        self.position = nn.Parameter(torch.randn(math.ceil(max_frames / stride), latent) * 0.02)
        self.progress = nn.Linear(2, latent)
        self.cross = nn.MultiheadAttention(latent, heads, kdim=text_dim, vdim=text_dim, batch_first=True)
        self.norm = nn.LayerNorm(latent)
        self.prediction = nn.Linear(latent, features)
        self.readout = nn.Linear(features, latent)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, text, text_pad, mask):
        bins = math.ceil(mask.shape[1] / self.stride)
        lengths = mask.sum(1)
        count = (lengths + self.stride - 1) // self.stride
        index = torch.arange(bins, device=mask.device)
        progress = ((index + 0.5) / count[:, None].clamp_min(1)).clamp(max=1.)
        scale = lengths.float().log()[:, None].expand_as(progress)
        query = self.position[:bins] + self.progress(torch.stack((progress, scale), -1))
        decoded = self.cross(query, text, text, key_padding_mask=text_pad, need_weights=False)[0]
        plan = self.prediction(self.norm(query + decoded))
        return plan

    def dense(self, plan, mask):
        """Linearly interpolate between bin centers and clamp to each clip's support."""
        index = torch.arange(mask.shape[1], device=mask.device)
        u = (index + 0.5) / self.stride - 0.5
        last = ((mask.sum(1) - 1) // self.stride)[:, None]
        u = torch.minimum(u.clamp_min(0), last)
        a = u.floor().long()
        b = (a + 1).minimum(last)
        ids = torch.arange(len(plan), device=plan.device)[:, None]
        value = torch.lerp(plan[ids, a], plan[ids, b], (u - a)[..., None])
        return self.readout(value)
