"""Part tokens with nonlinear finger encoders and high-resolution hand skips."""

import torch
from torch import nn

from ..geometry import so3
from .components import AxialBlock, Condition, LanguageAdapters, embedding, mlp


class PartInput(nn.Module):
    def __init__(self, manifest, dim):
        super().__init__()
        self.manifest = manifest
        self.body = mlp(len(manifest.body) * 6, dim, dim)
        self.fingers = nn.ModuleList(
            [mlp(len(f) * 6, dim, dim) for f in manifest.fingers_left + manifest.fingers_right]
        )
        self.left = mlp(len(manifest.fingers_left) * dim, dim, dim)
        self.right = mlp(len(manifest.fingers_right) * dim, dim, dim)
        self.expression = mlp(manifest.expression_dim, dim, dim)
        self.parts = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.dim = dim

    def forward(self, r, e, times):
        x, m = so3.to_6d(r), self.manifest
        fg = [
            net(x[..., f, :].flatten(-2))
            for f, net in zip(m.fingers_left + m.fingers_right, self.fingers)
        ]
        n = len(m.fingers_left)
        h = torch.stack(
            (
                self.body(x[..., m.body, :].flatten(-2)),
                self.left(torch.cat(fg[:n], -1)),
                self.right(torch.cat(fg[n:], -1)),
                self.expression(e),
            ),
            -2,
        )
        return h + self.parts + embedding(times, self.dim)[..., None, :]


class MotionHead(nn.Module):
    def __init__(self, manifest, dim, rotation_dim):
        super().__init__()
        self.manifest, self.rotation_dim = manifest, rotation_dim
        self.body = mlp(dim, len(manifest.body) * rotation_dim, dim)
        self.fingers = nn.ModuleList(
            [
                mlp(dim + len(f) * 6, len(f) * rotation_dim, dim)
                for f in manifest.fingers_left + manifest.fingers_right
            ]
        )
        self.expr = mlp(dim, manifest.expression_dim, dim)
        if rotation_dim == 6:
            identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            for head in [self.body, *self.fingers]:
                nn.init.normal_(head[-1].weight, std=0.001)
                with torch.no_grad():
                    head[-1].bias.copy_(identity.repeat(head[-1].out_features // 6))

    def forward(self, h, high_resolution=None):
        m, d = self.manifest, self.rotation_dim
        out = h.new_zeros(*h.shape[:2], m.joints, d)
        out[..., m.body, :] = self.body(h[..., 0, :]).unflatten(-1, (len(m.body), d))
        if high_resolution is None:
            high_resolution = h.new_zeros(*h.shape[:2], m.joints, 6)
        for k, (finger, head) in enumerate(zip(m.fingers_left + m.fingers_right, self.fingers)):
            part = 1 if k < len(m.fingers_left) else 2
            inp = torch.cat((h[..., part, :], high_resolution[..., finger, :].flatten(-2)), -1)
            out[..., finger, :] = head(inp).unflatten(-1, (len(finger), d))
        return out, self.expr(h[..., 3, :])


class DenseFlow(nn.Module):
    def __init__(
        self,
        manifest,
        text_dim,
        plan_dim=256,
        dim=512,
        layers=8,
        heads=8,
        ff=2048,
        dropout=0.1,
        auxiliary=False,
    ):
        super().__init__()
        if dim % 2:
            raise ValueError("Even hidden dimension required for time embeddings")
        self.input = PartInput(manifest, dim)
        self.condition = Condition(dim, text_dim, plan_dim)
        self.flow_time = mlp(dim, dim, dim)
        self.blocks = nn.ModuleList([AxialBlock(dim, heads, ff, dropout) for _ in range(layers)])
        self.adapters = LanguageAdapters(dim)
        self.velocity = MotionHead(manifest, dim, 3)
        self.endpoint = MotionHead(manifest, dim, 6) if auxiliary else None
        self.dim = dim

    def forward(self, r, e, s, text, text_mask, z, zmask, language, mask, times):
        h = self.input(r, e, times) + self.flow_time(embedding(s * 1000, self.dim))[:, None, None]
        context, cmask = self.condition(text, text_mask, language, mask.sum(1), z, zmask)
        for block in self.blocks:
            h = block(h, mask, context, cmask)
        h = self.adapters(h, language)
        w, v = self.velocity(h, so3.to_6d(r))
        aux = None
        if self.endpoint is not None:
            a, b = self.endpoint(h, so3.to_6d(r))
            aux = so3.from_6d(a), b * mask[..., None]
        return w * mask[..., None, None], v * mask[..., None], aux
