"""SDPA wrapper: True=allowed, finite all-masked rows, zero invalid queries."""

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, query, context, allowed, query_valid=None, bias=None):
        b, n, h = query.shape

        def project(layer, x):
            return layer(x).reshape(b, -1, self.heads, h // self.heads).transpose(1, 2)

        q, k, v = project(self.q, query), project(self.k, context), project(self.v, context)
        mask = allowed[:, None]
        if bias is not None:
            mask = torch.where(mask, bias[:, None].to(q.dtype), float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b, n, h)
        valid = allowed.any(-1)
        if query_valid is not None:
            valid = valid & query_valid
        return self.out(y) * valid[..., None]


def relation_key_bias(query_parts, key_parts, alpha, device):
    affected = (query_parts[:, None] < 4) & (key_parts[None, :] == 4)
    if alpha <= 0:
        return ~affected, torch.zeros_like(affected, dtype=torch.float32)
    import math

    bias = torch.where(affected, torch.tensor(math.log(alpha), device=device), 0.0)
    return torch.ones_like(affected, dtype=torch.bool), bias
