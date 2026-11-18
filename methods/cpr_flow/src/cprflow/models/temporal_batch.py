"""Vectorized true-length replicate boundaries, including every residual convolution.

Decoder internal true length is 2*ceil(T/2), even when final physical T is odd.
These functions reuse existing modules/parameters; no checkpoint tensor key changes.
"""

import torch
from torch import nn


def replicate_valid(x, lengths):
    positions = torch.minimum(
        torch.arange(x.shape[-1], device=x.device)[None], lengths[:, None] - 1
    )
    return x.gather(2, positions[:, None].expand(-1, x.shape[1], -1))


def apply_temporal(layers, x, lengths):
    from cprflow.models.ae import Residual

    for layer in layers:
        if isinstance(layer, nn.Conv1d):
            x = layer(replicate_valid(x, lengths))
            lengths = (
                lengths + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1
            ) // layer.stride[0] + 1
        elif isinstance(layer, Residual):
            residual, _ = apply_temporal(layer.net, x, lengths)
            x = x + residual
        elif isinstance(layer, nn.Upsample):
            x = layer(x)
            lengths = lengths * int(layer.scale_factor)
        else:
            x = layer(x)
    return x, lengths


def encode(model, x, lengths):
    from cprflow.types import PART_DIMS

    encoded = []
    for encoder, part in zip(model.encoders, x.split(PART_DIMS, -1)):
        y, n = apply_temporal(encoder, part.transpose(1, 2), lengths)
        mask = torch.arange(y.shape[-1], device=x.device)[None] < n[:, None]
        encoded.append(y.transpose(1, 2) * mask[..., None])
    return torch.stack(encoded, 1)


def decode(model, z, lengths):
    batch, _, maximum, _ = z.shape
    packed = z.permute(0, 2, 1, 3).reshape(batch, maximum, 512)
    fused = (packed + model.fusion(packed)).reshape(batch, maximum, 4, 128).permute(0, 2, 1, 3)
    outputs = []
    for p, decoder in enumerate(model.decoders):
        y, _ = apply_temporal(decoder, fused[:, p].transpose(1, 2), (lengths + 1) // 2)
        outputs.append(y.transpose(1, 2))
    physical_max = int(lengths.max())
    result = torch.cat(outputs, -1)[:, :physical_max]
    mask = torch.arange(physical_max, device=z.device)[None] < lengths[:, None]
    return result * mask[..., None]
