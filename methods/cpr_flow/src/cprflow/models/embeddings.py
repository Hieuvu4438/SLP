import math

import torch


def sinusoidal(values, dim, max_period=10000.0):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    angles = values.float()[..., None] * frequencies
    result = torch.cat((angles.cos(), angles.sin()), -1)
    if dim % 2:
        result = torch.nn.functional.pad(result, (0, 1))
    return result
