import torch


def lengths_mask(lengths, maximum=None):
    maximum = int(lengths.max()) if maximum is None else maximum
    return torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]


def pool2(x, lengths):
    """Valid pair mean, time dimension 1; never average odd final frame with zero."""
    mask = lengths_mask(lengths, x.shape[1])
    shape = (*mask.shape, *((1,) * (x.ndim - 2)))
    x = x * mask.reshape(shape)
    if x.shape[1] % 2:
        x = torch.cat((x, torch.zeros_like(x[:, :1])), 1)
        mask = torch.cat((mask, torch.zeros_like(mask[:, :1])), 1)
    summed = x[:, 0::2] + x[:, 1::2]
    count = (mask[:, 0::2].float() + mask[:, 1::2].float()).clamp_min(1)
    return summed / count.reshape(*count.shape, *((1,) * (x.ndim - 2)))


def derivative(x, times):
    dt = times[..., 1:] - times[..., :-1]
    if (dt <= 0).any():
        raise ValueError("Timestamps must strictly increase within each true clip")
    shape = (*dt.shape, *((1,) * (x.ndim - dt.ndim)))
    return (x[:, 1:] - x[:, :-1]) / dt.reshape(shape)
