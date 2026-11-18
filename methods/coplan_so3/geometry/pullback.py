"""Matrix-free pullback metric. The tangent prediction retains its gradient."""

import torch

from . import so3


def tangent_mse(dw, de, mask):
    n = mask.sum(1).clamp_min(1)
    rot = (dw.square() * mask[..., None, None]).sum((1, 2, 3)) / (n * dw.shape[-2] * 3)
    expr = (de.square() * mask[..., None]).sum((1, 2)) / (n * de.shape[-1])
    return (rot + expr).mean()


def feature_jvp(features, r, e, dw, de, mask, times, create_graph=True):
    r, e = r.detach(), e.detach()

    def fn(w, v):
        groups = features(r @ so3.exp(w), e + v, mask, times)
        return torch.cat([x.flatten(1) for x in groups.values()], -1)

    _, jv = torch.autograd.functional.jvp(
        fn,
        (torch.zeros_like(dw), torch.zeros_like(de)),
        (dw, de),
        create_graph=create_graph,
        strict=False,
    )
    return jv


def kfm_loss(features, r, e, dw, de, mask, times, tangent_weight=1.0, pullback_weight=1.0):
    if tangent_weight <= 0 or pullback_weight < 0:
        raise ValueError("SPD metric requires positive tangent weight")
    tangent = tangent_mse(dw, de, mask)
    jv = feature_jvp(features, r, e, dw, de, mask, times)
    pullback = jv.square().sum(1).mean()
    return tangent_weight * tangent + pullback_weight * pullback, {
        "tangent": tangent.detach(),
        "pullback": pullback.detach(),
    }
