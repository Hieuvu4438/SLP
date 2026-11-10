"""Endpoint-consistent linear conditional flow matching (DisCoRD/CFM formulation)."""

import math
import torch


def flow_path(target, time, noise):
    if target.shape != noise.shape or time.shape != target.shape[:1]:
        raise ValueError("Invalid flow path dimensions")
    t = time[:, None, None]
    return (1 - t) * noise + t * target, target - noise


def endpoint(state, time, velocity):
    return state + (1 - time[:, None, None]) * velocity


@torch.no_grad()
def sample_euler(model, noise, steps, condition, *, guidance=1.0, use_text=True, coupled=True):
    if steps < 1 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Positive steps and nonnegative guidance required")
    if model.training:
        raise ValueError("Set decoder.eval() before sampling")
    state = noise.clone()
    mask = torch.arange(noise.shape[1], device=noise.device)[None] < condition["lengths"][:, None]
    state = state.masked_fill(~mask[..., None], 0)
    nfe = 0
    for i in range(steps):
        t = torch.full((len(state),), i / steps, device=state.device, dtype=state.dtype)
        v = model(state, t, **condition, use_text=use_text, coupled=coupled)
        nfe += 1
        if use_text and guidance != 1:
            uncond = model(state, t, **condition, use_text=False, coupled=coupled)
            nfe += 1
            v = uncond + guidance * (v - uncond)
        state = (state + v / steps).masked_fill(~mask[..., None], 0)
    return state, nfe
