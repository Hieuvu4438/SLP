import math

import torch


class CosineSchedule:
    def __init__(self, steps=1000, offset=0.008, beta_min=1e-5, beta_max=0.999, device="cpu"):
        self.steps = steps
        k = torch.arange(steps + 1, device=device, dtype=torch.float64)
        raw = torch.cos(((k / steps + offset) / (1 + offset)) * math.pi / 2).square()
        raw = raw / raw[0]
        beta = (1 - raw[1:] / raw[:-1]).clamp(beta_min, beta_max)
        self.alpha_bar = torch.cat(
            (torch.ones(1, device=device), torch.cumprod(1 - beta, 0))
        ).float()

    def corrupt(self, x, noise, k):
        if ((k < 1) | (k > self.steps)).any():
            raise ValueError("DDPM train indices must be 1..K")
        a = self.alpha_bar[k].reshape(len(k), *((1,) * (x.ndim - 1)))
        return a.sqrt() * x + (1 - a).sqrt() * noise

    def indices(self, nfe):
        if nfe < 1 or nfe > self.steps:
            raise ValueError("DDIM NFE must be 1..K")
        return torch.linspace(self.steps, 1, nfe).round().long().tolist() + [0]

    def clean(self, state, eps, k):
        a = self.alpha_bar[k]
        return (state - (1 - a).sqrt() * eps) / a.sqrt()

    def ddim(self, state, eps, k, next_k):
        clean = self.clean(state, eps, k)
        a = self.alpha_bar[next_k]
        return a.sqrt() * clean + (1 - a).sqrt() * eps
