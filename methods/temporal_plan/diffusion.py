"""Cosine DDPM training and explicit DDIM reverse steps (clean-latent prediction)."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .tensors import check_sequence, valid_mask, zero_padding


@dataclass
class DenoisingLoss:
    loss: Tensor
    clean_prediction: Tensor
    timesteps: Tensor
    alpha_bar: Tensor


class CosineDiffusion(nn.Module):
    def __init__(self, steps: int = 1000, offset: float = .008):
        super().__init__()
        if steps < 1 or not math.isfinite(offset) or offset < 0:
            raise ValueError("invalid diffusion schedule")
        t = torch.arange(steps + 1, dtype=torch.float64) / steps
        target = torch.cos((t + offset)/(1 + offset) * math.pi/2).square()
        target = target / target[0]
        beta = (1 - target[1:] / target[:-1]).clamp(0, .999)
        alpha_bar = torch.cat((torch.ones(1, dtype=torch.float64), (1-beta).cumprod(0)))
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.steps = steps

    def _alpha(self, timesteps: Tensor, like: Tensor) -> Tensor:
        if timesteps.dtype != torch.long or timesteps.shape != (len(like),):
            raise ValueError("timesteps must be int64 [B]")
        if bool(((timesteps < 0) | (timesteps > self.steps)).any()):
            raise ValueError("timestep outside schedule")
        return self.alpha_bar[timesteps].to(like).reshape(-1, *([1]*(like.ndim-1)))

    def q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        if clean.shape != noise.shape:
            raise ValueError("noise and clean shapes disagree")
        a = self._alpha(timesteps, clean)
        return a.sqrt() * clean + (1-a).sqrt() * noise

    def training_loss(self, model: nn.Module, clean: Tensor, lengths: Tensor,
                      memory: Tensor, text_valid: Tensor, condition: Tensor, *,
                      generator: torch.Generator | None = None, plan_dropout: float = .1,
                      timesteps: Tensor | None = None, noise: Tensor | None = None
                      ) -> DenoisingLoss:
        check_sequence(clean, lengths)
        if not 0 <= plan_dropout <= 1:
            raise ValueError("invalid plan dropout probability")
        clean = zero_padding(clean, lengths)
        if timesteps is None:
            timesteps = torch.randint(1, self.steps+1, (len(clean),),
                                       device=clean.device, generator=generator)
        if noise is None:
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype,
                                generator=generator)
        noisy = zero_padding(self.q_sample(clean, timesteps, zero_padding(noise, lengths)), lengths)
        if model.training and plan_dropout:
            keep = torch.rand(len(clean), device=clean.device, generator=generator) >= plan_dropout
            condition = torch.where(keep[:, None, None], condition, torch.zeros_like(condition))
        prediction = model(noisy, timesteps.to(clean)/self.steps, memory, text_valid,
                           condition, lengths)
        mask = valid_mask(lengths, clean.shape[1])
        error = (prediction-clean).square().flatten(2).mean(-1)
        loss = (torch.where(mask, error, torch.zeros_like(error)).sum(-1) / lengths).mean()
        return DenoisingLoss(loss, prediction, timesteps, self.alpha_bar[timesteps])

    def ddim_step(self, state: Tensor, timesteps: Tensor, previous: Tensor,
                  clean_prediction: Tensor, *, eta: float = 0, noise: Tensor | None = None
                  ) -> Tensor:
        if not 0 <= eta <= 1:
            raise ValueError("DDIM eta must lie in [0,1]")
        a = self._alpha(timesteps, state)
        ap = self._alpha(previous, state)
        if bool((previous >= timesteps).any()):
            raise ValueError("DDIM requires strictly decreasing reverse timesteps")
        if clean_prediction.shape != state.shape:
            raise ValueError("prediction shape mismatch")
        epsilon = (state - a.sqrt()*clean_prediction) / (1-a).sqrt()
        variance = ((1-ap)/(1-a) * (1-a/ap)).clamp_min(0)
        sigma = eta * variance.sqrt()
        result = ap.sqrt()*clean_prediction + (1-ap-sigma.square()).clamp_min(0).sqrt()*epsilon
        if eta:
            if noise is None or noise.shape != state.shape:
                raise ValueError("stochastic DDIM requires explicitly supplied matching noise")
            result = result + sigma * noise
        return result

    @torch.no_grad()
    def sample(self, model: nn.Module, lengths: Tensor, memory: Tensor, text_valid: Tensor,
               condition: Tensor, *, latent_width: int, sampling_steps: int = 50,
               generator: torch.Generator | None = None, eta: float = 0,
               initial_noise: Tensor | None = None) -> Tensor:
        if model.training:
            raise RuntimeError("diffusion sampling requires eval()")
        if not 1 <= sampling_steps <= self.steps:
            raise ValueError("sampling_steps must be between one and training diffusion steps")
        if condition.shape[0] != len(lengths) or condition.shape[1] != int(lengths.max()):
            raise ValueError("condition must have the predicted-length canvas")
        shape = (len(lengths), int(lengths.max()), 4, latent_width)
        state = (torch.randn(shape, device=memory.device, dtype=memory.dtype, generator=generator)
                 if initial_noise is None else initial_noise.clone())
        if state.shape != shape:
            raise ValueError("initial noise shape mismatch")
        state = zero_padding(state, lengths)
        grid = torch.linspace(self.steps, 1, sampling_steps, device=state.device).round().long()
        previous_grid = torch.cat((grid[1:], grid.new_zeros(1)))
        for t, previous in zip(grid.tolist(), previous_grid.tolist()):
            ts = torch.full((len(lengths),), t, device=state.device, dtype=torch.long)
            ps = torch.full_like(ts, previous)
            prediction = model(state, ts.to(state)/self.steps, memory, text_valid,
                               condition, lengths)
            noise = (torch.randn(shape, device=state.device, dtype=state.dtype,
                                  generator=generator) if eta else None)
            state = zero_padding(self.ddim_step(state, ts, ps, prediction, eta=eta, noise=noise),
                                 lengths)
        return state
