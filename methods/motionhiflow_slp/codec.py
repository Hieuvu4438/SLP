"""Temporal VAE with separate body, left hand, right hand, and facial channels."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import PART_DIMS, ceil_div, valid_frames
from .pyramid import resample

PARTS = len(PART_DIMS)


class TemporalBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        channels = PARTS * width
        self.norm = nn.LayerNorm(channels)
        self.temporal = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=PARTS)
        self.mix = nn.Linear(channels, channels)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        flat = x.flatten(start_dim=2)
        features = self.norm(flat).masked_fill(~mask[..., None], 0)
        features = self.temporal(features.transpose(1, 2)).transpose(1, 2)
        features = flat + self.mix(F.gelu(features))
        return features.reshape_as(x).masked_fill(~mask[..., None, None], 0)


class MotionVAE(nn.Module):
    def __init__(self, width: int, latent_dim: int, depth: int, stride: int):
        super().__init__()
        if min(width, latent_dim, depth, stride) < 1:
            raise ValueError("Codec width, latent dimensions, depth and stride must be positive")
        self.stride = stride
        self.latent_dim = latent_dim
        self.inputs = nn.ModuleList(nn.Linear(dim, width) for dim in PART_DIMS)
        self.encoder = nn.ModuleList(TemporalBlock(width) for _ in range(depth))
        self.down = nn.Linear(width * stride, width)
        self.posterior = nn.Linear(width, latent_dim * 2)
        self.up = nn.Linear(latent_dim, width * stride)
        self.decoder = nn.ModuleList(TemporalBlock(width) for _ in range(depth))
        self.outputs = nn.ModuleList(nn.Linear(width, dim) for dim in PART_DIMS)
        self.register_buffer("latent_mean", torch.zeros(PARTS, latent_dim))
        self.register_buffer("latent_std", torch.ones(PARTS, latent_dim))

    def encode(self, poses: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        frames = poses.shape[1]
        mask = valid_frames(lengths, frames)
        pose_parts = poses.masked_fill(~mask[..., None], 0).split(PART_DIMS, dim=-1)
        x = torch.stack([layer(part) for layer, part in zip(self.inputs, pose_parts)], dim=2)
        x = x.masked_fill(~mask[..., None, None], 0)
        for block in self.encoder:
            x = block(x, mask)

        batch, _, parts, width = x.shape
        x = F.pad(x, (0, 0, 0, 0, 0, (-frames) % self.stride))
        x = x.reshape(batch, -1, self.stride, parts, width).transpose(2, 3).flatten(3)
        latent_lengths = ceil_div(lengths, self.stride)
        mask = valid_frames(latent_lengths, x.shape[1])[..., None, None]
        mean, logvar = self.posterior(F.gelu(self.down(x))).float().chunk(2, dim=-1)
        return mean.masked_fill(~mask, 0), logvar.clamp(-12, 8).masked_fill(~mask, 0), latent_lengths

    def decode(self, latent: Tensor, lengths: Tensor, size: int) -> Tensor:
        batch, tokens, parts, _ = latent.shape
        latent_mask = valid_frames(ceil_div(lengths, self.stride), tokens)[..., None, None]
        x = self.up(latent.masked_fill(~latent_mask, 0))
        x = x.reshape(batch, tokens, parts, self.stride, -1).transpose(2, 3)
        x = x.reshape(batch, tokens * self.stride, parts, -1)[:, :size]
        mask = valid_frames(lengths, size)
        x = x.masked_fill(~mask[..., None, None], 0)
        for block in self.decoder:
            x = block(x, mask)
        y = torch.cat([proj(x[:, :, i]) for i, proj in enumerate(self.outputs)], dim=-1)
        return y.float().masked_fill(~mask[..., None], 0)

    def flow_target(self, poses: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        mean, _, latent_lengths = self.encode(poses, lengths)
        mask = valid_frames(latent_lengths, mean.shape[1])[..., None, None]
        return ((mean - self.latent_mean) / self.latent_std).masked_fill(~mask, 0), latent_lengths

    def decode_flow(self, latent: Tensor, lengths: Tensor, size: int) -> Tensor:
        return self.decode(latent * self.latent_std + self.latent_mean, lengths, size)


def sequence_mean(values: Tensor, lengths: Tensor) -> Tensor:
    """Average over valid time positions first, then give every sequence equal weight."""
    mask = valid_frames(lengths, values.shape[1])
    per_frame = values.flatten(start_dim=2).mean(-1) if values.ndim > 2 else values
    per_clip = per_frame.masked_fill(~mask, 0).sum(1) / lengths.clamp_min(1)
    return per_clip.mean()


def reconstruction_error(
    predicted: Tensor, target: Tensor, lengths: Tensor, weights: Tensor
) -> Tensor:
    errors = (predicted.float() - target).abs().split(PART_DIMS, dim=-1)
    part_errors = torch.stack([error.mean(-1) for error in errors], dim=-1)
    return sequence_mean(part_errors @ (weights / weights.sum()), lengths)


def codec_losses(model: MotionVAE, poses: Tensor, lengths: Tensor, cfg: dict) -> dict[str, Tensor]:
    mean, logvar, latent_lengths = model.encode(poses, lengths)
    latent = mean
    if model.training:
        latent = mean + (0.5 * logvar).exp() * torch.randn_like(mean)
    reconstruction = model.decode(latent, lengths, poses.shape[1])
    weights = poses.new_tensor(cfg["part_weights"])
    rec = reconstruction_error(reconstruction, poses, lengths, weights)
    vel = reconstruction_error(reconstruction.diff(dim=1), poses.diff(dim=1),
                               lengths - 1, weights)
    kl = sequence_mean(0.5 * (mean.square() + logvar.exp() - logvar - 1), latent_lengths)
    pool = poses.new_zeros(())
    if cfg["pool_weight"] and model.training:
        scale = torch.empty((), device="cpu").uniform_(cfg["pool_min_scale"], 1).item()
        coarse_lengths = (latent_lengths * scale).floor().long().clamp_min(1)
        coarse_frames = (coarse_lengths * lengths + latent_lengths - 1) // latent_lengths
        coarse = resample(latent, latent_lengths, coarse_lengths)
        target = resample(poses, lengths, coarse_frames)
        decoded = model.decode(coarse, coarse_frames, target.shape[1])
        pool = reconstruction_error(decoded, target, coarse_frames, weights)
    loss = rec + cfg["velocity_weight"] * vel + cfg["kl_weight"] * kl + cfg["pool_weight"] * pool
    return {"loss": loss, "reconstruction": rec, "velocity": vel, "kl": kl, "pool": pool}
