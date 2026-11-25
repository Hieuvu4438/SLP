"""Compact latent (MoRAE Sec. 3.2): one attention layer compresses frozen teacher features into a
d-dimensional diagonal-Gaussian posterior per frame; a light Transformer decodes it straight back
to the 133-D pose (single-stage decoding, the paper's better variant).

``LatentSpace.kind`` selects the generative space:
  jepa  frozen Sign-JEPA teacher features -> compressor -> decoder (MoRAE)
  ae    same encoder architecture trained jointly by reconstruction, deterministic latent
  vae   as ae with a sampled posterior and KL (paper's reconstruction controls, Table 3)
  raw   the normalised 133-D pose itself (uncompressed control); decoder = identity
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import PART_SLICES, POSE_DIM, valid_frames, with_velocity
from .jepa import FrameEncoder, SignJEPA, frame_features
from .layers import Attention, Transformer, init_weights, masked_mean

KINDS = ("jepa", "ae", "vae", "raw")


class Compressor(nn.Module):
    """h = LN(H + MHA(LN(H))); (mean, logvar) = Linear(h). One attention layer, as in FAE/MoRAE."""

    def __init__(self, in_dim: int, latent_dim: int, attn_width: int, heads: int,
                 rope: bool = True):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.attn = Attention(in_dim, heads, inner=attn_width, rope=rope)
        self.norm_out = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, 2 * latent_dim)

    def forward(self, features: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        h = self.norm_out(features + self.attn(self.norm_in(features), valid))
        mean, logvar = self.head(h).float().chunk(2, dim=-1)
        return mean, logvar.clamp(-30.0, 20.0)


class MotionDecoder(nn.Module):
    def __init__(self, latent_dim: int, width: int, depth: int, heads: int,
                 out_dim: int = POSE_DIM, mlp_ratio: float = 4.0):
        super().__init__()
        self.inp = nn.Linear(latent_dim, width)
        self.transformer = Transformer(width, depth, heads, mlp_ratio, rope=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, z: Tensor, valid: Tensor) -> Tensor:
        return self.head(self.norm(self.transformer(self.inp(z), valid)[-1]))


class LatentSpace(nn.Module):
    """Encoder (frozen teacher or jointly trained) + compressor + decoder + flow normalisation."""

    def __init__(self, kind: str, latent_dim: int, encoder: dict, compressor: dict,
                 decoder: dict):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"latent.kind must be one of {KINDS}")
        self.kind = kind
        if kind == "raw":
            self.latent_dim = POSE_DIM
        else:
            self.latent_dim = latent_dim
            self.input_velocity = encoder["input_velocity"]
            self.feature_layers = encoder["feature_layers"]
            in_dim = POSE_DIM * (2 if self.input_velocity else 1)
            self.encoder = FrameEncoder(in_dim, encoder["width"], encoder["depth"],
                                        encoder["heads"], encoder["mlp_ratio"])
            self.compressor = Compressor(encoder["width"], latent_dim, compressor["attn_width"],
                                         compressor["heads"], compressor["rope"])
            self.decoder = MotionDecoder(latent_dim, decoder["width"], decoder["depth"],
                                         decoder["heads"])
            for module in (self.encoder, self.compressor, self.decoder):
                init_weights(module)
            if kind == "jepa":
                self.encoder.requires_grad_(False)
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_std", torch.ones(self.latent_dim))

    @property
    def trains_encoder(self) -> bool:
        return self.kind in ("ae", "vae")

    @property
    def stochastic(self) -> bool:
        return self.kind in ("jepa", "vae")

    def load_teacher(self, jepa: SignJEPA) -> None:
        if self.kind != "jepa":
            raise ValueError("Only latent.kind=jepa uses a Sign-JEPA teacher")
        if (jepa.input_velocity != self.input_velocity
                or jepa.feature_layers != self.feature_layers):
            raise ValueError("Sign-JEPA input or feature layers differ from the latent config")
        self.encoder.load_state_dict(jepa.teacher.state_dict())
        self.encoder.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.kind == "raw":
            return []
        modules = [self.compressor, self.decoder] + ([self.encoder] if self.trains_encoder else [])
        return [p for m in modules for p in m.parameters()]

    def features(self, poses: Tensor, lengths: Tensor) -> Tensor:
        valid = valid_frames(lengths, poses.shape[1])
        u = (with_velocity(poses, lengths) if self.input_velocity
             else poses * valid.unsqueeze(-1))
        if not self.trains_encoder:
            self.encoder.eval()
        with torch.set_grad_enabled(self.trains_encoder and torch.is_grad_enabled()):
            _, hidden = self.encoder(u, valid)
            return frame_features(hidden, self.feature_layers)

    def posterior(self, poses: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """(mean, logvar) [B, T, d] in un-normalised latent coordinates."""
        if self.kind == "raw":
            mean = poses.float() * valid_frames(lengths, poses.shape[1]).unsqueeze(-1)
            return mean, torch.full_like(mean, -30.0)
        valid = valid_frames(lengths, poses.shape[1])
        return self.compressor(self.features(poses, lengths), valid)

    def decode(self, z: Tensor, lengths: Tensor) -> Tensor:
        valid = valid_frames(lengths, z.shape[1]).unsqueeze(-1)
        if self.kind == "raw":
            return z.float() * valid
        return self.decoder(z, valid.squeeze(-1)).float() * valid

    # --- flow coordinates: per-channel standardisation with train statistics (paper Sec. 3.3)
    def to_flow(self, mean: Tensor) -> Tensor:
        return (mean.float() - self.latent_mean) / self.latent_std

    def from_flow(self, q: Tensor) -> Tensor:
        return q.float() * self.latent_std + self.latent_mean

    def encode_flow(self, poses: Tensor, lengths: Tensor) -> Tensor:
        mean, _ = self.posterior(poses, lengths)
        valid = valid_frames(lengths, poses.shape[1]).unsqueeze(-1)
        return self.to_flow(mean) * valid

    def decode_flow(self, q: Tensor, lengths: Tensor) -> Tensor:
        return self.decode(self.from_flow(q), lengths)


def part_mse(predicted: Tensor, target: Tensor, valid: Tensor, weights: dict[str, float]
             ) -> tuple[Tensor, dict[str, Tensor]]:
    """Part-weighted MSE. Each part's mean squared error is weighted by w_p * (its channel count),
    so equal weights give exactly the plain MSE over all 133 channels (paper Eq. 2)."""
    parts, sizes = {}, {}
    for name, sl in PART_SLICES.items():
        parts[name] = masked_mean((predicted[..., sl] - target[..., sl]).square(), valid)
        sizes[name] = sl.stop - sl.start
    norm = sum(weights[n] * sizes[n] for n in parts)
    total = sum(weights[n] * sizes[n] * parts[n] for n in parts) / norm
    return total, parts


def evenly_spaced_frames(lengths: Tensor, samples: int) -> Tensor:
    """[B, samples] frame indices at interval midpoints of each clip."""
    grid = 2 * torch.arange(samples, device=lengths.device) + 1
    return grid[None] * lengths[:, None] // (2 * samples)


class PoseFKLoss(nn.Module):
    """SOKE-protocol joint positions (cm) through the exact differentiable SMPL-X rig:
    12 upper-body points relative to the pelvis, 2 x 21 hand points relative to their wrist."""

    def __init__(self, stats=None):
        super().__init__()
        from slp.skeleton_fk import SkeletonFK

        self.fk = SkeletonFK(stats=stats).requires_grad_(False)

    def points(self, features: Tensor) -> tuple[Tensor, Tensor]:
        out = self.fk.from_features(features.float())
        body = out["body"] - out["pelvis"].unsqueeze(-2)
        hands = torch.cat([out["lhand"] - out["lhand"][..., :1, :],
                           out["rhand"] - out["rhand"][..., :1, :]], dim=-2)
        return 100 * body, 100 * hands

    def forward(self, predicted: Tensor, target: Tensor, lengths: Tensor, samples: int
                ) -> tuple[Tensor, Tensor]:
        samples = max(1, min(samples, predicted.shape[1]))
        frames = evenly_spaced_frames(lengths, samples)
        items = torch.arange(len(predicted), device=predicted.device)[:, None]
        with torch.autocast(predicted.device.type, enabled=False):
            pb, ph = self.points(predicted[items, frames])
            with torch.no_grad():
                tb, th = self.points(target[items, frames])
            return F.smooth_l1_loss(pb, tb), F.smooth_l1_loss(ph, th)


def autoencoder_losses(space: LatentSpace, poses: Tensor, lengths: Tensor, cfg: dict,
                       fk: PoseFKLoss | None = None) -> dict[str, Tensor]:
    """L = sum_p w_p MSE_p + beta KL (+ velocity and FK terms when enabled), paper Eq. (2)."""
    valid = valid_frames(lengths, poses.shape[1])
    mean, logvar = space.posterior(poses, lengths)
    if space.stochastic and space.training:
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
    else:
        z = mean
    recon = space.decode(z, lengths)
    target = poses.float()
    weights = dict(zip(PART_SLICES, cfg["part_weights"], strict=True))
    rec, parts = part_mse(recon, target, valid, weights)
    mse = masked_mean((recon - target).square(), valid)
    loss = rec
    out = {"mse": mse.detach(), **{f"mse_{k}": v.detach() for k, v in parts.items()}}
    if space.stochastic:
        kl = 0.5 * (mean.square() + logvar.exp() - logvar - 1).sum(-1)
        kl = masked_mean(kl, valid)
        loss = loss + cfg["kl_weight"] * kl
        out["kl"] = kl.detach()
    if cfg.get("velocity_weight", 0.0):
        v_pred, v_true = recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1]
        vel = masked_mean((v_pred - v_true).square(), valid[:, 1:])
        loss = loss + cfg["velocity_weight"] * vel
        out["velocity"] = vel.detach()
    if fk is not None and cfg.get("fk_weight", 0.0):
        body, hand = fk(recon, target, lengths, cfg["fk_frames"])
        loss = loss + cfg["fk_weight"] * (body + hand)
        out["fk_body"], out["fk_hand"] = body.detach(), hand.detach()
    out["loss"] = loss
    return out
