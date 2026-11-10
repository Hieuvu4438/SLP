"""Coupled expected-improvement regression; targets exist only during training."""

import torch
from torch import nn
from .losses import sequence_mask, surrogate_frames
from .rotations import rotation_6d_to_matrix, matrix_to_rotation_6d, slerp_matrix


def block_mean(x, mask, size):
    n = (x.shape[1] + size - 1) // size
    out, valid = [], []
    for i in range(n):
        a, z = i * size, min((i + 1) * size, x.shape[1])
        m = mask[:, a:z]
        weight = m.to(x.dtype)
        while weight.ndim < x[:, a:z].ndim:
            weight = weight.unsqueeze(-1)
        out.append((x[:, a:z] * weight).sum(1) / weight.sum(1).clamp_min(1))
        valid.append(m.any(1))
    return torch.stack(out, 1), torch.stack(valid, 1)


class ImprovementSelector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Base, candidate, abs difference, two 15D relations; no reference/target features.
        self.motion = nn.Sequential(
            nn.Linear(240 * 3 + 30, cfg.selector_width),
            nn.SiLU(),
            nn.Linear(cfg.selector_width, cfg.selector_width),
        )
        self.text = nn.Linear(cfg.text_dim, cfg.selector_width)
        self.language = nn.Embedding(cfg.languages, cfg.selector_width)
        self.code = nn.Linear(3 * cfg.code_dim, cfg.selector_width)
        self.whole_sentence = False
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.selector_width), nn.SiLU(), nn.Linear(cfg.selector_width, 2)
        )
        self.register_buffer("calibration_scale", torch.tensor(1.0))
        self.register_buffer("calibration_bias", torch.tensor(0.0))

    def forward(
        self,
        base,
        candidate,
        base_rel,
        candidate_rel,
        text,
        text_mask,
        lengths,
        language,
        *,
        codes,
        whole_sentence=False,
    ):
        if base.shape != candidate.shape or base.shape[-1] != 240:
            raise ValueError("Candidates must have a shared native timeline and 240 manual dims")
        valid = sequence_mask(lengths, base.shape[1])
        features = torch.cat(
            (base, candidate, (base - candidate).abs(), base_rel, candidate_rel), -1
        )
        features = features.masked_fill(~valid[..., None], 0)
        h = self.motion(features)
        code_features = self.code(codes.flatten(-2)).repeat_interleave(self.cfg.factor, dim=1)
        if code_features.shape[1] < base.shape[1]:
            raise ValueError("Codes do not cover candidate timeline")
        h = h + code_features[:, : base.shape[1]]
        text = text.masked_fill(~text_mask[..., None], 0)
        text_mean = text.sum(1) / text_mask.sum(1, keepdim=True).clamp_min(1)
        h = h + self.text(text_mean)[:, None] + self.language(language)[:, None]
        size = base.shape[1] if whole_sentence else self.cfg.block_size
        h, block_mask = block_mean(h, valid, size)
        output = self.head(h)
        raw_delta, win_logit = output.unbind(-1)
        return {
            "delta": raw_delta * self.calibration_scale + self.calibration_bias,
            "raw_delta": raw_delta,
            "win_logit": win_logit,
            "mask": block_mask,
        }

    @torch.no_grad()
    def calibrate(self, predicted, observed, mask):
        x, y = predicted[mask].double(), observed[mask].double()
        if len(x) < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Need at least two finite dev pairs for calibration")
        # Monotone affine least squares; never reverse the learned ranking.
        centered = x - x.mean()
        scale = (
            (centered * (y - y.mean())).sum() / centered.square().sum().clamp_min(1e-12)
        ).clamp_min(0)
        self.calibration_scale.copy_(scale)
        self.calibration_bias.copy_(y.mean() - scale * x.mean())


@torch.no_grad()
def improvement_targets(base, candidate, target, lengths, geometry, cfg, *, whole_sentence=False):
    valid = sequence_mask(lengths, base.shape[1])
    size = base.shape[1] if whole_sentence else cfg.block_size
    delta = surrogate_frames(base, target, geometry, cfg) - surrogate_frames(
        candidate, target, geometry, cfg
    )
    block_delta, block_mask = block_mean(delta, valid, size)
    # Local switch penalty: replacing block in a base sequence creates these boundary jumps.
    # This is explicitly a surrogate: multiple switches/blending interact, measured separately.
    if not whole_sentence and cfg.boundary_weight:
        for i in range(block_delta.shape[1]):
            start, end = i * size, min((i + 1) * size, base.shape[1])
            penalty = base.new_zeros(len(base))
            if start > 0:
                a = (candidate[:, start] - base[:, start - 1]) - (
                    base[:, start] - base[:, start - 1]
                )
                penalty += a.square().mean(-1) * (lengths > start)
            if end < base.shape[1]:
                a = (base[:, end] - candidate[:, end - 1]) - (base[:, end] - base[:, end - 1])
                penalty += a.square().mean(-1) * (lengths > end)
            block_delta[:, i] -= cfg.boundary_weight * penalty
    return block_delta, block_mask


def selector_objective(output, target):
    mask = output["mask"]
    if not mask.any():
        raise ValueError("No selector training blocks")
    regression = nn.functional.smooth_l1_loss(output["raw_delta"][mask], target[mask])
    binary = nn.functional.binary_cross_entropy_with_logits(
        output["win_logit"][mask], (target[mask] > 0).to(target.dtype)
    )
    # Calibration acts on raw regression output; classification is an auxiliary diagnostic.
    return regression + 0.1 * binary


@torch.no_grad()
def blend_candidates(base, candidate, delta, lengths, block_size, blend_frames=0):
    b, t, _ = base.shape
    if candidate.shape != base.shape or delta.shape != (b, (t + block_size - 1) // block_size):
        raise ValueError("Invalid candidate or block score dimensions")
    weight = (delta > 0).to(base.dtype).repeat_interleave(block_size, 1)[:, :t]
    # Symmetric interpolation at switches; equal choices stay exactly unchanged.
    if blend_frames:
        original = weight.clone()
        for j in range(block_size, t, block_size):
            lo, hi = max(0, j - blend_frames), min(t, j + blend_frames)
            native_hi = lengths.clamp(max=hi)
            positions = torch.arange(lo, hi, device=base.device)[None]
            alpha = (positions - lo) / (native_hi[:, None] - lo - 1).clamp_min(1)
            alpha = alpha.clamp(0, 1).to(base.dtype)
            switch = (original[:, j - 1] != original[:, j]) & (lengths > j)
            ramp = original[:, j - 1, None] * (1 - alpha) + original[:, j, None] * alpha
            active = switch[:, None] & (positions < native_hi[:, None])
            weight[:, lo:hi] = torch.where(active, ramp, weight[:, lo:hi])
    ra = rotation_6d_to_matrix(base.reshape(b, t, 40, 6))
    rb = rotation_6d_to_matrix(candidate.reshape(b, t, 40, 6))
    blended = matrix_to_rotation_6d(
        slerp_matrix(ra, rb, weight[..., None].expand(-1, -1, 40))
    ).flatten(-2)
    # Preserve exact candidates at endpoints, avoiding unnecessary round-trip drift.
    blended = torch.where(weight[..., None] == 0, base, blended)
    blended = torch.where(weight[..., None] == 1, candidate, blended)
    valid = sequence_mask(lengths, t)
    return blended.masked_fill(~valid[..., None], 0), weight.masked_fill(~valid, 0)


@torch.no_grad()
def calibration_diagnostics(selector, predicted, observed, win_logits):
    """Held-out training-surrogate diagnostics only, never final SLP benchmark scores."""
    calibrated = predicted * selector.calibration_scale + selector.calibration_bias
    order = calibrated.argsort(descending=True)
    bins = []
    for indices in torch.tensor_split(order, min(5, len(order))):
        bins.append(
            dict(
                count=len(indices),
                predicted_delta=float(calibrated[indices].mean()),
                observed_delta=float(observed[indices].mean()),
            )
        )
    coverage = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        count = max(1, round(len(order) * fraction))
        chosen = order[:count]
        coverage.append(
            dict(
                coverage=count / len(order),
                mean_observed_improvement=float(observed[chosen].mean()),
            )
        )
    return dict(
        pair_count=len(observed),
        surrogate_mae=float((calibrated - observed).abs().mean()),
        auxiliary_win_brier=float(
            (win_logits.sigmoid() - (observed > 0).to(observed.dtype)).square().mean()
        ),
        reliability_bins=bins,
        surrogate_coverage_curve=coverage,
    )
