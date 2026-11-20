"""Target/velocity training and checkpoint-compatible path re-planning."""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from . import geometry as G
from .coarse import coarse_targets
from .paths import PATHS, path_and_velocity, path_point


@dataclass(frozen=True)
class FlowSpec:
    representation: str = "quat"
    path: str = "naturalness"
    prediction: str = "target"
    w_pose: float = 10.
    w_expr: float = 0.1
    loss_reduction: str = "sum"
    endpoint_loss: str = "lift"
    input_rotation: str = "raw"
    source_ar: float = 0.
    coupling: str = "independent"
    position_weight: float = 0.
    velocity_weight: float = 0.
    aux_min_t: float = 0.
    huber_delta: float = 1.
    w_jaw: float = 0.

    def __post_init__(self):
        if self.coupling not in {"independent", "matched_ot"}:
            raise ValueError("Unknown source coupling strategy")
        if self.representation not in {"quat", "6d"} or self.path not in PATHS:
            raise ValueError("Unknown rotation representation or path")
        if self.prediction not in {"target", "velocity"} or self.loss_reduction not in {"sum", "mean"}:
            raise ValueError("Unknown prediction or loss reduction")
        if self.endpoint_loss not in {"lift", "canonical", "chordal"} or self.input_rotation not in {"raw", "canonical", "6d"}:
            raise ValueError("Unknown endpoint loss or input rotation policy")
        if self.representation == "6d" and (self.path != "linear" or self.endpoint_loss != "lift" or self.input_rotation != "raw"):
            raise ValueError("Euclidean 6D uses linear paths, lift MSE and raw input")
        if self.input_rotation == "6d" and self.path == "linear":
            raise ValueError("Sign-invariant quaternion features require an on-manifold path")
        if self.prediction == "velocity" and (self.position_weight or self.velocity_weight or self.w_jaw
                                             or self.endpoint_loss != "lift" or self.path == "projected_6d"):
            raise ValueError("These auxiliary losses or projected 6D paths require endpoint prediction")
        weights = (self.w_pose, self.w_expr, self.position_weight, self.velocity_weight, self.w_jaw)
        if any(not 0 <= w < math.inf for w in weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        if not -1 < self.source_ar < 1 or not 0 <= self.aux_min_t < 1 or not 0 < self.huber_delta < math.inf:
            raise ValueError("Invalid source correlation, auxiliary time gate, or Huber delta")

    @property
    def uses_fk(self):
        return bool(self.position_weight or self.velocity_weight)

    @property
    def manifold(self):
        return self.path != "linear"

    @property
    def rot_dim(self):
        return 4 if self.representation == "quat" else 6

    def model_input(self, x):
        if self.input_rotation == "canonical":
            return G.sign_convention(x)
        if self.input_rotation == "6d":
            return G.matrix_to_6d(G.quat_to_matrix(x))
        return x


def ar1_noise(innovation, correlation):
    if correlation == 0:
        return innovation
    multiplier = math.sqrt(1 - correlation ** 2)
    z = torch.cat((innovation[:, :1], innovation[:, 1:] * multiplier), dim=1)
    # Parallel prefix of the AR recurrence. Doubling the lag accumulates
    # the contribution of all preceding innovations in logarithmic passes.
    lag, coefficient = 1, correlation
    while lag < innovation.shape[1]:
        z = torch.cat((z[:, :lag], z[:, lag:] + coefficient * z[:, :-lag]), dim=1)
        lag *= 2
        coefficient *= coefficient
    return z


def source_from_innovations(spec, z):
    z = ar1_noise(z, spec.source_ar)
    return G.sign_convention(G.normalize(z)) if spec.representation == "quat" else z


def sample_source(spec, shape, device, generator=None):
    # Shape B,T,J. Correlation acts in physical motion time, before normalization.
    z = torch.randn(*shape, spec.rot_dim, device=device, generator=generator)
    return source_from_innovations(spec, z)


def seeded_generator(seed, key, stream, device):
    digest = hashlib.blake2b(f"{seed}\0{key}\0{stream}".encode(), digest_size=8).digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest, "little"))


def per_clip_sources(spec, keys, seed, frames, max_frames, device, n_joints=41, expr_dim=10):
    """Noise is invariant to batch order, time-grid changes and prefix truncation."""
    if not 0 < frames <= max_frames:
        raise ValueError("Invalid requested or canonical noise length")
    pose, expr = [], []
    for key in keys:
        pose_rng = seeded_generator(seed, key, "pose", device)
        expr_rng = seeded_generator(seed, key, "expr", device)
        pose.append(torch.randn(max_frames, n_joints, spec.rot_dim, device=device, generator=pose_rng)[:frames])
        expr.append(torch.randn(max_frames, expr_dim, device=device, generator=expr_rng)[:frames])
    return source_from_innovations(spec, torch.stack(pose)), torch.stack(expr)


def masked_mean(values, mask, denominator):
    return torch.where(mask, values, 0.).sum() / max(1, denominator)


def motion_residual(xhat, target, spec):
    if spec.endpoint_loss == "chordal":
        dot = (xhat * target).sum(-1).clamp(-1, 1)
        error = 4 * (1 - dot.square())
        return error.sum(-1) if spec.loss_reduction == "sum" else error.mean(-1)
    if spec.endpoint_loss == "canonical":
        xhat = G.sign_convention(xhat)
    error = (xhat - target).square().flatten(2)
    return error.sum(-1) if spec.loss_reduction == "sum" else error.mean(-1)


def auxiliary_losses(fk, pose, target, valid, times, inverse_dt, spec, counts, weights):
    """Huber position and signed finite-difference errors with logical-batch counts."""
    predict_matrix = G.quat_to_matrix if spec.representation == "quat" else G.sixd_to_matrix
    mask = valid & (times[:, None] >= spec.aux_min_t)
    # Compute FK only for valid frames used by this objective.
    features = fk(predict_matrix(pose[mask]))
    with torch.no_grad():
        reference = fk(predict_matrix(target[mask]))
    delta = features - reference
    pos = pose.new_zeros(())
    vel = pose.new_zeros(())
    if spec.position_weight:
        residual = F.huber_loss(delta, torch.zeros_like(delta), delta=spec.huber_delta, reduction="none")
        pos = (residual * weights).sum() / max(1, counts["frames"])
    if spec.velocity_weight:
        dense = delta.new_zeros(*mask.shape, delta.shape[-1])
        dense[mask] = delta
        pair_mask = mask[:, 1:] & mask[:, :-1] & (inverse_dt > 0)
        speed = (dense[:, 1:] - dense[:, :-1]) * inverse_dt[..., None]
        residual = F.huber_loss(speed, torch.zeros_like(speed), delta=spec.huber_delta, reduction="none")
        vel = masked_mean((residual * weights).sum(-1), pair_mask, counts["pairs"])
    return pos, vel


def flow_loss(gen, correction, spec, target, expr, mask, texts, sources, times, initial_pose,
              initial_expr, inverse_dt, counts, fk=None, weights=None, length_weight=1., coarse_weight=0.):
    t = times[:, None, None, None]
    with torch.no_grad():
        if spec.prediction == "velocity":
            x, target_v = path_and_velocity(spec.path, initial_pose, target, t, correction)
        else:
            x = path_point(spec.path, initial_pose, target, t, correction)
        if spec.manifold:
            x = G.normalize(x)
    e = torch.lerp(initial_expr, expr, times[:, None, None])
    text, text_pad = gen.encode_text(texts, sources, e.device)
    coarse_loss = e.new_zeros(())
    if gen.coarse is not None:
        plan = gen.coarse(text, text_pad, mask)
        coarse_context = gen.coarse.dense(plan, mask)
        values, bins = coarse_targets(target, expr, mask, gen.coarse.stride)
        coarse_loss = masked_mean((plan - values).square().mean(-1), bins, counts["bins"])
        pose, facial = gen(spec.model_input(x), e, times, text, text_pad, ~mask, coarse_context)
    else:
        pose, facial = gen(spec.model_input(x), e, times, text, text_pad, ~mask)
    n = counts["frames"]
    pos = vel = jaw = pose.new_zeros(())
    if spec.prediction == "target":
        if spec.representation == "quat":
            pose = G.normalize(pose)
        pose_loss = masked_mean(motion_residual(pose, target, spec), mask, n)
        expr_error = facial - expr
        if spec.uses_fk:
            pos, vel = auxiliary_losses(fk, pose, target, mask, times, inverse_dt, spec, counts, weights)
        if spec.w_jaw:
            if spec.representation == "quat":
                cosine = (pose[..., 40, :] * target[..., 40, :]).sum(-1).clamp(-1, 1)
                jaw_error = 4 * (1 - cosine.square())
            else:
                jaw_error = (pose[..., 40, :] - target[..., 40, :]).square().sum(-1)
            jaw = masked_mean(jaw_error, mask, n)
    else:
        if spec.manifold:
            pose = G.project(x, pose)
        error = (pose - target_v).square().flatten(2)
        per_frame = error.sum(-1) if spec.loss_reduction == "sum" else error.mean(-1)
        pose_loss = masked_mean(per_frame, mask, n)
        expr_error = facial - (expr - initial_expr)
    expr_error = expr_error.square()
    expr_per_frame = expr_error.sum(-1) if spec.loss_reduction == "sum" else expr_error.mean(-1)
    expr_loss = masked_mean(expr_per_frame, mask, n)
    length_values = gen.length_errors(text, text_pad, mask.sum(1))
    length_loss = length_values.sum() / counts["clips"]
    total = (spec.w_pose * pose_loss + spec.w_expr * expr_loss + spec.w_jaw * jaw
             + spec.position_weight * pos + spec.velocity_weight * vel + length_weight * length_loss
             + coarse_weight * coarse_loss)
    parts = {"pose": pose_loss, "expression": expr_loss, "jaw": jaw, "position": pos, "velocity": vel,
             "length": length_loss, "coarse": coarse_loss}
    return total, parts


def sampling_grid(steps, explicit=None):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("Sampler steps must be a positive integer")
    grid = [i / steps for i in range(steps + 1)] if explicit is None else list(explicit)
    if len(grid) != steps + 1 or grid[0] != 0 or grid[-1] != 1 or any(
            not 0 <= a < b <= 1 for a, b in zip(grid, grid[1:])):
        raise ValueError("time_grid must strictly increase from 0 to 1 and have steps+1 values")
    return grid


@torch.no_grad()
def sample(gen, correction, spec, text, text_pad, mask, initial_pose, initial_expr, grid,
           expression_update="current", return_diagnostics=False):
    if expression_update not in {"current", "anchored"}:
        raise ValueError(f"Unknown expression update {expression_update!r}")
    if expression_update == "anchored" and spec.prediction != "target":
        raise ValueError("Anchored expression requires target prediction")
    x, e = initial_pose, initial_expr
    last_target = None
    drift = []
    coarse = getattr(gen, "coarse", None)
    coarse_context = coarse.dense(coarse(text, text_pad, mask), mask) if coarse is not None else None
    for t, s in zip(grid, grid[1:]):
        times = x.new_full((len(x),), t)
        arguments = (spec.model_input(x), e, times, text, text_pad, ~mask)
        raw_pose, expr = gen(*arguments) if coarse_context is None else gen(*arguments, coarse_context)
        if spec.prediction == "target":
            target = G.normalize(raw_pose) if spec.representation == "quat" else raw_pose
            if spec.endpoint_loss == "canonical":
                target = G.sign_convention(target)
            if spec.path == "linear":
                x = torch.lerp(x, target, (s - t) / (1 - t))
            else:
                x = path_point(spec.path, initial_pose, target, x.new_full((len(x), 1, 1, 1), s), correction)
                x = G.normalize(x)
            if expression_update == "anchored":
                e = torch.lerp(initial_expr, expr, s)
            else:
                e = torch.lerp(e, expr, (s - t) / (1 - t))
            if return_diagnostics and last_target is not None:
                qa, qb = last_target, target
                if spec.representation == "6d":
                    qa, qb = (G.matrix_to_quat(G.sixd_to_matrix(y)) for y in (qa, qb))
                angle = G.relative_rotvec(qa, qb).norm(dim=-1).mean(-1)
                clip_drift = torch.where(mask, angle, 0.).sum(-1) / mask.sum(-1).clamp_min(1)
                drift.append(clip_drift.rad2deg())
            last_target = target
        else:
            step = s - t
            x = G.normalize(G.expmap(x, step * G.project(x, raw_pose))) if spec.manifold else x + step * raw_pose
            e = e + step * expr
    if spec.representation == "quat":
        x = G.normalize(x)
    return x, e, drift
