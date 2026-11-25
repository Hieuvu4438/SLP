"""Jacobi damped least-squares IK and reproducible conditional sources."""

import math

import torch

from .data import digest
from .relations import GROUPS, describe, ik_weights
from .rotations import canon, multiply, normalize, physical_angle, so3_exp, so3_log_short

ACTIVE = (4, 5, 6, 7, 8, 9, *(j for j in range(10, 40) if (j - 10) % 3 != 2))
ARMS = (
    "A-paper",
    "A-matched",
    "A-zero",
    "B",
    "C",
    "D",
    "E-rest",
    "E-shuffle",
    "F-oracle-dev",
    "F-both",
    "F-oracle-length",
)
UNIFORM = ("A-paper", "A-matched", "A-zero", "B")


def generator(device, seed, corpus, clip_id, stream, *draw):
    key = int(digest([seed, corpus, clip_id, stream, *draw])[:16], 16)
    return torch.Generator(device=device).manual_seed(key)


def choose_rest(clips, seed, limit):
    valid_indices = [c.q_valid.all(-1).nonzero().flatten() for c in clips]
    counts = torch.tensor([len(ix) for ix in valid_indices]).cumsum(0)
    if not counts[-1]:
        raise ValueError("Rest pose requires at least one fully valid training frame")
    chosen = torch.randperm(int(counts[-1]), generator=torch.Generator().manual_seed(seed))[:limit]
    clip_indexes = torch.searchsorted(counts, chosen, right=True)
    offsets = torch.cat((counts.new_zeros(1), counts[:-1]))
    sample = [
        (clip, int(valid_indices[clip][index - offsets[clip]]))
        for clip, index in zip(clip_indexes.tolist(), chosen.tolist())
    ]
    poses = torch.stack([clips[clip].q[i] for clip, i in sample])
    moment = torch.einsum("njc,njd->jcd", poses.double(), poses.double())
    mean = torch.linalg.eigh(moment).eigenvectors[..., -1].to(poses)
    i = physical_angle(poses, mean).square().mean(-1).argmin().item()
    clip, frame = sample[i]
    return poses[i], (clips[clip].id, frame)


def solve_ik(r, rest, settings, fk, scale, weights, valid=None, ablate=()):
    """Return sparse pose, per-sweep energy and relation residual, and solver counters."""
    index = torch.tensor(ACTIVE, device=r.device)
    rest = rest.to(r)
    initial = rest.expand(len(r), -1, -1).clone()
    w = ik_weights(weights, scale.to(r), ablate)
    if valid is not None:
        w = w * valid.to(r).sqrt()
    else:
        w = w.expand_as(r)
    dof = len(ACTIVE) * 3
    reg = math.sqrt(settings["rest_weight"] / dof)
    smooth = math.sqrt(settings["temporal_weight"] / dof)
    eye = torch.eye(dof, device=r.device, dtype=r.dtype)
    zero = r.new_zeros(len(ACTIVE), 3)

    with torch.no_grad():
        rest_points = fk(rest[None])[0]

    def relations(q, target, weight):
        predicted, _ = describe(fk(q), rest_points=rest_points)
        return (predicted - target) * weight

    def energy(q):
        moving = q[:, index]
        rel = relations(q, r, w).square().sum()
        anchor = so3_log_short(rest[index], moving).square().sum() * reg**2
        temporal = so3_log_short(moving[:-1], moving[1:]).square().sum() * smooth**2
        return rel + anchor + temporal, rel

    def update(q, delta):
        return q.index_copy(-2, index, normalize(multiply(q[..., index, :], so3_exp(delta))))

    status = {"damping_retries": 0, "fallbacks": 0, "accepted": []}
    q = initial
    current, relation = energy(q)
    history = [(float(current), float(relation))]
    # torch.no_grad() is appropriate for the outer sampler, but IK needs local AD.
    for _ in range(settings["sweeps"]):
        steps = []
        for k in range(len(q)):

            def residual(delta):
                moved = update(q[k], delta)
                pose = moved[index]
                parts = [
                    relations(moved[None], r[k : k + 1], w[k : k + 1]).flatten(),
                    (reg * so3_log_short(rest[index], pose)).flatten(),
                ]
                for adjacent in (k - 1, k + 1):
                    if 0 <= adjacent < len(q):
                        parts.append((smooth * so3_log_short(q[adjacent, index], pose)).flatten())
                return torch.cat(parts)

            with torch.enable_grad():
                b = residual(zero)
                j = torch.autograd.functional.jacobian(residual, zero, vectorize=True)
            j = j.flatten(1)
            lhs, rhs = j.T @ j, -(j.T @ b.detach())
            step = zero
            for mult in (1, 10):
                if mult == 10:
                    status["damping_retries"] += 1
                delta, info = torch.linalg.solve_ex(
                    lhs + (settings["damping"] * mult) ** 2 * eye, rhs
                )
                if int(info) == 0 and delta.isfinite().all():
                    step = delta.reshape_as(zero)
                    break
            else:
                status["fallbacks"] += 1
            cap = settings["step_cap_rad_per_joint"]
            step = step * (cap / step.norm(dim=-1, keepdim=True).clamp_min(cap))
            steps.append(step)
        accepted = 0.0
        for factor in settings["line_search"]:
            candidate = update(q, factor * torch.stack(steps))
            proposal, rel = energy(candidate)
            if proposal.isfinite() and proposal <= current:
                q, current, relation, accepted = candidate.detach(), proposal, rel, factor
                break
        status["accepted"].append(accepted)
        history.append((float(current), float(relation)))
    final_r, _ = describe(fk(q), rest_points=rest_points)
    status["group_rmse"] = {
        name: float((final_r[..., ids] - r[..., ids]).square().mean().sqrt())
        for name, ids in GROUPS.items()
    }
    return canon(q.detach()), history, status


def sample_source(arm, center, length, config, req, corpus, *, device, dtype, draw=()):
    def noise(shape, stream):
        rng = generator(device, req.seed, corpus, req.sample_id, stream, *draw)
        return torch.randn(shape, device=device, dtype=dtype, generator=rng)

    if arm in UNIFORM:
        q = normalize(noise((length, 41, 4), "pose"))
    else:
        if center is None or center.shape != (length, 41, 4):
            raise ValueError(f"{arm} requires a center on the requested grid")
        std = config["std_radians"]
        sigma = center.new_tensor(
            [std["upper"]] * 8 + [std["wrist"]] * 2 + [std["finger"]] * 30 + [std["jaw"]]
        )
        alpha = config["ar1_alpha"]
        pair = math.sqrt(1 - alpha**2)
        z = noise((length, 41, 3), "pose")
        state = [z[0]]
        for n in z[1:]:
            state.append(alpha * state[-1] + pair * n)
        q = multiply(
            center, so3_exp(torch.stack(state) * sigma[:, None] * config.get("std_multiplier", 1.0))
        )
    return canon(normalize(q)), noise((length, 10), "expression")


def donor_for(bank, request, predicted_length):
    candidates = [
        (abs(row["length"] - predicted_length), key, row["center"])
        for key, row in bank.items()
        if key != request.sample_id
        and row["sign"] == request.target_sign_language
        and row["text_hash"] != digest(request.text)
    ]
    if not candidates:
        raise ValueError(f"No language-matched donor for {request.sample_id}")
    _, key, center = min(candidates, key=lambda row: (row[0], row[1]))
    return key, center
