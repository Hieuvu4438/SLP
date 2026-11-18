"""Calibrated bounded correction with immutable original reference and relation plan."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_calibration(path, hashes):
    if path is None or not Path(path).is_file():
        raise ValueError(f"Guard disabled: missing calibration artifact {path}")
    data = json.loads(Path(path).read_text())
    if data.get("approved") is not True:
        raise ValueError("Guard disabled: calibration not approved")
    for key, value in hashes.items():
        if data.get(key) != value:
            raise ValueError(f"Guard calibration hash mismatch: {key}")
    required = (
        "epsilon_sem",
        "epsilon_parts",
        "confidence_threshold",
        "annotation_split",
        "sample_counts",
        "sensitivity",
        "false_alarm",
        "threshold_selection",
    )
    if any(key not in data for key in required) or data["annotation_split"] != "dev":
        raise ValueError("Guard requires complete dev-only calibration evidence")
    if set(data["epsilon_parts"]) != {"left", "right", "face"}:
        raise ValueError("Guard requires three calibrated part budgets")
    categories = ("handshape", "location", "orientation", "NMF")
    for category in categories:
        if data["sample_counts"].get(category, 0) <= 0:
            raise ValueError(f"No confirmed dev calibration contrasts: {category}")
        sensitivity = float(data["sensitivity"].get(category, 0))
        false_alarm = float(data["false_alarm"].get(category, 1))
        if (
            not torch.isfinite(torch.tensor([sensitivity, false_alarm])).all()
            or not 0.8 <= sensitivity <= 1
            or not 0 <= false_alarm <= 0.1
        ):
            raise ValueError(f"Calibration sensitivity/false-alarm gate failed: {category}")
    budgets = [data["epsilon_sem"], data["confidence_threshold"], *data["epsilon_parts"].values()]
    if not all(torch.isfinite(torch.tensor(float(value))) for value in budgets):
        raise ValueError("Nonfinite calibration trust budgets")
    if data["epsilon_sem"] < 0 or any(value < 0 for value in data["epsilon_parts"].values()):
        raise ValueError("Trust budgets must be nonnegative")
    return data


def guarded_correction(
    z,
    relation_plan,
    decode,
    relation_fn,
    teacher_fn,
    text_embedding,
    calibration,
    valid=None,
    max_steps=2,
    max_backtracking=3,
    step_size=0.05,
    max_latent_delta_rms=0.2,
    energy_improvement_min=1e-6,
    mode="semantic",
):
    if mode not in ("semantic", "feature_only", "unrestricted"):
        raise ValueError("Unknown correction mode")
    calibration = calibration or {}
    if mode != "unrestricted" and calibration.get("approved") is not True:
        return z, {"skip_reason": "calibration_not_approved", "proposed": 0, "accepted": 0}
    original = z.detach().clone()
    plan = relation_plan.detach().clone()
    mask = torch.ones_like(z, dtype=torch.bool) if valid is None else valid.expand_as(z)

    def rms(v):
        return (v.square()[mask].mean()).sqrt()

    def energy(x):
        return F.huber_loss(relation_fn(x), plan)

    with torch.no_grad():
        x0 = decode(original)
        feature0 = teacher_fn(x0) if mode != "unrestricted" else None
        score0 = (
            F.cosine_similarity(feature0["global"].float(), text_embedding.float()).mean()
            if feature0 is not None
            else x0.new_tensor(1.0)
        )
        e0 = energy(x0)
    finite_original = bool(
        torch.isfinite(x0).all() and torch.isfinite(score0) and torch.isfinite(e0)
    )
    logs = {
        "mode": mode,
        "proposed": 0,
        "accepted": 0,
        "skip_reason": None,
        "initial_energy": float(e0) if torch.isfinite(e0) else None,
        "initial_semantic": float(score0)
        if mode != "unrestricted" and torch.isfinite(score0)
        else None,
        "forward_calls": 1,
        "backward_calls": 0,
        "fk_calls": 1,
        "rejections": [],
    }
    if not finite_original:
        logs["skip_reason"] = "nonfinite_original"
        return original, logs
    if mode == "semantic" and score0 < calibration["confidence_threshold"]:
        logs["skip_reason"] = "below_calibrated_confidence"
        return original, logs
    logs["final_energy"] = float(e0)
    logs["final_semantic"] = float(score0) if mode != "unrestricted" else None
    current = original.clone()
    for _ in range(max_steps):
        current = current.detach().requires_grad_(True)
        with torch.enable_grad():
            x = decode(current)
            current_energy = energy(x)
            gradient = torch.autograd.grad(current_energy, current)[0] * mask
        logs["backward_calls"] += 1
        logs["forward_calls"] += 1
        logs["fk_calls"] += 1
        norm = rms(gradient)
        if not torch.isfinite(norm) or norm < 1e-10:
            logs["skip_reason"] = "tiny_or_nonfinite_gradient"
            break
        accepted = False
        for attempt in range(max_backtracking):
            delta = -step_size * (0.5**attempt) * gradient / norm.clamp_min(1e-10)
            candidate = current.detach() + delta
            logs["proposed"] += 1
            try:
                with torch.no_grad():
                    cx = decode(candidate)
                    ce = energy(cx)
                    if mode != "unrestricted":
                        feat = teacher_fn(cx)
                        cs = F.cosine_similarity(
                            feat["global"].float(), text_embedding.float()
                        ).mean()
                        distances = 1 - F.cosine_similarity(
                            feat["parts"].float(), feature0["parts"].float(), dim=-1
                        )
                        shifts = {
                            name: float(distances[..., index].max())
                            for name, index in [("left", 1), ("right", 2), ("face", 3)]
                        }
                    else:
                        cs = score0
                        distances = cx.new_zeros(1)
                        shifts = {}
                    finite = (
                        torch.isfinite(cx).all()
                        and torch.isfinite(ce)
                        and torch.isfinite(cs)
                        and torch.isfinite(distances).all()
                    )
                    # Rotation validity is checked on the decoded SO(3) matrices by the relation backend.
                    ok = finite and ce < current_energy.detach() - energy_improvement_min
                    if mode == "semantic":
                        ok = ok and cs >= score0 - calibration["epsilon_sem"]
                    if mode != "unrestricted":
                        ok = ok and all(
                            shifts[name] <= calibration["epsilon_parts"][name] for name in shifts
                        )
                    ok = ok and rms(candidate - original) <= max_latent_delta_rms
            except (ValueError, FloatingPointError) as exc:
                ok = False
                logs["rejections"].append(str(exc))
            logs["forward_calls"] += 1
            logs["fk_calls"] += 1
            if ok:
                current = candidate.detach()
                logs["accepted"] += 1
                logs.update(
                    final_energy=float(ce),
                    final_semantic=float(cs) if mode != "unrestricted" else None,
                    part_shifts=shifts,
                )
                accepted = True
                break
        if not accepted:
            logs["skip_reason"] = "no_candidate_within_budgets"
            current = current.detach()
            break
    logs["delta_geometry"] = logs["final_energy"] - logs["initial_energy"]
    logs["delta_semantic"] = (
        (logs["final_semantic"] - logs["initial_semantic"]) if mode != "unrestricted" else None
    )
    return current.detach(), logs
