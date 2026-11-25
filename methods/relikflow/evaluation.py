"""Protocol and semantic adapters around the shared evaluator."""

import importlib
import time
from pathlib import Path

import numpy as np
import torch

from slp.evaluation.flow_matching import bleu_counts, geometry
from .data import digest, phase_grid, resample, sha256, unpack133
from .relations import GROUPS, body_frame, describe, wrist_tips

PUBLISHED = [
    ("phoenix_2014t", "mono", 4.37, 5.57, 3.85, 1.07, 16.15),
    ("phoenix_2014t", "multi", 4.42, 5.75, 3.91, 1.08, 14.45),
    ("csl_daily", "mono", 5.73, 7.43, 5.10, 1.30, 15.14),
    ("csl_daily", "multi", 5.94, 7.73, 5.31, 1.33, 15.23),
    ("how2sign", "mono", 6.76, 9.72, 6.01, 2.07, 16.88),
    ("how2sign", "multi", 6.74, 9.77, 5.99, 2.06, 17.49),
]


def locked_protocol(protocol):
    effective = dict(protocol)
    for kind in ("bt", "siclip"):
        plugin = protocol.get(kind)
        if plugin is None:
            continue
        for required in ("callable", "id", "revision", "preprocessing", "checkpoint", "sha256"):
            if not plugin.get(required):
                raise ValueError(f"Missing {kind} protocol field {required}")
        if sha256(plugin["checkpoint"]) != plugin["sha256"]:
            raise ValueError(f"Scorer checkpoint checksum mismatch: {kind}")
        if kind == "bt" and plugin.get("input") == "fk_features":
            if not plugin.get("feature_groups") or not plugin.get("feature_dim"):
                raise ValueError("FK-based BT requires a locked feature mapping")
            if plugin.get("feature_stats") and (
                sha256(plugin["feature_stats"]) != plugin.get("feature_stats_sha256")
            ):
                raise ValueError("BT feature statistics checksum mismatch")
        if kind == "siclip" and sha256(plugin["candidate_pool"]) != plugin["candidate_sha256"]:
            raise ValueError("SiCLIP candidate pool checksum mismatch")
    return effective


def _plugin(spec):
    mod, fn = spec["callable"].split(":", 1)
    return getattr(importlib.import_module(mod), fn)


@torch.no_grad()
def bt_fk_features(pose, fk, spec):
    """Map physical motion to the joint channels named by a BT checkpoint's release."""
    if fk is None:
        raise ValueError("FK-based BT requires a geometry adapter")
    buffers = list(fk.buffers())
    device = buffers[0].device if buffers else "cpu"
    q, e = unpack133(pose.to(device))
    body, left, right = fk.score_joints(q, e)
    regions = {"body": body, "left_hand": left, "right_hand": right}
    parts = []
    for group in spec["feature_groups"]:
        points = regions[group["region"]]
        selected = points[:, group["indices"]]
        if "root" in group:
            selected = selected - points[:, group["root"], None, :]
        if "axes" in group:
            selected = selected[..., group["axes"]]
        parts.append(selected.flatten(1))
    result = torch.cat(parts, dim=-1)
    if result.shape[1] != spec["feature_dim"]:
        raise ValueError("BT feature width does not match the locked protocol")
    if spec.get("feature_stats"):
        with np.load(spec["feature_stats"], allow_pickle=False) as stats:
            mean, std = (
                torch.as_tensor(stats[k], device=device, dtype=result.dtype)
                for k in ("mean", "std")
            )
        if mean.shape != result.shape[1:] or std.shape != mean.shape or (std <= 0).any():
            raise ValueError("Invalid BT feature statistics")
        result = (result - mean) / std
    return result.cpu()


def semantics(pred, clip, protocol, *, metric="bt", fk=None):
    spec = protocol.get(metric)
    if not spec:
        return {f"{metric}_status": "unavailable"}
    if metric != "bt":
        raise ValueError("Use the corpus adapter for retrieval")
    inputs = (
        bt_fk_features(pred["pose133"], fk, spec)
        if spec.get("input") == "fk_features"
        else pred["pose133"]
    )
    texts = _plugin(spec)(inputs, pred["fps"], clip.row["text"], clip.row["corpus"], spec)
    hypothesis, reference = texts["hypothesis_tokens"], texts["reference_tokens"]
    if any(
        not isinstance(x, list) or not all(isinstance(t, str) for t in x)
        for x in (hypothesis, reference)
    ):
        raise ValueError("BT backend must return token lists")
    return {
        "bt_status": "verified" if spec.get("verified") else "reconstructed",
        "bt_hypothesis_tokens": hypothesis,
        "bt_reference_tokens": reference,
        "bt_counts": bleu_counts(hypothesis, reference),
        "bt_scorer_id": spec["id"],
    }


def jerk(points, fps):
    if len(points) < 4:
        return None
    return float((points.diff(n=3, dim=0) * fps**3).square().mean().sqrt())


def relation_metrics(pr, rr, ref_width, path):
    u, v = pr[path[:, 0]], rr[path[:, 1]]
    a, b = u[:, [0, 1, 2, 18, 19, 20]], v[:, [0, 1, 2, 18, 19, 20]]
    wrist = (a - b).reshape(-1, 2, 3).norm(dim=-1)
    axes = (u[:, GROUPS["axes"]].reshape(-1, 4, 3) * v[:, GROUPS["axes"]].reshape(-1, 4, 3)).sum(-1)
    result = {
        "wrist_torso_error": float(wrist.mean()),
        "wrist_torso_mm": float((wrist * ref_width[path[:, 1], None]).mean() * 1000),
        "palm_direction_deg": float(axes[:, [0, 2]].clamp(-1, 1).acos().rad2deg().mean()),
        "palm_normal_deg": float(axes[:, [1, 3]].clamp(-1, 1).acos().rad2deg().mean()),
    }
    for name in ("flex", "aperture", "face_chest", "inter"):
        result[f"{name}_mae"] = float((u[:, GROUPS[name]] - v[:, GROUPS[name]]).abs().mean())
    return result


@torch.no_grad()
def score_clip(pred, clip, fk, protocol=None):
    protocol = protocol or {}
    buffers = list(fk.buffers())
    device = buffers[0].device if buffers else "cpu"
    q, e = unpack133(pred["pose133"].to(device))
    r_q, r_e = clip.q.to(device), clip.e.to(device)
    start = time.perf_counter()
    decoded, reference = fk.score_joints(q, e), fk.score_joints(r_q, r_e)
    out, path = geometry(decoded, reference, protocol.get("body_pa_alignment", "all"))
    p, r = fk(q), fk(r_q)
    pr, _ = describe(p)
    rr, _ = describe(r)
    out.update(relation_metrics(pr, rr, body_frame(r)[1], path))
    points, _ = wrist_tips(p)
    refs, _ = wrist_tips(r)
    if pred["fps"] != clip.fps and len(points) > 1:
        n = 1 + round((len(points) - 1) / pred["fps"] * clip.fps)
        points = resample(points, phase_grid(n, device=points.device))
    j1, j2 = jerk(points, clip.fps), jerk(refs, clip.fps)
    out.update(
        T_reference=len(r),
        fps_reference=clip.fps,
        T_hat=pred["T_hat"],
        fps=pred["fps"],
        duration_mae_seconds=abs(pred["T_hat"] / pred["fps"] - len(r) / clip.fps),
        from_duration_cap=pred["T_hat"] == pred.get("length_cap"),
        jerk_pred=j1,
        jerk_ref=j2,
        jerk_ratio=j1 / j2 if j2 and j1 is not None else None,
        geometry_seconds=time.perf_counter() - start,
    )
    if "q0" in pred:
        from .rotations import physical_angle

        phase = phase_grid(len(r), device=r_q.device)
        for key in ("q0", "center"):
            if key not in pred:
                continue
            source = resample(pred[key].to(r_q), phase, quaternion=True)
            out[key + "_angle_rad"] = float(physical_angle(source, r_q).mean())
            rsource, _ = describe(fk(source))
            idx = np.arange(len(r))
            for name, val in relation_metrics(
                rsource, rr, body_frame(r)[1], np.column_stack((idx, idx))
            ).items():
                out[key + "_" + name] = val
    return out


def retrieval(outputs, corpus, protocol):
    spec = protocol.get("siclip")
    if spec is None:
        return {"status": "unavailable"}
    score = _plugin(spec)(outputs, corpus, spec)
    return {
        "status": "verified" if spec.get("verified") else "reconstructed",
        "recall_at_1": float(score),
        "scorer_id": spec["id"],
        "candidate_hash": spec["candidate_sha256"],
    }


def cohort_hash(rows):
    return digest(
        sorted(
            (r["clip_id"], r["corpus"], r["pose_fit_id"], r["text"], r.get("sha256")) for r in rows
        )
    )


def prediction_dir(root, split, arm, nfe):
    return Path(root) / "predictions" / split / arm / f"nfe_{nfe}"
