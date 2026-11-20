"""CPU diagnostics for the SignNMFlow proposal. Run from the repository root."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from methods.signmflow import geometry as G
from methods.signmflow.naturalness import NaturalnessMeasure, kmeans_joint
from methods.signmflow.paths import path_point
from methods.signmflow.representation import MotionCodec, load_hand_mean


def angle_quat(half_angle):
    return torch.tensor([math.cos(half_angle), math.sin(half_angle), 0, 0], dtype=torch.float64)


def rotation_probes():
    q0 = angle_quat(math.radians(89))[None, None]
    q1 = angle_quat(math.radians(-89))[None, None]
    t = torch.full((1, 1, 1), 0.5, dtype=torch.float64)
    baseline = path_point("geodesic", q0, q1, t)
    aligned = path_point("geodesic", q0, -q1, t)
    baseline_angle = float(2 * G.geodesic_distance(q0, q1).rad2deg())
    physical_angle = float(2 * G.geodesic_distance(q0, -q1).rad2deg())
    midpoint_gap = float(2 * G.geodesic_distance(baseline, aligned).rad2deg())
    assert abs(baseline_angle - 356) < 1e-8
    assert abs(physical_angle - 4) < 1e-8
    assert abs(midpoint_gap - 180) < 1e-8

    # Inference endpoints are only normalized and thus can be exactly antipodal.
    identity = angle_quat(0)[None, None]
    antipodal_endpoint = path_point("geodesic", identity, -identity, torch.ones_like(t))
    endpoint_error = float((antipodal_endpoint + identity).norm())
    assert endpoint_error == 2.0
    return {
        "lift_path_degrees": baseline_angle,
        "shortest_so3_degrees": physical_angle,
        "so3_midpoint_gap_degrees": midpoint_gap,
        "mse_sum_to_equivalent_lift": float((q1 - (-q1)).square().sum()),
        "antipodal_endpoint_lift_error_l2": endpoint_error,
    }


def naturalness_probes():
    proto = torch.stack((angle_quat(0), angle_quat(0.4)))[None]
    measure = NaturalnessMeasure(proto, torch.tensor([[0.5, 0.5]]), sigma=0.5, rho=1e-6).double()
    points = torch.stack((angle_quat(0.2), angle_quat(1.2)))[:, None]
    directions = torch.tensor([[-math.sin(0.2), math.cos(0.2), 0, 0], [0, 0, 1, 0]],
                              dtype=torch.float64)[:, None]
    assert torch.allclose(G.inner(points, directions), torch.zeros_like(points[..., :1]))
    mass = measure.log_mass(points).exp()[:, 0]
    energy = measure.quad_form(points, directions)[:, 0]
    assert mass[0] > mass[1] and energy[0] > 1000 * energy[1]

    # A stationary joint can have at least K frames and fewer than K unique rotations.
    duplicates = angle_quat(0).float().repeat(8, 1)
    try:
        kmeans_joint(duplicates, K=2)
    except RuntimeError as error:
        if "sum of probabilities <= 0" not in str(error):
            raise
        duplicate_error = str(error)
    else:
        raise AssertionError("Expected the current K-means++ duplicate-data failure")
    return {"kernel_mass": mass.tolist(), "energy_unit_tangent": energy.tolist(),
            "constant_joint_kmeans_error": duplicate_error}


def fk_probe():
    # Planar two-link chain at a straight pose with unit bones. F returns the tip.
    def tip(relative_angles):
        global_angles = relative_angles.cumsum(0)
        return torch.stack((global_angles.cos().sum(), global_angles.sin().sum()))

    relative_angles = torch.zeros(2, dtype=torch.float64)
    jacobian = torch.func.jacfwd(tip)(relative_angles)
    metric = jacobian.T @ jacobian + 0.1 * torch.eye(2, dtype=torch.float64)
    diagonal = torch.diag(metric.diag())
    single_velocities = torch.tensor([[1.0, 0], [0, 1.0]], dtype=torch.float64)
    paired_velocities = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    single = ((single_velocities @ metric) * single_velocities).sum(-1)
    paired = ((paired_velocities @ metric) * paired_velocities).sum(-1)
    paired_diagonal = ((paired_velocities @ diagonal) * paired_velocities).sum(-1)
    assert torch.linalg.eigvalsh(metric).min() > 0
    assert torch.allclose(paired, torch.tensor([9.2, 1.2], dtype=torch.float64))
    assert paired_diagonal[0] == paired_diagonal[1]
    return {"regularized_pullback": metric.tolist(), "single_joint_energy": single.tolist(),
            "paired_energy": paired.tolist(), "paired_diagonal_energy": paired_diagonal.tolist()}


def expression_probe():
    # Path time t=0.5 already uses the previous target 0. A new target proposes 1.
    source, previous_target, next_target = 0.0, 0.0, 1.0
    t, s = 0.5, 0.75
    state = (1 - t) * source + t * previous_target
    code_update = state + (s - t) / (1 - t) * (next_target - state)
    anchored_update = (1 - s) * source + s * next_target
    assert code_update == 0.5 and anchored_update == 0.75
    return {"current_state_recurrence": code_update, "retained_source_update": anchored_update}


def summary_stats(quats, adjacent_dots, generator):
    source = G.uniform_hemisphere(quats.shape[:-1], generator=generator)
    dots = (source * quats).sum(-1).clamp(-1, 1)
    angles = dots.acos().rad2deg() * 2
    excess = (angles - dots.abs().acos().rad2deg() * 2).clamp_min(0)
    summary = {}
    for part, joints in {"body": slice(0, 10), "hands": slice(10, 40), "jaw": slice(40, 41)}.items():
        summary[part] = {
            "sampled_source_target_negative_dot_rate": float((dots[:, joints] < 0).float().mean()),
            "rotation_cut_locus_abs_dot_lt_0.01_rate": float((dots[:, joints].abs() < 0.01).float().mean()),
            "mean_excess_projected_path_degrees": float(excess[:, joints].mean()),
            "canonical_adjacent_negative_dot_rate": float((adjacent_dots[:, joints] < 0).float().mean()),
            "target_abs_w_lt_0.05_rate": float((quats[:, joints, 0].abs() < 0.05).float().mean()),
        }
    return summary


def motion_speed(quats):
    previous, following = quats[:-1], quats[1:]
    dot = (previous * following).sum(-1).clamp(-1, 1)
    chord = (following - dot[..., None] * previous).norm(dim=-1)
    return 2 * torch.atan2(chord, dot.abs()).rad2deg()


def prediction_probe(archive_path, inputs, ids, targets, codec):
    predictions = []
    with np.load(archive_path) as archive:
        for i in ids:
            key = inputs[i]["name"].replace("/", "__")
            predictions.append(codec.encode(torch.from_numpy(archive[key]))[0])
    target_lengths = np.array([len(clip) for clip in targets])
    pred_lengths = np.array([len(clip) for clip in predictions])
    result = {
        "artifact": str(archive_path.relative_to(Path.cwd())),
        "length_mae_frames": float(np.abs(pred_lengths - target_lengths).mean()),
        "length_std_frames_target_then_prediction": [float(target_lengths.std()), float(pred_lengths.std())],
        "at_min_length_40_target_then_prediction": [float((target_lengths == 40).mean()),
                                                    float((pred_lengths == 40).mean())],
        "pooled_speed_degrees_per_frame_target_then_prediction": {},
    }
    for label, clips in (("target", targets), ("prediction", predictions)):
        speed = torch.cat([motion_speed(clip) for clip in clips])
        result["pooled_speed_degrees_per_frame_target_then_prediction"][label] = {
            part: {"mean": float(speeds.mean()), "p95": float(torch.quantile(speeds.flatten(), 0.95))}
            for part, speeds in (("body", speed[:, :10]), ("hands", speed[:, 10:40]), ("jaw", speed[:, 40:]))
        }
    return result


@torch.no_grad()
def cache_probes(root, seed):
    from slp.data.sign_smplx import load_stats

    deps = root / "third_party/SOKE/deps"
    mean, std = load_stats(str(deps / "stats"))
    codec = MotionCodec(mean, std, load_hand_mean(deps / "smpl_models"))
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    datasets = {}
    for dataset in ("phoenix_smplx", "csl_smplx", "how2sign_smplx_soke"):
        files = sorted((root / "outputs/signmflow/cache").glob(f"{dataset}_val_*.json"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one val cache for {dataset}, found {len(files)}")
        meta = files[0]
        metadata = json.loads(meta.read_text())
        offsets = metadata["offsets"]
        frames = np.load(meta.with_suffix(".npy"), mmap_mode="r")
        ids = sorted(rng.choice(len(offsets) - 1, min(256, len(offsets) - 1), replace=False))
        clips = [codec.encode(torch.from_numpy(frames[offsets[i]:offsets[i + 1]].copy()))[0] for i in ids]
        quats = torch.cat(clips)
        adjacent = torch.cat([(q[:-1] * q[1:]).sum(-1) for q in clips])
        datasets[dataset] = {"cache": meta.name, "sampled_clips": len(clips), "frames": len(quats),
                             "adjacent_pairs": len(adjacent), "parts": summary_stats(quats, adjacent, generator)}
        paths = list((root / f"outputs/runs/signmflow_v2_main_{dataset}/signmflow").glob("seed0_*/predictions.npz"))
        if len(paths) == 1:
            datasets[dataset]["v2_predictions"] = prediction_probe(paths[0], metadata["inputs"], ids, clips, codec)
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-cache", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"torch": str(torch.__version__), "rotation": rotation_probes(),
              "naturalness": naturalness_probes(), "fk": fk_probe(), "expression": expression_probe()}
    if args.local_cache:
        result["seed"] = 20261118
        result["val_cache"] = cache_probes(Path.cwd(), result["seed"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
