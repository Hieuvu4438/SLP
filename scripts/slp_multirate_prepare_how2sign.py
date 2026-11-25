"""Build the real-data dev-only pilot cohort from local How2Sign RGB and SOKE fits."""

import argparse
import csv
import json
import pickle
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from smplx.vertex_ids import vertex_ids

from methods.slp_multirate.config import resolve
from methods.slp_multirate.data import (
    SMPLX_BODY_21,
    SMPLX_HAND_15,
    CanonicalDataset,
    assign_splits,
    file_hash,
    prepare_data,
    save_npz,
    stable_seed,
    write_json,
    write_jsonl,
)
from methods.slp_multirate.geometry import JOINTS, PARENTS, SIDES, TIP_JOINTS, fk, so3_exp
from slp.data.sign_smplx import SOKE_BAD_HOW2SIGN_IDS

NAMES = (
    "pelvis",
    *SMPLX_BODY_21,
    "jaw",
    "left_eye",
    "right_eye",
    *(f"{side}_{name}" for side in SIDES for name in SMPLX_HAND_15),
)
INDICES = [NAMES.index(name) for name in JOINTS]
BODY_POINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "head": 0,
}
FIT_KEYS = ("smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose", "smplx_rhand_pose")
FPS = 25


def rig_spec(model):
    vertices = model["v_template"]
    rest = model["J_regressor"] @ vertices
    tree = model["kintree_table"][0]
    joints = []
    for name, parent in zip(JOINTS, PARENTS):
        i = NAMES.index(name)
        offset = np.zeros(3)
        if parent:
            j = NAMES.index(parent)
            if tree[i] != j:
                raise ValueError(f"Unexpected SMPL-X parent for {name}")
            offset = rest[i] - rest[j]
        joints.append(dict(name=name, parent=parent, offset=offset.tolist()))
    tips = {
        name: (
            vertices[vertex_ids["smplx"][name[0] + name.split("_")[1][:-1]]]
            - rest[NAMES.index(name)]
        ).tolist()
        for name in TIP_JOINTS
    }
    palms = {}
    for side in SIDES:
        w, middle, index, pinky = (
            rest[NAMES.index(f"{side}_{name}")] for name in ("wrist", "middle1", "index1", "pinky1")
        )
        x = w - middle
        x = x / np.linalg.norm(x)
        y = index - pinky
        y = y - (y @ x) * x
        y = y / np.linalg.norm(y)
        palms[side] = np.stack((x, y, np.cross(x, y)), axis=1).tolist()
    return dict(
        version="1.0",
        unit="m",
        handedness="right",
        palm_convention="palm_to_wrist",
        local_axes_id="soke_neural_sign_actors_smplx_local_v1",
        joints=joints,
        tip_offsets=tips,
        palm_offsets=palms,
    )


def catalog(packed):
    rows = []
    for split in ("train", "val"):
        csv_name = f"how2sign_realigned_{split}_preprocessed_fps.csv"
        with (packed / split / "re_aligned" / csv_name).open(newline="") as stream:
            for row in csv.DictReader(stream):
                clip = row["SENTENCE_NAME"]
                path = packed / "poses_video_level" / split / "poses" / f"{clip}.pkl"
                if row["SENTENCE"].strip() and clip not in SOKE_BAD_HOW2SIGN_IDS and path.exists():
                    rows.append(
                        dict(
                            clip_id=clip,
                            split=split,
                            origin_video_id=row["VIDEO_ID"],
                            text=row["SENTENCE"],
                            fit=path,
                            raw_split=split,
                        )
                    )
    return assign_splits(rows)


def load_fit(path, mean):
    with path.open("rb") as stream:
        fit = pickle.load(stream)
    root, body, left, right = (
        np.asarray(fit[k], dtype=np.float32).reshape(-1, n, 3)
        for k, n in zip(FIT_KEYS, (1, 21, 15, 15))
    )
    return np.concatenate(
        (root, body, np.zeros((len(root), 3, 3), np.float32), left + mean[0], right + mean[1]),
        axis=1,
    )[:, INDICES]


def rgb_times(video, n):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    s = json.loads(result.stdout)["streams"][0]
    rate = Fraction(s["avg_frame_rate"])
    if int(s["nb_frames"]) != n:
        raise ValueError("RGB and fit frame counts differ")
    if rate <= 0:
        raise ValueError("Invalid RGB fps")
    if rate == Fraction(s["r_frame_rate"]):
        return np.arange(n, dtype=np.float64) / float(rate), str(rate)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    pts = np.array(
        [
            float(frame["best_effort_timestamp_time"])
            for frame in json.loads(result.stdout)["frames"]
        ],
        dtype=np.float64,
    )
    if len(pts) != n or not np.isfinite(pts).all() or (np.diff(pts) <= 0).any():
        raise ValueError("Invalid RGB frame timestamps")
    return pts - pts[0], "variable"


def observation(kp, scores):
    valid = np.isfinite(kp).all(axis=-1) & np.isfinite(scores) & (scores >= 4.0)
    shoulders = valid[:, 5] & valid[:, 6]
    out = np.broadcast_to(shoulders[:, None], (len(kp), len(JOINTS))).copy()
    for j, name in enumerate(JOINTS):
        if name in BODY_POINTS:
            out[:, j] &= valid[:, BODY_POINTS[name]]
        for side, start in zip(SIDES, (91, 112)):
            if name.startswith(side + "_") and name[-1].isdigit():
                finger = ("thumb", "index", "middle", "ring", "pinky").index(
                    name.split("_")[1][:-1]
                )
                index = start + 1 + 4 * finger + int(name[-1]) - 1
                out[:, j] &= valid[:, index] & valid[:, start]
    return out, valid


def observed_turns(kp, valid, times, duration):
    """Sparse wrist reversals from independent RGB 2D pose tracks at 25 fps."""
    tick = np.arange(round(duration * FPS) + 1) / FPS
    right = np.searchsorted(times, tick).clip(1, len(times) - 1)
    left = right - 1
    a = ((tick - times[left]) / (times[right] - times[left])).clip(0, 1)
    geometry = kp[:, [5, 6, 9, 10]]
    pos = np.nan_to_num(
        geometry[left] * (1 - a[:, None, None]) + geometry[right] * a[:, None, None]
    )
    seen0 = valid[:, [5, 6, 9, 10]]
    seen = (seen0[left] | (a == 1)[:, None]) & (seen0[right] | (a == 0)[:, None])
    center = pos[:, :2].mean(1)
    width = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
    relative = (pos[:, 2:] - center[:, None]) / np.maximum(width, 0.03)[:, None, None]
    relative = median_filter(relative, size=(3, 1, 1), mode="nearest")
    observed = seen[:, 2:] & seen[:, :2].all(1)[:, None] & (width > 0.03)[:, None]
    evidence = np.zeros(len(tick))
    for f in range(4, len(tick) - 4):
        prev = relative[f] - relative[f - 3]
        next_ = relative[f + 3] - relative[f]
        speeds = np.linalg.norm(prev, axis=1) * np.linalg.norm(next_, axis=1)
        cosine = (prev * next_).sum(-1) / np.maximum(speeds, 1e-9)
        reliable = observed[f - 4 : f + 5].all(0) & (speeds >= 0.06**2)
        evidence[f] = (np.arccos(np.clip(cosine, -1, 1)) * reliable).max()
    peaks = sorted(np.flatnonzero(evidence >= np.pi / 3), key=lambda f: (-evidence[f], f))
    chosen = []
    for frame in peaks:
        if all(abs(frame - other) >= 20 for other in chosen):
            chosen.append(frame)
    return [
        {"kind": "turn", "time_s": float(tick[f])} for f in sorted(chosen) if tick[f] <= duration
    ]


def geometry_errors(rotations, kp, obs, rig):
    """Fit a nonreflecting 2D shoulder alignment, then check elbows and wrists."""
    sample = np.linspace(0, len(kp) - 1, min(12, len(kp)), dtype=int)
    p = fk(so3_exp(torch.from_numpy(rotations[sample])), rig).joints.numpy()
    use = tuple(k for k in BODY_POINTS if k != "head")
    j = [JOINTS.index(k) for k in use]
    q = kp[sample][:, [BODY_POINTS[k] for k in use]].astype(np.float64)
    visible = obs[sample][:, j]
    good = visible[:, :2].all(1) & (visible[:, 2:].sum(1) == 4)
    p, q = p[good][:, j, :2], q[good]
    if not len(p):
        return []
    dp = p[:, 0] - p[:, 1]
    dq = q[:, 0] - q[:, 1]
    width = np.linalg.norm(dq, axis=1)
    z_p = dp[:, 0] + 1j * dp[:, 1]
    z_q = dq[:, 0] + 1j * dq[:, 1]
    okay = (np.abs(z_p) > 0.05) & (width > 0.03)
    dp, dq, p, q, z_p, z_q, width = (x[okay] for x in (dp, dq, p, q, z_p, z_q, width))
    local = p - p[:, :2].mean(1)[:, None]
    proj = (local[:, :, 0] + 1j * local[:, :, 1]) * (z_q / z_p)[:, None]
    proj = np.stack((proj.real, proj.imag), -1) + q[:, :2].mean(1)[:, None]
    same = np.linalg.norm(proj[:, 2:] - q[:, 2:], axis=-1).mean(1) / width
    swap = np.linalg.norm(proj[:, 2:] - q[:, [3, 2, 5, 4]], axis=-1).mean(1) / width
    return list(zip(same.tolist(), swap.tolist(), strict=True))


def process(row, args, mean, rig):
    prefix, poses = (
        ("train", "train_pose") if row["raw_split"] == "train" else ("eval", "eval_pose")
    )
    clip = row["clip_id"]
    parent = args.source / prefix
    raw_path = parent / poses / f"{clip}.pkl"
    video = parent / "raw_videos" / f"{clip}.mp4"
    try:
        rotations = load_fit(row["fit"], mean)
        if len(rotations) < 2 or not np.isfinite(rotations).all():
            return "invalid_fit"
        times, _ = rgb_times(video, len(rotations))
        count = round(float(times[-1]) * FPS) + 1
        if not 8 <= count <= 320:
            return "outside_length"
        with raw_path.open("rb") as stream:
            keypoints = pickle.load(stream)
        kp, scores = (np.asarray(keypoints[k]).squeeze(1) for k in ("keypoints", "scores"))
        if kp.shape != (len(times), 133, 2) or scores.shape != kp.shape[:-1]:
            return "2d_frame_or_schema_mismatch"
        obs, valid = observation(kp, scores)
        if np.mean(obs[:, 12:]) < 0.30:
            return "low_hand_visibility"
    except (
        OSError,
        ValueError,
        KeyError,
        pickle.UnpicklingError,
        subprocess.SubprocessError,
    ) as err:
        return type(err).__name__
    read_events = row["split"] in ("dev_tune", "dev_gate")
    events = observed_turns(kp, valid, times, float(times[-1])) if read_events else []
    if row["split"] == "dev_gate" and not events:
        return "no_visible_independent_events"
    name = f"raw/{stable_seed(17, clip):016x}.npz"
    save_npz(
        args.out / name,
        rotations=rotations,
        joint_names=np.asarray(JOINTS),
        observation=obs,
        timestamps_s=times,
    )
    geom = geometry_errors(rotations, kp, obs, rig) if row["split"] == "dev_tune" else []
    return dict(
        clip_id=clip,
        origin_video_id=row["origin_video_id"],
        split=row["split"],
        text=row["text"],
        path=name,
        target_language="ASL",
        source_joint_schema="named_local_v1",
        provenance="How2Sign local research RGB and 133-point poses; SOKE Neural Sign Actors SMPL-X fits",
        local_axes_id=rig.local_axes_id,
        rotation_convention="axis_angle_radians",
        unit="m",
        handedness="right",
        events=events,
        event_source="how2sign_rgb_2d_wrist_turn_proxy" if read_events else None,
        event_types=["turn"] if read_events else [],
        quality={"fit_quality": "2d_visibility_masked", "occlusion": 1 - float(obs.mean())},
    ), geom


def prepare(args):
    if args.train < 1:
        raise ValueError("Need positive train limit")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    model = np.load(args.smplx, allow_pickle=False)
    mean = (model["hands_meanl"].reshape(15, 3), model["hands_meanr"].reshape(15, 3))
    from methods.slp_multirate.geometry import Rig

    path = args.out / "source_rig.json"
    write_json(path, rig_spec(model))
    rig = Rig.load(path)
    rows = catalog(args.packed)
    selected, rejected, geometry = [], Counter(), []
    for split, limit in (
        ("train", args.train),
        ("dev_tune", 275),
        ("dev_gate", None),
        ("dev_cal", 30),
    ):
        ordered = sorted(
            (r for r in rows if r["split"] == split), key=lambda r: stable_seed(17, r["clip_id"])
        )
        n = 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            for start in range(0, len(ordered), 48):
                for result in pool.map(
                    lambda r: process(r, args, mean, rig), ordered[start : start + 48]
                ):
                    if isinstance(result, str):
                        rejected[split, result] += 1
                    elif limit is None or n < limit:
                        row, scores = result
                        geometry.extend(scores)
                        selected.append(row)
                        n += 1
                if start % 240 == 0:
                    print(
                        f"{split}: accepted {n} from {min(len(ordered), start + 48)} checked",
                        flush=True,
                    )
                if limit is not None and n >= limit:
                    break
    manifest = args.out / "source.jsonl"
    write_jsonl(manifest, selected)
    canon = prepare_data(
        manifest, path, "named_npz", args.out / "canonical", resolve("p1", "D")["data"]
    )
    by_split = {
        s: CanonicalDataset(canon, rig, (s,)).rows
        for s in ("train", "dev_tune", "dev_gate", "dev_cal")
    }
    dev = by_split["dev_gate"]
    videos = sorted({r["origin_video_id"] for r in dev}, key=lambda v: stable_seed("pilot-gate", v))
    mid = len(videos) // 2
    counts = {}
    for pilot, allowed, train, tune, gate_size in (
        ("p1", set(videos[:mid]), 512, 128, 160),
        ("p2", set(videos[mid:]), 1500, 200, 200),
    ):
        gate = sorted(
            (r for r in dev if r["origin_video_id"] in allowed),
            key=lambda r: stable_seed(pilot, r["clip_id"]),
        )[:gate_size]
        clips = {
            s: [r["clip_id"] for r in subset]
            for s, subset in (
                ("train", by_split["train"][:train]),
                ("dev_tune", by_split["dev_tune"][:tune]),
                ("dev_gate", gate),
                ("dev_cal", by_split["dev_cal"]),
            )
        }
        counts[pilot] = dict(
            train=len(clips["train"]),
            dev_tune=len(clips["dev_tune"]),
            dev_gate=len(gate),
            gate_video_groups=len({r["origin_video_id"] for r in gate}),
        )
        write_json(
            args.out / f"{pilot}.json",
            dict(
                version="1.0", pilot=pilot, seed=17, manifest_sha256=file_hash(canon), clips=clips
            ),
        )
    e = np.asarray(geometry)
    coverage = []
    for r in dev:
        frames = {
            t
            for event in r["events"]
            for t in range(round(FPS * event["time_s"]) - 3, round(FPS * event["time_s"]) + 4)
            if 0 <= t < r["frames"]
        }
        coverage.append(len(frames) / r["frames"])
    report = dict(
        cohorts=counts,
        rejections={f"{s}/{k}": v for (s, k), v in rejected.items()},
        rig_sha256=rig.sha256,
        manifest_sha256=file_hash(canon),
        smplx_sha256=file_hash(args.smplx),
        evaluated_pose_frames=len(e),
        median_wrist_elbow_error_shoulder_widths=float(np.median(e[:, 0])),
        mean_correct_side_beats_swapped=float(np.mean(e[:, 0] < e[:, 1])),
        mean_event_window_fraction=float(np.mean(coverage)),
        event_source="RGB 2D wrist reversals",
    )
    write_json(args.out / "prep_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--smplx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1550)
    prepare(parser.parse_args())
