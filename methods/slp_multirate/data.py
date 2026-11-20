"""Source adapters, canonical manifests, split/cohort audit and masked batches."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .geometry import (
    JOINTS, PARENTS, SIDES, TIP_JOINTS, Rig, fk, identity6,
    interpolate_rotations, interpolation_indices, matrix_to_rot6d, rot6d_to_matrix, so3_exp,
)

SCHEMA = "1.0"
DEV_SPLITS = ("dev_tune", "dev_gate", "dev_cal")
SPLITS = ("train", *DEV_SPLITS, "test")
PILOT_CAPS = {"p0": (32, 16, 16), "p1": (512, 128, 160), "p2": (1500, 200, 200)}
SMPLX_BODY_21 = (
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
SMPLX_HAND_15 = tuple(f"{finger}{j}" for finger in (
    "index", "middle", "pinky", "ring", "thumb"
) for j in (1, 2, 3))


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path):
    with open(path, "rb") as stream:
        # file_digest was added in Python 3.11.
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts):
    return int(digest(parts)[:16], 16) % (2**63 - 1)


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, json_bytes(value) + b"\n")


def write_jsonl(path, rows):
    atomic_bytes(path, b"".join(json_bytes(row) + b"\n" for row in rows))


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_npz(path, **arrays):
    data = io.BytesIO()
    np.savez_compressed(data, **arrays)
    atomic_bytes(path, data.getvalue())


def read_npz(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as f:
        return {name: f[name] for name in f.files}


def _strict_meta(row, rig):
    for key in ("clip_id", "origin_video_id", "split", "target_language", "path",
                "source_joint_schema", "provenance", "local_axes_id", "rotation_convention",
                "unit", "handedness"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing or empty {key}")
    if not isinstance(row.get("text"), str):
        raise ValueError("Missing or invalid text")
    if row["rotation_convention"] not in ("rows", "columns", "axis_angle_radians"):
        raise ValueError("Unknown source rotation convention")
    if row["unit"] not in ("m", "mm") or row["handedness"] != "right":
        raise ValueError("Explicit source units and right-handed coordinates are required")
    if row["local_axes_id"] != rig.local_axes_id:
        raise ValueError("Rig and source local coordinate axes differ")
    if row["split"] not in ("dev", "val", *SPLITS):
        raise ValueError(f"Unknown split {row['split']}")


def _read_source(row, path, adapter):
    if adapter in ("synthetic", "named_npz"):
        if row["source_joint_schema"] != "named_local_v1":
            raise ValueError("Expected named_local_v1 NPZ")
        raw = path.read_bytes()
        arrays = read_npz(raw)
        required = {"rotations", "joint_names"}
        if not required <= arrays.keys():
            raise ValueError("Named source requires rotations and joint_names")
        return arrays, raw
    if adapter != "signspark_lmdb":
        raise ValueError(f"Unknown adapter {adapter}")
    if row["source_joint_schema"] not in ("smplx_body21_hand15_v1", "smplx_root_body21_hand15_v1"):
        raise ValueError("SignSparK needs the explicit full SMPL-X body 21/22 layout")
    if row["rotation_convention"] not in ("rows", "columns"):
        raise ValueError("SignSparK features must declare their 6D rotation convention")
    if not isinstance(row.get("key"), str) or not row["key"]:
        raise ValueError("SignSparK requires an explicit LMDB key")
    import lmdb

    with lmdb.open(str(path), readonly=True, create=False, subdir=path.is_dir(),
                   readahead=False) as db:
        with db.begin() as txn:
            raw = txn.get(row["key"].encode("utf-8"))
    if raw is None:
        raise ValueError("LMDB key not found")
    arrays = read_npz(raw)
    body = arrays["body_features"]
    left, right = arrays["left_features"], arrays["right_features"]
    names = SMPLX_BODY_21
    if row["source_joint_schema"] == "smplx_root_body21_hand15_v1":
        names = ("pelvis", *names)
    if (body.ndim != 2 or body.shape[-1] != 6 * len(names) or
            any(x.shape != (len(body), 90) for x in (left, right))):
        raise ValueError("Unexpected SignSparK arrays. A trimmed upper-body stream is insufficient")
    if "translation" in arrays and str(arrays["translation"].reshape(-1)[0]) != row["text"]:
        raise ValueError("Source manifest text differs from SignSparK record")
    joint_names = [*names, *(f"{s}_{n}" for s in SIDES for n in SMPLX_HAND_15)]
    rotations = np.concatenate((body, left, right), axis=-1).reshape(len(body), -1, 6)
    observations = arrays.get("observation")
    if observations is None:
        observations = np.ones(rotations.shape[:-1], dtype=bool)
    if "pelvis" not in names:
        # The stock LMDB exports body_pose only. Include the global rotation only
        # if supplied by the same fit. Identity is permitted for a verified canonical root.
        if "pelvis_rotation6d" in arrays:
            pelvis = arrays["pelvis_rotation6d"]
        elif row.get("pelvis_source") == "verified_identity":
            pelvis = identity6(len(body)).numpy()
        else:
            raise ValueError("Source lacks pelvis rotation. Provide same-fit root or verify "
                             "canonical identity root in the source manifest")
        rotations = np.concatenate((np.asarray(pelvis).reshape(len(body), 1, 6), rotations), axis=1)
        pelvis_obs = np.asarray(arrays.get("pelvis_observation", np.ones(len(body))), dtype=bool)
        observations = np.concatenate((pelvis_obs[:, None], observations), axis=1)
        joint_names = ["pelvis", *joint_names]
    arrays.update(rotations=rotations, joint_names=np.asarray(joint_names),
                  observation=observations)
    return arrays, raw


def _timestamps(arrays, row, count):
    source = arrays.get("timestamps_s", row.get("timestamps_s"))
    if source is None:
        fps = row.get("fps")
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Source FPS or timestamps_s must be supplied")
        source = np.arange(count, dtype=np.float64) / fps
    t = torch.as_tensor(source, dtype=torch.float64)
    if (t.shape != (count,) or not t.isfinite().all() or
            (t.numel() > 1 and (t.diff() <= 0).any())):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    clock = (t - t[0]).float()
    if not clock.isfinite().all() or (clock.diff() <= 0).any():
        raise ValueError("Source timestamps are outside FP32 clock resolution")
    return clock


def _segments(labels):
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Segment flags must be 0, 1 or 2")
    starts = []
    start = None
    for n, label in enumerate([*labels, 0]):
        if start is not None and label in (0, 2):
            starts.append([start, n - 1])
            start = None
        if label in (1, 2) and start is None:
            start = n
    return starts


def _contact(arrays, count):
    pairs = np.asarray(arrays.get("contact_pairs", np.empty((0, 2))), dtype=np.int64)
    values = np.asarray(arrays.get("contact", np.zeros((count, len(pairs)))), dtype=bool)
    valid = np.asarray(arrays.get("contact_valid", np.zeros_like(values)), dtype=bool)
    if (pairs.shape != (len(pairs), 2) or (pairs < 0).any() or (pairs >= 10).any()
            or (pairs[:, 0] >= pairs[:, 1]).any() or len(np.unique(pairs, axis=0)) != len(pairs)
            or values.shape != (count, len(pairs)) or valid.shape != values.shape):
        raise ValueError("Contact pairs must be unique sorted pairs of canonical tip IDs")
    return pairs, values, valid


def _canonicalize(arrays, row, fps, min_frames, max_frames):
    x = torch.as_tensor(arrays["rotations"], dtype=torch.float32)
    names = arrays["joint_names"].tolist()
    encoding = row["rotation_convention"]
    dim = 3 if encoding == "axis_angle_radians" else 6
    if (x.ndim != 3 or x.shape[1:] != (len(names), dim) or len(x) == 0
            or len(set(names)) != len(names) or not set(JOINTS) <= set(names)):
        raise ValueError("Missing/duplicate canonical joints or invalid pose shape")
    time = _timestamps(arrays, row, len(x))
    new_count = max(1, round(float(time[-1]) * fps) + 1)
    if not min_frames <= new_count <= max_frames:
        return None, {"reason": "too_short" if new_count < min_frames else "too_long",
                      "frames": new_count}
    order = [names.index(n) for n in JOINTS]
    obs = torch.as_tensor(arrays.get("observation", np.ones(x.shape[:2])), dtype=torch.bool)
    if obs.shape != x.shape[:2]:
        raise ValueError("Wrong source observation shape")
    x, obs = x[:, order], obs[:, order]
    nonfinite = obs & ~x.isfinite().all(-1)
    obs &= ~nonfinite
    x = x.masked_fill(~obs[..., None], 0)
    if dim == 3:
        matrix = so3_exp(x)
        degenerate = torch.zeros_like(obs)
    else:
        matrix, bad = rot6d_to_matrix(x, convention=encoding, return_fallback=True)
        degenerate = obs & bad
        obs &= ~bad
    matrix = torch.where(obs[..., None, None], matrix, torch.eye(3))
    target = torch.arange(new_count, dtype=torch.float32) / fps
    left, right, w = interpolation_indices(time, target)
    nearest = torch.where(w <= 0.5, left, right)
    if not torch.equal(time, target):
        matrix = interpolate_rotations(matrix, time, target)
        obs = ((obs[left] | (w == 1)[:, None]) & (obs[right] | (w == 0)[:, None]))
        matrix = torch.where(obs[..., None, None], matrix, torch.eye(3))
    flags = arrays.get("segment")
    if flags is not None and np.asarray(flags).shape != (len(x),):
        raise ValueError("Segment annotation length mismatch")
    segments = _segments(flags[nearest.numpy()]) if flags is not None else []
    pairs, contact, contact_valid = _contact(arrays, len(x))
    events = row.get("events", [])
    event_types = row.get("event_types", sorted({e["kind"] for e in events}))
    if (not isinstance(event_types, list) or
            any(not isinstance(kind, str) or not kind for kind in event_types) or
            (events or event_types) and not row.get("event_source")):
        raise ValueError("Annotated event types/times require a declared source")
    if not {e["kind"] for e in events} <= set(event_types):
        raise ValueError("A reference event type is missing from event_types")
    if contact_valid.any() and not row.get("contact_source"):
        raise ValueError("Contact labels require a declared contact_source")
    for event in events:
        if (not isinstance(event.get("kind"), str) or
                type(event.get("time_s")) not in (float, int) or
                not 0 <= event["time_s"] <= float(time[-1]) + 1e-6):
            raise ValueError("Events must specify a type and a relative time within the clip")
    return dict(rot6d=matrix_to_rot6d(matrix).numpy(), observation=obs.numpy(),
                timestamps_s=target.numpy(), segments=np.asarray(segments, dtype=np.int64).reshape(-1, 2),
                contact_pairs=pairs, contact=contact[nearest.numpy()], contact_valid=contact_valid[nearest.numpy()]), {
                    "event_types": event_types, "missing_segments": flags is None, "nonfinite_joint_frames": int(nonfinite.sum()),
                    "degenerate_joint_frames": int(degenerate.sum()), "source_frames": len(x),
                    "source_clock": "timestamps" if ("timestamps_s" in arrays or "timestamps_s" in row)
                    else "declared_fps",
                }


def assign_splits(rows, split_seed=17):
    """Assign generic dev by video group once, independently of the training seed."""
    domains = defaultdict(set)
    dev_video = set()
    for row in rows:
        split = row["split"]
        domain = "dev" if split in ("dev", "val", *DEV_SPLITS) else split
        video = row["origin_video_id"]
        domains[video].add(domain)
        if split in ("dev", "val"):
            dev_video.add(video)
        if split in DEV_SPLITS:
            domains[(video, "dev_subsplit")].add(split)
    conflicts = [v for v, splits in domains.items() if len(splits) != 1]
    if conflicts:
        raise ValueError(f"Origin video split overlap for {len(conflicts)} group(s)")
    if any((v, "dev_subsplit") in domains for v in dev_video):
        raise ValueError("Cannot mix generic dev and explicit dev subsplit for the same video")
    vids = sorted(dev_video, key=lambda v: (stable_seed(split_seed, v), v))
    n = len(vids)
    cut1 = min(n, max(1, round(n * .5)))
    cut2 = min(n, max(cut1 + int(n >= 2), round(n * .85)))
    mapping = {v: "dev_tune" if j < cut1 else "dev_gate" if j < cut2 else "dev_cal"
               for j, v in enumerate(vids)}
    return [dict(row, split=mapping[row["origin_video_id"]]) if row["split"] in ("dev", "val")
            else dict(row) for row in rows]


def prepare_data(source, rig_path, adapter, out, cfg, split_seed=17):
    """Prepare data atomically at manifest level and hash all inputs and canonical clips."""
    source, out = Path(source), Path(out)
    rig = Rig.load(rig_path)
    rows = read_jsonl(source)
    if not rows:
        raise ValueError("No source records")
    for row in rows:
        _strict_meta(row, rig)
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source clip ID")
    rows = assign_splits(rows, split_seed)
    out.mkdir(parents=True, exist_ok=True)
    accepted, excluded = [], []
    for row in rows:
        if row["target_language"] != cfg["pilot_language"]:
            excluded.append({"clip_id": row["clip_id"], "reason": "target_language"})
            continue
        if not row["text"].strip():
            excluded.append({"clip_id": row["clip_id"], "reason": "empty_text"})
            continue
        arrays, raw = _read_source(row, source.parent / row["path"], adapter)
        canonical, quality = _canonicalize(arrays, row, cfg["fps"], cfg["min_frames"], cfg["max_frames"])
        if canonical is None:
            excluded.append({"clip_id": row["clip_id"], **quality})
            continue
        count = len(canonical["timestamps_s"])
        raw_hash = hashlib.sha256(raw).hexdigest()
        rel = Path("clips") / (digest([row["clip_id"], raw_hash, rig.sha256, cfg["fps"],
                                      row["rotation_convention"], row.get("fps"),
                                      row.get("timestamps_s"), row.get("pelvis_source")]) + ".npz")
        save_npz(out / rel, **canonical)
        accepted.append({
            "version": SCHEMA, "clip_id": row["clip_id"], "origin_video_id": row["origin_video_id"],
            "split": row["split"], "target_language": row["target_language"], "text": row["text"],
            "path": str(rel), "frames": count, "fps": cfg["fps"],
            "rig_sha256": rig.sha256, "raw_sha256": raw_hash,
            "sha256": file_hash(out / rel), "provenance": row["provenance"],
            "source_joint_schema": row["source_joint_schema"], "quality": quality,
            "source_quality": {k: value for k, value in row.get("quality", {}).items()
                               if k in ("fit_quality", "fit_score", "occlusion")},
            "events": [{k: event[k] for k in ("kind", "time_s")} for event in row.get("events", [])],
            "event_source": row.get("event_source"), "contact_source": row.get("contact_source"),
            "event_types": quality["event_types"],
            "signer_available": bool(row.get("signer_id")), "adapter": adapter,
            "pelvis_source": row.get("pelvis_source", "source_joint"),
        })
    if not accepted:
        raise ValueError("All clips excluded by the canonical filters")
    manifest = out / "manifest.jsonl"
    write_jsonl(manifest, accepted)
    atomic_bytes(out / "rig.json", Path(rig_path).read_bytes())
    write_json(out / "prepare.json", {"version": SCHEMA, "adapter": adapter, "config": cfg,
               "split_seed": split_seed, "source_sha256": file_hash(source),
               "rig_sha256": rig.sha256, "manifest_sha256": file_hash(manifest),
               "excluded": excluded})
    write_json(out / "audit.json", audit_manifest(manifest, rig, "all"))
    return manifest


class CanonicalDataset(Dataset):
    def __init__(self, manifest, rig, splits, cohort=None, fps=None):
        self.manifest = Path(manifest)
        self.rig = rig
        self.cohort = Path(cohort).resolve() if cohort is not None else None
        self.rows = read_jsonl(manifest)
        assign_splits(self.rows)
        ids = [r["clip_id"] for r in self.rows]
        if len(set(ids)) != len(ids) or any(
            r["version"] != SCHEMA or r["rig_sha256"] != rig.sha256 or r["split"] not in SPLITS
            for r in self.rows
        ):
            raise ValueError("Manifest schema/rig mismatch, invalid split or duplicate clip IDs")
        if cohort is not None:
            file = json.loads(Path(cohort).read_bytes())
            if file["manifest_sha256"] != file_hash(manifest):
                raise ValueError("Cohort manifest hash mismatch")
            selected = [cid for split in splits for cid in file["clips"].get(split, ())]
            allowed = {r["clip_id"] for r in self.rows if r["split"] in splits}
            if not set(selected) <= allowed or len(set(selected)) != len(selected):
                raise ValueError("Cohort contains unknown, duplicate or cross-split clip IDs")
            selected = set(selected)
        else:
            selected = set(ids)
        self.rows = [r for r in self.rows if r["split"] in splits and r["clip_id"] in selected]
        if fps is not None and any(r.get("fps") != fps for r in self.rows):
            raise ValueError("Canonical FPS differs from run configuration")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.manifest.parent / row["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Canonical payload hash mismatch for {row['clip_id']}")
        a = read_npz(raw)
        time = torch.from_numpy(a["timestamps_s"])
        x = torch.from_numpy(a["rot6d"])
        obs = torch.from_numpy(a["observation"]).bool()
        if (len(time) != row["frames"] or time.shape != (len(x),) or
                x.shape != (len(time), 44, 6) or obs.shape != x.shape[:-1] or
                not time.isfinite().all() or (time.diff() <= 0).any() or
                ("fps" in row and not torch.allclose(time, torch.arange(len(time)) / row["fps"],
                                                     rtol=0, atol=1e-6)) or
                not x.isfinite().all() or time[0] != 0):
            raise ValueError(f"Invalid canonical payload for {row['clip_id']}")
        return dict(row=row, timestamps_s=time, body_rot6d=x[:, :12],
                    hands_rot6d=x[:, 12:].unflatten(1, (2, 16)),
                    body_obs=obs[:, :12], hands_obs=obs[:, 12:].unflatten(1, (2, 16)),
                    segments=a["segments"].tolist(),
                    **{k: torch.from_numpy(a[k]) for k in (
                        "contact", "contact_valid", "contact_pairs")})


def collate_clips(clips):
    if not clips:
        raise ValueError("Empty batch")
    lengths = torch.tensor([len(c["timestamps_s"]) for c in clips])
    width = int(lengths.max())
    batch = {"length": lengths, "frame_mask": torch.arange(width) < lengths[:, None],
             "text": [c["row"]["text"] for c in clips], "clips": clips}
    for key in ("timestamps_s", "body_rot6d", "hands_rot6d", "body_obs", "hands_obs"):
        values = [c[key] for c in clips]
        pad = (0, 0) * (values[0].ndim - 1)
        batch[key] = torch.stack([F.pad(v, (*pad, 0, width - len(v))) for v in values])
    return batch


def _clip_audit(c, rig):
    obs = torch.cat((c["body_obs"], c["hands_obs"].flatten(1, 2)), -1)
    rot = rot6d_to_matrix(torch.cat((c["body_rot6d"], c["hands_rot6d"].flatten(1, 2)), 1))
    pose = fk(rot, rig, obs)
    movement = pose.local_tips.diff(dim=0).norm(dim=-1)
    valid = pose.tip_mask[1:] & pose.tip_mask[:-1]
    wrist = pose.wrist.diff(dim=0).norm(dim=-1)
    wrist_valid = pose.hand_mask[1:] & pose.hand_mask[:-1]
    active = (((movement > .001) & valid).any(dim=(0, 2)) |
              ((wrist > .001) & wrist_valid).any(dim=0)).tolist()
    return {
        "frames": len(rot), "two_hands_active": all(active), "active_hands": sum(active),
        "occlusion": 1 - float(obs.float().mean()), "segments": len(c["segments"]),
        "contact_labels": int(c["contact_valid"].sum()),
        "contact_positives": int((c["contact_valid"] & c["contact"]).sum()),
        "visible_tip_frames": int(pose.tip_mask.sum()),
        "signer_available": c["row"]["signer_available"],
        "provenance": c["row"]["provenance"],
    }


def audit_manifest(manifest, rig, selection="all"):
    splits = SPLITS if selection == "all" else tuple(
        s for part in selection.split(",") for s in (DEV_SPLITS if part in ("dev", "val") else (part,))
    )
    if not set(splits) <= set(SPLITS):
        raise ValueError("Unknown audit split")
    data = CanonicalDataset(manifest, rig, splits)
    clips = [dict(_clip_audit(c, rig), clip_id=c["row"]["clip_id"], split=c["row"]["split"])
             for c in data]
    review = _stratified([(c["clip_id"], (c["split"], c["frames"] // 32, c["active_hands"],
                                         min(4, int(c["occlusion"] * 20)))) for c in clips], 32,
                         stable_seed("rig-review"))
    return {"version": SCHEMA, "manifest_sha256": file_hash(manifest), "rig_sha256": rig.sha256,
            "stratified_geometry_review_clip_ids": review,
            "contact_coverage_clips": sum(c["contact_labels"] > 0 for c in clips),
            "splits": {s: {"clips": sum(c["split"] == s for c in clips),
                           "frames_histogram": dict(Counter(str(c["frames"]) for c in clips
                                                            if c["split"] == s)),
                           "signer_missing": sum(not c["signer_available"] for c in clips
                                                 if c["split"] == s)}
                       for s in splits}, "clips": clips}


def _stratified(rows, cap, seed):
    groups = defaultdict(list)
    for cid, strata in rows:
        groups[strata].append(cid)
    rng = random.Random(seed)
    queues = list(groups.values())
    for group in queues:
        rng.shuffle(group)
    rng.shuffle(queues)
    picks = []
    while queues and len(picks) < cap:
        for group in queues:
            if group and len(picks) < cap:
                picks.append(group.pop())
        queues = [q for q in queues if q]
    return picks


def make_cohort(manifest, rig, pilot, seed, out):
    from .anchors import event_signals

    data = CanonicalDataset(manifest, rig, ("train", *DEV_SPLITS))
    by_split = defaultdict(list)
    for c in data:
        row = c["row"]
        stats = _clip_audit(c, rig)
        signals, coverage = event_signals(c, rig)
        density = float(((signals[:, :2] > 0) & coverage[:, :2]).any(-1).float().mean())
        strata = (min(stats["frames"] // 32, 8), stats["active_hands"], min(int(density * 4), 3))
        by_split[row["split"]].append((row["clip_id"], strata))
    caps = PILOT_CAPS.get(pilot, (len(data),) * 3)
    selected = {s: _stratified(by_split[s], cap, stable_seed(seed, s)) for s, cap in zip(
        ("train", *DEV_SPLITS), (*caps, len(data))
    )}
    write_json(out, {"version": SCHEMA, "pilot": pilot, "seed": seed,
                     "manifest_sha256": file_hash(manifest), "clips": selected})
    return selected


def create_synthetic(out, seed=17):
    """Public toy rig and fixed geometry fixture. No real signer data."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    offsets = [(0., 0., 0.), (0., .1, 0.), (0., .1, 0.), (0., .1, 0.),
               (0., .10, 0.), (0., .10, 0.), (.08, .04, 0.), (-.08, .04, 0.),
               (.09, 0., 0.), (-.09, 0., 0.), (.14, 0., 0.), (-.14, 0., 0.)]
    for side in (1, -1):
        offsets.append((.10 * side, 0., 0.))
        for finger in range(5):
            offsets.extend(((.018 * side, .01 * (finger - 2), 0.),
                            (.012 * side, 0., 0.), (.010 * side, 0., 0.)))
    rig_path = out / "rig.json"
    write_json(rig_path, {"version": SCHEMA, "unit": "m", "handedness": "right",
                         "palm_convention": "palm_to_wrist", "local_axes_id": "toy-v1",
                         "joints": [dict(name=name, parent=p, offset=offset)
                                    for name, p, offset in zip(JOINTS, PARENTS, offsets)],
                         "tip_offsets": {n: [.008 if n.startswith("left") else -.008, 0., 0.]
                                         for n in TIP_JOINTS},
                         "palm_offsets": {s: np.eye(3).tolist() for s in SIDES}})
    rig = Rig.load(rig_path)
    rng = torch.Generator().manual_seed(seed)
    rows = []
    for i, count in enumerate((8, 9, 11, 13, 15, 17, 19, 21, 1, 5, 8, 13, 17, 23, 11, 15, 24, 9, 13)):
        split = ("train" if i < 10 else "dev_tune" if i < 13 else "dev_gate" if i < 16
                 else "dev_cal" if i < 18 else "test")
        t = torch.linspace(0, 1, count)
        aa = torch.zeros(count, 44, 3)
        aa[:, 8, 2] = .18 * t.sin()
        aa[:, 10, 1] = .13 * (t * math.pi).sin()
        aa[:, 12, 0] = .12 * t
        aa[:, 13, 1] = (.65 + torch.rand((), generator=rng) * .3) * (t > .45)
        aa[:, 31, 2] = .50 * (t > .6)
        contact = np.zeros((count, 1), dtype=bool)
        if i % 3 == 1 and count >= 8:
            a, b = count // 3, 2 * count // 3
            aa[a:b] = 0
            # Rotate both shoulders by pi about the vertical Y axis, then rotate
            # elbows to place the middle tips together at the midsagittal plane.
            for side_id, (shoulder, elbow) in enumerate(((8, 10), (9, 11))):
                aa[a:b, shoulder, 1] = math.pi
                angle = math.acos((rig.shoulder_width / 2 - .14) / .148)
                aa[a:b, elbow, 1] = angle if side_id == 0 else -angle
            contact[a:b, 0] = True
        rotations = so3_exp(aa)
        obs = torch.ones(count, 44, dtype=torch.bool)
        if i % 4 == 0 and count > 4:
            obs[2, 16] = False
        path = f"raw_clips/{i:03}.npz"
        segment = np.zeros(count, dtype=np.int64)
        if count > 4:
            segment[1:-1] = 1
            segment[1] = 2
        save_npz(out / path, rotations=matrix_to_rot6d(rotations).numpy(),
                 joint_names=np.asarray(JOINTS), observation=obs.numpy(), segment=segment,
                 contact_pairs=[[2, 7]], contact=contact, contact_valid=np.ones_like(contact))
        rows.append({"clip_id": f"toy-{i:03}", "origin_video_id": f"toy-video-{i:03}",
                     "split": split, "target_language": "ASL", "text": f"toy sign {i % 4} length {count}",
                     "path": path, "source_joint_schema": "named_local_v1",
                     "provenance": "synthetic-fixture-v1", "fps": 25, "unit": "m",
                     "rotation_convention": "rows", "local_axes_id": "toy-v1", "handedness": "right",
                     "contact_source": "synthetic", "event_source": "synthetic",
                     "events": ([{"kind": "shape", "time_s": (count // 2) / 25}]
                                if count >= 5 else [])})
        if i == 1:
            pose = fk(rotations, rig, obs)
            save_npz(out / "golden.npz", joints_m=pose.joints.numpy(), tips_m=pose.tips.numpy(),
                     tips_wrist_widths=pose.local_tips.numpy(), contact=contact)
    write_jsonl(out / "raw.jsonl", rows)
    return out / "raw.jsonl", rig_path


def make_signer_diagnostic(raw_manifest, manifest, rig, out):
    """Identify a held-out development slice from private signer metadata."""
    raw = read_jsonl(raw_manifest)
    identities = {row["clip_id"]: row.get("signer_id") for row in raw}
    if len(identities) != len(raw):
        raise ValueError("Duplicate clip IDs in the private signer manifest")
    dataset = CanonicalDataset(manifest, rig, ("train", *DEV_SPLITS))
    if any(row["clip_id"] not in identities for row in dataset.rows):
        raise ValueError("Signer manifest does not cover the requested canonical manifest")
    train = {identities[row["clip_id"]] for row in dataset.rows if row["split"] == "train"
             and identities[row["clip_id"]]}
    selected, missing = defaultdict(list), Counter()
    excluded = Counter()
    for row in dataset.rows:
        split, name = row["split"], row["clip_id"]
        identity = identities[name]
        if not identity:
            missing[split] += 1
        elif split != "train":
            if identity in train:
                excluded[split] += 1
            else:
                selected[split].append(name)
    if missing["train"]:
        status = "insufficient_train_identity_coverage"
        selected.clear()
    elif not selected:
        status = "no_held_out_development_signers"
    else:
        status = "diagnostic_slice"
    report = {"version": SCHEMA, "status": status, "manifest_sha256": file_hash(manifest),
              "rig_sha256": rig.sha256, "clips": dict(selected), "missing_signer_counts": dict(missing),
              "seen_signer_exclusion_counts": dict(excluded)}
    write_json(out, report)
    return report
