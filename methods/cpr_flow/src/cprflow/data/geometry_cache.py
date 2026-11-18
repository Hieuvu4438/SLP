"""Offline ground-truth neutral FK cache, independent of learned AE/statistics."""

from pathlib import Path

import torch

from cprflow.data.cache import NumericCache, digest, file_hash
from cprflow.data.dataset import MotionDataset

FIELDS = ("joints", "body", "left", "right", "left_local", "right_local", "degenerate")


class GeometryCache:
    def __init__(self, config):
        from cprflow.training.artifacts import index_entry

        entry = index_entry(config["paths"]["cache"], "geometry")
        deps = entry["dependencies"]
        if deps["asset"] != file_hash(config["paths"]["smplx_model"]):
            raise ValueError("Stale ground-truth FK cache: SMPL-X asset changed")
        if deps["synthetic"] != config["run"]["synthetic"]:
            raise ValueError("Synthetic/real FK cache mixing is forbidden")
        self.names = deps["joint_names"]
        self.cache = NumericCache(Path(config["paths"]["cache"]) / "geometry", deps)

    def get(self, row, device="cpu"):
        key = digest({"uid": row["uid"], "processed": row["processed_sha256"]})
        arrays = self.cache.get(key)
        if any(len(a) != row["frames"] for a in arrays.values()):
            raise ValueError("Ground-truth FK cache frame mismatch")
        if all(k in arrays for k in FIELDS):
            out = {k: torch.as_tensor(arrays[k], device=device) for k in FIELDS}
            out["names"] = self.names[: out["joints"].shape[-2]]
            return out
        from cprflow.geometry.joint_maps import BODY_NAMES, hand_names, select

        joints = torch.as_tensor(arrays["joints"], device=device)
        left = select(joints, hand_names("left"), self.names)
        right = select(joints, hand_names("right"), self.names)
        return {
            "joints": joints,
            "names": self.names,
            "body": select(joints, BODY_NAMES, self.names),
            "left": left,
            "right": right,
            "left_local": left[:, 1:] - left[:, :1],
            "right_local": right[:, 1:] - right[:, :1],
            "degenerate": torch.as_tensor(arrays["degenerate"], device=device),
        }


def build_geometry_cache(config, splits):
    from cprflow.geometry.body_model import BodyModel
    from cprflow.training.artifacts import update_index

    device = config["run"]["device"]
    body = BodyModel(config["paths"]["smplx_model"], config["geometry"]["fk_chunk_frames"]).to(
        device
    )
    from cprflow.geometry.joint_maps import BODY_NAMES, hand_names

    required = (
        set(BODY_NAMES) | set(hand_names("left")) | set(hand_names("right")) | {"pelvis", "jaw"}
    )
    names = [name for name in body.names if name in required]
    indices = [body.names.index(name) for name in names]
    deps = {
        "asset": file_hash(config["paths"]["smplx_model"]),
        "profile": body.profile_hash,
        "version": "required_joint_positions_float32_v2",
        "joint_names": names,
        "synthetic": config["run"]["synthetic"],
    }
    cache = NumericCache(Path(config["paths"]["cache"]) / "geometry", deps)
    counts = {}
    with torch.no_grad():
        for split in splits:
            dataset = MotionDataset(
                Path(config["paths"]["manifests"]) / f"{split}.jsonl",
                split,
                config["data"]["datasets"],
            )
            for row in dataset:
                key = digest({"uid": row["uid"], "processed": row["processed_sha256"]})
                if cache.path(key).is_file():
                    continue
                result = body(torch.as_tensor(row["x"], device=device))
                cache.put(
                    key,
                    joints=result["joints"][:, indices].cpu().numpy(),
                    degenerate=result["degenerate"].cpu().numpy(),
                )
            counts[split] = len(dataset)
    update_index(
        config["paths"]["cache"],
        "geometry",
        {"dependencies": deps, "cache_key": cache.key, "counts": counts},
    )
    return {
        "kind": "geometry",
        "counts": counts,
        "learned_statistics": False,
        "synthetic": config["run"]["synthetic"],
    }
