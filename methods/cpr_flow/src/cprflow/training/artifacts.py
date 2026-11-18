"""Resolve artifact indices and verify that fitted/cached dependencies remain current."""

import json
from pathlib import Path

from cprflow.data.cache import file_hash
from cprflow.data.stats import Normalizer
from cprflow.training.checkpoint import load_checkpoint


def index_entry(root, kind):
    path = Path(root) / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact index {path}; prepare {kind} first")
    data = json.loads(path.read_text())
    if kind not in data:
        raise FileNotFoundError(f"{kind} absent from {path}")
    return data[kind]


def update_index(root, kind, entry):
    # Different cache builders can finish concurrently; preserve both index entries.
    import fcntl
    import os

    path = Path(root) / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[kind] = entry
        tmp = path.with_name(f".index.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)


def stats_path(config, kind):
    explicit = config["paths"].get(kind + "_stats")
    path = explicit or index_entry(config["paths"]["stats"], kind)["path"]
    meta = Path(str(path) + ".json")
    if not meta.is_file():
        raise FileNotFoundError(f"Missing statistics provenance {meta}")
    provenance = json.loads(meta.read_text())
    if provenance.get("array_sha256") != file_hash(path):
        raise ValueError(f"{kind} statistics numeric hash mismatch; rebuild statistics")
    if kind == "relation" and provenance.get("relation_definition") != {
        k: config["relation"][k] for k in ("threshold", "softness", "pooling")
    }:
        raise ValueError("Relation statistics definition changed")
    train = Path(config["paths"]["manifests"]) / "train.jsonl"
    if provenance["train_manifest_sha256"] != file_hash(train):
        raise ValueError(f"Stale {kind} statistics: train manifest changed")
    if provenance.get("max_train_frames") != config["data"]["max_train_frames"]:
        raise ValueError("Statistics train inclusion policy changed")
    if kind == "latent" and provenance["ae_sha256"] != file_hash(config["paths"]["ae_checkpoint"]):
        raise ValueError("Stale latent statistics: AE changed")
    if provenance.get("asset_sha256") and provenance["asset_sha256"] != file_hash(
        config["paths"]["smplx_model"]
    ):
        raise ValueError(f"Stale {kind} statistics: asset changed")
    return Path(path)


def normalizer(config, kind, device="cpu"):
    return Normalizer.load(stats_path(config, kind)).to(device)


def scales(config):
    import numpy as np

    path = stats_path(config, "activity")
    with np.load(path, allow_pickle=False) as data:
        return {key: float(data[key]) for key in data.files}


def load_weights(model, path, device="cpu", freeze=True):
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {path}")
    state = load_checkpoint(path)
    model.load_state_dict(state["ema"]["shadow"])
    model.to(device)
    if freeze:
        model.requires_grad_(False).eval()
    return model


def text_revisions(config):
    path = Path(config["paths"]["sources_lock"])
    lock = json.loads(path.read_text())
    revision = config["text"]["revision"] or lock["text"]["revision"]
    tokenizer = config["text"]["tokenizer_revision"] or lock["text"]["tokenizer_revision"]
    if (revision, tokenizer) != (lock["text"]["revision"], lock["text"]["tokenizer_revision"]):
        raise ValueError("Configured text revisions differ from source lock")
    return revision, tokenizer


class CachedFeatures:
    def __init__(self, config, kind):
        from cprflow.data.cache import NumericCache

        entry = index_entry(config["paths"]["cache"], kind)
        deps = entry["dependencies"]
        if deps.get("synthetic", False) != config["run"]["synthetic"]:
            raise ValueError("Synthetic/real cache mixing is forbidden")
        if kind in ("latent", "relations"):
            for key, path in [
                ("asset", config["paths"]["smplx_model"]),
                ("motion_stats", stats_path(config, "motion")),
            ]:
                if deps.get(key) != file_hash(path):
                    raise ValueError(f"Stale {kind} cache: {key} changed")
            if kind == "latent" and deps["ae"] != file_hash(config["paths"]["ae_checkpoint"]):
                raise ValueError("Stale latent cache: AE changed")
            if kind == "latent" and deps["latent_stats"] != file_hash(stats_path(config, "latent")):
                raise ValueError("Stale latent cache: latent stats changed")
            if kind == "relations" and deps["relation_stats"] != file_hash(
                stats_path(config, "relation")
            ):
                raise ValueError("Stale relation cache: relation stats changed")
        if kind == "relations" and deps.get("definition") != {
            k: config["relation"][k] for k in ("threshold", "softness", "pooling")
        }:
            raise ValueError("Stale relation cache: relation definition changed")
        if kind == "text":
            revision, tokenizer = text_revisions(config)
            if (
                deps["revision"] != revision
                or deps["tokenizer_revision"] != tokenizer
                or deps["max_tokens"] != config["text"]["max_tokens"]
            ):
                raise ValueError("Stale text cache")
        self.cache = NumericCache(Path(config["paths"]["cache"]) / kind, deps)
        self.kind = kind

    def get(self, row):
        from cprflow.data.cache import digest

        identity = {"uid": row["uid"], "processed": row["processed_sha256"]}
        if self.kind == "text":
            identity = {"text": row["text"], "language": row["sign_language_id"]}
        values = self.cache.get(digest(identity))
        import numpy as np

        if self.kind == "text":
            hidden = values["hidden"]
            if hidden.ndim != 2 or hidden.shape[-1] != 768 or not 0 < len(hidden) <= 512:
                raise ValueError("Invalid cached mT5 token shape")
        elif self.kind == "latent":
            if values["z"].shape != (4, (row["frames"] + 1) // 2, 128):
                raise ValueError("Cached latent length/profile mismatch")
        elif values["r"].shape != ((row["frames"] + 1) // 2, 33):
            raise ValueError("Cached relation length/profile mismatch")
        if any(not np.isfinite(array).all() for array in values.values()):
            raise ValueError("Nonfinite cached features")
        return values


def load_dependency(model, path, config, role, device="cpu"):
    """Frozen stages must have been trained with the same input representation and features."""
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"Missing trained {role} checkpoint: {path}")
    state = load_checkpoint(path)
    if state["config"]["run"]["task"] != role:
        raise ValueError(f"Expected {role} checkpoint")
    if state["config"]["run"]["synthetic"] != config["run"]["synthetic"]:
        raise ValueError("Synthetic/real checkpoint mixing is forbidden")
    expected = {
        "profile": config["data"]["profile"],
        "manifest": file_hash(Path(config["paths"]["manifests"]) / "train.jsonl"),
    }
    if role in ("ae", "teacher"):
        expected["motion_stats"] = file_hash(stats_path(config, "motion"))
        expected["asset"] = file_hash(config["paths"]["smplx_model"])
    if role in ("duration", "teacher"):
        expected["text_revision"], expected["tokenizer_revision"] = text_revisions(config)
    for key, value in expected.items():
        if state["provenance"].get(key) != value:
            raise ValueError(
                f"Frozen {role} dependency mismatch: {key}; explicit migration/retraining required"
            )
    model.load_state_dict(state["ema"]["shadow"])
    if role == "ae":
        model.execution = config["ae"].get("execution", "per_clip")
    return model.to(device).requires_grad_(False).eval()
