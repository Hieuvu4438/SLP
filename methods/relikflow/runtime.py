"""Resolved runs, atomic artifacts, RNG state, and stage-independent configuration."""

import importlib.metadata
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import digest, sha256
from .source import ARMS


def write_atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def json_write(path, obj):
    write_atomic(
        path,
        lambda dest: dest.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ),
    )


def torch_write(path, obj):
    def save(dest):
        with dest.open("wb") as handle:
            torch.save(obj, handle)

    write_atomic(path, save)


def checkpointed_records(directory, keys, compute, *, batch_size, resume, label):
    """Compute deterministic batches and commit each as an atomic checkpoint."""
    result = {}
    for start in range(0, len(keys), batch_size):
        batch = keys[start : start + batch_size]
        path = directory / f"{start:06d}.pt"
        if resume and path.is_file():
            records = torch.load(path, map_location="cpu", weights_only=True)
            if list(records) != batch:
                raise ValueError(f"Checkpoint keys do not match {label}: {path}")
        else:
            records = {key: compute(key) for key in batch}
            torch_write(path, records)
        result.update(records)
        print(f"{label}: {len(result)}/{len(keys)} saved", flush=True)
    return result


def npz_write(path, **obj):
    def save(dest):
        with dest.open("wb") as handle:
            np.savez_compressed(handle, **obj)

    write_atomic(path, save)


def json_read(path):
    return json.loads(Path(path).read_text())


def versions():
    names = ("torch", "numpy", "transformers", "sentencepiece", "PyYAML", "smplx", "pyarrow")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def git_revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def config_load(path, profile="full"):
    base = Path(__file__).with_name("config.yaml")
    config = yaml.safe_load(base.read_text())
    if path:
        config = merge(config, yaml.safe_load(Path(path).read_text()))
    if profile != "smoke" and config["text"].get("backend") == "fixture":
        raise ValueError("Fixture text backend is reserved for smoke")
    config["profile"] = profile
    if profile == "smoke":
        merge(
            config,
            dict(
                data=dict(corpus="csl_daily", fps={"csl_daily": 25.0}),
                text=dict(backend="fixture", fixture_width=16),
                planner=dict(width=24, layers=1, heads=4, ffn=32, batch_effective=4, max_epochs=1),
                direct_pose_head=dict(max_epochs=1, batch_effective=4),
                ik=dict(sweeps=2),
                bridge=dict(
                    correction_dims=[361, 32, 32, 164],
                    prototypes_per_joint=8,
                    kmeans_restarts=1,
                    kmeans_max_iter=8,
                    correction_epochs=1,
                    correction_batch_effective=4,
                ),
                generator=dict(
                    width=24,
                    layers=1,
                    heads=4,
                    ffn=32,
                    time_embedding=32,
                    max_epochs=1,
                    batch_effective=4,
                ),
                evaluation=dict(validation_period_epochs=1, bootstrap_repetitions=24),
                runtime=dict(device="cpu", microbatch=2),
            ),
        )
    elif profile == "pilot":
        config["bridge"]["correction_epochs"] = config["stopping"]["correction_pilot_epochs"]
        config["generator"]["max_epochs"] = config["stopping"]["generator_pilot_epochs"]
        config["generator"]["batch_effective"] = config["generator"]["batch_effective_pilot"]
    validate_config(config)
    return config


def validate_config(cfg):
    interval = cfg["runtime"].get("checkpoint_every_steps", 1)
    if type(interval) is not int or interval < 1:
        raise ValueError("runtime.checkpoint_every_steps must be a positive integer")
    if cfg["schema_version"] != 1 or cfg["arm"] not in ARMS:
        raise ValueError("Unsupported schema or arm")
    if cfg["planner"]["relation_dim"] != 39 or cfg["direct_pose_head"]["output_width"] != 246:
        raise ValueError("Invalid planner or direct-pose layout")
    if (
        cfg["ik"]["active_dof"] != 78
        or cfg["ik"]["damping"] <= 0
        or cfg["ik"]["step_cap_rad_per_joint"] <= 0
    ):
        raise ValueError("Invalid IK settings")
    if not 0 <= cfg["source"]["ar1_alpha"] < 1:
        raise ValueError("AR(1) correlation must be in [0,1)")
    if cfg["bridge"]["mode"] not in {"naturalness", "geodesic"}:
        raise ValueError("Invalid bridge mode")
    if cfg["bridge"]["kernel_sigma"] <= 0 or cfg["bridge"]["ridge_rho"] <= 0:
        raise ValueError("Invalid energy configuration")
    for name in ("planner", "generator"):
        width = cfg[name]["width"]
        if width % 2 or width % cfg[name]["heads"] or cfg[name]["layers"] < 1:
            raise ValueError(f"Invalid decoder dimensions for {name}")


def lock_assets(config, rows=()):
    candidates = [
        {
            "name": "smplx",
            "local_path": config["assets"].get("smplx"),
            "source_url": "https://smpl-x.is.tue.mpg.de/",
            "consumers": ["FK", "IK", "geometry"],
            "revision": config["assets"].get("smplx_revision"),
        },
        {
            "name": "SOKE",
            "local_path": config["assets"].get("soke_reference"),
            "source_url": "https://github.com/2000ZRL/SOKE",
            "consumers": ["codec parity", "geometry"],
            "revision": config["protocol"]["geometry_revision"],
        },
        {
            "name": "mBART",
            "local_path": config["text"].get("local_snapshot"),
            "source_url": "https://huggingface.co/facebook/mbart-large-cc25",
            "consumers": ["planner", "generator"],
            "revision": config["text"].get("revision"),
        },
        *config["assets"].get("extra", []),
    ]
    stats = config["data"].get("soke_stats")
    if stats or any(r.get("pose_layout") == "soke133_norm" for r in rows):
        from slp.evaluation.smplx_fk import stats_dir

        for name in ("mean.pt", "std.pt"):
            candidates.append(
                {
                    "name": f"SOKE-{name}",
                    "local_path": str(Path(stats or stats_dir()) / name),
                    "consumers": ["codec"],
                    "source_url": "https://github.com/2000ZRL/SOKE",
                    "revision": None,
                }
            )
    output = []
    for item in candidates:
        item = dict(item)
        path = Path(item["local_path"]) if item.get("local_path") else None
        checksum = None
        if path and path.is_file():
            checksum = sha256(path)
        elif path and path.is_dir():
            if item["name"] == "smplx":
                model = path / "smplx/SMPLX_NEUTRAL.npz"
                alternative = path / "smplx/SMPLX_NEUTRAL.pkl"
                model = model if model.is_file() else alternative
                checksum = sha256(model) if model.is_file() else None
            elif item["name"] == "SOKE":
                item["resolved_revision"] = git_revision(path)
                files = ("mGPT/data/H2S.py", "mGPT/utils/human_models.py", "mGPT/metrics/t2m.py")
                if all((path / f).is_file() for f in files):
                    checksum = digest({f: sha256(path / f) for f in files})
            else:
                checksum = digest(
                    {
                        str(f.relative_to(path)): sha256(f)
                        for f in sorted(path.rglob("*"))
                        if f.is_file()
                    }
                )
        if item.get("sha256") and item["sha256"] != checksum:
            raise ValueError(f"Asset SHA-256 mismatch: {item['name']}")
        item.update(sha256=checksum, availability="available" if checksum else "unavailable")
        output.append(item)
    return output
