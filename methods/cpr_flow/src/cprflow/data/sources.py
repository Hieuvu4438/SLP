"""Resolve sources once, hash actual bytes, download only authorized pinned pose data."""

import importlib.metadata
import json
import subprocess
import urllib.request
from pathlib import Path

from cprflow.data.cache import file_hash
from cprflow.types import DATASETS

REFERENCES = {
    "signspark": (
        "JianHe0628/SignSparK",
        [
            "DATA.md",
            "signspark/pose_datasets_lmdb.py",
            "tools/download_data.py",
            "tools/visualize.py",
            "LICENSE",
        ],
    ),
    "smplx": (
        "vchoutas/smplx",
        [
            "smplx/body_models.py",
            "smplx/joint_names.py",
            "smplx/vertex_joint_selector.py",
            "smplx/lbs.py",
            "LICENSE",
        ],
    ),
    "mardm": ("neu-3dvi/MARDM", ["models/AE.py", "LICENSE.txt"]),
    "dit": ("facebookresearch/DiT", ["models.py", "LICENSE.txt"]),
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "CPRFlow-source-audit/1.0"})
    with urllib.request.urlopen(req, timeout=40) as stream:
        return stream.read()


def require_access(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing access record {path}; record existing lab authorization for the three corpora"
        )
    record = json.loads(path.read_text())
    for dataset in DATASETS:
        if record.get(dataset, {}).get("authorized") is not True or not record[dataset].get(
            "reference"
        ):
            raise ValueError(f"No recorded authorization for {dataset}")
    return record


def audit_sources(config, resolve=False):
    from cprflow.training.logging import environment

    lock_path = Path(config["paths"]["sources_lock"])
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        # A copied lock remains usable without reference checkouts; present snapshots must match.
        for repo in lock["repositories"].values():
            if len(repo["revision"]) != 40:
                raise ValueError("Source revision must be a resolved commit")
            for item in repo["files"].values():
                if Path(item["path"]).is_file() and file_hash(item["path"]) != item["sha256"]:
                    raise ValueError(f"Source snapshot changed: {item['path']}")
        requirements = lock_path.parent / "requirements.lock.txt"
        if not requirements.is_file() or file_hash(requirements) != lock["dependency_lock_sha256"]:
            raise ValueError("Dependency lock missing or hash mismatch")
        asset = Path(config["paths"]["smplx_model"])
        if (
            lock["assets"]["smplx_sha256"]
            and asset.is_file()
            and file_hash(asset) != lock["assets"]["smplx_sha256"]
        ):
            raise ValueError("SMPL-X asset differs from source lock")
        return lock
    if not resolve:
        raise FileNotFoundError(f"{lock_path}: use --resolve-lock once")
    root = lock_path.parent / "source_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    lock = {"schema": "cprflow_sources_v1", "repositories": {}, "environment": environment()}
    for name, (repo, files) in REFERENCES.items():
        meta = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/HEAD"))
        revision = meta["sha"]
        snapshots = {}
        for relative in files:
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{relative}"
            content = fetch(url)
            target = root / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            snapshots[relative] = {
                "sha256": file_hash(target),
                "url": url,
                "path": str(target.resolve()),
            }
        lock["repositories"][name] = {
            "url": f"https://github.com/{repo}",
            "revision": revision,
            "files": snapshots,
        }
    from huggingface_hub import HfApi

    api = HfApi()
    lock["dataset"] = {
        "repo_id": "LionelLow/SignSparK_data",
        "repo_type": "dataset",
        "revision": api.dataset_info("LionelLow/SignSparK_data").sha,
    }
    revision = api.model_info("google/mt5-base").sha
    lock["text"] = {
        "repo_id": "google/mt5-base",
        "revision": revision,
        "tokenizer_revision": revision,
    }
    lock["dependencies"] = {
        dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
    }
    requirements = lock_path.parent / "requirements.lock.txt"
    freeze = subprocess.run(
        ["python", "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    requirements.write_text(freeze)
    lock["dependency_lock_sha256"] = file_hash(requirements)
    asset = Path(config["paths"]["smplx_model"])
    lock["assets"] = {
        "smplx_path": str(asset.resolve()),
        "smplx_sha256": file_hash(asset) if asset.is_file() else None,
        "num_expression_coeffs": 50,
        "num_betas": 10,
        "use_face_contour": False,
        "joint_mapper": None,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2))
    return lock


def download_authorized(config):
    require_access(config["paths"]["access_record"])
    lock = audit_sources(config)
    from huggingface_hub import snapshot_download

    patterns = []
    for _, _, prefix in DATASETS.values():
        for split in ("train", "dev", "test"):
            patterns.append(f"{split}/{prefix}{split}.lmdb/**")
    destination = Path(config["paths"]["source_pose"]) / lock["dataset"]["revision"]
    snapshot_download(
        lock["dataset"]["repo_id"],
        repo_type="dataset",
        revision=lock["dataset"]["revision"],
        allow_patterns=patterns,
        local_dir=str(destination),
    )
    for path in destination.rglob("*"):
        if path.is_file() and ".cache" not in path.parts:
            path.chmod(0o444)
    return destination
