import json
import platform
import subprocess
import sys
import time
import tarfile
from pathlib import Path

import torch
import yaml


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().mean().item()
    return value


class JsonLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **values):
        with self.path.open("a") as stream:
            stream.write(
                json.dumps(
                    {"wall_time": time.time(), **{k: json_value(v) for k, v in values.items()}},
                    allow_nan=False,
                )
                + "\n"
            )


def environment():
    return dict(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        tf32=torch.backends.cuda.matmul.allow_tf32,
        sdpa=hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    )


def record_run(directory, config, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (directory / "environment.json").write_text(json.dumps(environment(), indent=2))
    (directory / "command.txt").write_text(" ".join(sys.argv) + "\n")
    from cprflow.data.cache import file_hash

    method_root = Path(__file__).resolve().parents[3]
    code_files = [
        p
        for folder in ("src", "configs", "scripts")
        for p in (method_root / folder).rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml", ".sh")
    ]
    code_files += [
        p
        for p in method_root.iterdir()
        if p.is_file() and p.suffix in (".py", ".toml", ".yaml", ".md", ".txt")
    ]
    inventory = {str(p.relative_to(method_root)): file_hash(p) for p in sorted(code_files)}
    (directory / "source_inventory.json").write_text(json.dumps(inventory, indent=2))
    with tarfile.open(directory / "source_snapshot.tar.gz", "w:gz") as archive:
        for p in code_files:
            archive.add(p, arcname=str(p.relative_to(method_root)), recursive=False)
    for name, args in [
        ("git_commit.txt", ["git", "rev-parse", "HEAD"]),
        ("git_dirty.diff", ["git", "diff", "HEAD"]),
        ("git_status.txt", ["git", "status", "--short"]),
    ]:
        p = subprocess.run(args, capture_output=True, text=True)
        (directory / name).write_text(p.stdout + p.stderr)
