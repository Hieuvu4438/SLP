"""Strict, portable job manifest; commands are argv arrays, never shell strings."""

import json
from pathlib import Path

from cprflow.data.cache import digest, file_hash


def source_digest(root):
    root = Path(root)
    return digest(
        {
            str(p.relative_to(root)): file_hash(p)
            for p in sorted((root / "src/cprflow").rglob("*"))
            if p.suffix in (".py", ".yaml")
        }
    )


def validate_plan(plan):
    allowed = {
        "version",
        "id",
        "root",
        "state_dir",
        "global_lock",
        "max_concurrent",
        "poll_seconds",
        "disk",
        "environment",
        "jobs",
        "source_digest",
        "config_hashes",
        "scope",
        "seeds",
    }
    if set(plan) - allowed or plan.get("version") != 1:
        raise ValueError("Unknown plan fields/version")
    if type(plan["max_concurrent"]) is not int or not 1 <= plan["max_concurrent"] <= 2:
        raise ValueError("Hard limit: at most two concurrent payload processes")
    if plan["poll_seconds"] < 0.05:
        raise ValueError("Poll interval too short")
    names = {j["id"] for j in plan["jobs"]}
    if len(names) != len(plan["jobs"]):
        raise ValueError("Duplicate job ID")
    for job in plan["jobs"]:
        if set(job) - {
            "id",
            "argv",
            "deps",
            "gates",
            "outputs",
            "slots",
            "gpu_gib",
            "disk_gib",
            "locks",
            "max_attempts",
            "retry_seconds",
            "resume_dir",
            "warm_start",
            "priority",
        }:
            raise ValueError(f"Unknown job fields: {job['id']}")
        if not job["id"].replace(".", "_").replace("-", "_").isidentifier():
            raise ValueError("Job IDs must be safe identifiers")
        if not set(job["deps"]) <= names or job["id"] in job["deps"]:
            raise ValueError("Invalid dependency")
        if (
            not 1 <= job["slots"] <= plan["max_concurrent"]
            or job["gpu_gib"] < 0
            or job["disk_gib"] < 0
        ):
            raise ValueError("Invalid resource request")
        if job["max_attempts"] < 1 or job["retry_seconds"] < 0:
            raise ValueError("Invalid retry policy")
        if job["argv"] and (
            not isinstance(job["argv"], list) or not all(isinstance(a, str) for a in job["argv"])
        ):
            raise ValueError("Command must be an argv list")
        if job["argv"] and (len(job["argv"]) < 3 or job["argv"][1] not in ("-m", "-c")):
            raise ValueError("Payloads run Python modules/snippets in one worker process")
    visited = set()
    visiting = set()
    by_id = {j["id"]: j for j in plan["jobs"]}

    def visit(name):
        if name in visiting:
            raise ValueError("Dependency cycle")
        if name in visited:
            return
        visiting.add(name)
        for parent in by_id[name]["deps"]:
            visit(parent)
        visiting.remove(name)
        visited.add(name)

    for name in names:
        visit(name)
    return plan


def load_plan(path):
    return validate_plan(json.loads(Path(path).read_text()))


def atomic_json(path, data):
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)
