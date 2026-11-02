"""Capture code provenance so every result can be traced to exact code."""

from __future__ import annotations

import subprocess

from slp.paths import PROJECT_ROOT


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_info() -> dict:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD").strip() or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip() or None,
        "dirty": bool(status.strip()),
        "status": status,
    }


def git_diff() -> str:
    return _git("diff", "HEAD")
