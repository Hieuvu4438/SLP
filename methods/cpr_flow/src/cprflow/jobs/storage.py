"""Storage admission checks; never remove raw data, shared caches, or foreign runs."""

import os
import shutil
from pathlib import Path


def check_free_space(path, min_gib=None):
    minimum = float(os.environ.get("CPRFLOW_MIN_FREE_GIB", "0")) if min_gib is None else min_gib
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum:
        raise OSError(f"Disk reserve breached: {free:.2f}GiB free, require {minimum:.2f}GiB")
    return free


def directory_bytes(paths):
    total = 0
    stack = [Path(p) for p in paths if Path(p).exists()]
    while stack:
        path = stack.pop()
        if path.is_symlink():
            continue
        if path.is_file():
            total += path.stat().st_size
            continue
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    return total
