"""Content-addressed numeric caches. No pickle in processed/cache artifacts."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from cprflow.config import canonical_json


def digest(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(kind, **dependencies):
    return digest({"kind": kind, "version": "cprflow-impl-v1", **dependencies})


def save_numeric(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from cprflow.jobs.storage import check_free_space

    check_free_space(path.parent)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    if any(v.dtype.hasobject for v in arrays.values()):
        raise ValueError("Object arrays forbidden")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


class NumericCache:
    def __init__(self, root, dependencies):
        self.key = digest(dependencies)
        self.root = Path(root) / self.key
        self.root.mkdir(parents=True, exist_ok=True)
        meta = self.root / "dependencies.json"
        if meta.exists() and json.loads(meta.read_text()) != dependencies:
            raise ValueError("Cache provenance mismatch")
        meta.write_text(canonical_json(dependencies))

    def path(self, uid):
        return self.root / f"{digest(uid)}.npz"

    def put(self, uid, **arrays):
        save_numeric(self.path(uid), **arrays)

    def get(self, uid):
        with np.load(self.path(uid), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
