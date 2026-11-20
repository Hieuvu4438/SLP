"""Atomic checkpoint persistence and explicit compatibility checks."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch


def save_atomic(state, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@lru_cache(maxsize=32)
def _hash(path, size, modified):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    st = path.stat()
    return _hash(path, st.st_size, st.st_mtime_ns)


def read_checkpoint(path, signature, legacy=False):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    stored = state.get("signature")
    if stored is None:
        if not legacy:
            raise ValueError(f"Checkpoint {path} has no signature. Legacy reuse must be explicit.")
    elif stored != signature:
        mismatches = sorted(k for k in set(stored) | set(signature) if stored.get(k) != signature.get(k))
        raise ValueError(f"Incompatible checkpoint {path}. Differing fields: {', '.join(mismatches)}")
    return state


def write_json(path, content):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)
