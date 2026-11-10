"""Two-slot, atomic training recovery: model + optimizer + RNG + data cursor."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import math
import numpy as np
import torch
from .checkpoint import save_checkpoint, read_checkpoint


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def rng_state():
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "python": random.getstate(),
        "numpy": (numpy[0], torch.tensor(numpy[1].astype(np.int64)), *numpy[2:]),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype(np.uint32), *numpy[2:]))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class TrainingProgress:
    def __init__(self, directory):
        self.root = Path(directory) / "resume"
        self.pointer = self.root / "current.json"

    def exists(self):
        return self.pointer.exists()

    def save(self, model, tokenizer, runtime, optimizer, generator, metadata):
        self.root.mkdir(parents=True, exist_ok=True)
        previous = json.loads(self.pointer.read_text()) if self.pointer.exists() else {}
        slot = self.root / ("b" if previous.get("slot") == "a" else "a")
        slot.mkdir(exist_ok=True)
        save_checkpoint(model, slot / "model.pt", tokenizer=tokenizer, metadata=metadata)
        payload = {
            "version": 1,
            "runtime": runtime,
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "rng": rng_state(),
            "metadata": metadata,
        }
        torch.save(_cpu(payload), slot / "training.pt")
        best = self.root.parent / f"best_{model.stage}.pt"
        has_best = best.exists() and math.isfinite(runtime["best"])
        if has_best:
            shutil.copy2(best, slot / "best_model.pt")
        elif (slot / "best_model.pt").exists():
            (slot / "best_model.pt").unlink()
        record = {
            "best_sha256": digest(slot / "best_model.pt") if has_best else None,
            "slot": slot.name,
            "model_sha256": digest(slot / "model.pt"),
            "training_sha256": digest(slot / "training.pt"),
            "stage": model.stage,
            "optimizer_updates": runtime["updates"],
            "phase_start": runtime["phase_start"],
        }
        temporary = self.pointer.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        os.replace(temporary, self.pointer)  # old committed pair survives an interrupted save

    def load(self):
        record = json.loads(self.pointer.read_text())
        if record["slot"] not in ("a", "b"):
            raise ValueError("invalid recovery slot")
        slot = self.root / record["slot"]
        if (
            digest(slot / "model.pt") != record["model_sha256"]
            or digest(slot / "training.pt") != record["training_sha256"]
        ):
            raise ValueError("recovery checkpoint checksum mismatch")
        payload = torch.load(slot / "training.pt", map_location="cpu", weights_only=True)
        if payload["version"] != 1:
            raise ValueError("unsupported training recovery format")
        if record.get("best_sha256"):
            if digest(slot / "best_model.pt") != record["best_sha256"]:
                raise ValueError("best-checkpoint recovery checksum mismatch")
            payload["best_checkpoint_path"] = str(slot / "best_model.pt")
        return read_checkpoint(slot / "model.pt"), payload


class AtomicState:
    """Compact evaluator state with the same two-slot commit discipline."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.pointer = self.root / "current.json"

    def exists(self):
        return self.pointer.exists()

    def save(self, payload):
        self.root.mkdir(parents=True, exist_ok=True)
        old = json.loads(self.pointer.read_text()) if self.pointer.exists() else {}
        slot = "b.pt" if old.get("slot") == "a.pt" else "a.pt"
        path = self.root / slot
        torch.save(_cpu(payload), path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        record = {"slot": slot, "sha256": digest(path)}
        temporary = self.pointer.with_suffix(".tmp")
        temporary.write_text(json.dumps(record))
        os.replace(temporary, self.pointer)

    def load(self):
        record = json.loads(self.pointer.read_text())
        if (
            record["slot"] not in ("a.pt", "b.pt")
            or digest(self.root / record["slot"]) != record["sha256"]
        ):
            raise ValueError("invalid evaluator recovery checksum")
        return torch.load(self.root / record["slot"], map_location="cpu", weights_only=True)
