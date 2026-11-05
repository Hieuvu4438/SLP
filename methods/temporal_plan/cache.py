"""Bounded CPU/disk cache of deterministic teacher scores, never learned targets/state."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .segments import TeacherConfig, segment_scores


class TeacherScoreCache:
    def __init__(self, directory: str | Path | None = None, max_entries: int = 64):
        if max_entries < 0:
            raise ValueError("cache capacity cannot be negative")
        self.directory = None if directory is None else Path(directory)
        self.max_entries = max_entries
        self.memory = OrderedDict()
        self.scoring_code = hashlib.sha256(
            Path(__file__).with_name("segments.py").read_bytes()
        ).hexdigest()

    @torch.no_grad()
    def get(self, features, config: TeacherConfig, source_id: str | None = None):
        cpu = features.detach().cpu().contiguous()
        metadata = {"format": 1, "config": asdict(config), "shape": list(cpu.shape),
                    "dtype": str(cpu.dtype), "scoring_code_sha256": self.scoring_code,
                    "torch_version": str(torch.__version__)}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(cpu.numpy().tobytes())
        key = digest.hexdigest()
        if key in self.memory:
            scores = self.memory.pop(key)
            self.memory[key] = scores
            return scores.to(features).clone()
        path = None if self.directory is None else self.directory / (key + ".pt")
        if path is not None and path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            scores = payload["scores"]
            if (payload.get("key") != key or scores.shape != (len(cpu), config.max_duration) or
                    torch.isnan(scores).any() or torch.isposinf(scores).any() or
                    hashlib.sha256(scores.contiguous().numpy().tobytes()).hexdigest() !=
                    payload.get("scores_sha256")):
                raise ValueError("invalid cached teacher scores")
        else:
            scores = segment_scores(cpu, config).detach().cpu()
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=key, suffix=".tmp")
                os.close(fd)
                try:
                    torch.save({"key": key, "scores": scores, "source_id": source_id,
                                "metadata": metadata,
                                "scores_sha256": hashlib.sha256(
                                    scores.contiguous().numpy().tobytes()).hexdigest()}, temporary)
                    os.replace(temporary, path)
                finally:
                    if Path(temporary).exists():
                        Path(temporary).unlink()
        if self.max_entries:
            self.memory[key] = scores
            while len(self.memory) > self.max_entries:
                self.memory.popitem(last=False)
        return scores.to(features).clone()
