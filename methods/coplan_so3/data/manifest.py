"""A manifest is a checked contract, never inferred from tensor width."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class MotionManifest:
    joint_names: tuple[str, ...]
    smplx_joint_indices: tuple[int, ...]
    body: tuple[int, ...]
    left_hand: tuple[int, ...]
    right_hand: tuple[int, ...]
    fingers_left: tuple[tuple[int, ...], ...]
    fingers_right: tuple[tuple[int, ...], ...]
    expression_dim: int
    fps: float
    protocol: str
    extraction_version: str
    annotation_version: str
    shape_policy: str
    hand_convention: str
    fixed_parameters: str
    asset_checksums: dict[str, str]

    def __post_init__(self):
        j = len(self.joint_names)
        if not j or len(set(self.joint_names)) != j:
            raise ValueError("Joint names must be unique and nonempty")
        if len(self.smplx_joint_indices) != j or len(set(self.smplx_joint_indices)) != j:
            raise ValueError("Explicit unique SMPL-X mapping required")
        if any(i < 0 or i >= 55 for i in self.smplx_joint_indices):
            raise ValueError("SMPL-X indices must be in [0,55)")
        if not self.body or not self.left_hand or not self.right_hand:
            raise ValueError("Body and both hand branches must be nonempty")
        parts = self.body + self.left_hand + self.right_hand
        if sorted(parts) != list(range(j)):
            raise ValueError("Parts must partition the motion joints")
        for hand, fingers in (
            (self.left_hand, self.fingers_left),
            (self.right_hand, self.fingers_right),
        ):
            if sorted(i for f in fingers for i in f) != sorted(hand) or any(not f for f in fingers):
                raise ValueError("Finger groups must partition each hand")
        if self.expression_dim < 1 or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Positive expression dimension and fps required")
        if any(
            not getattr(self, k)
            for k in (
                "protocol",
                "extraction_version",
                "annotation_version",
                "shape_policy",
                "hand_convention",
                "fixed_parameters",
            )
        ):
            raise ValueError("Provenance and conventions must be explicit")

    @property
    def joints(self):
        return len(self.joint_names)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        for k in ("joint_names", "smplx_joint_indices", "body", "left_hand", "right_hand"):
            d[k] = tuple(d[k])
        for k in ("fingers_left", "fingers_right"):
            d[k] = tuple(tuple(f) for f in d[k])
        return cls(**d)

    def verify_assets(self, root):
        for name, expected in self.asset_checksums.items():
            if file_sha256(Path(root) / name) != expected:
                raise ValueError(f"Asset checksum mismatch: {name}")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
