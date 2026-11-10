from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class RaceConfig:
    code_dim: int = 512
    text_dim: int = 1024
    width: int = 256
    heads: int = 8
    layers: int = 6
    factor: int = 4
    languages: int = 3
    dropout: float = 0.1
    text_dropout: float = 0.1
    block_size: int = 16
    blend_frames: int = 4
    selector_width: int = 128
    flow_steps: int = 8
    joint_weight: float = 1.0
    relation_weight: float = 1.0
    rotation_weight: float = 0.1
    velocity_weight: float = 0.1
    boundary_weight: float = 0.1

    def __post_init__(self):
        for key in (
            "code_dim",
            "text_dim",
            "width",
            "heads",
            "layers",
            "factor",
            "languages",
            "block_size",
            "selector_width",
            "flow_steps",
        ):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if (
            not isinstance(self.blend_frames, int)
            or isinstance(self.blend_frames, bool)
            or not 0 <= 2 * self.blend_frames <= self.block_size
        ):
            raise ValueError("blend_frames must be in [0, block_size/2]")
        for key in ("dropout", "text_dropout"):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f"{key} must be in [0,1)")
        for key in (
            "joint_weight",
            "relation_weight",
            "rotation_weight",
            "velocity_weight",
            "boundary_weight",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")

    @classmethod
    def from_dict(cls, data):
        allowed = {f.name for f in fields(cls)}
        if set(data) - allowed:
            raise ValueError(f"Unknown model settings: {set(data) - allowed}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
