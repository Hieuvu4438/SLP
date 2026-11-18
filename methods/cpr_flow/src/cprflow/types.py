"""Public contracts. Generation requests deliberately contain no target fields."""

from dataclasses import dataclass

import numpy as np
import torch

PART_DIMS = (60, 90, 90, 56)
PART_NAMES = ("body", "left", "right", "face")
LANGUAGES = {"CSL": 0, "ASL": 1, "DGS": 2}
DATASETS = {
    "csl_daily": ("CSL", "zh", "CSL-Daily_reopt_"),
    "how2sign": ("ASL", "en", "How2Sign_reopt_"),
    "phoenix2014t": ("DGS", "de", "PHOENIX-2014T_reopt_"),
}
PROFILE = "canonical296_nativeleft_row6d_v1"


@dataclass
class MotionRecord:
    uid: str
    dataset_id: str
    sign_language_id: int
    spoken_language: str
    source_clip_id: str
    source_video_id: str | None
    signer_id: str | None
    split: str
    text: str
    motion: np.ndarray
    fps: float
    source_fps: float
    source_frames: int
    timestamps: np.ndarray
    preprocessing_hash: str
    asset_profile_hash: str


@dataclass
class MotionBatch:
    x: torch.Tensor
    lengths: torch.Tensor
    frame_mask: torch.Tensor
    timestamps: torch.Tensor
    language_id: torch.Tensor
    dataset_id: list[str]
    text: list[str]
    uid: list[str]


@dataclass
class TextBatch:
    hidden: torch.Tensor
    mask: torch.Tensor


@dataclass
class LatentBatch:
    z: torch.Tensor
    r: torch.Tensor | None
    latent_lengths: torch.Tensor
    latent_mask: torch.Tensor
    token_times: torch.Tensor


@dataclass(frozen=True)
class GenerationRequest:
    text: str
    sign_language: str
    seed: int
    duration_seconds: float | None = None

    def __post_init__(self):
        if self.sign_language not in LANGUAGES or not self.text.strip():
            raise ValueError("Generation requires nonempty text and CSL/ASL/DGS")
        if self.duration_seconds is not None and (
            not np.isfinite(self.duration_seconds) or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be finite and positive")


@dataclass
class GenerationResult:
    canonical_motion: np.ndarray
    fps: float
    predicted_frames: int
    duration_source: str
    seed: int
    metadata: dict
