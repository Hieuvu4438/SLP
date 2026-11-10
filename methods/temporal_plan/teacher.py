"""Training-only teacher plans. This module is never called by text-only generation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .normalization import FeatureNormalizer
from .plans import PlanBatch
from .segments import PartitionDistribution, SegmentCodec, TeacherConfig, phase_samples, segment_scores
from .tensors import check_sequence


def plan_phases(features: Tensor, durations: list[int], phase_steps: int = 8) -> Tensor:
    if sum(durations) != len(features) or not durations or min(durations) < 1:
        raise ValueError("partition must cover exactly the unpadded reference latent")
    flat = features.flatten(1)
    phases, start = [], 0
    for duration in durations:
        phases.append(phase_samples(flat[start:start+duration], phase_steps))
        start += duration
    return torch.stack(phases)


@dataclass
class TeacherTargets:
    plan: PlanBatch
    source_indices: Tensor


@torch.no_grad()
def teacher_partitions(features: Tensor, lengths: Tensor, config: TeacherConfig, *,
                       samples: int = 2, mode: str = "sample",
                       generator: torch.Generator | None = None,
                       score_cache=None, source_ids=None):
    check_sequence(features, lengths)
    if samples < 1 or mode not in ("sample", "map", "uniform"):
        raise ValueError("invalid number of plans or teacher mode")
    for b, length in enumerate(lengths.tolist()):
        valid_features = features[b, :length]
        scores = (segment_scores(valid_features, config) if score_cache is None else
                  score_cache.get(valid_features, config,
                                  None if source_ids is None else source_ids[b]))
        distribution = PartitionDistribution(scores, config.temperature)
        fixed_durations = distribution.mode() if mode != "sample" else None
        for _ in range(samples):
            if mode == "sample":
                durations = distribution.sample(generator)
            else:
                durations = list(fixed_durations)
                if mode == "uniform":
                    # Match K to MAP instead of changing count/capacity silently.
                    k = len(durations)
                    boundaries = torch.linspace(0, length, k+1).round().long()
                    durations = boundaries.diff().tolist()
            yield b, durations, plan_phases(valid_features, durations, config.phase_steps)


@torch.no_grad()
def teacher_targets(features: Tensor, lengths: Tensor, segment_codec: SegmentCodec,
                    content_normalizer: FeatureNormalizer, config: TeacherConfig, *,
                    samples: int = 2, mode: str = "sample",
                    generator: torch.Generator | None = None,
                    score_cache=None, source_ids=None) -> TeacherTargets:
    if segment_codec.training or any(p.requires_grad for p in segment_codec.parameters()):
        raise RuntimeError("teacher content encoder must be frozen before creating plan targets")
    codes, durations, sources = [], [], []
    for source, ds, phases in teacher_partitions(features, lengths, config, samples=samples,
                                                mode=mode, generator=generator,
                                                score_cache=score_cache, source_ids=source_ids):
        codes.append(content_normalizer(segment_codec.encode(phases)))
        durations.append(ds)
        sources.append(source)
    return TeacherTargets(PlanBatch.from_sequences(codes, durations),
                          torch.tensor(sources, device=features.device, dtype=torch.long))
