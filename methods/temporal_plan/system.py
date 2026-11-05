"""Complete neural method and stage objectives; data/optimization are separate."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import math

import torch
from torch import Tensor, nn

from .codec import MotionCodec
from .conditioning import temporal_condition
from .denoiser import JointDenoiser
from .diffusion import CosineDiffusion
from .geometry import rotation6d_to_matrix
from .losses import LossWeights, motion_reconstruction, posterior_kl
from .normalization import FeatureNormalizer
from .planner import TextToPlan
from .plans import PlanBatch
from .representation import MotionRepresentation
from .segments import SegmentCodec, TeacherConfig
from .teacher import teacher_partitions, teacher_targets
from .tensors import check_sequence, valid_mask, zero_padding
from .text import TextConditioner

STAGES = ("codec", "segment", "planner", "generator", "exposure")


@dataclass(frozen=True)
class ModelConfig:
    max_latents: int  # ceil(max TRAIN frames / 2), not a test/reference duration
    latent_width: int = 64
    codec_hidden: int = 128
    content_width: int = 128
    segment_hidden: int = 512
    hidden: int = 512
    heads: int = 8
    planner_layers: int = 4
    denoiser_layers: int = 8
    mixtures: int = 4
    stop_threshold: float = .5
    dropout: float = .1
    diffusion_steps: int = 1000
    sampling_steps: int = 50
    plan_samples: int = 2
    teacher_mode: str = "sample"
    plan_dropout: float = .1
    exposure_probability: float = .25
    overlap: float = 2.
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    losses: LossWeights = field(default_factory=LossWeights)

    def __post_init__(self):
        if self.max_latents < 1 or self.plan_samples < 1:
            raise ValueError("training-derived cap and plan count must be positive")
        if not 1 <= self.sampling_steps <= self.diffusion_steps:
            raise ValueError("invalid DDIM step count")
        if not 0 <= self.exposure_probability <= 1 or not 0 <= self.plan_dropout <= 1:
            raise ValueError("invalid dropout/exposure probability")
        if self.teacher_mode not in ("sample", "map", "uniform"):
            raise ValueError("unknown teacher mode")
        if not math.isfinite(self.overlap) or self.overlap < 0:
            raise ValueError("overlap must be finite and nonnegative")
        if not math.isfinite(self.stop_threshold) or not 0 < self.stop_threshold < 1:
            raise ValueError("stop threshold must lie strictly inside (0,1)")


@dataclass
class GeneratedMotion:
    raw133: Tensor
    frame_lengths: Tensor
    plans: PlanBatch
    stop_reasons: tuple[str, ...]
    rotation_fallbacks: Tensor


class TemporalPlanSystem(nn.Module):
    def __init__(self, config: ModelConfig, text: TextConditioner, rotation_offsets: Tensor):
        super().__init__()
        self.config, self.text = config, text
        self.representation = MotionRepresentation(rotation_offsets)
        self.motion_norm = FeatureNormalizer(256)
        self.latent_norm = FeatureNormalizer(4 * config.latent_width)
        self.content_norm = FeatureNormalizer(config.content_width)
        self.codec = MotionCodec(config.latent_width, config.codec_hidden)
        self.segment_codec = SegmentCodec(4*config.latent_width, config.content_width,
                                          config.segment_hidden, config.teacher.phase_steps)
        self.planner = TextToPlan(config.content_width, text.output_width, config.hidden,
                                  config.heads, config.planner_layers, config.mixtures,
                                  config.teacher.max_duration, config.max_latents, config.dropout,
                                  config.stop_threshold)
        self.denoiser = JointDenoiser(config.latent_width, config.content_width, text.output_width,
                                      config.hidden, config.heads, config.denoiser_layers,
                                      config.dropout)
        self.diffusion = CosineDiffusion(config.diffusion_steps)
        self.register_buffer("completed_stages", torch.zeros(len(STAGES), dtype=torch.bool))
        self.stage = "codec"
        self.score_cache = None  # CPU cache is not neural/checkpoint state
        self.train(False)

    def set_stage(self, stage: str):
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        index = STAGES.index(stage)
        if index and not bool(self.completed_stages[:index].all()):
            raise RuntimeError("complete preceding stages before training this stage")
        self.stage = stage
        for p in self.parameters():
            p.grad = None  # stale gradients must not survive stage/optimizer transitions
        return self.train(True)

    def mark_stage_completed(self):
        self.completed_stages[STAGES.index(self.stage)] = True

    def train(self, mode: bool = True):
        super().train(mode)
        if not hasattr(self, "stage"):
            return self
        self.requires_grad_(False)
        for child in self.children():
            child.eval()
        if self.stage == "codec":
            selected = (self.codec,)
        elif self.stage == "segment":
            selected = (self.segment_codec,)
        elif self.stage == "planner":
            selected = (self.planner, self.text.projection, self.text.language)
        else:
            selected = (self.denoiser,)
        for module in selected:
            module.requires_grad_(True)
            module.train(mode)
        self.text.encoder.requires_grad_(False).eval()
        return self

    def _prepare(self, raw: Tensor, lengths: Tensor):
        check_sequence(raw, lengths, 133)
        raw = raw[:, :int(lengths.max())]
        physical = self.representation.encode(raw, lengths)
        return zero_padding(self.motion_norm(physical), lengths), physical

    @torch.no_grad()
    def fit_motion_statistics(self, batches: Iterable[tuple[Tensor, Tensor]]):
        self.motion_norm.fit((self.representation.encode(raw, lengths), lengths)
                              for raw, lengths in batches)

    @torch.no_grad()
    def fit_latent_statistics(self, batches: Iterable[tuple[Tensor, Tensor]]):
        if not bool(self.completed_stages[0]):
            raise RuntimeError("train/freeze motion codec before fitting latent statistics")
        def means():
            for raw, lengths in batches:
                normalized, _ = self._prepare(raw, lengths)
                posterior = self.codec.encode(normalized, lengths)
                yield posterior.mean.flatten(2), posterior.lengths
        self.latent_norm.fit(means())

    def _normalize_latents(self, latents: Tensor, lengths: Tensor) -> Tensor:
        return zero_padding(self.latent_norm(latents.flatten(2)).reshape_as(latents), lengths)

    def _inverse_latents(self, latents: Tensor) -> Tensor:
        return self.latent_norm.inverse(latents.flatten(2)).reshape_as(latents)

    @torch.no_grad()
    def reference_latents(self, raw: Tensor, lengths: Tensor):
        normalized, _ = self._prepare(raw, lengths)
        posterior = self.codec.encode(normalized, lengths)
        return self._normalize_latents(posterior.mean, posterior.lengths), posterior.lengths

    @torch.no_grad()
    def fit_content_statistics(self, batches: Iterable[tuple[Tensor, Tensor]], *,
                               generator: torch.Generator | None = None):
        if not bool(self.completed_stages[:2].all()):
            raise RuntimeError("train motion and segment codecs before fitting content statistics")
        def codes():
            for raw, lengths in batches:
                features, latent_lengths = self.reference_latents(raw, lengths)
                for _, _, phases in teacher_partitions(
                    features, latent_lengths, self.config.teacher,
                    samples=self.config.plan_samples, mode="sample",
                    generator=generator, score_cache=self.score_cache,
                ):
                    c = self.segment_codec.encode(phases)
                    yield c[None], torch.tensor([len(c)], device=c.device)
        self.content_norm.fit(codes())

    def codec_loss(self, raw: Tensor, lengths: Tensor, *, fk: nn.Module | None = None,
                   generator: torch.Generator | None = None) -> dict[str, Tensor]:
        if self.stage != "codec":
            raise RuntimeError("codec objective requested in the wrong stage")
        normalized, physical = self._prepare(raw, lengths)
        reconstructed, posterior = self.codec(normalized, lengths, generator=generator)
        unnormalized = self.motion_norm.inverse(reconstructed)
        # Equal weight per training example, not per padded frame or per clip duration.
        # This also makes microbatch gradient accumulation well-defined.
        individual = [
            motion_reconstruction(unnormalized[b:b+1, :n], physical[b:b+1, :n],
                                  lengths[b:b+1], weights=self.config.losses, fk=fk,
                                  rotation_offsets=self.representation.rotation_offsets)
            for b, n in enumerate(lengths.tolist())
        ]
        losses = {key: torch.stack([item[key] for item in individual]).mean()
                  for key in individual[0]}
        kl = posterior_kl(posterior)
        return {**losses, "kl": kl, "total": losses["total"] + self.config.losses.kl * kl}

    def segment_loss(self, raw: Tensor, lengths: Tensor, *,
                     generator: torch.Generator | None = None,
                     source_ids=None) -> dict[str, Tensor]:
        if self.stage != "segment":
            raise RuntimeError("segment objective requested in the wrong stage")
        features, latent_lengths = self.reference_latents(raw, lengths)
        groups = [p for _, _, p in teacher_partitions(
            features, latent_lengths, self.config.teacher,
            samples=self.config.plan_samples, mode="sample",
            generator=generator, score_cache=self.score_cache, source_ids=source_ids,
        )]
        phases = torch.cat(groups)
        reconstructed, _ = self.segment_codec(phases)
        segment_error = (reconstructed-phases).square().flatten(1).mean(-1)
        loss = torch.stack([item.mean() for item in segment_error.split(
            [len(group) for group in groups]
        )]).mean()
        return dict(total=loss, reconstruction=loss)

    def _targets(self, features: Tensor, lengths: Tensor, generator, source_ids=None):
        return teacher_targets(features, lengths, self.segment_codec, self.content_norm,
                                self.config.teacher, samples=self.config.plan_samples,
                                mode=self.config.teacher_mode, generator=generator,
                                score_cache=self.score_cache, source_ids=source_ids)

    def planner_loss(self, raw: Tensor, lengths: Tensor, input_ids: Tensor,
                     attention_mask: Tensor, language_ids: Tensor, *, generator=None,
                     source_ids=None):
        if self.stage != "planner":
            raise RuntimeError("planner objective requested in the wrong stage")
        features, latent_lengths = self.reference_latents(raw, lengths)
        targets = self._targets(features, latent_lengths, generator, source_ids)
        memory, valid = self.text(input_ids, attention_mask, language_ids)
        return self.planner.loss(memory[targets.source_indices], valid[targets.source_indices],
                                 targets.plan)

    def generator_loss(self, raw: Tensor, lengths: Tensor, input_ids: Tensor,
                       attention_mask: Tensor, language_ids: Tensor, *, fk=None, generator=None,
                       source_ids=None):
        if self.stage not in ("generator", "exposure"):
            raise RuntimeError("generator objective requested in the wrong stage")
        features, latent_lengths = self.reference_latents(raw, lengths)
        targets = self._targets(features, latent_lengths, generator, source_ids)
        memory, text_valid = self.text(input_ids, attention_mask, language_ids)
        source = targets.source_indices
        plan = targets.plan
        exposure = torch.zeros(len(raw), device=raw.device, dtype=torch.bool)
        if self.stage == "exposure":
            exposure = torch.rand(len(raw), device=raw.device, generator=generator) < (
                self.config.exposure_probability
            )
            if bool(exposure.any()):
                with torch.no_grad():
                    predicted = self.planner.generate(memory, text_valid, generator=generator).plan
                contents, durations = [], []
                for index, b in enumerate(source.tolist()):
                    candidate = predicted if bool(exposure[b]) else plan
                    row = b if bool(exposure[b]) else index
                    k = int(candidate.lengths[row])
                    contents.append(candidate.contents[row, :k])
                    durations.append(candidate.durations[row, :k].tolist())
                plan = PlanBatch.from_sequences(contents, durations)
        condition = temporal_condition(plan, canvas_lengths=latent_lengths[source],
                                        rescale=self.stage == "exposure",
                                        overlap=self.config.overlap)
        denoise = self.diffusion.training_loss(
            self.denoiser, features[source], latent_lengths[source], memory[source],
            text_valid[source], condition.features, generator=generator,
            plan_dropout=self.config.plan_dropout,
        )
        decoded = self.codec.decode(self._inverse_latents(denoise.clean_prediction),
                                     latent_lengths[source], lengths[source])
        physical = self.motion_norm.inverse(decoded)
        target_physical = self.representation.encode(raw[:, :decoded.shape[1]], lengths)
        auxiliaries = []
        for i, b in enumerate(source.tolist()):
            n = int(lengths[b])
            losses = motion_reconstruction(
                physical[i:i+1, :n], target_physical[b:b+1, :n], lengths[b:b+1],
                weights=self.config.losses, fk=fk,
                rotation_offsets=self.representation.rotation_offsets,
            )
            # Generator spec uses decoded FK/velocity/expression, not an extra rotation term.
            w = self.config.losses
            auxiliaries.append(denoise.alpha_bar[i] *
                               (w.fk*losses["fk"] + w.velocity*losses["velocity"] +
                                w.expression*losses["expression"]))
        auxiliary = torch.stack(auxiliaries).mean()
        return dict(total=denoise.loss+auxiliary, denoising=denoise.loss, decoded=auxiliary,
                    exposure_fraction=exposure.float().mean(),
                    subcell_units=condition.subcell_units.float().mean())

    @torch.no_grad()
    def generate(self, input_ids: Tensor, attention_mask: Tensor, language_ids: Tensor, *,
                 generator: torch.Generator | None = None) -> GeneratedMotion:
        if self.training:
            raise RuntimeError("generation requires eval()")
        if not bool(self.completed_stages.all()):
            raise RuntimeError("all training stages or a completed checkpoint are required")
        memory, valid = self.text(input_ids, attention_mask, language_ids)
        generated = self.planner.generate(memory, valid, generator=generator)
        condition = temporal_condition(generated.plan, overlap=self.config.overlap)
        latent_lengths = generated.plan.latent_lengths
        latents = self.diffusion.sample(
            self.denoiser, latent_lengths, memory, valid, condition.features,
            latent_width=self.config.latent_width, sampling_steps=self.config.sampling_steps,
            generator=generator,
        )
        frame_lengths = 2*latent_lengths
        decoded = self.codec.decode(self._inverse_latents(latents), latent_lengths)
        physical = self.motion_norm.inverse(decoded)
        raw = self.representation.decode(physical, frame_lengths)
        _, fallback = rotation6d_to_matrix(physical[..., :246].reshape(
            *physical.shape[:-1], 41, 6), return_fallback=True)
        fallback = fallback & valid_mask(frame_lengths, raw.shape[1])[..., None]
        return GeneratedMotion(raw, frame_lengths, generated.plan, generated.stop_reasons,
                                fallback.sum((1, 2)))
