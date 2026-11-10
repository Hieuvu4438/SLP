from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from .data import InferenceBatch
from .flow import sample_euler
from .schema import manual_to_6d, restore_pose
from .selector import blend_candidates
from .geometry import geometry_signature


@dataclass
class Prediction:
    poses: torch.Tensor
    lengths: torch.Tensor
    base: torch.Tensor
    candidate: torch.Tensor
    weights: torch.Tensor
    delta: torch.Tensor | None
    nfe: int
    sample_ids: tuple[str, ...]


class RacePipeline(nn.Module):
    def __init__(self, decoder, normalizer, geometry, selector=None):
        super().__init__()
        self.decoder, self.normalizer, self.geometry, self.selector = (
            decoder,
            normalizer,
            geometry,
            selector,
        )

    @torch.no_grad()
    def predict(
        self,
        inputs: InferenceBatch,
        *,
        policy="coupled",
        seed=0,
        steps=None,
        guidance=1.0,
        use_text=None,
        coupled=None,
    ):
        if not isinstance(inputs, InferenceBatch):
            raise TypeError("predict accepts only InferenceBatch (no reference targets)")
        inputs.validate(self.decoder.cfg)
        signature = getattr(self.decoder, "expected_base_signature", None)
        if signature is not None and signature != inputs.base_signature:
            raise ValueError(
                "Inference inputs use different tokenizer/generator/statistics than training"
            )
        expected_geometry = getattr(self.decoder, "expected_geometry_signature", None)
        if expected_geometry is not None and expected_geometry != geometry_signature(self.geometry):
            raise ValueError("Inference geometry/shape/hand-mean codec differs from training")
        if self.training:
            raise ValueError("Set pipeline.eval() before inference")
        cfg = self.decoder.cfg
        use_text = getattr(self.decoder, "default_use_text", True) if use_text is None else use_text
        coupled = getattr(self.decoder, "default_coupled", True) if coupled is None else coupled
        if policy in ("coupled", "sentence") and self.selector is not None:
            expected = getattr(self.selector, "sampling_policy", None)
            actual = dict(
                candidate_steps=cfg.flow_steps if steps is None else steps,
                candidate_use_text=use_text,
                candidate_coupled=coupled,
            )
            if expected is not None and (actual != expected or guidance != 1.0):
                raise ValueError("Sampling policy differs from selector training/calibration")
        base = manual_to_6d(inputs.baseline, self.geometry.manual_mean)
        base_rel = self.geometry(base).relations
        cond = dict(
            codes=inputs.codes,
            token_lengths=inputs.token_lengths,
            text=inputs.text,
            text_mask=inputs.text_mask,
            lengths=inputs.lengths,
            language=inputs.language,
            relations=base_rel,
        )
        gen = torch.Generator(device=base.device).manual_seed(seed)
        noise = torch.zeros_like(base)
        for i, sample_id in enumerate(inputs.sample_ids):
            stable = int.from_bytes(
                hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()[:8], "little"
            )
            local = torch.Generator(device=base.device).manual_seed(stable)
            n = int(inputs.lengths[i])
            noise[i, :n] = torch.randn(
                (n, 240), device=base.device, dtype=base.dtype, generator=local
            )
        if policy == "base":
            candidate, nfe = base, 0
        else:
            candidate, nfe = sample_euler(
                self.decoder,
                noise,
                cfg.flow_steps if steps is None else steps,
                cond,
                guidance=guidance,
                use_text=use_text,
                coupled=coupled,
            )
            candidate = self.normalizer.inverse(candidate)
        # SO(3) projection and exact jaw/expression preservation happen for every candidate.
        candidate_pose = restore_pose(candidate, inputs.baseline, self.geometry.manual_mean)
        candidate = manual_to_6d(candidate_pose, self.geometry.manual_mean)
        nblocks = (base.shape[1] + cfg.block_size - 1) // cfg.block_size
        delta = None
        if policy in ("coupled", "sentence"):
            if self.selector is None:
                raise ValueError("A trained selector checkpoint is required for this policy")
            if self.selector.whole_sentence != (policy == "sentence"):
                raise ValueError("Selector selection unit differs from its training/calibration")
            output = self.selector(
                self.normalizer(base),
                self.normalizer(candidate),
                base_rel,
                self.geometry(candidate).relations,
                inputs.text,
                inputs.text_mask,
                inputs.lengths,
                inputs.language,
                codes=inputs.codes,
                whole_sentence=policy == "sentence",
            )
            delta = output["delta"]
            if policy == "sentence":
                delta = delta.expand(-1, nblocks)
            scores = delta
        elif policy == "base":
            scores = base.new_full((len(base), nblocks), -1)
        elif policy == "flow":
            scores = base.new_ones(len(base), nblocks)
        elif policy == "random":
            scores = torch.rand(len(base), nblocks, device=base.device, generator=gen) - 0.5
        else:
            raise ValueError(f"Unsupported inference policy {policy!r}")
        mixed, weights = blend_candidates(
            base, candidate, scores, inputs.lengths, cfg.block_size, cfg.blend_frames
        )
        pose = restore_pose(mixed, inputs.baseline, self.geometry.manual_mean)
        pose = torch.where((weights == 0)[..., None], inputs.baseline, pose)
        valid = torch.arange(pose.shape[1], device=pose.device)[None] < inputs.lengths[:, None]
        pose = pose.masked_fill(~valid[..., None], 0)
        if not torch.isfinite(pose).all():
            raise FloatingPointError("Nonfinite generated motion")
        return Prediction(
            pose,
            inputs.lengths.clone(),
            inputs.baseline,
            candidate_pose,
            weights,
            delta,
            nfe,
            inputs.sample_ids,
        )
