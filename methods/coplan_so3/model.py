"""Complete A/B/C objectives and inference model; no optimization on construction."""

from __future__ import annotations

import math

import torch
from torch import nn

from .flow import SourceConfig, interpolate, plan_fm_loss, source
from .geometry.pullback import kfm_loss
from .losses import contrastive, reconstruction
from .models.components import mlp
from .models.part_temporal_transformer import DenseFlow
from .models.plan_encoder_decoder import PlanAutoencoder
from .models.plan_prior import LengthModel, PlanPrior


class CoPlanSO3(nn.Module):
    def __init__(
        self,
        manifest,
        text_encoder,
        features,
        *,
        plan_dim=256,
        plan_hidden=256,
        plan_layers=4,
        dense_hidden=512,
        dense_layers=8,
        heads=8,
        dropout=0.1,
        stride=4,
        posterior_sigma=0.05,
        source_config=None,
        auxiliary_weight=0.0,
        fps_by_language=None,
    ):
        super().__init__()
        if (
            not math.isfinite(posterior_sigma)
            or not math.isfinite(auxiliary_weight)
            or posterior_sigma <= 0
            or auxiliary_weight < 0
        ):
            raise ValueError("Positive posterior sigma and nonnegative auxiliary weight required")
        if (
            min(plan_dim, plan_hidden, plan_layers, dense_hidden, dense_layers, heads, stride) < 1
            or plan_hidden % 2
            or dense_hidden % 2
            or plan_hidden % heads
            or dense_hidden % heads
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                "Positive dimensions/stride/layers, even hidden widths divisible by heads, "
                "and dropout in [0,1) required"
            )
        self.manifest, self.text_encoder, self.features = manifest, text_encoder, features
        self.posterior_sigma, self.auxiliary_weight = posterior_sigma, auxiliary_weight
        self.fps_by_language = tuple(fps_by_language or [manifest.fps] * 3)
        if len(self.fps_by_language) != 3 or any(
            not math.isfinite(f) or f <= 0 for f in self.fps_by_language
        ):
            raise ValueError("Three verified positive output frame rates for DGS/CSL/ASL required")
        self.source_config = source_config or SourceConfig()
        self.plan = PlanAutoencoder(
            manifest, plan_dim, plan_hidden, stride, plan_layers, heads, plan_hidden * 4, dropout
        )
        self.prior = PlanPrior(
            text_encoder.dim, plan_dim, plan_hidden, plan_layers, heads, plan_hidden * 4, dropout
        )
        self.length_model = LengthModel(text_encoder.dim)
        self.dense = DenseFlow(
            manifest,
            text_encoder.dim,
            plan_dim,
            dense_hidden,
            dense_layers,
            heads,
            dense_hidden * 4,
            dropout,
            auxiliary=auxiliary_weight > 0,
        )
        self.motion_projection = mlp(plan_dim, plan_dim, plan_hidden)
        self.text_projection = mlp(text_encoder.dim, plan_dim, plan_hidden)
        self.plan_dim = plan_dim
        self.stage = None
        self.set_stage("A")

    def set_stage(self, stage, finetune_text=False):
        if stage not in ("A", "B", "C"):
            raise ValueError("Expected stage A, B, or C")
        if stage != "A" and not self.plan.frozen:
            raise RuntimeError("Freeze plan and train statistics before B/C")
        if stage == "A" and self.plan.frozen:
            raise RuntimeError(
                "Frozen plan cannot be silently retrained; rebuild prior if changing q"
            )
        self.requires_grad_(False)
        modules = {
            "A": (self.plan, self.motion_projection, self.text_projection),
            "B": (self.prior, self.length_model),
            "C": (self.dense, self.prior, self.length_model),
        }[stage]
        for module in modules:
            module.requires_grad_(True)
        if finetune_text:
            self.text_encoder.requires_grad_(True)
        self.stage = stage
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Frozen condition encoders must not add dropout drift.
        if not any(p.requires_grad for p in self.text_encoder.parameters()):
            self.text_encoder.eval()
        self.features.eval()
        return self

    def encode_text(self, texts, language):
        return self.text_encoder(texts, language)

    def objective(
        self,
        batch,
        generator=None,
        *,
        pullback_weight=1.0,
        positives=None,
        exclude=None,
        auxiliary_multiplier=1.0,
    ):
        r, e, mask, times = batch.rotations, batch.expression, batch.mask, batch.timestamps
        text, tm = self.encode_text(batch.texts, batch.languages)
        if self.stage == "A":
            z, zm = self.plan.encode_raw(r, e, mask, times)
            pred = self.plan.decode_raw(z, zm, mask, times)
            rec = reconstruction(self.features, pred, (r, e), mask, times)
            pooled = (z * zm[..., None]).sum(1) / zm.sum(1, keepdim=True)
            tp = (text * tm[..., None]).sum(1) / tm.sum(1, keepdim=True)
            nce = contrastive(
                self.motion_projection(pooled),
                self.text_projection(tp),
                batch.languages,
                batch.texts,
                positives=positives,
                exclude=exclude,
            )
            hidden = (torch.rand(mask.shape, device=r.device, generator=generator) < 0.25) & mask
            # Guarantee one masked valid frame per example.
            hidden[:, 0] = True
            eye = torch.eye(3, dtype=r.dtype, device=r.device)
            mr = torch.where(hidden[..., None, None, None], eye, r)
            me = e.masked_fill(hidden[..., None], 0)
            mz, mzm = self.plan.encode_raw(mr, me, mask, times)
            mpred = self.plan.decode_raw(mz, mzm, mask, times)
            masked = reconstruction(self.features, mpred, (r, e), hidden, times)
            loss = rec + 0.1 * nce + 0.1 * masked
            return loss, {
                "reconstruction": rec.detach(),
                "nce": nce.detach(),
                "masked": masked.detach(),
            }
        with torch.no_grad():
            z1, zm = self.plan.posterior(r, e, mask, times, self.posterior_sigma, generator)
        prior = plan_fm_loss(
            self.prior, z1, zm, text, tm, batch.languages, batch.lengths, generator
        )
        length = self.length_model.nll(batch.lengths, text, tm, batch.languages)
        if self.stage == "B":
            return prior + length, {"prior": prior.detach(), "length": length.detach()}
        with torch.no_grad():
            rb, eb = self.plan.decode(z1, zm, mask, times)
            r0, e0 = source(rb, eb, mask, self.manifest, self.source_config, generator)
            s = torch.rand(len(r), device=r.device, generator=generator)
            rs, es, uw, ue = interpolate(r0, e0, r, e, s)
        w, v, aux = self.dense(rs, es, s, text, tm, z1, zm, batch.languages, mask, times)
        flow, terms = kfm_loss(
            self.features, rs, es, w - uw, v - ue, mask, times, pullback_weight=pullback_weight
        )
        loss = flow + prior + length
        terms.update(prior=prior.detach(), length=length.detach())
        if aux is not None:
            auxiliary = reconstruction(self.features, aux, (r, e), mask, times)
            loss = loss + self.auxiliary_weight * auxiliary_multiplier * auxiliary
            terms["auxiliary"] = auxiliary.detach()
        return loss, terms
