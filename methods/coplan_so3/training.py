"""Opt-in A/B/C optimization; importing this module never starts training.

No evaluator or test split is accessed. Stage C always updates prior and length
alongside dense flow, including when the shared text encoder is fine-tuned.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .data.dataset import collate

SOURCE_LANGUAGE = {"phoenix": 0, "csl": 1, "how2sign": 2, "DGS": 0, "CSL": 1, "ASL": 2}


class ProtocolDataset(Dataset):
    """Adapter for the existing normalized SOKE Split contract.

    Caller must declare the actual post-preprocessing fps in the manifest.
    Per-clip timestamps in inputs override uniform fps; no resampling is done.
    """

    def __init__(self, split, codec, manifest):
        if len(split.inputs) != len(split.targets):
            raise ValueError("Unpaired split")
        for x in split.inputs:
            if (
                "timestamps" not in x
                and "fps" not in x
                and not split.meta.get("timing_verified", False)
            ):
                raise ValueError(
                    "Verify physical timing for every train/validation clip before training"
                )
            language = x.get("sign_language", x.get("src"))
            if language not in SOURCE_LANGUAGE:
                raise ValueError("Explicit supported sign language or dataset source required")
        self.split, self.codec, self.manifest = split, codec.cpu(), manifest

    def __len__(self):
        return len(self.split.inputs)

    def __getitem__(self, i):
        x = self.split.inputs[i]
        with torch.no_grad():
            features = torch.as_tensor(self.split.targets[i], dtype=torch.float32)
            if features.ndim != 2 or features.shape[1] != 133 or not len(features):
                raise ValueError("Expected a nonempty [T,133] SOKE feature sequence")
            r, e = self.codec.encode(features)
        if "timestamps" in x:
            times = torch.as_tensor(x["timestamps"], dtype=torch.float32)
        else:
            if "fps" not in x and not self.split.meta.get("timing_verified", False):
                raise ValueError(
                    "Verify post-preprocessing timing or supply per-clip timestamps/fps; "
                    "the existing crop/resampling loader does not record physical time"
                )
            fps = float(x.get("fps", self.split.meta.get("fps", self.manifest.fps)))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Finite positive post-preprocessing fps required")
            times = torch.arange(len(features), dtype=torch.float32) / fps
        language = x.get("sign_language", x.get("src"))
        if language not in SOURCE_LANGUAGE:
            raise ValueError("Explicit supported sign language or dataset source required")
        return {
            "rotations": r,
            "expression": e,
            "timestamps": times,
            "text": x["text"],
            "language": SOURCE_LANGUAGE[language],
        }


def language_sampler(dataset, alpha, seed=0):
    """P(dataset i) proportional to N_i**alpha; each sample in i is uniform."""
    if not 0 <= alpha <= 1:
        raise ValueError("Language sampling alpha must be in [0,1]")
    if isinstance(dataset, ProtocolDataset):
        labels = [
            SOURCE_LANGUAGE[x.get("sign_language", x.get("src"))] for x in dataset.split.inputs
        ]
    elif hasattr(dataset, "records"):
        labels = [{"DGS": 0, "CSL": 1, "ASL": 2}[r["sign_language"]] for r in dataset.records]
    elif isinstance(dataset, torch.utils.data.ConcatDataset):
        # Components expose their labels without loading motion arrays.
        parts = [language_sampler(d, alpha, seed) for d in dataset.datasets]
        # Recompute global frequencies below, rather than balancing each part separately.
        labels = [label for part in parts for label in part.language_labels]
    else:
        raise TypeError(
            "Balanced sampling requires protocol/canonical datasets or their concatenation"
        )
    ids = torch.tensor(labels)
    counts = torch.bincount(ids, minlength=3).double().clamp_min(1)
    weights = counts[ids].pow(alpha - 1)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(ids), replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    sampler.language_labels = labels
    return sampler


@torch.no_grad()
def fit_feature_statistics(model, loader, device):
    model.eval()
    for batch in loader:
        b = batch.to(device)
        model.features.fit_batch(b.rotations, b.expression, b.mask, b.timestamps, split="train")
    model.features.finalize(split="train")


@torch.no_grad()
def finalize_plan(model, loader, device):
    model.plan.eval()
    for batch in loader:
        b = batch.to(device)
        z, zm = model.plan.encode_raw(b.rotations, b.expression, b.mask, b.timestamps)
        model.plan.stats.update(z, zm, split="train")
    model.plan.stats.finalize(split="train")
    model.plan.freeze()


def law_signature(model):
    return {
        "source": asdict(model.source_config),
        "stride": model.plan.stride,
        "fps_by_language": model.fps_by_language,
        "posterior_sigma": model.posterior_sigma,
        "feature_weights": model.features.weights,
        "group_weights": model.features.group_weights,
    }


def save_checkpoint(path, model, architecture_config, *, stage, extra=None):
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "manifest_sha256": model.manifest.digest,
        "config": architecture_config,
        "stage": stage,
        "posterior_sigma": model.posterior_sigma,
        "law_signature": law_signature(model),
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model):
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    if payload["format_version"] != 1 or payload["manifest_sha256"] != model.manifest.digest:
        raise ValueError("Checkpoint format or motion manifest mismatch")
    if payload["posterior_sigma"] != model.posterior_sigma:
        raise ValueError("Posterior noise law must match stages B/C and checkpoint")
    if payload["law_signature"] != law_signature(model):
        raise ValueError("Checkpoint source, stride, or metric law mismatch")
    model.load_state_dict(payload["model"], strict=True)
    model.set_stage(payload["stage"])
    return payload


def train_stage(model, stage, train_ds, val_ds, hp, device, ckpt, config, log=None):
    model.set_stage(stage, finetune_text=hp.get("finetune_text", False))
    params = [p for p in model.parameters() if p.requires_grad]
    text_ids = {id(p) for p in model.text_encoder.parameters()}
    main = [p for p in params if id(p) not in text_ids]
    text = [p for p in params if id(p) in text_ids]
    groups = [{"params": main, "lr": hp.get("lr", 2e-4)}]
    if text:
        groups.append({"params": text, "lr": hp.get("text_lr", 1e-5)})
    optimizer = torch.optim.AdamW(groups, weight_decay=hp.get("weight_decay", 0.01))
    batch_size, epochs = hp.get("batch_size", 16), hp.get("epochs", 1)
    if batch_size < 1 or epochs < 1:
        raise ValueError("Positive batch size and epochs required")
    rng = torch.Generator(device=device).manual_seed(hp.get("seed", 0))
    sampler = None
    if hp.get("language_sampling_alpha") is not None:
        sampler = language_sampler(train_ds, hp["language_sampling_alpha"], hp.get("seed", 0))
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(hp.get("seed", 0)),
    )
    validation = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate)
    if not len(loader) or not len(validation):
        raise ValueError("Nonempty train and validation required")
    total = epochs * len(loader)
    warmup = max(1, int(total * hp.get("warmup_fraction", 0.05)))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    best, step = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            ramp = min(1.0, (step + 1) / max(1, hp.get("pullback_warmup_steps", 1000)))
            loss, _ = model.objective(
                batch.to(device), rng, pullback_weight=ramp, auxiliary_multiplier=ramp
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite stage {stage} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params, hp.get("gradient_clip", 1.0), error_if_nonfinite=True
            )
            optimizer.step()
            schedule.step()
            step += 1
        model.eval()
        val_rng = torch.Generator(device=device).manual_seed(12345)
        value, n = 0.0, 0
        # JVP internally enables autograd even here; no parameter .grad is populated.
        with torch.no_grad():
            for batch in validation:
                loss, _ = model.objective(batch.to(device), val_rng)
                value += float(loss) * len(batch.texts)
                n += len(batch.texts)
        value /= n
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite validation objective")
        if log:
            log(f"CoPlan-SO3 stage={stage} epoch={epoch + 1} val_objective={value:.6f}")
        if value < best:
            best = value
            save_checkpoint(
                ckpt,
                model,
                config,
                stage=stage,
                extra={"epoch": epoch + 1, "validation_objective": best},
            )
    restore_checkpoint(ckpt, model)
