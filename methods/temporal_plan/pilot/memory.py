"""Disposable, bounded memory diagnostics. Never save experimental model weights."""
from __future__ import annotations
from dataclasses import replace
import gc
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from ..data import collate_motion, tokenize_texts
from ..plans import PlanBatch
from ..teacher import TeacherTargets
from ..conditioning import temporal_condition


def configure_budget():
    budget = int(os.environ["PILOT_GPU_MEMORY_BUDGET_BYTES"])
    if budget <= 0:
        raise ValueError("positive GPU memory budget required")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., budget/total), device=0)
    torch.cuda.reset_peak_memory_stats(0)
    return budget


def measured_memory():
    if not torch.cuda.is_initialized():
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def diagnostic_batch(train, count):
    # Worst TRAIN-derived motion canvas and max configured token length. No dev/test.
    index = max(range(len(train)), key=train.frame_count)
    batch = collate_motion([train[index] for _ in range(count)])
    frames = 2*train.max_latents
    if batch.raw.shape[1] < frames:
        batch.raw = torch.cat((batch.raw, batch.raw[:, -1:].expand(-1, frames-batch.raw.shape[1], -1)), 1)
    batch.lengths.fill_(frames)
    batch.texts = tuple((text+" ")*128 for text in batch.texts)
    return batch


def diagnostic_updates(runner, batch, generator, *, steps=2):
    parameters = [p for p in runner.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = runner._loss(batch, generator)["total"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite disposable memory diagnostic")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()  # Include actual AdamW moment allocations, not forward alone.
    runner.model.zero_grad(set_to_none=True)


def maximum_targets(model):
    # Unit durations cover the maximum supported plan width. These shape-only
    # targets are used ONLY in a discarded diagnostic, never in real training.
    def targets(features, lengths, generator, source_ids=None):
        contents, durations, sources = [], [], []
        for b, n in enumerate(lengths.tolist()):
            for _ in range(model.config.plan_samples):
                contents.append(features.new_zeros(n, model.config.content_width))
                durations.append([1]*n)
                sources.append(b)
        return TeacherTargets(PlanBatch.from_sequences(contents, durations),
                              torch.tensor(sources, device=features.device))
    model._targets = targets


def inference_canvas(model, tokenizer, batch):
    ids, mask = tokenize_texts(tokenizer, batch.texts, max_length=128, device="cuda")
    model.eval()
    with torch.no_grad():
        memory, valid = model.text(ids, mask, batch.language_ids.to("cuda"))
        n = model.config.max_latents
        plan = PlanBatch.from_sequences(
            [memory.new_zeros(n, model.config.content_width) for _ in batch.texts],
            [[1]*n for _ in batch.texts])
        # Full-width planner forward bounds its autoregressive prefix buffers.
        model.planner.loss(memory, valid, plan)
        condition = temporal_condition(plan, overlap=model.config.overlap)
        latents = model.diffusion.sample(model.denoiser, plan.latent_lengths, memory, valid,
            condition.features, latent_width=model.config.latent_width, sampling_steps=1,
            generator=torch.Generator(device="cuda").manual_seed(0))
        decoded = model.codec.decode(model._inverse_latents(latents), plan.latent_lengths)
        model.representation.decode(model.motion_norm.inverse(decoded), 2*plan.latent_lengths)


def probe_job(plan, identity):
    from .worker import datasets, fresh, shared, fk_backend, training_config, bt_model, bt_features
    from ..training import TrainingRunner
    from ..checkpoint import load_checkpoint
    train, _ = datasets(plan)
    count = 1 if identity.startswith("predict_") or identity == "evaluate" else plan["microbatch"]
    batch = diagnostic_batch(train, count)
    generator = torch.Generator(device="cuda").manual_seed(plan["seed"])
    stages = []
    if identity in ("preflight", "common"):
        model, tokenizer, backend = fresh(plan, train)
        model.to("cuda"); backend.to("cuda")
        # Statistics for this disposable model are train-only; real training fits
        # its own statistics from scratch, unchanged by diagnostics.
        model.fit_motion_statistics((
            (b.raw.to("cuda"), b.lengths.to("cuda")) for b in
            (collate_motion([train[i]]) for i in range(min(16, len(train))))))
        runner = TrainingRunner(model, tokenizer, training_config(plan), Path(plan["output"])/"unused", fk=backend)
        model.set_stage("codec")
        diagnostic_updates(runner, batch, generator); stages.append("codec_backward_and_adamw")
        if identity == "common":
            model.mark_stage_completed()
            b = batch.to("cuda")
            model.fit_latent_statistics([(b.raw, b.lengths)])
            model.set_stage("segment")
            diagnostic_updates(runner, batch, generator); stages.append("segment_backward_and_adamw")
            model.mark_stage_completed()
            model.fit_content_statistics([(b.raw, b.lengths)], generator=generator)
    elif identity in ("profile", "map", "sample"):
        modes = ("map", "sample") if identity == "profile" else (identity,)
        for mode in modes:
            model, tokenizer, _ = shared(plan, mode)
            backend = fk_backend(plan).to("cuda")
            maximum_targets(model)
            runner = TrainingRunner(model, tokenizer, training_config(plan), Path(plan["output"])/"unused", fk=backend)
            for stage in ("planner", "generator", "exposure"):
                model.set_stage(stage)
                diagnostic_updates(runner, batch, generator)
                stages.append(mode+"_"+stage+"_backward_and_adamw")
                model.mark_stage_completed()
            del model, tokenizer, backend, runner
            gc.collect(); torch.cuda.empty_cache()
    elif identity in ("evaluator", "evaluate"):
        model, _, _ = shared(plan)
        bt, tokenizer = bt_model()
        if identity == "evaluate":
            payload = torch.load(Path(plan["output"])/"evaluator.pt", map_location="cpu", weights_only=True)
            bt.restore_compact(payload["state"])
            bt.eval(); bt.requires_grad_(False)
        batch = batch.to("cuda")
        labels, mask = tokenize_texts(tokenizer, batch.texts, max_length=128, device="cuda")
        labels = labels.masked_fill(~mask.bool(), -100)
        with torch.no_grad():
            features = bt_features(model, batch.raw, batch.lengths)
        if identity == "evaluator":
            params = [p for p in bt.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=1e-4)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = bt(features, batch.lengths, labels)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite evaluator diagnostic")
                loss.backward(); optimizer.step()
            stages.append("evaluator_backward_and_adamw")
            del optimizer, loss
        bt.eval()
        with torch.no_grad():
            bt(features, batch.lengths, labels)
            bt.translate(features, batch.lengths, tokenizer)
            normalized, _ = model._prepare(batch.raw, batch.lengths)
            posterior = model.codec.encode(normalized, batch.lengths)
            rec = model.codec.decode(posterior.mean, posterior.lengths, batch.lengths)
            bt.translate(rec, batch.lengths, tokenizer)
            backend = fk_backend(plan).to("cuda")
            backend(batch.raw.flatten(0,1))
        stages.append("evaluator_and_codec_inference")
    elif identity.startswith("predict_"):
        mode = identity.removeprefix("predict_")
        ready = __import__("json").loads((Path(plan["output"])/f"{mode}_ready.json").read_text())
        model, tokenizer, _ = load_checkpoint(ready["checkpoint"], device="cuda", require_completed=True)
        inference_canvas(model, tokenizer, batch)
        stages.append("maximum_plan_and_motion_inference")
    else:
        raise ValueError("no memory probe for job " + identity)
    torch.cuda.synchronize()
    return {"stages": stages, "motion_frames": int(batch.lengths.max()),
        "microbatch": count, "text_tokens": 128, "optimizer_steps_per_training_stage": 2,
        "diagnostic_only": True, "experimental_weights_saved": False,
        "shape_source": "max TRAIN-derived cap; max token budget; max-width plan targets"}
