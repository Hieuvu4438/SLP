"""Five-stage training controller. Default CPU; no execution occurs on import."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import islice
import hashlib
import json
import math
from pathlib import Path
import time
import signal
import shutil

import torch
from torch.utils.data import DataLoader

from .checkpoint import read_checkpoint, restore_into, save_checkpoint
from .cache import TeacherScoreCache
from .data import MotionDataset, collate_motion, tokenize_texts, validate_train_val
from .system import TemporalPlanSystem, STAGES
from .progress import TrainingProgress, restore_rng


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cpu"
    seed: int = 0
    batch_size: int = 8
    accumulation: int = 8
    epochs: dict[str, int] = field(
        default_factory=lambda: {
            "codec": 100,
            "segment": 50,
            "planner": 100,
            "generator": 100,
            "exposure": 25,
        }
    )
    codec_lr: float = 2e-4
    lr: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    patience: int | None = 10
    max_text_length: int = 128
    teacher_cache_entries: int = 64
    allow_shared_source_groups: bool = False
    stage_max_updates: dict[str, int] = field(default_factory=dict)
    stage_max_seconds: dict[str, float] = field(default_factory=dict)
    stage_initial_updates: dict[str, int] = field(default_factory=dict)
    stage_restart_updates: dict[str, list[int]] = field(default_factory=dict)
    checkpoint_every_updates: int = 50

    def __post_init__(self):
        integers = (
            self.seed,
            self.batch_size,
            self.accumulation,
            self.max_text_length,
            self.teacher_cache_entries,
        )
        if any(isinstance(v, bool) or not isinstance(v, int) for v in integers):
            raise ValueError("seed, batching, text length and cache capacity must be integers")
        if set(self.epochs) != set(STAGES) or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.epochs.values()
        ):
            raise ValueError("provide positive integer epochs for all five stages")
        if min(self.batch_size, self.accumulation) < 1 or self.max_text_length < 2:
            raise ValueError("invalid batching/text budget")
        if self.patience is not None and (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, int)
            or self.patience < 1
        ):
            raise ValueError("patience must be positive or null")
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.codec_lr, self.lr, self.gradient_clip)
        ):
            raise ValueError("learning rates and gradient clipping must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("invalid weight decay")
        if self.teacher_cache_entries < 0:
            raise ValueError("invalid teacher cache capacity")
        if set(self.stage_max_updates) - set(STAGES) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in self.stage_max_updates.values()
        ):
            raise ValueError("invalid per-stage update budget")
        if set(self.stage_max_seconds) - set(STAGES) or any(
            not math.isfinite(v) or v <= 0 for v in self.stage_max_seconds.values()
        ):
            raise ValueError("invalid per-stage time budget")
        if self.checkpoint_every_updates < 1:
            raise ValueError("positive checkpoint interval required")
        for stage, n in self.stage_initial_updates.items():
            if (
                stage not in STAGES
                or not isinstance(n, int)
                or n < 0
                or n >= self.stage_max_updates.get(stage, math.inf)
            ):
                raise ValueError("invalid initial update count")
        for stage, points in self.stage_restart_updates.items():
            if (
                stage not in STAGES
                or points != sorted(set(points))
                or any(
                    not isinstance(n, int)
                    or n <= 0
                    or n >= self.stage_max_updates.get(stage, math.inf)
                    for n in points
                )
            ):
                raise ValueError("invalid optimizer restart schedule")
        torch.device(self.device)  # syntax validation; no device is initialized here


def weighted_backward(loss, batch_size: int, window_examples: int):
    """A short final microbatch has its correct example weight, not 1/accumulation."""
    if batch_size < 1 or window_examples < batch_size:
        raise ValueError("invalid microbatch accumulation weights")
    (loss * (batch_size / window_examples)).backward()


class TrainingRunner:
    def __init__(
        self,
        model: TemporalPlanSystem,
        tokenizer,
        config: TrainingConfig,
        output_dir: str | Path,
        *,
        fk=None,
        log_callback=None,
    ):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.output_dir = Path(output_dir)
        self.fk, self.log_callback = fk, log_callback
        self.device = torch.device(config.device)
        self.history: list[dict] = []

    def _loader(self, dataset, *, shuffle=False, seed=0):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_motion,
            generator=generator,
            num_workers=0,
        )

    def _motion_batches(self, dataset):
        for batch in self._loader(dataset):
            batch = batch.to(self.device)
            yield batch.raw, batch.lengths

    def _loss(self, batch, generator):
        batch = batch.to(self.device)
        stage = self.model.stage
        if stage == "codec":
            return self.model.codec_loss(batch.raw, batch.lengths, fk=self.fk, generator=generator)
        if stage == "segment":
            return self.model.segment_loss(
                batch.raw, batch.lengths, generator=generator, source_ids=batch.names
            )
        ids, mask = tokenize_texts(
            self.tokenizer, batch.texts, max_length=self.config.max_text_length, device=self.device
        )
        args = (batch.raw, batch.lengths, ids, mask, batch.language_ids)
        if stage == "planner":
            return self.model.planner_loss(*args, generator=generator, source_ids=batch.names)
        return self.model.generator_loss(
            *args, fk=self.fk, generator=generator, source_ids=batch.names
        )

    @torch.no_grad()
    def _validation(self, val, out_of_support, stage_index, deadline=math.inf):
        if val is None or out_of_support:
            return {
                "available": False,
                "out_of_support": out_of_support,
                "examples": 0 if val is None else len(val),
                "selection": "fixed_budget_last_epoch",
            }
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.seed + 100_000 + stage_index
        )
        total, examples = 0.0, 0
        for batch in self._loader(val):
            if time.monotonic() >= deadline:
                return {
                    "available": False,
                    "reason": "time_budget",
                    "examples": len(val),
                    "selection": "last_budget_update",
                }
            value = self._loss(batch, generator)["total"]
            if not bool(torch.isfinite(value)):
                raise FloatingPointError("non-finite validation objective")
            n = len(batch.lengths)
            total += float(value) * n
            examples += n
        return {
            "available": True,
            "objective": total / examples,
            "examples": examples,
            "selection": "conditional_validation_objective",
            "note": "teacher/reference-conditioned diagnostic, not text-only quality",
        }

    def _log(self, record):
        with (self.output_dir / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if self.log_callback is not None:
            self.log_callback(record)

    def fit(
        self,
        train: MotionDataset,
        val: MotionDataset | None = None,
        *,
        start_stage: str = "codec",
        stop_after: str = "exposure",
        parent_metadata: dict | None = None,
        resume: bool = False,
    ):
        validate_train_val(
            train, val, allow_shared_source_groups=self.config.allow_shared_source_groups
        )
        if start_stage not in STAGES or stop_after not in STAGES:
            raise ValueError("unknown stage interval")
        first, last = STAGES.index(start_stage), STAGES.index(stop_after)
        if last < first:
            raise ValueError("stage interval is reversed")
        if self.model.config.max_latents != train.max_latents:
            raise ValueError("model cap must be derived exclusively from max TRAIN length")
        progress = TrainingProgress(self.output_dir)
        recovery = None
        if resume:
            if not progress.exists():
                raise ValueError("no complete training recovery checkpoint")
            payload, recovery = progress.load()
            if recovery["metadata"]["train_fingerprint"] != train.fingerprint:
                raise ValueError("recovery training data mismatch")
            if recovery["metadata"]["training"] != asdict(self.config):
                raise ValueError("recovery optimizer/training configuration mismatch")
            restore_into(self.model, payload)
            first = STAGES.index(recovery["runtime"]["stage"])
            self.history = recovery["runtime"]["history"]
            if recovery.get("best_checkpoint_path"):
                shutil.copy2(
                    recovery["best_checkpoint_path"],
                    self.output_dir / f"best_{recovery['runtime']['stage']}.pt",
                )
        expected = torch.arange(len(STAGES), device=self.model.completed_stages.device) < first
        if not torch.equal(self.model.completed_stages, expected):
            raise ValueError(
                "fit starts fresh or at an explicit completed stage boundary, not resume"
            )
        if (
            first
            and recovery is None
            and (
                parent_metadata is None
                or parent_metadata.get("train_fingerprint") != train.fingerprint
            )
        ):
            raise ValueError("stage fork must identify the same training data")
        if self.model.config.losses.fk > 0 and self.fk is None:
            raise ValueError("primary training requires an explicit SMPL-X FK backend")
        if self.fk is not None and hasattr(self.fk, "rotation_offsets"):
            torch.testing.assert_close(
                self.model.representation.rotation_offsets.cpu(), self.fk.rotation_offsets.cpu()
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not resume and any(self.output_dir.iterdir()):
            raise FileExistsError(
                "choose a fresh training directory; existing runs are not overwritten"
            )
        self.model.to(self.device)
        if self.fk is not None:
            self.fk.to(self.device).eval()
        torch.manual_seed(self.config.seed)
        unsupported = (
            0
            if val is None
            else sum(
                (val.frame_count(i) + 1) // 2 > self.model.config.max_latents
                for i in range(len(val))
            )
        )
        metadata = (
            recovery["metadata"]
            if recovery
            else {
                "training": asdict(self.config),
                "train_fingerprint": train.fingerprint,
                "val_fingerprint": None if val is None else val.fingerprint,
                "fingerprint_scope": "record metadata and lengths, not every motion byte",
                "fps_by_language": train.fps_by_language,
                "validation_out_of_support": unsupported,
                "optimizer_state_saved": False,
                "exact_resume_supported": False,
                "training_recovery_supported": True,
                "training_recovery_manifest": "resume/current.json",
                "stage_boundary_fork_supported": True,
                "max_latents_source": "ceil(max training frames/2)",
                "pose_mean_sha256": hashlib.sha256(
                    self.model.representation.rotation_offsets.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "fk_context": self.fk.provenance()
                if self.fk is not None and hasattr(self.fk, "provenance")
                else None,
                "allow_shared_source_groups": self.config.allow_shared_source_groups,
                "shared_source_group_count": 0
                if val is None
                else len(
                    {(r["lang"], r["source_id"]) for r in train.records}
                    & {(r["lang"], r["source_id"]) for r in val.records}
                ),
                "parent_checkpoint": None
                if parent_metadata is None
                else parent_metadata.get("parent_checkpoint"),
                "legacy_warm_start_not_exact_resume": bool(self.config.stage_initial_updates),
                "warm_start_source": None
                if parent_metadata is None
                else parent_metadata.get("warm_start_source"),
            }
        )
        (self.output_dir / "run.json").write_text(json.dumps(metadata, indent=2))
        if first == 0 and recovery is None:
            self.model.fit_motion_statistics(self._motion_batches(train))
        elif first and not all(
            int(n.count) > 0
            for n in (self.model.motion_norm, self.model.latent_norm, self.model.content_norm)
        ):
            raise ValueError("branching requires fitted shared codec/content statistics")
        self.model.score_cache = TeacherScoreCache(
            self.output_dir / "teacher_scores", self.config.teacher_cache_entries
        )
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        previous = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            for stage_index in range(first, last + 1):
                stage = STAGES[stage_index]
                self.model.set_stage(stage)
                parameters = [p for p in self.model.parameters() if p.requires_grad]
                limit = self.config.stage_max_updates.get(stage, math.inf)
                updates = self.config.stage_initial_updates.get(stage, 0)
                started = time.monotonic()
                deadline = started + self.config.stage_max_seconds.get(stage, math.inf)
                points = self.config.stage_restart_updates.get(stage, [])
                best_path = self.output_dir / f"best_{stage}.pt"
                saved = (
                    recovery["runtime"]
                    if recovery is not None and recovery["runtime"]["stage"] == stage
                    else None
                )
                if saved:
                    updates = saved["updates"]
                phase_start = saved["phase_start"] if saved else updates
                reason = "epochs"
                while True:
                    phase_end = min([p for p in points if p > phase_start] + [limit])
                    optimizer = torch.optim.AdamW(
                        parameters,
                        lr=self.config.codec_lr if stage == "codec" else self.config.lr,
                        weight_decay=self.config.weight_decay,
                    )
                    phase_seed = self.config.seed + 1000 * stage_index + phase_start
                    generator = torch.Generator(device=self.device).manual_seed(phase_seed)
                    torch.manual_seed(phase_seed)
                    best, bad_epochs = math.inf, 0
                    epoch_start, cursor, epoch_total, epoch_examples = 0, 0, 0.0, 0
                    if saved:
                        optimizer.load_state_dict(recovery["optimizer"])
                        generator.set_state(recovery["generator"])
                        best, bad_epochs = saved["best"], saved["bad_epochs"]
                        epoch_start, cursor = saved["epoch"], saved["window_index"]
                        epoch_total, epoch_examples = saved["epoch_total"], saved["epoch_examples"]
                        restore_rng(recovery["rng"])

                    def checkpoint(epoch, window_index, total, examples):
                        runtime = {
                            "stage": stage,
                            "updates": updates,
                            "phase_start": phase_start,
                            "epoch": epoch,
                            "window_index": window_index,
                            "epoch_total": total,
                            "epoch_examples": examples,
                            "best": best,
                            "bad_epochs": bad_epochs,
                            "history": self.history,
                        }
                        progress.save(
                            self.model, self.tokenizer, runtime, optimizer, generator, metadata
                        )

                    if not saved:
                        checkpoint(0, 0, 0.0, 0)
                    saved = None
                    recovery = None
                    for epoch in range(epoch_start, self.config.epochs[stage]):
                        self.model.train()
                        iterator = iter(self._loader(train, shuffle=True, seed=phase_seed + epoch))
                        for window_index, window in enumerate(
                            iter(lambda: list(islice(iterator, self.config.accumulation)), [])
                        ):
                            if epoch == epoch_start and window_index < cursor:
                                continue
                            if stop_requested:
                                checkpoint(epoch, window_index, epoch_total, epoch_examples)
                                raise InterruptedError(
                                    "training stopped at a recoverable optimizer boundary"
                                )
                            if time.monotonic() >= deadline:
                                reason = "time_budget"
                                break
                            if updates >= phase_end:
                                break
                            examples = sum(len(b.lengths) for b in window)
                            optimizer.zero_grad(set_to_none=True)
                            for batch in window:
                                loss = self._loss(batch, generator)["total"]
                                if not bool(torch.isfinite(loss)):
                                    raise FloatingPointError(f"non-finite {stage} training loss")
                                weighted_backward(loss, len(batch.lengths), examples)
                                epoch_total += float(loss.detach()) * len(batch.lengths)
                                epoch_examples += len(batch.lengths)
                            torch.nn.utils.clip_grad_norm_(
                                parameters, self.config.gradient_clip, error_if_nonfinite=True
                            )
                            optimizer.step()
                            updates += 1
                            if updates % 10 == 0:
                                self._log(
                                    {
                                        "stage": stage,
                                        "epoch": epoch,
                                        "optimizer_updates": updates,
                                        "progress_only": True,
                                        "phase_start_update": phase_start,
                                        "wall_seconds": time.monotonic() - started,
                                    }
                                )
                            if (
                                updates % self.config.checkpoint_every_updates == 0
                                or updates >= phase_end
                                or stop_requested
                            ):
                                checkpoint(epoch, window_index + 1, epoch_total, epoch_examples)
                            if updates >= phase_end:
                                reason = "update_budget"
                                break
                        cursor = 0
                        if epoch_examples:
                            validation = self._validation(val, unsupported, stage_index, deadline)
                            self._log(
                                {
                                    "stage": stage,
                                    "epoch": epoch,
                                    "optimizer_updates": updates,
                                    "phase_start_update": phase_start,
                                    "train_objective": epoch_total / epoch_examples,
                                    "validation": validation,
                                }
                            )
                            if validation["available"]:
                                if validation["objective"] < best:
                                    best, bad_epochs = validation["objective"], 0
                                    save_checkpoint(
                                        self.model,
                                        best_path,
                                        tokenizer=self.tokenizer,
                                        metadata=metadata,
                                    )
                                else:
                                    bad_epochs += 1
                            checkpoint(epoch + 1, 0, 0.0, 0)
                        epoch_total, epoch_examples = 0.0, 0
                        if updates >= phase_end or reason == "time_budget":
                            break
                        if self.config.patience is not None and bad_epochs >= self.config.patience:
                            reason = "patience"
                            break
                    if updates >= phase_end and phase_end < limit:
                        # Same documented reset for both variants; carry raw last weights.
                        save_checkpoint(
                            self.model,
                            self.output_dir / f"phase_{stage}_{updates}.pt",
                            tokenizer=self.tokenizer,
                            metadata={
                                **metadata,
                                "actual_updates": updates,
                                "optimizer_reset_at_update": updates,
                                "history": self.history,
                            },
                        )
                        if best_path.exists():
                            best_path.replace(
                                self.output_dir / f"best_{stage}_before_reset_{updates}.pt"
                            )
                        phase_start = updates
                        self._log(
                            {"stage": stage, "optimizer_updates": updates, "optimizer_reset": True}
                        )
                        continue
                    break
                if updates == 0:
                    raise RuntimeError("a stage cannot be completed without optimizer updates")
                if updates < limit < math.inf:
                    save_checkpoint(
                        self.model,
                        self.output_dir / f"incomplete_{stage}.pt",
                        tokenizer=self.tokenizer,
                        metadata={
                            **metadata,
                            "budget_incomplete": True,
                            "actual_updates": updates,
                            "required_updates": limit,
                        },
                    )
                    raise RuntimeError("time/epoch budget ended before matched update target")
                if best_path.exists():
                    restore_into(self.model, read_checkpoint(best_path))
                self.model.mark_stage_completed()
                if stage == "codec":
                    self.model.fit_latent_statistics(self._motion_batches(train))
                elif stage == "segment":
                    self.model.fit_content_statistics(
                        self._motion_batches(train), generator=generator
                    )
                self.history.append(
                    {
                        "stage": stage,
                        "updates": updates,
                        "completion_reason": reason,
                        "wall_seconds": time.monotonic() - started,
                        "selection": "best_validation" if best_path.exists() else "last_epoch",
                        "warm_start_updates": self.config.stage_initial_updates.get(stage, 0),
                        "optimizer_restart_updates": points,
                    }
                )
                save_checkpoint(
                    self.model,
                    self.output_dir / f"stage_{stage}.pt",
                    tokenizer=self.tokenizer,
                    metadata={**metadata, "history": self.history},
                )
            self.model.eval()
            final_path = self.output_dir / (
                "final.pt" if last == len(STAGES) - 1 else f"partial_{STAGES[last]}.pt"
            )
            save_checkpoint(
                self.model,
                final_path,
                tokenizer=self.tokenizer,
                metadata={**metadata, "history": self.history},
            )
            return final_path
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
