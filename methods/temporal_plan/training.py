"""Five-stage training controller. Default CPU; no execution occurs on import."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import islice
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import read_checkpoint, restore_into, save_checkpoint
from .cache import TeacherScoreCache
from .data import MotionDataset, collate_motion, tokenize_texts, validate_train_val
from .system import TemporalPlanSystem, STAGES


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cpu"
    seed: int = 0
    batch_size: int = 8
    accumulation: int = 8
    epochs: dict[str, int] = field(default_factory=lambda: {
        "codec": 100, "segment": 50, "planner": 100, "generator": 100, "exposure": 25,
    })
    codec_lr: float = 2e-4
    lr: float = 1e-4
    weight_decay: float = .01
    gradient_clip: float = 1.
    patience: int | None = 10
    max_text_length: int = 128
    teacher_cache_entries: int = 64

    def __post_init__(self):
        integers = (self.seed, self.batch_size, self.accumulation,
                    self.max_text_length, self.teacher_cache_entries)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in integers):
            raise ValueError("seed, batching, text length and cache capacity must be integers")
        if set(self.epochs) != set(STAGES) or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.epochs.values()
        ):
            raise ValueError("provide positive integer epochs for all five stages")
        if min(self.batch_size, self.accumulation) < 1 or self.max_text_length < 2:
            raise ValueError("invalid batching/text budget")
        if self.patience is not None and (isinstance(self.patience, bool) or
                not isinstance(self.patience, int) or self.patience < 1):
            raise ValueError("patience must be positive or null")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.codec_lr, self.lr, self.gradient_clip)):
            raise ValueError("learning rates and gradient clipping must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("invalid weight decay")
        if self.teacher_cache_entries < 0:
            raise ValueError("invalid teacher cache capacity")
        torch.device(self.device)  # syntax validation; no device is initialized here


def weighted_backward(loss, batch_size: int, window_examples: int):
    """A short final microbatch has its correct example weight, not 1/accumulation."""
    if batch_size < 1 or window_examples < batch_size:
        raise ValueError("invalid microbatch accumulation weights")
    (loss * (batch_size / window_examples)).backward()


class TrainingRunner:
    def __init__(self, model: TemporalPlanSystem, tokenizer, config: TrainingConfig,
                 output_dir: str | Path, *, fk=None, log_callback=None):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.output_dir = Path(output_dir)
        self.fk, self.log_callback = fk, log_callback
        self.device = torch.device(config.device)
        self.history: list[dict] = []

    def _loader(self, dataset, *, shuffle=False, seed=0):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle,
                          collate_fn=collate_motion, generator=generator, num_workers=0)

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
            return self.model.segment_loss(batch.raw, batch.lengths, generator=generator,
                                            source_ids=batch.names)
        ids, mask = tokenize_texts(self.tokenizer, batch.texts,
                                   max_length=self.config.max_text_length, device=self.device)
        args = (batch.raw, batch.lengths, ids, mask, batch.language_ids)
        if stage == "planner":
            return self.model.planner_loss(*args, generator=generator, source_ids=batch.names)
        return self.model.generator_loss(*args, fk=self.fk, generator=generator,
                                          source_ids=batch.names)

    @torch.no_grad()
    def _validation(self, val, out_of_support, stage_index):
        if val is None or out_of_support:
            return {"available": False, "out_of_support": out_of_support,
                    "examples": 0 if val is None else len(val),
                    "selection": "fixed_budget_last_epoch"}
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.seed + 100_000 + stage_index
        )
        total, examples = 0., 0
        for batch in self._loader(val):
            value = self._loss(batch, generator)["total"]
            if not bool(torch.isfinite(value)):
                raise FloatingPointError("non-finite validation objective")
            n = len(batch.lengths)
            total += float(value) * n
            examples += n
        return {"available": True, "objective": total/examples, "examples": examples,
                "selection": "conditional_validation_objective",
                "note": "teacher/reference-conditioned diagnostic, not text-only quality"}

    def _log(self, record):
        with (self.output_dir / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if self.log_callback is not None:
            self.log_callback(record)

    def fit(self, train: MotionDataset, val: MotionDataset | None = None):
        validate_train_val(train, val)
        if self.model.config.max_latents != train.max_latents:
            raise ValueError("model cap must be derived exclusively from max TRAIN length")
        if bool(self.model.completed_stages.any()):
            raise ValueError("fit starts fresh; checkpoints are weight/inference snapshots, not resume")
        if self.model.config.losses.fk > 0 and self.fk is None:
            raise ValueError("primary training requires an explicit SMPL-X FK backend")
        if self.fk is not None and hasattr(self.fk, "rotation_offsets"):
            torch.testing.assert_close(self.model.representation.rotation_offsets.cpu(),
                                       self.fk.rotation_offsets.cpu())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise FileExistsError("choose a fresh training directory; existing runs are not overwritten")
        self.model.to(self.device)
        if self.fk is not None:
            self.fk.to(self.device).eval()
        torch.manual_seed(self.config.seed)
        unsupported = 0 if val is None else sum(
            (val.frame_count(i)+1)//2 > self.model.config.max_latents for i in range(len(val))
        )
        metadata = {
            "training": asdict(self.config), "train_fingerprint": train.fingerprint,
            "val_fingerprint": None if val is None else val.fingerprint,
            "fingerprint_scope": "record metadata and lengths, not every motion byte",
            "fps_by_language": train.fps_by_language, "validation_out_of_support": unsupported,
            "optimizer_state_saved": False, "exact_resume_supported": False,
            "max_latents_source": "ceil(max training frames/2)",
            "pose_mean_sha256": hashlib.sha256(
                self.model.representation.rotation_offsets.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "fk_context": (self.fk.provenance() if self.fk is not None and
                           hasattr(self.fk, "provenance") else None),
        }
        (self.output_dir / "run.json").write_text(json.dumps(metadata, indent=2))
        self.model.fit_motion_statistics(self._motion_batches(train))
        self.model.score_cache = TeacherScoreCache(self.output_dir / "teacher_scores",
                                                    self.config.teacher_cache_entries)
        for stage_index, stage in enumerate(STAGES):
            self.model.set_stage(stage)
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                parameters, lr=self.config.codec_lr if stage == "codec" else self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            generator = torch.Generator(device=self.device).manual_seed(
                self.config.seed + 1000*stage_index
            )
            torch.manual_seed(self.config.seed + 1000*stage_index)
            best, bad_epochs, updates = math.inf, 0, 0
            best_path = self.output_dir / f"best_{stage}.pt"
            for epoch in range(self.config.epochs[stage]):
                self.model.train()
                iterator = iter(self._loader(train, shuffle=True,
                                             seed=self.config.seed+1000*stage_index+epoch))
                epoch_total, epoch_examples = 0., 0
                while window := list(islice(iterator, self.config.accumulation)):
                    examples = sum(len(b.lengths) for b in window)
                    optimizer.zero_grad(set_to_none=True)
                    for batch in window:
                        losses = self._loss(batch, generator)
                        loss = losses["total"]
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"non-finite {stage} training loss")
                        n = len(batch.lengths)
                        weighted_backward(loss, n, examples)
                        epoch_total += float(loss.detach()) * n
                        epoch_examples += n
                    torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                validation = self._validation(val, unsupported, stage_index)
                record = {"stage": stage, "epoch": epoch, "optimizer_updates": updates,
                          "train_objective": epoch_total/epoch_examples, "validation": validation}
                self._log(record)
                if validation["available"]:
                    if validation["objective"] < best:
                        best, bad_epochs = validation["objective"], 0
                        save_checkpoint(self.model, best_path, tokenizer=self.tokenizer,
                                         metadata=metadata)
                    else:
                        bad_epochs += 1
                    if self.config.patience is not None and bad_epochs >= self.config.patience:
                        break
            if updates == 0:
                raise RuntimeError("a stage cannot be completed without optimizer updates")
            if best_path.exists():
                restore_into(self.model, read_checkpoint(best_path))
            self.model.mark_stage_completed()
            if stage == "codec":
                self.model.fit_latent_statistics(self._motion_batches(train))
            elif stage == "segment":
                self.model.fit_content_statistics(self._motion_batches(train), generator=generator)
            self.history.append({"stage": stage, "updates": updates,
                                 "selection": "best_validation" if best_path.exists() else "last_epoch"})
            save_checkpoint(self.model, self.output_dir / f"stage_{stage}.pt",
                             tokenizer=self.tokenizer, metadata={**metadata, "history": self.history})
        self.model.eval()
        final_path = self.output_dir / "final.pt"
        save_checkpoint(self.model, final_path, tokenizer=self.tokenizer,
                         metadata={**metadata, "history": self.history})
        return final_path
