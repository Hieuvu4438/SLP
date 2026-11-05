"""Repository BaseMethod bridge; all expensive work is explicit in fit/load."""
from __future__ import annotations

from pathlib import Path

from slp.method import BaseMethod


class Method(BaseMethod):
    def _dataset(self, split, role):
        import numpy as np
        from .data import MotionDataset

        if split.meta.get("split", role) != role:
            raise ValueError("Split metadata does not match the requested training/validation role")

        target_format = self.hparams.get("target_format")
        if target_format not in ("raw133", "normalized133"):
            raise ValueError("declare method.target_format as raw133 or normalized133")
        if not self.hparams.get("target_preprocessing"):
            raise ValueError("declare method.target_preprocessing; existing crops are not undone")
        records = []
        source_key = self.hparams.get("source_key", "source_id")
        fps = self.hparams.get("fps_by_language", {})
        for row in split.inputs:
            if row.get("split", role) != role:
                raise ValueError("input row declares a different split")
            lang = row["lang"]
            if source_key not in row:
                raise ValueError(f"explicit recording source key {source_key!r} is required")
            records.append({"name": row["name"], "text": row["text"], "lang": lang,
                            "source_id": row[source_key], "fps": row.get("fps", fps.get(lang)),
                            "split": role})
        stats = None
        if target_format == "normalized133":
            description = self.hparams.get("target_stats") or {}
            if not description.get("mean") or not description.get("std"):
                raise ValueError("normalized133 requires explicit 133D mean/std .npy paths")
            stats = tuple(np.load(description[k], allow_pickle=False) for k in ("mean", "std"))
        return MotionDataset(records, split.targets, split=role, normalized_stats=stats)

    def fit(self, train, val):
        import torch
        from .application import create_system
        from .checkpoint import read_checkpoint
        from .training import TrainingConfig, TrainingRunner

        incompatible = {"accuracy", "macro_f1", "dtw_jpe", "dtw_jpe_norm",
                        "sibleu_unigram", "sibleu4"}
        if incompatible.intersection(self.cfg.get("eval", {}).get("metrics", [])):
            raise ValueError(
                "configured metrics consume labels/joints/tokens, but this method predicts "
                "SMPL-X parameters; use the method CLI and an explicit shared pose-to-joints/"
                "back-translation evaluator"
            )
        train_data = self._dataset(train, "train")
        val_data = self._dataset(val, "val")
        cfg = TrainingConfig(**{**self.hparams.get("training", {}), "seed": self.ctx.seed})
        torch.manual_seed(cfg.seed)
        self.model, self.tokenizer, fk = create_system(
            train_data, self.hparams.get("model", {}),
            smplx_model_dir=self.hparams.get("smplx_model_dir"),
            rotation_offsets=self.hparams.get("rotation_offsets"),
            smplx_ext=self.hparams.get("smplx_ext", "npz"),
            allow_download=self.hparams.get("allow_download", False),
            text_width=self.hparams.get("text_width", 512),
        )
        def log(record):
            self.ctx.log_metrics({f"train/{record['stage']}": record["train_objective"]},
                                 step=record["epoch"])
        path = TrainingRunner(self.model, self.tokenizer, cfg, self.ctx.ckpt_dir,
                               fk=fk, log_callback=log).fit(train_data, val_data)
        self.metadata = read_checkpoint(path)["metadata"]
        self.metadata["bridge"] = {
            "target_format": self.hparams["target_format"],
            "preprocessing": self.hparams["target_preprocessing"],
            "mean": None if train_data.normalized_stats is None else (
                train_data.normalized_stats[0].tolist()),
            "std": None if train_data.normalized_stats is None else (
                train_data.normalized_stats[1].tolist()),
        }
        self.save()

    def predict(self, inputs):
        import numpy as np
        from .application import predict_records

        if not hasattr(self, "model"):
            raise RuntimeError("fit or load a temporal-plan checkpoint first")
        bridge = self.metadata.get("bridge", {"target_format": "raw133"})
        output = []
        for prediction in predict_records(self.model, self.tokenizer, inputs, self.metadata,
                                           seed=self.ctx.seed):
            raw = prediction["raw133"]
            if bridge["target_format"] == "normalized133":
                raw = (raw-np.asarray(bridge["mean"], dtype=np.float32)) / np.asarray(
                    bridge["std"], dtype=np.float32)
            output.append(raw)
        return output

    def save(self):
        from .checkpoint import save_checkpoint
        if not hasattr(self, "model"):
            raise RuntimeError("nothing has been fitted or loaded")
        save_checkpoint(self.model, self.ctx.ckpt_dir / "final.pt",
                         tokenizer=self.tokenizer, metadata=self.metadata)

    def load(self, ckpt_dir):
        from .checkpoint import load_checkpoint
        path = Path(ckpt_dir)
        path = path / "final.pt" if path.is_dir() else path
        self.model, self.tokenizer, self.metadata = load_checkpoint(
            path, device=self.hparams.get("training", {}).get("device", "cpu"),
            allow_download=self.hparams.get("allow_download", False),
            require_completed=True,
        )
