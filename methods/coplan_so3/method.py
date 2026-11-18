"""Repository method contract. fit is opt-in; predict accepts inputs only."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from slp.method import BaseMethod

from .data.dataset import collate
from .factory import build
from .sampling import generate
from .training import (
    ProtocolDataset,
    SOURCE_LANGUAGE,
    finalize_plan,
    fit_feature_statistics,
    restore_checkpoint,
    save_checkpoint,
    train_stage,
)


class Method(BaseMethod):
    def _setup(self):
        self.model, self.codec = build(self.hparams)
        self.device = torch.device(self.hparams.get("device", "cpu"))
        self.model.to(self.device)

    def fit(self, train, val):
        if train.meta.get("split", "train") != "train" or val.meta.get("split", "val") not in (
            "val",
            "dev",
        ):
            raise ValueError("fit requires train and validation splits")
        torch.manual_seed(self.cfg.get("seed", 0))
        self._setup()
        train_ds = ProtocolDataset(train, self.codec, self.model.manifest)
        val_ds = ProtocolDataset(val, self.codec, self.model.manifest)
        stats_loader = DataLoader(
            train_ds, batch_size=self.hparams.get("statistics_batch_size", 8), collate_fn=collate
        )
        fit_feature_statistics(self.model, stats_loader, self.device)
        for stage in ("A", "B", "C"):
            train_stage(
                self.model,
                stage,
                train_ds,
                val_ds,
                self.hparams["training"][stage],
                self.device,
                self.ctx.ckpt_dir / f"stage_{stage}.pt",
                self.hparams,
                log=self.ctx.logger.info,
            )
            if stage == "A":
                finalize_plan(self.model, stats_loader, self.device)
                save_checkpoint(
                    self.ctx.ckpt_dir / "plan_frozen.pt", self.model, self.hparams, stage="B"
                )
        self.save()

    def predict(self, inputs):
        if not hasattr(self, "model"):
            raise RuntimeError("Load a checkpoint before predict")
        result = []
        self.last_generation_metadata = []
        sampling = self.hparams.get("sampling", {})
        controlled = sampling.get("length_controlled", False)
        for i, x in enumerate(inputs):
            language = x.get("sign_language", x.get("src"))
            if language not in SOURCE_LANGUAGE:
                raise ValueError("Explicit sign language required")
            lengths = [x["length"]] if controlled else None
            out = generate(
                self.model,
                [x["text"]],
                [SOURCE_LANGUAGE[language]],
                seed=self.cfg.get("seed", 0) + i,
                total_budget=sampling.get("total_budget", 10),
                solver=sampling.get("solver", "euler"),
                lengths=lengths,
                max_frames=sampling.get("max_frames", 400),
            )
            n = int(out.lengths[0])
            prediction = self.codec.decode(out.rotations[0, :n].cpu(), out.expression[0, :n].cpu())
            result.append(prediction.numpy())
            self.last_generation_metadata.append(out.metadata)
        return result

    def save(self):
        save_checkpoint(self.ctx.ckpt_dir / "coplan_so3.pt", self.model, self.hparams, stage="C")

    def load(self, ckpt_dir):
        # Build from checkpoint config: reject a different manifest / posterior law
        # rather than accepting shape-compatible but semantically different motion.
        path = Path(ckpt_dir) / "coplan_so3.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.hparams = payload["config"]
        self._setup()
        restore_checkpoint(path, self.model)
        self.model.eval()
