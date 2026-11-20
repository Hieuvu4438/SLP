"""Apply MotionHiFlow to gloss-free text-to-sign on the repository's SMPL-X benchmarks."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from slp.data import Split
from slp.method import BaseMethod
from slp.paths import PROJECT_ROOT

from .codec import MotionVAE, codec_losses, sequence_mean
from .data import PART_DIMS, POSE_DIM, MotionBatch, MotionDataset, ceil_div, text_input
from .dit import MotionDiT
from .pyramid import PyramidFlow
from .text import TextEncoder
from .training import calibrate, make_loader, precision_context, train_stage

CHECKPOINT_VERSION = 1
CODEC_KEYS = ("width", "latent_dim", "depth", "stride")
DIT_KEYS = ("width", "heads", "double_blocks", "single_blocks", "dropout", "time_patch")
TEXT_KEYS = ("backend", "model", "max_tokens", "finetune", "byte_dim")


def atomic_write(path: Path, write: Callable) -> None:
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temp = Path(file.name)
        try:
            write(file)
            file.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


class Method(BaseMethod):
    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        hp = self.hparams
        requested = hp["device"]
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        )
        self.precision = hp["precision"]
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("method.precision must be fp32 or bf16")
        self.codec_cfg = hp["codec"]
        self.flow_cfg = hp["flow"]
        self.sampler = hp["sampler"]
        self.codec_args = {k: self.codec_cfg[k] for k in CODEC_KEYS}
        self.text_args = {k: hp["text"][k] for k in TEXT_KEYS}
        self.dit_args = {k: self.flow_cfg[k] for k in DIT_KEYS}
        self.pyramid = PyramidFlow(self.flow_cfg["scales"], self.flow_cfg["gamma"])
        self._validate_config()
        self.ready = False

    def _validate_config(self) -> None:
        sampler = self.sampler
        self.pyramid.step_counts(sampler["steps"])
        if not 0 < sampler["min_frames"] <= sampler["max_frames"] or sampler["batch_size"] < 1:
            raise ValueError("Invalid sampler frame limits or batch size")
        if not math.isfinite(sampler["cfg_scale"]) or sampler["cfg_scale"] < 0:
            raise ValueError("sampler.cfg_scale must be finite and nonnegative")
        if not 0 <= self.flow_cfg["cond_drop"] < 1 or not 0 <= self.flow_cfg["ema_decay"] < 1:
            raise ValueError("Invalid conditioning dropout or EMA decay")
        weights = self.codec_cfg["part_weights"]
        if len(weights) != len(PART_DIMS) or any(w <= 0 for w in weights):
            raise ValueError("codec.part_weights must contain four positive weights")
        if not 0 < self.codec_cfg["pool_min_scale"] <= 1:
            raise ValueError("Invalid codec.pool_min_scale")
        for stage in (self.codec_cfg, self.flow_cfg):
            if not 0 < stage["microbatch"] <= stage["batch_size"]:
                raise ValueError("Each microbatch must be between 1 and batch_size")
            if min(stage["updates"], stage["eval_every"]) < 1 or stage["warmup_updates"] < 0:
                raise ValueError("Each stage needs a positive update count and evaluation interval")

    def _codec(self) -> MotionVAE:
        return MotionVAE(**self.codec_args).to(self.device)

    def _flow(self) -> None:
        self.text = TextEncoder(**self.text_args).to(self.device)
        self.dit = MotionDiT(self.codec.latent_dim, self.text.output_dim, **self.dit_args)
        self.dit.to(self.device)

    def _loader(
        self, split: Split, stage: dict, *, limit: int | None, shuffle: bool,
    ) -> DataLoader:
        return make_loader(MotionDataset(split, limit), stage, self.hparams["workers"],
                           self.device, self.ctx.seed, shuffle)

    def _codec_state(self) -> dict:
        return {"version": CHECKPOINT_VERSION, "arch": self.codec_args,
                "model": self.codec.state_dict()}

    def _load_codec(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["version"] != CHECKPOINT_VERSION or checkpoint["arch"] != self.codec_args:
            raise ValueError(f"Codec architecture in {path} does not match this run")
        self.codec.load_state_dict(checkpoint["model"])

    def _fit_codec(self, train: Split, val: Split, train_limit, val_limit) -> float:
        cfg = self.codec_cfg
        codec_path = self.ctx.ckpt_dir / "codec.pt"
        self.codec = self._codec()
        init_codec = self.hparams["init_codec"]
        if init_codec:
            self._load_codec(PROJECT_ROOT / Path(init_codec).expanduser())
            self.ctx.logger.info("Loaded codec from %s", init_codec)
        else:
            optimizer = torch.optim.AdamW(self.codec.parameters(), lr=cfg["lr"],
                                          weight_decay=cfg["weight_decay"])

            def loss(batch):
                return codec_losses(self.codec, batch.poses, batch.lengths, cfg)

            train_stage(
                "codec", self._loader(train, cfg, limit=train_limit, shuffle=True),
                self._loader(val, cfg, limit=val_limit, shuffle=False), cfg,
                optimizer, (self.codec,), loss, lambda: loss,
                lambda: atomic_write(codec_path, lambda f: torch.save(self._codec_state(), f)),
                self.device, self.precision, self.ctx, select="reconstruction",
            )
            self._load_codec(codec_path)
        self.codec.eval().requires_grad_(False)
        limit = cfg["calibration_clips"] if train_limit is None else min(
            train_limit, cfg["calibration_clips"],
        )
        prior = calibrate(self.codec, self._loader(train, cfg, limit=limit, shuffle=False),
                          self.device, self.precision)
        atomic_write(codec_path, lambda f: torch.save(self._codec_state(), f))
        return prior

    def _flow_losses(
        self, batch: MotionBatch, dit: MotionDiT, validation_rng: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            clean, lengths = self.codec.flow_target(batch.poses, batch.lengths)
        tokens, text_mask = self.text(batch.texts, batch.languages)
        condition = dit.prepare_text(tokens, text_mask)
        duration = dit.log_duration(tokens, text_mask, batch.languages).float()
        duration_loss = F.l1_loss(duration, batch.lengths.float().log())
        duration_frames = duration.detach().clamp(
            max=math.log(self.sampler["max_frames"])
        ).exp().round().clamp_min(self.sampler["min_frames"])
        frame_error = F.l1_loss(duration_frames, batch.lengths.float())
        noise = torch.randn(clean.shape, device=self.device, generator=validation_rng)
        progress = torch.rand(len(clean), device=self.device, generator=validation_rng)
        drop = torch.zeros_like(progress, dtype=torch.bool)
        if validation_rng is None:
            drop = torch.rand_like(progress) < self.flow_cfg["cond_drop"]
            stage = torch.multinomial(self.pyramid.probabilities, 1).item()
            stages = [(stage, 1.0)]
        else:
            stages = [(i, stage.time_width) for i, stage in enumerate(self.pyramid.stages)]

        flow_loss = clean.new_zeros(())
        for i, weight in stages:
            scale = self.pyramid.stages[i].scale
            x, velocity, time, stage_lengths = self.pyramid.training_pair(
                clean, noise, lengths, i, progress,
            )
            predicted = dit(x, time, scale, stage_lengths, condition, batch.languages, drop)
            flow_loss = flow_loss + weight * sequence_mean(
                (predicted.float() - velocity).square(), stage_lengths
            )
        return {"loss": flow_loss + self.flow_cfg["duration_weight"] * duration_loss,
                "flow": flow_loss, "log_duration": duration_loss, "length_mae": frame_error}

    def _signature(self) -> dict:
        return {
            "codec": self.codec_args, "text": self.text_args, "dit": self.dit_args,
            "scales": self.flow_cfg["scales"], "gamma": self.flow_cfg["gamma"],
        }

    def _model_state(self, dit: MotionDiT) -> dict:
        return {
            "version": CHECKPOINT_VERSION, "signature": self._signature(),
            "codec": self.codec.state_dict(), "dit": dit.state_dict(),
            "text": self.text.state_dict() if self.text.trained else None,
        }

    def _restore_model(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["version"] != CHECKPOINT_VERSION
                or checkpoint["signature"] != self._signature()):
            raise ValueError(f"Checkpoint architecture in {path} does not match this run")
        self.codec.load_state_dict(checkpoint["codec"])
        self.dit.load_state_dict(checkpoint["dit"])
        if self.text.trained:
            self.text.load_state_dict(checkpoint["text"])
        self.codec.eval().requires_grad_(False)
        self.text.eval()
        self.dit.eval()
        self.ready = True

    def fit(self, train: Split, val: Split) -> None:
        self.ready = False
        hp = self.hparams
        train_limit = hp["debug_limit"]
        val_limit = min((limit for limit in (train_limit, hp["val_limit"]) if limit is not None),
                        default=None)
        self.ctx.logger.info("MotionHiFlow SLP on %s: %d train clips, %d validation clips",
                             self.device, len(train), len(val))
        prior = self._fit_codec(train, val, train_limit, val_limit)
        # Keep the flow initialization independent of codec training vs. reuse.
        torch.manual_seed(self.ctx.seed)
        self._flow()
        torch.nn.init.zeros_(self.dit.duration[-1].weight)
        torch.nn.init.constant_(self.dit.duration[-1].bias, prior)

        cfg = self.flow_cfg
        groups = [{"params": self.dit.parameters(), "lr": cfg["lr"]}]
        if self.text.trained:
            groups.append({"params": self.text.parameters(),
                           "lr": hp["text"]["lr"] if self.text_args["finetune"] else cfg["lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        ema = None
        if cfg["ema_decay"]:
            ema = AveragedModel(self.dit, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]))
            ema.eval().requires_grad_(False)
        evaluation_model = ema.module if ema is not None else self.dit

        def validation_loss():
            rng = torch.Generator(device=self.device).manual_seed(self.ctx.seed + 1)
            return lambda batch: self._flow_losses(batch, evaluation_model, rng)

        checkpoint = self.ctx.ckpt_dir / "model.pt"
        train_stage(
            "flow", self._loader(train, cfg, limit=train_limit, shuffle=True),
            self._loader(val, cfg, limit=val_limit, shuffle=False), cfg, optimizer,
            (self.text, self.dit), lambda batch: self._flow_losses(batch, self.dit),
            validation_loss,
            lambda: atomic_write(checkpoint,
                                 lambda f: torch.save(self._model_state(evaluation_model), f)),
            self.device, self.precision, self.ctx,
            after_step=None if ema is None else lambda: ema.update_parameters(self.dit),
        )
        self._restore_model(checkpoint)

    def save(self) -> None:
        # fit already stores the selected complete checkpoint atomically.
        if self.ready:
            path = self.ctx.ckpt_dir / "model.pt"
            atomic_write(path, lambda f: torch.save(self._model_state(self.dit), f))

    def load(self, ckpt_dir) -> None:
        self.ready = False
        self.codec = self._codec()
        self._flow()
        self._restore_model(Path(ckpt_dir) / "model.pt")

    def _noise(self, keys: list[str], lengths: Tensor) -> Tensor:
        """Give each utterance the same starting noise across batch sizes and scale ablations."""
        rows = []
        rng = torch.Generator(device=self.device)
        for key, length in zip(keys, lengths.tolist(), strict=True):
            seed = hashlib.blake2b(f"{self.ctx.seed}/{key}".encode(), digest_size=8).digest()
            rng.manual_seed(int.from_bytes(seed, "little"))
            rows.append(torch.randn((length, len(PART_DIMS), self.codec.latent_dim),
                                    device=self.device, generator=rng))
        return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)

    def _fixed_lengths(self, keys: list[str]) -> list[int] | None:
        path = self.sampler["lengths_from_npz"]
        if path is None:
            return None
        lengths = []
        with np.load(PROJECT_ROOT / Path(path).expanduser(), allow_pickle=False) as source:
            for name in keys:
                if name not in source:
                    raise ValueError(f"Missing predicted duration for {name!r} in {path}")
                clip = source[name]
                if (clip.ndim != 2 or clip.shape[1] != POSE_DIM
                        or not self.sampler["min_frames"] <= len(clip) <= self.sampler["max_frames"]):
                    raise ValueError(f"Invalid prediction shape {clip.shape} for {name!r} in {path}")
                lengths.append(len(clip))
        return lengths

    @torch.inference_mode()
    def predict(self, inputs):
        if not self.ready:
            raise RuntimeError("Fit or load MotionHiFlow SLP before prediction")
        if not inputs:
            return []
        settings = self.sampler
        batch_size = settings["batch_size"]
        cfg_scale = settings["cfg_scale"]
        predictions = []
        names = [str(item.get("name", i)) for i, item in enumerate(inputs)]
        keys = [name.replace("/", "__") for name in names]
        if self.hparams["export_predictions"] and len(set(keys)) != len(keys):
            raise ValueError("Duplicate sample keys after normalizing prediction names")
        fixed_lengths = self._fixed_lengths(keys)
        with precision_context(self.device, self.precision):
            for start in range(0, len(inputs), batch_size):
                batch = inputs[start:start + batch_size]
                texts, languages = zip(*(text_input(item) for item in batch))
                languages = torch.tensor(languages, device=self.device)
                tokens, mask = self.text(list(texts), languages)
                condition = self.dit.prepare_text(tokens, mask)
                if fixed_lengths is None:
                    log_lengths = self.dit.log_duration(tokens, mask, languages).float()
                    lengths = log_lengths.clamp(
                        math.log(settings["min_frames"]), math.log(settings["max_frames"])
                    ).exp().round().long()
                else:
                    lengths = torch.tensor(fixed_lengths[start:start + len(batch)], device=self.device)
                latent_lengths = ceil_div(lengths, self.codec.stride)
                rng_keys = [f"{lang}/{name}" for lang, name in zip(
                    languages.tolist(), names[start:start + len(batch)], strict=True,
                )]
                noise = self._noise(rng_keys, latent_lengths)
                conditioned = torch.zeros(len(batch), device=self.device, dtype=torch.bool)
                unconditioned = ~conditioned

                def velocity(x, time, scale, stage_lengths):
                    cond = self.dit(
                        x, time, scale, stage_lengths, condition, languages, conditioned
                    ).float()
                    if cfg_scale == 1:
                        return cond
                    uncond = self.dit(
                        x, time, scale, stage_lengths, condition, languages, unconditioned
                    ).float()
                    return uncond + cfg_scale * (cond - uncond)

                sampled = self.pyramid.sample(velocity, noise, latent_lengths, settings["steps"])
                frame_lengths = lengths.tolist()
                decoded = self.codec.decode_flow(sampled, lengths, max(frame_lengths)).cpu().numpy()
                predictions.extend(decoded[i, :length].copy()
                                   for i, length in enumerate(frame_lengths))

        if self.hparams["export_predictions"]:
            arrays = dict(zip(keys, predictions, strict=True))
            atomic_write(self.ctx.run_dir / "predictions.npz",
                         lambda f: np.savez_compressed(f, **arrays))
        return predictions
