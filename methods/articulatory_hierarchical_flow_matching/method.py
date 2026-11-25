"""Gloss-free text-to-sign generation with complementary conditional flows."""

from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from slp.data import Split
from slp.fingertip_fk import SigningEndpointLoss
from slp.method import BaseMethod
from slp.paths import PROJECT_ROOT
from slp.sign_text import TextEncoder
from slp.utils.torch_training import precision_context, train_stage

from .data import POSE_DIM, MotionBatch, MotionDataset, collate, text_input
from .flow import rank_loss, sample, sample_noise, training_stages
from .model import MotionTransformer
from .subspace import GROUP_DIMS, MODES, DescriptorExtractor, Representation, calibrate, envelope

CHECKPOINT_VERSION = 1


def atomic_write(path, writer):
    stream = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temp = Path(stream.name)
    try:
        with stream:
            writer(stream)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def device_from(value: str) -> torch.device:
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value)


class Method(BaseMethod):
    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        hp = self.hparams
        self.device = device_from(hp["device"])
        self.precision = hp["precision"]
        self.subspace = hp["subspace"]
        self.flow = hp["flow"]
        self.sampler = hp["sampler"]
        self.text_cfg = {key: value for key, value in hp["text"].items() if key != "lr"}
        self.arch = {key: self.flow[key] for key in ("width", "heads", "depth", "dropout")}
        self.use_energy = self.flow["energy_weight"] > 0
        self._validate_config()

        mode, rank, stride = (self.subspace[key] for key in ("mode", "rank", "stride"))
        self.representation = Representation(mode, rank, stride).to(self.device)
        self.text = TextEncoder(**self.text_cfg).to(self.device)
        self.model = MotionTransformer(
            self.text.output_dim, **self.arch, energy_rank=rank if self.use_energy else None,
        ).to(self.device)
        self.calibration = None
        self.ready = False
        self.fk_loss = None

    def _validate_config(self):
        subspace, flow, sampler = self.subspace, self.flow, self.sampler
        if subspace["mode"] not in MODES or not 1 <= subspace["rank"] <= sum(GROUP_DIMS):
            raise ValueError("Invalid subspace mode or rank")
        if subspace["mode"] == "single" and subspace["unconstrained"]:
            raise ValueError("Only a two-stage flow supports unconstrained refinement")
        if (subspace["stride"] < 1 or subspace["fk_chunk"] < 1
                or len(subspace["weights"]) != len(GROUP_DIMS)
                or any(not math.isfinite(w) or w < 0 for w in subspace["weights"])
                or not any(subspace["weights"])):
            raise ValueError("Invalid subspace stride, FK chunk size, or descriptor weights")
        if (flow["width"] < 2 or flow["width"] % 2 or flow["heads"] < 1
                or flow["width"] % flow["heads"] or flow["depth"] < 1):
            raise ValueError("Width must be even and divisible by the number of attention heads")
        positive = ("updates", "batch_size", "microbatch", "eval_every", "endpoint_fk_frames")
        if (any(flow[k] < 1 for k in positive) or self.hparams["workers"] < 0
                or any(self.hparams[k] is not None and self.hparams[k] < 1
                       for k in ("debug_limit", "val_limit"))):
            raise ValueError("Invalid training sizes or limits")
        if (self.precision not in ("bf16", "fp32") or not 0 <= flow["cond_drop"] < 1
                or not 0 <= flow["ema_decay"] < 1 or flow["warmup_updates"] < 0
                or any(not math.isfinite(flow[k]) or flow[k] < 0
                       for k in ("duration_weight", "energy_weight", "endpoint_fk_weight"))):
            raise ValueError("Invalid training precision, rates, or loss weights")
        if (len(sampler["steps"]) != 2 or any(n < 1 for n in sampler["steps"])
                or not 1 <= sampler["min_frames"] <= sampler["max_frames"]
                or sampler["batch_size"] < 1 or not math.isfinite(sampler["cfg_scale"])
                or sampler["cfg_scale"] < 0):
            raise ValueError("Invalid sampler options")

    def _loader(self, data, shuffle, *, calibrating=False):
        return DataLoader(
            data, batch_size=self.flow["batch_size"], shuffle=shuffle,
            num_workers=0 if calibrating else self.hparams["workers"],
            collate_fn=partial(collate, microbatch=self.flow["microbatch"]),
        )

    def _loss(self, batch: MotionBatch, model, text, rng=None):
        encoded, text_mask = text(batch.texts, batch.languages)
        condition = model.prepare(encoded, text_mask, batch.languages)
        duration = F.l1_loss(condition.log_length, batch.lengths.float().log())
        rep = self.representation
        projector = rep.projector(batch.lengths, batch.poses.shape[1])
        state = rep.encode(batch.poses).masked_fill(~projector.mask[..., None], 0)
        noise = torch.randn(state.shape, device=self.device, generator=rng)
        noise = noise.masked_fill(~projector.mask[..., None], 0)
        stages = training_stages(state, noise, projector, rep.scales, batch.lengths,
                                 self.subspace["unconstrained"])
        time = torch.rand(len(state), device=self.device, generator=rng)
        dropped = torch.zeros(len(state), device=self.device, dtype=torch.bool)
        stage_ids = range(len(stages))
        if rng is None:
            dropped = torch.rand_like(time) < self.flow["cond_drop"]
            stage_ids = (int(torch.randint(len(stages), ())),)
        u = time[:, None, None]
        flow_loss = state.new_zeros(())
        endpoint_loss = state.new_zeros(())
        for stage_id in stage_ids:
            entry = stages[stage_id]
            start, target, rank, constrain = entry
            coarse = stages[0].target if stage_id else None
            interpolated = torch.lerp(start, target, u)
            prediction = constrain(model(interpolated, time, stage_id, condition, batch.lengths,
                                         projector.mask, dropped, coarse))
            flow_loss = flow_loss + rank_loss(prediction - (target - start), rank)
            if self.fk_loss is not None and rng is None and stage_id == len(stages) - 1:
                endpoint = (interpolated + (1 - u) * prediction) * rep.scales[stage_id]
                if stage_id and not self.subspace["unconstrained"]:
                    endpoint = endpoint + coarse * rep.scales[0]
                with torch.autocast(self.device.type, enabled=False):
                    wrist, fingers = self.fk_loss(rep.decode(endpoint), batch.poses, batch.lengths,
                                                  self.flow["endpoint_fk_frames"])
                endpoint_loss = (wrist + fingers) * len(stages)
        flow_loss = flow_loss / len(stage_ids)
        losses = {"flow": flow_loss, "duration": duration}
        losses["length_mae"] = F.l1_loss(self._predict_lengths(condition).float(),
                                         batch.lengths.float())
        total = flow_loss + self.flow["duration_weight"] * duration
        if self.use_energy:
            expected, valid = envelope(state, batch.lengths, rep)
            error = F.smooth_l1_loss(condition.energy / rep.energy_std,
                                     expected / rep.energy_std, reduction="none")
            energy_loss = (error.mean(-1) * valid).sum() / valid.sum().clamp_min(1)
            losses["energy"] = energy_loss
            total = total + self.flow["energy_weight"] * energy_loss
        if self.fk_loss is not None and rng is None:
            total = total + self.flow["endpoint_fk_weight"] * endpoint_loss
            losses["endpoint_fk"] = endpoint_loss
        return {**losses, "loss": total}

    def fit(self, train: Split, val: Split) -> None:
        self.ready = False
        limit = self.hparams["debug_limit"]
        train_data = MotionDataset(train, limit)
        val_data = MotionDataset(val, limit or self.hparams["val_limit"])
        descriptors = None
        stats = self.cfg["data"].get("stats")
        if stats:
            stats = PROJECT_ROOT / Path(stats)
        if self.subspace["mode"] == "articulatory" or self.use_energy:
            fk_device = self.subspace["fk_device"]
            fk_device = self.device if fk_device == "auto" else torch.device(fk_device)
            descriptors = DescriptorExtractor(stats, fk_device, self.subspace["fk_chunk"])
        self.ctx.logger.info("Calibrating from %d train clips", len(train_data))
        manifest = [str(train.inputs[i].get("name", i)) for i in train_data.indices]
        self.calibration = calibrate(self.representation, self._loader(
            train_data, False, calibrating=True,
        ), self.subspace, descriptors, manifest, self.use_energy)
        self.ctx.logger.info("Calibrated %d frames, coarse fraction %.4f",
                             self.calibration["frames"], self.calibration["coarse_fraction"])
        del descriptors
        if self.flow["endpoint_fk_weight"]:
            self.fk_loss = SigningEndpointLoss(stats).to(self.device)

        groups = [{"params": self.model.parameters()}]
        if self.text.trained:
            lr = self.hparams["text"]["lr"] if self.text_cfg["finetune"] else self.flow["lr"]
            groups.append({"params": self.text.parameters(), "lr": lr})
        optimizer = torch.optim.AdamW(groups, lr=self.flow["lr"],
                                      weight_decay=self.flow["weight_decay"])
        models = (self.model, self.text)
        ema_model = deepcopy(self.model).eval().requires_grad_(False)
        ema_text = deepcopy(self.text).eval().requires_grad_(False) if self.text.trained else self.text
        ema_pairs = [(ema_model, self.model)]
        if self.text.trained:
            ema_pairs.append((ema_text, self.text))

        @torch.no_grad()
        def update_ema():
            for average, trainable in ema_pairs:
                torch._foreach_lerp_(list(average.parameters()), list(trainable.parameters()),
                                     1 - self.flow["ema_decay"])

        def val_loss():
            rng = torch.Generator(device=self.device).manual_seed(self.ctx.seed + 1009)
            return partial(self._loss, model=ema_model, text=ema_text, rng=rng)

        def write(name):
            self._write(self.ctx.ckpt_dir / name, ema_model, ema_text)

        train_stage("flow", self._loader(train_data, True), self._loader(val_data, False),
                    self.flow, optimizer, models, partial(self._loss, model=self.model, text=self.text),
                    val_loss, lambda: write("model.pt"), self.device, self.precision, self.ctx,
                    after_step=update_ema,
                    save_snapshot=lambda step: write(f"model_step{step:06d}.pt"))
        self.load(self.ctx.ckpt_dir)
        self.fk_loss = None

    def _signature(self):
        return {"architecture": self.arch, "text": self.text_cfg, "energy": self.use_energy,
                **{key: self.subspace[key] for key in ("mode", "stride", "rank", "unconstrained")}}

    def _write(self, path, model, text):
        state = {"version": CHECKPOINT_VERSION, "signature": self._signature(),
                 "calibration": self.calibration, "representation": self.representation.state_dict(),
                 "model": model.state_dict(), "text": text.state_dict() if text.trained else None}
        atomic_write(path, partial(torch.save, state))

    def save(self) -> None:
        if not self.ready:
            raise RuntimeError("Fit or load the model before saving")
        self._write(self.ctx.ckpt_dir / "model.pt", self.model, self.text)

    def load(self, ckpt_dir) -> None:
        self.load_checkpoint(Path(ckpt_dir) / "model.pt")

    def load_checkpoint(self, path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("version") != CHECKPOINT_VERSION
                or checkpoint.get("signature") != self._signature()):
            raise ValueError(f"Incompatible model checkpoint: {path}")
        self.representation.load_state_dict(checkpoint["representation"])
        self.model.load_state_dict(checkpoint["model"])
        if self.text.trained:
            self.text.load_state_dict(checkpoint["text"])
        self.calibration = checkpoint["calibration"]
        self.model.eval()
        self.text.eval()
        self.ready = True

    def _predict_lengths(self, condition):
        lower, upper = (self.sampler[key] for key in ("min_frames", "max_frames"))
        return condition.log_length.clamp(math.log(lower), math.log(upper)).exp().round().long()

    def _length_archive(self, keys):
        path = self.sampler["lengths_from_npz"]
        if path is None:
            return None
        lengths = []
        with np.load(PROJECT_ROOT / Path(path).expanduser(), allow_pickle=False) as archive:
            for key in keys:
                if key not in archive:
                    raise ValueError(f"No prediction for {key!r} in {path}")
                value = archive[key]
                if value.ndim != 2 or value.shape[1] != POSE_DIM or not (
                    self.sampler["min_frames"] <= len(value) <= self.sampler["max_frames"]
                ):
                    raise ValueError(f"Invalid prediction for {key!r} in {path}: {value.shape}")
                lengths.append(len(value))
        return lengths

    @torch.inference_mode()
    def generate(self, inputs, *, lengths=None, noise_seed=None, oracle_targets=None,
                 oracle_component=None):
        """Generate predictions and allow explicit component interventions for dev diagnostics."""
        if not self.ready:
            raise RuntimeError("Fit or load the model before prediction")
        if not len(inputs):
            return []
        names = [str(item.get("name", i)) for i, item in enumerate(inputs)]
        if (oracle_targets is None) != (oracle_component is None):
            raise ValueError("Supply both oracle_targets and oracle_component")
        if oracle_targets is not None:
            if (oracle_component not in ("coarse", "residual") or len(oracle_targets) != len(inputs)
                    or self.subspace["mode"] == "single" or self.subspace["unconstrained"]):
                raise ValueError("Oracle interventions require projected two-stage flow")
            oracle_lengths = [len(target) for target in oracle_targets]
            if lengths is not None and list(lengths) != oracle_lengths:
                raise ValueError("An oracle component requires matching frame counts")
            lengths = oracle_lengths
        if lengths is not None and (len(lengths) != len(inputs) or any(
            not isinstance(n, (int, np.integer)) or not 1 <= n <= self.sampler["max_frames"]
            for n in lengths
        )):
            raise ValueError("Expected a valid integer frame count for each input")
        seed = self.ctx.seed if noise_seed is None else noise_seed
        predictions = []
        batch_size = self.sampler["batch_size"]
        guidance = self.sampler["cfg_scale"]
        with precision_context(self.device, self.precision):
            for start in range(0, len(inputs), batch_size):
                stop = min(start + batch_size, len(inputs))
                texts, language_ids = zip(*(text_input(item) for item in inputs[start:stop]))
                lang = torch.tensor(language_ids, device=self.device)
                condition = self.model.prepare(*self.text(list(texts), lang), lang)
                frame_counts = self._predict_lengths(condition) if lengths is None else torch.tensor(
                    lengths[start:stop], device=self.device,
                )
                counts = frame_counts.tolist()
                keys = [f"{lang_id}/{name}" for lang_id, name in zip(language_ids, names[start:stop])]
                noise = sample_noise(keys, counts, seed, self.device)
                projector = self.representation.projector(frame_counts, noise.shape[1])
                kept = torch.zeros(len(noise), device=self.device, dtype=torch.bool)

                def velocity(state, time, stage, coarse):
                    cond = self.model(state, time, stage, condition, frame_counts,
                                      projector.mask, kept, coarse)
                    if guidance == 1:
                        return cond
                    uncond = self.model(state, time, stage, condition, frame_counts,
                                        projector.mask, ~kept, coarse)
                    return uncond + guidance * (cond - uncond)

                kwargs = {}
                if oracle_targets is not None:
                    rows = [torch.as_tensor(row, dtype=torch.float32, device=self.device)
                            for row in oracle_targets[start:stop]]
                    if any(row.ndim != 2 or row.shape[-1] != POSE_DIM or not torch.isfinite(row).all()
                           for row in rows):
                        raise ValueError("Oracle targets must be finite [T,133] poses")
                    target = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
                    kwargs[f"{oracle_component}_override"] = self.representation.encode(target)
                result = sample(velocity, noise, projector, self.representation, self.sampler["steps"],
                                self.subspace["unconstrained"], **kwargs)
                with torch.autocast(self.device.type, enabled=False):
                    result = self.representation.decode(result).cpu().numpy()
                if not np.isfinite(result[projector.mask.cpu().numpy()]).all():
                    raise RuntimeError("Non-finite generated pose")
                predictions.extend(row[:n].copy() for row, n in zip(result, counts, strict=True))
        return predictions

    @staticmethod
    def prediction_keys(inputs):
        keys = [str(item.get("name", i)).replace("/", "__") for i, item in enumerate(inputs)]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate IDs after prediction key normalization")
        return keys

    def predict(self, inputs):
        keys = self.prediction_keys(inputs) if (self.hparams["export_predictions"]
                                               or self.sampler["lengths_from_npz"]) else []
        predictions = self.generate(inputs, lengths=self._length_archive(keys))
        if self.hparams["export_predictions"]:
            arrays = dict(zip(keys, predictions, strict=True))
            atomic_write(self.ctx.run_dir / "predictions.npz",
                         lambda f: np.savez_compressed(f, **arrays))
        return predictions
