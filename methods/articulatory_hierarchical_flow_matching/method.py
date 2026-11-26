"""Gloss-free text-to-sign generation with complementary conditional flows."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from functools import partial
from itertools import chain, islice
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from slp.config import load_yaml
from slp.data import Split
from slp.fingertip_fk import SigningEndpointLoss
from slp.method import BaseMethod
from slp.paths import PROJECT_ROOT
from slp.sign_text import TextEncoder
from slp.utils.torch_training import precision_context, train_stage

from .data import POSE_DIM, MotionBatch, MotionDataset, collate, text_input, train_monitor_indices
from .endpoint import COARSE_LOSSES, FK_CHANNELS, extra_fk_losses, jitter_coarse, rotation_loss
from .flow import rank_loss, sample, sample_noise, training_stages
from .model import MotionTransformer
from .planning import CONTENT_DIM, MODES as PLAN_MODES
from .planning import fit_planner, frame_condition, make_planner, predict_plans
from .retrieval import condition_retrieval, fit_index
from .semantic import fit_teacher
from .subspace import (
    GROUP_DIMS, MODES, DescriptorExtractor, Representation, calibrate, checksum, envelope,
)

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


def manifest_hash(names):
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()


class Method(BaseMethod):
    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        hp = self.hparams
        self.device = device_from(hp["device"])
        self.precision = hp["precision"]
        self.subspace = hp["subspace"]
        self.rotation_chart = self.subspace.get("rotation_chart", "axis_angle")
        defaults = load_yaml(Path(__file__).with_name("config.yaml"))["method"]
        self.flow = {**defaults["flow"], **hp["flow"]}
        self.sampler = hp["sampler"]
        self.semantic_cfg = {**defaults["semantic"], **hp.get("semantic", {})}
        self.planner_cfg = {**defaults["planner"], **hp.get("planner", {})}
        self.refinement = {**defaults["refinement"], **hp.get("refinement", {})}
        self.retrieval_cfg = {**defaults["retrieval"], **hp.get("retrieval", {})}
        self.text_cfg = {key: value for key, value in hp["text"].items() if key != "lr"}
        self.arch = {key: self.flow[key] for key in ("width", "heads", "depth", "dropout")}
        self.use_energy = self.flow["energy_weight"] > 0
        self._validate_config()

        mode, rank, stride = (self.subspace[key] for key in ("mode", "rank", "stride"))
        self.representation = Representation(mode, rank, stride,
                                             tangent=self.rotation_chart == "tangent").to(self.device)
        self.text = TextEncoder(**self.text_cfg).to(self.device)
        self.use_plan = self.planner_cfg["mode"] != "off"
        self.use_retrieval = self.retrieval_cfg["mode"] != "off"
        self.model = MotionTransformer(
            self.text.output_dim, **self.arch, energy_rank=rank if self.use_energy else None,
            plan_dim=CONTENT_DIM * 3 if self.use_plan else None,
            retrieval=self.use_retrieval,
        ).to(self.device)
        self.planner = None
        self.planner_text = None
        self.plan_cache = {}
        self.plan_stops = {}
        self.retrieval_text = self.retrieval_index = None
        if self.use_plan:
            self.planner = make_planner(self.text.output_dim, self.sampler["max_frames"],
                                        self.planner_cfg).to(self.device)
            self.planner_text = (deepcopy(self.text).eval().requires_grad_(False)
                                 if self.text.trained else self.text)
        if self.use_retrieval:
            self.retrieval_text = (deepcopy(self.text).eval().requires_grad_(False)
                                   if self.text.trained else self.text)
        self.calibration = None
        self.ready = False
        self.fk_loss = None
        self.teacher = None
        self.bank = None
        self.semantic_weight = 0.0
        self.jitter_strength = 0.0

    def _validate_config(self):
        subspace, flow, sampler = self.subspace, self.flow, self.sampler
        if subspace["mode"] not in MODES or not 1 <= subspace["rank"] <= sum(GROUP_DIMS):
            raise ValueError("Invalid subspace mode or rank")
        if self.rotation_chart not in ("axis_angle", "tangent"):
            raise ValueError("Choose the axis_angle or tangent rotation chart")
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
                       for k in ("duration_weight", "energy_weight", "endpoint_fk_weight",
                                 *(f"{name}_weight" for name in COARSE_LOSSES)))):
            raise ValueError("Invalid training precision, rates, or loss weights")
        if (len(sampler["steps"]) != 2 or any(n < 1 for n in sampler["steps"])
                or not 1 <= sampler["min_frames"] <= sampler["max_frames"]
                or sampler["batch_size"] < 1 or not math.isfinite(sampler["cfg_scale"])
                or sampler["cfg_scale"] < 0):
            raise ValueError("Invalid sampler options")
        self._validate_auxiliary_options()

    def _validate_auxiliary_options(self):
        semantic, planner, refinement = self.semantic_cfg, self.planner_cfg, self.refinement
        retrieval = self.retrieval_cfg
        if semantic["mode"] not in ("off", "sentence", "word"):
            raise ValueError("semantic.mode must be off, sentence, or word")
        if planner["mode"] not in PLAN_MODES or planner["duration_source"] not in ("flow", "planner"):
            raise ValueError("Invalid planner mode or duration source")
        if retrieval["mode"] not in ("off", "nearest", "random") or retrieval["anchors"] < 2:
            raise ValueError("Invalid train-only retrieval options")
        if (any(x["mode"] != "off" for x in (semantic, planner, retrieval))
                and self.text_cfg["finetune"]):
            raise ValueError("Run the fine-tuned mBART control independently of the auxiliaries")
        if not 0 < semantic["monitor_fraction"] < 0.5:
            raise ValueError("semantic.monitor_fraction must be between 0 and 0.5")
        if semantic["mode"] != "off" and (
            min(semantic[k] for k in ("window", "width", "max_words", "negatives", "text_batch",
                                      "calibration_microbatches")) < 1
            or not 0 < semantic["gradient_ratio"] <= 1 or not 0 < semantic["max_df"] <= 1
            or not -1 <= semantic["min_retrieval_gain"] <= 1
            or not -1 < semantic["null_similarity"] < 1
            or any(not math.isfinite(semantic[k]) or semantic[k] <= 0
                   for k in ("temperature", "alignment_temperature"))
        ):
            raise ValueError("Invalid semantic teacher options")
        if planner["mode"] != "off":
            stride = planner["frame_stride"]
            minimum, target, maximum = (planner[k] for k in ("min_frames", "target_frames", "max_frames"))
            if (not 1 <= stride <= minimum <= target <= maximum or minimum % stride
                    or maximum % stride or maximum < 2 * minimum - stride
                    or min(planner[k] for k in ("width", "heads", "depth", "max_units")) < 1
                    or planner["width"] % planner["heads"]
                    or planner["max_units"] < max(1, math.ceil(self.sampler["max_frames"] / stride)
                                                  // (minimum // stride))):
                raise ValueError("Invalid planner capacity or span support")
        elif planner["duration_source"] == "planner":
            raise ValueError("Planner durations require an active planner")
        for aux in (semantic, planner):
            if aux["mode"] != "off" and (
                min(aux["updates"], aux["eval_every"]) < 1 or aux["warmup_updates"] < 0
                or any(not math.isfinite(aux[k]) or aux[k] <= 0 for k in ("lr", "grad_clip"))
                or not 0 <= aux["min_lr_ratio"] <= 1
            ):
                raise ValueError("Invalid auxiliary training schedule")
        jitter = refinement["jitter_scale"]
        if (not math.isfinite(jitter) or jitter < 0 or (jitter and (
            self.subspace["mode"] == "single" or not refinement["reference_run"]
            or not refinement["reference_checkpoint"] or refinement["calibration_clips"] < 1
        ))):
            raise ValueError("Coarse jitter needs a two-stage model and a train calibration checkpoint")
        if refinement["reference_train_only"] and (
            jitter or any(x["mode"] != "off" for x in (semantic, planner, retrieval))
        ):
            raise ValueError("The out-of-fold refinement reference is a standalone baseline")

    def _loader(self, data, shuffle, *, calibrating=False):
        return DataLoader(
            data, batch_size=self.flow["batch_size"], shuffle=shuffle,
            num_workers=0 if calibrating else self.hparams["workers"],
            collate_fn=partial(collate, microbatch=self.flow["microbatch"]),
        )

    def _local_plan(self, keys, lengths, size):
        plans = predict_plans(self, keys)
        context = frame_condition(plans, lengths, size, self.planner_cfg["mode"], self.device)
        return context, plans

    def _loss(self, batch: MotionBatch, model, text, rng=None, *, calibrating=False):
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
        elif calibrating:
            stage_ids = (0,)
        u = time[:, None, None]
        flow_loss = state.new_zeros(())
        endpoint_loss = state.new_zeros(())
        extras = {}
        for stage_id in stage_ids:
            entry = stages[stage_id]
            start, target, rank, constrain = entry
            coarse = stages[0].target if stage_id else None
            if stage_id and rng is None and self.jitter_strength:
                altered = jitter_coarse(coarse, torch.randn_like(state), projector,
                                        self.jitter_strength)
                if self.subspace["unconstrained"]:
                    start = start + (altered - coarse) * rep.scales[0] / rep.scales[1]
                coarse = altered
            interpolated = torch.lerp(start, target, u)
            local = {}
            if self.use_plan and stage_id == 0:
                keys = list(zip(batch.texts, batch.languages.tolist(), strict=True))
                local["plan"], _ = self._local_plan(keys, batch.lengths, state.shape[1])
            if self.use_retrieval and stage_id == 0:
                local["retrieval"] = condition_retrieval(
                    self, batch.texts, batch.languages, batch.lengths, state.shape[1], batch.indices,
                )
            prediction = constrain(model(interpolated, time, stage_id, condition, batch.lengths,
                                         projector.mask, dropped, coarse, **local))
            flow_loss = flow_loss + rank_loss(prediction - (target - start), rank)
            if (self.fk_loss is not None and rng is None and stage_id == len(stages) - 1
                    and self.flow["endpoint_fk_weight"]):
                endpoint = (interpolated + (1 - u) * prediction) * rep.scales[stage_id]
                if stage_id and not self.subspace["unconstrained"]:
                    endpoint = endpoint + stages[0].target * rep.scales[0]
                with torch.autocast(self.device.type, enabled=False):
                    wrist, fingers = self.fk_loss(rep.decode(endpoint), batch.poses, batch.lengths,
                                                  self.flow["endpoint_fk_frames"])
                endpoint_loss = (wrist + fingers) * len(stages)
            if stage_id == 0 and (rng is None or calibrating):
                extras.update(self._coarse_losses(batch, entry, interpolated, prediction, u,
                                                  dropped, calibrating))
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
        if self.fk_loss is not None and rng is None and self.flow["endpoint_fk_weight"]:
            total = total + self.flow["endpoint_fk_weight"] * endpoint_loss
            losses["endpoint_fk"] = endpoint_loss
        for name, value in extras.items():
            weight = self.semantic_weight if name == "semantic" else self.flow[f"{name}_weight"]
            total = total + weight * value
        losses.update(extras)
        return {**losses, "loss": total}

    def _coarse_losses(self, batch, entry, interpolated, velocity, time, dropped, calibrating):
        semantic = self.teacher is not None
        fk = not calibrating and self.flow["coarse_fk_weight"]
        rotation = not calibrating and self.flow["coarse_rotation_weight"]
        channels = tuple(name for name in FK_CHANNELS if not calibrating
                         and self.flow[f"{name}_weight"])
        if not (semantic or fk or rotation or channels):
            return {}
        rep = self.representation
        losses = {}
        with torch.autocast(self.device.type, enabled=False):
            endpoint = entry.constrain(interpolated + (1 - time) * velocity) * rep.scales[0]
            poses = rep.decode(endpoint)
            if semantic:
                keys = self.bank.candidates(batch.indices)
                loss, _, _, _ = self.teacher.objective(poses, batch.lengths, keys, self.bank,
                                                       frozen=True, active=~dropped)
                losses["semantic"] = loss
            if fk or rotation or channels:
                target = rep.decode(entry.target * rep.scales[0])
                samples = self.flow["endpoint_fk_frames"]
                if fk:
                    wrist, fingers = self.fk_loss(poses, target, batch.lengths, samples,
                                                  shoulder_relative=True)
                    losses["coarse_fk"] = wrist + fingers
                if rotation:
                    losses["coarse_rotation"] = rotation_loss(poses, target, batch.lengths,
                                                               self.fk_loss, samples)
                if channels:
                    losses.update(extra_fk_losses(poses, target, batch.lengths, self.fk_loss,
                                                  samples, channels))
        return losses

    def _calibrate_semantic_weight(self, loader):
        params = tuple(self.model.parameters())
        limit = self.semantic_cfg["calibration_microbatches"]
        rng = torch.Generator(device=self.device).manual_seed(self.ctx.seed + 1277)
        gradient_sq = torch.zeros(2, device=self.device)

        def norm_sq(grads):
            used = [g.detach().float().square().sum() for g in grads if g is not None]
            return torch.stack(used).sum()

        self.model.eval()
        self.text.eval()
        count = 0
        for batch in islice(chain.from_iterable(loader), limit):
            with precision_context(self.device, self.precision):
                values = self._loss(batch.to(self.device), self.model, self.text, rng,
                                    calibrating=True)
            for i, key in enumerate(("flow", "semantic")):
                gradients = torch.autograd.grad(values[key], params, retain_graph=i == 0,
                                                allow_unused=True)
                gradient_sq[i] += norm_sq(gradients)
            count += 1
        flow_norm, semantic_norm = (gradient_sq / count).sqrt().tolist()
        if not math.isfinite(flow_norm + semantic_norm) or min(flow_norm, semantic_norm) < 1e-8:
            raise ValueError("Cannot calibrate the semantic loss from degenerate coarse gradients")
        self.semantic_weight = self.semantic_cfg["gradient_ratio"] * flow_norm / semantic_norm
        return {"weight": self.semantic_weight, "coarse_cfm_grad_rms": flow_norm,
                "semantic_grad_rms": semantic_norm, "microbatches": count}

    @torch.no_grad()
    def _calibrate_jitter(self, data, kept, held):
        source = PROJECT_ROOT / Path(self.refinement["reference_run"]).expanduser()
        path = source / "checkpoints" / self.refinement["reference_checkpoint"]
        cfg = load_yaml(source / "config.yaml")
        if cfg.get("method_name") != self.cfg["method_name"] or cfg.get("eval_split") != "val":
            raise ValueError("Coarse jitter requires a reference run selected on dev")
        if cfg["data"] != self.cfg["data"]:
            raise ValueError("Coarse jitter reference and current training data must match")
        cfg["method"]["device"] = str(self.device)
        reference = Method(cfg, self.ctx)
        if reference.subspace["mode"] == "single":
            raise ValueError("Coarse jitter needs a hierarchical reference flow")
        reference.load_checkpoint(path)
        training_ids = [str(data.split.inputs[data.indices[i]].get("name", data.indices[i]))
                        for i in kept]
        if (reference.calibration["manifest_sha256"] != manifest_hash(training_ids)
                or reference.calibration.get("train_source_manifest_sha256")
                != self.calibration["manifest_sha256"]):
            raise ValueError("Train an out-of-fold jitter reference with reference_train_only=true")
        if reference.calibration["rig"] != self.calibration["rig"]:
            raise ValueError("Coarse jitter reference rig differs")
        index = np.linspace(0, len(held) - 1, min(self.refinement["calibration_clips"], len(held)),
                            dtype=int)
        rows = [data.indices[held[i]] for i in index]
        inputs = [data.split.inputs[i] for i in rows]
        targets = [data.split.targets[i] for i in rows]
        poses = reference.generate(inputs, lengths=list(map(len, targets)), component="coarse",
                                   noise_seed=self.ctx.seed + 2017)
        rms = []
        rep = self.representation
        for pred, ref in zip(poses, targets, strict=True):
            lengths = torch.tensor([len(ref)], device=self.device)
            projector = rep.projector(lengths, len(ref))
            pairs = torch.as_tensor(np.stack((pred, ref)), device=self.device)
            a, b = rep.encode(pairs)
            delta = (a - b)[None]
            error = projector.coarse(delta) / rep.scales[0]
            rms.append((error.square().sum() / projector.coarse_rank[0]).sqrt().item())
        median = float(np.median(rms))
        self.jitter_strength = median * self.refinement["jitter_scale"]
        return {"rms_median": median, "strength": self.jitter_strength,
                "checkpoint_sha256": checksum(path), "ids": [item["name"] for item in inputs]}

    def fit(self, train: Split, val: Split) -> None:
        self.ready = False
        self.teacher = self.bank = None
        self.semantic_weight = self.jitter_strength = 0.0
        self.plan_cache.clear()
        self.plan_stops.clear()
        limit = self.hparams["debug_limit"]
        train_data = MotionDataset(train, limit)
        val_data = MotionDataset(val, limit or self.hparams["val_limit"])
        aux = self.semantic_cfg["mode"] != "off" or self.use_plan or self.use_retrieval
        reference_only = self.refinement["reference_train_only"]
        manifest = [str(train.inputs[i].get("name", i)) for i in train_data.indices]
        if aux or reference_only or self.refinement["jitter_scale"]:
            kept, held = train_monitor_indices(train_data, self.semantic_cfg["monitor_fraction"])
        if reference_only:
            source_manifest = manifest_hash(manifest)
            train_data.indices = [train_data.indices[i] for i in kept]
            manifest = [manifest[i] for i in kept]
        train_data.include_indices = aux
        descriptors = None
        stats = self.cfg["data"].get("stats")
        if stats:
            stats = PROJECT_ROOT / Path(stats)
        if self.subspace["mode"] == "articulatory" or self.use_energy or self.use_plan:
            fk_device = self.subspace["fk_device"]
            fk_device = self.device if fk_device == "auto" else torch.device(fk_device)
            descriptors = DescriptorExtractor(stats, fk_device, self.subspace["fk_chunk"])
        measure_fk = self.flow["endpoint_fk_weight"] or any(
            self.flow[f"{name}_weight"] for name in COARSE_LOSSES
        )
        if measure_fk or self.representation.chart is not None:
            self.fk_loss = SigningEndpointLoss(stats).to(self.device)
        if self.representation.chart is not None:
            self.representation.chart.fit(self.fk_loss, self._loader(train_data, False,
                                                                       calibrating=True))
        if not measure_fk:
            self.fk_loss = None
        self.ctx.logger.info("Calibrating from %d train clips", len(train_data))
        self.calibration = calibrate(self.representation, self._loader(
            train_data, False, calibrating=True,
        ), self.subspace, descriptors, manifest, self.use_energy)
        self.ctx.logger.info("Calibrated %d frames, coarse fraction %.4f",
                             self.calibration["frames"], self.calibration["coarse_fraction"])
        if reference_only:
            self.calibration["train_source_manifest_sha256"] = source_manifest
        if aux or self.refinement["jitter_scale"]:
            # Auxiliary stages have their own seed stream. Common flow initial weights and
            # stochastic draws otherwise keep the seed used by the historical baseline.
            devices = [self.device] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.ctx.seed + 1709)
                if devices:
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(self.ctx.seed + 1709)
                self._fit_auxiliaries(train_data, kept, held, descriptors)
        del descriptors
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

    def _fit_auxiliaries(self, data, kept, held, descriptors):
        fit_subset = Subset(data, kept)
        fit = self._loader(fit_subset, True, calibrating=True)
        monitor = self._loader(Subset(data, held), False, calibrating=True)
        self.calibration["auxiliary_train_split"] = {
            "fit_clips": len(kept), "monitor_clips": len(held),
            "fit_sha256": manifest_hash([str(data.split.inputs[data.indices[i]]["name"])
                                        for i in kept]),
            "monitor_sha256": manifest_hash([str(data.split.inputs[data.indices[i]]["name"])
                                            for i in held]),
        }
        if self.use_plan:
            self.calibration["planner"] = fit_planner(self, data, kept, fit, monitor, descriptors)
        if self.use_retrieval:
            self.retrieval_index = fit_index(self, data)
            self.calibration["retrieval"] = {
                "train_manifest_sha256": self.calibration["manifest_sha256"],
                "clips": len(data), "anchors": self.retrieval_cfg["anchors"],
                "source": "normalized_coarse_train_keyframes",
            }
        if self.semantic_cfg["mode"] != "off":
            self.teacher, self.bank, metadata = fit_teacher(self, data, kept, held, fit, monitor)
            metadata.update(self._calibrate_semantic_weight(
                self._loader(fit_subset, True, calibrating=True),
            ))
            teacher = self.ctx.ckpt_dir / "semantic_teacher.pt"
            atomic_write(teacher, partial(torch.save, {
                "settings": self.semantic_cfg, "monitor": metadata,
                "teacher": self.teacher.state_dict(),
            }))
            metadata["teacher_sha256"] = checksum(teacher)
            self.calibration["semantic"] = metadata
            self.ctx.logger.info("Semantic loss weight %.5g, monitor retrieval %.4f (length %.4f)",
                                 self.semantic_weight, metadata["monitor"]["retrieval_top1"],
                                 metadata["monitor"]["length_only_top1"])
        if self.refinement["jitter_scale"]:
            self.calibration["refinement"] = self._calibrate_jitter(data, kept, held)

    def _signature(self):
        signature = {"architecture": self.arch, "text": self.text_cfg, "energy": self.use_energy,
                     **{key: self.subspace[key]
                        for key in ("mode", "stride", "rank", "unconstrained")}}
        if self.representation.chart is not None:
            signature["rotation_chart"] = self.rotation_chart
        if self.use_retrieval:
            signature["retrieval"] = self.retrieval_cfg
        if self.use_plan:
            signature["planner"] = {key: self.planner_cfg[key] for key in (
                "mode", "duration_source", "width", "heads", "depth", "frame_stride",
                "max_units", "max_frames",
            )}
            signature["planner"]["capacity"] = self.planner.max_latents
        return signature

    def _write(self, path, model, text):
        state = {"version": CHECKPOINT_VERSION, "signature": self._signature(),
                 "calibration": self.calibration, "representation": self.representation.state_dict(),
                 "model": model.state_dict(), "text": text.state_dict() if text.trained else None}
        if self.use_plan:
            state["planner"] = self.planner.state_dict()
            if self.planner_text is not self.text:
                state["planner_text"] = self.planner_text.state_dict()
        if self.use_retrieval:
            state["retrieval_index"] = self.retrieval_index
            if self.retrieval_text is not self.text:
                state["retrieval_text"] = self.retrieval_text.state_dict()
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
        if self.use_plan:
            self.planner.load_state_dict(checkpoint["planner"])
            self.planner.eval().requires_grad_(False)
            if self.planner_text is not self.text:
                self.planner_text.load_state_dict(checkpoint["planner_text"])
                self.planner_text.eval()
            self.plan_cache.clear()
            self.plan_stops.clear()
        if self.use_retrieval:
            self.retrieval_index = {k: v.to(self.device)
                                    for k, v in checkpoint["retrieval_index"].items()}
            if self.retrieval_text is not self.text:
                self.retrieval_text.load_state_dict(checkpoint["retrieval_text"])
                self.retrieval_text.eval()
        self.teacher = self.bank = None
        self.semantic_weight = self.jitter_strength = 0.0
        self.calibration = checkpoint["calibration"]
        self.model.eval()
        self.text.eval()
        self.ready = True

    def _predict_lengths(self, condition):
        lower, upper = (self.sampler[key] for key in ("min_frames", "max_frames"))
        return condition.log_length.clamp(math.log(lower), math.log(upper)).exp().round().long()

    def _native_lengths(self, condition, texts, languages):
        if self.use_plan and self.planner_cfg["duration_source"] == "planner":
            keys = list(zip(texts, languages, strict=True))
            stride = self.planner_cfg["frame_stride"]
            return torch.tensor([sum(d) * stride for _, d in predict_plans(self, keys)],
                                device=self.device).clamp(self.sampler["min_frames"],
                                                          self.sampler["max_frames"])
        return self._predict_lengths(condition)

    @torch.inference_mode()
    def predict_durations(self, inputs):
        """Native text-only lengths, independently of a diagnostic archive or pose sampling."""
        if not self.ready:
            raise RuntimeError("Fit or load the model before prediction")
        batch_size = self.sampler["batch_size"]
        result = []
        with precision_context(self.device, self.precision):
            for start in range(0, len(inputs), batch_size):
                texts, language_ids = zip(*(text_input(item)
                                            for item in inputs[start:start + batch_size]),
                                          strict=True)
                lang = torch.tensor(language_ids, device=self.device)
                condition = self.model.prepare(*self.text(list(texts), lang), lang)
                result.extend(self._native_lengths(condition, texts, language_ids).tolist())
        return result

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
                if value.ndim == 0 and np.issubdtype(value.dtype, np.integer):
                    length = int(value)
                elif value.ndim == 2 and value.shape[1] == POSE_DIM:
                    length = len(value)
                else:
                    raise ValueError(f"Invalid prediction for {key!r} in {path}: {value.shape}")
                if not self.sampler["min_frames"] <= length <= self.sampler["max_frames"]:
                    raise ValueError(f"Invalid prediction for {key!r} in {path}: {value.shape}")
                lengths.append(length)
        return lengths

    @torch.inference_mode()
    def generate(self, inputs, *, lengths=None, noise_seed=None, oracle_targets=None,
                 oracle_component=None, component="full"):
        """Generate predictions and allow explicit component interventions for dev diagnostics."""
        if not self.ready:
            raise RuntimeError("Fit or load the model before prediction")
        if not len(inputs):
            return []
        if component not in ("full", "coarse") or (
            component == "coarse" and oracle_component == "residual"
        ):
            raise ValueError("Invalid generated component for this intervention")
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
                frame_counts = (self._native_lengths(condition, texts, language_ids)
                                if lengths is None else torch.tensor(lengths[start:stop],
                                                                      device=self.device))
                local = {}
                if self.use_plan:
                    texts_and_languages = list(zip(texts, language_ids, strict=True))
                    local["plan"], _ = self._local_plan(texts_and_languages, frame_counts,
                                                        int(frame_counts.max()))
                if self.use_retrieval:
                    local["retrieval"] = condition_retrieval(
                        self, list(texts), lang, frame_counts, int(frame_counts.max()),
                    )
                counts = frame_counts.tolist()
                keys = [f"{lang_id}/{name}" for lang_id, name in zip(language_ids, names[start:stop])]
                noise = sample_noise(keys, counts, seed, self.device)
                projector = self.representation.projector(frame_counts, noise.shape[1])
                kept = torch.zeros(len(noise), device=self.device, dtype=torch.bool)

                def velocity(state, time, stage, coarse):
                    kwargs = local if stage == 0 else {}
                    cond = self.model(state, time, stage, condition, frame_counts,
                                      projector.mask, kept, coarse, **kwargs)
                    if guidance == 1:
                        return cond
                    uncond = self.model(state, time, stage, condition, frame_counts,
                                        projector.mask, ~kept, coarse, **kwargs)
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
                                self.subspace["unconstrained"], coarse_only=component == "coarse",
                                **kwargs)
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
