"""SignRAE: MoRAE (arXiv 2607.29180) transferred to text-to-sign production.

fit():  Sign-JEPA pre-training (or ``init_jepa``) -> compact latent autoencoder (or ``init_ae``)
        -> train-split latent calibration + cache -> text-conditioned latent flow (DiT-M + mBART).
        ``init_flow`` warm-starts the generator from a flow pre-trained on the union of datasets
        (pretrain.py --stage flow) before fine-tuning on this dataset.
predict(): duration head -> seeded Gaussian noise -> guided ODE (RK4 by default) -> decoder ->
        normalised [T, 133] poses; ``sampler.ensemble`` K > 1 averages K samples in pose space
        (a point estimate of E[pose | text]). Metrics are computed by the shared evaluator only.
See PLAN.md for the design and README.md for the commands.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from slp.data import Split
from slp.method import BaseMethod

from .autoencoder import KINDS, LatentSpace
from .data import LatentDataset, POSE_DIM, subset_indices, text_input, valid_frames
from .dit import DurationHead, LatentDiT
from .flow import SOLVERS, cfg_combine, integrate, sample_time, training_pair
from .layers import masked_mean
from .stages import (atomic_save, encode_split, load_jepa, load_space, resolve, save_space,
                     space_spec, train_autoencoder, train_jepa)
from .text import TextEncoder
from .training import evaluate_loader, make_loader, make_optimizer, precision_context, train_stage

CHECKPOINT_VERSION = 1
DIT_KEYS = ("width", "depth", "heads", "ff", "dropout", "text_mode", "max_frames")
TEXT_KEYS = ("backend", "model", "max_tokens", "byte_dim")
STAGE_KEYS = ("batch_size", "microbatch", "updates", "warmup_updates", "eval_every", "lr",
              "min_lr_ratio", "betas", "weight_decay", "grad_clip")


class Generator(nn.Module):
    """Everything the flow stage trains (EMA is taken over this module)."""

    def __init__(self, text: TextEncoder, dit: LatentDiT, duration: DurationHead):
        super().__init__()
        self.text, self.dit, self.duration = text, dit, duration


class Method(BaseMethod):
    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        hp = self.hparams
        requested = hp["device"]
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                                   if requested == "auto" else requested)
        self.precision = hp["precision"]
        self.flow_cfg, self.sampler = hp["flow"], hp["sampler"]
        self.dit_args = {k: self.flow_cfg[k] for k in DIT_KEYS}
        self.text_args = {k: hp["text"][k] for k in TEXT_KEYS}
        self.text_finetune = bool(hp["text"].get("finetune", False))
        self.stats = cfg.get("data", {}).get("stats")
        self._validate()
        self.ready = False

    # ------------------------------------------------------------------ config
    def _validate(self) -> None:
        hp = self.hparams
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("method.precision must be fp32 or bf16")
        if hp["latent"]["kind"] not in KINDS:
            raise ValueError(f"method.latent.kind must be one of {KINDS}")
        for name in ("jepa", "ae", "flow"):
            stage = hp[name]
            missing = [k for k in STAGE_KEYS if k not in stage]
            if missing:
                raise ValueError(f"method.{name} is missing {missing}")
            if not 0 < stage["microbatch"] <= stage["batch_size"]:
                raise ValueError(f"method.{name}: need 0 < microbatch <= batch_size")
            if stage["updates"] < 1 or stage["eval_every"] < 1 or stage["warmup_updates"] < 0:
                raise ValueError(f"method.{name}: invalid update schedule")
        mask = hp["jepa"]["mask"]
        if not 0 < mask["ratio"] < 1:
            raise ValueError("method.jepa.mask.ratio must be in (0, 1)")
        if len(hp["ae"]["part_weights"]) != 4 or min(hp["ae"]["part_weights"]) <= 0:
            raise ValueError("method.ae.part_weights needs four positive weights")
        s = self.sampler
        if s["solver"] not in SOLVERS or s["steps"] < 1:
            raise ValueError(f"method.sampler.solver must be one of {SOLVERS}, steps >= 1")
        if not (math.isfinite(s["cfg_scale"]) and s["cfg_scale"] >= 0 and s["temperature"] >= 0):
            raise ValueError("Invalid sampler.cfg_scale or sampler.temperature")
        if not 0 < s["min_frames"] <= s["max_frames"] <= self.flow_cfg["max_frames"]:
            raise ValueError("Need 0 < sampler.min_frames <= max_frames <= flow.max_frames")
        if not 0 <= self.flow_cfg["cond_drop"] < 1 or not 0 <= self.flow_cfg["ema_decay"] < 1:
            raise ValueError("Invalid flow.cond_drop or flow.ema_decay")
        if self.flow_cfg["select"] not in ("last", "flow"):
            raise ValueError("method.flow.select must be last or flow")
        if int(s.get("ensemble", 1)) < 1:
            raise ValueError("method.sampler.ensemble must be >= 1")
        if not 0 <= self.flow_cfg.get("token_drop", 0.0) < 1:
            raise ValueError("method.flow.token_drop must be in [0, 1)")
        if hp["text"].get("lr_scale", 1.0) <= 0:
            raise ValueError("method.text.lr_scale must be positive")
        if hp.get("init_flow"):
            if not hp["init_ae"]:
                raise ValueError("method.init_flow needs method.init_ae (the same latent space)")
            if hp["latent"]["recalibrate"]:
                raise ValueError("method.init_flow needs latent.recalibrate=false: the pretrained "
                                 "flow was trained in the init_ae standardisation")

    def _limits(self) -> tuple[int | None, int | None]:
        train_limit = self.hparams["debug_limit"]
        val_limit = min((x for x in (train_limit, self.hparams["val_limit"]) if x is not None),
                        default=None)
        return train_limit, val_limit

    # ------------------------------------------------------------------ stages 1-2
    def _fit_space(self, train: Split, val: Split, train_limit, val_limit) -> None:
        hp, ckpt = self.hparams, self.ctx.ckpt_dir
        kind = hp["latent"]["kind"]
        if hp["init_ae"]:
            self.space, self.spec = load_space(hp["init_ae"], self.device,
                                               need_calibration=not hp["latent"]["recalibrate"])
            if self.spec["kind"] != kind or (kind != "raw"
                                              and self.spec["latent_dim"] != hp["latent"]["dim"]):
                raise ValueError(f"init_ae has kind={self.spec['kind']} "
                                 f"d={self.spec['latent_dim']}, config asks {kind} "
                                 f"d={hp['latent']['dim']}")
            self.ctx.logger.info("Loaded compact latent from %s", hp["init_ae"])
            return
        arch = None
        if kind == "jepa":
            if hp["init_jepa"]:
                jepa, arch = load_jepa(hp["init_jepa"], self.device)
                self.ctx.logger.info("Loaded Sign-JEPA from %s", hp["init_jepa"])
            else:
                jepa, arch = train_jepa(train, val, hp, self.device, self.precision, self.ctx,
                                        ckpt / "jepa.pt", train_limit, val_limit)
        self.spec = space_spec(hp, arch)
        torch.manual_seed(self.ctx.seed)
        self.space = LatentSpace(**self.spec).to(self.device)
        if kind == "jepa":
            self.space.load_teacher(jepa)
            del jepa
        train_autoencoder(self.space, self.spec, train, val, hp, self.device, self.precision,
                          self.ctx, ckpt / "ae.pt", train_limit, val_limit, self.stats)
        self.space.eval().requires_grad_(False)

    def _cache_latents(self, train: Split, val: Split, train_limit, val_limit):
        """Encode train/val clips once; (re)calibrate flow statistics on the train clips."""
        hp, workers = self.hparams, self.hparams["workers"]
        recalibrate = hp["latent"]["recalibrate"] or not hp["init_ae"]
        train_idx = subset_indices(len(train), train_limit)
        val_idx = subset_indices(len(val), val_limit)
        raw = encode_split(self.space, train, train_idx, self.device, workers, standardised=False)
        if recalibrate:
            values = torch.cat([raw[i] for i in train_idx]).double()
            self.space.latent_mean.copy_(values.mean(0).float())
            self.space.latent_std.copy_(values.std(0).clamp_min(hp["latent"]["min_std"]).float())
        mean, std = self.space.latent_mean.cpu(), self.space.latent_std.cpu()
        train_q = [((raw[i] - mean) / std) for i in train_idx]
        val_raw = encode_split(self.space, val, val_idx, self.device, workers, standardised=False)
        val_q = [((val_raw[i] - mean) / std) for i in val_idx]
        log_len = float(np.mean([math.log(len(q)) for q in train_q]))
        info = {"clips": len(train_q), "frames": int(sum(len(q) for q in train_q)),
                "std_min": float(std.min()), "std_max": float(std.max()),
                "mean_log_length": log_len, "recalibrated": bool(recalibrate)}
        save_space(self.space, self.spec, self.ctx.ckpt_dir / "ae.pt",
                   extra={"calibration": info, "data": self.cfg["data"]["name"]})
        self.ctx.logger.info("Latent cache: %s", info)
        return (train_q, [train.inputs[i] for i in train_idx],
                val_q, [val.inputs[i] for i in val_idx], log_len)

    # ------------------------------------------------------------------ stage 3
    def _build_generator(self) -> Generator:
        text = TextEncoder(**self.text_args, finetune=self.text_finetune)
        dit = LatentDiT(self.space.latent_dim, text.output_dim, **self.dit_args)
        return Generator(text, dit, DurationHead(text.output_dim)).to(self.device)

    def _flow_losses(self, batch, gen: Generator, rng: torch.Generator | None = None) -> dict:
        q, lengths = batch.values.float(), batch.lengths
        valid = valid_frames(lengths, q.shape[1])
        tokens, mask = gen.text(batch.texts, batch.languages)
        if rng is None:
            drop = torch.rand(len(q), device=q.device) < self.flow_cfg["cond_drop"]
            mask = self._drop_tokens(mask)
        else:
            drop = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        condition = gen.dit.condition(tokens, mask, batch.languages, drop)
        t = sample_time(len(q), self.flow_cfg["time_sampling"], q.device, rng)
        noise = torch.randn(q.shape, device=q.device, generator=rng)
        x_t, target = training_pair(q, noise, t)
        predicted = gen.dit(x_t, t, valid, condition).float()
        flow = masked_mean((predicted - target).square(), valid)
        log_t = gen.duration(tokens, mask, batch.languages)
        duration = F.l1_loss(log_t, lengths.float().log())
        return {"loss": flow + self.flow_cfg["duration_weight"] * duration, "flow": flow,
                "log_duration": duration}

    def _drop_tokens(self, mask: Tensor) -> Tensor:
        """Training-time text token dropout (regularises the text pathway); keeps >= 1 token."""
        p = self.flow_cfg.get("token_drop", 0.0)
        if not p:
            return mask
        kept = mask & (torch.rand(mask.shape, device=mask.device) >= p)
        empty = ~kept.any(1, keepdim=True)
        return torch.where(empty, mask, kept)

    def _load_flow_init(self, gen: Generator) -> None:
        """Warm start from a pretrained flow checkpoint in the same latent space."""
        path = self.hparams["init_flow"]
        state = torch.load(resolve(path), map_location="cpu", weights_only=True)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported init_flow checkpoint version")
        if state["dit_args"] != self.dit_args or state["text_args"] != self.text_args:
            raise ValueError("init_flow DiT/text architecture differs from this config")
        if state["space_spec"] != self.spec:
            raise ValueError("init_flow was trained in a different latent space")
        for key in ("latent_mean", "latent_std"):
            if not torch.allclose(state["space"][key].float(), getattr(self.space, key).cpu()):
                raise ValueError(f"init_flow {key} differs from the current latent statistics")
        missing, unexpected = gen.load_state_dict(state["generator"], strict=False)
        if unexpected or any(not k.startswith("text.encoder.") for k in missing):
            raise ValueError(f"init_flow generator mismatch: {missing} {unexpected}")
        self.ctx.logger.info("Warm-started the flow generator from %s", path)

    def _optimizer(self, gen: Generator) -> torch.optim.Optimizer:
        cfg = self.flow_cfg
        if not (self.text_finetune and gen.text.backend == "mbart"):
            return make_optimizer(gen.parameters(), cfg)
        text = {id(p) for p in gen.text.encoder.parameters()}
        rest = [p for p in gen.parameters() if id(p) not in text]
        scale = self.hparams["text"].get("lr_scale", 1.0)
        return make_optimizer(rest, cfg, extra=(list(gen.text.encoder.parameters()), scale))

    def _state(self, gen: Generator) -> dict:
        decoder_only = {k: v for k, v in self.space.state_dict().items()
                        if not k.startswith(("encoder.", "compressor."))}
        return {"version": CHECKPOINT_VERSION, "space_spec": self.spec, "space": decoder_only,
                "text_args": self.text_args, "dit_args": self.dit_args,
                "text_finetune": gen.text.finetune,
                "generator": {k: v for k, v in gen.state_dict().items()
                              if not k.startswith("text.encoder.") or gen.text.trained}}

    def _restore(self, state: dict) -> None:
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported SignRAE checkpoint version")
        if state["text_args"] != self.text_args or state["dit_args"] != self.dit_args:
            raise ValueError("Checkpoint text/DiT architecture differs from this config")
        self.spec = state["space_spec"]
        self.space = LatentSpace(**self.spec)
        missing, unexpected = self.space.load_state_dict(state["space"], strict=False)
        if unexpected or any(not k.startswith(("encoder.", "compressor.")) for k in missing):
            raise ValueError(f"Latent decoder state mismatch: {missing} {unexpected}")
        self.space.to(self.device).eval().requires_grad_(False)
        self.text_finetune = bool(state.get("text_finetune", False))
        self.gen = self._build_generator()
        missing, unexpected = self.gen.load_state_dict(state["generator"], strict=False)
        if unexpected or any(not k.startswith("text.encoder.") for k in missing):
            raise ValueError(f"Generator state mismatch: {missing} {unexpected}")
        self.gen.eval().requires_grad_(False)
        self.ready = True

    def fit(self, train: Split, val: Split) -> None:
        self.ready = False
        hp, cfg = self.hparams, self.flow_cfg
        train_limit, val_limit = self._limits()
        self.ctx.logger.info("SignRAE (%s latent) on %s: %d train / %d val clips",
                             hp["latent"]["kind"], self.device, len(train), len(val))
        self._fit_space(train, val, train_limit, val_limit)
        train_q, train_items, val_q, val_items, log_len = self._cache_latents(
            train, val, train_limit, val_limit)

        torch.manual_seed(self.ctx.seed)  # same generator init whether or not stages were reused
        gen = self._build_generator()
        gen.duration.init_prior(log_len)
        if hp.get("init_flow"):
            self._load_flow_init(gen)
        optimizer = self._optimizer(gen)
        # EMA over the trained modules only (the frozen mBART encoder is shared, not copied).
        trained = nn.ModuleDict({"dit": gen.dit, "duration": gen.duration,
                                 **({"text": gen.text} if gen.text.trained else {})})
        evaluated = gen
        ema = None
        if cfg["ema_decay"]:
            ema = AveragedModel(trained, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]),
                                use_buffers=True)
            ema.eval().requires_grad_(False)
            avg = ema.module
            evaluated = Generator(avg["text"] if "text" in avg else gen.text, avg["dit"],
                                  avg["duration"])
        # Latents are already in memory: no worker processes (they would copy the cache).
        loader = make_loader(LatentDataset(train_q, train_items), cfg, 0, self.device,
                             self.ctx.seed, shuffle=True)
        val_loader = make_loader(LatentDataset(val_q, val_items), cfg, 0, self.device,
                                 self.ctx.seed, shuffle=False)

        def validate() -> dict:
            rng = torch.Generator(device=self.device).manual_seed(self.ctx.seed + 1)
            return evaluate_loader(val_loader, lambda b: self._flow_losses(b, evaluated, rng),
                                   self.device, self.precision)

        path = self.ctx.ckpt_dir / "model.pt"

        def snapshot(step: int) -> None:
            atomic_save({**self._state(evaluated), "step": step},
                        path.with_name(f"model_step{step:06d}.pt"))

        train_stage("flow", loader, cfg, optimizer, (gen,),
                    lambda batch: self._flow_losses(batch, gen), self.device, self.precision,
                    self.ctx, validate=validate,
                    select=None if cfg["select"] == "last" else "flow",
                    save=lambda: atomic_save(self._state(evaluated), path),
                    after_step=None if ema is None else lambda _: ema.update_parameters(trained),
                    snapshot=snapshot)
        self._restore(torch.load(path, map_location="cpu", weights_only=True))

    def save(self) -> None:
        """fit() already wrote checkpoints/model.pt (and jepa.pt / ae.pt)."""

    def load(self, ckpt_dir) -> None:
        self.load_checkpoint(Path(ckpt_dir) / "model.pt")

    def load_checkpoint(self, path) -> None:
        self.ready = False
        self._restore(torch.load(resolve(path), map_location="cpu", weights_only=True))

    # ------------------------------------------------------------------ inference
    def _noise(self, keys: list[str], lengths: list[int]) -> Tensor:
        """Per-utterance noise: identical across batch sizes and sampler settings."""
        rows, rng = [], torch.Generator(device=self.device)
        for key, n in zip(keys, lengths, strict=True):
            digest = hashlib.blake2b(f"{self.ctx.seed}/{key}".encode(), digest_size=8).digest()
            rng.manual_seed(int.from_bytes(digest, "little"))
            rows.append(torch.randn((n, self.space.latent_dim), device=self.device, generator=rng))
        return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)

    def _fixed_lengths(self, keys: list[str]) -> list[int] | None:
        path = self.sampler.get("lengths_from_npz")
        if not path:
            return None
        out = []
        with np.load(resolve(path), allow_pickle=False) as source:
            for key in keys:
                if key not in source:
                    raise ValueError(f"No prediction for {key!r} in {path}")
                clip = source[key]
                if clip.ndim != 2 or clip.shape[1] != POSE_DIM:
                    raise ValueError(f"Invalid prediction shape {clip.shape} for {key!r}")
                out.append(int(np.clip(len(clip), self.sampler["min_frames"],
                                       self.sampler["max_frames"])))
        return out

    @torch.inference_mode()
    def _sample(self, texts: list[str], languages: Tensor, keys: list[str],
                lengths: list[int] | None) -> list[np.ndarray]:
        s, gen = self.sampler, self.gen
        with precision_context(self.device, self.precision):
            tokens, mask = gen.text(texts, languages)
            if lengths is None:
                log_t = gen.duration(tokens, mask, languages).float()
                lo, hi = math.log(s["min_frames"]), math.log(s["max_frames"])
                lengths = log_t.clamp(lo, hi).exp().round().long().tolist()
            n = len(texts)
            length_t = torch.tensor(lengths, device=self.device)
            valid = valid_frames(length_t, max(lengths))
            w = float(s["cfg_scale"])
            no_drop = torch.zeros(n, dtype=torch.bool, device=self.device)
            if w == 1.0:
                cond = gen.dit.condition(tokens, mask, languages, no_drop)

                def field(state, t):
                    return gen.dit(state, t, valid, cond).float()
            else:
                both = gen.dit.condition(torch.cat([tokens, tokens]), torch.cat([mask, mask]),
                                         torch.cat([languages, languages]),
                                         torch.cat([no_drop, ~no_drop]))
                valid2 = torch.cat([valid, valid])

                def field(state, t):
                    out = gen.dit(torch.cat([state, state]), torch.cat([t, t]), valid2,
                                  both).float()
                    return cfg_combine(out[:n], out[n:], w)

            total = None
            members = int(s.get("ensemble", 1))
            for k in range(members):  # member 0 uses the single-sample noise (comparable runs)
                member_keys = keys if k == 0 else [f"{key}#{k}" for key in keys]
                x = self._noise(member_keys, lengths) * s["temperature"]
                q = integrate(field, x, s["steps"], s["solver"], valid)
                with torch.autocast(self.device.type, enabled=False):
                    poses = self.space.decode_flow(q.float(), length_t)
                total = poses if total is None else total + poses
            poses = (total / members).cpu().numpy()
        return [poses[i, :k].astype(np.float32, copy=True) for i, k in enumerate(lengths)]

    def predict(self, inputs):
        if not self.ready:
            raise RuntimeError("Fit or load SignRAE before prediction")
        if not inputs:
            return []
        names = [str(item.get("name", i)) for i, item in enumerate(inputs)]
        keys = [name.replace("/", "__") for name in names]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate sample keys after normalising names")
        fixed = self._fixed_lengths(keys)
        predictions: list[np.ndarray] = []
        size = self.sampler["batch_size"]
        for start in range(0, len(inputs), size):
            chunk = inputs[start:start + size]
            texts, languages = zip(*(text_input(item) for item in chunk))
            languages = torch.tensor(languages, device=self.device)
            noise_keys = [f"{lang}/{name}" for lang, name in
                          zip(languages.tolist(), names[start:start + size], strict=True)]
            lengths = None if fixed is None else fixed[start:start + size]
            predictions.extend(self._sample(list(texts), languages, noise_keys, lengths))
        if self.hparams["export_predictions"]:
            arrays = dict(zip(keys, predictions, strict=True))
            out = self.ctx.run_dir / "predictions.npz"
            tmp = out.with_name(".predictions.tmp.npz")
            np.savez_compressed(tmp, **arrays)
            tmp.replace(out)
        return predictions
