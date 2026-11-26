"""SignRAE: MoRAE (arXiv 2607.29180) transferred to text-to-sign production.

fit():  Sign-JEPA pre-training (or ``init_jepa``) -> compact latent autoencoder (or ``init_ae``)
        -> train-split latent calibration + cache -> text-conditioned latent flow (DiT-M + mBART).
        ``init_flow`` warm-starts the generator from a flow pre-trained on the union of datasets
        (pretrain.py --stage flow) before fine-tuning on this dataset.
predict(): duration head -> seeded Gaussian noise -> guided ODE (RK4 by default) -> decoder ->
        normalised [T, 133] poses; ``sampler.ensemble`` K > 1 combines K samples (pose-space mean
        by default). Metrics are computed by the shared evaluator only.

v3 options (PLAN_V3.md; every default reproduces v2 exactly):
  C1 flow.prediction = target (x1 prediction) + sampler.solver = replan_implied | replan_anchored
  C2 flow.fk_weight_{body,hand}: joint-position loss in the SOKE metric frame on decoded x1
  C3 flow.gloss_weight + gloss.recognizer: CTC of a frozen gloss recognizer on x1 (train glosses)
  C4 retrieval.enabled: train-only text retrieval -> memory tokens in the DiT cross-attention
  C5 sampler.decoder = medoid | dba over sampler.ensemble samples
  C6 duration.{token_count, loss, calibrate}
See PLAN.md / PLAN_V3.md for the design and README.md for the commands.
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

from .autoencoder import KINDS, LatentSpace, PoseFKLoss
from .data import LatentDataset, POSE_DIM, load_pose, subset_indices, text_input, valid_frames
from .decode import dba, medoid
from .dit import DurationHead, LatentDiT
from .flow import (PREDICTIONS, REPLAN, SOLVERS, cfg_combine, integrate, replan, sample_time,
                   target_weight, to_target, to_velocity, training_pair)
from .layers import masked_mean
from .stages import (atomic_save, encode_split, latent_stats, load_jepa, load_space, resolve,
                     save_space, space_spec, train_autoencoder, train_jepa)
from .text import TextEncoder
from .training import evaluate_loader, make_loader, make_optimizer, precision_context, train_stage

CHECKPOINT_VERSION = 1
DIT_KEYS = ("width", "depth", "heads", "ff", "dropout", "text_mode", "max_frames")
TEXT_KEYS = ("backend", "model", "max_tokens", "byte_dim")
STAGE_KEYS = ("batch_size", "microbatch", "updates", "warmup_updates", "eval_every", "lr",
              "min_lr_ratio", "betas", "weight_decay", "grad_clip")
DECODERS = ("mean", "medoid", "dba")
RETRIEVAL_FILE = "retrieval.pt"
# Defaults of the v3 flow keys (identical to config.yaml; used when a run config predates v3).
FLOW_DEFAULTS = {"prediction": "velocity", "target_weighting": "uniform", "fk_weight_body": 0.0,
                 "fk_weight_hand": 0.0, "fk_t_min": 0.3, "fk_frames": 16, "gloss_weight": 0.0,
                 "gloss_t_min": 0.5, "gloss_start": 5000}
# x1 prediction: rk4 / heun evaluate the field at t = 1, where v = (x1 - x) / (1 - t) is undefined.
TARGET_SOLVERS = ("euler", "replan_implied", "replan_anchored")


class Generator(nn.Module):
    """Everything the flow stage trains (EMA is taken over this module)."""

    def __init__(self, text: TextEncoder, dit: LatentDiT, duration: DurationHead):
        super().__init__()
        self.text, self.dit, self.duration = text, dit, duration


def _opt(section: dict, key: str, default):
    """Config value with a default: run configs written before v3 lack the new keys."""
    return section.get(key, default) if isinstance(section, dict) else default


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
        self.retrieval_cfg = hp.get("retrieval") or {}
        self.duration_cfg = hp.get("duration") or {}
        # Architecture / parameterisation (overwritten from the checkpoint by _restore).
        self.prediction = _opt(self.flow_cfg, "prediction", FLOW_DEFAULTS["prediction"])
        self.memory_args = self._memory_args_from_config()
        self.duration_args = {"token_count": bool(_opt(self.duration_cfg, "token_count", False))}
        self.duration_bias = 0.0
        self.index = self.bank = None
        self._fk = self._recognizer = None
        self._step = 0
        self._validate()
        self.ready = False

    def _memory_args_from_config(self) -> dict | None:
        r = self.retrieval_cfg
        if not _opt(r, "enabled", False):
            return None
        mem = _opt(r, "memory", {}) or {}
        return {"slots": int(_opt(r, "k", 3)), "layers": int(mem.get("layers", 2)),
                "heads": int(mem.get("heads", 4)), "stride": int(mem.get("stride", 4))}

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
        s, f = self.sampler, self.flow_cfg
        if s["solver"] not in SOLVERS + REPLAN or s["steps"] < 1:
            raise ValueError(f"method.sampler.solver must be one of {SOLVERS + REPLAN}, "
                             "steps >= 1")
        if not _opt(s, "length_scale", 1.0) > 0:
            raise ValueError("method.sampler.length_scale must be positive")
        if not (math.isfinite(s["cfg_scale"]) and s["cfg_scale"] >= 0 and s["temperature"] >= 0):
            raise ValueError("Invalid sampler.cfg_scale or sampler.temperature")
        if not 0 < s["min_frames"] <= s["max_frames"] <= f["max_frames"]:
            raise ValueError("Need 0 < sampler.min_frames <= max_frames <= flow.max_frames")
        if not 0 <= f["cond_drop"] < 1 or not 0 <= f["ema_decay"] < 1:
            raise ValueError("Invalid flow.cond_drop or flow.ema_decay")
        if f["select"] not in ("last", "flow"):
            raise ValueError("method.flow.select must be last or flow")
        members = int(s.get("ensemble", 1))
        if members < 1:
            raise ValueError("method.sampler.ensemble must be >= 1")
        decoder = _opt(s, "decoder", "mean")
        if decoder not in DECODERS:
            raise ValueError(f"method.sampler.decoder must be one of {DECODERS}")
        if decoder != "mean" and members < 2:
            raise ValueError("sampler.decoder medoid/dba needs sampler.ensemble >= 2")
        if not 0 <= f.get("token_drop", 0.0) < 1:
            raise ValueError("method.flow.token_drop must be in [0, 1)")
        if hp["text"].get("lr_scale", 1.0) <= 0:
            raise ValueError("method.text.lr_scale must be positive")
        if self.prediction not in PREDICTIONS:
            raise ValueError(f"method.flow.prediction must be one of {PREDICTIONS}")
        if _opt(f, "target_weighting", FLOW_DEFAULTS["target_weighting"]) not in ("uniform", "velocity"):
            raise ValueError("method.flow.target_weighting must be uniform or velocity")
        for key in ("fk_weight_body", "fk_weight_hand", "gloss_weight"):
            if _opt(f, key, FLOW_DEFAULTS[key]) < 0:
                raise ValueError(f"method.flow.{key} must be >= 0")
        for key in ("fk_t_min", "gloss_t_min"):
            if not 0 <= _opt(f, key, FLOW_DEFAULTS[key]) < 1:
                raise ValueError(f"method.flow.{key} must be in [0, 1)")
        self._check_solver(self.prediction)
        if self._fk_active() and int(_opt(f, "fk_frames", FLOW_DEFAULTS["fk_frames"])) < 1:
            raise ValueError("method.flow.fk_frames must be >= 1")
        if self._gloss_active() and not _opt(hp.get("gloss") or {}, "recognizer", None):
            raise ValueError("method.flow.gloss_weight > 0 needs method.gloss.recognizer")
        if self.memory_args is not None:
            if f["text_mode"] != "xattn":
                raise ValueError("retrieval needs flow.text_mode=xattn")
            if not 0 <= _opt(self.retrieval_cfg, "dropout", 0.25) < 1:
                raise ValueError("method.retrieval.dropout must be in [0, 1)")
        if _opt(self.duration_cfg, "loss", "l1") not in ("l1", "l2"):
            raise ValueError("method.duration.loss must be l1 or l2")
        if hp.get("init_flow"):
            if not hp["init_ae"]:
                raise ValueError("method.init_flow needs method.init_ae (the same latent space)")
            if hp["latent"]["recalibrate"]:
                raise ValueError("method.init_flow needs latent.recalibrate=false: the pretrained "
                                 "flow was trained in the init_ae standardisation")

    def _check_solver(self, prediction: str) -> None:
        if prediction == "target" and self.sampler["solver"] not in TARGET_SOLVERS:
            raise ValueError(f"flow.prediction=target needs sampler.solver in {TARGET_SOLVERS} "
                             f"(got {self.sampler['solver']!r}, which evaluates t = 1)")

    def _fk_active(self) -> bool:
        f = self.flow_cfg
        return (_opt(f, "fk_weight_body", FLOW_DEFAULTS["fk_weight_body"]) > 0
                or _opt(f, "fk_weight_hand", FLOW_DEFAULTS["fk_weight_hand"]) > 0)

    def _gloss_active(self) -> bool:
        return _opt(self.flow_cfg, "gloss_weight", FLOW_DEFAULTS["gloss_weight"]) > 0

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

    def _cache_latents(self, train: Split, val: Split, train_limit, val_limit) -> dict:
        """Encode train/val clips once; (re)calibrate flow statistics on the train clips."""
        hp, workers = self.hparams, self.hparams["workers"]
        recalibrate = hp["latent"]["recalibrate"] or not hp["init_ae"]
        train_idx = subset_indices(len(train), train_limit)
        val_idx = subset_indices(len(val), val_limit)
        raw = encode_split(self.space, train, train_idx, self.device, workers, standardised=False)
        if recalibrate:
            mean, std = latent_stats(raw, train_idx, hp["latent"]["min_std"])
            self.space.latent_mean.copy_(mean)
            self.space.latent_std.copy_(std)
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
        return {"train_q": train_q, "train_items": [train.inputs[i] for i in train_idx],
                "val_q": val_q, "val_items": [val.inputs[i] for i in val_idx],
                "log_len": log_len, "train_idx": train_idx, "val_idx": val_idx}

    # ------------------------------------------------------------------ v3 helpers
    def _fk_module(self) -> PoseFKLoss:
        if self._fk is None:
            self._fk = PoseFKLoss(resolve(self.stats) if self.stats else None).to(self.device)
        return self._fk

    def _poses(self, split: Split, indices: list[int]) -> list[Tensor]:
        """GT poses (fp16 to halve the memory) for the metric-frame FK loss (C2)."""
        return [torch.from_numpy(load_pose(split, i)).half() for i in indices]

    def _load_recognizer(self) -> None:
        from .gloss import load_recognizer

        path = self.hparams["gloss"]["recognizer"]
        model, vocab, mean, std = load_recognizer(path, self.device)
        if not (torch.allclose(mean, self.space.latent_mean.cpu(), atol=1e-5)
                and torch.allclose(std, self.space.latent_std.cpu(), atol=1e-5)):
            raise ValueError("gloss recognizer was trained in a different latent "
                             "standardisation (train it with the same ae.pt / recalibration)")
        if model.inp.in_features != self.space.latent_dim:
            raise ValueError("gloss recognizer latent dimension differs from the latent space")
        self._recognizer, self._gloss_vocab = model, vocab
        self.ctx.logger.info("Gloss recognizer %s (%d glosses)", path, len(vocab) - 2)

    def _build_retrieval(self, train_items: list[dict], train_q: list[Tensor]) -> None:
        from .retrieval import TextIndex

        r = self.retrieval_cfg
        texts = [text_input(item)[0] for item in train_items]
        self.index = TextIndex(texts, tuple(_opt(r, "ngram", [2, 4])),
                               float(_opt(r, "length_weight", 0.1)))
        self.bank = [q.half() for q in train_q]
        atomic_save({"index": self.index.state(), "bank": self.bank,
                     "k": self.memory_args["slots"]}, self.ctx.ckpt_dir / RETRIEVAL_FILE)

    def _retrieve(self, texts: list[str], train_ids: list[int] | None) -> Tensor:
        """[N, k] retrieved train indices. ``train_ids`` (train queries): never the clip itself,
        and with ``retrieval.exclude_duplicates`` none of its verbatim repeats either."""
        k = self.memory_args["slots"]
        exclude = train_ids is not None and bool(_opt(self.retrieval_cfg,
                                                      "exclude_duplicates", True))
        return torch.from_numpy(self.index.query(texts, k, exclude_groups=exclude,
                                                 query_ids=train_ids))

    def _memory(self, indices: Tensor | None, training: bool):
        if self.memory_args is None or indices is None:
            return None, None
        from .retrieval import gather_memory

        indices = indices.clone().cpu()
        if training:  # retrieval dropout: the whole memory of a sample is removed
            p = float(_opt(self.retrieval_cfg, "dropout", 0.25))
            gone = torch.rand(len(indices)) < p
            indices[gone] = -1
        return gather_memory(self.bank, indices, self.flow_cfg["max_frames"], self.device)

    # ------------------------------------------------------------------ stage 3
    def _build_generator(self) -> Generator:
        text = TextEncoder(**self.text_args, finetune=self.text_finetune)
        dit = LatentDiT(self.space.latent_dim, text.output_dim, **self.dit_args,
                        memory=self.memory_args)
        duration = DurationHead(text.output_dim, **self.duration_args)
        return Generator(text, dit, duration).to(self.device)

    def _flow_losses(self, batch, gen: Generator, rng: torch.Generator | None = None) -> dict:
        f = self.flow_cfg
        training = rng is None
        q, lengths = batch.values.float(), batch.lengths
        valid = valid_frames(lengths, q.shape[1])
        tokens, mask = gen.text(batch.texts, batch.languages)
        full_mask = mask
        if training:
            drop = torch.rand(len(q), device=q.device) < f["cond_drop"]
            mask = self._drop_tokens(mask)
        else:
            drop = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        memory, memory_valid = self._memory(batch.memory, training)
        condition = gen.dit.condition(tokens, mask, batch.languages, drop, memory, memory_valid)
        t = sample_time(len(q), f["time_sampling"], q.device, rng)
        noise = torch.randn(q.shape, device=q.device, generator=rng)
        x_t, target = training_pair(q, noise, t)
        out = gen.dit(x_t, t, valid, condition).float()
        if self.prediction == "velocity":
            flow = masked_mean((out - target).square(), valid)
        else:
            weight = target_weight(t, _opt(f, "target_weighting",
                                           FLOW_DEFAULTS["target_weighting"]))
            flow = masked_mean((out - q).square() * weight[:, None, None], valid)
        # v2 pooled the token-dropped mask; the token-count feature must see the same number of
        # tokens as at inference, so it uses the undropped mask.
        dur_mask = full_mask if self.duration_args.get("token_count") else mask
        log_t = gen.duration(tokens, dur_mask, batch.languages)
        if _opt(self.duration_cfg, "loss", "l1") == "l2":
            duration = F.mse_loss(log_t, lengths.float().log())
        else:
            duration = F.l1_loss(log_t, lengths.float().log())
        loss = flow + f["duration_weight"] * duration
        losses = {"flow": flow, "log_duration": duration}
        if self._fk_active() or self._gloss_active():
            x1 = to_target(out, x_t, t, self.prediction)
        if self._fk_active():
            body, hand = self._fk_losses(x1, batch.poses, lengths, t)
            loss = (loss + _opt(f, "fk_weight_body", FLOW_DEFAULTS["fk_weight_body"]) * body
                    + _opt(f, "fk_weight_hand", FLOW_DEFAULTS["fk_weight_hand"]) * hand)
            losses.update(fk_body=body, fk_hand=hand)
        if self._gloss_active():
            gloss = self._gloss_loss(x1, valid, lengths, batch.glosses, t)
            if self._step >= int(_opt(f, "gloss_start", FLOW_DEFAULTS["gloss_start"])):
                loss = loss + f["gloss_weight"] * gloss
            losses["gloss"] = gloss
        losses["loss"] = loss
        return losses

    def _fk_losses(self, x1: Tensor, poses: Tensor | None, lengths: Tensor, t: Tensor):
        """C2: smooth-L1 (cm) of SOKE-frame joint positions of decode(x1) against the GT pose
        (upper body relative to the pelvis, hands relative to their wrists), for t >= fk_t_min."""
        if poses is None:
            raise ValueError("FK loss needs GT poses in the batch")
        sel = t >= _opt(self.flow_cfg, "fk_t_min", FLOW_DEFAULTS["fk_t_min"])
        if not bool(sel.any()):
            zero = x1.sum() * 0.0
            return zero, zero
        decoded = self.space.decode_flow(x1[sel], lengths[sel])
        target = poses[sel][:, : decoded.shape[1]].float()
        return self._fk_module()(decoded, target, lengths[sel],
                                 int(_opt(self.flow_cfg, "fk_frames", FLOW_DEFAULTS["fk_frames"])))

    def _gloss_loss(self, x1: Tensor, valid: Tensor, lengths: Tensor,
                    glosses: list[Tensor] | None, t: Tensor) -> Tensor:
        """C3: CTC of the frozen recognizer on x1 for t >= gloss_t_min (clips with glosses)."""
        from .gloss import ctc

        if glosses is None:
            raise ValueError("gloss loss needs gloss ids in the batch")
        sel = (t >= _opt(self.flow_cfg, "gloss_t_min", FLOW_DEFAULTS["gloss_t_min"])
               ).nonzero().flatten().tolist()
        if not sel:
            return x1.sum() * 0.0
        idx = torch.tensor(sel, device=x1.device)
        log_probs = self._recognizer(x1.index_select(0, idx), valid.index_select(0, idx))
        return ctc(log_probs, lengths.index_select(0, idx), [glosses[i] for i in sel])

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
        if state.get("duration_args", {"token_count": False}) != self.duration_args:
            raise ValueError("init_flow duration head differs from this config")
        if state.get("flow_prediction", "velocity") != self.prediction:
            raise ValueError("init_flow was trained with a different flow.prediction")
        if state["space_spec"] != self.spec:
            raise ValueError("init_flow was trained in a different latent space")
        for key in ("latent_mean", "latent_std"):
            if not torch.allclose(state["space"][key].float(), getattr(self.space, key).cpu()):
                raise ValueError(f"init_flow {key} differs from the current latent statistics")
        missing, unexpected = gen.load_state_dict(state["generator"], strict=False)
        allowed = ("text.encoder.", "dit.memory.")  # a new memory encoder starts from scratch
        if unexpected or any(not k.startswith(allowed) for k in missing):
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
                "flow_prediction": self.prediction, "memory_args": self.memory_args,
                "duration_args": self.duration_args, "duration_bias": float(self.duration_bias),
                "generator": {k: v for k, v in gen.state_dict().items()
                              if not k.startswith("text.encoder.") or gen.text.trained}}

    def _restore(self, state: dict, path: Path | None = None) -> None:
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
        # Checkpoints written before v3 lack these keys: velocity, no memory, plain duration.
        self.prediction = state.get("flow_prediction", "velocity")
        self._check_solver(self.prediction)
        self.memory_args = state.get("memory_args")
        self.duration_args = state.get("duration_args", {"token_count": False})
        self.duration_bias = float(state.get("duration_bias", 0.0))
        self.gen = self._build_generator()
        missing, unexpected = self.gen.load_state_dict(state["generator"], strict=False)
        if unexpected or any(not k.startswith("text.encoder.") for k in missing):
            raise ValueError(f"Generator state mismatch: {missing} {unexpected}")
        self.gen.eval().requires_grad_(False)
        if self.memory_args is not None:
            from .retrieval import TextIndex

            if path is None:
                raise ValueError("a retrieval checkpoint needs its directory for retrieval.pt")
            extra = torch.load(resolve(Path(path).with_name(RETRIEVAL_FILE)), map_location="cpu",
                               weights_only=True)
            self.index = TextIndex.from_state(extra["index"])
            self.bank = list(extra["bank"])
        self.ready = True

    def fit(self, train: Split, val: Split) -> None:
        self.ready = False
        hp, cfg = self.hparams, self.flow_cfg
        train_limit, val_limit = self._limits()
        self.ctx.logger.info("SignRAE (%s latent) on %s: %d train / %d val clips",
                             hp["latent"]["kind"], self.device, len(train), len(val))
        self._fit_space(train, val, train_limit, val_limit)
        cache = self._cache_latents(train, val, train_limit, val_limit)
        train_q, train_items = cache["train_q"], cache["train_items"]
        val_q, val_items = cache["val_q"], cache["val_items"]

        extras_train: dict = {}
        extras_val: dict = {}
        if self._fk_active():
            extras_train["poses"] = self._poses(train, cache["train_idx"])
            extras_val["poses"] = self._poses(val, cache["val_idx"])
        if self._gloss_active():
            self._load_recognizer()
            extras_train["glosses"] = [self._gloss_vocab.encode(i.get("gloss")) for i in train_items]
            extras_val["glosses"] = [self._gloss_vocab.encode(i.get("gloss")) for i in val_items]
        if self.memory_args is not None:
            self._build_retrieval(train_items, train_q)
            extras_train["memory"] = self._retrieve([text_input(i)[0] for i in train_items],
                                                    list(range(len(train_items))))
            extras_val["memory"] = self._retrieve([text_input(i)[0] for i in val_items], None)

        torch.manual_seed(self.ctx.seed)  # same generator init whether or not stages were reused
        gen = self._build_generator()
        gen.duration.init_prior(cache["log_len"])
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
        loader = make_loader(LatentDataset(train_q, train_items, **extras_train), cfg, 0,
                             self.device, self.ctx.seed, shuffle=True)
        val_loader = make_loader(LatentDataset(val_q, val_items, **extras_val), cfg, 0,
                                 self.device, self.ctx.seed, shuffle=False)

        def validate() -> dict:
            rng = torch.Generator(device=self.device).manual_seed(self.ctx.seed + 1)
            return evaluate_loader(val_loader, lambda b: self._flow_losses(b, evaluated, rng),
                                   self.device, self.precision)

        path = self.ctx.ckpt_dir / "model.pt"

        def snapshot(step: int) -> None:
            atomic_save({**self._state(evaluated), "step": step},
                        path.with_name(f"model_step{step:06d}.pt"))

        def after_step(step: int) -> None:
            self._step = step
            if ema is not None:
                ema.update_parameters(trained)

        self._step = 0
        train_stage("flow", loader, cfg, optimizer, (gen,),
                    lambda batch: self._flow_losses(batch, gen), self.device, self.precision,
                    self.ctx, validate=validate,
                    select=None if cfg["select"] == "last" else "flow",
                    save=lambda: atomic_save(self._state(evaluated), path),
                    after_step=after_step, snapshot=snapshot)
        self._restore(torch.load(path, map_location="cpu", weights_only=True), path)
        if _opt(self.duration_cfg, "calibrate", False):
            self._calibrate_duration(val_items, [len(q) for q in val_q], path)

    @torch.inference_mode()
    def _calibrate_duration(self, items: list[dict], lengths: list[int], path: Path) -> None:
        """C6: shift log T by the median residual on the fit's validation clips (the 256-clip
        'tune' subset when val_limit=256; the remaining dev clips stay untouched for
        confirmation). The train residuals are useless: the head memorises train lengths.
        Only model.pt gets the bias: model_step*.pt snapshots keep 0 (uncalibrated)."""
        residuals = []
        for start in range(0, len(items), 64):
            chunk = items[start:start + 64]
            texts, languages = zip(*(text_input(i) for i in chunk))
            languages = torch.tensor(languages, device=self.device)
            with precision_context(self.device, self.precision):
                tokens, mask = self.gen.text(list(texts), languages)
                pred = self.gen.duration(tokens, mask, languages).float().cpu()
            true = torch.tensor(lengths[start:start + 64], dtype=torch.float32).log()
            residuals.append(true - pred)
        self.duration_bias = float(torch.cat(residuals).median())
        state = torch.load(path, map_location="cpu", weights_only=True)
        state["duration_bias"] = self.duration_bias
        atomic_save(state, path)
        self.ctx.logger.info("Duration calibration: log-length bias %+.4f", self.duration_bias)

    def save(self) -> None:
        """fit() already wrote checkpoints/model.pt (and jepa.pt / ae.pt / retrieval.pt)."""

    def load(self, ckpt_dir) -> None:
        self.load_checkpoint(Path(ckpt_dir) / "model.pt")

    def load_checkpoint(self, path) -> None:
        self.ready = False
        path = resolve(path)
        self._restore(torch.load(path, map_location="cpu", weights_only=True), path)

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

    def _points(self, poses: Tensor) -> np.ndarray:
        """[T, 54, 3] SOKE-frame points (cm) of one pose sequence, for the C5 decoders."""
        body, hands = self._fk_module().points(poses.to(self.device).float())
        return torch.cat([body, hands], dim=-2).cpu().double().numpy()

    def _combine(self, members: list[Tensor], lengths: list[int]) -> list[np.ndarray]:
        """C5: medoid or DBA of the K member samples of each clip."""
        decoder = _opt(self.sampler, "decoder", "mean")
        out = []
        for i, n in enumerate(lengths):
            seqs = [m[i, :n].float().cpu() for m in members]
            points = [self._points(s) for s in seqs]
            best, _ = medoid(points)
            if decoder == "medoid":
                out.append(seqs[best].numpy().astype(np.float32, copy=True))
                continue
            centre = dba([s.double().numpy() for s in seqs], points,
                         lambda c: self._points(torch.from_numpy(c)), best,
                         int(_opt(self.sampler, "dba_iterations", 5)))
            out.append(centre.astype(np.float32, copy=False))
        return out

    @torch.inference_mode()
    def _sample(self, texts: list[str], languages: Tensor, keys: list[str],
                lengths: list[int] | None) -> list[np.ndarray]:
        s, gen = self.sampler, self.gen
        with precision_context(self.device, self.precision):
            tokens, mask = gen.text(texts, languages)
            if lengths is None:
                log_t = gen.duration(tokens, mask, languages).float() + self.duration_bias
                scale = float(_opt(s, "length_scale", 1.0))
                if scale != 1.0:  # P0.2 length-sensitivity analysis only (never for results)
                    log_t = log_t + math.log(scale)
                lo, hi = math.log(s["min_frames"]), math.log(s["max_frames"])
                lengths = log_t.clamp(lo, hi).exp().round().long().tolist()
            n = len(texts)
            length_t = torch.tensor(lengths, device=self.device)
            valid = valid_frames(length_t, max(lengths))
            w = float(s["cfg_scale"])
            no_drop = torch.zeros(n, dtype=torch.bool, device=self.device)
            memory = memory_valid = None
            if self.memory_args is not None:
                memory, memory_valid = self._memory(self._retrieve(texts, None), False)
            if w == 1.0:
                cond = gen.dit.condition(tokens, mask, languages, no_drop, memory, memory_valid)

                def output(state, t):
                    return gen.dit(state, t, valid, cond).float()
            else:
                pair = (lambda x: None if x is None else torch.cat([x, x]))
                both = gen.dit.condition(torch.cat([tokens, tokens]), torch.cat([mask, mask]),
                                         torch.cat([languages, languages]),
                                         torch.cat([no_drop, ~no_drop]), pair(memory),
                                         pair(memory_valid))
                valid2 = torch.cat([valid, valid])

                def output(state, t):
                    out = gen.dit(torch.cat([state, state]), torch.cat([t, t]), valid2,
                                  both).float()
                    # Guidance on the raw output: the velocity <-> x1 maps are affine in it for
                    # fixed (state, t), so this equals guidance on either parameterisation.
                    return cfg_combine(out[:n], out[n:], w)

            def field(state, t):
                return to_velocity(output(state, t), state, t, self.prediction)

            def predict_x1(state, t):
                return to_target(output(state, t), state, t, self.prediction)

            decoder = _opt(s, "decoder", "mean")
            members = int(s.get("ensemble", 1))
            total, kept = None, []
            for k in range(members):  # member 0 uses the single-sample noise (comparable runs)
                member_keys = keys if k == 0 else [f"{key}#{k}" for key in keys]
                x = self._noise(member_keys, lengths) * s["temperature"]
                if s["solver"] in REPLAN:
                    q = replan(predict_x1, x, s["steps"], s["solver"], valid)
                else:
                    q = integrate(field, x, s["steps"], s["solver"], valid)
                with torch.autocast(self.device.type, enabled=False):
                    poses = self.space.decode_flow(q.float(), length_t)
                if decoder == "mean":
                    total = poses if total is None else total + poses
                else:
                    kept.append(poses)
        if decoder != "mean":
            return self._combine(kept, lengths)
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
