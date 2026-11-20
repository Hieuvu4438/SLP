"""Training stages of SignNMFlow.

Stage 1  prototypes : per-joint K-means on all training frames (App. C.2)
Stage 2  correction : epsilon_theta minimising the naturalness kinetic energy (Alg. 1, Eq. 12)
Stage 3  generator  : phi along the frozen naturalness-guided path (Alg. 2, Eq. 13)

Optimisation (Table 6): AdamW, lr 2e-4 with cosine decay, weight decay 0, fp32;
eps_theta batch 8 / 100 epochs, phi batch 64 per GPU / 300 epochs (4 GPUs in the paper;
``effective_batch`` reproduces the global batch with gradient accumulation on fewer GPUs).
Distributed data parallel is used automatically under ``torchrun``.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from . import geometry as G
from .data import MotionTextDataset, collate
from .flow import FlowSpec, flow_loss
from .naturalness import NaturalnessMeasure
from .paths import CorrectionNet, path_and_velocity


def dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class EMA:
    """Exponential moving average of the generator weights (docs/proposal7 E3).

    shadow <- decay * shadow + (1 - decay) * param after every optimiser step; buffers are copied.
    Kept in fp32 on the parameters' device; saved in generator.pt under "ema"."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must lie in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        params = dict(model.named_parameters())
        for k, v in model.state_dict().items():
            if k in params and params[k].requires_grad:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict):
        for k in self.shadow:
            self.shadow[k].copy_(sd[k].to(self.shadow[k].device))

    def weights_for(self, model: torch.nn.Module) -> dict:
        """EMA weights cast to the model's dtypes (for model.load_state_dict)."""
        ref = model.state_dict()
        return {k: v.to(ref[k].dtype) for k, v in self.shadow.items()}


def cosine_lambda(total_steps: int, warmup: int = 0):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * p))
    return f


# ----------------------------------------------------------------------------- stage 2
def train_correction(corr: CorrectionNet, measure: NaturalnessMeasure, dataset: MotionTextDataset,
                     codec, hp: dict, device, log=print, ckpt: Path | None = None, metrics=None):
    """Alg. 1: sample x0 ~ p0, x1 ~ p1, t ~ U[0,1]; minimise xdot^T G(x_t)^-1 xdot."""
    corr.to(device).train()
    measure.to(device)
    bs, epochs = int(hp["batch_size"]), int(hp["epochs"])
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate,
                        num_workers=int(hp.get("workers", 4)), drop_last=True)
    opt = torch.optim.AdamW(corr.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(total))
    start, step = 0, 0
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        corr.load_state_dict(st["model"])
        if "opt" in st and st.get("epoch", epochs) < epochs:
            opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, step = st.get("epoch", epochs), st.get("step", 0)
        log(f"correction: resumed at epoch {start}")
    chunk = int(hp.get("frame_chunk", 0)) or None
    for ep in range(start, epochs):
        t0, run = time.time(), 0.0
        for feats, mask, _, _ in loader:
            feats, mask = feats.to(device), mask.to(device)
            x1 = codec.encode(feats)[0][mask]  # valid frames [N,J,4]
            B = feats.shape[0]
            seq_id = torch.arange(B, device=device)[:, None].expand_as(mask)[mask]
            t = torch.rand(B, device=device)[seq_id][:, None, None]  # one t per sequence
            x0 = G.uniform_hemisphere(x1.shape[:2], device=device)
            if chunk is None:
                xt, v = path_and_velocity("naturalness", x0, x1, t, corr)
                loss = measure.quad_form(xt, v).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
            else:  # identical gradient, bounded memory
                opt.zero_grad(set_to_none=True)
                n = x1.shape[0]
                loss = 0.0
                for s in range(0, n, chunk):
                    xt, v = path_and_velocity("naturalness", x0[s:s + chunk], x1[s:s + chunk],
                                              t[s:s + chunk], corr)
                    li = measure.quad_form(xt, v).sum(-1).sum() / n
                    li.backward()
                    loss += float(li)
                loss = torch.tensor(loss)
            opt.step()
            sched.step()
            step += 1
            run += float(loss.detach())
        msg = {"corr_energy": run / max(1, len(loader)), "lr": sched.get_last_lr()[0]}
        log(f"correction epoch {ep + 1}/{epochs} energy={msg['corr_energy']:.5g} ({time.time() - t0:.0f}s)")
        if metrics:
            metrics(msg, step=ep + 1)
        if ckpt is not None:
            torch.save({"model": corr.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep + 1, "step": step}, ckpt)
    corr.eval()
    return corr


@torch.no_grad()
def geodesic_energy_gap(corr, measure, dataset, codec, device, n_batches=20, bs=8):
    """Diagnostic: naturalness kinetic energy of the learned path vs. the plain geodesic."""
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate)
    tot = {"geodesic": 0.0, "naturalness": 0.0}
    for i, (feats, mask, _, _) in enumerate(loader):
        if i >= n_batches:
            break
        x1 = codec.encode(feats.to(device))[0][mask.to(device)]
        x0 = G.uniform_hemisphere(x1.shape[:2], device=device)
        t = torch.rand(x1.shape[0], 1, 1, device=device)
        for kind in tot:
            with torch.enable_grad():
                xt, v = path_and_velocity(kind, x0, x1, t, corr if kind == "naturalness" else None)
            tot[kind] += float(measure.quad_form(xt, v).sum(-1).mean())
    return {k: v / max(1, min(n_batches, len(loader))) for k, v in tot.items()}


# ----------------------------------------------------------------------------- stage 3
def train_generator(gen, corr, spec: FlowSpec, dataset: MotionTextDataset, codec, hp: dict, device,
                    log=print, ckpt: Path | None = None, metrics=None, val_dataset=None, joint_weights=None,
                    position_loss=None):
    rank, world = dist_info()
    gen.to(device).train()
    if corr is not None:
        corr.to(device).eval().requires_grad_(False)
    model = gen
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(gen, device_ids=[device.index],
                                                          find_unused_parameters=True)
    # batch_size = per-GPU batch of the paper (64); each batch is processed in micro-batches whose
    # losses are normalised by the whole batch's valid-frame count, so the gradient equals the
    # 64-sample batch gradient; `accum` such batches emulate the 4-GPU data parallelism
    bs, epochs = int(hp["batch_size"]), int(hp["epochs"])
    micro = min(bs, int(hp.get("micro_batch") or bs))
    accum = max(1, int(hp.get("effective_batch", bs * world)) // (bs * world))
    sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=bs, shuffle=sampler is None, sampler=sampler,
                        collate_fn=collate, num_workers=int(hp.get("workers", 4)), drop_last=True,
                        pin_memory=True, persistent_workers=int(hp.get("workers", 4)) > 0)
    text_lr = float(hp.get("text_lr", hp["lr"]))
    tparams = [p for n, p in gen.named_parameters() if n.startswith("text.")]
    oparams = [p for n, p in gen.named_parameters() if not n.startswith("text.")]
    groups = [{"params": oparams, "lr": float(hp["lr"])}]
    if tparams and not hp.get("freeze_text", False):
        groups.append({"params": tparams, "lr": text_lr})
    elif tparams:
        for p in tparams:
            p.requires_grad_(False)
    opt = torch.optim.AdamW(groups, weight_decay=float(hp["weight_decay"]))
    steps_per_epoch = max(1, len(loader) // accum)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(total, int(hp.get("warmup_steps", 0))))
    start, step = 0, 0
    ema_decay = float(hp.get("ema") or 0.0)
    ema = None
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        gen.load_state_dict(st["model"])
        if "opt" in st and st.get("epoch", epochs) < epochs:
            opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, step = st.get("epoch", epochs), st.get("step", 0)
        if ema_decay > 0:
            ema = EMA(gen, ema_decay)
            if "ema" in st:
                ema.load_state_dict(st["ema"])
            else:
                log("generator: checkpoint has no EMA state; EMA restarts from the current weights")
        log(f"generator: resumed at epoch {start}")
    elif ema_decay > 0:
        ema = EMA(gen, ema_decay)
    clip = hp.get("grad_clip")
    w_len = float(hp.get("length_loss_weight", 1.0))
    p_uncond = float(hp.get("p_uncond") or 0.0)
    save_every = int(hp.get("save_every", 10))
    for ep in range(start, epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        t0 = time.time()
        acc = {"loss": 0.0, "loss_pose": 0.0, "loss_expr": 0.0, "loss_len": 0.0}
        nb = 0
        opt.zero_grad(set_to_none=True)
        for i, (feats, mask, texts, srcs) in enumerate(loader):
            B = feats.shape[0]
            n_frames = mask.sum().to(device)
            last_of_step = (i + 1) % accum == 0
            for s0 in range(0, B, micro):
                m = mask[s0:s0 + micro]
                T = int(m.sum(1).max())
                f = feats[s0:s0 + micro, :T].to(device, non_blocking=True)
                m = m[:, :T].to(device, non_blocking=True)
                rot, expr = codec.encode(f)
                text, text_pad = gen.encode_text(texts[s0:s0 + micro], srcs[s0:s0 + micro], device)
                sync = last_of_step and s0 + micro >= B
                ctx = model.no_sync() if (world > 1 and not sync) else contextlib.nullcontext()
                # the length head always sees the real text (it is trained on detached features)
                l_len_text, l_len_pad = text, text_pad
                if p_uncond > 0:  # C7: text dropout for classifier-free guidance
                    text, text_pad = gen.drop_text(text, text_pad, torch.rand(m.shape[0]) < p_uncond)
                with ctx:
                    loss, parts = flow_loss(model, corr, spec, rot, expr, m, text, text_pad, denom=n_frames,
                                            joint_weights=joint_weights, position_loss=position_loss)
                    share = m.shape[0] / B   # length loss is a per-sequence mean
                    l_len = gen.length_loss(gen.predict_log_length(l_len_text, l_len_pad), m.sum(1)) * share
                    ((loss + w_len * l_len) / accum).backward()
                acc["loss"] += float(loss.detach()); acc["loss_len"] += float(l_len.detach())
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + float(v)
            nb += 1
            if last_of_step:
                if clip:
                    torch.nn.utils.clip_grad_norm_(gen.parameters(), float(clip))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(gen)
                step += 1
        msg = {k: v / max(1, nb) for k, v in acc.items()}
        msg["lr"] = sched.get_last_lr()[0]
        if val_dataset is not None and rank == 0 and ((ep + 1) % save_every == 0 or ep + 1 == epochs):
            msg.update(validation_loss(gen, corr, spec, val_dataset, codec, device, bs,
                                       joint_weights=joint_weights, position_loss=position_loss))
        if rank == 0:
            log(f"generator epoch {ep + 1}/{epochs} " + " ".join(f"{k}={v:.4g}" for k, v in msg.items())
                + f" ({time.time() - t0:.0f}s)")
            if metrics:
                metrics(msg, step=ep + 1)
            if ckpt is not None:
                st = {"model": gen.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "epoch": ep + 1, "step": step}
                if ema is not None:
                    st["ema"] = ema.state_dict()
                torch.save(st, ckpt.with_suffix(".tmp"))
                os.replace(ckpt.with_suffix(".tmp"), ckpt)
                if (ep + 1) % save_every == 0:
                    snap = {"model": gen.state_dict(), "epoch": ep + 1}
                    if ema is not None:
                        snap["ema"] = ema.state_dict()
                    torch.save(snap, ckpt.parent / f"generator_ep{ep + 1}.pt")
    if ema is not None and hp.get("ema_eval", True):
        gen.load_state_dict(ema.weights_for(gen))  # sample with the EMA weights
        log(f"generator: using EMA weights (decay {ema_decay}) for sampling")
    gen.eval()
    return gen


@torch.no_grad()
def validation_loss(gen, corr, spec, dataset, codec, device, bs=64, max_batches=50, joint_weights=None,
                    position_loss=None):
    """Flow loss on the validation split (monitoring only; never the reported metric)."""
    gen.eval()
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=collate)
    g = torch.random.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(1234)
    tot, n = 0.0, 0
    for i, (feats, mask, texts, srcs) in enumerate(loader):
        if i >= max_batches:
            break
        feats, mask = feats.to(device), mask.to(device)
        rot, expr = codec.encode(feats)
        text, text_pad = gen.encode_text(texts, srcs, device)
        loss, _ = flow_loss(gen, corr, spec, rot, expr, mask, text, text_pad, joint_weights=joint_weights,
                            position_loss=position_loss)
        tot += float(loss); n += 1
    torch.random.set_rng_state(g[0])
    if g[1] is not None:
        torch.cuda.set_rng_state_all(g[1])
    gen.train()
    return {"val_loss": tot / max(1, n)}
