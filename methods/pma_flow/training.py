"""Single-GPU trainer shared by Stage A and Stage B ("training hygiene", proposal 5 Sec. 3.2 E).

* AdamW, linear warm-up + cosine decay over optimiser steps, optional gradient clipping;
* exact large-batch emulation: each ``batch_size`` batch is processed in micro-batches whose
  losses are normalised by the batch's valid-frame count, ``effective_batch // batch_size``
  batches are accumulated per optimiser step;
* mBART: own learning rate, frozen for the first ``freeze_text_epochs`` epochs (the replication
  saw the text features collapse in epochs 2-4 when fine-tuned at the decoder lr);
* EMA of all weights; validation every ``val_every`` epochs on the EMA weights with a
  method-supplied proxy (never a reported metric); the best EMA weights are kept;
* bf16 autocast for the network (losses / FK in fp32), resumable checkpoints.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import collate


def cosine_lambda(total_steps: int, warmup: int = 0, floor: float = 0.0):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return f


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def state_dict(self):
        return {"n": self.n, "shadow": self.shadow}

    def load_state_dict(self, st):
        self.n, self.shadow = st["n"], st["shadow"]

    def weights(self, model) -> dict:
        """Full state dict of ``model`` with the EMA floats substituted."""
        sd = model.state_dict()
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v) for k, v in sd.items()}


@contextlib.contextmanager
def swapped(model, state: dict | None):
    """Temporarily load ``state`` into ``model`` (no-op for None)."""
    if state is None:
        yield
        return
    backup = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(backup)


def train_network(net, step_loss, dataset, hp: dict, device, log=print, ckpt: Path | None = None,
                  metrics=None, validate=None, tag: str = "net"):
    """Train ``net``; ``step_loss(net, feats, mask, text, text_pad, anchors, denom) -> (loss, parts)``.

    ``validate(net) -> dict`` with a ``score`` (lower is better), run on the EMA weights.
    On return ``net`` holds the selected weights (best-validated EMA, or final EMA)."""
    net.to(device).train()
    bs, epochs = int(hp["batch_size"]), int(hp["epochs"])
    micro = min(bs, int(hp.get("micro_batch") or bs))
    accum = max(1, int(hp.get("effective_batch", bs)) // bs)
    workers = int(hp.get("workers", 4))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=workers,
                        drop_last=len(dataset) >= bs, pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0)
    tparams = [p for n, p in net.named_parameters() if n.startswith("text.")]
    oparams = [p for n, p in net.named_parameters() if not n.startswith("text.")]
    groups = [{"params": oparams, "lr": float(hp["lr"])}]
    if tparams:
        groups.append({"params": tparams, "lr": float(hp.get("text_lr", hp["lr"]))})
    opt = torch.optim.AdamW(groups, weight_decay=float(hp.get("weight_decay", 0.0)),
                            betas=tuple(hp.get("betas", (0.9, 0.999))))
    steps_per_epoch = max(1, len(loader) // accum)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_lambda(total, int(hp.get("warmup_steps", 0)), float(hp.get("lr_floor", 0.0))))
    ema = EMA(net, float(hp["ema_decay"])) if hp.get("ema_decay") else None
    start, step, best, best_ep = 0, 0, math.inf, -1
    best_path = ckpt.parent / f"{tag}_best.pt" if ckpt is not None else None
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if ema is not None and st.get("ema") is not None:
            ema.load_state_dict(st["ema"])
        start, step = st["epoch"], st["step"]
        best, best_ep = st.get("best", math.inf), st.get("best_epoch", -1)
        log(f"{tag}: resumed at epoch {start}")
    freeze_ep = int(hp.get("freeze_text_epochs", 0))
    clip = hp.get("grad_clip")
    w_len = float(hp.get("length_loss_weight", 1.0))
    val_every = int(hp.get("val_every", 10))
    amp = hp.get("amp", "bf16") == "bf16" and device.type == "cuda"
    select = hp.get("select", "best") == "best" and validate is not None

    for ep in range(start, epochs):
        frozen = ep < freeze_ep
        for p in tparams:
            p.requires_grad_(not frozen)
        t0 = time.time()
        acc, nb = {}, 0
        opt.zero_grad(set_to_none=True)
        for i, (feats, mask, texts, srcs, anc) in enumerate(loader):
            B = feats.shape[0]
            denom = mask.sum().to(device).float()
            for s0 in range(0, B, micro):
                m = mask[s0:s0 + micro]
                T = int(m.sum(1).max())
                f = feats[s0:s0 + micro, :T].to(device, non_blocking=True)
                m = m[:, :T].to(device, non_blocking=True)
                a = None if anc is None else anc[s0:s0 + micro, :T].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    with torch.no_grad() if frozen else contextlib.nullcontext():
                        text, text_pad = net.encode_text(texts[s0:s0 + micro], srcs[s0:s0 + micro], device)
                    loss, parts = step_loss(net, f, m, text, text_pad, a, denom)
                    share = m.shape[0] / B
                    l_len = net.length_loss(net.predict_log_length(text, text_pad).float(), m.sum(1)) * share
                ((loss + w_len * l_len) / accum).backward()
                parts = {"loss": loss.detach(), "loss_len": l_len.detach(), **parts}
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + float(v) * (1.0 if k in ("loss", "loss_len") or
                                                           not k.startswith("err") else share)
            nb += 1
            if (i + 1) % accum == 0:
                if clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), float(clip))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if ema is not None:
                    ema.update(net)
        msg = {k: v / max(1, nb) for k, v in acc.items()}
        msg["lr"] = sched.get_last_lr()[0]
        last = ep + 1 == epochs
        if validate is not None and ((ep + 1) % val_every == 0 or last):
            net.eval()
            with swapped(net, ema.weights(net) if ema is not None else None):
                v = validate(net)
            net.train()
            msg.update({f"val_{k}": x for k, x in v.items()})
            if v["score"] < best:
                best, best_ep = v["score"], ep + 1
                if best_path is not None:
                    w = ema.weights(net) if ema is not None else net.state_dict()
                    torch.save({"model": w, "epoch": ep + 1, "score": best}, best_path)
        log(f"{tag} epoch {ep + 1}/{epochs} " + " ".join(f"{k}={x:.4g}" for k, x in msg.items())
            + f" ({time.time() - t0:.0f}s)")
        if metrics:
            metrics({f"{tag}/{k}": x for k, x in msg.items()}, step=ep + 1)
        if ckpt is not None:
            st = {"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "ema": ema.state_dict() if ema is not None else None, "epoch": ep + 1, "step": step,
                  "best": best, "best_epoch": best_ep}
            torch.save(st, ckpt.with_suffix(".tmp"))
            os.replace(ckpt.with_suffix(".tmp"), ckpt)

    for p in tparams:
        p.requires_grad_(True)
    if select and best_path is not None and best_path.exists():
        net.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
        log(f"{tag}: selected EMA weights of epoch {best_ep} (val score {best:.5g})")
    elif ema is not None:
        net.load_state_dict(ema.weights(net))
        log(f"{tag}: final EMA weights")
    net.eval()
    return {"best_score": best, "best_epoch": best_ep}
