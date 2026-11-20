"""Training on fixed splits, with logical-batch normalization and epoch resume."""

import math
import time

import torch
from torch.utils.data import DataLoader

from .checkpoints import read_checkpoint, save_atomic
from .coupling import pair_sources
from .data import collate
from .flow import flow_loss, sample_source
from .paths import path_and_velocity, path_point


def rng(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def loader_for(dataset, hp, seed, shuffle):
    return DataLoader(dataset, batch_size=int(hp["batch_size"]), shuffle=shuffle,
                      num_workers=int(hp["workers"]), collate_fn=collate, generator=rng(seed))


def make_scheduler(opt, steps, warmup=0):
    if steps < 1 or not 0 <= warmup <= steps:
        raise ValueError("Invalid total optimizer steps or warmup")

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def restore_epoch(module, opt, scheduler, resume, signature, plan, device):
    if resume is None:
        return 0
    state = read_checkpoint(resume, signature)
    if state.get("plan") != plan:
        raise ValueError("Resume requires the same data, seed, optimizer, schedule and loss plan")
    module.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["rng_cuda"], device)
    return int(state["epoch"])


def save_epoch(module, opt, scheduler, path, signature, plan, epoch, device):
    save_atomic({"model": module.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch, "signature": signature,
                 "plan": plan, "rng_cpu": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, path)


def training_plan(hp, data, seed, extra):
    return {"data": data.key, "seed": seed, "hyperparameters": {
        k: v for k, v in hp.items() if k not in {"resume_from", "save_every"}}, **extra}


def train_correction(correction, energy, path, spec, data, codec, hp, device, seed, ctx,
                     signature, resume=None):
    opt = torch.optim.AdamW(correction.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    epochs = int(hp["epochs"])
    batches = math.ceil(len(data) / int(hp["batch_size"]))
    scheduler = make_scheduler(opt, epochs * batches)
    chunk = int(hp["frame_chunk"])
    if chunk < 0:
        raise ValueError("correction.frame_chunk must be nonnegative")
    plan = training_plan(hp, data, seed, {"device": str(device), "path": path})
    first = restore_epoch(correction, opt, scheduler, resume, signature, plan, device)
    for epoch in range(first, epochs):
        now = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        correction.train()
        total, frames, ot_clips, seen = 0., 0, 0, 0
        noise = rng(seed + 41_000_003 + epoch, device)
        for f, mask, texts, languages, _ in loader_for(data, hp, seed + 41_000_003 + epoch, True):
            f, mask = f.to(device), mask.to(device)
            target, expressions = codec.encode(f)
            source = sample_source(spec, target.shape[:-1], device, noise)
            if spec.coupling == "matched_ot":
                source, _, paired = pair_sources(source, expressions, target, mask, texts, languages,
                                                 spec.representation)
                ot_clips += paired
            seen += len(f)
            t = torch.rand(len(f), 1, 1, 1, device=device, generator=noise)
            t = t.expand(*mask.shape, 1, 1)[mask]
            target, source = target[mask], source[mask]
            n = len(target)
            size = chunk or n
            opt.zero_grad(set_to_none=True)
            for begin in range(0, n, size):
                select = slice(begin, begin + size)
                q, dq = path_and_velocity(path, source[select], target[select], t[select], correction)
                values = energy(q, dq)
                loss = values.sum() / n
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite correction energy")
                loss.backward()
                total += float(loss.detach()) * n
            clip = hp.get("grad_clip")
            if clip:
                torch.nn.utils.clip_grad_norm_(correction.parameters(), float(clip), error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            frames += n
        correction.eval()
        logs = {"correction_energy": total / frames, "correction_seconds": time.monotonic() - now,
                "correction_ot_fraction": ot_clips / seen}
        if device.type == "cuda":
            logs["correction_peak_cuda_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        ctx.log_metrics(logs, step=epoch + 1)
        save_epoch(correction, opt, scheduler, ctx.ckpt_dir / "correction.pt", signature, plan, epoch + 1, device)
    return correction.eval().requires_grad_(False)


@torch.no_grad()
def correction_diagnostics(correction, energy, spec, data, codec, device, seed, max_frames=128):
    from . import geometry as G

    goal = G.normalize(codec.encode(data[0][0][:max_frames].to(device))[0])
    noise = rng(seed + 91, device)
    source = sample_source(spec, (1, *goal.shape[:-1]), device, noise)[0]
    t = torch.rand(len(goal), 1, 1, device=device, generator=noise)
    baseline = "so3" if spec.path == "so3_corrected" else "geodesic"
    results = {}
    for path, net in [(baseline, None), (spec.path, correction)]:
        q, dq = path_and_velocity(path, source, goal, t, net)
        results[f"energy_{path}"] = float(energy(q, dq).mean())
        results[f"angular_speed_{path}"] = float(G.body_velocity(q, dq).norm(dim=-1).mean())
        # Estimate per-joint path lengths by a common fixed SO(3) quadrature grid.
        prev = source
        length = q.new_zeros(q.shape[:-1])
        for value in torch.linspace(0, 1, 17, device=device)[1:]:
            state = path_point(path, source, goal, torch.ones_like(t) * value, net)
            length = length + G.relative_rotvec(prev, state).norm(dim=-1)
            prev = state
        results[f"path_radians_{path}"] = float(length.mean())
    return results


def batch_objective(gen, correction, spec, codec, batch, noise, hp, device, fk, weights, backward=0., gradient_probe=False):
    feats, mask, texts, sources, inverse_dt = batch
    mask = mask.to(device)
    inverse_dt = inverse_dt.to(device)
    b, t = mask.shape
    n_joints = gen.n_joints
    initial_pose = sample_source(spec, (b, t, n_joints), device, noise)
    initial_expr = torch.randn(b, t, gen.expr_dim, device=device, generator=noise)
    times = torch.rand(b, device=device, generator=noise)
    paired = 0
    if spec.coupling == "matched_ot":
        target, _ = codec.encode(feats.to(device))
        initial_pose, initial_expr, paired = pair_sources(
            initial_pose, initial_expr, target, mask, texts, sources, spec.representation)
    counts = {"frames": int(mask.sum()), "pairs": int((inverse_dt > 0).sum()), "clips": b}
    if gen.coarse is not None:
        stride = gen.coarse.stride
        counts["bins"] = int(((mask.sum(1) + stride - 1) // stride).sum())
    micro = int(hp["micro_batch"])
    if micro < 1:
        raise ValueError("generator.micro_batch must be positive")
    sums = {"ot_fraction": paired / b}
    for start in range(0, b, micro):
        sl = slice(start, start + micro)
        end = int(mask[sl].sum(1).max())
        target, expr = codec.encode(feats[sl, :end].to(device))
        loss, parts = flow_loss(gen, correction, spec, target, expr, mask[sl, :end], texts[sl],
                                sources[sl], times[sl], initial_pose[sl, :end], initial_expr[sl, :end],
                                inverse_dt[sl, :end - 1], counts, fk, weights, hp["length_loss_weight"],
                                hp["coarse_loss_weight"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite generator objective")
        if gradient_probe and start == 0:
            norm = {}
            for name, coefficient in (("pose", spec.w_pose), ("position", spec.position_weight),
                                      ("velocity", spec.velocity_weight)):
                value = parts[name]
                if coefficient and value.requires_grad:
                    gradient, = torch.autograd.grad(coefficient * value, gen.out.weight,
                                                    retain_graph=True, allow_unused=True)
                    norm[name] = float(gradient.norm()) if gradient is not None else 0.
            for key, value in norm.items():
                sums[f"grad_{key}"] = value
            if norm.get("pose", 0.) > 0:
                for key in ("position", "velocity"):
                    if key in norm:
                        sums[f"grad_ratio_{key}"] = norm[key] / norm["pose"]
        if backward:
            (loss * backward).backward()
        for name, value in {**parts, "loss": loss.detach()}.items():
            sums[name] = sums.get(name, 0.) + float(value.detach())
    return sums


def train_generator(gen, correction, spec, data, val, codec, hp, device, seed, ctx, signature,
                    fk=None, weights=None, resume=None, upstream=None):
    text, other = [], []
    for name, param in gen.named_parameters():
        (text if name.startswith("text.") else other).append(param)
    groups = [{"params": other, "lr": float(hp["lr"])}]
    if hp["freeze_text"]:
        gen.text.requires_grad_(False)
    else:
        groups.append({"params": text, "lr": float(hp["text_lr"])})
    opt = torch.optim.AdamW(groups, weight_decay=float(hp["weight_decay"]))
    epochs, bs = int(hp["epochs"]), int(hp["batch_size"])
    effective = int(hp["effective_batch"])
    if min(bs, effective, epochs) < 1 or effective % bs:
        raise ValueError("effective_batch must be a positive multiple of batch_size")
    accum = effective // bs
    n_batches = math.ceil(len(data) / bs)
    scheduler = make_scheduler(opt, epochs * math.ceil(n_batches / accum), int(hp["warmup_steps"]))
    plan = training_plan(hp, data, seed, {"device": str(device), "upstream": upstream,
                                         "flow": spec.__dict__, "weights": weights.tolist() if weights is not None else None})
    first = restore_epoch(gen, opt, scheduler, resume, signature, plan, device)
    for epoch in range(first, epochs):
        now = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        gen.train()
        if hp["freeze_text"]:
            gen.text.eval()
        noise = rng(seed + 5_000_003 + epoch, device)
        totals = {}
        opt.zero_grad(set_to_none=True)
        loader = loader_for(data, hp, seed + 7_000_003 + epoch, True)
        for i, batch in enumerate(loader):
            group_start = (i // accum) * accum
            group_size = min(accum, n_batches - group_start)
            current = batch_objective(gen, correction, spec, codec, batch, noise, hp, device, fk, weights,
                                      backward=1. / group_size,
                                      gradient_probe=hp["probe_gradients"] and i == 0)
            for name, value in current.items():
                totals[name] = totals.get(name, 0.) + value
            if i - group_start + 1 == group_size:
                if hp["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(gen.parameters(), float(hp["grad_clip"]), error_if_nonfinite=True)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
        period = int(hp["save_every"])
        snapshot = period > 0 and (epoch + 1) % period == 0
        logs = {f"train_{key}": value if key.startswith("grad_") else value / n_batches
                for key, value in totals.items()}
        if snapshot or epoch + 1 == epochs:
            gen.eval()
            with torch.no_grad():
                iterator = loader_for(val, hp, seed + 20, False)
                holdout_noise = rng(seed + 23, device)
                val_loss = []
                for i, batch in enumerate(iterator):
                    if i >= int(hp["validation_batches"]):
                        break
                    val_loss.append(batch_objective(gen, correction, spec, codec, batch, holdout_noise,
                                                    hp, device, fk, weights)["loss"])
                if val_loss:
                    logs["val_loss"] = sum(val_loss) / len(val_loss)
        logs["generator_seconds"] = time.monotonic() - now
        if device.type == "cuda":
            logs["generator_peak_cuda_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        logs["learning_rate"] = scheduler.get_last_lr()[0]
        ctx.log_metrics(logs, step=epoch + 1)
        save_epoch(gen, opt, scheduler, ctx.ckpt_dir / "generator.pt", signature, plan, epoch + 1, device)
        if snapshot:
            save_atomic({"model": gen.state_dict(), "epoch": epoch + 1, "signature": signature},
                        ctx.ckpt_dir / f"generator_ep{epoch + 1}.pt")
    return gen.eval()
