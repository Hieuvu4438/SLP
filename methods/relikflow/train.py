"""Deterministic effective batching, global masked means, and resumable stages."""

import math
import time

import torch
from torch import distributed as dist

from .runtime import json_write, rng_state, set_rng, torch_write
from .source import generator


class Process:
    def __init__(self, device):
        import os

        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = device

    def sum(self, tensor):
        if self.world > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def barrier(self):
        if self.world > 1:
            dist.barrier()


def groups_for_epoch(clips, batch, seed, epoch, multilingual=False, shuffle=True):
    rng = generator("cpu", seed, "train", "sampler", "epoch", epoch)
    if not shuffle:
        order = list(range(len(clips)))
        return [order[i : i + batch] for i in range(0, len(order), batch)]
    if multilingual:
        sizes = {}
        for c in clips:
            key = c.row["corpus"]
            sizes[key] = sizes.get(key, 0) + 1
        # Per-clip mass yields aggregate mass proportional to sqrt(corpus size).
        weight = torch.tensor([sizes[c.row["corpus"]] ** -0.5 for c in clips])
        order = torch.multinomial(weight, len(clips), replacement=True, generator=rng).tolist()
    else:
        order = torch.randperm(len(clips), generator=rng).tolist()
    groups = []
    for start in range(0, len(order), batch * 12):
        window = sorted(order[start : start + batch * 12], key=lambda i: len(clips[i].q))
        groups.extend(window[i : i + batch] for i in range(0, len(window), batch))
    return [groups[i] for i in torch.randperm(len(groups), generator=rng).tolist()]


def run_stage(
    model,
    clips,
    settings,
    weights,
    count_terms,
    forward_terms,
    validate,
    dest,
    signature,
    process,
    *,
    seed,
    microbatch,
    resume=False,
    multilingual=False,
    shuffle=True,
):
    """Divide each local numerator by the global denominator before summing gradients.

    Counts depend only on masks. This avoids retaining all microbatch activations.
    """
    if not clips or microbatch < 1 or settings["epochs"] < 1:
        raise ValueError(f"No training items or invalid budget for {dest.name}")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=settings["lr"],
        weight_decay=settings["decay"],
        betas=tuple(settings.get("betas", (0.9, 0.999))),
    )
    steps = math.ceil(len(clips) / settings["batch"]) * settings["epochs"]
    warmup = int(settings.get("warmup", 0) * steps)

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    last = dest / "last.pt"
    best_path = dest / "best.pt"
    epoch0 = batch0 = step = stale = frames = 0
    best = None
    best_state = None
    history = []
    if resume and last.is_file():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature or checkpoint["world_size"] != process.world:
            raise ValueError(f"Run or sampler topology changed for {last}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch0, batch0, step = checkpoint["epoch"], checkpoint["batch"], checkpoint["step"]
        stale, best = checkpoint["stale"], checkpoint["best"]
        best_state = checkpoint["best_state"]
        if process.rank == 0 and best_state is not None:
            torch_write(best_path, best_state)
        history, frames = checkpoint["history"], checkpoint["frames"]
        set_rng(checkpoint["rng"][process.rank])
    elif last.exists():
        raise ValueError(f"Stage already exists at {dest}. Use --resume or a new output directory.")

    def save(epoch, batch):
        rng = [rng_state()]
        if process.world > 1:
            source = rng[0]
            rng = [None] * process.world
            dist.all_gather_object(rng, source)
        if process.rank == 0:
            torch_write(
                last,
                {
                    "signature": signature,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": None,
                    "epoch": epoch,
                    "batch": batch,
                    "step": step,
                    "rng": rng,
                    "world_size": process.world,
                    "stale": stale,
                    "best": best,
                    "best_state": best_state,
                    "history": history,
                    "frames": frames,
                },
            )

    elapsed_from = time.perf_counter()
    if process.rank == 0:
        print(f"{dest.name}: training from epoch {epoch0 + 1}, step {step}", flush=True)
    for epoch in range(epoch0, settings["epochs"]):
        if stale >= settings.get("patience", math.inf):
            break
        model.train()
        epoch_groups = groups_for_epoch(
            clips, settings["batch"], seed, epoch, multilingual, shuffle
        )
        for batch_index, indices in enumerate(epoch_groups):
            if epoch == epoch0 and batch_index < batch0:
                continue
            local = [
                (clips[i], batch_index * settings["batch"] + n)
                for n, i in enumerate(indices)
                if n % process.world == process.rank
            ]
            chunks = [local[i : i + microbatch] for i in range(0, len(local), microbatch)]
            counts = torch.zeros(len(weights), dtype=torch.float64, device=process.device)
            names = list(weights)
            for chunk in chunks:
                terms = count_terms([c for c, _ in chunk])
                counts += counts.new_tensor([float(terms.get(key, 0)) for key in names])
            process.sum(counts)
            if not (counts > 0).any():
                history.append({"epoch": epoch, "batch": batch_index, "skipped_empty": True})
                save(epoch, batch_index + 1)
                continue
            optimizer.zero_grad(set_to_none=True)
            diagnostics = []
            nums = counts.new_zeros(counts.shape)
            for chunk in chunks:
                terms, logs = forward_terms(chunk, epoch)
                diagnostics.extend(logs)
                loss = None
                for j, key in enumerate(names):
                    if counts[j] > 0:
                        num = terms[key]
                        nums[j] += num.detach().double()
                        weighted = weights[key] * num / counts[j].to(num)
                        loss = weighted if loss is None else loss + weighted
                if loss is not None:
                    loss.backward()
            process.sum(nums)
            for p in params:
                if process.world > 1:
                    used = process.sum(
                        torch.tensor(p.grad is not None, device=process.device, dtype=torch.int32)
                    )
                    if used.item():
                        p.grad = process.sum(p.grad if p.grad is not None else torch.zeros_like(p))
            grads = [p.grad for p in params if p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
            if not bool(nums.isfinite().all()) or not finite:
                raise FloatingPointError(
                    f"Non-finite loss/gradient at {dest}, IDs {[clips[i].id for i in indices]}"
                )
            if settings.get("clip"):
                torch.nn.utils.clip_grad_norm_(params, settings["clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 and process.rank == 0:
                print(f"{dest.name}: first optimizer step complete", flush=True)
            frames += sum(len(clips[i].q) for i in indices)
            history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "frames": frames,
                    "losses": {
                        k: float(nums[j] / counts[j]) for j, k in enumerate(names) if counts[j] > 0
                    },
                    "omitted": [k for j, k in enumerate(names) if counts[j] == 0],
                    "finite_gradients": finite,
                    "diagnostics": diagnostics,
                }
            )
            if step % settings.get("checkpoint_every_steps", 1) == 0:
                save(epoch, batch_index + 1)
        epoch_no = epoch + 1
        val = None
        if validate and (
            epoch_no % settings.get("val_period", 1) == 0 or epoch_no == settings["epochs"]
        ):
            model.eval()
            with torch.no_grad():
                key, val = validate()
            key = list(key)
            if not all(math.isfinite(v) for v in key):
                raise FloatingPointError(f"Non-finite validation metric at {dest}")
            if best is None or key < best:
                best, stale = key, 0
                best_state = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch_no,
                    "step": step,
                    "selection": key,
                    "signature": signature,
                }
            else:
                stale += 1
        if process.rank == 0 and best_state is not None:
            torch_write(best_path, best_state)
        history.append(
            {
                "epoch_completed": epoch_no,
                "step": step,
                "elapsed_seconds": time.perf_counter() - elapsed_from,
                "validation": val,
            }
        )
        save(epoch_no, 0)
        if process.rank == 0:
            print(
                f"{dest.name}: epoch {epoch_no}/{settings['epochs']}, step {step}, dev={val}",
                flush=True,
            )
    if best_state is None and process.rank == 0:
        torch_write(
            best_path,
            {
                "model": model.state_dict(),
                "epoch": settings["epochs"],
                "step": step,
                "signature": signature,
            },
        )
    process.barrier()
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True)["model"])
    if process.rank == 0:
        from .runtime import write_atomic
        import json

        write_atomic(
            dest / "dev_history.jsonl",
            lambda path: path.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in history)
            ),
        )
        json_write(
            dest / "budget.json",
            {
                "world_size": process.world,
                "microbatch": microbatch,
                "effective_batch": settings["batch"],
                "steps": step,
                "valid_frames": frames,
            },
        )
    return model
