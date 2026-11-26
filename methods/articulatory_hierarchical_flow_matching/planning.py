"""Train-only movement segmentation and local, text-predicted flow conditioning."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from slp.temporal_boundaries import boundary_evidence
from slp.temporal_planner import PlanBatch, TextToPlan
from slp.utils.torch_training import train_stage

from .data import text_input

CONTENT_DIM = 83
PART_WIDTHS = (6, 12, 30, 35)
MODES = ("off", "uniform", "predicted", "shuffled")


def make_planner(text_dim, max_frames, cfg):
    stride = cfg["frame_stride"]
    return TextToPlan(content_width=CONTENT_DIM, text_width=text_dim,
                      hidden=cfg["width"], heads=cfg["heads"], layers=cfg["depth"],
                      max_duration=cfg["max_frames"] // stride,
                      max_latents=math.ceil(max_frames / stride))


def valid_directions(observation: Tensor):
    valid = torch.ones_like(observation, dtype=torch.bool)
    for side in range(2):
        orientation = slice(6 + side * 6, 12 + side * 6)
        ok = observation[..., orientation].norm(dim=-1) > 0.5
        valid[..., orientation] = ok[..., None]
        valid[..., 18 + side * 15:33 + side * 15] = ok[..., None]
    return valid


def teacher_plan(observation: Tensor, mean: Tensor, std: Tensor, cfg: dict):
    # A stride cell contains only the available frames, including the last incomplete cell.
    stride = cfg["frame_stride"]
    length = math.ceil(len(observation) / stride)
    valid = valid_directions(observation)
    standard = ((observation - mean) / std).clamp(-8, 8).masked_fill(~valid, 0)
    extra = length * stride - len(observation)
    total = F.pad(standard, (0, 0, 0, extra)).reshape(length, stride, CONTENT_DIM).sum(1)
    count = F.pad(valid, (0, 0, 0, extra)).reshape(length, stride, CONTENT_DIM).sum(1)
    cells = total / count.clamp_min(1)
    parts = [F.pad(p, (0, max(PART_WIDTHS) - p.shape[-1]))
             * (max(PART_WIDTHS) / p.shape[-1]) ** 0.5 for p in cells.split(PART_WIDTHS, dim=-1)]
    evidence = boundary_evidence(torch.stack(parts, dim=1), (1, 2, 4)).double().numpy()

    # Viterbi over positive-duration partitions. The duration prior and boundary reward
    # admit short events while preventing a constant trace from fragmenting into single cells.
    minimum = cfg["min_frames"] // stride
    maximum = cfg["max_frames"] // stride
    expected = cfg["target_frames"] / stride
    best = np.full(length + 1, -np.inf)
    best[0] = 0.0
    back = np.zeros(length + 1, dtype=np.int64)
    for end in range(1, length + 1):
        low = min(minimum, length)
        allowed = np.arange(low, min(maximum, end) + 1)
        if not len(allowed):
            continue
        score = (best[end - allowed] + 0.8 * evidence[end]
                 - 0.45 - 0.35 * ((allowed - expected) / expected) ** 2)
        choice = int(score.argmax())
        back[end] = allowed[choice]
        best[end] = score[choice]
    if not np.isfinite(best[-1]):
        raise ValueError(f"Segment duration support cannot cover {length} latent cells")
    content, durations = [], []
    end = length
    while end:
        duration = int(back[end])
        stop = min(end * stride, len(standard))
        start = (end - duration) * stride
        content.append(standard[start:stop].sum(0) / valid[start:stop].sum(0).clamp_min(1))
        durations.append(duration)
        end -= duration
    return torch.stack(content[::-1]), durations[::-1]


def positive_allocation(durations: list[int], frames: int) -> list[int]:
    """Hamilton rounding after reserving one frame per segment."""
    if frames < 1 or not durations or min(durations) < 1:
        raise ValueError("Plan durations and frame budget must be positive")
    durations = durations[:frames]
    budget = frames - len(durations)
    total = sum(durations)
    bases = [1 + budget * d // total for d in durations]
    priority = sorted(range(len(durations)), key=lambda i: (-(budget * durations[i] % total), i))
    for i in priority[:frames - sum(bases)]:
        bases[i] += 1
    return bases


def unpack(plan: PlanBatch):
    return [(code[:n].detach().cpu(), duration[:n].tolist())
            for code, duration, n in zip(plan.contents, plan.durations, plan.lengths.tolist(),
                                         strict=True)]


def frame_condition(plans, lengths, size, mode, device) -> Tensor:
    per_clip = []
    for (contents, durations), frames in zip(plans, lengths.tolist(), strict=True):
        spans = positive_allocation(durations if mode != "uniform" else [1] * len(durations),
                                    frames)
        units = contents[:len(spans)].to(device)
        if mode == "shuffled" and len(units) > 1:
            units = units.roll(1, dims=0)
        active = torch.arange(len(spans), device=device).repeat_interleave(
            torch.tensor(spans, device=device), output_size=frames,
        )
        # A frame sees the content of its own unit and immediate neighbors.
        # Zero padding is explicit rather than repeating a boundary segment.
        neighbors = F.pad(units, (0, 0, 1, 1))
        context = torch.cat((neighbors[active], neighbors[active + 1], neighbors[active + 2]), -1)
        per_clip.append(F.pad(context, (0, 0, 0, size - frames)))
    return torch.stack(per_clip)


@torch.no_grad()
def predict_plans(method, keys):
    # Cache train predictions from the frozen text-to-plan model. Keys carry no pose or lengths.
    missing = list(dict.fromkeys(key for key in keys if key not in method.plan_cache))
    device = method.device
    max_units = min(method.planner_cfg["max_units"], method.planner.max_latents)
    with torch.autocast(device.type, enabled=False):
        for start in range(0, len(missing), method.sampler["batch_size"]):
            rows = missing[start:start + method.sampler["batch_size"]]
            lang = torch.tensor([key[1] for key in rows], device=device)
            encoded, mask = method.planner_text([key[0] for key in rows], lang)
            result = method.planner.generate_greedy(encoded.float(), mask, max_units=max_units)
            method.plan_cache.update(zip(rows, unpack(result.plan), strict=True))
            method.plan_stops.update(zip(rows, result.stop_reasons, strict=True))
    return [method.plan_cache[key] for key in keys]


def fit_planner(method, data, kept, train_loader, monitor_loader, descriptor):
    cfg = method.planner_cfg
    method.planner_text.eval()
    observations = []
    totals = torch.zeros(CONTENT_DIM, dtype=torch.float64)
    squares, counts = totals.clone(), totals.clone()
    training = set(kept)
    for i, index in enumerate(data.indices):
        poses = torch.as_tensor(data.split.targets[index])
        values = descriptor(poses[None])[0, :, :CONTENT_DIM].cpu()
        observations.append(values)
        if i in training:
            valid = valid_directions(values)
            clean = values.double().masked_fill(~valid, 0)
            counts += valid.sum(0)
            totals += clean.sum(0)
            squares += clean.square().sum(0)
    mean = totals / counts.clamp_min(1)
    std = (squares / counts.clamp_min(1) - mean.square()).clamp_min(1e-8).sqrt()
    targets = [teacher_plan(pose, mean.float(), std.float(), cfg) for pose in observations]
    del observations
    planner = method.planner
    planner.requires_grad_(True)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=cfg["lr"], weight_decay=0.01)
    best = {}

    def criterion(batch):
        e, d = zip(*(targets[i] for i in batch.indices), strict=True)
        plan = PlanBatch.from_sequences(list(e), list(d)).to(method.device)
        encoded, mask = method.planner_text(batch.texts, batch.languages)
        values = planner.loss(encoded.float(), mask, plan)
        return {"loss": values.pop("total"), **values}

    def save_best():
        best["state_dict"] = {k: v.detach().cpu().clone() for k, v in planner.state_dict().items()}

    train_stage("planner", train_loader, monitor_loader, cfg, optimizer, (planner,), criterion,
                lambda: criterion, save_best, method.device, "fp32", method.ctx)
    planner.load_state_dict(best["state_dict"])
    planner.eval().requires_grad_(False)
    keys = [text_input(data.split.inputs[i]) for i in data.indices]
    plans = predict_plans(method, keys)
    return {"teacher_mean": mean, "teacher_std": std, "descriptor_rig": descriptor.provenance,
            "mean_target_units": sum(len(x[1]) for x in targets) / len(targets),
            "mean_generated_units": sum(len(x[1]) for x in plans) / len(plans),
            "eos_rate": sum(method.plan_stops[key] == "eos" for key in keys) / len(keys)}
