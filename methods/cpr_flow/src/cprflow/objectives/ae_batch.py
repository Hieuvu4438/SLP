"""Pack valid auxiliary frames across clips, keep every per-clip reduction unchanged."""

import torch
import torch.nn.functional as F

from cprflow.objectives.geometry import fk_loss, rotation_loss, velocity_loss
from cprflow.types import PART_DIMS


def split_geometry(geometry, sizes):
    offset = 0
    result = []
    for size in sizes:
        result.append(
            {
                key: (value[offset : offset + size] if isinstance(value, torch.Tensor) else value)
                for key, value in geometry.items()
            }
        )
        offset += size
    return result


def objective(
    pred, batch, normalizer, body_model, scales, step, dense, weights, stride, target_geometry
):
    lengths = batch.lengths.detach().cpu().tolist()
    sampled = []
    targets = []
    indices = []
    params = []
    for i, n in enumerate(lengths):
        params.append(
            torch.stack(
                [
                    F.huber_loss(p, t)
                    for p, t in zip(
                        pred[i, :n].split(PART_DIMS, -1), batch.x[i, :n].split(PART_DIMS, -1)
                    )
                ]
            ).mean()
        )
        idx = torch.arange(
            0 if dense else step % stride, n, 1 if dense else stride, device=pred.device
        )
        indices.append(idx)
        sampled.append(normalizer.inverse(pred[i, :n])[idx])
        targets.append(normalizer.inverse(batch.x[i, :n])[idx])
    sizes = [len(x) for x in sampled]
    predicted = split_geometry(body_model(torch.cat(sampled, 0)), sizes)
    if target_geometry is None:
        with torch.no_grad():
            target = split_geometry(body_model(torch.cat(targets, 0)), sizes)
    else:
        target = [
            {
                key: (value[idx.to(value.device)] if isinstance(value, torch.Tensor) else value)
                for key, value in geo.items()
            }
            for geo, idx in zip(target_geometry, indices)
        ]
    vectors = []
    logs = {key: [] for key in ("param", "rotation", "fk", "velocity", "degenerate6d_rate")}
    for i, (pp, tt, pg, tg, idx) in enumerate(zip(sampled, targets, predicted, target, indices)):
        terms = {
            "param": params[i],
            "rotation": rotation_loss(pp, tt),
            "fk": fk_loss(pg, tg),
            "velocity": velocity_loss(
                pg, tg, pp[:, 246:], tt[:, 246:], batch.timestamps[i, idx], scales
            ),
        }
        vectors.append(sum(weights[key] * value for key, value in terms.items()))
        for key, value in terms.items():
            logs[key].append(value.detach())
        logs["degenerate6d_rate"].append(pg["degenerate"].float().mean().detach())
    return torch.stack(vectors), {key: torch.stack(value) for key, value in logs.items()}
