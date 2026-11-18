import torch
import torch.nn.functional as F

from cprflow.objectives.geometry import fk_loss, rotation_loss, velocity_loss
from cprflow.types import PART_DIMS


def ae_objective(
    pred,
    batch,
    normalizer,
    body_model,
    scales,
    step=0,
    dense=False,
    weights=None,
    auxiliary_stride=2,
    target_geometry=None,
    batched_fk=False,
):
    weights = weights or {"param": 1.0, "rotation": 0.5, "fk": 1.0, "velocity": 0.1}
    if batched_fk and len(batch.lengths) > 1:
        from cprflow.objectives.ae_batch import objective

        return objective(
            pred,
            batch,
            normalizer,
            body_model,
            scales,
            step,
            dense,
            weights,
            auxiliary_stride,
            target_geometry,
        )
    totals = []
    logs = {k: [] for k in weights}
    logs["degenerate6d_rate"] = []
    for i, n in enumerate(batch.lengths.detach().cpu().tolist()):
        n = int(n)
        param = torch.stack(
            [
                F.huber_loss(p, t)
                for p, t in zip(
                    pred[i, :n].split(PART_DIMS, -1), batch.x[i, :n].split(PART_DIMS, -1)
                )
            ]
        ).mean()
        p, t = normalizer.inverse(pred[i, :n]), normalizer.inverse(batch.x[i, :n])
        indices = torch.arange(
            0 if dense else step % auxiliary_stride,
            n,
            1 if dense else auxiliary_stride,
            device=p.device,
        )
        pp, tt = p[indices], t[indices]
        pg = body_model(pp)
        if target_geometry is None:
            with torch.no_grad():
                tg = body_model(tt)
        else:
            tg = {
                key: (value[indices] if isinstance(value, torch.Tensor) else value)
                for key, value in target_geometry[i].items()
            }
        terms = {
            "param": param,
            "rotation": rotation_loss(pp, tt),
            "fk": fk_loss(pg, tg),
            "velocity": velocity_loss(
                pg, tg, pp[:, 246:], tt[:, 246:], batch.timestamps[i, indices], scales
            ),
        }
        totals.append(sum(weights[k] * v for k, v in terms.items()))
        logs["degenerate6d_rate"].append(pg["degenerate"].float().mean().detach())
        for k, v in terms.items():
            logs[k].append(v.detach())
    return torch.stack(totals), {k: torch.stack(v) for k, v in logs.items()}
