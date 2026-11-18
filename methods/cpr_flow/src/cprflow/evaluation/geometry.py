import numpy as np
import torch

from cprflow.evaluation.dtw import dtw_jpe
from cprflow.geometry.relations import relation33
from cprflow.geometry.temporal import pool2


def geometry_metrics(pred, target, body_model, relation_normalizer=None):
    with torch.no_grad():
        p, t = body_model(pred), body_model(target)
        metrics = {}
        alignments = {}
        for group in ("body", "left", "right", "left_local", "right_local"):
            result = dtw_jpe(p[group].cpu().numpy(), t[group].cpu().numpy())
            metrics[f"controlled_v1_dtw_jpe_{group}"] = result["distance"]
            metrics[f"controlled_v1_dtw_path_length_{group}"] = result["path_length"]
            alignments[group] = result["path"]
        # One shared global-hand alignment for relation components.
        hp = torch.cat((p["left"], p["right"]), -2).cpu().numpy()
        ht = torch.cat((t["left"], t["right"]), -2).cpu().numpy()
        result = dtw_jpe(hp, ht)
        metrics["controlled_v1_dtw_jpe_global_hand21"] = result["distance"]
        pr = pool2(
            relation33(p["joints"], p["names"])[None], torch.tensor([len(pred)], device=pred.device)
        )[0]
        tr = pool2(
            relation33(t["joints"], t["names"])[None],
            torch.tensor([len(target)], device=target.device),
        )[0]
        low_path = list(dict.fromkeys((i // 2, j // 2) for i, j in result["path"]))
        pi, ti = zip(*low_path)
        diff = (pr[list(pi)] - tr[list(ti)]).abs().cpu().numpy()
        if relation_normalizer is not None:
            pn, tn = relation_normalizer(pr), relation_normalizer(tr)
            metrics["controlled_v1_relation_normalized_huber"] = float(
                torch.nn.functional.huber_loss(pn[list(pi)], tn[list(ti)])
            )
        for name, sl in [
            ("position", slice(0, 9)),
            ("normal", slice(9, 15)),
            ("distance", slice(15, 24)),
            ("proximity", slice(24, 33)),
        ]:
            metrics[f"controlled_v1_relation_{name}_mae"] = float(diff[:, sl].mean())
        metrics["controlled_v1_degenerate6d_rate"] = float(p["degenerate"].float().mean())
    return metrics


def kabsch(pred, target, similarity=False):
    p, t = np.asarray(pred), np.asarray(target)
    pc, tc = p.mean(-2, keepdims=True), t.mean(-2, keepdims=True)
    a, b = p - pc, t - tc
    u, s, vh = np.linalg.svd(np.swapaxes(a, -1, -2) @ b)
    correction = np.ones_like(s)
    correction[..., -1] = np.linalg.det(u @ vh)
    rotation = (u * correction[..., None, :]) @ vh
    out = a @ rotation
    if similarity:
        scale = (s * correction).sum(-1) / (a * a).sum((-2, -1)).clip(1e-12)
        out = out * scale[..., None, None]
    return out + tc
