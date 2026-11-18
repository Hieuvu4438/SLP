"""Physical time diagnostics; correspondence is explicit for variable-length sequences."""

import numpy as np
import torch

from cprflow.geometry.rotations import rotation_6d_to_matrix, geodesic_distance
from cprflow.objectives.geometry import activities
from cprflow.objectives.lag import lag_loss


def angular_velocities(motion, timestamps):
    r = rotation_6d_to_matrix(motion[:, :246].reshape(-1, 41, 6))
    dt = (timestamps[1:] - timestamps[:-1]).clamp_min(1e-8)
    v = geodesic_distance(r[1:], r[:-1]) / dt[:, None]
    return {"head": v[:, 3], "left": v[:, 10:25].mean(-1), "right": v[:, 25:40].mean(-1)}


def high_frequency_fraction(values, dt, cutoff_hz=6.0):
    if len(values) < 4:
        return None
    x = np.asarray(values, dtype=np.float64)
    power = np.abs(np.fft.rfft(x - x.mean(0), axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(x), dt)
    return float(power[frequencies >= cutoff_hz].sum() / max(power.sum(), 1e-12))


def temporal_metrics(pred, target, body_model, motion_norm, scales, target_times):
    pt = torch.arange(len(pred), device=pred.device) / 30
    tt = torch.as_tensor(target_times, device=target.device, dtype=torch.float32)
    out = {}
    if len(pred) < 2 or len(target) < 2:
        return {"controlled_v1_temporal_coverage": 0.0}
    for label, velocities in [
        ("generated", angular_velocities(pred, pt)),
        ("reference", angular_velocities(target, tt)),
    ]:
        for group, v in velocities.items():
            out[f"controlled_v1_{label}_{group}_angular_velocity_rad_s"] = float(v.mean())
            hf = high_frequency_fraction(
                v.cpu().numpy(), float((pt if label == "generated" else tt).diff().mean())
            )
            if hf is not None:
                out[f"controlled_v1_{label}_{group}_high_frequency_fraction"] = hf
    pg, tg = body_model(pred), body_model(target)
    pa = activities(pred, pg, pt, motion_norm, scales)
    ta = activities(target, tg, tt, motion_norm, scales)
    # Lag comparison on endpoint-normalized time; reported separately from native duration.
    aligned = torch.nn.functional.interpolate(
        pa.T[None], size=len(ta), mode="linear", align_corners=True
    )[0].T
    lag, coverage = lag_loss(aligned, ta)
    out["controlled_v1_lag_kl_normalized_time"] = float(lag)
    out["controlled_v1_lag_valid_windows"] = coverage
    out["controlled_v1_temporal_coverage"] = 1.0
    return out
