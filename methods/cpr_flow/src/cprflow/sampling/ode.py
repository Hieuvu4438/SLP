"""Masked Euler/Heun and deterministic DDIM; count actual field forwards."""

import torch

from cprflow.objectives.diffusion import CosineSchedule
from cprflow.sampling.cfg import classifier_free_guidance


@torch.no_grad()
def integrate(model, z, r, conditions, nfe=16, cfg=1.0, solver="euler", oracle_relation=None):
    if nfe < 1:
        raise ValueError("NFE must be positive")
    if solver == "heun" and nfe % 2:
        raise ValueError("Heun requires an even field-call budget")
    steps = nfe if solver == "euler" else nfe // 2
    if solver not in ("euler", "heun"):
        raise ValueError("Unknown ODE solver")
    valid = conditions["valid"]
    forwards = 0
    relation_noise = r.clone() if oracle_relation is not None and r is not None else None
    if oracle_relation is not None and (r is None or oracle_relation.shape != r.shape):
        raise ValueError("OracleR diagnostic requires a matching P5 relation endpoint")
    times = []
    for j in range(steps):
        t = z.new_full((len(z),), j / steps)
        if oracle_relation is not None:
            r = (1 - j / steps) * relation_noise + (j / steps) * oracle_relation
        pred, calls = classifier_free_guidance(model, cfg, z=z, r=r, t=t, **conditions)
        forwards += calls
        times.append(j / steps)
        nz = z + pred.z / steps
        nr = None if r is None else r + pred.r / steps
        if oracle_relation is not None:
            nr = (1 - (j + 1) / steps) * relation_noise + ((j + 1) / steps) * oracle_relation
        if solver == "heun":
            t = z.new_full((len(z),), (j + 1) / steps)
            second, calls = classifier_free_guidance(model, cfg, z=nz, r=nr, t=t, **conditions)
            forwards += calls
            times.append((j + 1) / steps)
            nz = z + (pred.z + second.z) / (2 * steps)
            if oracle_relation is None:
                nr = None if r is None else r + (pred.r + second.r) / (2 * steps)
        z = nz * valid[:, None, :, None]
        r = None if nr is None else nr * valid[..., None]
    return (
        z,
        r,
        {
            "solver": solver,
            "solver_steps": steps,
            "vector_field_calls": nfe,
            "batch_equivalent_forwards": forwards,
            "times": times,
            "oracle_relation": oracle_relation is not None,
        },
    )


@torch.no_grad()
def ddim_sample(model, z, conditions, nfe=16, cfg=1.0, schedule=None):
    schedule = schedule or CosineSchedule(device=z.device)
    indices = schedule.indices(nfe)
    forwards = 0
    for k, next_k in zip(indices[:-1], indices[1:]):
        pred, calls = classifier_free_guidance(
            model, cfg, z=z, r=None, t=z.new_full((len(z),), k / schedule.steps), **conditions
        )
        forwards += calls
        z = schedule.ddim(z, pred.z, k, next_k) * conditions["valid"][:, None, :, None]
    return (
        z,
        None,
        {
            "solver": "ddim_eta0",
            "solver_steps": nfe,
            "vector_field_calls": nfe,
            "batch_equivalent_forwards": forwards,
            "timestep_indices": indices,
        },
    )
