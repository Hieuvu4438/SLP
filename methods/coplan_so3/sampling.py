"""Current-state Lie integration; exact flow-network evaluation accounting."""

from dataclasses import dataclass

import torch

from .data.dataset import length_mask
from .flow import source
from .geometry import so3

BUDGETS = {5: (2, 3), 10: (4, 6), 20: (5, 15)}


def lie_step(field, r, e, s, h, method="euler"):
    if method not in ("euler", "midpoint"):
        raise ValueError("Expected euler or midpoint")
    w, v = field(r, e, s)
    if method == "midpoint":
        rm, em = r @ so3.exp(0.5 * h * w), e + 0.5 * h * v
        w, v = field(rm, em, s + 0.5 * h)
    return r @ so3.exp(h * w), e + h * v


def sample_plan(prior, text, text_mask, languages, lengths, zmask, plan_dim, nfe, generator):
    if nfe < 1:
        raise ValueError("Plan prior requires a positive NFE budget")
    z = (
        torch.randn(
            len(text),
            zmask.shape[1],
            plan_dim,
            device=text.device,
            dtype=text.dtype,
            generator=generator,
        )
        * zmask[..., None]
    )
    for k in range(nfe):
        s = text.new_full((len(text),), k / nfe)
        z = z + prior(z, s, text, text_mask, languages, lengths, zmask) / nfe
        z = z * zmask[..., None]
    return z


@dataclass
class Generation:
    rotations: torch.Tensor
    expression: torch.Tensor
    lengths: torch.Tensor
    metadata: dict


@torch.no_grad()
def generate(
    model,
    texts,
    languages,
    *,
    seed=0,
    total_budget=10,
    lengths=None,
    solver="euler",
    max_frames=400,
    stochastic_length=False,
):
    if total_budget not in BUDGETS or solver not in ("euler", "midpoint"):
        raise ValueError("Budgets: 5/10/20; solvers: euler/midpoint")
    if not model.plan.frozen or not model.plan.stats.ready:
        raise RuntimeError("Load a trained frozen plan and statistics before generation")
    if not texts:
        raise ValueError("Nonempty batch required")
    model.eval()
    device = next(model.parameters()).device
    languages = torch.as_tensor(languages, device=device, dtype=torch.long)
    if languages.shape != (len(texts),) or ((languages < 0) | (languages > 2)).any():
        raise ValueError("One valid sign-language ID per input required")
    generator = torch.Generator(device=device).manual_seed(seed)
    text, tm = model.encode_text(texts, languages)
    controlled = lengths is not None
    if lengths is None:
        lengths = model.length_model.predict(
            text, tm, languages, max_frames, generator, stochastic_length
        )
    lengths = torch.as_tensor(lengths, device=device)
    if (
        lengths.shape != (len(texts),)
        or lengths.is_floating_point()
        or (lengths < 1).any()
        or (lengths > max_frames).any()
    ):
        raise ValueError("Integer lengths in [1,max_frames] required; no silent truncation")
    mask, zm = length_mask(lengths), model.plan.slot_mask(lengths)
    np, nd = BUDGETS[total_budget]
    z = sample_plan(model.prior, text, tm, languages, lengths, zm, model.plan_dim, np, generator)
    times = torch.arange(mask.shape[1], device=device, dtype=text.dtype)[None].expand(
        len(texts), -1
    )
    fps = text.new_tensor(model.fps_by_language)[languages]
    times = times / fps[:, None]
    rb, eb = model.plan.decode(z, zm, mask, times)
    r, e = source(rb, eb, mask, model.manifest, model.source_config, generator)
    calls = 0

    def field(r, e, s):
        nonlocal calls
        calls += 1
        ss = text.new_full((len(texts),), float(s))
        w, v, _ = model.dense(r, e, ss, text, tm, z, zm, languages, mask, times)
        return w, v

    if solver == "euler":
        for k in range(nd):
            r, e = lie_step(field, r, e, k / nd, 1 / nd)
    else:
        # Use a final Euler interval for odd NFE. Both solver choices consume
        # exactly the requested budget; midpoint stages count as two evaluations.
        steps = (nd + 1) // 2
        for k in range(steps):
            kind = "midpoint" if calls + 2 <= nd else "euler"
            r, e = lie_step(field, r, e, k / steps, 1 / steps, kind)
    so3.assert_valid(r)
    return Generation(
        r,
        e,
        lengths,
        {
            "length_controlled": controlled,
            "stochastic_length": stochastic_length,
            "seed": seed,
            "plan_nfe": np,
            "dense_nfe": calls,
            "total_flow_nfe": np + calls,
            "solver": solver,
            "dense_euler_steps": nd if solver == "euler" else nd % 2,
            "dense_midpoint_steps": 0 if solver == "euler" else nd // 2,
            "manifest_sha256": model.manifest.digest,
            "expression_coordinates": "normalized",
            "fps": float(fps[0]) if (fps == fps[0]).all() else fps.tolist(),
        },
    )
