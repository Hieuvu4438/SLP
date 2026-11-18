"""Stage A objective, Stage B residual geodesic flow, samplers, CFG and the alpha knob.

Stage A (anchor):  mu = normalize(net(c, T)), trained with the reconstruction loss -> an
                   estimate of the conditional mean in joint space.
Stage B (flow) on (S^3)^41 x R^10, x1-prediction (SignNMFlow Eq. 13) along geodesics:
    source  'anchor' : x0 = Exp_mu(sigma_j xi), xi ~ N(0, I_3) in T_mu S^3;  e0 = e_mu + s_e n
            'uniform': x0 ~ uniform hemisphere, e0 ~ N(0, I)          (matched SignNMFlow-GE)
    path    x_t = Exp_{x0}(t Log_{x0} x1*),  e_t = (1-t) e0 + t e1
            x1* = x1 aligned to the hemisphere of x0 (shortest rotation) when ``align_pairs``
    loss    reconstruction_loss(normalize(phi(x_t, e_t, t, c[, mu])), x1)
Sampler (x1-prediction, NFE = steps):
    euler : x <- Exp_x( dt/(1-t) Log_x x^1 )          (DDIM-like; last step returns x^1)
    replan: x <- Exp_{x0}( (t+dt) Log_{x0} x^1 )      (SignNMFlow Eq. 53 with the geodesic)
CFG (on the prediction):  x^1 = Exp_{x^u}( w Log_{x^u} x^c ),  e^ = e^u + w (e^c - e^u),
    applied for t in [t_lo, t_hi), conditional-only elsewhere. w = 1 disables it.
Alpha knob:  y_alpha = Exp_mu( alpha Log_mu x ),  alpha = 0 -> mu, alpha = 1 -> the sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slp import geometry as G

from .losses import LossWeights, reconstruction_loss
from .representation import EXPR_DIM, N_JOINTS, unflatten


@dataclass
class FlowSpec:
    source: str = "anchor"          # anchor | uniform
    align_pairs: bool = True
    t_sampling: str = "logit_normal"  # uniform | logit_normal
    t_mean: float = 0.0
    t_std: float = 1.0
    t_uniform_mix: float = 0.2      # share of uniform t (logit-normal alone ~never visits t ~ 0,
                                    # where the sampler's first step and NFE 1 are evaluated)
    cfg_drop: float = 0.1           # text dropout probability (0 = no CFG training)

    def __post_init__(self):
        assert self.source in ("anchor", "uniform")
        assert self.t_sampling in ("uniform", "logit_normal")


def sample_t(spec: FlowSpec, n: int, device, generator=None) -> torch.Tensor:
    u = torch.rand(n, device=device, generator=generator)
    if spec.t_sampling == "uniform":
        return u
    t = torch.sigmoid(torch.randn(n, device=device, generator=generator) * spec.t_std + spec.t_mean)
    if spec.t_uniform_mix > 0:
        mix = torch.rand(n, device=device, generator=generator) < spec.t_uniform_mix
        t = torch.where(mix, u, t)
    return t


def sample_source(spec: FlowSpec, shape_bt, device, anchor=None, sigma=None, expr_sigma=None,
                  generator=None):
    """x0 [B,T,41,4], e0 [B,T,10]. ``anchor`` flat [B,T,174]; sigma [41,1]; expr_sigma [10]."""
    B, T = shape_bt
    if spec.source == "uniform":
        x0 = G.uniform_hemisphere((B, T, N_JOINTS), device=device, generator=generator)
        e0 = torch.randn(B, T, EXPR_DIM, device=device, generator=generator)
        return x0, e0
    mu, emu = unflatten(anchor)
    mu = G.normalize(mu)
    x0 = G.expmap(mu, G.tangent_gaussian(mu, sigma, generator=generator))
    e0 = emu + torch.randn(emu.shape, device=device, generator=generator) * expr_sigma
    return G.normalize(x0), e0


# ----------------------------------------------------------------------------- Stage A
def anchor_loss(net, w: LossWeights, fk, q1, e1, mask, text, text_pad, denom=None):
    pose, expr = net(None, None, None, text, text_pad, ~mask)
    return reconstruction_loss(w, fk, G.normalize(pose), expr, q1, e1, mask, denom)


# ----------------------------------------------------------------------------- Stage B
def flow_loss(net, spec: FlowSpec, w: LossWeights, fk, q1, e1, mask, text, text_pad, anchor=None,
              sigma=None, expr_sigma=None, denom=None):
    B, T = mask.shape
    dev = q1.device
    with torch.no_grad():
        x0, e0 = sample_source(spec, (B, T), dev, anchor, sigma, expr_sigma)
        x1 = G.align_sign(q1, x0) if spec.align_pairs else q1
        t = sample_t(spec, B, dev)
        x_t = G.normalize(G.geodesic(x0, x1, t[:, None, None, None]))
        e_t = (1 - t[:, None, None]) * e0 + t[:, None, None] * e1
    if spec.cfg_drop > 0:
        drop = torch.rand(B, device=dev) < spec.cfg_drop
        text, text_pad = net.drop_text(text, text_pad, drop)
    pose, expr = net(x_t, e_t, t, text, text_pad, ~mask, anchor=anchor)
    return reconstruction_loss(w, fk, G.normalize(pose), expr, q1, e1, mask, denom, q_target=x1)


def _predict(net, x, e, t, text, text_pad, pad, anchor, cfg_w, null):
    pose, expr = net(x, e, t, text, text_pad, pad, anchor=anchor)
    xc = G.normalize(pose)
    if cfg_w == 1.0 or null is None:
        return xc, expr
    pu, eu = net(x, e, t, null[0], null[1], pad, anchor=anchor)
    xu = G.normalize(pu)
    xg = G.normalize(G.geodesic(xu, G.align_sign(xc, xu), cfg_w))
    return xg, eu + cfg_w * (expr - eu)


@torch.no_grad()
def sample(net, spec: FlowSpec, text, text_pad, lengths, steps: int = 5, sampler: str = "euler",
           anchor=None, sigma=None, expr_sigma=None, cfg_w: float = 1.0,
           cfg_interval=(0.0, 1.0), generator=None):
    """Returns quats [B,T,41,4], expression [B,T,10], mask [B,T]; padded to max(lengths)."""
    assert sampler in ("euler", "replan")
    B, T = len(lengths), int(max(lengths))
    dev = text.device
    mask = torch.arange(T, device=dev)[None] < torch.as_tensor(lengths, device=dev)[:, None]
    if anchor is not None:
        anchor = anchor[:, :T]
    x0, e0 = sample_source(spec, (B, T), dev, anchor, sigma, expr_sigma, generator)
    null = net.null_like(text, text_pad) if cfg_w != 1.0 else None
    x, e = x0, e0
    dt = 1.0 / steps
    for k in range(steps):
        t = k * dt
        tb = torch.full((B,), t, device=dev)
        use_cfg = cfg_interval[0] <= t < cfg_interval[1]
        x1, e1 = _predict(net, x, e, tb, text, text_pad, ~mask, anchor, cfg_w if use_cfg else 1.0, null)
        if k == steps - 1:
            x, e = x1, e1
            break
        # pair with the representative the training path used (aligned to the path start or not)
        if sampler == "euler":
            x = G.geodesic(x, G.align_sign(x1, x) if spec.align_pairs else x1, dt / (1 - t))
        else:
            x = G.geodesic(x0, G.align_sign(x1, x0) if spec.align_pairs else x1, t + dt)
        x = G.normalize(x)
        e = e + dt / (1 - t) * (e1 - e)
    return G.normalize(x), e, mask


@torch.no_grad()
def predict_anchor(net, text, text_pad, lengths):
    B, T = len(lengths), int(max(lengths))
    dev = text.device
    mask = torch.arange(T, device=dev)[None] < torch.as_tensor(lengths, device=dev)[:, None]
    pose, expr = net(None, None, None, text, text_pad, ~mask)
    return G.normalize(pose), expr, mask


def interpolate(anchor_flat, q, e, alpha: float):
    """y_alpha on (S^3)^41 x R^10 between the anchor (alpha = 0) and the sample (alpha = 1)."""
    mu, emu = unflatten(anchor_flat)
    mu = G.normalize(mu)
    y = G.normalize(G.geodesic(mu, G.align_sign(q, mu), alpha))
    return y, emu + alpha * (e - emu)


@torch.no_grad()
def residual_scales(dataset, anchors_ok: bool, codec, device, max_items: int = 4000, bs: int = 64):
    """Per-joint tangent std sigma_j = sqrt(E|Log_mu x1|^2 / 3) and per-dim expression RMS of
    (x1 - mu) over a dataset that carries anchors. Returns (sigma [41,1], expr_sigma [10])."""
    from torch.utils.data import DataLoader

    from .data import collate

    assert anchors_ok
    n = min(len(dataset), max_items)
    loader = DataLoader(dataset.subset(range(n)), batch_size=bs, shuffle=False, collate_fn=collate)
    sq = torch.zeros(N_JOINTS, device=device, dtype=torch.float64)
    se = torch.zeros(EXPR_DIM, device=device, dtype=torch.float64)
    cnt = 0
    for feats, mask, _, _, anc in loader:
        feats, mask, anc = feats.to(device), mask.to(device), anc.to(device)
        q1, e1 = codec.encode(feats)
        mu, emu = unflatten(anc)
        mu = G.normalize(mu)
        v = G.logmap(mu, G.align_sign(q1, mu))
        sq += (v.pow(2).sum(-1)[mask]).sum(0).double()
        se += ((e1 - emu).pow(2)[mask]).sum(0).double()
        cnt += int(mask.sum())
    sigma = (sq / max(cnt, 1) / 3.0).sqrt().float()[:, None]
    es = (se / max(cnt, 1)).sqrt().float()
    return sigma, es
