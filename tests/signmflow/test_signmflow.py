"""Unit tests for methods/signmflow (paper arXiv:2609.33339)."""

import math

import numpy as np
import pytest
import torch

from methods.signmflow import geometry as G
from methods.signmflow.flow import FlowSpec, flow_loss, sample
from methods.signmflow.model import SignGenerator
from methods.signmflow.naturalness import NaturalnessMeasure, extract_prototypes
from methods.signmflow.paths import CorrectionNet, path_and_velocity, path_point, velocity_chain_rule
from methods.signmflow.representation import MotionCodec



@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _rand_quats(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return G.sign_convention(G.normalize(torch.randn(*shape, 4, generator=g)))


def test_sign_convention_eq37():
    q = G.normalize(torch.randn(1000, 4))
    s = G.sign_convention(q)
    assert torch.allclose(s, G.sign_convention(-q))  # q and -q -> same representative
    assert (s[:, 0] >= 0).all()
    edge = torch.tensor([[0.0, -1, 0, 0], [0, 0, -0.6, 0.8], [0, 0, 0, -1], [-1.0, 0, 0, 0]])
    out = G.sign_convention(edge)
    first_nz = [row[row != 0][0] for row in out]
    assert all(v > 0 for v in first_nz)


def test_rotation_round_trips():
    aa = torch.randn(500, 3) * 1.2
    R = G.aa_to_matrix(aa)
    assert torch.allclose(G.aa_to_matrix(G.quat_to_aa(G.aa_to_quat(aa))), R, atol=1e-8)
    assert torch.allclose(G.aa_to_matrix(G.sixd_to_aa(G.aa_to_6d(aa))), R, atol=1e-8)
    assert torch.allclose(G.quat_to_matrix(G.matrix_to_quat(R)), R, atol=1e-8)


def test_exp_log_inverse_and_distance():
    x, y = _rand_quats(200, seed=1), _rand_quats(200, seed=2)
    v = G.logmap(x, y)
    assert torch.allclose(G.inner(x, v), torch.zeros(200, 1), atol=1e-10)  # tangent
    assert torch.allclose(G.expmap(x, v), y, atol=1e-8)
    assert torch.allclose(v.norm(dim=-1), G.geodesic_distance(x, y), atol=1e-8)


def test_property1_linear_interpolation_leaves_sphere():
    q0, q1 = _rand_quats(100, seed=3), _rand_quats(100, seed=4)
    t = torch.rand(100, 1) * 0.98 + 0.01
    qt = (1 - t) * q0 + t * q1
    expect = 1 - t[:, 0] * (1 - t[:, 0]) * ((q0 - q1) ** 2).sum(-1)  # Eq. 29
    assert torch.allclose((qt ** 2).sum(-1), expect, atol=1e-10)
    assert ((qt ** 2).sum(-1) < 1).all()


def test_naturalness_path_exact_on_manifold_and_endpoints():
    torch.manual_seed(0)
    net = CorrectionNet(n_joints=5, hidden=32)
    for p in net.parameters():  # arbitrary (non-zero) raw output, Sec. C.4.2
        torch.nn.init.normal_(p, std=0.5)
    x0, x1 = _rand_quats(64, 5, seed=5), _rand_quats(64, 5, seed=6)
    for tv in (0.0, 0.3, 0.77, 1.0):
        t = torch.full((64, 1, 1), tv)
        xt = path_point("naturalness", x0, x1, t, net)
        assert torch.allclose(xt.norm(dim=-1), torch.ones(64, 5), atol=1e-10)  # Eq. 51
        if tv == 0.0:
            assert torch.allclose(xt, x0, atol=1e-10)
        if tv == 1.0:
            assert torch.allclose(xt, x1, atol=1e-8)
    eps = net(torch.full((64, 1, 1), 0.4), x0, x1)
    assert torch.allclose(G.inner(x0, eps), torch.zeros(64, 5, 1), atol=1e-10)  # Eq. 49


def test_autodiff_velocity_matches_algorithm1_chain_rule():
    torch.manual_seed(1)
    net = CorrectionNet(n_joints=3, hidden=16)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.3)
    x0, x1 = _rand_quats(32, 3, seed=7), _rand_quats(32, 3, seed=8)
    t = torch.rand(32, 1, 1)
    _, v = path_and_velocity("naturalness", x0, x1, t, net)
    assert torch.allclose(v, velocity_chain_rule(x0, x1, t, net), atol=1e-8)
    h = 1e-6
    fd = (path_point("naturalness", x0, x1, t + h, net) - path_point("naturalness", x0, x1, t - h, net)) / (2 * h)
    assert torch.allclose(v, fd, atol=1e-6)


def test_geodesic_velocity_constant_speed():
    x0, x1 = _rand_quats(16, 2, seed=9), _rand_quats(16, 2, seed=10)
    d = G.geodesic_distance(x0, x1)
    for tv in (0.1, 0.5, 0.9):
        _, v = path_and_velocity("geodesic", x0, x1, torch.full((16, 1, 1), tv))
        assert torch.allclose(v.norm(dim=-1), d, atol=1e-8)


def test_naturalness_quadratic_form_matches_dense_metric():
    J, K = 3, 7
    protos, w = _rand_quats(J, K, seed=11), torch.rand(J, K) + 0.1
    w = w / w.sum(-1, keepdim=True)
    m = NaturalnessMeasure(protos, w, sigma=0.5, rho=1e-3)
    x = _rand_quats(20, J, seed=12)
    v = G.project(x, torch.randn(20, J, 4))
    dense = torch.einsum("njd,njde,nje->nj", v, m.metric_inverse(x), v)
    assert torch.allclose(m.quad_form(x, v), dense, rtol=1e-6, atol=1e-10)
    # direct Eq. 7-8 computation: G = m (S + rho I)^-1 on the tangent space
    delta = G.logmap(x[..., None, :], protos.expand(20, -1, -1, -1))
    wt = w * torch.exp(-(delta ** 2).sum(-1) / (2 * 0.5 ** 2))
    S = torch.einsum("njk,njkd,njke->njde", wt, delta, delta)
    mass = wt.sum(-1)
    ginv = (S + 1e-3 * torch.eye(4)) / mass[..., None, None]
    assert torch.allclose(m.quad_form(x, v), torch.einsum("njd,njde,nje->nj", v, ginv, v), rtol=1e-6)
    assert (m.quad_form(x, v) > 0).all()


def test_kmeans_prototypes():
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)  # two of the random centres are close; fixed data keeps k-means++ deterministic
        centers = _rand_quats(1, 5, seed=13).float()[0]
        pts = G.sign_convention(G.normalize(centers[torch.randint(5, (5000,))] + 0.01 * torch.randn(5000, 4)))
        frames = torch.stack([pts, pts.flip(0)], 1)  # [N,J=2,4]
        protos, weights = extract_prototypes(frames, K=5, iters=30, device="cpu")
        assert protos.shape == (2, 5, 4) and weights.shape == (2, 5)
        assert torch.allclose(protos.norm(dim=-1), torch.ones(2, 5), atol=1e-5)
        assert torch.allclose(weights.sum(-1), torch.ones(2), atol=1e-6) and (weights > 0).all()
        d = G.geodesic_distance(protos[0][:, None], centers[None]).min(0).values
        assert (d < 0.05).all()
    finally:
        torch.set_default_dtype(torch.float64)


def test_kmeans_runs_lloyd_to_convergence():
    """Regression: the stopping rule must not fire on the first pass (prev = inf)."""
    from methods.signmflow.naturalness import kmeans_joint
    torch.set_default_dtype(torch.float32)
    try:
        q = G.sign_convention(G.normalize(torch.randn(20000, 4, generator=torch.Generator().manual_seed(0))))

        def inertia(c):
            return float(((q[:, None] - c[None]) ** 2).sum(-1).min(1).values.mean())
        one, _ = kmeans_joint(q, 64, iters=1, seed=0)
        many, _ = kmeans_joint(q, 64, iters=300, seed=0)
        assert inertia(many) < 0.97 * inertia(one)
    finally:
        torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("rep", ["quat", "6d"])
def test_codec_round_trip_preserves_rotations(rep):
    mean, std = np.random.default_rng(0).normal(size=133) * 0.1, np.full(133, 0.5)
    hm = np.random.default_rng(1).normal(size=(41, 3)).astype(np.float32) * 0.2
    codec = MotionCodec(mean, std, hm, rep)
    feats = torch.randn(6, 133) * 0.8
    rot, expr = codec.encode(feats)
    assert rot.shape == (6, 41, 4 if rep == "quat" else 6) and expr.shape == (6, 10)
    back = codec.decode(rot, expr)
    raw = lambda f: (f[..., :123] * torch.as_tensor(std[:123]) + torch.as_tensor(mean[:123])).reshape(6, 41, 3)
    assert torch.allclose(G.aa_to_matrix(raw(back) + torch.as_tensor(hm)),
                          G.aa_to_matrix(raw(feats) + torch.as_tensor(hm)), atol=1e-6)
    assert torch.allclose(back[..., 123:], feats[..., 123:])
    if rep == "quat":
        assert (rot[..., 0] >= 0).all()


def _tiny_gen(rot_dim):
    return SignGenerator(rot_dim=rot_dim, latent=32, layers=2, heads=4, ff=64, time_dim=64,
                         dropout=0.0, max_frames=50, text_model=None, text_dim=16)


@pytest.mark.parametrize("spec", [FlowSpec(), FlowSpec(path="geodesic"), FlowSpec(path="linear"),
                                  FlowSpec("6d", "linear"), FlowSpec(prediction="velocity"),
                                  FlowSpec("6d", "linear", "velocity")])
def test_flow_loss_and_sampler_shapes(spec):
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)
        D = 4 if spec.representation == "quat" else 6
        gen, corr = _tiny_gen(D), CorrectionNet(n_joints=41, hidden=32)
        x1 = _rand_quats(2, 9, 41, seed=1).float() if D == 4 else torch.randn(2, 9, 41, 6)
        e1, mask = torch.randn(2, 9, 10), torch.ones(2, 9, dtype=torch.bool)
        mask[1, 6:] = False
        text, tpad = torch.randn(2, 5, 16), torch.zeros(2, 5, dtype=torch.bool)
        loss, parts = flow_loss(gen, corr, spec, x1, e1, mask, text, tpad)
        loss.backward()
        assert torch.isfinite(loss)
        rot, e, m = sample(gen, corr, spec, text, tpad, [9, 6], steps=3)
        assert rot.shape == (2, 9, 41, D) and e.shape == (2, 9, 10)
        if D == 4:
            assert torch.allclose(rot.norm(dim=-1), torch.ones(2, 9, 41), atol=1e-5)
    finally:
        torch.set_default_dtype(torch.float64)


def test_replanned_sampler_ends_at_prediction():
    """Eq. 53 at t + dt = 1 places the state exactly at the normalised prediction."""
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)
        gen, corr = _tiny_gen(4), CorrectionNet(n_joints=41, hidden=32)
        text, tpad = torch.randn(1, 3, 16), torch.zeros(1, 3, dtype=torch.bool)
        calls = []
        orig = gen.forward

        def spy(*a, **k):
            out = orig(*a, **k)
            calls.append(out)
            return out
        gen.forward = spy
        rot, e, _ = sample(gen, corr, FlowSpec(), text, tpad, [7], steps=5)
        assert len(calls) == 5
        assert torch.allclose(rot, G.normalize(calls[-1][0]), atol=1e-5)
        assert torch.allclose(e, calls[-1][1], atol=1e-5)
    finally:
        torch.set_default_dtype(torch.float64)


def test_masked_mse_reduction_and_microbatch_exactness():
    from methods.signmflow.flow import _masked_mse
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(8, 12, 41, 4, generator=g), torch.randn(8, 12, 41, 4, generator=g)
    mask = torch.arange(12)[None] < torch.randint(3, 13, (8, 1), generator=g)
    full = _masked_mse(a, b, mask, "sum")
    assert torch.allclose(full, 164 * _masked_mse(a, b, mask, "mean"))
    n = mask.sum()
    chunks = sum(_masked_mse(a[s:s + 3], b[s:s + 3], mask[s:s + 3], "sum", denom=n) for s in range(0, 8, 3))
    assert torch.allclose(chunks, full)
