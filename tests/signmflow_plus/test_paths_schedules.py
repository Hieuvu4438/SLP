"""signmflow_plus C2/C3: extra paths (E1, E1-lin) and noise-level schedules (E2-F)."""

import pytest
import torch

from methods.signmflow_plus import geometry as G
from methods.signmflow_plus.flow import FlowSpec, flow_loss, sample, time_grid
from methods.signmflow_plus.model import SignGenerator
from methods.signmflow_plus.paths import CorrectionNet, parallel_part, path_and_velocity, path_point, time_shift


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _q(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return G.sign_convention(G.normalize(torch.randn(*shape, 4, generator=g)))


def _random_corr(J, seed=0):
    torch.manual_seed(seed)
    net = CorrectionNet(n_joints=J, hidden=16)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.4)
    return net


def test_time_shift_matches_sd3_noise_level_and_endpoints():
    t = torch.linspace(0, 1, 101)
    for s in (0.5, 1.0, 2.5, 4.0):
        k = time_shift(t, s)
        n = s * (1 - t) / (1 + (s - 1) * (1 - t))  # SD3 noise level, probes/probe_theory.py
        assert torch.allclose(k, 1 - n, atol=1e-12)
        assert k[0] == 0 and abs(float(k[-1]) - 1) < 1e-12
        assert (k[1:] > k[:-1]).all()
    assert (time_shift(t[1:-1], 2.5) < t[1:-1]).all()  # s > 1: slower progress (more time at noise)


def test_geoshift_with_s1_is_geodesic_and_has_shifted_progress():
    x0, x1 = _q(32, 5, seed=1), _q(32, 5, seed=2)
    t = torch.rand(32, 1, 1)
    assert torch.allclose(path_point("geoshift", x0, x1, t, shift=1.0), path_point("geodesic", x0, x1, t))
    xs = path_point("geoshift", x0, x1, t, shift=2.5)
    d = G.geodesic_distance(x0, xs) / G.geodesic_distance(x0, x1)
    assert torch.allclose(d, time_shift(t[..., 0], 2.5).expand_as(d), atol=1e-8)


def test_ng_parallel_stays_on_the_geodesic_curve():
    J = 4
    corr = _random_corr(J, seed=3)
    x0, x1 = _q(64, J, seed=4), _q(64, J, seed=5)
    t = torch.rand(64, 1, 1) * 0.9 + 0.05
    xp = path_point("ng_parallel", x0, x1, t, corr)
    xi = G.logmap(x0, x1)
    om = G.logmap(x0, xp)  # tangent of the reached point must be parallel to xi
    cos = (om * xi).sum(-1) / (om.norm(dim=-1) * xi.norm(dim=-1))
    assert torch.allclose(cos.abs(), torch.ones_like(cos), atol=1e-8)
    eps = corr(t, x0, x1)
    par = parallel_part(eps, xi)
    assert torch.allclose(((eps - par) * xi).sum(-1), torch.zeros(64, J), atol=1e-10)
    # endpoints are unchanged
    assert torch.allclose(path_point("ng_parallel", x0, x1, torch.zeros(64, 1, 1), corr), x0, atol=1e-10)
    assert torch.allclose(path_point("ng_parallel", x0, x1, torch.ones(64, 1, 1), corr), x1, atol=1e-8)


def test_linear_renorm_on_sphere_and_velocity_tangent():
    x0, x1 = _q(16, 3, seed=6), _q(16, 3, seed=7)
    t = torch.rand(16, 1, 1)
    xt, v = path_and_velocity("linear_renorm", x0, x1, t)
    assert torch.allclose(xt.norm(dim=-1), torch.ones(16, 3), atol=1e-12)
    assert torch.allclose(G.inner(xt, v), torch.zeros(16, 3, 1), atol=1e-10)
    h = 1e-6
    fd = (path_point("linear_renorm", x0, x1, t + h) - path_point("linear_renorm", x0, x1, t - h)) / (2 * h)
    assert torch.allclose(v, fd, atol=1e-6)


def test_time_grid_defaults_and_variants():
    assert time_grid(5) == [k * 0.2 for k in range(6)]
    for p, s in ((0.7, 1.0), (1.5, 1.0), (1.0, 2.5), (1.5, 3.0)):
        g = time_grid(6, p, s)
        assert g[0] == 0.0 and abs(g[-1] - 1.0) < 1e-12
        assert all(b > a for a, b in zip(g, g[1:]))


def _tiny_gen(seed=0):
    torch.manual_seed(seed)
    return SignGenerator(rot_dim=4, latent=32, layers=2, heads=4, ff=64, time_dim=64,
                         dropout=0.0, max_frames=50, text_model=None, text_dim=16)


@pytest.mark.parametrize("path", ["geoshift", "ng_parallel", "linear_renorm"])
@pytest.mark.parametrize("train_shift", [1.0, 2.5])
def test_new_paths_train_and_sample(path, train_shift):
    torch.set_default_dtype(torch.float32)
    try:
        gen, corr = _tiny_gen(), CorrectionNet(n_joints=41, hidden=32)
        spec = FlowSpec(path=path, shift=2.5, train_shift=train_shift)
        x1, e1 = _q(2, 7, 41, seed=1).float(), torch.randn(2, 7, 10)
        mask = torch.ones(2, 7, dtype=torch.bool)
        text, tpad = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
        loss, _ = flow_loss(gen, corr, spec, x1, e1, mask, text, tpad)
        loss.backward()
        assert torch.isfinite(loss)
        rot, e, _ = sample(gen, corr, spec, text, tpad, [7, 5], steps=4, grid_power=1.5, grid_shift=2.0)
        assert torch.allclose(rot.norm(dim=-1), torch.ones(2, 7, 41), atol=1e-5)
    finally:
        torch.set_default_dtype(torch.float64)


def test_defaults_reproduce_the_signmflow_replica_exactly():
    """Paper defaults: loss and sampler are identical to methods/signmflow (same seeds)."""
    from methods.signmflow import flow as ref_flow
    from methods.signmflow.paths import CorrectionNet as RefCorr

    torch.set_default_dtype(torch.float32)
    try:
        gen, corr = _tiny_gen(1), _random_corr(41, seed=2).float()
        ref_corr = RefCorr(n_joints=41, hidden=16)
        ref_corr.load_state_dict(corr.state_dict())
        x1, e1 = _q(2, 6, 41, seed=3).float(), torch.randn(2, 6, 10)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[1, 4:] = False
        text, tpad = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
        for path in ("naturalness", "geodesic", "linear"):
            torch.manual_seed(5)
            a, _ = flow_loss(gen, corr, FlowSpec(path=path), x1, e1, mask, text, tpad)
            torch.manual_seed(5)
            b, _ = ref_flow.flow_loss(gen, ref_corr, ref_flow.FlowSpec(path=path), x1, e1, mask, text, tpad)
            assert torch.equal(a, b), path
            ra = sample(gen, corr, FlowSpec(path=path), text, tpad, [6, 4], steps=5,
                        generator=torch.Generator().manual_seed(9))
            rb = ref_flow.sample(gen, ref_corr, ref_flow.FlowSpec(path=path), text, tpad, [6, 4], steps=5,
                                 generator=torch.Generator().manual_seed(9))
            assert all(torch.equal(u, v) for u, v in zip(ra, rb)), path
    finally:
        torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- C1 fixed lengths
def test_fixed_lengths_table_round_trip(tmp_path):
    import json

    import numpy as np

    from methods.signmflow_plus.method import read_lengths
    from methods.signmflow_plus.tools.lengths import lengths_from_npz

    np.savez(tmp_path / "p.npz", **{"dev__a": np.zeros((50, 133)), "dev__b": np.zeros((73, 133))})
    table = lengths_from_npz(tmp_path / "p.npz")
    json.dump(table, open(tmp_path / "l.json", "w"))
    assert read_lengths(tmp_path / "l.json", ["dev/b", "dev/a"], cap=400) == [73, 50]
    with pytest.raises(KeyError):
        read_lengths(tmp_path / "l.json", ["dev/c"], cap=400)
    with pytest.raises(ValueError):
        read_lengths(tmp_path / "l.json", ["dev/a"], cap=40)


# ----------------------------------------------------------------------------- C8 wrapped-normal source
def test_wrapped_normal_source():
    from methods.signmflow_plus.flow import sample_source

    mu = G.sign_convention(G.normalize(torch.tensor([[0.9, 0.1, -0.2, 0.3], [0.7, 0.5, 0.1, -0.4]])))
    x = G.wrapped_normal(mu, 0.25, (20000, 2), generator=torch.Generator().manual_seed(0))
    assert torch.allclose(x.norm(dim=-1), torch.ones(20000, 2), atol=1e-12)
    d2 = (G.geodesic_distance(mu.expand_as(x), x) ** 2).mean(0)
    assert torch.allclose(d2, torch.full((2,), 3 * 0.25 ** 2), rtol=0.05)  # E|v|^2 = 3 sigma^2
    tiny = G.wrapped_normal(mu, 1e-6, (10, 2))
    assert torch.allclose(tiny, mu.expand_as(tiny), atol=1e-5)  # sigma0 -> 0: the mean pose
    spec = FlowSpec(source="wrapped_normal", source_sigma=0.25)
    with pytest.raises(ValueError):
        sample_source(spec, (3, 2), "cpu")
    assert sample_source(spec, (3, 2), "cpu", mean=mu).shape == (3, 2, 4)
    with pytest.raises(ValueError):
        FlowSpec("6d", "linear", source="wrapped_normal")


def test_wrapped_normal_source_trains_and_samples():
    torch.set_default_dtype(torch.float32)
    try:
        gen = _tiny_gen()
        mu = _q(41, seed=3).float()
        spec = FlowSpec(path="geodesic", source="wrapped_normal", source_sigma=0.25)
        x1, e1 = _q(2, 5, 41, seed=4).float(), torch.randn(2, 5, 10)
        mask = torch.ones(2, 5, dtype=torch.bool)
        text, tpad = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
        loss, _ = flow_loss(gen, None, spec, x1, e1, mask, text, tpad, source_mean=mu)
        assert torch.isfinite(loss)
        rot, _, _ = sample(gen, None, spec, text, tpad, [5, 3], steps=3, source_mean=mu)
        assert torch.allclose(rot.norm(dim=-1), torch.ones(2, 5, 41), atol=1e-5)
    finally:
        torch.set_default_dtype(torch.float64)
