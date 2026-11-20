"""signmflow_plus C7: text dropout and classifier-free guidance on the manifold."""

import pytest
import torch

from methods.signmflow_plus import geometry as G
from methods.signmflow_plus.flow import FlowSpec, guided, sample
from methods.signmflow_plus.model import SignGenerator


def _gen(null_text=True, seed=0):
    torch.manual_seed(seed)
    g = SignGenerator(rot_dim=4, latent=32, layers=2, heads=4, ff=64, time_dim=64, dropout=0.0,
                      max_frames=20, text_model=None, text_dim=16, null_text=null_text)
    if g.null_text is not None:
        torch.nn.init.normal_(g.null_text)
    return g.eval()


def test_drop_text_replaces_only_dropped_samples():
    gen = _gen()
    text, pad = torch.randn(3, 5, 16), torch.zeros(3, 5, dtype=torch.bool)
    pad[2, 3:] = True
    t2, p2 = gen.drop_text(text, pad, torch.tensor([False, True, False]))
    assert torch.equal(t2[0], text[0]) and torch.equal(t2[2], text[2]) and torch.equal(p2[2], pad[2])
    assert torch.allclose(t2[1], gen.null_text[0].expand(5, -1))
    assert p2[1].tolist() == [False, True, True, True, True]
    with pytest.raises(RuntimeError):
        _gen(null_text=False).drop_text(text, pad, torch.ones(3, dtype=torch.bool))


def test_paper_config_has_no_extra_parameter():
    assert "null_text" not in dict(_gen(null_text=False).named_parameters())


def test_guidance_weights():
    gen = _gen()
    x = G.normalize(torch.randn(2, 6, 41, 4))
    e, t = torch.randn(2, 6, 10), torch.full((2,), 0.3)
    text, tpad = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
    pad = torch.zeros(2, 6, dtype=torch.bool)
    spec = FlowSpec(path="geodesic")
    with torch.no_grad():
        c = gen(x, e, t, text, tpad, pad)
        u = gen(x, e, t, *gen.drop_text(text, tpad, torch.ones(2, dtype=torch.bool)), pad)
        p1, e1 = guided(gen, x, e, t, text, tpad, pad, spec, 1.0)
        assert torch.equal(p1, c[0]) and torch.equal(e1, c[1])
        p0, e0 = guided(gen, x, e, t, text, tpad, pad, spec, 0.0)
        assert torch.allclose(p0, G.normalize(u[0]), atol=1e-5) and torch.allclose(e0, u[1], atol=1e-6)
        p2, e2 = guided(gen, x, e, t, text, tpad, pad, spec, 2.0)
        assert torch.allclose(p2.norm(dim=-1), torch.ones(2, 6, 41), atol=1e-5)  # stays on the manifold
        xc, xu = G.normalize(c[0]), G.normalize(u[0])  # w = 2: twice as far from x^u along the geodesic
        assert torch.allclose(G.geodesic_distance(xu, p2), 2 * G.geodesic_distance(xu, xc), atol=1e-4)
        assert torch.allclose(e2, u[1] + 2 * (c[1] - u[1]), atol=1e-5)


def test_sampler_with_guidance_runs_and_w1_is_plain():
    gen = _gen()
    text, tpad = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
    spec = FlowSpec(path="geodesic")
    a = sample(gen, None, spec, text, tpad, [6, 4], steps=3, generator=torch.Generator().manual_seed(0))
    b = sample(gen, None, spec, text, tpad, [6, 4], steps=3, generator=torch.Generator().manual_seed(0), cfg_w=1.0)
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    rot, _, _ = sample(gen, None, spec, text, tpad, [6, 4], steps=3, cfg_w=2.5)
    assert torch.allclose(rot.norm(dim=-1), torch.ones(2, 6, 41), atol=1e-5)
