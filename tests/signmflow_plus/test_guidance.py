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


# ----------------------------------------------------------------------------- C9 self-conditioning
def _sc_gen():
    torch.manual_seed(0)
    return SignGenerator(rot_dim=4, latent=32, layers=2, heads=4, ff=64, time_dim=64, dropout=0.0,
                         max_frames=20, text_model=None, text_dim=16, self_cond=True)


def test_self_conditioning_training_feeds_detached_first_estimate():
    from methods.signmflow_plus.flow import flow_loss

    gen = _sc_gen()
    calls = []
    orig = gen.forward

    def spy(*a, **k):
        out = orig(*a, **k)
        calls.append((k.get("sc"), out))
        return out
    gen.forward = spy
    x1 = G.normalize(torch.randn(2, 5, 41, 4))
    e1, mask = torch.randn(2, 5, 10), torch.ones(2, 5, dtype=torch.bool)
    text, tpad = torch.randn(2, 3, 16), torch.zeros(2, 3, dtype=torch.bool)
    spec = FlowSpec(path="geodesic", self_cond=True, self_cond_p=1.0)
    loss, _ = flow_loss(gen, None, spec, x1, e1, mask, text, tpad)
    loss.backward()
    assert len(calls) == 2 and calls[0][0] is None
    sc = calls[1][0]
    assert not sc[0].requires_grad and torch.allclose(sc[0], G.normalize(calls[0][1][0]))
    calls.clear()
    flow_loss(gen, None, FlowSpec(path="geodesic", self_cond=True, self_cond_p=0.0), x1, e1, mask, text, tpad)
    assert len(calls) == 1 and calls[0][0] is None


def test_self_conditioning_sampler_and_input_checks():
    gen = _sc_gen().eval()
    text, tpad = torch.randn(1, 3, 16), torch.zeros(1, 3, dtype=torch.bool)
    rot, _, _ = sample(gen, None, FlowSpec(path="geodesic", self_cond=True), text, tpad, [6], steps=3)
    assert torch.allclose(rot.norm(dim=-1), torch.ones(1, 6, 41), atol=1e-5)
    plain = _gen(null_text=False)
    x, e = torch.randn(1, 4, 41, 4), torch.randn(1, 4, 10)
    with pytest.raises(ValueError):
        plain(x, e, torch.zeros(1), text, tpad, torch.zeros(1, 4, dtype=torch.bool), sc=(x, e))
