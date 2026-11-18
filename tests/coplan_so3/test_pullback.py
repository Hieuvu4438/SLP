import torch

from methods.coplan_so3.geometry import so3
from methods.coplan_so3.geometry.physical_time_features import orders
from methods.coplan_so3.geometry.pullback import feature_jvp, kfm_loss


def test_physical_seconds_nonuniform_and_masks():
    times = torch.tensor([[0.0, 0.1, 0.3, 99.0]])
    x = times.square()[..., None]
    mask = torch.tensor([[True, True, True, False]])
    values = orders(x, times, mask)
    torch.testing.assert_close(values[2][0][0, 0], torch.tensor([2.0]))
    assert not values[2][1][0, 1]


def test_jvp_finite_difference_gradient_and_spd(features, batch):
    features.double()
    r, e = batch.rotations.double(), batch.expression.double()
    times, mask = batch.timestamps.double(), batch.mask
    features.fit_batch(r, e, mask, times, split="train")
    features.finalize(split="train")
    torch.manual_seed(31)
    dw = (torch.randn(*r.shape[:-2], 3, dtype=r.dtype) * 0.01).requires_grad_()
    de = (torch.randn_like(e) * 0.01).requires_grad_()
    jv = feature_jvp(features, r, e, dw, de, mask, times)

    def fn(a):
        return torch.cat(
            [x.flatten(1) for x in features(r @ so3.exp(a * dw), e + a * de, mask, times).values()],
            -1,
        )

    h = 1e-5
    torch.testing.assert_close(jv, (fn(h) - fn(-h)) / (2 * h), atol=2e-6, rtol=1e-5)
    loss, terms = kfm_loss(features, r, e, dw, de, mask, times)
    grad = torch.autograd.grad(loss, (dw, de), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad)
    direction = torch.randn_like(dw)

    def energy(a):
        return kfm_loss(features, r, e, dw + a * direction, de, mask, times)[0]

    fd = (energy(h) - energy(-h)) / (2 * h)
    torch.testing.assert_close((grad[0] * direction).sum(), fd, atol=1e-4, rtol=1e-5)
    assert terms["pullback"] > 0


def test_padding_invariance(features, batch):
    features.fit_batch(
        batch.rotations, batch.expression, batch.mask, batch.timestamps, split="train"
    )
    features.finalize(split="train")
    r, e, m, t = batch.rotations, batch.expression, batch.mask, batch.timestamps
    dw, de = torch.randn(*r.shape[:-2], 3), torch.randn_like(e)
    a = kfm_loss(features, r, e, dw, de, m, t)[0]
    rp = torch.cat((r, torch.eye(3).expand(2, 3, 5, 3, 3)), 1)
    ep = torch.nn.functional.pad(e, (0, 0, 0, 3))
    mp = torch.nn.functional.pad(m, (0, 3))
    tp = torch.nn.functional.pad(t, (0, 3))
    wp = torch.nn.functional.pad(dw, (0, 0, 0, 0, 0, 3))
    vp = torch.nn.functional.pad(de, (0, 0, 0, 3))
    b = kfm_loss(features, rp, ep, wp, vp, mp, tp)[0]
    torch.testing.assert_close(a, b)
