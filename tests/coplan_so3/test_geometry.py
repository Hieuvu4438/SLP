import math

import pytest
import torch

from methods.coplan_so3.flow import interpolate
from methods.coplan_so3.geometry import so3
from methods.coplan_so3.sampling import lie_step


def test_exp_log_branch_and_derivatives():
    axis = torch.tensor([[1.0, 2.0, 3.0], [-3.0, 2.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    axis /= axis.norm(dim=-1, keepdim=True)
    for angle in (0.0, 1e-9, 1e-3, 1.0, math.pi - 1e-7, math.pi):
        w = axis * angle
        r = so3.exp(w)
        so3.assert_valid(r, 1e-12)
        torch.testing.assert_close(so3.exp(so3.log(r)), r, atol=1e-10, rtol=1e-9)
    v = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(so3.exp, (v,))
    assert torch.autograd.gradgradcheck(so3.exp, (v,))
    assert torch.autograd.gradcheck(lambda x: so3.log(so3.exp(x)), (v,))


def test_seam_and_target_velocity():
    v = torch.tensor([[0.0, 0.0, math.pi - 0.01], [0.0, 0.0, -math.pi + 0.01]], dtype=torch.float64)
    r0, r1 = so3.exp(v[:1]), so3.exp(v[1:])
    torch.testing.assert_close(so3.distance(r0, r1), torch.tensor([0.02], dtype=v.dtype))
    e0, e1 = v[:1], v[1:]
    # Add physical frame and joint axes.
    r0, r1 = r0[:, None, None], r1[:, None, None]
    e0, e1 = e0[:, None], e1[:, None]
    s = torch.tensor([0.73], dtype=v.dtype)
    r, e, w, u = interpolate(r0, e0, r1, e1, s)
    h = 1e-6
    rp, ep, *_ = interpolate(r0, e0, r1, e1, s + h)
    rm, em, *_ = interpolate(r0, e0, r1, e1, s - h)
    torch.testing.assert_close((rp - rm) / (2 * h), r @ so3.hat(w), atol=1e-9, rtol=1e-7)
    torch.testing.assert_close((ep - em) / (2 * h), u)
    for endpoint in (0.0, 1.0):
        assert torch.isfinite(interpolate(r0, e0, r1, e1, torch.tensor([endpoint]))[2]).all()


def test_6d_degenerate_and_columns():
    x = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]])
    r, bad = so3.from_6d(x, return_degenerate=True)
    assert bad.all()
    so3.assert_valid(r)
    random = so3.exp(torch.randn(10, 3))
    torch.testing.assert_close(so3.from_6d(so3.to_6d(random)), random)


@pytest.mark.parametrize("solver", ["euler", "midpoint"])
def test_current_state_sampler_and_convergence(solver):
    r = torch.eye(3, dtype=torch.float64)[None]
    e = torch.zeros(1, 1, dtype=torch.float64)
    w = torch.tensor([[0.1, 0.2, 0.3]], dtype=r.dtype)

    def field(r, e, s):
        return w * (1 + s), torch.ones_like(e) * (1 + s)

    errors = []
    for n in (10, 20):
        rr, ee = r, e
        for k in range(n):
            rr, ee = lie_step(field, rr, ee, k / n, 1 / n, solver)
        so3.assert_valid(rr, 1e-12)
        errors.append(float((rr - so3.exp(1.5 * w)).abs().max()))
        assert abs(float(ee) - 1.5) < 0.06
    if solver == "euler":
        assert errors[1] < 0.51 * errors[0]
    else:
        assert errors[0] < 1e-12


def test_curved_path_exact_derivative():
    from methods.coplan_so3.flow import exact_path_velocity

    a = torch.tensor([[[[0.2, -0.1, 0.3]]]], dtype=torch.float64)
    b = torch.tensor([[[[-0.1, 0.3, 0.2]]]], dtype=torch.float64)
    s = torch.tensor([0.37], dtype=torch.float64)

    def path(t):
        t = t[:, None, None, None]
        r = so3.exp(t * a) @ so3.exp(t * (1 - t) * b)
        e = (t[..., 0] * t[..., 0]).expand(1, 1, 2)
        return r, e

    r, e, w, v = exact_path_velocity(path, s)
    h = 1e-6
    rp, ep = path(s + h)
    rm, em = path(s - h)
    torch.testing.assert_close(r @ so3.hat(w), (rp - rm) / (2 * h), atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(v, (ep - em) / (2 * h), atol=1e-10, rtol=1e-8)
