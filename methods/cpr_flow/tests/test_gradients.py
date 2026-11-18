import pytest
import torch

from cprflow.models.ae import MotionAE
from cprflow.objectives.geometry import activities


@pytest.mark.requires_body_model
def test_fk_gradient_to_latent_frozen_ae(body_model, normalizer):
    torch.manual_seed(8)
    ae = MotionAE(normalizer).freeze()
    z = torch.randn(1, 4, 3, 128, requires_grad=True)
    physical = ae.decode(z, torch.tensor([5]))
    geo = body_model(physical[0])
    geo["left"].square().sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.norm() > 0
    assert all(p.grad is None for p in ae.parameters())
    assert all(p.grad is None for p in body_model.parameters())


@pytest.mark.requires_body_model
def test_static_activity_finite_gradient(body_model, normalizer, identity_motion):
    x = identity_motion(4).requires_grad_()
    geo = body_model(x)
    scales = {
        key: 1.0 for key in ("activity_left", "activity_right", "activity_head", "activity_face")
    }
    act = activities(x, geo, torch.arange(4) / 30, normalizer, scales)
    act.sum().backward()
    assert torch.isfinite(x.grad).all()
