import torch

from cprflow.models.ae import MotionAE


def test_output_and_gradients_padding_invariant():
    torch.manual_seed(2)
    ae = MotionAE().eval()
    short = torch.randn(1, 5, 296, requires_grad=True)
    batch = torch.randn(2, 13, 296, requires_grad=True)
    with torch.no_grad():
        batch[0, :5] = short[0]
    alone, z1 = ae(short, torch.tensor([5]))
    together, z2 = ae(batch, torch.tensor([5, 13]))
    torch.testing.assert_close(alone[0], together[0, :5], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(z1[0], z2[0, :, :3], rtol=1e-5, atol=1e-6)
    grad1 = torch.autograd.grad(alone.square().sum(), short)[0]
    grad2 = torch.autograd.grad(together[0, :5].square().sum(), batch)[0]
    torch.testing.assert_close(grad1[0], grad2[0, :5], rtol=1e-4, atol=1e-5)
    assert torch.equal(grad2[0, 5:], torch.zeros_like(grad2[0, 5:]))
