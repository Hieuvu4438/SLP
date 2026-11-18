import torch

from cprflow.objectives.lag import correlations, lag_loss


def test_delayed_signal_positive_peak_and_static_prediction():
    torch.manual_seed(17)
    a = torch.randn(32)
    b = torch.cat((torch.randn(3), a[:-3]))
    assert int(correlations(a, b).argmax()) - 15 == 3
    target = torch.stack((a, b, a, b), -1)
    pred = torch.zeros_like(target, requires_grad=True)
    loss, coverage = lag_loss(pred, target)
    assert loss > 0 and coverage > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    zero, count = lag_loss(pred, torch.zeros_like(target))
    assert count == 0 and zero == 0
