"""signmflow_plus training options: EMA (C4)."""

import pytest
import torch

from methods.signmflow_plus.training import EMA


def test_ema_update_rule_buffers_and_state_round_trip():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    ema = EMA(net, 0.9)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    net.train()(torch.randn(8, 3))  # changes BatchNorm running stats (buffers)
    ema.update(net)
    w = net[0].weight
    assert torch.allclose(ema.shadow["0.weight"], 0.9 * before["0.weight"] + 0.1 * w)
    assert torch.equal(ema.shadow["1.running_mean"], net[1].running_mean)  # buffers copied
    assert int(ema.shadow["1.num_batches_tracked"]) == 1
    other = EMA(net, 0.9)
    other.load_state_dict(ema.state_dict())
    assert all(torch.equal(other.shadow[k], ema.shadow[k]) for k in ema.shadow)
    net.load_state_dict(ema.weights_for(net))
    assert torch.allclose(net[0].weight, ema.shadow["0.weight"])


def test_ema_frozen_parameters_are_copied_not_averaged():
    net = torch.nn.Linear(2, 2)
    net.bias.requires_grad_(False)
    ema = EMA(net, 0.5)
    with torch.no_grad():
        net.bias.add_(2.0)
        net.weight.add_(2.0)
    ema.update(net)
    assert torch.equal(ema.shadow["bias"], net.bias)
    assert not torch.equal(ema.shadow["weight"], net.weight)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1])
def test_ema_rejects_invalid_decay(decay):
    with pytest.raises(ValueError):
        EMA(torch.nn.Linear(2, 2), decay)
