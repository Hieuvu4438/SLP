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


# ----------------------------------------------------------------------------- C5 lever-arm weights
def test_masked_mse_joint_weights():
    from methods.signmflow_plus.flow import _masked_mse

    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(3, 5, 41, 4, generator=g), torch.randn(3, 5, 41, 4, generator=g)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[2, 3:] = False
    assert torch.allclose(_masked_mse(a, b, mask, joint_weights=torch.ones(41)), _masked_mse(a, b, mask))
    w = torch.rand(41, generator=g)
    manual = ((((a - b) ** 2).sum(-1) * w).sum(-1) * mask).sum() / mask.sum()
    assert torch.allclose(_masked_mse(a, b, mask, joint_weights=w), manual)


def _smplx_available():
    from slp.evaluation.smplx_fk import smplx_dir
    return (smplx_dir() / "smplx" / "SMPLX_NEUTRAL.npz").exists()


@pytest.mark.skipif(not _smplx_available(), reason="licensed SMPL-X model files not present")
def test_soke_lever_arm_weights_structure():
    from methods.signmflow_plus.levers import SMPLX_OF_ROT, soke_lever_arm_weights

    w = soke_lever_arm_weights(floor=0.1)
    i = {j: k for k, j in enumerate(SMPLX_OF_ROT)}
    assert w.shape == (41,) and abs(float(w.sum()) - 41) < 1e-3  # acceptance test of C5
    assert w[i[22]] == pytest.approx(w.min())  # jaw: no measured point, floored
    for side in ((16, 18, 20, 27), (17, 19, 21, 42)):  # shoulder > elbow > wrist > distal finger joint
        sh, el, wr, tip = (w[i[j]] for j in side)
        assert sh > el > wr > tip > 0
    # proximal finger joints carry more than distal ones (index finger: 25 -> 26 -> 27)
    assert w[i[25]] > w[i[26]] > w[i[27]]
    # symmetric skeleton -> nearly symmetric weights
    assert w[i[16]] == pytest.approx(w[i[17]], rel=0.05)
