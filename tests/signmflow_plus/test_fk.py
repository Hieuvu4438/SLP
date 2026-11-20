"""signmflow_plus C6: differentiable sparse SMPL-X FK matches the evaluation FK."""

import numpy as np
import pytest
import torch

from slp.evaluation.smplx_fk import smplx_dir

pytestmark = pytest.mark.skipif(not (smplx_dir() / "smplx" / "SMPLX_NEUTRAL.npz").exists(),
                                reason="licensed SMPL-X model files not present")


@pytest.fixture(scope="module")
def setup():
    from methods.signmflow_plus.fk import SparseSMPLX
    from methods.signmflow_plus.representation import MotionCodec, load_hand_mean
    from slp.evaluation.smplx_fk import SMPLXJoints, soke_stats

    mean, std = soke_stats()
    codec = MotionCodec(mean, std, load_hand_mean(smplx_dir()), "quat")
    return SparseSMPLX("cpu"), SMPLXJoints(device="cpu"), codec


def _feats(n=40, seed=0):
    # normalised features: sign-like poses around the data mean, plus some extreme ones
    return torch.from_numpy(np.random.default_rng(seed).normal(scale=0.8, size=(n, 133)).astype(np.float32))


def test_sparse_fk_matches_evaluation_fk(setup):
    from slp.evaluation.smplx_fk import UPPER_BODY_IDX

    fk, ref, codec = setup
    f = _feats()
    q, e = codec.encode(f)
    body, lh, rh = fk(q, e * codec.std[123:] + codec.mean[123:])
    j, l, r = ref(f.numpy())
    err = max(float((body - j[:, UPPER_BODY_IDX]).norm(dim=-1).max()),
              float((lh - l).norm(dim=-1).max()), float((rh - r).norm(dim=-1).max()))
    assert err < 1e-3  # acceptance test of C6: below 1 mm (float32)
    assert fk.n_support < 2000


def test_position_losses_zero_at_target_and_differentiable(setup):
    from methods.signmflow_plus import geometry as G
    from methods.signmflow_plus.fk import PositionLoss

    fk, _, codec = setup
    f = _feats(2 * 6, seed=1).reshape(2, 6, 133)
    q, e = codec.encode(f)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    loss = PositionLoss(fk, codec.mean[123:], codec.std[123:], 1.0, 1.0, 1.0)
    total, parts = loss(q, e, q, e, mask)
    assert float(total) < 1e-6 and set(parts) == {"loss_fk_body", "loss_fk_hand", "loss_fk_vel"}
    qh = G.normalize(q + 0.05 * torch.randn_like(q)).requires_grad_(True)
    total, parts = loss(qh, e, q, e, mask)
    total.backward()
    assert float(total.detach()) > 0 and torch.isfinite(qh.grad).all() and qh.grad.abs().sum() > 0
    assert qh.grad[1, 4:].abs().sum() == 0  # padded frames get no gradient
    # sign invariance: q and -q are the same rotation
    t2, _ = loss(-q, e, q, e, mask)
    assert float(t2) < 1e-6


def test_flow_loss_with_position_loss(setup):
    from methods.signmflow_plus.fk import PositionLoss
    from methods.signmflow_plus.flow import FlowSpec, flow_loss
    from methods.signmflow_plus.model import SignGenerator

    fk, _, codec = setup
    torch.manual_seed(0)
    gen = SignGenerator(rot_dim=4, latent=32, layers=1, heads=4, ff=64, time_dim=64, dropout=0.0,
                        max_frames=20, text_model=None, text_dim=16)
    q, e = codec.encode(_feats(2 * 5, seed=2).reshape(2, 5, 133))
    mask = torch.ones(2, 5, dtype=torch.bool)
    text, tpad = torch.randn(2, 3, 16), torch.zeros(2, 3, dtype=torch.bool)
    pl = PositionLoss(fk, codec.mean[123:], codec.std[123:], 0.1, 1.0, 0.5)
    loss, parts = flow_loss(gen, None, FlowSpec(path="geodesic"), q, e, mask, text, tpad, position_loss=pl)
    loss.backward()
    assert torch.isfinite(loss) and {"loss_fk_body", "loss_fk_hand", "loss_fk_vel"} <= set(parts)
    with pytest.raises(ValueError):
        flow_loss(gen, None, FlowSpec(path="geodesic", prediction="velocity"), q, e, mask, text, tpad,
                  position_loss=pl)
