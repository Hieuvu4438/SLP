"""slp.geometry (S^3 / quaternion ops) and slp.skeleton_fk (sparse differentiable SMPL-X FK)."""

import math

import numpy as np
import pytest
import torch

from slp import geometry as G


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _q(n, seed):
    g = torch.Generator().manual_seed(seed)
    return G.sign_convention(G.normalize(torch.randn(n, 4, generator=g)))


def test_rotation_round_trips():
    aa = torch.randn(500, 3) * 1.2
    R = G.aa_to_matrix(aa)
    assert torch.allclose(G.aa_to_matrix(G.quat_to_aa(G.aa_to_quat(aa))), R, atol=1e-8)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(500, 3, 3), atol=1e-10)
    q = G.aa_to_quat(aa)
    assert torch.allclose(G.quat_to_matrix(-q), R, atol=1e-12)  # q and -q: same rotation


def test_exp_log_inverse_and_distance():
    x, y = _q(300, 1), _q(300, 2)
    v = G.logmap(x, y)
    assert torch.allclose(G.inner(x, v), torch.zeros(300, 1), atol=1e-10)
    assert torch.allclose(G.expmap(x, v), y, atol=1e-8)
    assert torch.allclose(v.norm(dim=-1), G.geodesic_distance(x, y), atol=1e-8)
    assert torch.allclose(G.geodesic(x, y, 0.0), x, atol=1e-10)
    assert torch.allclose(G.geodesic(x, y, 1.0), y, atol=1e-8)
    mid = G.geodesic(x, y, 0.5)
    assert torch.allclose(G.geodesic_distance(x, mid), 0.5 * G.geodesic_distance(x, y), atol=1e-8)


def test_align_sign_gives_shortest_rotation():
    x, y = _q(400, 3), _q(400, 4)
    ya = G.align_sign(y, x)
    assert (G.inner(ya, x) >= 0).all()
    assert torch.allclose(G.quat_to_matrix(ya), G.quat_to_matrix(y), atol=1e-12)
    assert (G.geodesic_distance(x, ya) <= math.pi / 2 + 1e-9).all()


def test_tangent_gaussian_is_tangent_and_isotropic():
    x = _q(1, 5).expand(200000, 4)
    v = G.tangent_gaussian(x, 0.3, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(G.inner(x, v), torch.zeros(200000, 1), atol=1e-10)
    # E|v|^2 = 3 sigma^2 (3-D tangent space)
    assert abs(float(v.pow(2).sum(-1).mean()) - 3 * 0.09) < 3e-3


def test_sign_convention_canonical():
    q = G.normalize(torch.randn(1000, 4))
    assert torch.allclose(G.sign_convention(q), G.sign_convention(-q))
    assert (G.sign_convention(q)[:, 0] >= 0).all()


# --------------------------------------------------------------------------- skeleton FK
def _smplx_available():
    try:
        import smplx  # noqa: F401

        from slp.evaluation.smplx_fk import smplx_dir, stats_dir
        return (smplx_dir() / "smplx" / "SMPLX_NEUTRAL.npz").exists() and (stats_dir() / "mean.pt").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _smplx_available(), reason="SMPL-X model / SOKE stats not installed")
def test_skeleton_fk_matches_full_smplx_and_is_differentiable():
    torch.set_default_dtype(torch.float32)
    from slp.evaluation.smplx_fk import UPPER_BODY_IDX, SMPLXJoints
    from slp.skeleton_fk import SkeletonFK

    ref, fk = SMPLXJoints(device="cpu"), SkeletonFK()
    f = torch.from_numpy(np.random.RandomState(0).randn(64, 133).astype(np.float32) * 0.8)
    gj, gl, gr = ref(f.numpy())
    o = fk.from_features(f)
    assert torch.allclose(o["body"], gj[:, UPPER_BODY_IDX], atol=1e-5)
    assert torch.allclose(o["lhand"], gl, atol=1e-5)
    assert torch.allclose(o["rhand"], gr, atol=1e-5)
    assert torch.allclose(o["pelvis"], gj[:, 0], atol=1e-6)
    q = G.aa_to_quat(torch.randn(4, 41, 3) * 0.3).requires_grad_(True)
    fk.from_quats(q)["rhand"].sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad[:, 25:40].abs().sum() > 0
