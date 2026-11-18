import pytest

from slp.evaluation import evaluate
from slp.evaluation.metrics import accuracy, macro_f1


def test_accuracy():
    assert accuracy([0, 1, 1, 2], [0, 1, 2, 2]) == pytest.approx(0.75)


def test_macro_f1_perfect_and_known():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    # class0: tp=1 fp=0 fn=1 -> 2/3 ; class1: tp=1 fp=1 fn=0 -> 2/3
    assert macro_f1([0, 1, 1], [0, 0, 1]) == pytest.approx(2 / 3)


def test_evaluate_length_mismatch():
    with pytest.raises(ValueError):
        evaluate([0, 1], [0], ["accuracy"])


# --------------------------------------------------------------- SLP metrics
import numpy as np  # noqa: E402

from slp.evaluation.metrics import (  # noqa: E402
    _pairwise_jpe, dtw_cost, dtw_jpe, dtw_jpe_norm, sibleu4, sibleu_unigram)


def _seq(T, J=4, seed=0):
    return np.random.default_rng(seed).normal(size=(T, J, 3))


def test_dtw_identical_is_zero():
    x = _seq(10)
    assert dtw_jpe([x], [x]) == pytest.approx(0.0)
    assert dtw_jpe_norm([x], [x]) == pytest.approx(0.0)


def test_dtw_root_translation_invariant():
    x = _seq(8)
    assert dtw_jpe([x + np.array([1.0, -2.0, 3.0])], [x]) == pytest.approx(0.0, abs=1e-9)


def test_dtw_cost_known_example():
    # optimal path (0,0)->(1,1)->(2,1): 0 + 0 + 0 = 0 with 3 steps
    cost = np.array([[0.0, 5.0], [5.0, 0.0], [9.0, 0.0]])
    acc, n = dtw_cost(cost)
    assert acc == pytest.approx(0.0) and n == 3
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    acc, n = dtw_cost(cost)  # diagonal 1+4=5 (vs 1+2+4, 1+3+4)
    assert acc == pytest.approx(5.0) and n == 2


def test_dtw_norm_equals_accumulated_over_path():
    x, y = _seq(6, seed=1), _seq(9, seed=2)
    acc, n = dtw_cost(_pairwise_jpe(x, y))
    assert dtw_jpe([x], [y]) == pytest.approx(acc)
    assert dtw_jpe_norm([x], [y]) == pytest.approx(acc / n)
    assert max(len(x), len(y)) <= n <= len(x) + len(y) - 1


def test_dtw_accumulated_scales_with_length_norm_does_not():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 4, 3))
    off = np.zeros((1, 4, 3)); off[0, 1:, 0] = 0.1  # constant 7.5 cm offset on 3 of 4 joints
    short_t, long_t = np.repeat(base, 10, 0), np.repeat(base, 40, 0)
    short_p, long_p = short_t + off, long_t + off
    assert dtw_jpe([long_p], [long_t]) == pytest.approx(4 * dtw_jpe([short_p], [short_t]))
    assert dtw_jpe_norm([long_p], [long_t]) == pytest.approx(dtw_jpe_norm([short_p], [short_t]))


def test_sibleu_unigram_matches_madis_formula_and_is_order_free():
    ref = [[1, 2, 3, 4]]
    assert sibleu_unigram([[1, 2, 3, 4]], ref) == pytest.approx(100.0)
    assert sibleu_unigram([[4, 3, 2, 1]], ref) == pytest.approx(100.0)  # order ignored
    # hyp shorter: overlap 2 / hyp 2, BP = exp(1 - 4/2)
    assert sibleu_unigram([[1, 9]], ref) == pytest.approx(100 * np.exp(-1) * 0.5)


def test_sibleu4_is_order_sensitive():
    ref = [[1, 2, 3, 4, 5, 6]]
    assert sibleu4([[1, 2, 3, 4, 5, 6]], ref) == pytest.approx(100.0)
    assert sibleu4([[6, 5, 4, 3, 2, 1]], ref) == pytest.approx(0.0)


# ------------------------------------------------------ SOKE protocol metrics
from slp.evaluation.metrics import pairwise_part_cost  # noqa: E402


def _soke_rigid_align(A, B):
    """Verbatim port of SOKE mGPT/utils/human_models.rigid_transform_3D + rigid_align."""
    n, _ = A.shape
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb) / n
    U, s, V = np.linalg.svd(H)
    R = V.T @ U.T
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = V.T @ U.T
    c = 1 / np.var(A, axis=0).sum() * np.sum(s)
    t = -(c * R) @ ca + cb
    return ((c * R) @ A.T).T + t


def _soke_l2_dist_align(x, y, wanted=None, align_idx=None):
    x = _soke_rigid_align(x, y) if align_idx is None else x - x[align_idx:align_idx + 1] + y[align_idx:align_idx + 1]
    if wanted is not None:
        x, y = x[wanted], y[wanted]
    return np.mean(np.sqrt(((x - y) ** 2).sum(axis=1)))


@pytest.mark.parametrize("align,idx", [("root", 0), ("procrustes", None)])
@pytest.mark.parametrize("wanted", [None, [1, 3, 4]])
def test_pairwise_part_cost_matches_soke(align, idx, wanted):
    g, r = _seq(5, J=6, seed=1), _seq(7, J=6, seed=2)
    ref = np.array([[_soke_l2_dist_align(a, b, wanted, idx) for b in r] for a in g])
    np.testing.assert_allclose(pairwise_part_cost(g, r, wanted, align), ref, rtol=1e-6, atol=1e-9)


def test_procrustes_cost_similarity_invariant():
    from scipy.spatial.transform import Rotation

    x = _seq(4, J=8, seed=3)
    R = Rotation.from_rotvec([0.3, -1.2, 0.7]).as_matrix()
    y = 2.5 * x @ R.T + np.array([1.0, -2.0, 0.5])
    assert np.allclose(np.diag(pairwise_part_cost(x, y, None, "procrustes")), 0, atol=1e-8)
    assert dtw_cost(pairwise_part_cost(x, y, None, "procrustes"))[0] == pytest.approx(0, abs=1e-7)


def _has_smplx():
    from slp.evaluation.smplx_fk import smplx_dir, stats_dir
    try:
        import smplx  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return (smplx_dir() / "smplx" / "SMPLX_NEUTRAL.npz").exists() and (stats_dir() / "mean.pt").exists()


@pytest.mark.skipif(not _has_smplx(), reason="licensed SMPL-X model / SOKE stats not available")
def test_soke_metrics_end_to_end():
    from slp.evaluation import evaluate
    from slp.evaluation.smplx_fk import default_fk

    rng = np.random.default_rng(0)
    gt = [rng.normal(scale=0.3, size=(12, 133)).astype(np.float32) for _ in range(2)]
    j, lh, rh = default_fk()(gt[0])
    assert j.shape == (12, 144, 3) and lh.shape == (12, 21, 3) and rh.shape == (12, 21, 3)
    names = ["soke_dtw_jpe_all", "soke_dtw_jpe_body", "soke_dtw_jpe_hand",
             "soke_dtw_pa_jpe_all", "soke_dtw_pa_jpe_body", "soke_dtw_pa_jpe_hand",
             "soke_dtw_pa_jpe_body_ua"]
    same = evaluate(gt, gt, names)
    assert all(v == pytest.approx(0, abs=1e-6) for v in same.values())
    noisy = [g + rng.normal(scale=0.5, size=g.shape).astype(np.float32) for g in gt]
    res = evaluate(noisy, gt, names)
    assert all(v > 0 for v in res.values())


@pytest.mark.skipif(not _has_smplx(), reason="licensed SMPL-X model / SOKE stats not available")
def test_soke_pa_body_upper_aligned_bounded_by_root_aligned():
    """Similarity fit on the evaluated joints (least squares) should not exceed root alignment here."""
    from slp.evaluation import evaluate

    rng = np.random.default_rng(1)
    gt = [rng.normal(scale=0.3, size=(10, 133)).astype(np.float32) for _ in range(2)]
    noisy = [g + rng.normal(scale=0.5, size=g.shape).astype(np.float32) for g in gt]
    r = evaluate(noisy, gt, ["soke_dtw_jpe_body", "soke_dtw_pa_jpe_body_ua"])
    assert 0 < r["soke_dtw_pa_jpe_body_ua"] <= r["soke_dtw_jpe_body"] + 1e-9
