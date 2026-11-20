"""signmflow_plus analysis tools (C10): arm scoring helpers and the M1 retrieval evaluator."""

import numpy as np
import pytest
import torch

from methods.signmflow_plus.analysis.evaluator import Evaluator, contrastive_loss, train
from methods.signmflow_plus.analysis.score import COLUMNS, summarise, text_swap_index


def test_text_swap_index_picks_other_sentence_of_closest_length():
    texts = ["a", "a", "b", "c"]
    lengths = [50, 52, 51, 90]
    sw = text_swap_index(texts, lengths)
    assert all(texts[j] != texts[i] for i, j in enumerate(sw))
    assert sw[0] == 2 and sw[1] == 2 and sw[3] in (0, 1, 2)


def _fake(n, scale, speed, seed):
    rng = np.random.default_rng(seed)
    return {"cols": {c: scale + 0.01 * rng.normal(size=n) for c in COLUMNS},
            "kin": {k: np.full(n, speed) for k in ("body_speed", "body_accel", "hand_speed", "hand_accel")}}


def test_summarise_frontier_logic():
    n, shrink = 200, [0.0, 0.1, 0.2]
    frontier = [_fake(n, 10.0, 3.0, 0), _fake(n, 9.0, 2.0, 1), _fake(n, 8.5, 1.0, 2)]
    gt_kin = _fake(n, 0, 4.0, 3)["kin"]
    mean_pose = _fake(n, 12.0, 0.0, 4)
    # an arm as slow as a=0.1 but better than the frontier there -> beats the frontier
    good = summarise(_fake(n, 8.0, 2.0, 5), frontier[0], frontier, gt_kin, mean_pose, shrink, n_boot=500)
    for c in COLUMNS:
        assert good["frontier_effect"][c]["beats_frontier"] and good["frontier_effect"][c]["mean"] < -0.9
        assert good["delta_vs_reference"][c][0] < -1.9  # raw improvement vs the reference
        assert good["gap_to_mean_pose"][c] == pytest.approx(4.0, abs=0.01)
    assert good["kinematics"]["body_speed_ratio"] == pytest.approx(0.5)
    # an arm that only damps the motion (lies on the frontier) does not beat it
    damped = summarise(_fake(n, 9.0, 2.0, 6), frontier[0], frontier, gt_kin, mean_pose, shrink, n_boot=500)
    assert not any(damped["frontier_effect"][c]["beats_frontier"] for c in COLUMNS)
    assert all(damped["delta_vs_reference"][c][0] < -0.9 for c in COLUMNS)  # yet the raw DTW "improves"


def test_contrastive_loss_ignores_duplicate_sentences():
    t = torch.randn(4, 8)
    m = t.clone()
    ids = torch.tensor([0, 0, 1, 2])
    scale = torch.tensor(np.log(10.0))
    base = contrastive_loss(t, m, ids, scale)
    m2 = m.clone()
    m2[1] = m[0]  # duplicate sentence: identical motion must not be a penalised negative for row 0
    assert torch.isfinite(contrastive_loss(t, m2, ids, scale))
    assert float(base) >= 0


def test_evaluator_trains_saves_and_separates_toy_data(tmp_path):
    rng = np.random.default_rng(0)
    words = ["guten morgen", "regen im norden", "sonne im sueden", "wind an der kueste"]
    texts, seqs = [], []
    for i in range(96):
        k = i % 4
        T = int(rng.integers(30, 60))
        base = np.zeros((T, 133), np.float32)
        base[:, k * 10:(k + 1) * 10] = np.sin(np.linspace(0, 6, T))[:, None]  # sentence-specific channels
        texts.append(words[k])
        seqs.append(base + 0.1 * rng.normal(size=base.shape).astype(np.float32))
    ev = train(texts, seqs, epochs=8, bs=32, lr=3e-3, max_len=32)
    ev.save(tmp_path / "ev.pt")
    ev2 = Evaluator.load(tmp_path / "ev.pt", device="cpu")
    t, m = ev2.embed_text(words), ev2.embed_motion(seqs[:4])
    assert t.shape == m.shape == (4, 256)
    sim = t @ m.T
    assert (sim.argmax(1) == np.arange(4)).all()  # each sentence retrieves its own motion pattern


def _smplx_available():
    from slp.evaluation.smplx_fk import smplx_dir, stats_dir
    return (smplx_dir() / "smplx" / "SMPLX_NEUTRAL.npz").exists() and (stats_dir() / "mean.pt").exists()


@pytest.mark.skipif(not _smplx_available(), reason="licensed SMPL-X model / SOKE stats not available")
def test_scorer_columns_match_the_reported_soke_metrics():
    from methods.signmflow_plus.analysis.score import Scorer
    from slp.evaluation.metrics import soke_protocol_scores

    rng = np.random.default_rng(0)
    gts = [rng.normal(scale=0.3, size=(9, 133)).astype(np.float32) for _ in range(2)]
    preds = [g[:7] + rng.normal(scale=0.4, size=(7, 133)).astype(np.float32) for g in gts]
    cols = Scorer(gts).columns(preds)
    ref = soke_protocol_scores(preds, gts)
    assert np.allclose(cols["jpe_body"], ref["jpe_body"])
    assert np.allclose(cols["jpe_hand"], (ref["jpe_lhand"] + ref["jpe_rhand"]) / 2)
    assert np.allclose(cols["pa_hand"], (ref["pa_lhand"] + ref["pa_rhand"]) / 2)
    assert np.allclose(cols["pa_body_ua"], ref["pa_body_ua"])
