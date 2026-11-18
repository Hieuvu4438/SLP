import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from cprflow.config import DEFAULTS, load_config
from cprflow.data.cache import NumericCache, cache_key
from cprflow.evaluation.aggregate import aggregate_reports
from cprflow.evaluation.dtw import dtw_jpe
from cprflow.objectives.semantic import multi_positive_nce
from cprflow.sampling.cfg import classifier_free_guidance
from test_attention_masks import inputs, nonzero_model


def test_config_unknown_key_and_profiles():
    config = Path(__file__).parents[1] / "configs/base.yaml"
    with pytest.raises(ValueError, match="Unknown"):
        load_config(config, ["model.hidden=100"])
    with pytest.raises(ValueError, match="Locked"):
        load_config(config, ["geometry.num_expression=10"])
    for file in (config.parent / "runs").glob("*.yaml"):
        load_config(file)


def test_cache_invalidation_and_numeric_only(tmp_path):
    a = cache_key("latent", ae="old", stats="one", source="v1")
    b = cache_key("latent", ae="new", stats="one", source="v1")
    assert a != b
    cache = NumericCache(tmp_path, {"ae": "old"})
    cache.put("fixture", x=np.zeros((2, 3)))
    np.testing.assert_array_equal(cache.get("fixture")["x"], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Object"):
        cache.put("bad", x=np.array([{}], dtype=object))


def test_cfg_scale_one_skips_null_and_keeps_language_length():
    model = nonzero_model()
    inp = inputs()
    expected = model(**inp, drop_text=torch.zeros(2, dtype=torch.bool)).z
    result, calls = classifier_free_guidance(model, 1.0, **inp)
    assert calls == 1
    torch.testing.assert_close(result.z, expected, rtol=0, atol=0)
    result, calls = classifier_free_guidance(model, 2.0, **inp)
    assert calls == 2 and torch.isfinite(result.z).all()


def test_dtw_exact_path_average_and_tie_break():
    pred = np.array([0, 2, 3], dtype=float)[:, None, None] * np.ones((1, 1, 3))
    target = np.array([0, 1, 3], dtype=float)[:, None, None] * np.ones((1, 1, 3))
    result = dtw_jpe(pred, target)
    assert result["path"] == [(0, 0), (1, 1), (2, 2)]
    assert result["distance"] == pytest.approx(np.sqrt(3) / 3)
    zero = dtw_jpe(np.zeros((2, 1, 3)), np.zeros((3, 1, 3)))
    assert zero["path"] == [(0, 0), (0, 1), (1, 2)]


def test_protocol_mixing_rejected():
    report = {
        "protocol": "controlled_v1",
        "duration_source": "predicted",
        "relation_source": "generated",
        "oracle": False,
        "profile": "296",
        "metric_version": "v1",
        "asset_sha256": "asset",
        "fitting_revision": "r",
        "evaluator_sha256": "e",
        "dataset": "csl_daily",
        "metrics": {"jpe": 1},
        "coverage": 1,
        "checkpoint_sha256": "checkpoint",
        "run_id": "fixture",
        "split": "dev",
        "seeds": [1],
        "synthetic": True,
        "stats_sha256": {"motion": "stats"},
        "inference_correction": "off",
        "inference_teacher_sha256": None,
        "inference_calibration_sha256": None,
    }
    other = {**report, "dataset": "how2sign"}
    assert aggregate_reports([report, other])["metrics"]["jpe"] == 1
    for field, value in [("oracle", True), ("duration_source", "oracle"), ("profile", "133")]:
        with pytest.raises(ValueError, match="mixed"):
            aggregate_reports([report, {**other, field: value}])


def test_multi_positive_duplicates_and_empty_negatives():
    q = torch.tensor([[1.0, 0], [0, 1]], requires_grad=True)
    positive = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
    loss, valid = multi_positive_nce(q, q, positive, torch.ones(2, 2, dtype=torch.bool))
    assert not valid.any() and not loss.any()
    loss.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_statistics_numeric_tampering_is_rejected(tmp_path):
    from cprflow.data.cache import file_hash, save_numeric
    from cprflow.data.stats import save_stats
    from cprflow.training.artifacts import stats_path, update_index

    config = copy.deepcopy(DEFAULTS)
    config["paths"]["manifests"] = str(tmp_path / "manifests")
    config["paths"]["stats"] = str(tmp_path / "stats")
    (tmp_path / "manifests").mkdir()
    manifest = tmp_path / "manifests/train.jsonl"
    manifest.write_text("synthetic_fixture")
    stats = tmp_path / "stats/motion.npz"
    save_stats(
        stats,
        np.zeros(296),
        np.ones(296),
        {"train_manifest_sha256": file_hash(manifest), "max_train_frames": 1024},
    )
    update_index(config["paths"]["stats"], "motion", {"path": str(stats)})
    assert stats_path(config, "motion") == stats
    save_numeric(stats, mean=np.ones(296), std=np.ones(296))
    with pytest.raises(ValueError, match="numeric hash mismatch"):
        stats_path(config, "motion")


def test_conditional_diversity_requires_annotation_and_keeps_prompt_weights():
    from cprflow.evaluation.retrieval import conditional_diversity

    x = np.array([[0.0], [2.0], [1.0], [5.0], [10.0]])
    prompts = ["a", "a", "b", "b", "b"]
    assert conditional_diversity(x, prompts)["metric_status"] == "unavailable"
    result = conditional_diversity(x, prompts, [True, True, True, True, False])
    assert result["mean_per_prompt"] == 3.0 and result["valid_prompts"] == 2
