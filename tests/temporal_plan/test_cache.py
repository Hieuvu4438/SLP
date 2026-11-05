from dataclasses import replace

import pytest
import torch

from methods.temporal_plan import cache as cache_module
from methods.temporal_plan.cache import TeacherScoreCache
from methods.temporal_plan.segments import TeacherConfig, segment_scores


def test_teacher_cache_has_exact_values_correct_invalidation_and_cpu_lru(tmp_path, monkeypatch):
    features = torch.randn(7, 4, 3)
    cfg = TeacherConfig(max_duration=4)
    expected = segment_scores(features, cfg)
    calls = []
    original = cache_module.segment_scores
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(cache_module, "segment_scores", counted)
    cache = TeacherScoreCache(tmp_path, max_entries=1)
    actual = cache.get(features, cfg, "source_one")
    torch.testing.assert_close(actual, expected)
    actual.zero_()  # consumers cannot corrupt the cached CPU tensor
    torch.testing.assert_close(cache.get(features, cfg, "another_source"), expected)
    assert len(calls) == 1
    another = TeacherScoreCache(tmp_path, max_entries=1)
    torch.testing.assert_close(another.get(features, cfg), expected)
    assert len(calls) == 1
    cache.get(features+1, cfg)
    cache.get(features, replace(cfg, penalty=.9))
    assert len(calls) == 3 and len(cache.memory) == 1
    assert all(score.device.type == "cpu" for score in cache.memory.values())
    assert len(list(tmp_path.glob("*.pt"))) == 3
    payload = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert "source_id" in payload and "scoring_code_sha256" in payload["metadata"]


def test_corrupted_cache_is_rejected(tmp_path):
    features = torch.randn(3, 4, 2)
    cfg = TeacherConfig(max_duration=2)
    TeacherScoreCache(tmp_path, max_entries=0).get(features, cfg)
    file = next(tmp_path.glob("*.pt"))
    payload = torch.load(file, weights_only=True)
    payload["scores"][0, 0] += 1
    torch.save(payload, file)
    with pytest.raises(ValueError, match="cached"):
        TeacherScoreCache(tmp_path, max_entries=0).get(features, cfg)
