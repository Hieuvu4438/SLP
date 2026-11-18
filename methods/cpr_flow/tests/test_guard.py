import torch

from cprflow.sampling.guard import guarded_correction


def features(x):
    value = x.flatten().mean()
    embedding = torch.stack((torch.cos(value), torch.sin(value)))[None]
    return {"global": embedding, "parts": embedding[:, None].repeat(1, 4, 1)}


def test_semantic_violation_rolls_back_exactly():
    original = torch.zeros(1, 4, 1, 2)
    plan = torch.ones_like(original)
    calibration = {
        "approved": True,
        "confidence_threshold": 0.5,
        "epsilon_sem": 0.0,
        "epsilon_parts": {"left": 0.0, "right": 0.0, "face": 0.0},
    }
    z, log = guarded_correction(
        original,
        plan,
        lambda x: x,
        lambda x: x,
        features,
        torch.tensor([[1.0, 0.0]]),
        calibration,
        step_size=0.1,
    )
    assert log["proposed"] == 3 and log["accepted"] == 0
    assert torch.equal(z, original)
    assert torch.equal(plan, torch.ones_like(plan))


def test_acceptance_and_unapproved_gate():
    original = torch.zeros(1, 4, 1, 2)
    calibration = {
        "approved": True,
        "confidence_threshold": 0.5,
        "epsilon_sem": 0.1,
        "epsilon_parts": {"left": 0.1, "right": 0.1, "face": 0.1},
    }
    z, log = guarded_correction(
        original,
        torch.ones_like(original),
        lambda x: x,
        lambda x: x,
        features,
        torch.tensor([[1.0, 0.0]]),
        calibration,
    )
    assert log["accepted"] > 0 and log["delta_geometry"] < 0
    unchanged, log = guarded_correction(
        original,
        torch.ones_like(original),
        lambda x: x,
        lambda x: x,
        features,
        torch.tensor([[1.0, 0.0]]),
        {"approved": False},
    )
    assert torch.equal(unchanged, original) and log["accepted"] == 0


def test_invalid_candidate_geometry_is_rejected():
    original = torch.zeros(1, 4, 1, 2)

    def relation(x):
        if x.abs().max() > 0.001:
            raise ValueError("invalid candidate geometry")
        return x

    calibration = {
        "approved": True,
        "confidence_threshold": 0.5,
        "epsilon_sem": 0.1,
        "epsilon_parts": {"left": 0.1, "right": 0.1, "face": 0.1},
    }
    output, log = guarded_correction(
        original,
        torch.ones_like(original),
        lambda x: x,
        relation,
        features,
        torch.tensor([[1.0, 0.0]]),
        calibration,
    )
    assert torch.equal(output, original) and log["accepted"] == 0
    assert len(log["rejections"]) == 3


def test_calibration_requires_hashes_counts_and_dev_annotations(tmp_path):
    import json
    import pytest
    from cprflow.sampling.guard import load_calibration

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"approved": False}))
    with pytest.raises(ValueError, match="not approved"):
        load_calibration(path, {"teacher_sha256": "current"})
    path.write_text(json.dumps({"approved": True, "teacher_sha256": "old"}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_calibration(path, {"teacher_sha256": "current"})


def test_unrestricted_control_needs_no_semantic_approval_or_fake_score():
    original = torch.zeros(1, 4, 1, 2)
    output, log = guarded_correction(
        original,
        torch.ones_like(original),
        lambda x: x,
        lambda x: x,
        None,
        None,
        None,
        mode="unrestricted",
    )
    assert log["accepted"] > 0 and not torch.equal(output, original)
    assert log["initial_semantic"] is None and log["delta_semantic"] is None


def test_feature_only_and_semantic_controls_are_distinct():
    def crafted_features(x):
        feature = features(x)
        feature["parts"] = torch.tensor([[[1.0, 0.0]]]).repeat(1, 4, 1)
        return feature

    original = torch.zeros(1, 4, 1, 2)
    calibration = {
        "approved": True,
        "confidence_threshold": 0.5,
        "epsilon_sem": 0.0,
        "epsilon_parts": {"left": 0.01, "right": 0.01, "face": 0.01},
    }
    args = (
        original,
        torch.ones_like(original),
        lambda x: x,
        lambda x: x,
        crafted_features,
        torch.tensor([[1.0, 0.0]]),
        calibration,
    )
    feature_only, feature_log = guarded_correction(*args, mode="feature_only")
    semantic, semantic_log = guarded_correction(*args, mode="semantic")
    assert feature_log["accepted"] > 0 and semantic_log["accepted"] == 0
    assert not torch.equal(feature_only, original) and torch.equal(semantic, original)
