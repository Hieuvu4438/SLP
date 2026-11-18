import copy

import numpy as np
import pytest
import torch

from cprflow.config import DEFAULTS
from cprflow.data.cache import file_hash, save_numeric
from cprflow.data.geometry_cache import GeometryCache, build_geometry_cache
from cprflow.data.manifest import write_jsonl
from cprflow.types import PROFILE


@pytest.mark.requires_body_model
def test_offline_ground_truth_cache_matches_fk_and_invalidates_asset(
    tmp_path, body_model, identity_motion
):
    config = copy.deepcopy(DEFAULTS)
    config["run"]["synthetic"] = True
    config["paths"].update(
        cache=str(tmp_path / "cache"),
        manifests=str(tmp_path / "manifests"),
        smplx_model=str(body_model.asset_path),
    )
    x = identity_motion(4)
    path = tmp_path / "motion.npz"
    save_numeric(path, motion=x.numpy(), timestamps=np.arange(4) / 30)
    row = {
        "uid": "csl_daily:synthetic_geometry",
        "dataset_id": "csl_daily",
        "sign_language_id": 0,
        "profile": PROFILE,
        "split": "train",
        "frames": 4,
        "text": "synthetic fixture",
        "processed_path": str(path),
        "processed_sha256": file_hash(path),
        "synthetic": True,
    }
    write_jsonl(tmp_path / "manifests/train.jsonl", [row])
    result = build_geometry_cache(config, ["train"])
    assert result["counts"] == {"train": 1} and result["learned_statistics"] is False
    cached = GeometryCache(config).get(row)
    actual = body_model(x)
    for group in ("body", "left", "right", "left_local", "right_local"):
        torch.testing.assert_close(cached[group], actual[group])
        assert not cached[group].requires_grad
    selected = [actual["names"].index(name) for name in cached["names"]]
    torch.testing.assert_close(cached["joints"], actual["joints"][:, selected])
    assert len(cached["names"]) == 52
    from cprflow.geometry.relations import relation33

    torch.testing.assert_close(
        relation33(cached["joints"], cached["names"]), relation33(actual["joints"], actual["names"])
    )
    fake = tmp_path / "different_asset.npz"
    fake.write_bytes(b"synthetic changed asset fixture")
    config["paths"]["smplx_model"] = str(fake)
    with pytest.raises(ValueError, match="asset changed"):
        GeometryCache(config)
