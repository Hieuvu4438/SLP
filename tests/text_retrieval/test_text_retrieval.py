"""text_retrieval baselines (docs/proposal7 E0.4)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from methods.text_retrieval.method import Method, resample


def _method(tmp_path, **hp):
    ctx = SimpleNamespace(seed=0, logger=SimpleNamespace(info=lambda *a: None))
    return Method({"method": hp}, ctx)


def _split():
    rng = np.random.default_rng(0)
    inputs = [{"name": f"train/{i}", "text": t} for i, t in enumerate(["guten morgen", "regen im norden",
                                                                      "sonne im sueden", "regen im norden"])]
    targets = [rng.normal(size=(10 + 5 * i, 133)).astype(np.float32) for i in range(4)]
    return SimpleNamespace(inputs=inputs, targets=targets)


def test_retrieval_returns_the_nearest_training_motion(tmp_path):
    tr = _split()
    m = _method(tmp_path, mode="retrieval")
    m.fit(tr, None)
    out = m.predict([{"name": "dev/0", "text": "sonne im sueden"}, {"name": "dev/1", "text": "regen im norden!"}])
    assert np.array_equal(out[0], tr.targets[2])
    assert any(np.array_equal(out[1], tr.targets[j]) for j in (1, 3))  # tie between identical sentences


def test_mean_pose_and_fixed_lengths(tmp_path):
    tr = _split()
    json.dump({"dev/0": 33, "dev__1": 7}, open(tmp_path / "l.json", "w"))
    m = _method(tmp_path, mode="mean_pose", lengths_file=str(tmp_path / "l.json"))
    m.fit(tr, None)
    out = m.predict([{"name": "dev/0", "text": "x"}, {"name": "dev/1", "text": "y"}])
    assert [o.shape for o in out] == [(33, 133), (7, 133)] and all((o == 0).all() for o in out)
    r = _method(tmp_path, mode="retrieval", lengths_file=str(tmp_path / "l.json"))
    r.fit(tr, None)
    assert r.predict([{"name": "dev/0", "text": "guten morgen"}])[0].shape == (33, 133)


def test_resample_endpoints_and_linear():
    x = np.arange(5, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    y = resample(x, 9)
    assert y.shape == (9, 3) and y[0, 0] == 0 and y[-1, 0] == 4 and np.allclose(np.diff(y[:, 0]), 0.5)
    assert resample(x, 5) is x


def test_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        _method(tmp_path, mode="nope").fit(_split(), None)
