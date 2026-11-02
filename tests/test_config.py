from slp.config import compose, deep_merge, parse_overrides


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}}
    out = deep_merge(a, {"x": {"y": 9}})
    assert out == {"x": {"y": 9, "z": 2}}
    assert a == {"x": {"y": 1, "z": 2}}


def test_parse_overrides_types():
    o = parse_overrides(["method.lr=0.01", "eval.metrics=[accuracy]", "a.b.c=true"])
    assert o["method"]["lr"] == 0.01
    assert o["eval"]["metrics"] == ["accuracy"]
    assert o["a"]["b"]["c"] is True


def test_compose_precedence():
    cfg = compose("baseline_majority", "toy", ["data.n_samples=99", "method.foo=1"])
    assert cfg["method_name"] == "baseline_majority"
    assert cfg["data"]["name"] == "toy"
    assert cfg["data"]["n_samples"] == 99        # CLI beats data config
    assert cfg["method"]["foo"] == 1
    assert "accuracy" in cfg["eval"]["metrics"]  # inherited from base
