"""signmflow_plus arm registry: every override names an existing config key; commands are well formed."""

import pytest
import yaml

from methods.signmflow_plus import experiments as X
from slp.paths import METHODS_DIR


def _has_key(cfg, dotted):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return True


@pytest.mark.parametrize("arm", X.ARMS, ids=lambda a: a.id)
def test_overrides_exist_in_the_method_config(arm):
    cfg = yaml.safe_load(open(METHODS_DIR / arm.method / "config.yaml"))
    for o in arm.overrides + (X.BY_ID[arm.from_arm].overrides if arm.from_arm else []):
        key = o.split("=", 1)[0]
        assert _has_key(cfg, key), f"{arm.id}: {key} is not a key of methods/{arm.method}/config.yaml"


def test_ids_unique_and_commands_use_fixed_lengths():
    assert len(X.BY_ID) == len(X.ARMS)
    for arm in X.ARMS:
        for cmd in X.commands(arm, "phoenix_smplx"):
            assert cmd.startswith("python scripts/run.py") and "--split" not in cmd  # val only
            if arm.method == "signmflow_plus":
                assert "method.sampler.lengths_file=" in cmd
            if arm.from_arm or arm.from_v2:
                assert "method.init.generator=" in cmd
