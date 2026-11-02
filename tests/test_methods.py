"""Conformance checks applied automatically to EVERY folder in methods/."""

import re

import pytest

from slp.method import BaseMethod
from slp.paths import METHODS_DIR
from slp.registry import get_method, list_methods

METHODS = list_methods()


@pytest.mark.parametrize("name", METHODS)
def test_method_implements_interface(name):
    cls = get_method(name)
    assert issubclass(cls, BaseMethod)
    assert (METHODS_DIR / name / "config.yaml").exists(), "each method needs config.yaml"
    assert (METHODS_DIR / name / "README.md").exists(), "each method needs README.md"


@pytest.mark.parametrize("name", METHODS)
def test_no_cross_method_imports(name):
    pat = re.compile(r"^\s*(from|import)\s+methods\.(\w+)", re.MULTILINE)
    for py in (METHODS_DIR / name).rglob("*.py"):
        for _, other in pat.findall(py.read_text()):
            assert other == name, (
                f"{py} imports methods.{other}; move shared code into src/slp/ instead"
            )
