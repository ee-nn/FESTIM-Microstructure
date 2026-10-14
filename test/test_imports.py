"""Every submodule imports with the solver stack and console targets exist."""

import importlib
import pkgutil

import pytest

import festim_microstructure


def _all_modules():
    """List every importable submodule in the package."""
    pkg = festim_microstructure
    return sorted(
        m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")
    )


@pytest.mark.parametrize("name", _all_modules())
def test_submodule_imports(name):
    """Verify each advertised submodule can be imported."""
    importlib.import_module(name)


@pytest.mark.parametrize(
    "target",
    [
        "festim_microstructure.check:main",
    ],
)
def test_console_script_targets_exist(target):
    """Verify each declared console entry point resolves to a callable."""
    module, attr = target.split(":")
    assert callable(getattr(importlib.import_module(module), attr))
