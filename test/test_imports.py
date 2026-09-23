"""Every submodule imports, and the console-script targets exist.

When dolfinx / festim are absent they are stubbed, so that a wrong relative
import or a missing name is still caught without a FEniCS install.
"""

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
def test_submodule_imports(stubbed_heavy_deps, name):
    """Verify each advertised submodule can be imported."""
    importlib.import_module(name)


@pytest.mark.parametrize(
    "target",
    [
        "festim_microstructure.check:main",
    ],
)
def test_console_script_targets_exist(stubbed_heavy_deps, target):
    """Verify each declared console entry point resolves to a callable."""
    module, attr = target.split(":")
    assert callable(getattr(importlib.import_module(module), attr))


def test_pure_modules_need_no_fenics():
    """These must import with nothing but numpy/scipy/matplotlib/pillow."""
    for name in (
        "festim_microstructure",
        "festim_microstructure._binaries",
        "festim_microstructure.check",
        "festim_microstructure.voronoi",
        "festim_microstructure.meshing.neper",
        "festim_microstructure.ebsd",
        "festim_microstructure.ebsd.orientation",
        "festim_microstructure.ebsd.diagnostics",
        "festim_microstructure.meshing.diagnostics",
        "festim_microstructure.ebsd.convert",
        "festim_microstructure.formats.ctf",
        "festim_microstructure.formats.tesr",
        "festim_microstructure.formats.msh4",
        "festim_microstructure.formats.provenance",
        "festim_microstructure.plotting",
    ):
        importlib.import_module(name)
