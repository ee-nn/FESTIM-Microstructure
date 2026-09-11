"""Every submodule imports, and the console-script targets exist.

When dolfinx / festim are absent they are stubbed, so that a wrong relative
import or a missing name is still caught without a FEniCS install.
"""

import importlib
import pkgutil
import sys
import types
from unittest import mock

import pytest
from conftest import HAVE_FENICS

import festim_microstructure

HEAVY = (
    "dolfinx",
    "dolfinx.io",
    "dolfinx.io.gmsh",
    "dolfinx.mesh",
    "dolfinx.fem",
    "dolfinx.geometry",
    "festim",
    "ufl",
    "gmsh",
    "mpi4py",
    "mpi4py.MPI",
    "petsc4py",
    "petsc4py.PETSc",
)


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {"__init__": lambda self, *a, **k: None})
        return cls if name[:1].isupper() else mock.MagicMock(name=name)


@pytest.fixture(scope="module")
def stubbed_heavy_deps():
    if HAVE_FENICS:
        yield
        return
    added = []
    for name in HEAVY:
        if name not in sys.modules:
            sys.modules[name] = _Stub(name)
            added.append(name)
    sys.modules["mpi4py"].MPI = sys.modules["mpi4py.MPI"]
    sys.modules["petsc4py"].PETSc = sys.modules["petsc4py.PETSc"]
    sys.modules["dolfinx"].io = sys.modules["dolfinx.io"]
    sys.modules["dolfinx.io"].gmsh = sys.modules["dolfinx.io.gmsh"]
    sys.modules["festim"].k_B = 8.617333262e-5
    yield
    for name in added:
        sys.modules.pop(name, None)


def _all_modules():
    pkg = festim_microstructure
    return sorted(
        m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")
    )


@pytest.mark.parametrize("name", _all_modules())
def test_submodule_imports(stubbed_heavy_deps, name):
    importlib.import_module(name)


@pytest.mark.parametrize(
    "target",
    [
        "festim_microstructure.check:main",
        "festim_microstructure.meshing.voronoi:main",
        "festim_microstructure.meshing.ebsd.pipeline:main",
    ],
)
def test_console_script_targets_exist(stubbed_heavy_deps, target):
    module, attr = target.split(":")
    assert callable(getattr(importlib.import_module(module), attr))


def test_pure_modules_need_no_fenics():
    """These must import with nothing but numpy/scipy/matplotlib/pillow."""
    for name in (
        "festim_microstructure",
        "festim_microstructure._binaries",
        "festim_microstructure.check",
        "festim_microstructure.meshing.voronoi",
        "festim_microstructure.meshing.neper",
        "festim_microstructure.meshing.ebsd.ctf",
        "festim_microstructure.meshing.ebsd.orientation",
        "festim_microstructure.meshing.ebsd.segmentation_error",
        "festim_microstructure.meshing.ebsd.grain_area_change",
        "festim_microstructure.meshing.ebsd.mesh_overlay",
        "festim_microstructure.meshing.ebsd.micrograph",
    ):
        importlib.import_module(name)
