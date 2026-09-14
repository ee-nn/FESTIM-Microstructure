import importlib.util
import sys
import types
from unittest import mock

import pytest

HAVE_FENICS = all(
    importlib.util.find_spec(m) is not None for m in ("dolfinx", "festim", "gmsh")
)

requires_fenics = pytest.mark.skipif(
    not HAVE_FENICS, reason="needs dolfinx, festim and gmsh"
)


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
