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
    """Stand in for an unavailable solver or meshing module."""

    def __getattr__(self, name):
        """Return a stub for an unavailable heavy dependency."""
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {"__init__": lambda self, *a, **k: None})
        return cls if name[:1].isupper() else mock.MagicMock(name=name)


@pytest.fixture(scope="module")
def stubbed_heavy_deps():
    """Temporarily stub unavailable solver imports for module-level tests."""
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


@pytest.fixture
def ctf_writer():
    """Create complete Oxford text maps with optional quality/phase fields."""

    def write(path, euler, spacing=(1.0, 1.0), mad=None, phases=None, reverse=False):
        import numpy as np

        euler = np.asarray(euler)
        ny, nx, _ = euler.shape
        phases = np.ones((ny, nx), dtype=int) if phases is None else np.asarray(phases)
        mad = np.full((ny, nx), 0.1) if mad is None else np.asarray(mad)
        header = (
            f"Channel Text File\nXCells\t{nx}\nYCells\t{ny}\n"
            f"XStep\t{spacing[0]}\nYStep\t{spacing[1]}\n"
            "Phases\t2\n1;1;1\t90;90;90\tCubic\t11\t229\n1;1;1\t90;90;90\tOtherCubic\t11\t229\n"
            "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n"
        )
        rows = []
        for y in range(ny):
            for x in range(nx):
                values = [
                    phases[y, x],
                    x * spacing[0],
                    y * spacing[1],
                    8,
                    0,
                    *euler[y, x],
                    mad[y, x],
                    100,
                    100,
                ]
                rows.append("\t".join(str(v) for v in values) + "\n")
        path.write_text(header + "".join(reversed(rows) if reverse else rows))
        return path

    return write
