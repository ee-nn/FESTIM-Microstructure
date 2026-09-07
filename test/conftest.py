import importlib.util

import pytest

HAVE_FENICS = all(
    importlib.util.find_spec(m) is not None for m in ("dolfinx", "festim", "gmsh")
)

requires_fenics = pytest.mark.skipif(
    not HAVE_FENICS, reason="needs dolfinx, festim and gmsh"
)
