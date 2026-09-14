"""Microstructure generation and import: Voronoi (Gmsh), Neper, EBSD.

``voronoi`` and ``neper`` are importable without FEniCS; ``ebsd.pipeline`` is
not. Submodules resolve on attribute access, so ``import
festim_microstructure.meshing`` followed by ``meshing.voronoi`` works and costs
only what is actually touched.
"""

from .._lazy import lazy_namespace

_SUBMODULES = ("ebsd", "neper", "voronoi")

_NAMES = {
    "MeshSizing": ".voronoi",
    "VoronoiMicrostructure": ".voronoi",
    "VoronoiMicrostructure3D": ".voronoi",
    "NeperMicrostructure": ".neper",
    "NeperOptions": ".neper",
    "NeperRun": ".neper",
    "TesrMeshOptions": ".neper",
    "mesh_tesr": ".neper",
    "read_mesh": ".neper",
    "run_neper": ".neper",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
