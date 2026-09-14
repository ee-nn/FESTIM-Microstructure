"""Microstructure-resolved hydrogen transport tools for FESTIM.

The curated geometry and network API is imported eagerly. Solver-dependent
names resolve on first access, keeping standalone EBSD tools and ``fm-check``
usable without FEniCS.
"""

from ._lazy import lazy_namespace
from .meshing.neper import NeperMicrostructure as NeperMicrostructure
from .meshing.neper import NeperOptions as NeperOptions
from .meshing.neper import NeperRun as NeperRun
from .meshing.neper import TesrMeshOptions as TesrMeshOptions
from .microstructure import BoundaryNetwork as BoundaryNetwork
from .microstructure import MeshedMicrostructure as MeshedMicrostructure
from .microstructure import Microstructure as Microstructure
from .voronoi import MeshSizing as MeshSizing
from .voronoi import VoronoiMicrostructure as VoronoiMicrostructure
from .voronoi import VoronoiMicrostructure3D as VoronoiMicrostructure3D

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"

_SUBMODULES = (
    "check",
    "ebsd",
    "exports",
    "fem",
    "formats",
    "materials",
    "meshing",
    "microstructure",
    "models",
    "plotting",
    "voronoi",
)
_NAMES = {
    "EbsdMicrostructure": ".meshing.ebsd",
    "EbsdOptions": ".meshing.ebsd",
    "MicroModel": ".models.resolved",
    "Physics": ".materials",
    "ShortCircuitParams": ".models.fisher",
    "ShortCircuitProblem": ".models.fisher",
    "SolveOptions": ".models.resolved",
    "build": ".models.resolved",
    "Grain": ".fem.subdomains",
    "GrainBoundaryNetwork": ".fem.subdomains",
    "GrainSurface": ".fem.subdomains",
    "TaggedGrainBoundaryNetwork": ".fem.subdomains",
}
__getattr__, __dir__, __all__ = lazy_namespace(
    __name__,
    _SUBMODULES,
    _NAMES,
    eager=[
        "__version__",
        "BoundaryNetwork",
        "MeshedMicrostructure",
        "Microstructure",
        "NeperMicrostructure",
        "NeperOptions",
        "NeperRun",
        "TesrMeshOptions",
        "MeshSizing",
        "VoronoiMicrostructure",
        "VoronoiMicrostructure3D",
    ],
)
