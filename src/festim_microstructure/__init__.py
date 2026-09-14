"""Microstructure-resolved hydrogen transport tools for FESTIM.

The curated geometry and network API is imported eagerly. Solver-dependent
names resolve on first access, keeping standalone EBSD tools and ``fm-check``
usable without FEniCS.
"""

from typing import TYPE_CHECKING

from ._lazy import lazy_namespace
from .meshing.neper import NeperMicrostructure as NeperMicrostructure
from .meshing.neper import NeperOptions as NeperOptions
from .meshing.neper import NeperRun as NeperRun
from .meshing.neper import TesrMeshOptions as TesrMeshOptions
from .microstructure import BoundaryNetwork as BoundaryNetwork
from .microstructure import MeshedMicrostructure as MeshedMicrostructure
from .microstructure import Microstructure as Microstructure
from .microstructure import TaggedPolycrystal as TaggedPolycrystal
from .voronoi import MeshSizing as MeshSizing
from .voronoi import VoronoiMicrostructure as VoronoiMicrostructure
from .voronoi import VoronoiMicrostructure3D as VoronoiMicrostructure3D

if TYPE_CHECKING:
    from . import check as check
    from . import ebsd as ebsd
    from . import exports as exports
    from . import fem as fem
    from . import formats as formats
    from . import materials as materials
    from . import meshing as meshing
    from . import microstructure as microstructure
    from . import plotting as plotting
    from . import resolved as resolved
    from . import voronoi as voronoi
    from .fem.subdomains import Grain as Grain
    from .fem.subdomains import GrainBoundaryNetwork as GrainBoundaryNetwork
    from .fem.subdomains import GrainSurface as GrainSurface
    from .fem.subdomains import TaggedGrainBoundaryNetwork as TaggedGrainBoundaryNetwork
    from .materials import Physics as Physics
    from .meshing.ebsd import EbsdMicrostructure as EbsdMicrostructure
    from .meshing.ebsd import EbsdOptions as EbsdOptions
    from .resolved import MicroModel as MicroModel
    from .resolved import SolveOptions as SolveOptions
    from .resolved import build as build

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
    "plotting",
    "resolved",
    "voronoi",
)
_NAMES = {
    "EbsdMicrostructure": ".meshing.ebsd",
    "EbsdOptions": ".meshing.ebsd",
    "MicroModel": ".resolved",
    "Physics": ".materials",
    "SolveOptions": ".resolved",
    "build": ".resolved",
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
        "TaggedPolycrystal",
        "NeperMicrostructure",
        "NeperOptions",
        "NeperRun",
        "TesrMeshOptions",
        "MeshSizing",
        "VoronoiMicrostructure",
        "VoronoiMicrostructure3D",
    ],
)
