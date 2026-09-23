"""Microstructures for FESTIM: meshes, tagged subdomains, coefficient fields.

The package creates and interprets polycrystals -- Voronoi, Neper, EBSD --
converts them into FESTIM subdomains and diffusivity fields, and supplies the
post-processing a grain-boundary study needs. The transport model itself is
declared with FESTIM: species, exchange terms, boundary conditions, settings
and solving are ``festim.HydrogenTransportProblemDiscontinuous``'s, exactly as
in its manifold documentation. See ``examples/`` for the pattern.

The curated geometry and network API is imported eagerly. Solver-dependent
names resolve on first access, keeping standalone EBSD tools and ``fm-check``
usable without FEniCS.
"""

from typing import TYPE_CHECKING

from ._lazy import lazy_namespace
from .meshing.neper import NeperMesh as NeperMesh
from .meshing.neper import NeperSettings as NeperSettings
from .meshing.neper import TesrMeshOptions as TesrMeshOptions
from .microstructure import BoundaryNetwork as BoundaryNetwork
from .microstructure import MeshedMicrostructure as MeshedMicrostructure
from .microstructure import Microstructure as Microstructure
from .microstructure import TaggedPolycrystal as TaggedPolycrystal
from .voronoi import MeshSizing as MeshSizing
from .voronoi import VoronoiMicrostructure as VoronoiMicrostructure

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
    from . import voronoi as voronoi
    from .fem.subdomains import Grain as Grain
    from .fem.subdomains import GrainBoundaryNetwork as GrainBoundaryNetwork
    from .fem.subdomains import GrainSurface as GrainSurface
    from .meshing.ebsd import EbsdMicrostructure as EbsdMicrostructure
    from .meshing.ebsd import EbsdOptions as EbsdOptions

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
    "voronoi",
)
_NAMES = {
    "EbsdMicrostructure": ".meshing.ebsd",
    "EbsdOptions": ".meshing.ebsd",
    "Grain": ".fem.subdomains",
    "GrainBoundaryNetwork": ".fem.subdomains",
    "GrainSurface": ".fem.subdomains",
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
        "NeperMesh",
        "NeperSettings",
        "TesrMeshOptions",
        "MeshSizing",
        "VoronoiMicrostructure",
    ],
)
