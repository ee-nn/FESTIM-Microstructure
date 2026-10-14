"""Microstructures for FESTIM: meshes, tagged subdomains, coefficient fields.

The package creates and interprets polycrystals -- Voronoi and EBSD --
converts them into FESTIM subdomains and diffusivity fields, and supplies the
post-processing a grain-boundary study needs. The transport model itself is
declared with FESTIM: species, exchange terms, boundary conditions, settings
and solving are ``festim.HydrogenTransportProblemDiscontinuous``'s, exactly as
in its manifold documentation. See ``examples/`` for the pattern.

The public API uses ordinary imports and requires the FESTIM/DOLFINx stack.
"""

from . import (
    check,
    ebsd,
    exports,
    fem,
    formats,
    materials,
    meshing,
    microstructure,
    plotting,
    voronoi,
)
from .fem.subdomains import Grain, GrainBoundaryNetwork, GrainSurface
from .meshing.ebsd import EbsdMicrostructure, EbsdOptions
from .meshing.upxo import UpxoMeshOptions
from .microstructure import (
    BoundaryNetwork,
    MeshedMicrostructure,
    Microstructure,
    TaggedPolycrystal,
)
from .voronoi import MeshSizing, VoronoiMicrostructure

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"

__all__ = [
    "BoundaryNetwork",
    "EbsdMicrostructure",
    "EbsdOptions",
    "Grain",
    "GrainBoundaryNetwork",
    "GrainSurface",
    "MeshSizing",
    "MeshedMicrostructure",
    "Microstructure",
    "TaggedPolycrystal",
    "UpxoMeshOptions",
    "VoronoiMicrostructure",
    "__version__",
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
]
