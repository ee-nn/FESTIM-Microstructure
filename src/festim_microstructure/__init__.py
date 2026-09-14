"""Microstructure-resolved hydrogen transport tools for FESTIM.

Used alongside FESTIM itself::

    import festim as F
    import festim_microstructure as fm

    micro = fm.VoronoiMicrostructure.create(size=20e-6, n_seeds=64)
    model = fm.build(micro, fm.Physics(T=600.0), bcs).run()

The names below are the curated surface; everything else stays reachable
through the subpackages (``fm.meshing``, ``fm.models``, ``fm.fem``,
``fm.postprocessing``, ``fm.microstructure``).

Attribute access is lazy (:mod:`._lazy`). ``import festim_microstructure``
pulls in nothing heavier than the standard library, so the pure-Python half of
the package -- the EBSD converter, the Voronoi geometry, the orientation maths
-- is usable without a DOLFINx install. FEniCS is imported only when a name
that needs it is first touched.
"""

from ._lazy import lazy_namespace

try:
    from ._version import __version__
except ImportError:  # not installed (running from a checkout without pip -e)
    __version__ = "0+unknown"

#: Subpackages reachable as attributes of the top-level package.
_SUBMODULES = (
    "check",
    "fem",
    "meshing",
    "microstructure",
    "models",
    "postprocessing",
)

#: The curated top-level API: name -> the module that defines it.
_NAMES = {
    # The contract every microstructure builder implements.
    "BoundaryNetwork": ".microstructure",
    "MeshedMicrostructure": ".microstructure",
    "Microstructure": ".microstructure",
    # Microstructure builders.
    "EbsdMicrostructure": ".meshing.ebsd.pipeline",
    "EbsdOptions": ".meshing.ebsd.pipeline",
    "MeshSizing": ".meshing.voronoi",
    "NeperMicrostructure": ".meshing.neper",
    "NeperOptions": ".meshing.neper",
    "NeperRun": ".meshing.neper",
    "TesrMeshOptions": ".meshing.neper",
    "VoronoiMicrostructure": ".meshing.voronoi",
    "VoronoiMicrostructure3D": ".meshing.voronoi",
    # Transport models.
    "MicroModel": ".models.resolved",
    "Physics": ".models.properties",
    "ShortCircuitParams": ".models.fisher",
    "ShortCircuitProblem": ".models.fisher",
    "SolveOptions": ".models.resolved",
    "build": ".models.resolved",
    # FESTIM subdomains for grains and grain-boundary networks.
    "Grain": ".fem.subdomains",
    "GrainBoundaryNetwork": ".fem.subdomains",
    "GrainSurface": ".fem.subdomains",
    "TaggedGrainBoundaryNetwork": ".fem.subdomains",
}

__getattr__, __dir__, __all__ = lazy_namespace(
    __name__, _SUBMODULES, _NAMES, eager=["__version__"]
)
