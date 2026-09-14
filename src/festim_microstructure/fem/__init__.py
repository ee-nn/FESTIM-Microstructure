"""FESTIM/DOLFINx glue: subdomains for grains and networks, and solver settings.

Everything here imports ``festim`` and ``dolfinx`` at module level, which is
what separates it from :mod:`~festim_microstructure.meshing`: a microstructure
can be built and inspected without FEniCS, but a subdomain cannot exist without
it. Keeping the two apart is what lets the top-level package stay importable on
a machine that has neither.
"""

from .._lazy import lazy_namespace

_SUBMODULES = ("solvers", "subdomains")

_NAMES = {
    "ATOL": ".solvers",
    "DIRECT_SOLVER_OPTIONS": ".solvers",
    "tune_direct_solver": ".solvers",
    "Grain": ".subdomains",
    "GrainBoundaryNetwork": ".subdomains",
    "GrainSurface": ".subdomains",
    "TaggedGrainBoundaryNetwork": ".subdomains",
    "facet_midpoints": ".subdomains",
    "interior_facet_mask": ".subdomains",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
