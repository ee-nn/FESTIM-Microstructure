"""FESTIM/DOLFINx glue: subdomains for grains and networks, and a solver workaround.

Everything here imports ``festim`` and ``dolfinx`` at module level, which is
what separates it from :mod:`~festim_microstructure.meshing`: a microstructure
can be built and inspected without FEniCS, but a subdomain cannot exist without
it. Keeping the two apart is what lets the top-level package stay importable on
a machine that has neither.
"""

from typing import TYPE_CHECKING

from .._lazy import lazy_namespace

if TYPE_CHECKING:
    from . import solvers as solvers
    from . import subdomains as subdomains
    from .solvers import tune_direct_solver as tune_direct_solver
    from .subdomains import Grain as Grain
    from .subdomains import GrainBoundaryNetwork as GrainBoundaryNetwork
    from .subdomains import GrainSurface as GrainSurface
    from .subdomains import facet_midpoints as facet_midpoints
    from .subdomains import grain_boundary_network as grain_boundary_network
    from .subdomains import grain_subdomains as grain_subdomains
    from .subdomains import grain_surfaces as grain_surfaces
    from .subdomains import interior_facet_mask as interior_facet_mask

_SUBMODULES = ("solvers", "subdomains")

_NAMES = {
    "tune_direct_solver": ".solvers",
    "Grain": ".subdomains",
    "GrainBoundaryNetwork": ".subdomains",
    "GrainSurface": ".subdomains",
    "facet_midpoints": ".subdomains",
    "grain_boundary_network": ".subdomains",
    "grain_subdomains": ".subdomains",
    "grain_surfaces": ".subdomains",
    "interior_facet_mask": ".subdomains",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
