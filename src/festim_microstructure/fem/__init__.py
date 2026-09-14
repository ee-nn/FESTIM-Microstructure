"""FESTIM/DOLFINx glue: subdomains for grains and networks, and solver settings.

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
    from .solvers import ATOL as ATOL
    from .solvers import DIRECT_SOLVER_OPTIONS as DIRECT_SOLVER_OPTIONS
    from .solvers import tune_direct_solver as tune_direct_solver
    from .subdomains import Grain as Grain
    from .subdomains import GrainBoundaryNetwork as GrainBoundaryNetwork
    from .subdomains import GrainSurface as GrainSurface
    from .subdomains import TaggedGrainBoundaryNetwork as TaggedGrainBoundaryNetwork
    from .subdomains import (
        check_network_covers_grain_boundaries as check_network_covers_grain_boundaries,
    )
    from .subdomains import facet_midpoints as facet_midpoints
    from .subdomains import interior_facet_mask as interior_facet_mask

_SUBMODULES = ("solvers", "subdomains")

_NAMES = {
    "ATOL": ".solvers",
    "DIRECT_SOLVER_OPTIONS": ".solvers",
    "tune_direct_solver": ".solvers",
    "Grain": ".subdomains",
    "GrainBoundaryNetwork": ".subdomains",
    "GrainSurface": ".subdomains",
    "TaggedGrainBoundaryNetwork": ".subdomains",
    "check_network_covers_grain_boundaries": ".subdomains",
    "facet_midpoints": ".subdomains",
    "interior_facet_mask": ".subdomains",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
