"""FESTIM/DOLFINx glue: grain and network subdomains, and solver helpers."""

from . import solvers, subdomains
from .solvers import tune_direct_solver
from .subdomains import (
    Grain,
    GrainBoundaryNetwork,
    GrainSurface,
    facet_midpoints,
    grain_boundary_network,
    grain_subdomains,
    grain_surfaces,
    interior_facet_mask,
)

__all__ = [
    "Grain",
    "GrainBoundaryNetwork",
    "GrainSurface",
    "facet_midpoints",
    "grain_boundary_network",
    "grain_subdomains",
    "grain_surfaces",
    "interior_facet_mask",
    "solvers",
    "subdomains",
    "tune_direct_solver",
]
