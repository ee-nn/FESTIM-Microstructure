"""Reusable grain-boundary transport models and coefficient fields.

Study workflows and publication reproductions live in ``examples/``.
"""

from .._lazy import lazy_namespace

_SUBMODULES = ("fisher", "properties", "resolved")

_NAMES = {
    "ShortCircuitParams": ".fisher",
    "ShortCircuitProblem": ".fisher",
    "Physics": ".properties",
    "crystal_diffusivity_field": ".properties",
    "gb_diffusivity_field": ".properties",
    "MicroModel": ".resolved",
    "SolveOptions": ".resolved",
    "averages": ".resolved",
    "build": ".resolved",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
