"""Reusable grain-boundary transport models."""

from .._lazy import lazy_namespace

_SUBMODULES = ("fisher", "resolved")
_NAMES = {
    "ShortCircuitParams": ".fisher",
    "ShortCircuitProblem": ".fisher",
    "MicroModel": ".resolved",
    "SolveOptions": ".resolved",
    "build": ".resolved",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
