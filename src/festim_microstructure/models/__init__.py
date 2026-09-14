"""Reusable grain-boundary transport models."""

from typing import TYPE_CHECKING

from .._lazy import lazy_namespace

if TYPE_CHECKING:
    from . import fisher as fisher
    from . import resolved as resolved
    from .fisher import ShortCircuitParams as ShortCircuitParams
    from .fisher import ShortCircuitProblem as ShortCircuitProblem
    from .resolved import MicroModel as MicroModel
    from .resolved import SolveOptions as SolveOptions
    from .resolved import build as build

_SUBMODULES = ("fisher", "resolved")
_NAMES = {
    "ShortCircuitParams": ".fisher",
    "ShortCircuitProblem": ".fisher",
    "MicroModel": ".resolved",
    "SolveOptions": ".resolved",
    "build": ".resolved",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
