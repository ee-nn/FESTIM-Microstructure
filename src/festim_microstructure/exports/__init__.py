"""Measures and post-processing for solved microstructure problems."""

from typing import TYPE_CHECKING

from .._lazy import lazy_namespace

if TYPE_CHECKING:
    from . import averages as averages
    from . import measures as measures
    from .measures import component_count as component_count
    from .measures import inventory as inventory
    from .measures import junction_only_below as junction_only_below
    from .measures import submesh_measure as submesh_measure

_SUBMODULES = ("measures", "averages")
_NAMES = {
    "component_count": ".measures",
    "inventory": ".measures",
    "junction_only_below": ".measures",
    "submesh_measure": ".measures",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
