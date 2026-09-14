"""Reusable numerical measures for solved microstructure problems."""

from .._lazy import lazy_namespace

_SUBMODULES = ("measures",)

_NAMES = {
    "component_count": ".measures",
    "inventory": ".measures",
    "junction_only_below": ".measures",
    "submesh_measure": ".measures",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
