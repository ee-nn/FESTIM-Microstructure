"""Tessellation meshing, boundary-network reconstruction, and diagnostics."""

from .._lazy import lazy_namespace

_SUBMODULES = ("neper", "ebsd", "diagnostics")
_NAMES = {
    "NeperMicrostructure": ".neper",
    "NeperOptions": ".neper",
    "NeperRun": ".neper",
    "TesrMeshOptions": ".neper",
    "mesh_tesr": ".neper",
    "run_neper": ".neper",
    "EbsdMicrostructure": ".ebsd",
    "EbsdOptions": ".ebsd",
    "run_ebsd_pipeline": ".ebsd",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
