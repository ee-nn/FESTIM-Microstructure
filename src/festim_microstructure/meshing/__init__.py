"""Tessellation meshing, boundary-network reconstruction, and diagnostics."""

from typing import TYPE_CHECKING

from .._lazy import lazy_namespace

if TYPE_CHECKING:
    from . import diagnostics as diagnostics
    from . import ebsd as ebsd
    from . import neper as neper
    from .ebsd import EbsdMicrostructure as EbsdMicrostructure
    from .ebsd import EbsdOptions as EbsdOptions
    from .ebsd import run_ebsd_pipeline as run_ebsd_pipeline
    from .neper import NeperMesh as NeperMesh
    from .neper import NeperSettings as NeperSettings
    from .neper import TesrMeshOptions as TesrMeshOptions
    from .neper import mesh_tesr as mesh_tesr

_SUBMODULES = ("neper", "ebsd", "diagnostics")
_NAMES = {
    "NeperMesh": ".neper",
    "NeperSettings": ".neper",
    "TesrMeshOptions": ".neper",
    "mesh_tesr": ".neper",
    "EbsdMicrostructure": ".ebsd",
    "EbsdOptions": ".ebsd",
    "run_ebsd_pipeline": ".ebsd",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
