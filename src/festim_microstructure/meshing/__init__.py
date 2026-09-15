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
    from .neper import NeperMicrostructure as NeperMicrostructure
    from .neper import NeperSettings as NeperSettings
    from .neper import NeperRun as NeperRun
    from .neper import TesrMeshOptions as TesrMeshOptions
    from .neper import mesh_tesr as mesh_tesr
    from .neper import run_neper as run_neper

_SUBMODULES = ("neper", "ebsd", "diagnostics")
_NAMES = {
    "NeperMicrostructure": ".neper",
    "NeperSettings": ".neper",
    "NeperRun": ".neper",
    "TesrMeshOptions": ".neper",
    "mesh_tesr": ".neper",
    "run_neper": ".neper",
    "EbsdMicrostructure": ".ebsd",
    "EbsdOptions": ".ebsd",
    "run_ebsd_pipeline": ".ebsd",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
