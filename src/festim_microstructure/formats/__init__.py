"""File readers and writers, independent of conversion and modelling."""

from typing import TYPE_CHECKING

from .._lazy import lazy_namespace

if TYPE_CHECKING:
    from . import ctf as ctf
    from . import msh4 as msh4
    from . import provenance as provenance
    from . import tesr as tesr
    from .ctf import CtfMap as CtfMap
    from .msh4 import StatFile as StatFile
    from .msh4 import read_mesh as read_mesh
    from .msh4 import read_msh4 as read_msh4
    from .tesr import TesrData as TesrData
    from .tesr import read_tesr as read_tesr
    from .tesr import read_tesr_full as read_tesr_full
    from .tesr import write_tesr as write_tesr

_SUBMODULES = ("ctf", "tesr", "msh4", "provenance")
_NAMES = {
    "TesrData": ".tesr",
    "read_tesr": ".tesr",
    "read_tesr_full": ".tesr",
    "write_tesr": ".tesr",
    "CtfMap": ".ctf",
    "StatFile": ".msh4",
    "read_mesh": ".msh4",
    "read_msh4": ".msh4",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
