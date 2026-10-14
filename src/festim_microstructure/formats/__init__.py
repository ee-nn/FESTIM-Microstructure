"""File readers and writers, independent of conversion and modelling."""

from typing import TYPE_CHECKING

from .._lazy import lazy_namespace

if TYPE_CHECKING:
    from . import ctf as ctf
    from . import ebsd as ebsd
    from . import msh4 as msh4
    from . import provenance as provenance
    from .ctf import CtfMap as CtfMap
    from .ebsd import EbsdData as EbsdData
    from .ebsd import read_ebsd as read_ebsd
    from .ebsd import write_ebsd as write_ebsd
    from .msh4 import read_mesh as read_mesh
    from .msh4 import read_msh4 as read_msh4

_SUBMODULES = ("ctf", "ebsd", "msh4", "provenance")
_NAMES = {
    "EbsdData": ".ebsd",
    "read_ebsd": ".ebsd",
    "write_ebsd": ".ebsd",
    "CtfMap": ".ctf",
    "read_mesh": ".msh4",
    "read_msh4": ".msh4",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
