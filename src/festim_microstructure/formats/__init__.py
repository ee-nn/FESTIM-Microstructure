"""File readers and writers, independent of conversion and modelling."""

from .._lazy import lazy_namespace

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
