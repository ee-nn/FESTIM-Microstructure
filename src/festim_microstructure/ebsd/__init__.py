"""Measured orientation maps to raster tessellations."""

from .._lazy import lazy_namespace

_SUBMODULES = (
    "orientation",
    "segmentation",
    "morphology",
    "settings",
    "convert",
    "diagnostics",
)
_NAMES = {
    "ConversionResult": ".convert",
    "CtfConversion": ".convert",
    "MeasureOptions": ".convert",
    "measure_tesr_against_ctf": ".convert",
    "Settings": ".settings",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
