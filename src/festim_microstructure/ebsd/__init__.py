"""Measured orientation maps to raster tessellations."""

from typing import TYPE_CHECKING

from .._lazy import lazy_namespace

if TYPE_CHECKING:
    from . import convert as convert
    from . import diagnostics as diagnostics
    from . import morphology as morphology
    from . import orientation as orientation
    from . import segmentation as segmentation
    from . import settings as settings
    from .convert import ConversionResult as ConversionResult
    from .convert import CtfConversion as CtfConversion
    from .convert import MeasureOptions as MeasureOptions
    from .convert import measure_tesr_against_ctf as measure_tesr_against_ctf
    from .settings import Settings as Settings

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
