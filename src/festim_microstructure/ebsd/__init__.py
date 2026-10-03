"""Measured orientation maps imported through UPXO/DefDAP."""

from typing import TYPE_CHECKING

from .._lazy import lazy_namespace

if TYPE_CHECKING:
    from . import convert as convert
    from . import diagnostics as diagnostics
    from . import orientation as orientation
    from . import settings as settings
    from .convert import ConversionResult as ConversionResult
    from .convert import CtfConversion as CtfConversion
    from .convert import MeasureOptions as MeasureOptions
    from .convert import measure_against_ctf as measure_against_ctf
    from .settings import Settings as Settings

_SUBMODULES = (
    "orientation",
    "settings",
    "convert",
    "diagnostics",
)
_NAMES = {
    "ConversionResult": ".convert",
    "CtfConversion": ".convert",
    "MeasureOptions": ".convert",
    "measure_against_ctf": ".convert",
    "Settings": ".settings",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
