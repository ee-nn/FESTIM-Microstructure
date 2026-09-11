"""Microstructure-resolved hydrogen transport tools for FESTIM.

``meshing`` builds/imports polycrystals, ``models`` provides reusable transport
models, and ``postprocessing`` provides numerical measures. Study workflows live
in ``examples/``. Heavy dependencies are imported only by modules that need them.
"""

try:
    from ._version import __version__
except ImportError:  # not installed (running from a checkout without pip -e)
    __version__ = "0+unknown"

__all__ = ["__version__"]
