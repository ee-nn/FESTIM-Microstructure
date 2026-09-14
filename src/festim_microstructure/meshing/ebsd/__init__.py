"""EBSD orientation maps -> Neper raster tessellations -> conforming meshes.

``ctf``, ``orientation``, ``segmentation_error``, ``grain_area_change``,
``mesh_overlay`` and ``micrograph`` are pure NumPy/matplotlib; ``pipeline``
needs dolfinx and Neper, and is imported only if something asks for it.
"""

from ..._lazy import lazy_namespace

_SUBMODULES = (
    "ctf",
    "grain_area_change",
    "mesh_overlay",
    "micrograph",
    "orientation",
    "pipeline",
    "segmentation_error",
)

_NAMES = {
    "ConversionResult": ".ctf",
    "CtfConversion": ".ctf",
    "convert": ".ctf",
    "measure_tesr_against_ctf": ".ctf",
    "EbsdMicrostructure": ".pipeline",
    "EbsdOptions": ".pipeline",
    "run_ebsd_pipeline": ".pipeline",
}

__getattr__, __dir__, __all__ = lazy_namespace(__name__, _SUBMODULES, _NAMES)
