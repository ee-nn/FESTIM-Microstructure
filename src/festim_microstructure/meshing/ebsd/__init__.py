"""EBSD orientation maps -> Neper raster tessellations -> conforming meshes.

``ctf``, ``orientation``, ``segmentation_error``, ``grain_area_change``,
``mesh_overlay`` and ``micrograph`` are pure NumPy/matplotlib; ``pipeline``
needs dolfinx and Neper.
"""

__all__ = [
    "ctf",
    "grain_area_change",
    "mesh_overlay",
    "micrograph",
    "orientation",
    "pipeline",
    "segmentation_error",
]
