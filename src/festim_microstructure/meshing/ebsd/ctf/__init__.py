"""Convert Channel ``.ctf`` EBSD maps to Neper ``.tesr`` raster tessellations.

The converter segments cubic-orientation pixels, writes provenance and optional
quality diagnostics, and preserves the orientation convention Neper expects.

This was one module; it is now split by stage, and the names it exported are
re-exported here, so ``from festim_microstructure.meshing.ebsd.ctf import X``
keeps working:

``settings``
    :class:`Settings` and the Laue-group table
``reader``
    the ``.ctf`` parser and the quality mask
``segmentation``
    flood-fill into grains, prune, average each grain's orientation
``morphology``
    hole filling and the two raster topologies ``neper -M`` rejects
``tesr`` / ``provenance``
    the two files a conversion writes
``diagnostics``
    check images and the read-back verification (matplotlib / Neper / POV-Ray)
``convert``
    :class:`CtfConversion`, which runs the stages, and :func:`convert`
"""

from .convert import (
    ConversionResult,
    CtfConversion,
    MeasureOptions,
    convert,
    measure_tesr_against_ctf,
)
from .diagnostics import (
    QualityPanels,
    render_checks,
    verify_readback,
    write_quality_png,
)
from .morphology import STRUCT4, fill_holes, make_meshable
from .provenance import settings_from_provenance, write_provenance
from .reader import CtfMap, build_grid, crop_grid
from .segmentation import grain_mean_orientations, relabel_and_prune, segment_grains
from .settings import CUBIC_LAUE, LAUE_TO_CRYSYM, Settings
from .tesr import TesrData, write_tesr

__all__ = [
    "CUBIC_LAUE",
    "LAUE_TO_CRYSYM",
    "STRUCT4",
    "ConversionResult",
    "CtfConversion",
    "CtfMap",
    "MeasureOptions",
    "QualityPanels",
    "Settings",
    "TesrData",
    "build_grid",
    "convert",
    "crop_grid",
    "fill_holes",
    "grain_mean_orientations",
    "make_meshable",
    "measure_tesr_against_ctf",
    "relabel_and_prune",
    "render_checks",
    "segment_grains",
    "settings_from_provenance",
    "verify_readback",
    "write_provenance",
    "write_quality_png",
    "write_tesr",
]
