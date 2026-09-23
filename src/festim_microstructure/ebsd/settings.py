"""Conversion settings, and the Channel-to-Neper crystal-symmetry table."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields

import numpy as np

from festim_microstructure.formats.provenance import read_provenance

__all__ = ["CUBIC_LAUE", "Settings", "settings_from_provenance"]


def settings_from_provenance(path):
    """Rebuild the `Settings` of a conversion from its provenance json."""
    rec = read_provenance(path)
    known = {f.name for f in dataclass_fields(Settings)}
    return Settings(**{k: v for k, v in rec.items() if k in known})


CUBIC_LAUE = (10, 11)


@dataclass
class Settings:
    """Configuration for :func:`convert`, including saved provenance."""

    ctf: str
    #: Crop before segmentation to avoid unpruned Neper slivers.
    crop: str | None = None
    phase: int = 1  #: phase to keep
    threshold: float = 10.0  #: grain boundary misorientation, degrees
    max_mad: float = 1.0  #: MAD cutoff
    min_bands: int = 0  #: minimum Bands; 0 disables the test
    allow_error: bool = False  #: keep points whose Error column is non-zero
    min_pixels: int = 5  #: discard grains smaller than this
    #: TESR length scale; keep it near unity for Neper's absolute tolerances.
    scale: float = 1.0
    #: Mirror y because EBSD and Neper use opposite vertical directions.
    flip_y: bool = False
    active: bool = False  #: write orientations active rather than passive
    fill: bool = True  #: grow cells into unassigned voxels; see fill_holes
    #: Repair raster topologies that ``neper -M`` cannot reconstruct.
    topology_fix: bool = True
    voxel_ori: bool = True  #: write **oridata/**oridef (large; needed by -V)
    #: Write quality and segmentation diagnostics.
    diagnostics: bool = False
    #: Optional Neper binary for rendered checks; absence is non-fatal.
    neper: str | None = "neper"
    povray: str = "povray"  #: neper -V renders through this

    @property
    def unit(self):
        """Return the display unit implied by the spatial scale."""
        return "um" if np.isclose(self.scale, 1.0) else f"x{self.scale:g} um"
