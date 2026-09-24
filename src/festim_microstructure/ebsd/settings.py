"""Conversion settings, and the Channel-to-Neper crystal-symmetry table."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields

import numpy as np

from festim_microstructure.ebsd.orientation import euler_bunge_to_quat
from festim_microstructure.formats.provenance import read_provenance

__all__ = ["CUBIC_LAUE", "Settings", "settings_from_provenance"]


def settings_from_provenance(path):
    """Rebuild the `Settings` of a conversion from its provenance json."""
    rec = read_provenance(path)
    if rec.get("active"):
        raise ValueError(
            f"{path} records active=True, an option that wrote inverse rotations "
            "under a 'rodrigues:passive' label and has been removed; re-run the "
            "conversion"
        )
    known = {f.name for f in dataclass_fields(Settings)}
    return Settings(**{k: v for k, v in rec.items() if k in known})


#: Laue 11 (m-3m) only: Laue 10 (m-3) has 12 proper rotations, not 24.
CUBIC_LAUE = (11,)


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
    #: Reverse the y axis because EBSD and Neper use opposite vertical
    #: directions. Written as a proper frame change, (x, y, z) -> (x, -y, -z):
    #: the rows are mirrored and every orientation is rotated 180 deg about x.
    flip_y: bool = False
    #: Rotation taking the Euler-angle frame onto the map frame, as Bunge
    #: angles in degrees, left-multiplied onto every measured orientation;
    #: equivalent to MTEX's ``'EulerCorrection', rotation.byEuler(...)``.
    #: ``None`` assumes the two frames coincide. MTEX assumes (180, 0, 0) for
    #: .ctf files; verify against a known specimen feature before relying on it.
    euler_correction: tuple[float, float, float] | None = None
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
    def euler_correction_quat(self):
        """``euler_correction`` as a unit quaternion, or ``None``."""
        if self.euler_correction is None:
            return None
        angles = np.asarray(self.euler_correction, dtype=float)
        if angles.shape != (3,):
            raise ValueError("euler_correction must be three Bunge angles in degrees")
        return euler_bunge_to_quat(*angles)

    @property
    def unit(self):
        """Return the display unit implied by the spatial scale."""
        return "um" if np.isclose(self.scale, 1.0) else f"x{self.scale:g} um"
