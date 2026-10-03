"""Quality, crop, frame and UPXO/DefDAP import settings."""

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
    #: Crop the measured map before segmentation.
    crop: str | None = None
    phase: int = 1  #: phase to keep
    threshold: float = 10.0  #: grain boundary misorientation, degrees
    max_mad: float = 1.0  #: MAD cutoff
    min_bands: int = 0  #: minimum Bands; 0 disables the test
    allow_error: bool = False  #: keep points whose Error column is non-zero
    min_pixels: int = 5  #: discard grains smaller than this
    #: Multiply both source pixel spacings by this factor.
    scale: float = 1.0
    #: Proper frame change: mirror rows and rotate orientations 180 deg about x.
    flip_y: bool = False
    #: Rotation taking the Euler frame onto the map frame (Bunge degrees).
    euler_correction: tuple[float, float, float] | None = None
    #: Assign unindexed/pruned pixels to the nearest surviving grain for geometry.
    #: Original membership is retained and these pixels do not affect grain means.
    fill: bool = True
    diagnostics: bool = False
    #: Optional UPXO interpreter; defaults to the combined active environment.
    python: str | None = None
    timeout: float = 600.0

    def __post_init__(self):
        if not isinstance(self.phase, int) or self.phase < 1:
            raise ValueError("phase must be a positive integer")
        if not isinstance(self.min_pixels, int) or self.min_pixels < 1:
            raise ValueError("min_pixels must be a positive integer")
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 63:
            raise ValueError("threshold must be between 0 and 63 degrees")
        if not np.isfinite(self.max_mad) or self.max_mad < 0 or self.min_bands < 0:
            raise ValueError("quality cutoffs must be finite and nonnegative")
        if (
            not np.isfinite(self.scale)
            or self.scale <= 0
            or not np.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("scale and timeout must be finite and positive")
        self.euler_correction_quat

    @property
    def euler_correction_quat(self):
        """``euler_correction`` as a unit quaternion, or ``None``."""
        if self.euler_correction is None:
            return None
        angles = np.asarray(self.euler_correction, dtype=float)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError(
                "euler_correction must be three finite Bunge angles in degrees"
            )
        return euler_bunge_to_quat(*angles)

    @property
    def unit(self):
        """Return the display unit implied by the spatial scale."""
        return "um" if np.isclose(self.scale, 1.0) else f"x{self.scale:g} um"
