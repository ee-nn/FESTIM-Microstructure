"""Read a Channel ``.ctf`` and put its pixel table on the map grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..orientation import euler_bunge_to_quat
from .settings import LAUE_TO_CRYSYM

__all__ = ["CtfMap", "build_grid", "crop_grid"]


class CtfMap:
    """A parsed Channel Text File: header fields plus the pixel table."""

    def __init__(self, path):
        self.path = Path(path)
        self.header = {}
        self.phases = []
        self._parse()

    def _parse(self):
        with open(self.path, errors="replace") as fh:
            lines = fh.read().splitlines()

        col_row = None
        for i, line in enumerate(lines):
            fields = line.split("\t")
            key = fields[0].strip()
            if key == "Phase" and len(fields) > 5 and "Euler1" in fields:
                col_row = i
                break
            if key in ("XCells", "YCells"):
                self.header[key] = int(float(fields[1]))
            elif key in ("XStep", "YStep"):
                self.header[key] = float(fields[1])
            elif key == "Phases":
                self.header["Phases"] = int(fields[1])
            elif ";" in key and len(fields) >= 5:
                # a phase line: "a;b;c <tab> al;be;ga <tab> name <tab> laue <tab> sg"
                try:
                    self.phases.append(
                        {"name": fields[2].strip(), "laue": int(fields[3])}
                    )
                except (ValueError, IndexError):
                    pass

        if col_row is None:
            raise ValueError(
                f"{self.path}: no column header row found. Expected a line "
                "starting with 'Phase' and containing 'Euler1'."
            )
        for key in ("XCells", "YCells", "XStep", "YStep"):
            if key not in self.header:
                raise ValueError(f"{self.path}: header is missing {key}")

        self.columns = [c.strip() for c in lines[col_row].split("\t") if c.strip()]
        for required in ("Phase", "X", "Y", "Euler1", "Euler2", "Euler3"):
            if required not in self.columns:
                raise ValueError(f"{self.path}: no '{required}' column")

        data = np.genfromtxt(
            self.path, skip_header=col_row + 1, usecols=range(len(self.columns))
        )
        if data.ndim == 1:
            data = data[None, :]
        self.table = {c: data[:, i] for i, c in enumerate(self.columns)}
        self.npoints = data.shape[0]

    def __getitem__(self, key):
        return self.table[key]

    def has(self, key):
        return key in self.table

    @property
    def shape(self):
        return self.header["YCells"], self.header["XCells"]

    def crysym(self, phase_index=1):
        if not self.phases:
            return None, None
        ph = self.phases[phase_index - 1]
        return LAUE_TO_CRYSYM.get(ph["laue"]), ph


def build_grid(ctf, phase, max_mad, require_zero_error, min_bands):
    """Place the pixel table on the (ny, nx) grid and build the quality mask.

    Points are indexed from their X/Y coordinates rather than from row order,
    so a file that is not written in strict raster order still lands correctly
    and a truncated file leaves holes rather than shearing the map.
    """
    ny, nx = ctf.shape
    ix = np.rint(ctf["X"] / ctf.header["XStep"]).astype(int)
    iy = np.rint(ctf["Y"] / ctf.header["YStep"]).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not inside.all():
        print(f"  warning: {int((~inside).sum())} points fall outside XCells x YCells")

    good = inside & (ctf["Phase"] == phase)
    if require_zero_error and ctf.has("Error"):
        good &= ctf["Error"] == 0
    if ctf.has("MAD"):
        good &= ctf["MAD"] <= max_mad
    if min_bands and ctf.has("Bands"):
        good &= ctf["Bands"] >= min_bands

    euler = np.stack((ctf["Euler1"], ctf["Euler2"], ctf["Euler3"]), axis=-1)
    quat = euler_bunge_to_quat(euler[:, 0], euler[:, 1], euler[:, 2])

    qgrid = np.zeros((ny, nx, 4))
    qgrid[..., 0] = 1.0
    ok = np.zeros((ny, nx), dtype=bool)
    qgrid[iy[inside], ix[inside]] = quat[inside]
    ok[iy[good], ix[good]] = True

    # Preserve per-pixel quality fields for diagnostics.
    diag = {}
    for col, fill in (("Error", -1), ("MAD", np.nan), ("Bands", -1), ("Phase", -1)):
        if ctf.has(col):
            g = np.full((ny, nx), fill, dtype=float)
            g[iy[inside], ix[inside]] = ctf[col][inside]
            diag[col] = g
    return qgrid, ok, diag


def crop_grid(qgrid, ok, spec, xstep, ystep):
    """Cut a rectangular window out of the map, before segmentation.

    Cropping here rather than in Neper matters for two reasons:
     1. The segmentation, the prune and the cell ids all describe
        the same region, and a clipped grain is either big enough
        to keep or dropped like any other.

     2. Per-voxel orientations. Possible bug is that Neper 5.0.0 cannot read
        back a raster when the file carries a `**oridata` section and has
        been through (auto)crop. Cropping upstream keeps the file small enough
        to keep orientations, so -V colouring & -S intragranular measures still work.

    Bounds are in the .ctf's own 'as- acquired' length units,
    i.e. before any flip_y.
    """
    try:
        x0, x1, y0, y1 = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(
            f"crop={spec!r}: expected four comma-separated numbers, "
            "xmin,xmax,ymin,ymax, in the same units as XStep"
        )
    ny, nx = ok.shape
    ix0, ix1 = max(round(x0 / xstep), 0), min(round(x1 / xstep), nx)
    iy0, iy1 = max(round(y0 / ystep), 0), min(round(y1 / ystep), ny)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        raise SystemExit(
            f"crop={spec} keeps {max(ix1 - ix0, 0)} x {max(iy1 - iy0, 0)} "
            f"pixels. The map is {nx} x {ny} pixels of {xstep} x {ystep}, "
            f"i.e. {nx * xstep:g} x {ny * ystep:g} in those units."
        )
    window = (slice(iy0, iy1), slice(ix0, ix1))
    return qgrid[window], ok[window], window
