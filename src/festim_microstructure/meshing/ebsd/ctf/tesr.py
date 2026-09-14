"""Write a Neper ``.tesr`` raster tessellation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["TesrData", "write_tesr"]


@dataclass
class TesrData:
    """Everything one ``.tesr`` holds, in the row order it will be written in.

    Grouped because the seven arrays and scalars always travel together: they
    are one raster, not seven independent arguments, and the writer had no way
    to catch two of them being swapped.
    """

    cellids: Any  #: (ny, nx) int, 0 = empty
    ori_cell: Any  #: (ncell, 3) Rodrigues, one per grain
    voxsize: tuple  #: (dx, dy)
    crysym: str
    ori_vox: Any = None  #: (ny, nx, 3) Rodrigues, or None to omit **oridata
    oridef: Any = None  #: (ny, nx) bool quality mask, written as **oridef

    @property
    def ncells(self) -> int:
        return int(self.cellids.max())


def write_tesr(path, raster: TesrData, precision=12):
    """Write the .tesr described by ``raster``.

    Section layout follows the EBSD tutorial and the file-format reference:
    https://neper.info/doc/tutorials/ebsd_process.html
    https://neper.info/doc/fileformat.html

    Voxels run with x varying fastest, matching the 4 x 3 example in the
    tutorial where twelve `**data` values describe a map four wide.
    """
    cellids, voxsize, crysym = raster.cellids, raster.voxsize, raster.crysym
    ori_cell, ori_vox, oridef = raster.ori_cell, raster.ori_vox, raster.oridef
    ny, nx = cellids.shape
    fmt = f"%.{precision}f"

    with open(path, "w") as fh:
        fh.write("***tesr\n")
        fh.write(" **format\n   2.2\n")
        fh.write(" **general\n   2\n")
        fh.write(f"   {nx} {ny}\n")
        fh.write(f"   {voxsize[0]:.12g} {voxsize[1]:.12g}\n")

        ncells = int(cellids.max())
        fh.write(" **cell\n")
        fh.write(f"   {ncells}\n")
        fh.write("  *id\n")
        ids = np.arange(1, ncells + 1)
        for lo in range(0, ncells, 20):
            fh.write("   " + " ".join(str(i) for i in ids[lo : lo + 20]) + "\n")
        fh.write("  *crysym\n")
        fh.write(f"   {crysym}\n")
        fh.write("  *ori\n")
        fh.write("   rodrigues:passive\n")
        for r in ori_cell:
            fh.write("   " + " ".join(fmt % v for v in r) + "\n")

        fh.write(" **data\n   ascii\n")
        flat = cellids.ravel()
        for lo in range(0, flat.size, 40):
            fh.write(" ".join(str(int(v)) for v in flat[lo : lo + 40]) + "\n")

        if ori_vox is not None:
            fh.write(" **oridata\n   rodrigues:passive\n   ascii\n")
            for r in ori_vox.reshape(-1, 3):
                fh.write("   " + " ".join(fmt % v for v in r) + "\n")
            fh.write(" **oridef\n   ascii\n")
            flags = oridef.ravel().astype(int)
            for lo in range(0, flags.size, 60):
                fh.write(" ".join(str(v) for v in flags[lo : lo + 60]) + "\n")

        fh.write("***end\n")
