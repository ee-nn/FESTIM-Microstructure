"""Stage 1 of the EBSD pipeline: an Oxford/Channel .ctf map -> Neper .tesr.

Pure Python (NumPy + SciPy + matplotlib); Neper is only used, if found, for the
two rendered check images. Run::

    python examples/ebsd_ctf_to_tesr.py
"""

from pathlib import Path

import festim_microstructure as fm

HERE = Path(__file__).resolve().parent

res = fm.ebsd.convert.convert(
    str(HERE / "data" / "D7 PBF SS316L.ctf"),
    str(HERE / "data" / "d7.tesr"),
    min_pixels=15,
    diagnostics=True,
    max_mad=1.5,
    allow_error=True,
    crop="0,306,0,306",
)
print(f"segmentation error (rms): {res.rms_deg:.3f} deg")
