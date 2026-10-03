"""Import CTF grains, mesh with UPXO, and validate the FESTIM network.

Run in festim-microstructure-upxo before ebsd_gb_diffusion.py.
"""

from pathlib import Path

import festim_microstructure as fm

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "results" / Path(__file__).stem
CTF = str(HERE / "data" / "D7 PBF SS316L.ctf")

options = fm.EbsdOptions(
    ctf=CTF,
    import_settings=fm.ebsd.Settings(
        ctf=CTF,
        min_pixels=15,
        max_mad=1.5,
        allow_error=True,
        crop="0,306,0,306",
        diagnostics=True,
    ),
    mesh=fm.UpxoMeshOptions(smooth_lambda=0.25, smooth_mu=-0.265, smooth_iter=5),
    unit=1e-6,
    theta_min=10.0,
)
base = fm.meshing.ebsd.run_ebsd_pipeline(options, workdir=OUTPUT_DIR, force=True)
print(f"Validated mesh and network: {base}")
