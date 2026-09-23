"""Convert an Oxford/Channel .ctf map to TESR and a simulation mesh in metres.

Requires Neper and Gmsh, found through FM_NEPER_BIN / FM_GMSH_BIN or PATH.
Run this before ``ebsd_gb_diffusion.py``.
"""

from pathlib import Path

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HERE = Path(__file__).resolve().parent

res = fm.ebsd.convert.convert(
    str(HERE / "data" / "D7 PBF SS316L.ctf"),
    str(OUTPUT_DIR / "d7.tesr"),
    min_pixels=15,
    diagnostics=True,
    max_mad=1.5,
    allow_error=True,
    crop="0,306,0,306",
)
print(f"segmentation error (rms): {res.rms_deg:.3f} deg")

# Mesh in raster units, then export the final mesh and extent in metres.
ebsd = fm.EbsdOptions(
    tesr=str(OUTPUT_DIR / "d7.tesr"),
    unit=1e-6,
    theta_min=10.0,
    mesh=fm.TesrMeshOptions(
        rcl=0.25, mesh_qual_min=0.7, tesr_smooth_fact=0.25, tesr_smooth_iter=5
    ),
)
base = fm.meshing.ebsd.run_ebsd_pipeline(ebsd, workdir=OUTPUT_DIR, force=True)
mesh, cell_tags, facet_tags = fm.formats.msh4.read_mesh(base, gdim=2)
micro = fm.EbsdMicrostructure.from_mesh(
    base,
    mesh,
    cell_tags,
    facet_tags,
    fm.meshing.ebsd.read_extent(base),
    theta_min=ebsd.theta_min,
)
micro.check_orientations()
fm.meshing.ebsd.write_network_png(
    base,
    mesh,
    micro,
    OUTPUT_DIR / f"{ebsd.stem}-unscaled-raw.tesr",
    ebsd.unit,
    fm.meshing.ebsd.unit_name(ebsd.unit),
)
fm.meshing.ebsd.cleanup_unscaled_files(base)
