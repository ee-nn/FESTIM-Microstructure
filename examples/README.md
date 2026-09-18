# Examples

Scripts in this directory are complete workflows, parameter studies, and paper
reproductions. They are intentionally not included in the installed
`festim_microstructure` API: their defaults, outputs, figures, and input data
belong to a particular study rather than being stable library contracts.

Import the package as `import festim_microstructure as fm`. Use curated names
such as `fm.Physics`, `fm.build`, and `fm.VoronoiMicrostructure`, and access
specialised helpers through namespaces such as `fm.voronoi`, `fm.meshing`,
`fm.formats`, and `fm.exports`.

- `voronoi_polycrystal_2d.py`, `voronoi_polycrystal_3d.py`, and
  `neper_voronoi_network.py` demonstrate short-circuit diffusion.
- `ebsd_ctf_to_tesr.py` and `ebsd_gb_diffusion.py` demonstrate the EBSD path.
- `gb_homogenisation.py`, `gb_validation.py`, and `gb_figures.py` are one RVE
  study, run in that order.
- `li2022_fig4.py` and `li2022_fig4_codim.py` reproduce and compare the two
  grain-boundary formulations used for Li et al. (2022), Fig. 4.

## Outputs

Each script writes generated files under `examples/results/<script_name>/`,
independent of the directory from which it is run. For example, the 2D Voronoi
exports are in `examples/results/voronoi_polycrystal_2d/`. Meshes, diagnostics,
figures, and JSON files use the same per-script layout. Rerunning a script can
replace its own outputs; different scripts use separate folders.

The EBSD conversion writes `results/ebsd_ctf_to_tesr/d7.tesr`, which
`ebsd_gb_diffusion.py` reads by default. The original CTF stays in `examples/data/`.

`gb_figures.py` reads `results/gb_homogenisation/identified.json` and
`results/gb_validation/validation.json` by default and writes to
`results/gb_figures/`. Its `--rve` and `--validation` options accept other input
paths. For the two JSON-producing scripts, `--out` is relative to that script's
results folder and must stay inside it. Homogenisation field exports include
the cell size and seed in their names to preserve each case in a sweep.

`plot_li2022_fig4.py` recreates the two Li Fig. 4 comparison PNGs from their
CSV outputs, requiring only NumPy and Matplotlib:

```bash
python examples/plot_li2022_fig4.py
```

Running the file reads `results/li2022_fig4/li2022-fig4.csv` and
`results/li2022_fig4_codim/li2022-fig4-codim-thin.csv`, relative to the script,
and saves each PNG beside its CSV. No command-line options are needed.
The analytical reference curves are recomputed; all simulation markers are
read from the CSV files.
