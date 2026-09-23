# Examples

Scripts in this directory are complete workflows, parameter studies, and paper
reproductions. They are intentionally not included in the installed
`festim_microstructure` API: their defaults, outputs, figures, and input data
belong to a particular study rather than being stable library contracts.

Import the package as `import festim_microstructure as fm`. Use curated names
such as `fm.VoronoiMicrostructure`, `fm.TaggedPolycrystal` and
`fm.GrainBoundaryNetwork`, and access specialised helpers through namespaces
such as `fm.voronoi`, `fm.meshing`, `fm.fem`, `fm.materials`, `fm.formats`, and
`fm.exports`. The transport model in every script is declared with FESTIM
directly -- species, exchange terms, boundary conditions, settings, the
`HydrogenTransportProblemDiscontinuous` -- following FESTIM's manifold
documentation; the package supplies the subdomains, the coefficient fields and
the post-processing. `gb_homogenisation.py` contains the complete RVE study,
sharing one cell-problem declaration across identification and validation.

- `voronoi_polycrystal_2d.py`, `voronoi_polycrystal_3d.py`, and
  `neper_voronoi_network.py` demonstrate short-circuit diffusion.
- `ebsd_ctf_to_tesr.py` and `ebsd_gb_diffusion.py` demonstrate the EBSD path.
- `gb_homogenisation.py` identifies effective diffusivity, validates it against
  permeation and uptake, and generates the study figures in one run.
- `li2022_fig4.py` and `li2022_fig4_codim.py` reproduce and compare the two
  grain-boundary formulations used for Li et al. (2022), Fig. 4.

## Outputs

Each script writes generated files under `examples/results/<script_name>/`,
independent of the directory from which it is run. For example, the 2D Voronoi
exports are in `examples/results/voronoi_polycrystal_2d/`. Meshes, diagnostics,
figures, and JSON files use the same per-script layout. Rerunning a script can
replace its own outputs; different scripts use separate folders.

The EBSD preprocessing writes `results/ebsd_ctf_to_tesr/d7.tesr` and
`poly.msh4` in metres, with SI extent and grain-orientation files alongside it.
`ebsd_gb_diffusion.py` reads these prepared files; its simulation exports go to
`results/ebsd_gb_diffusion/`. Temporary raster-unit meshing files use an
`-unscaled` stem and are deleted after all diagnostic images have been written.
Rerun preprocessing to replace meshes generated with the old unit convention.
The original CTF stays in `examples/data/`.

The consolidated study writes `identified.json`, `validation.json`, and
`fig_*.png` under `results/gb_homogenisation/`. Validation and the field map
reuse the first size/seed's identification and transport parameters. Use
`--skip-validation`, `--skip-transient`, `--skip-figures`, or `--skip-fields`
to omit stages. `--plot-only` regenerates summary figures from the JSON without
solving; the field map is generated during a simulation run. `--out` and
`--validation-out` select JSON filenames within the study's results directory.
The exchange-rate plot requires `--k-sweep`; the RVE plot requires multiple
identifications. Field exports include cell size and seed in their names.

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
