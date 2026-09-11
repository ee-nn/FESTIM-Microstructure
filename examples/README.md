# Examples

Scripts in this directory are complete workflows, parameter studies, and paper
reproductions. They are intentionally not included in the installed
`festim_microstructure` API: their defaults, outputs, figures, and input data
belong to a particular study rather than being stable library contracts.

Use the reusable pieces from `festim_microstructure.meshing`,
`festim_microstructure.models`, `festim_microstructure.subdomains`, and
`festim_microstructure.postprocessing.measures` when building a new workflow.

- `voronoi_polycrystal_2d.py`, `voronoi_polycrystal_3d.py`, and
  `neper_voronoi_network.py` demonstrate short-circuit diffusion.
- `ebsd_ctf_to_tesr.py` and `ebsd_gb_diffusion.py` demonstrate the EBSD path.
- `gb_homogenisation.py`, `gb_validation.py`, and `gb_figures.py` are one RVE
  study, run in that order.
- `li2022_fig4.py` and `li2022_fig4_codim.py` reproduce and compare the two
  grain-boundary formulations used for Li et al. (2022), Fig. 4.
