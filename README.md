# FESTIM-Microstructure

Tools and example workflows for modeling hydrogen transport through
microstructures with [FESTIM](https://festim.readthedocs.io/). The repository
contains scripts for creating Voronoi and Neper grain structures, converting
EBSD data into computational meshes, and running hydrogen diffusion simulations
on grain and grain-boundary networks.

## Installation

The project uses Conda for the compiled numerical dependencies and installs
FESTIM through pip. Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
or [Miniforge](https://github.com/conda-forge/miniforge), then run:

```bash
conda env create -f environment.yml
conda activate festim-microstructure
```

The environment includes FESTIM 2.2-cr2, Neper 5.0.0 or later, DOLFINx,
Gmsh, MPI, NumPy, SciPy, and Matplotlib. Confirm that the main command-line
tools are available with:

```bash
python -c "import festim, dolfinx, gmsh; print('FESTIM-Microstructure environment ready')"
neper -V
gmsh --version
```

If the environment already exists, update it after pulling changes with:

```bash
conda env update -f environment.yml --prune
```
