# FESTIM-Microstructure

Microstructure-resolved hydrogen transport modelling built on
[FESTIM](https://festim.readthedocs.io/).

`festim-microstructure` is a companion package to FESTIM, similar to 
other FESTIM add-ons such as [`festim-gui`](https://github.com/festim-dev/festim-gui) and
[`festim-niuq`](https://pypi.org/project/festim-niuq/). `festim-microstructure` is
alongside festim in FESTIM in scripts that use it: 

```python
import festim as F
import festim_microstructure as fm
```

## What it provides

| Area | Contents |
| --- | --- |
| Microstructure generation | 2D and 3D Voronoi polycrystals via Gmsh; Neper-backed tessellations |
| EBSD import | Conversion of EBSD orientation maps into conforming grain meshes |
| Grain-boundary models | Fisher-type short-circuit diffusion; grain-boundary homogenisation |
| Benchmarks | Diaz-Rodriguez baseline, dimensional and non-dimensionalised |

## Requirements

The stack is split across two package managers, because the compiled parts are
not installable from PyPI:

- **conda-forge** supplies DOLFINx, the Gmsh Python API, Neper, and FESTIM's
  compiled dependencies (`scifem`, `io4dolfinx`).
- **PyPI** supplies FESTIM itself. Versions 2.1 and 2.2rc* are not yet on
  conda-forge, where the newest 2.x build is `2.0b2.post2`.

**Platforms:** linux-64, osx-64, osx-arm64. There is no Windows build of Neper
on conda-forge, and DOLFINx is easiest to obtain on Linux/macOS, so Windows
users should work inside WSL2. FESTIM's own installation guide makes the same
recommendation.

## Installation

Install [Miniforge](https://github.com/conda-forge/miniforge) (or Miniconda),
then:

```bash
git clone https://github.com/ee-nn/FESTIM-Microstructure.git
cd FESTIM-Microstructure
conda env create -f environment.yml
conda activate festim-microstructure
pip install -e .
```

`pip install -e .` is required — without it the modules under `src/` are not on
the import path and none of the examples will run.

For a development install including test and lint extras:

```bash
pip install -e ".[test,lint]"
```

### Verifying the install

```bash
python -c "import dolfinx, gmsh, festim, festim_microstructure; \
           print(dolfinx.__version__, festim.__version__)"
neper -V
gmsh --version
mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```

The `import gmsh` check matters: on conda-forge the `gmsh` package installs only
the executable and shared library. The Python module comes from the separate
`python-gmsh` package, which is why `environment.yml` lists both.

### Updating

```bash
conda env update -f environment.yml --prune
pip install -e .
```

### Version coupling

FESTIM is pinned in `environment.yml`, and DOLFINx is pinned to a minor range
alongside it. These two must be bumped together: FESTIM moved to DOLFINx 0.11
and replaced `adios4dolfinx` with `io4dolfinx` in
[festim-dev/FESTIM#1173](https://github.com/festim-dev/FESTIM/pull/1173),
merged one day before the 2.1 release. An unpinned DOLFINx will drift out from
under a pinned FESTIM and fail at import or at assembly.

If `pip` attempts to build `scifem` from source during installation, the
conda-forge `scifem` build is older than the pinned FESTIM requires. Either
relax the FESTIM pin or install it without resolving dependencies:

```bash
pip install --no-deps festim==2.2rc2
pip check
```

## Repository layout

```
FESTIM-Microstructure/
├── environment.yml
├── pyproject.toml
├── LICENSE
├── README.md
├── src/festim_microstructure/
│   ├── __init__.py
│   ├── meshing/
│   │   ├── voronoi.py          # Gmsh Voronoi polycrystals (2D/3D)
│   │   ├── neper.py            # Neper tessellation wrapper
│   │   └── ebsd.py             # EBSD map -> conforming mesh
│   ├── models/
│   │   ├── fisher.py           # Fisher grain-boundary short circuit
│   │   ├── homogenisation.py   # GB homogenisation
│   │   └── baselines.py        # Diaz-Rodriguez benchmarks
│   └── postprocessing/
├── examples/                   # runnable scripts, one per workflow
└── test/                       # pytest suite
```

Scripts that were previously at the repository root now live either as library
modules under `src/festim_microstructure/` or as runnable examples under
`examples/`. Nothing importable should sit at the repository root.

## Usage

Generate a 2D Voronoi polycrystal and run a grain-boundary diffusion problem:

```bash
python examples/voronoi_polycrystal_2d.py
```

Or from the console entry points declared in `pyproject.toml`:

```bash
fm-voronoi --n-grains 64 --domain-size 100e-6 --out mesh/poly2d.msh
```

Parallel runs use MPI directly, as with any DOLFINx program:

```bash
mpirun -n 8 python examples/gb_homogenisation.py
```

## Testing

```bash
pytest
pytest --cov=festim_microstructure --cov-report=term-missing
ruff check src test
ruff format --check src test
```

## Citing

If this package contributes to published work, please cite FESTIM alongside it —
see `CITATION.cff` in the [FESTIM
repository](https://github.com/festim-dev/FESTIM).

## Contributing

Issues and pull requests are welcome. Run `ruff check` and `pytest` before
opening a PR. FESTIM's own [developer
guide](https://festim.readthedocs.io/en/latest/devguide/index.html) is a
reasonable reference for style and review conventions.

## Getting help

For FESTIM questions rather than microstructure questions, use the FESTIM
[Discourse](https://festim.discourse.group/) or Slack channel.

## License

Apache-2.0, matching FESTIM.