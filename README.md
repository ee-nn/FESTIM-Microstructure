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
├── README.md
├── docs/
│   └── gb_homogenisation.md    # the homogenisation study, with figures
├── src/festim_microstructure/
│   ├── __init__.py
│   ├── subdomains.py           # GrainBoundaryNetwork (locator), TaggedGrainBoundaryNetwork
│   │                           #   (facet tags), Grain, GrainSurface
│   ├── solvers.py              # ATOL and the MUMPS workaround, see docs/
│   ├── _binaries.py            # find_binary(): FM_NEPER_BIN / FM_GMSH_BIN / FM_POVRAY_BIN
│   ├── meshing/
│   │   ├── voronoi.py          # 2D/3D Voronoi polycrystals via Gmsh; `fm-voronoi`
│   │   ├── neper.py            # neper -T / -M wrapper, stat readers, raster meshing
│   │   └── ebsd/
│   │       ├── ctf.py          # .ctf -> .tesr converter (pure Python)
│   │       ├── orientation.py  # quaternions, cubic symmetry, disorientation
│   │       ├── pipeline.py     # .tesr -> mesh -> EbsdMicrostructure; `fm-ebsd`
│   │       └── ...             # segmentation_error, grain_area_change, figures
│   ├── models/
│   │   ├── fisher.py           # ShortCircuitProblem: one lattice + one network
│   │   ├── properties.py       # Physics, per-grain / per-boundary coefficient fields
│   │   ├── resolved.py         # one subdomain per grain, coupled through the network
│   │   ├── homogenisation.py   # identify an anisotropic D_eff; `fm-homogenise`
│   │   ├── validation.py       # does D_eff predict what it was not fitted to?
│   │   └── diaz_rodriguez*.py  # Diaz-Rodriguez et al. (2022) benchmarks
│   └── postprocessing/
│       ├── measures.py         # inventory, submesh length/area, component count
│       └── figures.py          # the figures in docs/
├── examples/                   # one runnable driver per workflow, plus data/
└── test/                       # pytest; the FEniCS-dependent tests skip without it
```

Nothing importable sits at the repository root. The pure-NumPy parts of the
package -- the EBSD converter, the orientation algebra, the Voronoi
tessellation geometry, the Neper stat readers -- import and test without
dolfinx or FESTIM installed; the mesh builders and models import them lazily.

## Usage

```python
import festim as F
import festim_microstructure as fm
from festim_microstructure.meshing.voronoi import build_mesh, near_segments, voronoi_segments
from festim_microstructure.models.fisher import ShortCircuitParams, ShortCircuitProblem
from festim_microstructure.subdomains import GrainBoundaryNetwork
```

Each example is a `Setup` dataclass plus a `main()`; edit the dataclass or
import `main` and pass your own:

```bash
python examples/voronoi_polycrystal_2d.py       # in-process Gmsh tessellation
python examples/voronoi_polycrystal_3d.py
python examples/neper_voronoi_network.py        # needs neper + gmsh executables
python examples/ebsd_ctf_to_tesr.py             # stage 1 of the EBSD pipeline
python examples/ebsd_gb_diffusion.py            # stages 2-3 + the transport model
python examples/fisher_grain_boundary.py        # single boundary vs Le Claire
python examples/gb_homogenisation.py --sizes 2e-6 3e-6 4e-6 --out rve.json
python examples/gb_validation.py --out validation.json
python examples/gb_figures.py --rve rve.json --validation validation.json
```

Console entry points from `pyproject.toml`:

```bash
fm-voronoi --n-grains 64 --domain-size 100e-6 --aspect 4 --out poly2d.msh
fm-ebsd examples/data/"D7 PBF SS316L.ctf" --out results --crop 0,306,0,306
fm-homogenise --k-sweep 1e-6 1e-4 1e-2 3 --out rve.json
```

Neper, Gmsh and POV-Ray are found on `PATH` or through `FM_NEPER_BIN`,
`FM_GMSH_BIN`, `FM_POVRAY_BIN`; Neper may live in its own conda environment.
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