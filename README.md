# FESTIM-Microstructure

⚠️ This repo is in its early stages and is actively being built. ⚠️ 

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

The stack is split across two package managers and **two conda environments**:

- **conda-forge** supplies DOLFINx, the Gmsh Python API, FESTIM's compiled
  dependencies (`scifem`, `io4dolfinx`), and Neper.
- **PyPI** supplies FESTIM itself. Versions 2.1 and 2.2rc* are not yet on
  conda-forge, where the newest 2.x build is `2.0b2.post2`.
- **Neper lives in its own environment.** conda-forge's `neper` is built
  against `scotch 6.1.x`, which pins `zlib <1.3`, while `fenics-dolfinx >=0.10`
  needs `libzlib >=1.3.2` (through `libadios2`). No single solve can contain
  both; `conda env create` on a combined file fails with
  `LibMambaUnsatisfiableError` on exactly that pair. Since Neper is only ever
  run as a subprocess -- nothing in the Python stack links against it -- the
  split costs nothing: the FEniCSx environment just needs to know the path.

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

# 1. the FEniCSx / FESTIM environment (this is the one you work in)
conda env create -f environment.yml
conda activate festim-microstructure
pip install -e .

# 2. Neper, in its own environment (never activated)
conda env create -f environment-neper.yml

# 3. tell environment 1 where environment 2's executables are
tools/link-neper-env.sh
conda activate festim-microstructure   # re-activate so the variables load
fm-check
```

Step 3 records `FM_NEPER_BIN`, `FM_GMSH_BIN` and `FM_POVRAY_BIN` as conda
environment variables of `festim-microstructure` (`conda env config vars`), so
they are exported on activate and cleared on deactivate; nothing is written to
your shell profile. If you prefer, skip the script and export the same three
variables yourself -- or pass paths explicitly (`run_neper(..., neper_bin=...)`).
Resolution order is always explicit argument, then the variable, then `PATH`.

The Neper environment is optional. Without it, the Gmsh-based Voronoi route
(`fm-voronoi`, `examples/voronoi_polycrystal_*.py`), the EBSD `.ctf` converter,
and the homogenisation study all work; only Neper-backed tessellations and EBSD
*meshing* need it. The POV-Ray render checks are optional within that: on
linux-64, `conda install -n neper-env conda-forge::povray` and re-run step 3.

`pip install -e .` is required -- without it the modules under `src/` are not
on the import path and none of the examples will run. For a development install
including test and lint extras:

```bash
pip install -e ".[test,lint]"
```

### Verifying the install

```bash
fm-check
```

prints the versions of the Python-side stack and where each external program
resolves to, and exits non-zero if the FEniCSx side is broken. It looks like:

```
python-side stack
  festim_microstructure  0.1.dev15
  ...
  dolfinx                0.10.0
  festim                 2.2rc2
  gmsh (python-gmsh)     4.13.1
external programs (explicit path > FM_*_BIN > PATH)
  neper    /home/you/miniforge3/envs/neper-env/bin/neper  [env var]  neper 5.0.0
  gmsh     /home/you/miniforge3/envs/neper-env/bin/gmsh   [env var]  4.13.1
  povray   not found      (FM_POVRAY_BIN unset)
```

For MPI, additionally:

```bash
mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```

The `gmsh (python-gmsh)` line matters: on conda-forge the `gmsh` package
installs only the executable and shared library, and the Python module comes
from the separate `python-gmsh` package. `environment.yml` lists `python-gmsh`;
`environment-neper.yml` lists `gmsh`, because the executable is what `neper -M`
calls, and it should be the one Neper was built alongside. Keeping them apart
means the two Gmsh builds never mix.

Paths with whitespace are rejected (`find_binary` raises): Neper re-tokenizes
its arguments, so a conda prefix such as `~/my envs/neper-env` is torn into
fragments by the time Neper sees it. Put the environments somewhere without a
space, or symlink them.

### Updating

```bash
conda env update -f environment.yml --prune
pip install -e .
conda env update -f environment-neper.yml --prune
tools/link-neper-env.sh        # only needed if an env was recreated or moved
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
├── environment.yml             # FEniCSx / FESTIM env (the one you work in)
├── environment-neper.yml       # Neper + gmsh executable, kept separate (see Requirements)
├── tools/link-neper-env.sh     # records the Neper paths on the FEniCSx env
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
│   ├── check.py                # `fm-check`: report versions and where the programs resolve
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
│   └── postprocessing/
│       ├── measures.py         # inventory, submesh length/area, component count
├── examples/                   # runnable workflows; not part of the library API
│   ├── gb_homogenisation.py    # RVE identification study
│   ├── gb_validation.py        # validation of the RVE result
│   ├── gb_figures.py           # bespoke figures for the study
│   ├── li2022_fig4*.py         # Li et al. (2022) reproduction scripts
│   └── ...                     # usage examples and data/
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
from festim_microstructure.meshing.voronoi import (
    build_mesh,
    near_segments,
    voronoi_segments,
)
from festim_microstructure.models.fisher import ShortCircuitParams, ShortCircuitProblem
from festim_microstructure.subdomains import GrainBoundaryNetwork
```

Examples are executable workflows, intentionally kept outside the installed
API. Edit their setup values or copy the relevant package-level building blocks
into your own script:

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
python examples/li2022_fig4.py                 # volumetric-band reproduction
python examples/li2022_fig4_codim.py           # codim-1 reproduction
```

Console entry points from `pyproject.toml`:

```bash
fm-voronoi --n-grains 64 --domain-size 100e-6 --aspect 4 --out poly2d.msh
fm-ebsd examples/data/"D7 PBF SS316L.ctf" --out results --crop 0,306,0,306
```

Neper, Gmsh and POV-Ray are found through `FM_NEPER_BIN`, `FM_GMSH_BIN`,
`FM_POVRAY_BIN` (set by `tools/link-neper-env.sh`), falling back to `PATH`;
`fm-check` shows what resolved. Every Neper subprocess runs with those
directories prepended to its `PATH`, so Neper finds Gmsh and POV-Ray by itself.
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
