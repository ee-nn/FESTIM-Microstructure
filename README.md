# FESTIM-Microstructure

⚠️ This repo is in its early stages and is actively being built. ⚠️ 

Microstructure-scale hydrogen transport modelling built on
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
| Microstructure generation | 2D and 3D Voronoi polycrystals via Gmsh |
| EBSD import | Conversion of EBSD orientation maps into conforming grain meshes |
| FESTIM subdomains and fields | Tagged grains, the boundary network as one codim-1 subdomain, per-grain surface patches, lattice tensor and per-boundary diffusivity fields |
| Post-processing | Grain-plus-network inventories and flux averages, homogenisation and short-circuit estimates, network topology checks |
| Benchmarks | Diaz-Rodriguez baseline, dimensional and non-dimensionalised |

The transport model itself is FESTIM's. Every grain is a
`festim.VolumeSubdomain` with a `festim.Species` of its own, the grain-boundary
network is one codimension-one `VolumeSubdomain` carrying one species, and each
grain exchanges with it through one `ParticleFluxBC`/`ParticleSource` pair --
exactly the pattern in FESTIM's manifold documentation. `festim_microstructure`
creates and interprets microstructures, converts them into FESTIM-compatible
subdomains and coefficient fields, and supplies specialised analysis; FESTIM
owns species, reactions, boundary conditions, settings, solving and standard
exports.

## Requirements

The solver and EBSD mesher share a Python 3.13 conda environment. Dependencies
come from two package managers:

- **conda-forge** supplies DOLFINx, the Gmsh Python API, FESTIM's compiled
  dependencies (`scifem`, `io4dolfinx`).
- **PyPI** supplies FESTIM itself. Versions 2.1 and 2.2rc* are not yet on
  conda-forge, where the newest 2.x build is `2.0b2.post2`.
- **UPXO requires Python 3.13** and shares the combined environment with FESTIM.

**Platforms:** linux-64, osx-64, osx-arm64. Windows users should work inside
WSL2 for the FEniCSx stack.

## Installation

Install [Miniforge](https://github.com/conda-forge/miniforge) (or Miniconda).
For a combined Python 3.13 solver and UPXO environment:

```bash
conda env create -f environment-festim-microstructure-upxo.yml
conda activate festim-microstructure-upxo
python -m pip install -e .
unset FM_UPXO_PYTHON
fm-check
python examples/ebsd_ctf_to_mesh.py
```

Run these commands from the repository checkout. The combined environment
installs the scientific and FEniCSx packages through conda and pins the evaluated
UPXO 1.3.1 commit and DefDAP 0.93.6. With Python 3.13 and UPXO installed, the importer and mesher automatically use
the active interpreter. An explicit `UpxoMeshOptions(python=...)` or
`FM_UPXO_PYTHON` still takes precedence.
CI creates this combined environment from scratch and runs the complete test
suite, including CTF-to-FESTIM transport and solver tests. Missing dependencies
or skipped tests fail that job.
Run `FM_REQUIRE_FULL_STACK=1 python -m pytest -q test/test_ebsd_si_mesh.py`
after installation to check mesh reconstruction, boundary exchange and inventory
conservation locally.

For solver workflows using the original Python 3.12 environment:

```bash
git clone https://github.com/ee-nn/FESTIM-Microstructure.git
cd FESTIM-Microstructure

# The FEniCSx / FESTIM environment
conda env create -f environment.yml
conda activate festim-microstructure
pip install -e .
```

`pip install -e .` is required -- without it the modules under `src/` are not
on the import path and none of the examples will run. For a development install
including test and lint extras:

```bash
pip install -e ".[test,lint]"
```

For VS Code, select the `festim-microstructure-upxo` Conda environment with
**Python: Select Interpreter**. The project configures `src/` as an analysis
search path for Pylance/Pyright; running the examples still requires the
editable install in the selected environment. If imports stay underlined after
changing environments, run **Developer: Reload Window**.

### Verifying the install

```bash
fm-check
```

prints the versions of the Python stack and probes the configured UPXO EBSD
worker. It exits non-zero if dependencies fail to import or the worker fails
its health check. It looks like:

```
python-side stack
  festim_microstructure  0.1.dev15
  ...
  dolfinx                0.10.0
  festim                 2.2rc2
  gmsh (python-gmsh)     4.13.1
  ...
  UPXO interpreter       /home/you/miniforge3/envs/festim-microstructure-upxo/bin/python
  EBSD worker            OK (runtime imports verified)
    upxo                 1.3.1
    defdap               0.93.6
```

For MPI, additionally:

```bash
mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```

The `gmsh (python-gmsh)` line matters: on conda-forge the `gmsh` package
installs only the executable and shared library, and the Python module comes
from the separate `python-gmsh` package. Both environment files include the
Python API used by the mesh builders.

### Updating

For the combined solver and EBSD environment:

```bash
conda env update -n festim-microstructure-upxo -f environment-festim-microstructure-upxo.yml --prune
conda activate festim-microstructure-upxo
python -m pip install --no-deps -e .
unset FM_UPXO_PYTHON
fm-check --ebsd
FM_REQUIRE_FULL_STACK=1 python -m pytest -q test/test_ebsd_si_mesh.py
```

For the original solver environment, use `conda env update -f environment.yml
--prune` and reinstall the package there.

Import and mesh cache identities probe the selected worker before reuse. They
record its Python executable/version, all installed distribution versions,
source provenance and installation-record fingerprints. Changing
`FM_UPXO_PYTHON` or reinstalling dependencies invalidates the matching cache.
The probe also rejects incompatible UPXO/DefDAP before a cached result is used.
Manual edits to installed dependency files or editable dependency source trees
are not fingerprinted: use immutable dependency installations, or `force=True`
after such edits. Project worker/importer code is fingerprinted separately.

`fm-check` probes an available EBSD worker and reports dependency versions and
runtime-import failures. `fm-check --ebsd` also fails when no worker is configured;
use it before running EBSD examples. The worker probe uses the selected
interpreter and does not require UPXO in a separate solver interpreter.

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
├── environment.yml             # Python 3.12 FEniCSx / FESTIM environment
├── environment-festim-microstructure-upxo.yml  # combined Python 3.13 environment
├── pyproject.toml
├── README.md
├── docs/
│   └── gb_homogenisation.md    # the homogenisation study, with figures
├── src/festim_microstructure/
│   ├── __init__.py              # public API with ordinary imports
│   ├── check.py                # fm-check environment diagnostic
│   ├── microstructure.py       # microstructure protocols, TaggedPolycrystal
│   ├── materials.py            # lattice tensor field, short-circuit estimates
│   ├── plotting.py             # raster colours, Matplotlib defaults, scale bars
│   ├── formats/                # file I/O
│   │   ├── ctf.py
│   │   ├── ebsd.py             # native grain/pixel arrays and masks
│   │   ├── msh4.py             # tagged Gmsh mesh readers
│   │   └── provenance.py
│   ├── ebsd/                   # UPXO/DefDAP measured grain import
│   │   ├── orientation.py
│   │   ├── settings.py
│   │   ├── convert.py
│   │   └── diagnostics.py
│   ├── voronoi/                # generated 2D/3D tessellations and Gmsh meshes
│   │   ├── geometry.py         # dimension-dispatched tessellation geometry
│   │   ├── gmsh_builder.py
│   │   └── polycrystal.py
│   ├── meshing/                # tessellation -> mesh -> BoundaryNetwork
│   │   ├── ebsd.py
│   │   ├── upxo.py             # UPXO worker and export adapter
│   │   └── diagnostics.py
│   ├── fem/
│   │   ├── solvers.py          # tune_direct_solver: the MUMPS workspace workaround
│   │   └── subdomains.py       # Grain, GrainBoundaryNetwork, GrainSurface + factories
│   └── exports/
│       ├── measures.py         # submesh measures and network topology checks
│       └── averages.py         # grain+network averages, inventory, fields, VTX output
├── examples/                   # runnable workflows; not part of the library API
│   ├── gb_homogenisation.py    # RVE identification, validation and figures
│   ├── li2022_fig4*.py         # Li et al. (2022) reproduction scripts
│   └── ...                     # usage examples and data/
└── test/                       # pytest in the combined FESTIM/UPXO environment
```

Nothing importable sits at the repository root. The package imports its public
API directly and requires the FESTIM/DOLFINx stack. Run the examples, conversion
tools, `fm-check`, and tests from an installed solver environment; the combined
Python 3.13 environment also supplies UPXO/DefDAP for EBSD workflows.

## Usage

A microstructure comes from the package; the model is declared with FESTIM.
The names you are most likely to want sit on the package itself:

```python
import dolfinx
import festim as F
import numpy as np

import festim_microstructure as fm

micro = fm.VoronoiMicrostructure.create(size=100e-6, n_seeds=64)
# The same entry point builds 3D structures with dim=3.
micro_3d = fm.VoronoiMicrostructure.create(size=100e-6, n_seeds=64, dim=3)

NETWORK_ID, SURFACE_ID_0 = 1_000_000, 2_000_000  # above every grain id
D_bulk, c0 = 1e-11, 1.0

# Coefficients: one lattice tensor per grain in a parent-mesh field, and an
# ordinary FESTIM material for the boundaries.
D_lattice, tensors = fm.materials.crystal_diffusivity_field(micro, D_bulk)
grains = fm.fem.subdomains.grain_subdomains(micro, F.Material(D=D_lattice))
network = fm.fem.subdomains.grain_boundary_network(
    NETWORK_ID, micro, F.Material(D_0=2.85e-7, E_D=0.12)
)
grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
c_gb = F.Species("c_gb", subdomains=[network])

# Each grain exchanges k (c_grain - c_gb) with the boundary slab; the network
# equation is written per unit slab width delta, hence k / delta on its side.
k = dolfinx.fem.Constant(micro.mesh, 3.0)
width = dolfinx.fem.Constant(micro.mesh, 1e-9)
sources, bcs = [], []
for c_grain in grain_species:
    exchange = {"c_g": c_grain, "c_n": c_gb}
    sources.append(F.ParticleSource(
        value=lambda c_g, c_n: (k / width) * (c_g - c_n), species=c_gb,
        volume=network, species_dependent_value=exchange))
    bcs.append(F.ParticleFluxBC(
        subdomain=network, species=c_grain,
        value=lambda c_g, c_n: k * (c_n - c_g), species_dependent_value=exchange))

# A charged surface: the patch each grain owns, and the network's mouths on it.
patches, mouths = fm.fem.subdomains.grain_surfaces(
    micro.mesh, grains, lambda x: np.isclose(x[1], micro.size), SURFACE_ID_0
)
species_of = dict(zip((g.id for g in grains), grain_species))
bcs += [F.FixedConcentrationBC(subdomain=p, value=c0, species=species_of[p.grain_id])
        for p in patches]
bcs.append(F.FixedConcentrationBC(subdomain=mouths, value=c0, species=c_gb))

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(micro.mesh),
    subdomains=[*grains, network, *patches, mouths],
    species=[*grain_species, c_gb],
    sources=sources,
    boundary_conditions=bcs,
    temperature=600.0,
    settings=F.Settings(atol=1e-25, rtol=1e-10, transient=False),
)
model.initialise()
fm.fem.solvers.tune_direct_solver(model)  # MUMPS workspace; see its docstring
model.run()

total = fm.exports.averages.inventory(grains, grain_species, network, c_gb, 1e-9)
```

There is one transport model. Every grain carries a lattice field of its own and
exchanges with a single codimension-one boundary network at the rate `k`; the
classical Fisher picture, one continuous lattice field for the whole
polycrystal, is the limit of it in which the boundary offers no resistance to
permeation, reached when `fm.materials.interface_resistance_ratio` is small.
A boundary-dependent diffusivity is one value per tessellation entity,
`grain_boundary_network(..., diffusivity_by_entity=...)`, which the network
turns into a field on its submesh when FESTIM creates it.

`fm.__all__` lists the curated top-level API. Classes such as `fm.Grain`,
`fm.TaggedPolycrystal`, and `fm.EbsdMicrostructure` are imported directly
alongside the geometry and subpackages.

Additional helpers are available through the subpackages:

```python
fm.voronoi.network_tensor
fm.ebsd.CtfConversion
fm.exports.averages.averages
fm.fem.subdomains.GrainBoundaryNetwork
fm.exports.measures.submesh_measure
```

The examples use a single package import, with curated names at the top level
and specialised helpers under their modules:

```python
import festim_microstructure as fm

fm.TaggedPolycrystal
fm.GrainBoundaryNetwork
fm.fem.subdomains.grain_surfaces
fm.materials.crystal_diffusivity_field
fm.voronoi.build_mesh
fm.voronoi.near
fm.voronoi.tessellate
```

Examples are executable workflows, intentionally kept outside the installed
API. Edit their setup values or copy the relevant package-level building blocks
into your own script:

```bash
python examples/voronoi_polycrystal_2d.py       # in-process Gmsh tessellation
python examples/voronoi_polycrystal_3d.py
python examples/ebsd_ctf_to_mesh.py             # EBSD conversion + SI mesh
python examples/ebsd_gb_diffusion.py            # transport using the saved SI mesh
python examples/fisher_grain_boundary.py        # single boundary vs Le Claire
python examples/gb_homogenisation.py --sizes 2e-6 3e-6 4e-6
python examples/li2022_fig4.py                 # volumetric-band reproduction
python examples/li2022_fig4_codim.py           # codim-1 reproduction
```

Generated files go to `examples/results/<script_name>/`; see
[example output locations](examples/README.md#outputs).

Generate meshes through Python functions:

```python
from pathlib import Path

import festim_microstructure as fm

micro = fm.VoronoiMicrostructure.create(
    size=100e-6, n_seeds=64, aspect=4, cells_per_grain=10, msh_path="poly2d.msh"
)
print(micro.report())

workdir = Path("results")
workdir.mkdir(parents=True, exist_ok=True)
ctf = "examples/data/D7 PBF SS316L.ctf"
options = fm.EbsdOptions(
    ctf=ctf,
    import_settings=fm.ebsd.Settings(
        ctf=ctf, min_pixels=15, max_mad=1.5, allow_error=True,
        crop="0,306,0,306", diagnostics=True,
    ),
    mesh=fm.UpxoMeshOptions(),
)
base = fm.meshing.ebsd.run_ebsd_pipeline(options, workdir=workdir)

```

The EBSD chain is **CTF → UPXO/DefDAP grains → UPXO mesh → FESTIM**.
Quality and phase filters and the crop apply before UPXO's `EBSDReader` detects
grains. DefDAP supplies grain labels and cubic symmetry equivalents. The importer
aligns quaternion signs before averaging to avoid DefDAP 0.93.6 cancellation; the
package translates its quaternion convention and applies explicit sample-frame
corrections. The current boundary-angle diagnostics support cubic m-3m (Laue 11).

`Settings(fill=True)` assigns unindexed/pruned pixels to the nearest surviving
grain for meshing. The archive retains their original membership and quality
mask, and those pixels never contribute to grain means or indexed orientation
error. Use `fill=False` to retain unmeshed regions. `min_pixels` and `threshold`
control DefDAP grain detection. `flip_y` mirrors rows and rotates both pixel and
grain orientations by 180° about sample x; `euler_correction` is a separate,
explicit Euler-to-map frame correction.

Configure smoothing and mesh sizes with `UpxoMeshOptions`. Defaults use five
Taubin passes (λ=0.25, μ=−0.265), explicit pixel seeds, and no grain merging or
diagonal repairs. `smoothing="none"` preserves raster boundaries. Sizes are in
pixel coordinates; both pixel spacings are applied when exporting to metres.
The pipeline rejects changes to grain IDs, components, holes, neighbor pairs,
coverage, mesh conformity or grain areas. Default smoothing limits are 5%
per-grain area change and one pixel of boundary displacement. There is no silent
fallback. With `fill=False`, `smoothing="none"` is the supported void-preserving
route: Taubin can move void boundaries and fail coverage validation.
`thin_grain_px=1.5` enables optional protection for thin grains; the default is
`0` (disabled). The adapter always sets `close_staircase=False` so protection
does not enable label-changing preprocessing. All preservation checks still apply.

Outputs include `<stem>.msh4`, `<stem>-ebsd.npz` (grain/pixel orientations,
labels, masks and spacings), `<stem>-metadata.json` (SI extent), import provenance,
`<stem>-validation.json`, and `<stem>-festim.json`. Quantitative diagnostics cover
source orientation transcription, indexed/filled error populations, grain area
and equivalent-diameter changes, displaced area, and grain-to-mesh identity.
Mesh diagnostic images are `<stem>-check-area.png` (grain-area changes),
`<stem>-check-mesh.png` (mesh over source pixels), and
`<stem>-check-network.png` (FESTIM boundary selection and disorientation).
Optional images show quality rejection, grains and cubic IPF-Z, orientation
error, mesh overlays, and the actual FESTIM boundary network.

For separate import/inspection, use
`fm.ebsd.convert.convert(ctf, "grains.npz", ...)` and
`fm.ebsd.measure_against_ctf(ctf, "grains.npz")`. Low-level
`fm.meshing.upxo.mesh_ebsd("grains.npz", ...)` meshes an existing native archive.
`force=False` checks source/settings/code and output hashes before cache reuse.
A failed meshing worker retains the previous accepted mesh and writes a failure
validation report and log.

In the combined Python 3.13 environment, the active interpreter runs both UPXO
and FESTIM. `FM_UPXO_PYTHON`, `Settings(python=...)`, or
`UpxoMeshOptions(python=...)` can select a compatible importer/mesher interpreter.
The combined environment pins UPXO's evaluated commit and its DefDAP extra.
UPXO remains an external dependency with its upstream GPL-3.0 license.

The returned Voronoi mesh is in metres; the optional Gmsh file uses units of
`size`. The `fm-check` environment diagnostic remains a console command.

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

GPL-3.0.
