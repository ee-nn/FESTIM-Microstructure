# UPXO 2D EBSD meshing evaluation

**Result: the 2D EBSD pathway uses pinned UPXO 1.3.1 with explicit pixel
seeds and FESTIM’s export adapter. Exact polygons pass all eight fixtures;
gentle Taubin passes the real crop and six synthetic fixtures. Serial
CTF-to-FESTIM transport validates boundary exchange and inventory conservation.**

The historical evaluation ran on October 2, 2026, on branch `upxo`, against UPXO 1.2.0 at
[`fef845ba10a3e849a5bb6b0a3c2ad89098035739`](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/tree/fef845ba10a3e849a5bb6b0a3c2ad89098035739).
The evaluation initially left the production pipeline unchanged. The subsequent
replacement described below now uses the evaluated path for EBSD meshing.
The code imports UPXO; it does not copy or modify UPXO source.

## UPXO 1.3.1 compatibility refresh

The production pin is now UPXO **1.3.1**, revision
[`a53885ef0a7a06f6b3fb195747363ae0a19bb127`](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/tree/a53885ef0a7a06f6b3fb195747363ae0a19bb127),
with DefDAP **0.93.6**. Both workers share one compatibility check.
The migration scope is the **2D EBSD pathway**; generated Neper polycrystals
remain an optional supported backend.

Rerun on October 9, 2026, with the DefDAP importer and UPXO 1.3.1 in the
combined environment. Geometry, mesh, export and serial DOLFINx reconstruction
reproduced the supplied review across all eight inputs:

| Path | Accepted cases |
|---|---:|
| Exact raster polygons | 8/8 |
| Gentle Taubin, explicit pixel seeds | 7/8 |
| Automatic seeds, existing harness settings | 0/8 |

The real crop retained 264 grains and 777 arcs with gentle Taubin: maximum
area change 1.285%, boundary displacement 0.298 pixels, no lost or added
neighbor pairs, and 124,004 triangles. Automatic seeds now construct and export
all eight meshes, resolving the old construction errors, but still fail the
acceptance gates. The real crop has 20.2% maximum area change and 1.229-pixel
boundary displacement. Explicit pixel seeds remain the production policy.
All 24 adapted exports reconstructed successfully in DOLFINx; the rejected
cases still fail geometry acceptance. Native exports still lacked facet tags.
Observed metrics and input provenance are checked in as
[`upxo_evaluation_131.json`](upxo_evaluation_131.json). The older tables below
remain historical measurements against 1.2.0.

Native physical curve tags remain absent, so the FESTIM export adapter stays.
DefDAP symmetry-aware averaging also stays: UPXO's newer
`grain_average_euler_deg()` uses an arithmetic quaternion mean. Rejected pixels
remain excluded from grain means. `EbsdMicrostructure.from_mesh()` now creates
vertex-to-cell connectivity itself; callers need no extra topology setup.

For `fill=False`, the supported void-preserving route is
`UpxoMeshOptions(smoothing="none")`. Gentle Taubin moves the interior-void
boundary and appropriately fails coverage validation. The checks are unchanged.
Optional `thin_grain_px=1.5` enables thin-grain protection; the default is 0.
The adapter explicitly passes `close_staircase=False` to prevent the protection
option from triggering label-changing preprocessing.

The required `ebsd-full-stack` CI job installs the combined Python 3.13
environment from scratch. It runs CTF-to-archive-to-tagged-mesh reconstruction,
a closed FESTIM grain/boundary transport problem (inventory conservation and
nonzero boundary exchange), and existing solver tests. Any dependency skip fails
the job; this failure policy was also verified with a deliberately missing worker.
Local validation passed **49 integration/worker/import tests with zero skips**
and **95 additional regression tests**. Ruff lint/format and `pip check` passed.
Local command:

```bash
FM_REQUIRE_FULL_STACK=1 python -m pytest -q test/test_ebsd_si_mesh.py \
  test/test_short_circuit.py test/test_subdomains.py test/test_gb_field.py \
  test/test_gb_homogenisation.py
```

## Full EBSD replacement

The production chain now imports CTF grains through UPXO's modern
`EBSDReader` and DefDAP 0.93.6, meshes with UPXO, and validates the actual
DOLFINx/FESTIM network. Custom segmentation, Neper topology repair, TESR
handoffs, optional Neper rendering and temporary unscaled mesh exports have
been removed. Generic TESR readers and generated Neper tessellations remain
separate APIs.

Quality/phase masks and crop apply before detection. DefDAP's grain means are
symmetry-aware; UPXO's simpler `grain_avg_quats` was not used because it only
resolves quaternion sign ambiguity. Both original membership and filled geometry
are persisted, so rejected/pruned pixels cannot bias grain means and subsequent
source comparisons reproduce the same diagnostic populations. Transcription,
frame changes, quality, indexed/filled errors, IPF-Z, grain areas/ECD, overlays
and FESTIM network diagnostics now operate on native arrays and the final SI mesh.

The default meshing policy remains five gentle Taubin passes (λ=0.25,
μ=−0.265), explicit pixel seeds, no grain merging and no diagonal repair.
Geometry and mesh validation are mandatory, including actual triangle
components/holes. Exact polygons use `smoothing="none"`.
The combined environment pins the UPXO revision and DefDAP EBSD extra.
The combined Python 3.13 environment was installed and tested locally on
October 9, 2026: UPXO 1.3.1, DefDAP 0.93.6, FESTIM 2.2rc2, DOLFINx 0.11.0,
SciFEM 0.26.0, NumPy 2.5.3 and Gmsh 4.15.2. CTF-to-FESTIM transport tests
pass for both exact polygons and gentle Taubin, including inventory conservation
and transfer into an initially empty grain boundary.

The complete example crop (191 × 191 pixels at 1.6 µm spacing) passed:

- 264 DefDAP grains, all retained by the mesh.
- 35,297 original grain pixels; 1,152 pruned and 32 unindexed pixels filled
  explicitly for geometry and excluded from grain means/indexed error.
- Indexed orientation RMS 5.734°, filled-pixel RMS 39.040°; source transcription
  maximum 1.79 × 10⁻⁵°.
- Maximum absolute grain-area change 1.29%; total area unchanged.
- FESTIM reconstructed 777 arcs (710 interior, 67 surface) and 447 junctions.

The tables below record the original **mesher-only** evaluation, performed
using the previous preprocessing labels. They explain the selected smoothing
policy; they do not claim the DefDAP importer was part of that original audit.
The current harness prepares real-map labels through the new importer, so future
runs use DefDAP labels and means and can differ from those historical results.

## Historical 1.2.0 results

Eight inputs were evaluated through five paths: exact raster polygons, Taubin,
moving-average, mean-coordinate smoothing, and Taubin with automatic seeds.
There were 40 main runs and four follow-up controls. All eight exact-polygon runs
passed geometry, meshing, export, and serial DOLFINx checks after adding FESTIM's
required tags.

| Input | Exact polygons | Taubin | Moving average | Mean coordinates | Automatic seeds |
|---|---|---|---|---|---|
| Enclosed grain | Pass | Pass | Fail | Fail | Construction error |
| Nested enclosed grains | Pass | Pass | Fail | Fail | Construction error |
| Two disconnected regions with the same grain ID | Pass | Pass | Fail | Fail | Construction error |
| Diagonal point contact | Pass | Pass | Fail | Fail | Construction error |
| One- and two-pixel twin ribbons | Pass | Fail | Fail | Fail | Construction error |
| Three-grain junction | Pass | Pass | Pass | Fail | Construction error |
| Interior void | Pass | Fail | Fail | Fail | Construction error |
| Real CTF crop | Pass | Pass | Fail | Fail | Fail |

"Pass" includes the evaluation-only export adapter. None of the 37 native
exports carried physical facet tags. An import pass by itself does not mean a
smoothed mesh is faithful to the original scan.

UPXO's own `test_confmesh2d_gmsh.py` also passed: **12 tests**. The evaluation's
seven audit tests passed in the UPXO environment; the FESTIM environment passed
six and skipped the one requiring Shapely/Rasterio. The targeted project checks
passed **27 tests**, with that same optional-dependency skip.

### Real measured map

The input was `examples/data/D7 PBF SS316L.ctf`, using the existing example's
crop `0,306,0,306`, `min_pixels=15`, `max_mad=1.5`, `allow_error=True`, and
10-degree segmentation threshold. Filling rejected pixels remained enabled;
**Neper-specific topology repair was disabled**.

- The crop has 191 × 191 pixels, 1.6 µm spacing, and a 305.6 × 305.6 µm extent.
- Segmentation retained **264 grains and five enclosed boundary loops**.
- Running the existing Neper topology repair on the same labels reduces this to
  **259 grains**, absorbing original IDs 58, 153, 162, 170, and 260. No diagonal
  pinches were repaired in this crop.
- UPXO's exact-polygon mesh retained all 264 grains and the five loops. It had
  **69,364 nodes, 137,198 triangles, 777 tagged boundary arcs**, and a minimum
  triangle angle of **21.23°**. The run took approximately **20 seconds**,
  including geometry checks and export; this is not a
  controlled speed comparison with Neper.
- No inverted/zero-area triangles, nonmanifold facets, duplicate coordinate
  nodes, or missing boundary segments were found. Maximum mesh-versus-polygon
  grain-area error was **7.14 × 10⁻⁸ relative**.
- DOLFINx 0.11 imported the adapted mesh. Every triangle's grain ID was checked
  against its pre-export owner using its centroid. Boundary grain pairs,
  original orientation rows, all facet IDs, metre coordinates, and per-arc
  lengths survived the round trip.

The five-pass smoothing results on that same crop were:

| Path | Maximum grain-area change | Input-polygon gap | Input-polygon overlap | Grain pairs lost in final mesh |
|---|---|---|---|---|
| Exact polygons | 0% | 0% | 0% | 0 |
| Taubin, explicit pixel seeds | 4.09% | 0% | 0% | 0 |
| Moving average, explicit pixel seeds | 19.47% | 0.0663% | 0.642% | 1 |
| Mean coordinates, explicit pixel seeds | 21.72% | 0.482% | 0.557% | 15 |
| Taubin, automatic seeds | 52.33% | 2.44% | Numerical noise | 0 |

Taubin preserves all grain IDs, components, holes, and neighbor pairs. Its
maximum boundary displacement is **0.522 pixels**, and its mesh contains
**116,195 triangles** with a minimum angle of **20.76°**. All smoothed CTF
outputs still imported after tagging; importability alone does not detect the
moving-average paths' geometry errors.

Two controls distinguish reconstruction fidelity from filter strength:

| Input | Zero-pass reconstruction | Gentler Taubin (λ=0.25, μ=−0.265, five passes) |
|---|---|---|
| Real CTF crop | Pass | Pass: 1.29% maximum area change, 0.298 px displacement |
| Thin-twin fixture | Pass | Pass: 2.05% maximum area change, 0.148 px displacement |

Zero-pass reconstruction preserves all labels, components, holes, and neighbor
pairs, with maximum area error below 4 × 10⁻⁷ relative. Thus the explicit-seed
reconstruction itself is faithful at this precision. The thin-twin failure with
the stronger Taubin weights is avoidable by reducing filter strength.

### Historical 1.2.0 limitations

1. **Native export is missing FESTIM's boundary tags.**
   `confMesh2dGMSH` creates physical surfaces, but no physical curves. Its native
   `.msh` imports with cell tags and no facet tags. The current FESTIM reader
   requires facets. Grain physical IDs are also assigned automatically rather
   than explicitly set to the original ID; the evaluation uses explicit IDs.
2. **Moving-average paths discard holes.**
   The enclosing grain's hole disappears in both moving-average and
   mean-coordinate reconstruction. On the simple island input this produces
   approximately 9.95% overlapping input-polygon area. The mesher can carve
   islands back out, but that does not repair all upstream geometry or
   grain-neighbor data. The void control is actually filled by both paths.
3. **Thin grains can change substantially.**
   The largest grain-area change in the thin-twin fixture was 8.26% for Taubin,
   41.87% for moving average, and 69.95% for mean coordinates.
4. **Historical automatic-seeding failures (1.2.0).**
   All seven synthetic fixtures fail before smoothing with
   `ValueError: A linearring requires at least 4 coordinates`, inside
   `GrainManifold2D._generate_clipped_polygons`. The real map constructs a mesh,
   but its input polygons leave 2.44% of the domain uncovered. This tests the
   automatic seed generator with jitter disabled, not every possible setting.
5. **Historical connectivity limitation (resolved locally).**
   The older `EbsdMicrostructure.from_mesh` midpoint query needed vertex-to-cell
   connectivity. The production method now creates it itself. The evaluation creates `0 -> mesh.topology.dim` before calling
   it; without that setup DOLFINx 0.11 raises a missing-connectivity error.

## What the evaluation adapter does

The standalone harness is `tools/evaluate_upxo.py`; it is not a public backend.

- Exact input polygons come from Rasterio, preserving interior rings and
  disconnected components. Collinear edges are subdivided at integer grid
  vertices so adjacent grains supply identical segment endpoints to UPXO.
  This avoids assuming that separately simplified polygon edges will be noded
  by the mesher. No labels, enclosed grains, or diagonal contacts are repaired.
- The three explicit-seed paths use one interior seed per pixel. Coordinates
  are just below pixel-centre half-integers, so UPXO's nearest-index lookup
  samples the intended pixel. The automatic-seed path uses its own generator.
  Main smoothing paths use five passes, `area_threshold=0`,
  `fix_diagonal=False`, `merge_enclosed=False`, and zero seeding jitter.
  Thus the smoothing results concern controlled reconstruction/smoothing,
  not a wholesale use of the defaults for Monte Carlo grain structures. The
  follow-up controls use zero passes or weaker Taubin weights, respectively.
  The seed sampling margin is 10⁻⁶ pixels; a regression test checks every seed
  against its original pixel, including the first row and column.
- All meshes use first-order triangles, Gmsh algorithm 6, boundary size
  0.75 pixels, bulk size 4.5 pixels, and one meshing thread. Quads are disabled.
- The adapter retains UPXO's triangles and their owners. It derives facets
  from triangle adjacency, groups them into arcs ending at junctions, and keeps
  separate closed loops between the same grain pair separate. Meshio/Gmsh
  re-export those triangles as MSH 4.1 with explicit grain and facet tags.
  It also writes the existing orientation/extent sidecars.
- Export scales both axes to metres. Synthetic inputs deliberately have
  unequal pixel spacings (1.6 × 0.8 µm) to exercise this.

The gates are: preserve IDs, components, holes, and neighbor pairs; no invalid
polygons; relative partition gap/overlap/excess below 10⁻⁷; maximum grain-area
change no more than 5%; maximum boundary displacement no more than one pixel;
positive triangle areas; conforming boundary segments; mesh-versus-input-polygon
area error below 10⁻⁷; and successful tagged DOLFINx reconstruction.
These are conservative evaluation thresholds, not user-specified production
requirements. The void fixture holds its original void geometry fixed, so even
Taubin's small void-boundary movement fails the partition gate.

## Reproduction and artifacts

The following version list describes the historical 1.2.0 run; the checkout
command selects the current 1.3.1 pin. The combined environment file is the
preferred installation for reproducing current mesh and solver validation.

UPXO was installed in `/tmp/upxo-eval-env`, separate from the existing Python
3.12 FESTIM environment. Relevant versions were Python 3.13.15, UPXO 1.2.0,
Gmsh 4.15.2, Shapely 2.1.2, Rasterio 1.5.2, NumPy 2.5.3, SciPy 1.18.1,
PyVista 0.49.0, and Meshio 5.3.5.

To reproduce on Linux with Conda and the existing FESTIM environment available:

```bash
git clone https://github.com/Design-By-Fundamentals-UKAEA/UPXO.git /tmp/upxo-review
git -C /tmp/upxo-review checkout a53885ef0a7a06f6b3fb195747363ae0a19bb127
conda create --prefix /tmp/upxo-eval-env -c conda-forge python=3.13 pip libglu -y
/tmp/upxo-eval-env/bin/python -m pip install "/tmp/upxo-review[ebsd]" \
  numpy==2.5.3 scipy==1.18.1 shapely==2.1.2 gmsh==4.15.2 \
  rasterio==1.5.2 meshio==5.3.5 pyvista==0.49.0 pytest

# Run prepare and verify with the existing FESTIM environment's Python.
FM_UPXO_PYTHON=/tmp/upxo-eval-env/bin/python python tools/evaluate_upxo.py prepare
/tmp/upxo-eval-env/bin/python tools/evaluate_upxo.py mesh --jobs 2 --timeout 180
/tmp/upxo-eval-env/bin/python tools/evaluate_upxo.py mesh \
  --methods raw taubin_gentle automatic_seeds
python tools/evaluate_upxo.py verify

/tmp/upxo-eval-env/bin/python -m pytest \
  /tmp/upxo-review/tests/meshing/test_confmesh2d_gmsh.py -q
/tmp/upxo-eval-env/bin/python -m pytest test/test_upxo_evaluation.py -q
```

The actual run borrowed OpenGL shared libraries from the existing Conda
environment via `LD_LIBRARY_PATH`, because the pip Gmsh wheel needs
`libGLU.so.1` even for headless meshing. Installing `libglu` in the isolated
environment avoids borrowing those libraries. Legacy `pyvoro`/TetGen and DefDAP
were not needed for this evaluation.

Generated artifacts are under ignored `examples/results/upxo-evaluation/`:

- `summary.json` and `summary.md`: every scenario, metrics, errors, and verdict.
- `environment.json`, `solver_environment.json`, `requirements-lock.txt`, and
  `input_provenance.json`: source revision, dependency versions, and CTF checksum.
- `<case>/<method>/`: native export, tagged SI export, expected mesh arrays,
  orientation/extent sidecars, `result.json`, and `run.log`.
- `ctf_map/<method>/overlay.png`: original CTF pixels with reconstructed polygons.
- `neper-topology-repair.json`: the independent comparison with the existing
  topology workaround; `project-tests.log` and `verify.log`: validation logs.

This is a serial evaluation of one measured crop and small synthetic fixtures.
It does not establish parallel import behavior, transport-solver equivalence,
performance superiority, or safe smoothing parameters for other scans.

## Original recommendation (superseded by full EBSD replacement)

Keep the existing segmentation, quality filtering, orientation algebra, and
raster diagnostics. Add an optional UPXO polygon-meshing backend with explicit
grain/facet tagging and topology-preserving polygon preparation. The exact
polygon route can retain enclosed grains and remove Neper's absorption workaround
for this backend. Controlled Taubin smoothing with explicit seeds is also a
credible candidate; gentler weights pass both the real-map and thin-twin checks.
Keep the geometry validation gates and avoid the moving-average paths and
automatic seeding until their observed failures are addressed. DefDAP import and
replacement of the existing CTF segmentation were not evaluated.

Resolve the previously identified Python-version and GPL-3.0/Apache-2.0
integration questions before making UPXO a package dependency. This evaluation
has not changed the project's declared dependencies or licensing.
