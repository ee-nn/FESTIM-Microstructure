"""Short-circuit diffusion through the grain-boundary network of a 3D Voronoi
polycrystal.

Same idea as the 2D example: the entire network is declared as **one** codim-1
subdomain with **one** species. In 3D "codim-1" means the network is made of
*facets* (triangles) rather than edges, so ``dim=2`` in a ``tdim == 3`` mesh, and the
junctions are triple *lines* (edges shared by three faces) plus quadruple *points*.
The submesh is connected through those shared edges exactly as the 2D one was
connected through shared vertices, so nothing about the FESTIM setup changes.

What does change relative to the 2D script:

* A Voronoi ridge in 3D is a convex *polygon*, not a segment, so Liang-Barsky is
  replaced by Sutherland-Hodgman clipping against the six half-spaces of the box.
* The network is tagged in gmsh with a physical group and picked up from the facet
  tags instead of being located geometrically.
  The geometric route still works (``locate_geometrically=True`` uses
  :func:`near_faces`) but a point-to-polygon distance is much more delicate than a
  point-to-segment distance, and the fragment operation already knows exactly which
  facets are grain boundaries.
* The mouths are the curves where the network meets the charged face, which
  :func:`~festim_microstructure.resolved.build` derives from the mesh dimension.
* The grain-boundary volume fraction is delta * area / L**3 instead of
  delta * length / L**2.

Run::

    python examples/voronoi_polycrystal_3d.py
"""

from pathlib import Path

import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
L = 1.0  # specimen size
n_seeds = 12  # number of grains
seed = 3

D_B = 1e-3  # lattice diffusivity
D_GB = 30.0  # grain-boundary diffusivity
delta = 1e-3  # grain-boundary width
k_exchange = 1.0  # bulk <-> grain-boundary exchange
c0 = 1.0
t_end = 3.0
dt = 0.05

# Mesh sizing
h_gb = 0.05  # mesh size at the boundaries
h_bulk = 0.15  # mesh size in the grain interiors
locate_geometrically = False  # use near_faces instead of the facet tags

# Microstructure
faces = fm.voronoi.voronoi_faces(n_seeds, L, np.random.default_rng(seed))
mesh, facet_tags = fm.voronoi.build_mesh_3d(
    faces, L, fm.MeshSizing(h_gb=h_gb, h_bulk=h_bulk)
)
# voronoi_faces draws its seeds from the generator before anything else, so
# the same seed reproduces them; each cell then takes the id of its nearest
# seed image, which is what makes the grains separable subdomains.
seeds = np.random.default_rng(seed).uniform(0, L, (n_seeds, 3))
cell_tags, grain_ids = fm.voronoi.grain_tags_from_seeds(mesh, seeds, L)
tagged = not locate_geometrically
micro = fm.VoronoiMicrostructure3D(
    size=L,
    n_seeds=n_seeds,
    seed=seed,
    seeds=seeds,
    faces=faces,
    mesh=mesh,
    cell_tags=cell_tags,
    facet_tags=facet_tags if tagged else None,
    gb_tag=fm.voronoi.GB_TAG_3D if tagged else None,
    grain_ids=grain_ids,
    # ids come from the seed images and need not be contiguous, so the
    # orientations are indexed by the largest of them. Untextured here.
    orientations=np.zeros(int(grain_ids.max())),
    h_gb=h_gb,
)
physics = fm.Physics(
    T=500.0,  # with E_D = 0 on both phases, nothing depends on it
    D_0_bulk=D_B,
    E_D_bulk=0.0,
    D_0_gb=D_GB,
    E_D_gb=0.0,
    delta=delta,
    k_exchange=k_exchange,
    crystal_anisotropy=1.0,
)
bcs = [("charged", lambda x: np.isclose(x[2], L), c0)]
solve = fm.SolveOptions(
    transient=True, final_time=t_end, stepsize=dt, atol=1e-14, rtol=1e-12
)

# Build and solve
model = fm.build(micro, physics, bcs, solve=solve).run()
fm.exports.averages.write_vtx(model, OUTPUT_DIR / "voronoi3d", time=t_end)
network, cgb_fast = model.network, model.network_solution

# Inspect the mesh and boundary network
lines, triple_length, quadruple = fm.voronoi.triple_lines(faces)
face_area = sum(fm.voronoi.polygon_area(poly) for poly in faces)
sub_area = fm.exports.measures.submesh_measure(network)
print(f"microstructure: {n_seeds} seeds, {len(faces)} boundary polygons")
print(f"  triple lines                    : {len(lines)} (length {triple_length:.3f})")
print(f"  quadruple points                : {len(quadruple)}")
print(
    f"  connected components (faces)    : {fm.voronoi.connected_components_3d(faces)}"
)
print(
    f"  connected components (submesh)  : "
    f"{fm.exports.measures.component_count(network)}"
)
n_cells = mesh.topology.index_map(3).size_global
print(f"  mesh                            : {n_cells} cells")
print(f"  grains (tagged pieces)          : {micro.n_grains}")
print(
    f"  network captured by the submesh : {sub_area:.4f} of {face_area:.4f}"
    f" ({100 * sub_area / face_area:.2f} %)"
)
print(
    f"  interior facets                 : {model.model.manifold_is_interior(network)}"
)

# Compare fast boundaries with lattice diffusion
fast = fm.exports.averages.inventory(model)
depth = fm.exports.measures.junction_only_below(faces, axis=2, top=L)
gb_z = cgb_fast.function_space.tabulate_dof_coordinates()[:, 2]
deep = gb_z < depth
c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

# A second build, not model.set_physics(...): a transient problem cannot be
# re-run from t = 0 in place.
reference_physics = fm.Physics(
    T=500.0,
    D_0_bulk=D_B,
    E_D_bulk=0.0,
    D_0_gb=D_B,
    E_D_gb=0.0,
    delta=delta,
    k_exchange=k_exchange,
    crystal_anisotropy=1.0,
)
reference = fm.build(micro, reference_physics, bcs, solve=solve).run()
ref = fm.exports.averages.inventory(reference)

print(
    f"\nafter t = {t_end} (lattice diffusion alone reaches "
    f"~{2 * np.sqrt(D_B * t_end):.3g})"
)
print(f"  inventory with fast boundaries : {fast:.4e}")
print(f"  inventory with D_gb = D_b      : {ref:.4e}")
print(f"  enhancement                    : x {fast / ref:.1f}")
f_gb = delta * face_area / L**3
print(f"  boundary volume fraction f     : {f_gb:.3e}")
print(
    f"  Hart bound f D_gb + (1-f) D_b  : "
    f"{fm.materials.hart_bound(f_gb, D_GB, D_B):.3e}"
    f"  (vs D_b = {D_B:.3e})"
)
beta = fm.materials.beta_parameter(delta, D_GB, D_B, t_end)
print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")
print(
    f"  interface / lattice resistance : "
    f"{physics.interface_resistance_ratio(L * n_seeds ** (-1 / 3)):.1e}"
    f"  (Fisher is the 0 limit)"
)

c_grain_deep = fm.exports.averages.lattice_mean_below(model, axis=2, depth=depth)
print("\njunction transport: no boundary touching the charged face reaches below")
print(f"z = {depth:.3f}, so everything the network holds there has")
print("crossed at least one triple line.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
