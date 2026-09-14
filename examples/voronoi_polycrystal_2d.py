"""Short-circuit diffusion through the grain-boundary network of a Voronoi polycrystal.

The point of the example is that the entire network is declared as **one** codim-1
subdomain with **one** species. That is what makes triple junctions work: the submesh
built from all the grain-boundary facets is topologically connected, so a single
continuous field lives on it and hydrogen crosses from one boundary to another with no
junction condition to write. The grains each carry a lattice field of their own and
exchange with that network at the rate ``k``; see
:mod:`festim_microstructure.resolved`.

Run::

    python examples/voronoi_polycrystal_2d.py
"""

from pathlib import Path

import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
L = 1.0e-4  # specimen size, 100 microns
n_seeds = 12  # number of grains
seed = 3  # rng seed, so the microstructure is reproducible

E_M_BULK = 0.20  # eV, DFT, Diaz-Rodriguez Fig. 3
E_M_GB = 0.12  # eV, DFT, Diaz-Rodriguez Fig. 3
D0_B = 1.9e-7  # m^2/s, assumed
T = 500.0  # K
delta = 1e-9  # grain-boundary width
k_exchange = 3.0  # bulk <-> grain-boundary exchange, m/s
c0 = 5.6e27  # surface concentration
t_end = 1.0
dt = 1e-3

# Diffusivities at the simulation temperature
D0_GB = 1.5 * D0_B  # 2D vs 3D random-walk prefactor
D_B = D0_B * np.exp(-E_M_BULK / (F.k_B * T))
D_GB = D0_GB * np.exp(-E_M_GB / (F.k_B * T))

# Mesh sizing
h_gb = 0.004 * L  # mesh size at the boundaries
h_bulk = 0.04 * L  # mesh size in the grain interiors

# Microstructure
segments = fm.voronoi.voronoi_segments(n_seeds, L, np.random.default_rng(seed))
mesh, cell_tags, n_grains = fm.voronoi.build_mesh(
    segments, L, fm.MeshSizing(h_gb=h_gb, h_bulk=h_bulk)
)
# Built here rather than through VoronoiMicrostructure.create, which would
# mesh the cell to its own sizing: this keeps the mesh the example asked for.
# Untextured, and crystal_anisotropy is 1 below, so the angles never enter.
micro = fm.VoronoiMicrostructure(
    size=L,
    n_seeds=n_seeds,
    seed=seed,
    segments=segments,
    mesh=mesh,
    cell_tags=cell_tags,
    grain_ids=np.arange(1, n_grains + 1),
    orientations=np.zeros(n_grains),
    h_gb=h_gb,
)
physics = fm.Physics(
    T=T,
    D_0_bulk=D_B,  # E_D = 0: the Arrhenius factor is already included above
    E_D_bulk=0.0,
    D_0_gb=D_GB,
    E_D_gb=0.0,
    delta=delta,
    k_exchange=k_exchange,
    crystal_anisotropy=1.0,
)
bcs = [("charged", lambda x: np.isclose(x[1], L), c0)]
solve = fm.SolveOptions(
    transient=True, final_time=t_end, stepsize=dt, atol=1e-8, rtol=1e-6
)

# Build and solve
model = fm.build(micro, physics, bcs, solve=solve).run()
fm.exports.averages.write_vtx(model, OUTPUT_DIR / "voronoi", time=t_end)
network, cgb_fast = model.network, model.network_solution

# Inspect the mesh and boundary network
ridge_length = sum(float(np.linalg.norm(q - p)) for p, q in segments)
facet_length = fm.exports.measures.submesh_measure(network)
print(
    f"microstructure: {n_seeds} seeds, {n_grains} grains, "
    f"{len(segments)} boundary segments"
)
print(f"triple junctions inside box: {len(fm.voronoi.triple_junctions(segments, L))}")
print(f"connected components (ridges): {fm.voronoi.connected_components(segments, L)}")
print(f"connected components (submesh): {fm.exports.measures.component_count(network)}")
print(f"  facets dropped on the outer box : {network.n_dropped}")
n_cells = mesh.topology.index_map(2).size_global
print(f"  mesh                            : {n_cells} cells")
print(
    f"  network captured by the submesh : {facet_length:.4e} of {ridge_length:.4e}"
    f" ({100 * facet_length / ridge_length:.2f} %)"
)
print(
    f"  interior facets                 : {model.model.manifold_is_interior(network)}"
)

# Compare fast boundaries with lattice diffusion
fast = fm.exports.averages.inventory(model)
depth = fm.exports.measures.junction_only_below(
    [np.array(seg) for seg in segments], axis=1, top=L
)
gb_y = cgb_fast.function_space.tabulate_dof_coordinates()[:, 1]
deep = gb_y < depth
c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

# The reference is a second build rather than model.set_physics(...): a
# transient problem cannot be re-run from t = 0 in place.
reference_physics = fm.Physics(
    T=T,
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
f_gb = delta * ridge_length / L**2
print(f"  boundary area fraction f       : {f_gb:.3e}")
print(
    f"  Hart bound f D_gb + (1-f) D_b  : "
    f"{fm.materials.hart_bound(f_gb, D_GB, D_B):.3e}"
    f"  (vs D_b = {D_B:.3e})"
)
beta = fm.materials.beta_parameter(delta, D_GB, D_B, t_end)
print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")
print(
    f"  interface / lattice resistance : "
    f"{physics.interface_resistance_ratio(L / np.sqrt(n_grains)):.1e}"
    f"  (Fisher is the 0 limit)"
)

c_grain_deep = fm.exports.averages.lattice_mean_below(model, axis=1, depth=depth)
print("\njunction transport: no boundary touching the charged surface reaches below")
print(f"y = {depth:.3e}, so everything the network holds there has crossed")
print("at least one triple junction.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
if c_grain_deep > 1e-12 * c0:
    print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
else:
    print("  ratio                          : n/a - nothing has reached this depth")
