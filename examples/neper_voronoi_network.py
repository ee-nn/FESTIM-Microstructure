"""Short-circuit diffusion through the grain-boundary network of a Neper
polycrystal.

Same formulation as the in-situ Voronoi version: the whole network is **one**
codim-1 subdomain carrying **one** species, and every grain carries a lattice
species of its own. What changes is only where the microstructure comes from --
see :mod:`festim_microstructure.meshing.neper` for what Neper adds (``domface``,
``theta``, exact junction topology) and for how the binaries are found
(``FM_NEPER_BIN`` / ``FM_GMSH_BIN``, or ``PATH``).

Neper tags every tessellation face separately, so the network is the *list* of
face ids above the disorientation threshold rather than one marker; that list is
what ``gb_tag`` carries here. Constructing the :class:`fm.NeperMesh` runs Neper
and reads what it wrote, and unpacks into the four things the model is built
from.

Run::

    python examples/neper_voronoi_network.py
    mpirun -n 4 python examples/neper_voronoi_network.py
"""

from pathlib import Path
from typing import Any

import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem

# Simulation parameters
L = 1.0  # specimen size (Neper's default domain is cube(1,1,1))
n_cells = 100  # number of grains; Neper is happy into the 1e5 range
seed = 1  # -id, the rng seed, so the microstructure is reproducible

D_B = 1e-3
D_GB = 30.0
delta = 1e-3
k_exchange = 1.0
c0 = 1.0
t_end = 3.0
dt = 0.05

theta_dependent_D = False  # see gb_diffusivity_field, CHECK before enabling
settings = fm.NeperSettings(
    stem="poly",
    workdir=OUTPUT_DIR,
    force=False,
    theta_min=0.0,  # keep only boundaries above this disorientation (deg)
)

# Build the microstructure and simulation
neper = fm.NeperMesh(n_cells, seed, settings)
mesh, cell_tags, facet_tags, network_ids = neper

# The tessellation knows the topology; this is the same polycrystal as the
# model sees it: one tagged subdomain per grain, the network as face ids.
# Untextured, and crystal_anisotropy is 1 below, so the angles never enter.

poly = fm.TaggedPolycrystal(
    mesh=mesh,
    cell_tags=cell_tags,
    facet_tags=facet_tags,
    gb_tag=network_ids,
    name=f"neper {n_cells} cells, seed {seed}",
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

model = fm.build(poly, physics, bcs, solve=solve)
if theta_dependent_D:
    # the submesh has to exist before a field can live on it, so initialise
    # once, swap the material in, and initialise again
    model.initialise()
    model.network.material = F.Material(
        D_0=fm.materials.gb_diffusivity_field(model.network, neper.theta, D_B, D_GB),
        E_D=0.0,
    )
    model.initialise(force=True)
model.run()
fm.exports.averages.write_vtx(model, str(neper.base.parent / "neper"), time=t_end)
network, cgb_fast = model.network, model.network_solution

# Inspect the mesh and boundary network
print(neper.report())
area_mesh, area_tess = (
    fm.exports.measures.submesh_measure(network),
    neper.network_measure,
)
n_comp = fm.exports.measures.component_count(network)
# ``NeperMesh`` can be unpacked as an array-like object by static type
# checkers, although the runtime value is the DOLFINx mesh.
mesh_obj: Any = mesh
n_mesh_cells = mesh_obj.topology.index_map(mesh_obj.topology.dim).size_global
print(f"  mesh                            : {n_mesh_cells} cells")
print(
    f"  network captured by the submesh : {area_mesh:.4f} of {area_tess:.4f}"
    f" ({100 * area_mesh / area_tess:.2f} %)"
)
if n_comp is not None:
    print(f"  connected components            : {n_comp}")
print(
    f"  interior facets                 : {model.model.manifold_is_interior(network)}"
)

# Analyse grain-boundary transport
fast = fm.exports.averages.inventory(model)
depth = neper.junction_only_below(L)
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
reference = fm.build(poly, reference_physics, bcs, solve=solve).run()
ref = fm.exports.averages.inventory(reference)

print(
    f"\nafter t = {t_end} (lattice diffusion alone reaches "
    f"~{2 * np.sqrt(D_B * t_end):.3g})"
)
print(f"  inventory with fast boundaries : {fast:.4e}")
print(f"  inventory with D_gb = D_b      : {ref:.4e}")
print(f"  enhancement                    : x {fast / ref:.1f}")
f_gb = delta * area_tess / L**3
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
    f"{physics.interface_resistance_ratio(L * n_cells ** (-1 / 3)):.1e}"
    f"  (Fisher is the 0 limit)"
)

c_grain_deep = fm.exports.averages.lattice_mean_below(model, axis=2, depth=depth)
print("\njunction transport: no boundary touching the charged face reaches below")
print(f"z = {depth:.3f}, so everything the network holds there has")
print("crossed at least one triple line.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
