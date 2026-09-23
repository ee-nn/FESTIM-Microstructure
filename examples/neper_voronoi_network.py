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
from. The model itself is FESTIM's, declared as in the Voronoi examples.

Run::

    python examples/neper_voronoi_network.py
    mpirun -n 4 python examples/neper_voronoi_network.py
"""

from pathlib import Path
from typing import Any

import dolfinx
import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem

NETWORK_ID = 1_000_000  # above every grain id; a manifold shares the surface ids
SURFACE_ID_0 = 2_000_000  # the per-grain surface patches are numbered from here

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

theta_dependent_D = False  # D_GB above theta_c only, D_B below; CHECK before use
theta_c = 15.0  # degrees
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
# ``NeperMesh`` can be unpacked as an array-like object by static type
# checkers, although the runtime value is the DOLFINx mesh.
mesh_obj: Any = mesh
T = 500.0  # with E_D = 0 on both phases, nothing depends on it


def grain_network_problem(D_gb):
    """The transport model, as FESTIM declarations; see the Voronoi examples."""
    D_lattice, _ = fm.materials.crystal_diffusivity_field(poly, D_B)
    grains = fm.fem.subdomains.grain_subdomains(poly, F.Material(D=D_lattice))
    # One material for every boundary, or a diffusivity per tessellation face
    # from its disorientation; the network builds that field on its submesh.
    network = fm.fem.subdomains.grain_boundary_network(
        NETWORK_ID,
        poly,
        None if theta_dependent_D else F.Material(D_0=D_gb, E_D=0.0),
        diffusivity_by_entity=(
            np.where(neper.theta >= theta_c, D_gb, D_B) if theta_dependent_D else None
        ),
    )
    grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    species_of = dict(zip((g.id for g in grains), grain_species, strict=True))

    k = dolfinx.fem.Constant(mesh_obj, k_exchange)
    width = dolfinx.fem.Constant(mesh_obj, delta)
    sources, boundary_conditions = [], []
    for c_grain in grain_species:
        exchange = {"c_g": c_grain, "c_n": c_gb}
        sources.append(
            F.ParticleSource(
                value=lambda c_g, c_n: (k / width) * (c_g - c_n),
                species=c_gb,
                volume=network,
                species_dependent_value=exchange,
            )
        )
        boundary_conditions.append(
            F.ParticleFluxBC(
                subdomain=network,
                species=c_grain,
                value=lambda c_g, c_n: k * (c_n - c_g),
                species_dependent_value=exchange,
            )
        )

    charged = lambda x: np.isclose(x[2], L)  # noqa: E731
    patches, mouths = fm.fem.subdomains.grain_surfaces(
        mesh_obj, grains, charged, SURFACE_ID_0
    )
    boundary_conditions += [
        F.FixedConcentrationBC(subdomain=p, value=c0, species=species_of[p.grain_id])
        for p in patches
    ]
    boundary_conditions.append(
        F.FixedConcentrationBC(subdomain=mouths, value=c0, species=c_gb)
    )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh_obj),
        subdomains=[*grains, network, *patches, mouths],
        species=[*grain_species, c_gb],
        sources=sources,
        boundary_conditions=boundary_conditions,
        temperature=T,
        settings=F.Settings(
            atol=1e-14, rtol=1e-12, transient=True, final_time=t_end, stepsize=dt
        ),
    )
    model.initialise()
    fm.fem.solvers.tune_direct_solver(model)
    return model, grains, grain_species, network, c_gb


model, grains, grain_species, network, cgb_species = grain_network_problem(D_GB)
model.run()
fm.exports.averages.write_vtx(
    grains,
    grain_species,
    network,
    cgb_species,
    str(neper.base.parent / "neper"),
    time=t_end,
)
cgb_fast = cgb_species.subdomain_to_post_processing_solution[network]

# Inspect the mesh and boundary network
print(neper.report())
area_mesh, area_tess = (
    fm.exports.measures.submesh_measure(network),
    neper.network_measure,
)
n_comp = fm.exports.measures.component_count(network)
n_mesh_cells = mesh_obj.topology.index_map(mesh_obj.topology.dim).size_global
print(f"  mesh                            : {n_mesh_cells} cells")
print(
    f"  network captured by the submesh : {area_mesh:.4f} of {area_tess:.4f}"
    f" ({100 * area_mesh / area_tess:.2f} %)"
)
if n_comp is not None:
    print(f"  connected components            : {n_comp}")
print(f"  interior facets                 : {model.manifold_is_interior(network)}")

# Analyse grain-boundary transport
fast = fm.exports.averages.inventory(grains, grain_species, network, cgb_species, delta)
depth = neper.junction_only_below(L)
gb_z = cgb_fast.function_space.tabulate_dof_coordinates()[:, 2]
deep = gb_z < depth
c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

# A second problem with D_gb = D_b: a transient problem cannot be re-run
# from t = 0 in place.
reference, ref_grains, ref_species, ref_network, ref_cgb = grain_network_problem(D_B)
reference.run()
ref = fm.exports.averages.inventory(
    ref_grains, ref_species, ref_network, ref_cgb, delta
)

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
resistance = fm.materials.interface_resistance_ratio(
    k_exchange, L * n_cells ** (-1 / 3), D_B
)
print(f"  interface / lattice resistance : {resistance:.1e}  (Fisher is the 0 limit)")

c_grain_deep = fm.exports.averages.lattice_mean_below(
    grains, grain_species, axis=2, depth=depth
)
print("\njunction transport: no boundary touching the charged face reaches below")
print(f"z = {depth:.3f}, so everything the network holds there has")
print("crossed at least one triple line.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
