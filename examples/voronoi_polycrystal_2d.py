"""Short-circuit diffusion through the grain-boundary network of a Voronoi polycrystal.

The point of the example is that the entire network is declared as **one** codim-1
subdomain with **one** species. That is what makes triple junctions work: the submesh
built from all the grain-boundary facets is topologically connected, so a single
continuous field lives on it and hydrogen crosses from one boundary to another with no
junction condition to write. The grains each carry a lattice field of their own and
exchange with that network at the rate ``k``.

The model is FESTIM's: this package supplies the tagged grains, the network, the
per-grain surface patches and the post-processing; everything from the species to
the ``HydrogenTransportProblemDiscontinuous`` is declared below the way FESTIM's
manifold documentation shows.

Run::

    python examples/voronoi_polycrystal_2d.py
"""

from pathlib import Path

import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_ID = 1_000_000  # above every grain id; a manifold shares the surface ids
SURFACE_ID_0 = 2_000_000  # the per-grain surface patches are numbered from here

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
seeds, boundaries = fm.voronoi.tessellate(
    n_seeds, L, np.random.default_rng(seed), dim=2
)
mesh_data = fm.voronoi.build_mesh(
    boundaries, L, fm.MeshSizing(h_gb=h_gb, h_bulk=h_bulk), dim=2, seeds=seeds
)
# Built here rather than through VoronoiMicrostructure.create, which would
# mesh the cell to its own sizing: this keeps the mesh the example asked for.
# Untextured, and crystal_anisotropy is 1 below, so the angles never enter.
micro = fm.VoronoiMicrostructure(
    dim=2,
    size=L,
    n_seeds=n_seeds,
    seed=seed,
    seeds=seeds,
    boundaries=boundaries,
    mesh=mesh_data.mesh,
    cell_tags=mesh_data.cell_tags,
    facet_tags=mesh_data.facet_tags,
    gb_tag=mesh_data.gb_tag,
    grain_ids=mesh_data.grain_ids,
    orientations=np.zeros(len(mesh_data.grain_ids)),
    h_gb=h_gb,
)
mesh, n_grains = micro.mesh, micro.n_grains
tdim = mesh.topology.dim


def grain_network_problem(D_gb):
    """The transport model, as FESTIM declarations.

    Every grain is a ``VolumeSubdomain`` with a ``Species`` of its own, the
    network is one codim-1 ``VolumeSubdomain`` with one species, and each grain
    exchanges ``k (c_grain - c_gb)`` with the boundary slab: a ``ParticleFluxBC``
    on the grain and, since the network equation is written per unit slab width,
    a ``ParticleSource`` of ``k / delta`` times the same jump on the network. The
    charged surface is split into what each grain owns, plus the network's
    mouths on it.
    """
    # Coefficients: the lattice tensor field (isotropic here, one tensor per
    # grain) and an ordinary FESTIM material for the boundaries. E_D = 0 on
    # both, since the Arrhenius factor is already in D_B and D_GB.
    D_lattice, _ = fm.materials.crystal_diffusivity_field(micro, D_B)
    grains = fm.fem.subdomains.grain_subdomains(micro, F.Material(D=D_lattice))
    network = fm.fem.subdomains.grain_boundary_network(
        NETWORK_ID, micro, F.Material(D_0=D_gb, E_D=0.0)
    )
    grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    species_of = dict(zip((g.id for g in grains), grain_species, strict=True))

    sources, boundary_conditions = [], []
    for c_grain in grain_species:
        exchange = {"c_g": c_grain, "c_n": c_gb}
        sources.append(
            F.ParticleSource(
                value=lambda c_g, c_n: (k_exchange / delta) * (c_g - c_n),
                species=c_gb,
                volume=network,
                species_dependent_value=exchange,
            )
        )
        boundary_conditions.append(
            F.ParticleFluxBC(
                subdomain=network,
                species=c_grain,
                value=lambda c_g, c_n: k_exchange * (c_n - c_g),
                species_dependent_value=exchange,
            )
        )

    charged = lambda x: np.isclose(x[1], L)  # noqa: E731
    patches, mouths = fm.fem.subdomains.grain_surfaces(
        mesh, grains, charged, SURFACE_ID_0
    )
    boundary_conditions += [
        F.FixedConcentrationBC(subdomain=p, value=c0, species=species_of[p.grain_id])
        for p in patches
    ]
    boundary_conditions.append(
        F.FixedConcentrationBC(subdomain=mouths, value=c0, species=c_gb)
    )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        subdomains=[*grains, network, *patches, mouths],
        species=[*grain_species, c_gb],
        sources=sources,
        boundary_conditions=boundary_conditions,
        temperature=T,
        settings=F.Settings(
            atol=1e-8, rtol=1e-6, transient=True, final_time=t_end, stepsize=dt
        ),
    )
    model.initialise()
    fm.fem.solvers.tune_direct_solver(model)  # MUMPS workspace, see its docstring
    return model, grains, grain_species, network, c_gb


# Build and solve
model, grains, grain_species, network, cgb_species = grain_network_problem(D_GB)
model.run()
fm.exports.averages.write_vtx(
    grains, grain_species, network, cgb_species, OUTPUT_DIR / "voronoi", time=t_end
)
cgb_fast = cgb_species.subdomain_to_post_processing_solution[network]

# Inspect the mesh and boundary network
ridge_length = fm.voronoi.network_measure(boundaries, dim=2)
facet_length = fm.exports.measures.submesh_measure(network)
print(
    f"microstructure: {n_seeds} seeds, {n_grains} grains, "
    f"{len(boundaries)} boundary segments"
)
topology = fm.voronoi.junctions(boundaries, dim=2, size=L)
print(f"triple junctions inside box: {len(topology.points)}")
print(
    "connected components (ridges): "
    f"{fm.voronoi.connected_components(boundaries, dim=2, size=L)}"
)
print(f"connected components (submesh): {fm.exports.measures.component_count(network)}")
print(f"  facets dropped on the outer box : {network.n_dropped}")
n_cells = mesh.topology.index_map(2).size_global
print(f"  mesh                            : {n_cells} cells")
print(
    f"  network captured by the submesh : {facet_length:.4e} of {ridge_length:.4e}"
    f" ({100 * facet_length / ridge_length:.2f} %)"
)
print(f"  interior facets                 : {model.manifold_is_interior(network)}")

# Compare fast boundaries with lattice diffusion
fast = fm.exports.averages.inventory(grains, grain_species, network, cgb_species, delta)
depth = fm.exports.measures.junction_only_below(
    [np.array(boundary) for boundary in boundaries], axis=1, top=L
)
gb_y = cgb_fast.function_space.tabulate_dof_coordinates()[:, 1]
deep = gb_y < depth
c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

# The reference is a second problem with D_gb = D_b: a transient problem
# cannot be re-run from t = 0 in place.
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
f_gb = delta * ridge_length / L**2
print(f"  boundary area fraction f       : {f_gb:.3e}")
print(
    f"  Hart bound f D_gb + (1-f) D_b  : "
    f"{fm.materials.hart_bound(f_gb, D_GB, D_B):.3e}"
    f"  (vs D_b = {D_B:.3e})"
)
beta = fm.materials.beta_parameter(delta, D_GB, D_B, t_end)
print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")
resistance = fm.materials.interface_resistance_ratio(
    k_exchange, L / np.sqrt(n_grains), D_B
)
print(f"  interface / lattice resistance : {resistance:.1e}  (Fisher is the 0 limit)")

c_grain_deep = fm.exports.averages.lattice_mean_below(
    grains, grain_species, axis=1, depth=depth
)
print("\njunction transport: no boundary touching the charged surface reaches below")
print(f"y = {depth:.3e}, so everything the network holds there has crossed")
print("at least one triple junction.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
if c_grain_deep > 1e-12 * c0:
    print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
else:
    print("  ratio                          : n/a - nothing has reached this depth")
