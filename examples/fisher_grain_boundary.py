"""Compare a codimension-one Fisher GB model with the Whipple/Le Claire result.

One boundary at ``x = 0`` between two grains, with a charged surface at
``y = 0``. The cell is symmetric about the boundary, so the right-hand grain is
exactly the half-cell the Whipple/Le Claire analysis is written for, and the
left-hand grain is what supplies the boundary slab's second face. The slab
therefore receives ``k (c_1 - c_gb) + k (c_2 - c_gb)``, which is the ``2 k`` of
Fisher's equation, without either side of it being assumed.

The model is FESTIM's, declared as in the Voronoi examples; here the "network"
is the single interior mesh line at ``x = 0``, located geometrically.
"""

from pathlib import Path

from mpi4py import MPI

import dolfinx
import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT, RIGHT = 1, 2  # grain tags either side of the boundary at x = 0
NETWORK_ID = 1_000_000  # above every grain id; a manifold shares the surface ids
SURFACE_ID_0 = 2_000_000  # the per-grain surface patches are numbered from here


def eval_on(fn, points):
    """Evaluate a fenics Function at a list of (x, y, 0) points."""
    fn_mesh = fn.function_space.mesh
    tree = dolfinx.geometry.bb_tree(fn_mesh, fn_mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(fn_mesh, candidates, points)
    cells = []
    for i, point in enumerate(points):
        hits = colliding.links(i)
        if not len(hits):
            raise ValueError(
                f"sample point {i} at {point} lies outside the function mesh"
            )
        cells.append(hits[0])
    return fn.eval(points, cells).reshape(-1)


def two_grains(mesh):
    """Tag the cells left of the boundary ``LEFT`` and those right of it
    ``RIGHT``. Every cell lies wholly on one side: ``x = 0`` is a mesh line."""
    tdim = mesh.topology.dim
    index_map = mesh.topology.index_map(tdim)
    cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    values = np.where(midpoints[:, 0] < 0.0, LEFT, RIGHT).astype(np.int32)
    return dolfinx.mesh.meshtags(mesh, tdim, cells, values)


# Simulation parameters
E_M_BULK = 0.20  # eV, DFT, Diaz-Rodriguez Fig. 3
E_M_GB = 0.12  # eV, DFT, Diaz-Rodriguez Fig. 3
D0_B = 1.9e-7  # m^2/s, assumed
T = 500.0  # K
delta = 1e-9  # grain-boundary width
k_exchange = 1.0  # bulk <-> grain-boundary exchange coefficient
c0 = 5.6e27  # surface concentration
LX = 0.1 / 1e4  # half grain width
LY = 1.5 / 1e4  # depth
NX = 100
NY = 150
t_end = 1.0
dt = 1e-3

# Diffusivities at the simulation temperature
D_B = D0_B * np.exp(-E_M_BULK / (F.k_B * T))
D_GB = 1.5 * D0_B * np.exp(-E_M_GB / (F.k_B * T))

# Build the microstructure and simulation
# The full cell, [-LX, LX] x [0, LY]. Its right half is meshed exactly as
# the half-cell was, so the profiles below are the same numbers.
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [np.array([-LX, 0.0]), np.array([LX, LY])], [2 * NX, NY]
)
micro = fm.TaggedPolycrystal(
    mesh=mesh,
    cell_tags=two_grains(mesh),
    grain_ids=np.array([LEFT, RIGHT]),
    tolerance=0.05 * LX / NX,
    # A line subdomain carries the GB transport equation; here it is the
    # single interior mesh line at x = 0.
    network_locator=lambda x: np.isclose(x[0], 0.0, atol=1e-11),
    name="one boundary between two grains",
)

# The transport model, as FESTIM declarations. E_D = 0 on both phases: the
# Arrhenius factor is already included above.
D_lattice, _ = fm.materials.crystal_diffusivity_field(micro, D_B)
grains = fm.fem.subdomains.grain_subdomains(micro, F.Material(D=D_lattice))
network = fm.fem.subdomains.grain_boundary_network(
    NETWORK_ID, micro, F.Material(D_0=D_GB, E_D=0.0)
)
grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
cgb_species = F.Species("c_gb", subdomains=[network])
species_of = dict(zip((g.id for g in grains), grain_species, strict=True))

# Large ``k`` approaches Fisher's local-equilibrium assumption.
k = dolfinx.fem.Constant(mesh, k_exchange)
width = dolfinx.fem.Constant(mesh, delta)
sources, boundary_conditions = [], []
for c_grain in grain_species:
    exchange = {"c_g": c_grain, "c_n": cgb_species}
    sources.append(
        F.ParticleSource(
            value=lambda c_g, c_n: (k / width) * (c_g - c_n),
            species=cgb_species,
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

charged = lambda x: np.isclose(x[1], 0.0, atol=1e-11)  # noqa: E731
patches, mouths = fm.fem.subdomains.grain_surfaces(mesh, grains, charged, SURFACE_ID_0)
boundary_conditions += [
    F.FixedConcentrationBC(subdomain=p, value=c0, species=species_of[p.grain_id])
    for p in patches
]
boundary_conditions.append(
    F.FixedConcentrationBC(subdomain=mouths, value=c0, species=cgb_species)
)

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(mesh),
    subdomains=[*grains, network, *patches, mouths],
    species=[*grain_species, cgb_species],
    sources=sources,
    boundary_conditions=boundary_conditions,
    temperature=T,
    settings=F.Settings(
        atol=1e-8, rtol=1e-6, transient=True, final_time=t_end, stepsize=dt
    ),
)
model.initialise()
fm.fem.solvers.tune_direct_solver(model)
model.run()
fm.exports.averages.write_vtx(
    grains, grain_species, network, cgb_species, OUTPUT_DIR / "fisher", time=t_end
)

cb_fn = species_of[RIGHT].subdomain_to_post_processing_solution[grains[RIGHT - 1]]
cg_fn = cgb_species.subdomain_to_post_processing_solution[network]

# Analysis
order = np.argsort(cg_fn.function_space.tabulate_dof_coordinates()[:, 1])
y_gb = cg_fn.function_space.tabulate_dof_coordinates()[order, 1]
c_gb_vals = cg_fn.x.array[order]

bulk_only = 2 * np.sqrt(D_B * t_end)
print(f"lattice diffusion alone would reach ~{bulk_only:.3g}")
print("grain boundary profile:")
for depth_1e4 in (0.0, 0.25, 0.5, 0.75, 1.0):
    y = depth_1e4 / 1e4
    print(
        f"   y = {depth_1e4:.2f} x 10^-4 m   "
        f"c_gb = {np.interp(y, y_gb, c_gb_vals):.4e}"
    )

# Check the local-equilibrium approximation.
probe = np.linspace(0.05 / 1e4, 1.0 / 1e4, 20)
# Evaluate just inside the right-grain submesh at x = 0 and x = LX.
eps_x = 1e-6 * LX / NX
pts = np.column_stack([np.full_like(probe, eps_x), probe, np.zeros_like(probe)])
ratio = eval_on(cb_fn, pts) / np.interp(probe, y_gb, c_gb_vals)
print(f"\nlocal equilibrium c_b(0,y)/c_gb(y): {ratio.min():.4f} .. {ratio.max():.4f}")

# Whipple/Le Claire type-B fit: ln(cbar) vs. y**(6/5). One grain and half
# the slab, which is the half-cell the closed form is written for.
depths = np.linspace(0.15, 1.0, 25) / 1e4  # metres, within [0, LY]
xs = np.linspace(0.0, LX, 200)
sample_xs = np.clip(xs, eps_x, LX - eps_x)
cbar = np.array(
    [
        (
            np.trapezoid(
                eval_on(
                    cb_fn,
                    np.column_stack(
                        [sample_xs, np.full_like(xs, y), np.zeros_like(xs)]
                    ),
                ),
                xs,
            )
            + (delta / 2) * np.interp(y, y_gb, c_gb_vals)
        )
        / (LX + delta / 2)
        for y in depths
    ]
)

slope, intercept = np.polyfit(depths**1.2, np.log(cbar), 1)
residual = np.abs(np.log(cbar) - (slope * depths**1.2 + intercept)).max()
recovered = 1.322 * np.sqrt(D_B / t_end) * (-slope) ** (-5.0 / 3.0)

alpha = delta / (2 * np.sqrt(D_B * t_end))
beta = fm.materials.beta_parameter(delta, D_GB, D_B, t_end)
print(
    f"\nLe Claire analysis (valid for alpha << 1, beta >> 1: "
    f"alpha = {alpha:.3f}, beta = {beta:.0f})"
)
print(f"   d ln(cbar) / d y**(6/5) = {slope:.4f}  (max fit residual {residual:.3f})")
print(f"   recovered delta*D_gb    = {recovered:.4e}")
print(f"   input     delta*D_gb    = {delta * D_GB:.4e}")
error = 100 * abs(recovered / (delta * D_GB) - 1)
print(f"   error                   = {error:.1f} %")
