"""Short-circuit diffusion through the grain-boundary network of a 3D Voronoi
polycrystal.

Runs the equiaxed case plus two aspect=3 cases elongated parallel (z) and
perpendicular (x) to hydrogen entry from z=L. The elongated cases use the same
tessellation with its axes exchanged. Each case writes separate grain and
network BP time series under examples/results/voronoi_polycrystal_3d/, including
the initial state and every solved timestep.

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
  :func:`festim_microstructure.voronoi.near`) but point-to-polygon distance is
  more delicate than a
  point-to-segment distance, and the fragment operation already knows exactly which
  facets are grain boundaries.
* The mouths are the curves where the network meets the charged face, which
  :func:`~festim_microstructure.fem.subdomains.grain_surfaces` derives from the
  mesh dimension.
* The grain-boundary volume fraction is delta * area / L**3 instead of
  delta * length / L**2.

Run::

    python examples/voronoi_polycrystal_3d.py
"""

from pathlib import Path

import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem

NETWORK_ID = 1_000_000  # above every grain id; a manifold shares the surface ids
SURFACE_ID_0 = 2_000_000  # the per-grain surface patches are numbered from here

# Simulation parameters
L = 1.0  # specimen size
n_seeds = 36  # number of grains
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
locate_geometrically = False  # use the geometric locator instead of facet tags
T = 500.0  # with E_D = 0 on both phases, nothing depends on it


class GrainNetworkProblem(F.HydrogenTransportProblemDiscontinuous):
    """Write the combined fields after FESTIM refreshes each timestep's solution."""

    field_writer = None

    def post_processing(self):
        """Keep FESTIM's post-processing and append the current grain/network fields."""
        super().post_processing()
        if self.field_writer is not None:
            self.field_writer(float(self.t))


def create_microstructure(aspect, aspect_axis):
    """Build one case with grain tags matching its anisotropic geometry."""
    seeds, boundaries = fm.voronoi.tessellate(
        n_seeds,
        L,
        np.random.default_rng(seed),
        dim=3,
        aspect=aspect,
        aspect_axis=aspect_axis,
    )
    mesh_data = fm.voronoi.build_mesh(
        boundaries,
        L,
        fm.MeshSizing(h_gb=h_gb, h_bulk=h_bulk),
        dim=3,
        seeds=seeds,
        aspect=aspect,
        aspect_axis=aspect_axis,
    )
    tagged = not locate_geometrically
    return fm.VoronoiMicrostructure(
        dim=3,
        size=L,
        n_seeds=n_seeds,
        seed=seed,
        seeds=seeds,
        boundaries=boundaries,
        mesh=mesh_data.mesh,
        cell_tags=mesh_data.cell_tags,
        facet_tags=mesh_data.facet_tags if tagged else None,
        gb_tag=mesh_data.gb_tag if tagged else None,
        grain_ids=mesh_data.grain_ids,
        # Seed-image ids need not be contiguous; index by the largest id.
        orientations=np.zeros(int(mesh_data.grain_ids.max())),
        h_gb=h_gb,
        aspect=aspect,
        aspect_axis=aspect_axis,
    )


def grain_network_problem(micro, D_gb):
    """The transport model, as FESTIM declarations; see the 2D example."""
    mesh = micro.mesh
    D_lattice, _ = fm.materials.crystal_diffusivity_field(micro, D_B)
    grains = fm.fem.subdomains.grain_subdomains(micro, F.Material(D=D_lattice))
    # From the facet tags when the microstructure carries them, by the
    # geometric locator otherwise (locate_geometrically above).
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

    charged = lambda x: np.isclose(x[2], L)  # noqa: E731
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

    model = GrainNetworkProblem(
        mesh=F.Mesh(mesh),
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


def run_case(name, aspect, aspect_axis, prefix):
    """Solve and export one grain alignment, then report its uptake."""
    print(f"\n=== {name}: aspect={aspect:g}, elongation along {'xyz'[aspect_axis]} ===")
    micro = create_microstructure(aspect, aspect_axis)
    mesh, boundaries = micro.mesh, micro.boundaries
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, grains, grain_species, network, cgb_species = grain_network_problem(
        micro, D_GB
    )
    with fm.exports.averages.vtx_series(
        grains, grain_species, network, cgb_species, OUTPUT_DIR / prefix
    ) as write_fields:
        model.field_writer = write_fields
        try:
            model.post_processing()  # include the initial state at t=0
            model.run()
        finally:
            model.field_writer = None
    cgb_fast = cgb_species.subdomain_to_post_processing_solution[network]

    # Inspect the mesh and boundary network
    topology = fm.voronoi.junctions(boundaries, dim=3)
    face_area = fm.voronoi.network_measure(boundaries, dim=3)
    sub_area = fm.exports.measures.submesh_measure(network)
    print(f"microstructure: {n_seeds} seeds, {len(boundaries)} boundary polygons")
    print(
        f"  triple lines                    : {len(topology.lines)} "
        f"(length {topology.line_measure:.3f})"
    )
    print(f"  quadruple points                : {len(topology.points)}")
    print(
        f"  connected components (faces)    : "
        f"{fm.voronoi.connected_components(boundaries, dim=3)}"
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
    print(f"  interior facets                 : {model.manifold_is_interior(network)}")

    # Inspect the fast-boundary transport solution
    fast = fm.exports.averages.inventory(
        grains, grain_species, network, cgb_species, delta
    )
    depth = fm.exports.measures.junction_only_below(boundaries, axis=2, top=L)
    gb_z = cgb_fast.function_space.tabulate_dof_coordinates()[:, 2]
    deep = gb_z < depth
    c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

    print(
        f"\nafter t = {t_end} (lattice diffusion alone reaches "
        f"~{2 * np.sqrt(D_B * t_end):.3g})"
    )
    print(f"  inventory with fast boundaries : {fast:.4e}")
    f_gb = delta * face_area / L**3
    print(f"  boundary volume fraction f     : {f_gb:.3e}")
    print(
        f"  Hart bound f D_gb + (1-f) D_b  : "
        f"{fm.materials.hart_bound(f_gb, D_GB, D_B):.3e}"
        f"  (vs D_b = {D_B:.3e})"
    )
    beta = fm.materials.beta_parameter(delta, D_GB, D_B, t_end)
    print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")
    resistance = fm.materials.interface_resistance_ratio(
        k_exchange, L * n_seeds ** (-1 / 3), D_B
    )
    print(
        f"  interface / lattice resistance : {resistance:.1e}  (Fisher is the 0 limit)"
    )

    if not deep.any():
        print(
            "\njunction transport: no sampled region below directly loaded boundaries."
        )
        return fast

    c_grain_deep = fm.exports.averages.lattice_mean_below(
        grains, grain_species, axis=2, depth=depth
    )
    print("\njunction transport: no boundary touching the charged face reaches below")
    print(f"z = {depth:.3f}, so everything the network holds there has")
    print("crossed at least one triple line.")
    print(f"  max c on the network there     : {c_deep:.4e}")
    print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
    ratio = c_deep / c_grain_deep if c_grain_deep > 0 else np.nan
    print(f"  ratio                          : x {ratio:.0f}")
    return fast


def main():
    """Run equiaxed, parallel and perpendicular grains with identical loading."""
    cases = (
        ("equiaxed", 1.0, 0, "voronoi3d"),
        ("parallel", 3.0, 2, "voronoi3d_parallel"),
        ("perpendicular", 3.0, 0, "voronoi3d_perpendicular"),
    )
    inventories = [
        (name, run_case(name, aspect, axis, prefix))
        for name, aspect, axis, prefix in cases
    ]
    print(f"\nHydrogen inventory at t={t_end:g} (loading from z=L):")
    for name, value in inventories:
        print(f"  {name:<15}: {value:.4e}")


if __name__ == "__main__":
    main()
