"""End-to-end smoke test of the grain/network model on a tiny 2D Voronoi cell.

The model is declared the way FESTIM's manifold documentation shows -- one
``VolumeSubdomain`` and ``Species`` per grain, the network as one
codimension-one subdomain, one ``ParticleFluxBC``/``ParticleSource`` pair per
grain -- with this package supplying the tagged subdomains, the lattice tensor
field, and the post-processing.
"""

import numpy as np
import pytest
from conftest import requires_fenics

pytestmark = requires_fenics

NETWORK_ID = 1_000_000
SURFACE_ID_0 = 2_000_000


def grain_network_problem(micro, D_bulk, D_gb, delta, k_exchange, bcs, settings):
    """A grain/network problem for ``micro``: FESTIM declarations only.

    ``bcs`` is a list of ``(locator, value)`` pairs, each applied to every grain
    touching that surface and to the network's mouths on it. Returns the
    problem and the handles the post-processing reads.
    """
    import dolfinx
    import festim as F

    import festim_microstructure as fm

    scalar = dolfinx.default_scalar_type
    mesh = micro.mesh
    D_lattice, tensors = fm.materials.crystal_diffusivity_field(micro, D_bulk)
    grains = fm.fem.subdomains.grain_subdomains(micro, F.Material(D=D_lattice))
    network = fm.fem.subdomains.grain_boundary_network(
        NETWORK_ID, micro, F.Material(D_0=D_gb, E_D=0.0)
    )
    grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    species_of = dict(zip((g.id for g in grains), grain_species, strict=True))

    # Each grain exchanges k (c_grain - c_gb) with the boundary slab; the
    # network equation is per unit slab width, hence k / delta on its side.
    k = dolfinx.fem.Constant(mesh, scalar(k_exchange))
    width = dolfinx.fem.Constant(mesh, scalar(delta))
    sources, boundary_conditions = [], []
    for c_grain in grain_species:
        sources.append(
            F.ParticleSource(
                value=lambda c_g, c_n: (k / width) * (c_g - c_n),
                species=c_gb,
                volume=network,
                species_dependent_value={"c_g": c_grain, "c_n": c_gb},
            )
        )
        boundary_conditions.append(
            F.ParticleFluxBC(
                subdomain=network,
                species=c_grain,
                value=lambda c_g, c_n: k * (c_n - c_g),
                species_dependent_value={"c_g": c_grain, "c_n": c_gb},
            )
        )

    subdomains = [*grains, network]
    next_id = SURFACE_ID_0
    for locator, value in bcs:
        patches, mouths = fm.fem.subdomains.grain_surfaces(
            mesh, grains, locator, next_id
        )
        next_id = mouths.id + 1
        subdomains += [*patches, mouths]
        boundary_conditions += [
            F.FixedConcentrationBC(
                subdomain=patch, value=value, species=species_of[patch.grain_id]
            )
            for patch in patches
        ]
        boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=mouths, value=value, species=c_gb)
        )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        subdomains=subdomains,
        species=[*grain_species, c_gb],
        sources=sources,
        boundary_conditions=boundary_conditions,
        temperature=500.0,
        settings=settings,
    )
    model.show_progress_bar = False
    return model, grains, grain_species, tensors, network, c_gb


@pytest.fixture(scope="module")
def micro():
    from festim_microstructure.voronoi import VoronoiMicrostructure

    return VoronoiMicrostructure.create(
        size=1.0, n_seeds=6, seed=0, cells_per_grain=8, bulk_coarsening=4.0
    )


@pytest.mark.fenics
def test_fast_boundaries_increase_inventory(micro):
    import festim as F

    from festim_microstructure.exports.averages import inventory
    from festim_microstructure.fem.solvers import tune_direct_solver

    D_b, delta, k = 1e-3, 1e-3, 1.0
    bcs = [(lambda x: np.isclose(x[1], micro.size), 1.0)]

    def uptake(D_gb):
        settings = F.Settings(
            atol=1e-25, rtol=1e-10, transient=True, final_time=0.2, stepsize=0.02
        )
        model, grains, species, _, network, c_gb = grain_network_problem(
            micro, D_b, D_gb, delta, k, bcs, settings
        )
        model.initialise()
        tune_direct_solver(model)
        model.run()
        return inventory(grains, species, network, c_gb, delta)

    fast = uptake(D_gb=30.0)
    ref = uptake(D_gb=D_b)
    assert fast > ref


@pytest.mark.fenics
def test_cell_problem_flux_and_local_equilibrium(micro):
    """A steady uniform-gradient cell problem: the flux runs against the
    gradient, the boundaries carry more than the lattice would alone, and the
    grain/network mismatch is what the exchange rate says it is."""
    import festim as F

    from festim_microstructure.exports.averages import averages, equilibrium_error
    from festim_microstructure.fem.solvers import tune_direct_solver

    D_b, D_gb, delta, k = 1e-3, 30.0, 1e-3, 1.0
    G = np.array([1.0, 0.0])
    bcs = [
        (
            lambda x: np.full_like(x[0], True, dtype=bool),
            lambda x: G[0] * x[0] + G[1] * x[1],
        )
    ]
    settings = F.Settings(atol=1e-25, rtol=1e-10, transient=False)
    model, grains, species, tensors, network, c_gb = grain_network_problem(
        micro, D_b, D_gb, delta, k, bcs, settings
    )
    model.initialise()
    tune_direct_solver(model)
    assert model.manifold_is_interior(network)
    model.run()

    q, grad_c, area = averages(grains, species, tensors, network, c_gb, delta, D_gb)
    assert np.isclose(area, micro.size**2)
    assert np.isclose(grad_c[0], 1.0, rtol=2e-2)
    assert q[0] < -D_b  # the network adds to the lattice flux
    assert (
        0.0 < equilibrium_error(grains, species, network, c_gb, micro.tolerance) < 1.0
    )


@pytest.mark.fenics
def test_geometric_network_covers_all_grain_boundaries():
    from festim_microstructure.fem.subdomains import facet_midpoints
    from festim_microstructure.voronoi import (
        MeshSizing,
        build_mesh,
        near,
        snap,
        tessellate,
    )

    L, h_gb = 1.0, 0.05
    _, boundaries = tessellate(6, L, np.random.default_rng(0), dim=2)
    boundaries = snap(boundaries, 0.1 * h_gb, L, dim=2)
    mesh_data = build_mesh(boundaries, L, MeshSizing(h_gb=h_gb, h_bulk=4 * h_gb), dim=2)
    mesh, cell_tags = mesh_data.mesh, mesh_data.cell_tags

    def locator(x):
        return near(x, boundaries, 0.05 * h_gb, dim=2)

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    index_map = cell_tags.topology.index_map(tdim)
    values = np.zeros(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    values[cell_tags.indices] = cell_tags.values

    facets = np.arange(mesh.topology.index_map(tdim - 1).size_local, dtype=np.int32)
    offsets = facet_to_cell.offsets
    interior = facets[(offsets[facets + 1] - offsets[facets]) == 2]
    pair = facet_to_cell.array[offsets[interior][:, None] + np.arange(2)]
    boundary_facets = interior[values[pair[:, 0]] != values[pair[:, 1]]]
    assert boundary_facets.size > 0
    assert locator(facet_midpoints(mesh, boundary_facets).T).all()
