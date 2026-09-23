"""A per-boundary diffusivity read from tessellation tags back onto the submesh.

The submesh of a tagged network exists only once FESTIM has initialised the
problem, so the network builds the field itself when FESTIM creates the
submesh; the check matches submesh cells to parent facets by position,
independently of the entity map the field itself uses.
"""

import numpy as np
import pytest
from conftest import requires_fenics
from test_short_circuit import NETWORK_ID, SURFACE_ID_0

pytestmark = requires_fenics


@pytest.mark.fenics
def test_diffusivity_by_entity_follows_the_tags():
    """Verify diffusivity by entity follows the tags."""
    import dolfinx
    import festim as F
    from scipy.spatial import KDTree

    import festim_microstructure as fm
    from festim_microstructure.fem.subdomains import facet_midpoints

    micro = fm.VoronoiMicrostructure.create(
        size=1.0, n_seeds=6, seed=0, cells_per_grain=8, bulk_coarsening=4.0
    )
    mesh, cell_tags = micro.mesh, micro.cell_tags
    # The 2D builder carries no facet tags, so tag the grain-grain facets here,
    # as if they were three tessellation entities.
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    index_map = cell_tags.topology.index_map(tdim)
    grain_of = np.zeros(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    grain_of[cell_tags.indices] = cell_tags.values
    facets = np.arange(mesh.topology.index_map(tdim - 1).size_local, dtype=np.int32)
    offsets = facet_to_cell.offsets
    interior = facets[(offsets[facets + 1] - offsets[facets]) == 2]
    pair = facet_to_cell.array[offsets[interior][:, None] + np.arange(2)]
    gb = interior[grain_of[pair[:, 0]] != grain_of[pair[:, 1]]]
    entity = 1 + gb % 3
    facet_tags = dolfinx.mesh.meshtags(mesh, 1, gb, entity.astype(np.int32))
    poly = fm.TaggedPolycrystal(
        mesh=mesh, cell_tags=micro.cell_tags, facet_tags=facet_tags, gb_tag=[1, 2, 3]
    )
    theta = np.array([5.0, 20.0, 40.0])  # entity 1 is low-angle
    D_by_entity = np.where(theta >= 15.0, 100.0, 1.0)

    D_lattice, _ = fm.materials.crystal_diffusivity_field(poly, 1e-3)
    grains = fm.fem.subdomains.grain_subdomains(poly, F.Material(D=D_lattice))
    with pytest.raises(RuntimeError, match="no submesh"):
        fm.fem.subdomains.grain_boundary_network(
            NETWORK_ID, poly, None, diffusivity_by_entity=D_by_entity
        ).entity_field(D_by_entity)
    network = fm.fem.subdomains.grain_boundary_network(
        NETWORK_ID, poly, None, diffusivity_by_entity=D_by_entity
    )
    grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    k = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    sources, bcs = [], []
    for c_grain in grain_species:
        exchange = {"c_g": c_grain, "c_n": c_gb}
        sources.append(
            F.ParticleSource(
                value=lambda c_g, c_n: k * (c_g - c_n),
                species=c_gb,
                volume=network,
                species_dependent_value=exchange,
            )
        )
        bcs.append(
            F.ParticleFluxBC(
                subdomain=network,
                species=c_grain,
                value=lambda c_g, c_n: k * (c_n - c_g),
                species_dependent_value=exchange,
            )
        )
    patches, mouths = fm.fem.subdomains.grain_surfaces(
        mesh, grains, lambda x: np.isclose(x[1], 1.0), SURFACE_ID_0
    )
    species_of = dict(zip((g.id for g in grains), grain_species, strict=True))
    bcs += [
        F.FixedConcentrationBC(subdomain=p, value=1.0, species=species_of[p.grain_id])
        for p in patches
    ]
    bcs.append(F.FixedConcentrationBC(subdomain=mouths, value=1.0, species=c_gb))
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        subdomains=[*grains, network, *patches, mouths],
        species=[*grain_species, c_gb],
        sources=sources,
        boundary_conditions=bcs,
        temperature=500.0,
        settings=F.Settings(atol=1e-25, rtol=1e-10, transient=False),
    )
    model.show_progress_bar = False
    model.initialise()
    field = network.material.D
    assert field is not None

    # Independent check: match every submesh cell to its parent facet by
    # position and read the entity the tag array gives that facet.
    sub = network.submesh
    n_sub = sub.topology.index_map(sub.topology.dim).size_local
    V = field.function_space
    mid_sub = dolfinx.mesh.compute_midpoints(sub, sub.topology.dim, np.arange(n_sub))
    _, nearest = KDTree(facet_midpoints(mesh, gb)).query(mid_sub)
    expected = np.where(theta[entity[nearest] - 1] >= 15.0, 100.0, 1.0)
    got = field.x.array[V.dofmap.list[:n_sub].reshape(-1)]
    assert np.array_equal(got, expected)
    assert (got == 1.0).any() and (got == 100.0).any()
    # any other per-entity quantity reads back the same way
    assert np.array_equal(
        network.entity_field(theta).x.array[V.dofmap.list[:n_sub].reshape(-1)],
        theta[entity[nearest] - 1],
    )

    fm.fem.solvers.tune_direct_solver(model)
    model.run()
    total = fm.exports.averages.inventory(grains, grain_species, network, c_gb, 1e-3)
    assert np.isfinite(total) and total > 0
