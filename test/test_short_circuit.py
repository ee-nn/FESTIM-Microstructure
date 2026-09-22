"""End-to-end smoke test of the short-circuit model on a tiny 2D Voronoi cell."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import requires_fenics

pytestmark = requires_fenics


@pytest.mark.fenics
def test_fast_boundaries_increase_inventory():
    from festim_microstructure.exports.averages import inventory
    from festim_microstructure.materials import Physics
    from festim_microstructure.model import SolveOptions, build
    from festim_microstructure.voronoi import VoronoiMicrostructure

    micro = VoronoiMicrostructure.create(
        size=1.0, n_seeds=6, seed=0, cells_per_grain=8, bulk_coarsening=4.0
    )
    # E_D = 0 everywhere, so D_bulk and D_gb are the prefactors as written.
    physics = Physics(
        D_0_bulk=1e-3,
        E_D_bulk=0.0,
        D_0_gb=30.0,
        E_D_gb=0.0,
        delta=1e-3,
        k_exchange=1.0,
        crystal_anisotropy=1.0,
    )
    bcs = [("charged", lambda x: np.isclose(x[1], micro.size), 1.0)]
    solve = SolveOptions(transient=True, final_time=0.2, stepsize=0.02)

    fast = inventory(build(micro, physics, bcs, solve=solve).run())
    slow = replace(physics, D_0_gb=physics.D_0_bulk, E_D_gb=0.0)
    ref = inventory(build(micro, slow, bcs, solve=solve).run())
    assert fast > ref


@pytest.mark.fenics
def test_geometric_network_covers_all_grain_boundaries():
    from festim_microstructure.fem.subdomains import facet_midpoints
    from festim_microstructure.voronoi import (
        MeshSizing,
        build_mesh,
        near_segments,
        snap_segments,
        voronoi_segments,
    )

    L, h_gb = 1.0, 0.05
    segments = snap_segments(
        voronoi_segments(6, L, np.random.default_rng(0)), 0.1 * h_gb, L
    )
    mesh, cell_tags, _ = build_mesh(segments, L, MeshSizing(h_gb=h_gb, h_bulk=4 * h_gb))

    def locator(x):
        return near_segments(x, segments, 0.05 * h_gb)

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
    boundaries = interior[values[pair[:, 0]] != values[pair[:, 1]]]
    assert boundaries.size > 0
    assert locator(facet_midpoints(mesh, boundaries).T).all()
