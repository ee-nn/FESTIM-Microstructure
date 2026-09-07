"""End-to-end smoke test of the short-circuit model on a tiny 2D Voronoi cell."""

import numpy as np
import pytest
from conftest import requires_fenics

pytestmark = requires_fenics


@pytest.mark.fenics
def test_fast_boundaries_increase_inventory():
    import festim as F

    from festim_microstructure.meshing.voronoi import (
        build_mesh,
        near_segments,
        snap_segments,
        voronoi_segments,
    )
    from festim_microstructure.models.fisher import (
        ShortCircuitParams,
        ShortCircuitProblem,
    )
    from festim_microstructure.postprocessing.measures import inventory
    from festim_microstructure.subdomains import GrainBoundaryNetwork

    L, h_gb = 1.0, 0.05
    segments = snap_segments(
        voronoi_segments(6, L, np.random.default_rng(0)), 0.1 * h_gb, L
    )
    mesh, _, _ = build_mesh(segments, L, h_gb, 4 * h_gb)
    network = GrainBoundaryNetwork(
        id=2,
        material=F.Material(D_0=1.0, E_D=0.0),
        locator=lambda x: near_segments(x, segments, 0.05 * h_gb),
        dim=1,
    )
    params = ShortCircuitParams(
        D_b=1e-3, D_gb=30.0, delta=1e-3, k_exchange=1.0, t_end=0.2, dt=0.02
    )
    problem = ShortCircuitProblem(mesh, network, lambda x: np.isclose(x[1], L), params)
    _, cb, cgb = problem.solve(params.D_gb)
    fast = inventory(cb, cgb, params.delta)
    _, cb, cgb = problem.solve(params.D_b)
    ref = inventory(cb, cgb, params.delta)
    assert fast > ref
