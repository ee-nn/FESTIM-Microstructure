"""Conforming meshes must retain the grain ownership of elongated tessellations."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from festim_microstructure import VoronoiMicrostructure

pytestmark = pytest.mark.fenics


@pytest.mark.parametrize("axis", [0, 2])
def test_elongated_3d_mesh_cells_stay_inside_their_tagged_grains(axis):
    """Every tetrahedron vertex belongs to its tagged seed's metric Voronoi cell."""
    micro = VoronoiMicrostructure.create(
        size=1.0,
        n_seeds=6,
        seed=3,
        dim=3,
        aspect=3.0,
        aspect_axis=axis,
        cells_per_grain=3,
    )
    mesh = micro.mesh
    stretch = np.ones(3)
    stretch[axis] = micro.aspect
    shifts = np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), -1)
    images = (micro.seeds + shifts.reshape(-1, 1, 3)).reshape(-1, 3) / stretch
    cells = micro.cell_tags.indices
    geom = mesh.geometry
    dofmap = geom.dofmaps[0] if hasattr(geom, "dofmaps") else geom.dofmap
    corners = geom.x[dofmap[cells]] / stretch
    nearest, _ = KDTree(images).query(corners.reshape(-1, 3))
    assigned = images[micro.cell_tags.values - 1]
    distances = np.linalg.norm(corners - assigned[:, None, :], axis=2)
    assert np.all(distances.ravel() <= nearest + 1e-7)

    # Grain interfaces and physical boundary tags must describe the same facets.
    mesh.topology.create_connectivity(2, 3)
    f_to_c = mesh.topology.connectivity(2, 3)
    grain_tags = np.zeros(dofmap.shape[0], dtype=np.int32)
    grain_tags[cells] = micro.cell_tags.values
    interfaces = []
    for facet in range(mesh.topology.index_map(2).size_local):
        adjacent = f_to_c.links(facet)
        if len(adjacent) == 2 and grain_tags[adjacent[0]] != grain_tags[adjacent[1]]:
            interfaces.append(facet)
    assert interfaces
    assert np.array_equal(interfaces, micro.facet_tags.find(micro.gb_tag))
