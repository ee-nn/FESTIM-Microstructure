"""FESTIM subdomains for grain-boundary networks.

Networks are one connected codimension-one subdomain, so junction transport is
continuous. They can be located geometrically or read from facet tags.
"""

import dolfinx
import festim as F
import numpy as np

__all__ = [
    "Grain",
    "GrainBoundaryNetwork",
    "GrainSurface",
    "TaggedGrainBoundaryNetwork",
]


def facet_midpoints(mesh, facets):
    """Return facet midpoints as ``(n, 3)`` coordinates."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
    offsets = facet_to_vertex.offsets
    vertices = facet_to_vertex.array[offsets[facets][:, None] + np.arange(tdim)]
    return mesh.geometry.x[vertices].mean(axis=1)


def interior_facet_mask(mesh, facets):
    """Return whether each facet is shared by two cells."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    offsets = mesh.topology.connectivity(tdim - 1, tdim).offsets
    return (offsets[facets + 1] - offsets[facets]) == 2


class Grain(F.VolumeSubdomain):
    """One grain, located from its Gmsh cell tag."""

    def __init__(self, id, material, cell_tags):
        super().__init__(id=id, material=material)
        self.cell_tags = cell_tags

    def locate_subdomain_entities(self, mesh):
        return self.cell_tags.find(self.id).astype(np.int32)


class GrainBoundaryNetwork(F.VolumeSubdomain):
    """One codimension-one network located by a pointwise predicate.

    Midpoints avoid false positives at junctions. Exterior facets are dropped so
    FESTIM receives a wholly interior manifold.
    """

    def __init__(self, id, material, locator, dim, drop_exterior=True):
        super().__init__(id=id, material=material, dim=dim)
        self.network_locator = locator
        self.drop_exterior = drop_exterior
        self.n_dropped = 0
        self.entity_ids = None

    def locate_subdomain_entities(self, mesh):
        tdim = mesh.topology.dim
        candidates = dolfinx.mesh.locate_entities(mesh, tdim - 1, self.network_locator)
        if candidates.size == 0:
            return candidates.astype(np.int32)
        keep = self.network_locator(facet_midpoints(mesh, candidates).T)
        if self.drop_exterior:
            interior = interior_facet_mask(mesh, candidates)
            self.n_dropped = int((keep & ~interior).sum())
            keep &= interior
        return candidates[keep].astype(np.int32)


class TaggedGrainBoundaryNetwork(F.VolumeSubdomain):
    """One codimension-one network selected from facet tags.

    ``entity_ids_of_facets`` stays aligned with the submesh parent map.
    """

    def __init__(self, id, material, facet_tags, entity_ids, dim):
        super().__init__(id=id, material=material, dim=dim)
        self.facet_tags = facet_tags
        self.entity_ids = np.asarray(entity_ids, dtype=np.int32)
        self.entity_ids_of_facets = None

    def locate_subdomain_entities(self, mesh):
        keep = np.isin(self.facet_tags.values, self.entity_ids)
        self.entity_ids_of_facets = self.facet_tags.values[keep].astype(np.int32)
        return self.facet_tags.indices[keep].astype(np.int32)


class GrainSurface(F.SurfaceSubdomain):
    """The part of an outer surface belonging to one grain."""

    def __init__(self, id, grain_id, cell_tags, locator):
        super().__init__(id=id, locator=locator)
        self.grain_id = grain_id
        self.cell_tags = cell_tags
        self._cache = {}

    def locate_boundary_facet_indices(self, mesh):
        # FESTIM locates each patch twice; cache by mesh identity.
        key = id(mesh)
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = self._locate(mesh)
        return self._cache[key]

    def _locate(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
        facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, self.locator)

        # Index the adjacency list's flat array through its offsets rather than
        # calling links() per facet, as FESTIM does in
        # HydrogenTransportProblem.manifold_is_interior: this runs once per grain
        # per boundary condition over every boundary facet of the mesh, so the
        # Python loop it replaces costs n_grains x n_bcs x n_boundary_facets
        # iterations on every build. locate_entities_boundary returns exterior
        # facets only, and an exterior facet has exactly one adjacent cell, which
        # sits at offsets[facet].
        adjacent = facet_to_cell.array[facet_to_cell.offsets[facets]]
        keep = np.isin(adjacent, self.cell_tags.find(self.grain_id))
        return facets[keep].astype(np.int32)
