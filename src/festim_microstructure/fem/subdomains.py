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
    """One codimension-one network, declared as a single connected subdomain.

    Its facets come either from the mesh's facet tags -- ``facet_tags`` and the
    ``entity_ids`` to keep -- or from a pointwise ``locator``, for a mesh that
    carries no tags. Exactly one of the two, since a network located both ways
    at once has no defined answer when they disagree.

    A tag says what a facet *is*; a predicate only says where it is, so the
    geometric route needs two guards the tagged one does not. Midpoints avoid
    false positives at junctions, where every vertex of a facet can sit on the
    network while the facet itself lies across the junction. ``drop_exterior``
    then removes the trace the network leaves on the wall of the box, which a
    distance cannot tell from the network itself, and which would leave FESTIM
    with a manifold that is neither wholly interior nor wholly on the boundary.
    It therefore defaults on for a locator and off for tags; pass it explicitly
    to filter a tagged network too, or to keep a network that is meant to sit on
    the boundary of the mesh. ``n_dropped`` records what it removed, either way.

    ``entity_ids_of_facets`` stays aligned with the submesh parent map, which is
    what lets a per-boundary field be read back onto the submesh. Only the
    tagged route can fill it: see
    :func:`~festim_microstructure.materials.gb_diffusivity_field`.
    """

    def __init__(
        self,
        id,
        material,
        dim,
        *,
        facet_tags=None,
        entity_ids=None,
        locator=None,
        drop_exterior=None,
    ):
        super().__init__(id=id, material=material, dim=dim)
        tagged = facet_tags is not None
        if tagged == (locator is not None):
            raise TypeError(
                f"grain-boundary network {id} must be located either by "
                "facet_tags and entity_ids or by a locator, and was given "
                f"{'both' if tagged else 'neither'}"
            )
        if tagged and entity_ids is None:
            raise TypeError(
                f"grain-boundary network {id} was given facet_tags without "
                "entity_ids, so there is nothing saying which tags are "
                "boundaries; pass the ids to keep"
            )
        self.facet_tags = facet_tags
        # one tag or a sequence of them: a Neper or EBSD mesh tags every
        # tessellation face separately, so its network is a list of ids
        self.entity_ids = (
            None
            if entity_ids is None
            else np.atleast_1d(np.asarray(entity_ids, dtype=np.int32))
        )
        self.network_locator = locator
        self.drop_exterior = (not tagged) if drop_exterior is None else drop_exterior
        self.n_dropped = 0
        self.entity_ids_of_facets = None

    def locate_subdomain_entities(self, mesh):
        facets, ids = (
            self._from_tags()
            if self.facet_tags is not None
            else (self._from_locator(mesh), None)
        )
        if self.drop_exterior and facets.size:
            interior = interior_facet_mask(mesh, facets)
            self.n_dropped = int((~interior).sum())
            facets = facets[interior]
            ids = None if ids is None else ids[interior]
        self.entity_ids_of_facets = ids
        return facets.astype(np.int32)

    def _from_tags(self):
        assert self.facet_tags is not None
        assert self.entity_ids is not None
        keep = np.isin(self.facet_tags.values, self.entity_ids)
        return self.facet_tags.indices[keep], self.facet_tags.values[keep].astype(
            np.int32
        )

    def _from_locator(self, mesh):
        assert self.network_locator is not None
        tdim = mesh.topology.dim
        candidates = dolfinx.mesh.locate_entities(mesh, tdim - 1, self.network_locator)
        if candidates.size == 0:
            return candidates
        return candidates[self.network_locator(facet_midpoints(mesh, candidates).T)]


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
