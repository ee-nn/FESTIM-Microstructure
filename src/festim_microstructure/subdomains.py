"""FESTIM subdomains for a grain-boundary network.

The whole network is declared as **one** codim-1 subdomain carrying **one**
species. That is what makes triple junctions work with no junction condition to
write: the submesh built from every grain-boundary facet is connected, so a
single continuous field lives on it. FESTIM gives each adjacent grain an
interior-facet integral of its own on which that grain is the ``"+"`` side, so a
facet shared by two grains is integrated once per grain -- exactly the two-sided
feeding the physics asks for.

There are two ways to say which facets belong to the network, and hence two
classes:

* :class:`GrainBoundaryNetwork` locates them **geometrically**, from a callable
  that says whether a point lies on the network. This is the route for meshes
  built in-process from a Voronoi tessellation (:mod:`.meshing.voronoi`).
* :class:`TaggedGrainBoundaryNetwork` reads them from the **facet tags** the
  mesh was written with. This is the route for Neper meshes and for EBSD rasters
  meshed by Neper, where every tessellation face / edge is an element set.

Both expose ``entity_ids`` after :meth:`locate_subdomain_entities` has run: the
tessellation face/edge id of every located facet, in order, which is what
:func:`festim_microstructure.models.properties.gb_diffusivity_field` needs to put
a per-boundary coefficient on the submesh. For the geometric class it is
``None`` (there is no tessellation id to report).
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


class Grain(F.VolumeSubdomain):
    """One grain (Voronoi cell), located from its gmsh physical group.

    Cells are read straight from the tags the mesh was generated with. Note that
    FESTIM calls ``locate_subdomain_entities(mesh)`` with the mesh, so the
    argument has to stay even though this implementation does not use it.
    """

    def __init__(self, id, material, cell_tags):
        super().__init__(id=id, material=material)
        self.cell_tags = cell_tags

    def locate_subdomain_entities(self, mesh):
        return self.cell_tags.find(self.id).astype(np.int32)


class GrainBoundaryNetwork(F.VolumeSubdomain):
    """The whole network as one codim-1 subdomain, located geometrically.

    ``locate_subdomain_entities`` is overridden rather than passing a ``locator``
    to FESTIM: ``locate_entities`` marks a facet when *all its vertices* satisfy
    the locator, which near a triple junction also catches short facets that
    merely touch two different boundaries (and in 3D, a triangle with all three
    vertices on a triple line lies on no grain boundary in particular). Testing
    the facet midpoint as well selects the network exactly.

    With ``drop_exterior`` (the default) facets on the outer boundary of the mesh
    are then dropped. FESTIM requires a manifold to be wholly interior (``dS``
    measure) or wholly exterior (``ds`` measure), and cannot have both in one
    form. Those facets sit on a surface that carries a boundary condition anyway.
    ``n_dropped`` records how many went.

    Args:
        id: subdomain id.
        material: a :class:`festim.Material` for the boundary (``D_gb``).
        locator: callable ``(3, n) -> (n,) bool``, true on the network. For a
            2D Voronoi tessellation this is
            ``lambda x: near_segments(x, segments, tol)``; a
            :class:`~festim_microstructure.meshing.voronoi.VoronoiMicrostructure`
            provides it as ``micro.locator``.
        dim: topological dimension of the network: ``1`` in a 2D mesh, ``2`` in
            a 3D mesh.
        drop_exterior: drop facets on the outer boundary of the mesh.
    """

    def __init__(self, id, material, locator, dim, drop_exterior=True):
        super().__init__(id=id, material=material, dim=dim)
        self.network_locator = locator
        self.drop_exterior = drop_exterior
        self.n_dropped = 0
        self.entity_ids = None

    def locate_subdomain_entities(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
        candidates = dolfinx.mesh.locate_entities(mesh, tdim - 1, self.network_locator)
        if candidates.size == 0:
            return candidates.astype(np.int32)
        x = mesh.geometry.x
        midpoints = np.array(
            [x[facet_to_vertex.links(f)].mean(axis=0) for f in candidates]
        )
        # TODO again, switch to a vectorized format to reduced computational overhead
        keep = self.network_locator(midpoints.T)
        if self.drop_exterior:
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
            interior = np.array(
                [len(facet_to_cell.links(f)) == 2 for f in candidates], dtype=bool
            )  # TODO see above
            self.n_dropped = int((keep & ~interior).sum())
            keep &= interior
        return candidates[keep].astype(np.int32)


class TaggedGrainBoundaryNetwork(F.VolumeSubdomain):
    """The whole network as one codim-1 subdomain, read from the facet tags.

    The facets come straight from the tags, so the delicate part of the
    geometric version -- the all-vertices problem of ``locate_entities`` near a
    junction -- does not arise. There is nothing geometric left to get wrong.

    Args:
        id: subdomain id.
        material: a :class:`festim.Material` for the boundary.
        facet_tags: the ``MeshTags`` of the facets, as read by ``gmshio``.
        entity_ids: the tag values that make up the network -- Neper face ids in
            3D, edge ids in 2D, or a single gmsh physical-group id.
        dim: topological dimension of the network: ``1`` in a 2D mesh, ``2`` in
            a 3D mesh.

    After :meth:`locate_subdomain_entities`, ``entity_ids_of_facets`` holds the
    tag value of each located facet in the order they were returned;
    ``dolfinx.mesh.create_submesh`` keeps that order, so the submesh's parent map
    indexes straight into it.
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
    """The part of an outer surface that belongs to one grain.

    A :class:`festim.SurfaceSubdomain` belongs to exactly one volume subdomain, so
    a surface crossing several grains has to be declared once per grain.
    """

    def __init__(self, id, grain_id, cell_tags, locator):
        super().__init__(id=id, locator=locator)
        self.grain_id = grain_id
        self.cell_tags = cell_tags

    def locate_boundary_facet_indices(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
        cells = set(self.cell_tags.find(self.grain_id).tolist())
        facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, self.locator)

        keep = [f for f in facets if any(c in cells for c in facet_to_cell.links(f))]
        # TODO again, switch to a vectorized format to reduced computational overhead

        return np.array(keep, dtype=np.int32)
