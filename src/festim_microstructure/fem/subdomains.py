"""FESTIM subdomains for polycrystals: grains, the boundary network, and the
part of an outer surface each grain owns.

FESTIM's own :class:`festim.VolumeSubdomain` locates its cells with a geometric
predicate; a polycrystal already knows its grains from the cell tags, so
:class:`Grain` reads them from there. The network is **one** connected
codimension-one subdomain, so transport through a junction is continuous and
needs no junction condition; see :class:`GrainBoundaryNetwork`. An outer surface
that crosses several independently solved grains has to be split into one
:class:`GrainSurface` per grain, because FESTIM maps a surface subdomain to a
single volume subdomain.

The three factory functions at the end build these from a
:class:`~festim_microstructure.microstructure.MeshedMicrostructure`. Everything
else -- species, the exchange terms, boundary conditions, settings, the
:class:`festim.HydrogenTransportProblemDiscontinuous` itself -- is written in
the application, as FESTIM's manifold documentation shows.
"""

import dolfinx
import festim as F
import numpy as np

__all__ = [
    "Grain",
    "GrainBoundaryNetwork",
    "GrainSurface",
    "facet_midpoints",
    "grain_boundary_network",
    "grain_subdomains",
    "grain_surfaces",
    "interior_facet_mask",
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

    After FESTIM has located the network, ``located_facets`` holds the parent
    facets it was given and ``entity_ids_of_facets`` the tessellation entity
    (the Neper face or EBSD edge) behind each of them, aligned. Only the tagged
    route can fill them, and they are what lets a per-boundary quantity be read
    back onto the submesh: :meth:`entity_field` turns an array with one value
    per tessellation entity into a DG0 function on the submesh.

    A boundary-dependent diffusivity is passed the same way, as
    ``diffusivity_by_entity``, one value per entity in id order (evaluated at
    the temperature of the run: a field carries no Arrhenius law). FESTIM asks
    a manifold's material for its coefficient on the submesh, which exists only
    once the problem is initialised, so the network builds the field itself as
    soon as FESTIM creates that submesh and wraps it as ``festim.Material(D=...)``
    in place of ``material``, which may then be ``None``.
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
        diffusivity_by_entity=None,
    ):
        super().__init__(id=id, material=material, dim=dim)
        tagged = facet_tags is not None
        if diffusivity_by_entity is not None and not tagged:
            raise TypeError(
                f"grain-boundary network {id} was given a diffusivity per entity "
                "but no facet_tags, and only tags say which entity a facet is"
            )
        if material is None and diffusivity_by_entity is None:
            raise TypeError(
                f"grain-boundary network {id} needs a material, or a diffusivity "
                "per entity to build one from"
            )
        self.diffusivity_by_entity = (
            None
            if diffusivity_by_entity is None
            else np.asarray(diffusivity_by_entity, dtype=float)
        )
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
        self.located_facets = None
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
        self.located_facets = facets.astype(np.int32)
        self.entity_ids_of_facets = ids
        return self.located_facets

    def create_subdomain(self, mesh, marker):
        super().create_subdomain(mesh, marker)
        if self.diffusivity_by_entity is not None:
            self.material = F.Material(
                D=self.entity_field(self.diffusivity_by_entity, name="D_gb")
            )

    def entity_field(self, values_by_entity, name="entity_field"):
        """A DG0 function on the submesh with ``values_by_entity[id - 1]`` on
        every cell that came from tessellation entity ``id``.

        Needs the tagged route and the submesh, which FESTIM creates in
        ``initialise()``. Each submesh cell is one parent facet -- FESTIM keeps
        the map as ``cell_map`` -- and the parent facet's entity is what the tags
        said when the network was located.
        """
        submesh = getattr(self, "submesh", None)
        if submesh is None or self.entity_ids_of_facets is None:
            raise RuntimeError(
                f"grain-boundary network {self.id} has no submesh yet, or was not "
                "located from facet tags: an entity field needs both"
            )
        values = np.asarray(values_by_entity, dtype=float)
        # Entity of every parent facet, looked up by facet index.
        parent = self.parent_mesh
        facet_map = parent.topology.index_map(parent.topology.dim - 1)
        entity_of = np.zeros(facet_map.size_local + facet_map.num_ghosts, np.int32)
        entity_of[self.located_facets] = self.entity_ids_of_facets
        tdim = submesh.topology.dim
        index_map = submesh.topology.index_map(tdim)
        cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
        parent_facets = self.cell_map.sub_topology_to_topology(cells, inverse=False)
        # every owned cell came from a facet the network was located on; only
        # a ghost may not have, and scatter_forward() below overwrites those
        assert (entity_of[parent_facets[: index_map.size_local]] > 0).all()
        V = dolfinx.fem.functionspace(submesh, ("DG", 0))
        field = dolfinx.fem.Function(V, name=name)
        field.x.array[V.dofmap.list[cells].reshape(-1)] = values[
            entity_of[parent_facets] - 1
        ]
        field.x.scatter_forward()  # ghost cells take their owner's value
        return field

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


def grain_subdomains(micro, material):
    """One :class:`Grain` per tagged grain of ``micro``, all sharing ``material``.

    The usual material is ``festim.Material(D=field)`` with the field from
    :func:`~festim_microstructure.materials.crystal_diffusivity_field`, so that
    each grain reads its own tensor from the one parent-mesh coefficient.
    """
    return [
        Grain(id=int(g), material=material, cell_tags=micro.cell_tags)
        for g in micro.grain_ids
    ]


def grain_boundary_network(
    id, micro, material, drop_exterior=None, diffusivity_by_entity=None
):
    """The network of ``micro`` as one codimension-one subdomain.

    Read from ``micro.facet_tags`` and ``micro.gb_tag`` when the mesh carries
    tags, and located by ``micro.locator`` otherwise; ``id`` must be unique
    among every volume *and* surface subdomain of the problem, since a manifold
    is tagged in the facet tags. See :class:`GrainBoundaryNetwork` for
    ``drop_exterior`` and ``diffusivity_by_entity``.
    """
    dim = micro.mesh.topology.dim - 1
    if micro.facet_tags is not None:
        return GrainBoundaryNetwork(
            id,
            material,
            dim,
            facet_tags=micro.facet_tags,
            entity_ids=micro.gb_tag,
            drop_exterior=drop_exterior,
            diffusivity_by_entity=diffusivity_by_entity,
        )
    return GrainBoundaryNetwork(
        id, material, dim, locator=micro.locator, drop_exterior=drop_exterior
    )


def grain_surfaces(mesh, grains, locator, first_id):
    """Split the outer surface ``locator`` selects into what each grain owns.

    Returns the non-empty per-grain patches, numbered from ``first_id``, and
    the *mouths* of the network on that surface: the codimension-two
    :class:`festim.SurfaceSubdomain` (the endpoints of the network in 2D, its rim
    in 3D) that carries the network species' boundary condition there. It takes
    the id after the last patch. A grain that does not touch the surface gets no
    patch, and its id is skipped.

    A boundary condition on the surface is then one
    :class:`festim.FixedConcentrationBC` per patch for that grain's species, and
    one on the mouths for the network species.
    """
    tdim = mesh.topology.dim
    patches = []
    next_id = first_id
    for grain in grains:
        patch = GrainSurface(next_id, grain.id, grain.cell_tags, locator)
        next_id += 1
        if patch.locate_boundary_facet_indices(mesh).size:
            patches.append(patch)
    mouths = F.SurfaceSubdomain(id=next_id, dim=tdim - 2, locator=locator)
    return patches, mouths
