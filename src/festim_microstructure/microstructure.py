"""The microstructure contract the builders implement and the helpers consume.

Three protocols, because there are really two different things in this package
that both get called "a microstructure":

* :class:`MeshedMicrostructure` -- a mesh with one tagged subdomain per grain,
  which is what the subdomain factories in
  :mod:`~festim_microstructure.fem.subdomains`, the coefficient fields in
  :mod:`~festim_microstructure.materials` and the post-processing in
  :mod:`~festim_microstructure.exports` read. Implemented by the Voronoi
  builders, and by :class:`TaggedPolycrystal` for a mesh that came from
  somewhere else.
* :class:`BoundaryNetwork` -- a tessellation topology with a per-entity
  disorientation and masks selecting the boundaries to keep. Implemented by the
  Neper and EBSD readers.

Both are declared as :pep:`544` protocols, so an implementation conforms by
having the right attributes rather than by inheriting: ``NeperMesh``
is built from stat files and has no mesh, and forcing it into a common base
class would mean giving it attributes it cannot fill. They are
``runtime_checkable``, so ``isinstance`` answers whether an object conforms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BoundaryNetwork",
    "MeshedMicrostructure",
    "Microstructure",
    "TaggedPolycrystal",
]


@runtime_checkable
class Microstructure(Protocol):
    """What every microstructure in this package can say about itself.

    Deliberately thin. ``n_grains`` is *not* here: a :class:`NeperMesh` built by
    :meth:`~festim_microstructure.meshing.neper.NeperMesh.from_base` is read
    from the face/edge/vertex stat files, which say nothing about how many cells
    the tessellation had, so its report names the count only when it was given
    one.
    """

    def report(self) -> str:
        """A human-readable summary, one line per quantity."""
        ...


@runtime_checkable
class MeshedMicrostructure(Microstructure, Protocol):
    """A meshed polycrystal: what the subdomain and field helpers read.

    ``facet_tags``/``gb_tag`` may be ``None``: a builder that cannot tag the
    grain-boundary facets leaves them unset and
    :func:`~festim_microstructure.fem.subdomains.grain_boundary_network`
    locates the network geometrically through :meth:`locator` instead. The
    attributes must still *exist*, so that consumers can test them rather than
    reach for ``getattr(micro, "facet_tags", None)`` and silently take the
    wrong branch when the name is simply misspelled.
    """

    #: ``dolfinx.mesh.Mesh`` covering the whole cell.
    mesh: Any
    #: ``MeshTags`` on the cells, one value per grain id.
    cell_tags: Any
    #: ``MeshTags`` on the facets, or ``None`` when the network is geometric.
    facet_tags: Any
    #: The facet-tag value, or values, marking the grain boundaries; ``None``
    #: when the network is geometric. A Neper or EBSD mesh tags every
    #: tessellation face separately, so its network is a sequence of ids.
    gb_tag: Any
    #: One orientation per grain, indexed by ``grain_id - 1`` (radians).
    orientations: np.ndarray

    @property
    def n_grains(self) -> int:
        """Number of tagged grains, i.e. ``len(grain_ids)``."""
        ...

    @property
    def grain_ids(self) -> np.ndarray:
        """The cell-tag values, 1-based, in ascending order."""
        ...

    @property
    def tolerance(self) -> float:
        """Distance below which a point counts as lying on the network."""
        ...

    def locator(self, points: np.ndarray) -> np.ndarray:
        """``(3, n)`` points in, boolean "is on the network" out."""
        ...


@runtime_checkable
class BoundaryNetwork(Microstructure, Protocol):
    """A tessellation's grain-boundary topology, as Neper and EBSD report it.

    Entities are boundaries: faces in 3D, edges in 2D. Every array here is in
    id order, so ``theta[k]`` belongs to entity ``k + 1``.
    """

    @property
    def theta(self) -> np.ndarray:
        """Disorientation of every boundary, degrees, in id order."""
        ...

    @property
    def interior_mask(self) -> np.ndarray:
        """True for boundaries between two grains, i.e. not free surface."""
        ...

    @property
    def network_mask(self) -> np.ndarray:
        """:attr:`interior_mask` restricted to ``theta > theta_min``."""
        ...

    @property
    def network_ids(self) -> np.ndarray:
        """1-based ids of the boundaries in the network."""
        ...

    @property
    def network_measure(self) -> float:
        """Total size of the network: length in 2D, area in 3D."""
        ...


@dataclass
class TaggedPolycrystal:
    """A :class:`MeshedMicrostructure` made from a mesh and its tags.

    The adapter for a mesh this package did not generate: a Neper or EBSD mesh
    read with :func:`festim_microstructure.formats.msh4.read_mesh`, or one built
    by hand. The Voronoi builders return microstructures of their own and do not
    need it.

    Only ``mesh`` and ``cell_tags`` are required. ``grain_ids`` defaults to the
    distinct cell-tag values across all ranks, and ``orientations`` to zero for
    every grain, which is the right answer whenever the ``crystal_anisotropy``
    of :func:`~festim_microstructure.materials.crystal_diffusivity_field` is
    one: the lattice tensor is then isotropic and the angles do not enter the
    model. Pass the measured angles to give the grains a texture.

    Supply either ``facet_tags`` with ``gb_tag`` (one value or a sequence) or a
    ``network_locator``;
    :func:`~festim_microstructure.fem.subdomains.grain_boundary_network`
    prefers the tags.
    """

    mesh: Any
    cell_tags: Any
    facet_tags: Any = None
    gb_tag: Any = None
    grain_ids: Any = None
    orientations: Any = None
    tolerance: float = 0.0
    """Distance below which a point counts as lying on the network. Only read
    by the geometric route and by ``exports.averages.equilibrium_error``."""
    network_locator: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False
    )
    name: str = "tagged mesh"

    def __post_init__(self):
        """Populate grain IDs and orientation slots from the mesh when omitted."""
        if self.grain_ids is None:
            self.grain_ids = _global_tag_values(self.mesh, self.cell_tags)
        self.grain_ids = np.asarray(self.grain_ids, dtype=np.int32)
        if self.orientations is None:
            # Indexed by grain id - 1, so it is the largest id that sets the
            # length, not the number of grains: ids need not be contiguous.
            self.orientations = np.zeros(int(self.grain_ids.max()))

    @property
    def n_grains(self) -> int:
        """Number of tagged grains in the mesh."""
        return len(self.grain_ids)

    def locator(self, points: np.ndarray) -> np.ndarray:
        """Classify points using the configured geometric network locator."""
        if self.network_locator is None:
            raise TypeError(
                f"{type(self).__name__} {self.name!r} was built without a "
                "network_locator, so the network can only come from facet tags. "
                "Pass facet_tags and gb_tag, or a network_locator."
            )
        return self.network_locator(points)

    def report(self) -> str:
        """Summarize the grain count and network location method."""
        where = "facet tags" if self.facet_tags is not None else "geometric"
        return "\n".join(
            [
                f"microstructure: {self.name}",
                f"  grains (tagged pieces)         : {self.n_grains}",
                f"  network located by             : {where}",
            ]
        )


def _global_tag_values(mesh, tags) -> np.ndarray:
    """The distinct values of ``tags``, gathered over every rank, ascending.

    Taken locally, the answer is whatever cells this rank happens to own, and a
    model built from it would carry a different species list on every rank.
    """
    local = np.unique(np.asarray(tags.values))
    return np.unique(np.concatenate(mesh.comm.allgather(local)))
