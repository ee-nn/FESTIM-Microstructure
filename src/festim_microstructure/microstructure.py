"""The microstructure contract the builders implement and the models consume.

Three protocols, because there are really two different things in this package
that both get called "a microstructure":

* :class:`MeshedMicrostructure` -- a mesh with one tagged subdomain per grain,
  which is what :func:`festim_microstructure.models.resolved.build` needs.
  Implemented by the Voronoi builders.
* :class:`BoundaryNetwork` -- a tessellation topology with a per-entity
  disorientation and masks selecting the boundaries to keep. Implemented by the
  Neper and EBSD readers.

Both are declared as :pep:`544` protocols, so an implementation conforms by
having the right attributes rather than by inheriting: ``NeperMicrostructure``
is built from stat files and has no mesh, and forcing it into a common base
class would mean giving it attributes it cannot fill.

The protocols are ``runtime_checkable`` so that :func:`require` can name the
attributes that are missing at the call site, instead of letting the caller
find out through an ``AttributeError`` raised somewhere inside form assembly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BoundaryNetwork",
    "MeshedMicrostructure",
    "Microstructure",
    "missing_members",
    "require",
]


@runtime_checkable
class Microstructure(Protocol):
    """What every microstructure in this package can say about itself.

    Deliberately thin. ``n_grains`` is *not* here: a
    :class:`NeperMicrostructure` is read from the face/edge/vertex stat files
    and genuinely does not know how many cells the tessellation had, which is
    why its ``report`` takes the count as an argument.
    """

    def report(self) -> str:
        """A human-readable summary, one line per quantity."""


@runtime_checkable
class MeshedMicrostructure(Microstructure, Protocol):
    """A meshed polycrystal: the contract of :func:`...models.resolved.build`.

    ``facet_tags``/``gb_tag`` may be ``None``: a builder that cannot tag the
    grain-boundary facets leaves them unset and the network is located
    geometrically through :meth:`locator` instead. The attributes must still
    *exist*, so that consumers can test them rather than reach for
    ``getattr(micro, "facet_tags", None)`` and silently take the wrong branch
    when the name is simply misspelled.
    """

    #: ``dolfinx.mesh.Mesh`` covering the whole cell.
    mesh: Any
    #: ``MeshTags`` on the cells, one value per grain id.
    cell_tags: Any
    #: ``MeshTags`` on the facets, or ``None`` when the network is geometric.
    facet_tags: Any
    #: The facet-tag value marking the grain boundaries, or ``None``.
    gb_tag: int | None
    #: One orientation per grain, indexed by ``grain_id - 1`` (radians).
    orientations: np.ndarray

    @property
    def n_grains(self) -> int:
        """Number of tagged grains, i.e. ``len(grain_ids)``."""

    @property
    def grain_ids(self) -> np.ndarray:
        """The cell-tag values, 1-based, in ascending order."""

    @property
    def tolerance(self) -> float:
        """Distance below which a point counts as lying on the network."""

    def locator(self, points: np.ndarray) -> np.ndarray:
        """``(3, n)`` points in, boolean "is on the network" out."""


@runtime_checkable
class BoundaryNetwork(Microstructure, Protocol):
    """A tessellation's grain-boundary topology, as Neper and EBSD report it.

    Entities are boundaries: faces in 3D, edges in 2D. Every array here is in
    id order, so ``theta[k]`` belongs to entity ``k + 1``.
    """

    @property
    def theta(self) -> np.ndarray:
        """Disorientation of every boundary, degrees, in id order."""

    @property
    def interior_mask(self) -> np.ndarray:
        """True for boundaries between two grains, i.e. not free surface."""

    @property
    def network_mask(self) -> np.ndarray:
        """:attr:`interior_mask` restricted to ``theta > theta_min``."""

    @property
    def network_ids(self) -> np.ndarray:
        """1-based ids of the boundaries in the network."""

    @property
    def network_measure(self) -> float:
        """Total size of the network: length in 2D, area in 3D."""


def _declared_members(protocol: type) -> set[str]:
    """The attribute and method names a protocol class declares.

    Python 3.12 exposes this as ``__protocol_attrs__``; 3.10 and 3.11 do not,
    so walk the MRO instead. Protocol machinery is dunder-named and therefore
    already excluded by the leading-underscore test.
    """
    names: set[str] = set()
    for klass in protocol.__mro__:
        if klass.__name__ in ("Protocol", "Generic", "object"):
            continue
        names |= set(getattr(klass, "__annotations__", {}))
        names |= set(vars(klass))
    return {name for name in names if not name.startswith("_")}


def _has(obj: object, name: str) -> bool:
    """Does ``obj`` provide ``name``?

    ``hasattr`` is not enough when ``obj`` is a *class*: a dataclass field
    without a default is declared but has no class attribute, so checking a
    class directly would report every such field as missing. The declaration in
    ``__dataclass_fields__`` counts, which lets a caller validate a type before
    building an instance of it.
    """
    if hasattr(obj, name):
        return True
    return name in getattr(obj, "__dataclass_fields__", ())


def missing_members(obj: object, protocol: type) -> list[str]:
    """Names ``protocol`` declares that ``obj`` does not have.

    ``obj`` may be an instance or a class. ``runtime_checkable`` protocols only
    support ``isinstance``, which answers yes/no; for an error message worth
    reading the caller wants the names.
    """
    return sorted(name for name in _declared_members(protocol) if not _has(obj, name))


def require(obj: object, protocol: type, context: str = "") -> object:
    """Return ``obj``, or raise ``TypeError`` naming what it is missing.

    Called at the top of the functions that consume a microstructure, so that
    an incompatible one is rejected at the call site with the list of
    attributes it would need, rather than part-way through a build.
    """
    missing = missing_members(obj, protocol)
    if missing:
        where = f" {context}" if context else ""
        raise TypeError(
            f"{type(obj).__name__} does not satisfy {protocol.__name__}"
            f"{where}: missing {', '.join(missing)}. See "
            "festim_microstructure.microstructure for the contract."
        )
    return obj
