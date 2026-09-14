"""Voronoi polycrystals: the tessellation, its mesh, and the grain tags.

Both classes implement
:class:`~festim_microstructure.microstructure.MeshedMicrostructure`, so
:func:`~festim_microstructure.resolved.build` accepts either. What used
to be two parallel classes with the same methods spelled differently is now one
base plus the parts that genuinely depend on the dimension.

The dataclasses are ``kw_only``: the fields are numerous and several are
optional, and keyword construction keeps a base-class field from dictating the
order of a subclass one. Build them with :meth:`create` rather than by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .geometry2d import (
    connected_components,
    near_segments,
    network_tensor,
    snap_segments,
    triple_junctions,
    voronoi_segments,
)
from .geometry3d import (
    connected_components_3d,
    near_faces,
    network_area,
    triple_lines,
    voronoi_faces,
)
from .gmsh_builder import (
    GB_TAG_3D,
    MeshSizing,
    build_mesh,
    build_mesh_3d,
    grain_tags_from_seeds,
)

__all__ = ["VoronoiMicrostructure", "VoronoiMicrostructure3D"]


@dataclass(kw_only=True)
class _VoronoiPolycrystal:
    """Fields and behaviour common to the 2D and 3D Voronoi microstructures."""

    #: Geometric dimension; set by each subclass.
    dim: ClassVar[int]

    size: float
    n_seeds: int
    seed: int
    mesh: Any
    cell_tags: Any
    grain_ids: np.ndarray
    orientations: np.ndarray  # one angle per grain id (indexed by id - 1), radians
    h_gb: float
    facet_tags: Any = None
    gb_tag: int | None = None

    @property
    def n_grains(self) -> int:
        return len(self.grain_ids)

    @property
    def domain_measure(self) -> float:
        """Area of the cell in 2D, volume in 3D."""
        return self.size**self.dim

    @property
    def tolerance(self) -> float:
        """Distance below which a point counts as lying on the network."""
        return 0.05 * self.h_gb

    def misorientation(self, grain_a, grain_b):
        """Misorientation angle between two grains, folded into ``[0, pi/2]``.

        The hook for misorientation-dependent boundary properties: a boundary's
        diffusivity or its exchange rate with the grains can be made a function
        of this instead of a constant.
        """
        delta = abs(self.orientations[grain_a - 1] - self.orientations[grain_b - 1])
        delta %= np.pi
        return min(delta, np.pi - delta)

    def _mesh_line(self):
        cells = self.mesh.topology.index_map(self.dim).size_global
        return (
            f"  mesh                           : {cells} cells,"
            f" h_gb = {1e9 * self.h_gb:.0f} nm"
        )

    @staticmethod
    def _sizing(h_gb, bulk_coarsening):
        return MeshSizing(h_gb=h_gb, h_bulk=bulk_coarsening * h_gb)


# --------------------------------------------------------------- 2D polycrystal


@dataclass(kw_only=True)
class VoronoiMicrostructure(_VoronoiPolycrystal):
    """A 2D Voronoi polycrystal, its conforming mesh, and the grain tags.

    The network is located geometrically (``facet_tags`` stays ``None``), so
    :meth:`locator` is the authority on what counts as a boundary.

    ``n_grains`` is generally larger than ``n_seeds``: the ridges of the
    periodically tiled seed set cut the box into more pieces than there are
    seeds inside it, and each piece is a grain in its own right.
    """

    dim: ClassVar[int] = 2

    aspect: float = 1.0
    segments: list = field(default_factory=list)

    @classmethod
    def create(
        cls,
        size,
        n_seeds,
        aspect=1.0,
        seed=0,
        cells_per_grain=14,
        bulk_coarsening=8.0,
        comm=None,
        msh_path=None,
    ):
        """Generate the tessellation and mesh it.

        Args:
            size: box side (m).
            n_seeds: number of Voronoi seeds.
            aspect: grain elongation along x.
            seed: rng seed.
            cells_per_grain: mesh cells across the *short* axis of a grain, which
                sets ``h_gb``.
            bulk_coarsening: ``h_bulk / h_gb``.
            comm: MPI communicator; defaults to MPI.COMM_WORLD.
            msh_path: optional Gmsh output path, in units of ``size``
                (the returned mesh is in metres).
        """
        rng = np.random.default_rng(seed)
        segments = voronoi_segments(n_seeds, size, rng, aspect)
        # the short axis of a grain: equal-area grains of aspect ratio `aspect`
        short_axis = np.sqrt(size**2 / n_seeds / aspect)
        h_gb = short_axis / cells_per_grain
        segments = snap_segments(segments, 0.1 * h_gb, size)
        mesh, cell_tags, n_grains = build_mesh(
            segments,
            size,
            cls._sizing(h_gb, bulk_coarsening),
            comm=comm,
            msh_path=msh_path,
        )
        # an untextured polycrystal: orientations uniform on [0, pi). Replace this
        # with the measured Euler angles to drive the model from EBSD or Neper.
        orientations = rng.uniform(0.0, np.pi, n_grains)
        return cls(
            size=size,
            n_seeds=n_seeds,
            aspect=aspect,
            seed=seed,
            segments=segments,
            mesh=mesh,
            cell_tags=cell_tags,
            grain_ids=np.arange(1, n_grains + 1),
            orientations=orientations,
            h_gb=h_gb,
        )

    def locator(self, points):
        return near_segments(points, self.segments, self.tolerance)

    @property
    def network_measure(self) -> float:
        """Total ridge length (m)."""
        return float(np.trace(network_tensor(self.segments)))

    #: Dimension-specific spelling of :attr:`network_measure`, kept for callers.
    @property
    def ridge_length(self) -> float:
        return self.network_measure

    @property
    def area(self) -> float:
        return self.domain_measure

    def area_fractions(self):
        """Each tagged piece's share of the cell, from the mesh triangle areas."""
        mesh = self.mesh
        mesh.topology.create_connectivity(2, 0)
        cells = mesh.topology.connectivity(2, 0).array.reshape(-1, 3)
        corners = mesh.geometry.x[cells][:, :, :2]
        edge_a = corners[:, 1] - corners[:, 0]
        edge_b = corners[:, 2] - corners[:, 0]
        areas = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
        tags = np.zeros(cells.shape[0], dtype=np.int64)
        tags[self.cell_tags.indices] = self.cell_tags.values
        per_grain = np.bincount(tags, weights=areas, minlength=self.n_grains + 1)
        return per_grain[1:] / per_grain[1:].sum()

    def oversized_pieces(self, factor=3.0):
        """Pieces holding more than ``factor`` times an average grain's area.

        The guard against a tessellation that did not actually tessellate. If a
        ridge fails to reach the edge of the cell, OpenCASCADE leaves it embedded
        inside a face rather than splitting it, and every grain it should have
        separated merges into one piece. Nothing else here notices: the ridge is
        still meshed, still in the network, and the check that every grain-grain
        facet is in the network still passes, because those facets now have the
        same tag on both sides. Only the *size* of the piece gives it away.
        """
        fractions = self.area_fractions()
        return fractions[fractions > factor / self.n_seeds]

    def report(self):
        tensor = network_tensor(self.segments)
        evals, evecs = np.linalg.eigh(tensor)
        length = self.network_measure
        lines = [
            f"microstructure: {self.n_seeds} grains in a {1e6 * self.size:.1f} um box,"
            f" aspect {self.aspect:g}, seed {self.seed}",
            f"  ridge segments                 : {len(self.segments)}",
            f"  triple junctions (interior)    : "
            f"{len(triple_junctions(self.segments, self.size))}",
            f"  connected components           : "
            f"{connected_components(self.segments, self.size)}",
            f"  ridge length / area            : "
            f"{length / self.domain_measure:.4g} 1/m",
            f"  network tensor eigenvalues     : "
            f"{evals[1] / length:.3f}, {evals[0] / length:.3f}"
            f" (of the trace)",
            f"  strong direction               : "
            f"({evecs[0, 1]:+.3f}, {evecs[1, 1]:+.3f})",
            f"  grains (tagged pieces)         : {self.n_grains}"
            f" (from {self.n_seeds} seeds)",
            f"  largest piece / average grain  : "
            f"{self.area_fractions().max() * self.n_seeds:.2f}"
            + (
                "   <-- SUSPECT: a ridge probably failed to split a face"
                if len(self.oversized_pieces()) > 0
                else ""
            ),
            self._mesh_line(),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------- 3D polycrystal


@dataclass(kw_only=True)
class VoronoiMicrostructure3D(_VoronoiPolycrystal):
    """A 3D Voronoi polycrystal meshed conformingly, with one cell tag per grain.

    The same contract as :class:`VoronoiMicrostructure`. The network is provided
    as facet tags (``facet_tags``, ``gb_tag``) rather than geometrically;
    :meth:`locator` is still available for the coverage check.

    The gmsh model is built in units of ``size`` and rescaled afterwards:
    OpenCASCADE's absolute tolerance (~1e-7) exceeds a sub-micron box.
    """

    dim: ClassVar[int] = 3

    seeds: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # metres
    faces: list = field(default_factory=list)  # polygons, metres

    @classmethod
    def create(
        cls, size, n_seeds, seed=0, cells_per_grain=8, bulk_coarsening=4.0, comm=None
    ):
        rng = np.random.default_rng(seed)
        seeds_unit = rng.uniform(0.0, 1.0, (n_seeds, 3))  # the draw voronoi_faces makes
        faces_unit = voronoi_faces(n_seeds, 1.0, np.random.default_rng(seed))
        spacing_unit = n_seeds ** (-1.0 / 3.0)
        h_gb_unit = spacing_unit / cells_per_grain
        mesh, facet_tags = build_mesh_3d(
            faces_unit, 1.0, cls._sizing(h_gb_unit, bulk_coarsening), comm=comm
        )
        mesh.geometry.x[:] *= size
        cell_tags, grain_ids = grain_tags_from_seeds(mesh, seeds_unit * size, size)
        # orientations are indexed by grain id, which here is the image-seed index
        orientations = rng.uniform(0.0, np.pi, int(grain_ids.max()))
        return cls(
            size=size,
            n_seeds=n_seeds,
            seed=seed,
            seeds=seeds_unit * size,
            faces=[f * size for f in faces_unit],
            mesh=mesh,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            gb_tag=GB_TAG_3D,
            grain_ids=grain_ids,
            orientations=orientations,
            h_gb=h_gb_unit * size,
        )

    def locator(self, points):
        return near_faces(points, self.faces, self.tolerance)

    @property
    def network_measure(self) -> float:
        """Total grain-boundary area (m^2)."""
        return network_area(self.faces)

    @property
    def volume(self) -> float:
        return self.domain_measure

    def report(self):
        lines, length, quadruple = triple_lines(self.faces)
        area = self.network_measure
        return "\n".join(
            [
                f"microstructure: {self.n_seeds} seeds in a "
                f"{1e6 * self.size:.1f} um cube, seed {self.seed}",
                f"  boundary polygons              : {len(self.faces)}",
                f"  connected components           : "
                f"{connected_components_3d(self.faces)}",
                f"  triple lines                   : {len(lines)} "
                f"(total length {length:.4g} m)",
                f"  quadruple points               : {len(quadruple)}",
                f"  boundary area / volume         : "
                f"{area / self.domain_measure:.4g} 1/m",
                f"  grains (tagged pieces)         : {self.n_grains}"
                f" (from {self.n_seeds} seeds)",
                self._mesh_line(),
            ]
        )
