"""Dimension-agnostic Voronoi polycrystals and their conforming meshes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import (
    connected_components,
    junctions,
    near,
    network_measure,
    network_tensor,
    snap,
    tessellate,
)
from .gmsh_builder import MeshSizing, build_mesh

__all__ = ["VoronoiMicrostructure"]


@dataclass(kw_only=True)
class VoronoiMicrostructure:
    """A periodic 2D or 3D Voronoi polycrystal.

    Use :meth:`create` with ``dim=2`` or ``dim=3``. The public representation is
    the same in both cases: ``boundaries`` contains codimension-one entities,
    represented by endpoint pairs in 2D and planar polygon rings in 3D.
    """

    dim: int
    size: float
    n_seeds: int
    seed: int
    mesh: Any
    cell_tags: Any
    grain_ids: np.ndarray
    orientations: np.ndarray
    h_gb: float
    seeds: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    boundaries: list = field(default_factory=list)
    aspect: float = 1.0
    facet_tags: Any = None
    gb_tag: int | None = None

    @classmethod
    def create(
        cls,
        size,
        n_seeds,
        dim=2,
        aspect=1.0,
        seed=0,
        cells_per_grain=None,
        bulk_coarsening=None,
        comm=None,
        msh_path=None,
    ):
        """Generate and mesh a periodic Voronoi microstructure.

        ``aspect`` controls elongation along x and is currently available in
        2D only. Dimension-specific meshing defaults preserve the established
        resolutions: 14 cells per grain and 8x bulk coarsening in 2D, versus
        8 cells per grain and 4x bulk coarsening in 3D.
        """
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim!r}")
        if cells_per_grain is None:
            cells_per_grain = 14 if dim == 2 else 8
        if bulk_coarsening is None:
            bulk_coarsening = 8.0 if dim == 2 else 4.0

        rng = np.random.default_rng(seed)
        seeds, boundaries = tessellate(n_seeds, size, rng, dim, aspect=aspect)
        spacing = (size**dim / n_seeds / aspect) ** (1.0 / dim)
        h_gb = spacing / cells_per_grain
        boundaries = snap(boundaries, 0.1 * h_gb, size, dim)
        sizing = MeshSizing(h_gb=h_gb, h_bulk=bulk_coarsening * h_gb)
        mesh_data = build_mesh(
            boundaries,
            size,
            sizing,
            dim,
            seeds=seeds,
            comm=comm,
            msh_path=msh_path,
        )
        orientations = rng.uniform(0.0, np.pi, int(mesh_data.grain_ids.max()))
        return cls(
            dim=dim,
            size=size,
            n_seeds=n_seeds,
            seed=seed,
            mesh=mesh_data.mesh,
            cell_tags=mesh_data.cell_tags,
            facet_tags=mesh_data.facet_tags,
            gb_tag=mesh_data.gb_tag,
            grain_ids=mesh_data.grain_ids,
            orientations=orientations,
            h_gb=h_gb,
            seeds=seeds,
            boundaries=boundaries,
            aspect=aspect,
        )

    @property
    def n_grains(self):
        return len(self.grain_ids)

    @property
    def domain_measure(self):
        """Area in 2D or volume in 3D."""
        return self.size**self.dim

    @property
    def tolerance(self):
        return 0.05 * self.h_gb

    @property
    def network_measure(self):
        """Total boundary length in 2D or area in 3D."""
        return network_measure(self.boundaries, self.dim)

    def locator(self, points):
        return near(points, self.boundaries, self.tolerance, self.dim)

    def misorientation(self, grain_a, grain_b):
        """Misorientation angle folded into ``[0, pi/2]``."""
        delta = abs(self.orientations[grain_a - 1] - self.orientations[grain_b - 1])
        delta %= np.pi
        return min(delta, np.pi - delta)

    def grain_fractions(self):
        """Each tagged grain's fraction of the domain measure."""
        mesh = self.mesh
        mesh.topology.create_connectivity(self.dim, 0)
        cells = mesh.topology.connectivity(self.dim, 0).array.reshape(-1, self.dim + 1)
        corners = mesh.geometry.x[cells][:, :, : self.dim]
        edges = corners[:, 1:] - corners[:, :1]
        measures = np.abs(np.linalg.det(edges)) / math.factorial(self.dim)
        tags = np.zeros(cells.shape[0], dtype=np.int64)
        tags[self.cell_tags.indices] = self.cell_tags.values
        n_tags = int(self.grain_ids.max()) + 1
        per_grain = np.bincount(tags, weights=measures, minlength=n_tags)
        selected = per_grain[self.grain_ids]
        return selected / selected.sum()

    def oversized_grains(self, factor=3.0):
        """Grain fractions exceeding ``factor`` times the expected average."""
        fractions = self.grain_fractions()
        return fractions[fractions > factor / self.n_seeds]

    def report(self):
        measure = self.network_measure
        tensor = network_tensor(self.boundaries, self.dim)
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        topology = junctions(self.boundaries, self.dim, size=self.size)
        fractions = self.grain_fractions()
        unit = "square" if self.dim == 2 else "cube"
        entity = "segments" if self.dim == 2 else "polygons"
        lines = [
            f"microstructure: {self.n_seeds} seeds in a "
            f"{1e6 * self.size:.1f} um {unit}, dimension {self.dim}, seed {self.seed}",
            f"  boundary {entity:<17}: {len(self.boundaries)}",
            f"  connected components           : "
            f"{connected_components(self.boundaries, self.dim, size=self.size)}",
        ]
        if self.dim == 2:
            lines.append(f"  triple junctions (interior)    : {len(topology.points)}")
        else:
            lines.extend(
                [
                    f"  triple lines                    : {len(topology.lines)} "
                    f"(total length {topology.line_measure:.4g} m)",
                    f"  quadruple points                : {len(topology.points)}",
                ]
            )
        trace = np.trace(tensor)
        lines.extend(
            [
                f"  boundary measure / domain       : "
                f"{measure / self.domain_measure:.4g} 1/m",
                "  network tensor eigenvalues      : "
                + ", ".join(f"{value / trace:.3f}" for value in eigenvalues[::-1])
                + " (of the trace)",
                "  strong direction                : ("
                + ", ".join(f"{x:+.3f}" for x in eigenvectors[:, -1])
                + ")",
                f"  grains (tagged pieces)          : {self.n_grains} "
                f"(from {self.n_seeds} seeds)",
                f"  largest piece / average grain   : "
                f"{fractions.max() * self.n_seeds:.2f}"
                + (
                    "   <-- SUSPECT: a boundary probably failed to split the domain"
                    if len(self.oversized_grains()) > 0
                    else ""
                ),
                f"  mesh                            : "
                f"{self.mesh.topology.index_map(self.dim).size_global} cells, "
                f"h_gb = {1e9 * self.h_gb:.0f} nm",
            ]
        )
        return "\n".join(lines)
