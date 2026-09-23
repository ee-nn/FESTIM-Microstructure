"""Dimension-agnostic periodic Voronoi geometry.

Public operations accept ``dim`` and dispatch to the appropriate segment (2D)
or polygon (3D) implementation. Everything here is pure NumPy/SciPy; Gmsh and
DOLFINx belong in :mod:`.gmsh_builder`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Voronoi

__all__ = [
    "Junctions",
    "connected_components",
    "junctions",
    "near",
    "network_measure",
    "network_tensor",
    "snap",
    "tessellate",
]


def _check_dim(dim):
    """Require a supported spatial dimension of two or three."""
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")


class _UnionFind:
    """Track connected components using disjoint-set union."""

    def __init__(self):
        """Initialize an empty parent map."""
        self._parent = {}

    def find(self, item):
        """Return the representative of an item with path compression."""
        self._parent.setdefault(item, item)
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a, b):
        """Merge the components containing two items."""
        self._parent[self.find(a)] = self.find(b)

    def add(self, item):
        """Register an item in its own component if it is new."""
        self.find(item)

    def n_components(self, keys):
        """Count distinct components represented by the given keys."""
        return len({self.find(key) for key in keys})


def _quantise(point, scale):
    """Round a point to integer coordinates at the given scale."""
    return tuple(np.round(np.asarray(point) / scale).astype(np.int64))


def _clip_segment_to_box(p, q, box):
    """Liang-Barsky clipping in an arbitrary-dimensional axis-aligned box."""
    d = q - p
    t0, t1 = 0.0, 1.0
    for axis in range(len(box)):
        for num, den in ((-d[axis], p[axis]), (d[axis], box[axis] - p[axis])):
            if abs(num) < 1e-15:
                if den < 0:
                    return None
                continue
            ratio = den / num
            if num < 0:
                t0 = max(t0, ratio)
            else:
                t1 = min(t1, ratio)
    return None if t0 > t1 else (p + t0 * d, p + t1 * d)


def _order_ring(poly, normal):
    """Order coplanar polygon vertices around their center."""
    centre = poly.mean(axis=0)
    u = poly[0] - centre
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = np.arctan2((poly - centre) @ v, (poly - centre) @ u)
    return poly[np.argsort(angles)]


def _clip_polygon_to_box(poly, size):
    """Clip a 3D polygon to the positive box of the given size."""
    planes = []
    for axis in range(3):
        normal = np.zeros(3)
        normal[axis] = 1.0
        planes.extend(((normal, 0.0), (-normal, -size)))
    out = poly
    for normal, offset in planes:
        if len(out) < 3:
            return None
        signed = out @ normal - offset
        new = []
        for i in range(len(out)):
            j = (i + 1) % len(out)
            if signed[i] >= -1e-12:
                new.append(out[i])
            crosses = (signed[i] > 0) != (signed[j] > 0)
            if crosses and abs(signed[i] - signed[j]) > 1e-15:
                fraction = signed[i] / (signed[i] - signed[j])
                new.append(out[i] + fraction * (out[j] - out[i]))
        out = np.array(new) if new else np.zeros((0, 3))
    if len(out) < 3:
        return None
    ring = [out[0]]
    for point in out[1:]:
        if np.linalg.norm(point - ring[-1]) > 1e-10:
            ring.append(point)
    if len(ring) > 1 and np.linalg.norm(ring[-1] - ring[0]) < 1e-10:
        ring.pop()
    return np.array(ring) if len(ring) >= 3 else None


def _polygon_area(poly):
    """Compute the area of a planar polygon by center-based triangles."""
    centre = poly.mean(axis=0)
    return 0.5 * sum(
        float(
            np.linalg.norm(
                np.cross(poly[i] - centre, poly[(i + 1) % len(poly)] - centre)
            )
        )
        for i in range(len(poly))
    )


def _face_edges(poly, scale):
    """Yield quantized, direction-independent polygon edges."""
    for i in range(len(poly)):
        a = _quantise(poly[i], scale)
        b = _quantise(poly[(i + 1) % len(poly)], scale)
        yield tuple(sorted((a, b)))


def _network_scale(boundaries, size=None):
    """Choose a coordinate tolerance relative to network extent."""
    if size is None:
        if not boundaries:
            return 1.0
        points = np.vstack(boundaries)
        size = float(np.ptp(points, axis=0).max())
    return 1e-9 * size


def tessellate(n_seeds, size, rng, dim, aspect=1.0, min_boundary_measure=1e-8):
    """Return ``(seeds, boundaries)`` for a periodic Voronoi tessellation.

    Boundaries are endpoint pairs in 2D and ordered planar polygons in 3D.
    ``aspect`` stretches 2D grains along x; 3D elongation is not implemented.
    ``min_boundary_measure`` is relative to the unit box used by the 3D
    construction.
    """
    _check_dim(dim)
    if dim == 2:
        return _tessellate_2d(n_seeds, size, rng, aspect)
    if aspect != 1.0:
        raise NotImplementedError("aspect is currently supported only in 2D")
    return _tessellate_3d(n_seeds, size, rng, min_boundary_measure)


def _tessellate_2d(n_seeds, size, rng, aspect):
    """Construct periodic 2D Voronoi ridges clipped to the square.

    Args:
        n_seeds: Number of random seeds.
        size: Side length of the output square.
        rng: NumPy random generator.
        aspect: Stretch factor applied to the seed pattern along x.

    Returns:
        Seed coordinates and clipped boundary endpoint pairs.
    """
    box = np.array([size / aspect, size])
    raw = np.column_stack(
        [rng.uniform(0, box[0], n_seeds), rng.uniform(0, box[1], n_seeds)]
    )
    tiled = np.vstack(
        [raw + np.array([dx, dy]) * box for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    )
    vor = Voronoi(tiled)
    stretch = np.array([aspect, 1.0])
    boundaries = []
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        clipped = _clip_segment_to_box(vor.vertices[a], vor.vertices[b], box)
        if clipped is None:
            continue
        p, q = clipped[0] * stretch, clipped[1] * stretch
        if np.linalg.norm(q - p) > 1e-12 * size:
            boundaries.append((p, q))
    return raw * stretch, boundaries


def _tessellate_3d(n_seeds, size, rng, min_boundary_measure):
    """Construct periodic 3D Voronoi faces clipped to the cube.

    Args:
        n_seeds: Number of random seeds.
        size: Side length of the output cube.
        rng: NumPy random generator.
        min_boundary_measure: Minimum face area in the unit cube.

    Returns:
        Seed coordinates and clipped planar boundary polygons.
    """
    # Work in a unit cube so clipping tolerances do not depend on physical scale.
    seeds = rng.uniform(0.0, 1.0, (n_seeds, 3))
    tiled = np.vstack(
        [
            seeds + np.array([dx, dy, dz])
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
    )
    vor = Voronoi(tiled)
    boundaries, seen = [], set()
    for ridge, (i, j) in zip(vor.ridge_vertices, vor.ridge_points, strict=True):
        if -1 in ridge or len(ridge) < 3:
            continue
        normal = tiled[j] - tiled[i]
        normal /= np.linalg.norm(normal)
        offset = 0.5 * (tiled[i] + tiled[j]) @ normal
        poly = vor.vertices[ridge]
        poly = poly - np.outer(poly @ normal - offset, normal)
        clipped = _clip_polygon_to_box(_order_ring(poly, normal), 1.0)
        if clipped is None or _polygon_area(clipped) < min_boundary_measure:
            continue
        key = tuple(np.round(clipped.mean(axis=0), 7))
        if key not in seen:
            seen.add(key)
            boundaries.append(clipped * size)
    return seeds * size, boundaries


def snap(boundaries, tol, size, dim):
    """Snap boundary vertices to a grid; currently needed only in 2D."""
    _check_dim(dim)
    if dim == 3:
        return boundaries
    snapped = []
    for p, q in boundaries:
        a = _clamp_to_box(np.round(p / tol) * tol, tol, size)
        b = _clamp_to_box(np.round(q / tol) * tol, tol, size)
        if np.linalg.norm(b - a) > 0.5 * tol:
            snapped.append((a, b))
    return snapped


def _clamp_to_box(point, tol, size):
    """Snap coordinates close to the box walls onto the walls."""
    point = point.copy()
    for axis in range(len(point)):
        if abs(point[axis]) < tol:
            point[axis] = 0.0
        elif abs(point[axis] - size) < tol:
            point[axis] = size
    return point


def near(points, boundaries, tol, dim):
    """Return whether each point is within ``tol`` of the boundary network."""
    _check_dim(dim)
    return (
        _near_segments(points, boundaries, tol)
        if dim == 2
        else _near_polygons(points, boundaries, tol)
    )


def _near_segments(points, boundaries, tol):
    """Mark points closer than ``tol`` to any 2D segment."""
    p = np.asarray(points)[:2]
    hit = np.zeros(p.shape[1], dtype=bool)
    for a, b in boundaries:
        direction = b - a
        t = np.clip(((p.T - a) @ direction) / (direction @ direction), 0.0, 1.0)
        closest = a + np.outer(t, direction)
        hit |= np.sum((p.T - closest) ** 2, axis=1) < tol**2
    return hit


def _near_polygons(points, boundaries, tol):
    """Mark points closer than ``tol`` to any 3D polygon."""
    points = np.asarray(points)
    hit = np.zeros(points.shape[1], dtype=bool)
    for poly in boundaries:
        normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        normal /= np.linalg.norm(normal)
        rel = points.T - poly[0]
        signed = rel @ normal
        projected = points.T - np.outer(signed, normal)
        inside = np.ones(points.shape[1], dtype=bool)
        best = np.full(points.shape[1], np.inf)
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            edge = b - a
            cross = np.cross(np.broadcast_to(edge, projected.shape), projected - a)
            inside &= (cross @ normal) > -tol
            t = np.clip(((projected - a) @ edge) / (edge @ edge), 0.0, 1.0)
            closest = a + np.outer(t, edge)
            best = np.minimum(best, np.linalg.norm(projected - closest, axis=1))
        hit |= np.where(inside, np.abs(signed), np.hypot(signed, best)) < tol
    return hit


def network_measure(boundaries, dim):
    """Total boundary length in 2D or area in 3D."""
    _check_dim(dim)
    if dim == 2:
        return float(sum(np.linalg.norm(q - p) for p, q in boundaries))
    return float(sum(_polygon_area(poly) for poly in boundaries))


def network_tensor(boundaries, dim):
    """Tangential second-moment tensor of the boundary network."""
    _check_dim(dim)
    tensor = np.zeros((dim, dim))
    if dim == 2:
        for p, q in boundaries:
            direction = q - p
            length = np.linalg.norm(direction)
            if length:
                tangent = direction / length
                tensor += length * np.outer(tangent, tangent)
        return tensor
    identity = np.eye(3)
    for poly in boundaries:
        normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        normal /= np.linalg.norm(normal)
        tensor += _polygon_area(poly) * (identity - np.outer(normal, normal))
    return tensor


def connected_components(boundaries, dim, size=None):
    """Number of components joined at endpoints (2D) or polygon edges (3D)."""
    _check_dim(dim)
    uf = _UnionFind()
    if dim == 2:
        if size is None:
            raise ValueError("size is required for 2D connectivity")
        scale = 1e-9 * size
        keys = []
        for p, q in boundaries:
            a, b = _quantise(p, scale), _quantise(q, scale)
            uf.union(a, b)
            keys.extend((a, b))
        return uf.n_components(keys)
    scale = _network_scale(boundaries, size)
    edge_to_boundary = defaultdict(list)
    for index, poly in enumerate(boundaries):
        uf.add(index)
        for edge in _face_edges(poly, scale):
            edge_to_boundary[edge].append(index)
    for shared in edge_to_boundary.values():
        for index in shared[1:]:
            uf.union(shared[0], index)
    return uf.n_components(range(len(boundaries)))


@dataclass(frozen=True)
class Junctions:
    """Line and point junctions; unavailable kinds are represented as empty."""

    lines: list
    points: list
    line_measure: float = 0.0


def junctions(boundaries, dim, size=None):
    """Return network junctions with the same result shape in 2D and 3D."""
    _check_dim(dim)
    if dim == 2:
        if size is None:
            raise ValueError("size is required for 2D junction detection")
        scale = 1e-9 * size
        ends = Counter(_quantise(point, scale) for edge in boundaries for point in edge)
        tol = 1e-9 * size
        points = [
            np.array(key, dtype=float) * scale
            for key, count in ends.items()
            if count >= 3
            and all(tol < coordinate * scale < size - tol for coordinate in key)
        ]
        return Junctions(lines=[], points=points)
    scale = _network_scale(boundaries, size)
    edge_count = Counter(
        edge for poly in boundaries for edge in _face_edges(poly, scale)
    )
    line_keys = [edge for edge, count in edge_count.items() if count >= 3]
    lines = [
        tuple(np.asarray(vertex, dtype=float) * scale for vertex in edge)
        for edge in line_keys
    ]
    length = sum(float(np.linalg.norm(np.array(b) - np.array(a))) for a, b in lines)
    corner_count = Counter(vertex for edge in line_keys for vertex in edge)
    points = [
        np.asarray(vertex, dtype=float) * scale
        for vertex, count in corner_count.items()
        if count >= 4
    ]
    return Junctions(lines=lines, points=points, line_measure=length)
