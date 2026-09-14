"""Periodic 3D Voronoi faces, and the measures taken directly off them.

The mirror image of :mod:`.geometry2d`: ridges become planar polygons, shared
endpoints become shared polygon edges, and triple junctions become triple lines.
Pure NumPy/SciPy.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import Voronoi

from ._geometry import UnionFind, clip_polygon_to_box, order_ring, polygon_area

__all__ = [
    "connected_components_3d",
    "face_edges",
    "near_faces",
    "network_area",
    "triple_lines",
    "voronoi_faces",
]


def voronoi_faces(n_seeds, size, rng, min_face_area=1e-8):
    """Voronoi ridges of a periodically tiled seed set in 3D, clipped to the box.

    Returns a list of planar polygons, each an ``(n, 3)`` ring. Clipping can
    leave slivers at the walls; polygons below ``min_face_area`` are dropped.
    """
    pts = rng.uniform(0, size, (n_seeds, 3))
    tiled = np.vstack(
        [
            pts + np.array([dx, dy, dz]) * size
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
    )
    vor = Voronoi(tiled)
    faces, seen = [], set()
    for ridge, (i, j) in zip(vor.ridge_vertices, vor.ridge_points, strict=True):
        if -1 in ridge or len(ridge) < 3:
            continue
        # a ridge lies on the perpendicular bisector of its two seeds; projecting onto
        # that exact plane makes the polygon planar to machine precision, which is what
        # OCC's addPlaneSurface wants
        n = tiled[j] - tiled[i]
        n /= np.linalg.norm(n)
        d = 0.5 * (tiled[i] + tiled[j]) @ n
        poly = vor.vertices[ridge]
        poly = poly - np.outer(poly @ n - d, n)
        clipped = clip_polygon_to_box(order_ring(poly, n), size)
        if clipped is None or polygon_area(clipped) < min_face_area:
            continue
        key = tuple(np.round(clipped.mean(axis=0), 7))  # periodic images can coincide
        if key in seen:
            continue
        seen.add(key)
        faces.append(clipped)
    return faces


def near_faces(points, faces, tol=1e-7):
    """Vectorised test: is each point within ``tol`` of any polygon?

    Distance to a convex polygon is the distance to its plane when the projection
    falls inside the ring, and the distance to the nearest edge otherwise. Only
    needed for meshes that arrive without facet tags;
    :func:`~.gmsh_builder.build_mesh_3d` tags the grain-boundary facets, which is
    the safer route.
    """
    p = np.asarray(points)
    hit = np.zeros(p.shape[1], dtype=bool)
    for poly in faces:
        n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        n /= np.linalg.norm(n)
        rel = p.T - poly[0]
        s = rel @ n
        proj = p.T - np.outer(s, n)
        inside = np.ones(p.shape[1], dtype=bool)
        best = np.full(p.shape[1], np.inf)
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            e = b - a
            inside &= (np.cross(np.broadcast_to(e, proj.shape), proj - a) @ n) > -tol
            t = np.clip(((proj - a) @ e) / (e @ e), 0.0, 1.0)
            best = np.minimum(best, np.linalg.norm(proj - (a + np.outer(t, e)), axis=1))
        dist = np.where(inside, np.abs(s), np.hypot(s, best))
        hit |= dist < tol
    return hit


def face_edges(poly):
    for i in range(len(poly)):
        a = tuple(np.round(poly[i], 7))
        b = tuple(np.round(poly[(i + 1) % len(poly)], 7))
        yield tuple(sorted((a, b)))


def connected_components_3d(faces):
    """Union-find over shared polygon edges (the 2D version uses shared endpoints)."""
    uf = UnionFind()
    edge_to_face = defaultdict(list)
    for k, poly in enumerate(faces):
        uf.add(k)
        for e in face_edges(poly):
            edge_to_face[e].append(k)
    for shared in edge_to_face.values():
        for k in shared[1:]:
            uf.union(shared[0], k)
    return uf.n_components(range(len(faces)))


def triple_lines(faces):
    """Edges shared by three or more polygons, and the quadruple points they meet at.

    Returns ``(lines, total_length, quadruple_points)``.
    """
    edge_count = Counter(e for poly in faces for e in face_edges(poly))
    lines = [e for e, n in edge_count.items() if n >= 3]
    length = sum(float(np.linalg.norm(np.array(b) - np.array(a))) for a, b in lines)
    corner_count = Counter(v for e in lines for v in e)
    quadruple = [v for v, n in corner_count.items() if n >= 4]
    return lines, length, quadruple


def network_area(faces):
    """Total area of the boundary polygons: the 3D analogue of ridge length."""
    return float(sum(polygon_area(poly) for poly in faces))
