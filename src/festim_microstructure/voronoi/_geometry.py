"""Dimension-independent geometry shared by the 2D and 3D Voronoi builders.

Everything here is pure NumPy: no Gmsh, no DOLFINx, no tessellation. The 2D and
3D modules are near mirror images of each other, so whatever does not actually
depend on the dimension lives here rather than twice over.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "UnionFind",
    "clip_polygon_to_box",
    "clip_to_box",
    "order_ring",
    "polygon_area",
    "quantise",
]


class UnionFind:
    """Union-find over hashable keys, with path compression.

    Both network-connectivity counts (endpoints in 2D, shared polygon edges in
    3D) are the same disjoint-set problem, so they share this.
    """

    def __init__(self):
        self._parent: dict = {}

    def find(self, a):
        parent = self._parent
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(self, a, b):
        self._parent[self.find(a)] = self.find(b)

    def add(self, a):
        self.find(a)

    def n_components(self, keys=None):
        keys = self._parent if keys is None else keys
        return len({self.find(k) for k in keys})


def quantise(point, scale):
    """A hashable key for a point, rounded to ``scale``.

    Vertices that a tessellation produced independently but that are meant to
    be the same point agree only to floating-point noise, so connectivity is
    decided on a quantised key rather than on equality.
    """
    return tuple(np.round(np.asarray(point) / scale).astype(np.int64))


def clip_to_box(p, q, box):
    """Liang-Barsky clip of the segment ``pq`` to ``[0, box[0]] x [0, box[1]]``."""
    d = q - p
    t0, t1 = 0.0, 1.0
    for num, den in (
        (-d[0], p[0]),
        (d[0], box[0] - p[0]),
        (-d[1], p[1]),
        (d[1], box[1] - p[1]),
    ):
        if abs(num) < 1e-15:
            if den < 0:
                return None
            continue
        r = den / num
        if num < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    return None if t0 > t1 else (p + t0 * d, p + t1 * d)


def order_ring(poly, normal):
    """Sort the vertices of a planar polygon into a ring by angle about the centroid.

    Qhull returns 3D ridge vertices in cyclic order in every case tested here, but the
    scipy documentation only describes ``ridge_vertices`` as the indices of the
    vertices forming the ridge, without promising an order, so sorting costs nothing
    and removes the assumption.
    """
    c = poly.mean(axis=0)
    u = poly[0] - c
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return poly[np.argsort(np.arctan2((poly - c) @ v, (poly - c) @ u))]


def clip_polygon_to_box(poly, size):
    """Sutherland-Hodgman clip of a convex polygon to ``[0, size]^3``."""
    planes = []
    for k in range(3):
        n = np.zeros(3)
        n[k] = 1.0
        planes.append((n, 0.0))  # keeps x_k >= 0
        planes.append((-n, -size))  # keeps x_k <= size
    out = poly
    for n, d in planes:
        if len(out) < 3:
            return None
        s = out @ n - d
        new = []
        for i in range(len(out)):
            j = (i + 1) % len(out)
            if s[i] >= -1e-12:
                new.append(out[i])
            if (s[i] > 0) != (s[j] > 0) and abs(s[i] - s[j]) > 1e-15:
                t = s[i] / (s[i] - s[j])
                new.append(out[i] + t * (out[j] - out[i]))
        out = np.array(new) if new else np.zeros((0, 3))
    if len(out) < 3:
        return None
    ring = [out[0]]
    for p in out[1:]:
        if np.linalg.norm(p - ring[-1]) > 1e-10:
            ring.append(p)
    if len(ring) > 1 and np.linalg.norm(ring[-1] - ring[0]) < 1e-10:
        ring.pop()
    return np.array(ring) if len(ring) >= 3 else None


def polygon_area(poly):
    """Area of a planar polygon given as an ordered ring, by the fan rule."""
    c = poly.mean(axis=0)
    return 0.5 * sum(
        float(np.linalg.norm(np.cross(poly[i] - c, poly[(i + 1) % len(poly)] - c)))
        for i in range(len(poly))
    )
