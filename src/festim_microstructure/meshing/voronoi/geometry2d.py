"""Periodic 2D Voronoi ridges, and the measures taken directly off them.

Pure NumPy/SciPy; nothing here needs Gmsh or DOLFINx.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial import Voronoi

from ._geometry import UnionFind, clip_to_box, quantise

__all__ = [
    "connected_components",
    "near_segments",
    "network_tensor",
    "snap_segments",
    "triple_junctions",
    "voronoi_segments",
]


def voronoi_segments(n_seeds, size, rng, aspect=1.0):
    """Ridges of a periodic Voronoi tessellation of the box ``[0, size]^2``.

    The seeds are tiled over the 3x3 neighbourhood before tessellating, so the
    ridges that reach the box edge are the ones a periodic tessellation would
    have, and the microstructure tiles.

    Args:
        n_seeds: number of grains in the box.
        size: box side (m).
        rng: a ``numpy`` generator, so the microstructure is reproducible.
        aspect: grain elongation along x. The tessellation is built in the
            coordinate ``(x / aspect, y)`` and mapped back, which stretches every
            grain by ``aspect`` along x without breaking periodicity.

    Returns:
        list of ``(p, q)`` endpoint pairs, in metres.
    """
    box = np.array([size / aspect, size])
    pts = np.column_stack(
        [rng.uniform(0, box[0], n_seeds), rng.uniform(0, box[1], n_seeds)]
    )
    tiled = np.vstack(
        [pts + np.array([dx, dy]) * box for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    )
    vor = Voronoi(tiled)
    stretch = np.array([aspect, 1.0])
    segments = []
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        clipped = clip_to_box(vor.vertices[a], vor.vertices[b], box)
        if clipped is None:
            continue
        p, q = (clipped[0] * stretch, clipped[1] * stretch)
        if np.linalg.norm(q - p) > 1e-12 * size:
            segments.append((p, q))
    return segments


def snap_segments(segments, tol, size):
    """Snap ridge endpoints to a grid of size ``tol`` and drop what collapses.

    A Voronoi tessellation produces the occasional ridge far shorter than any
    mesh size -- a quadruple junction that random seeds resolved into two triple
    junctions a nanometre apart. Meshing those is hopeless and dropping them
    outright would disconnect the network at that junction, which is exactly the
    thing the codim-1 formulation exists to get right. Snapping merges the two
    junctions into one instead, so connectivity survives.

    Endpoints that the clip put *on* the box edge are then put back on it
    exactly. ``size / tol`` is not a whole number, so rounding moves them off,
    and when it moves one inward it leaves a gap of a fraction of a nanometre
    between the ridge and the edge of the cell. OpenCASCADE will not split a face
    across a gap: the ridge ends up embedded inside the face instead of dividing
    it, and every grain that ridge should have separated silently merges into one
    enormous subdomain. It is a spectacular failure from a rounding error a
    thousand times smaller than an element.
    """
    snapped = []
    for p, q in segments:
        a = _clamp_to_box(np.round(p / tol) * tol, tol, size)
        b = _clamp_to_box(np.round(q / tol) * tol, tol, size)
        if np.linalg.norm(b - a) > 0.5 * tol:
            snapped.append((a, b))
    return snapped


def _clamp_to_box(point, tol, size):
    """Put a point that snapping nudged off the box edge back onto it exactly."""
    point = point.copy()
    for i in (0, 1):
        if abs(point[i]) < tol:
            point[i] = 0.0
        elif abs(point[i] - size) < tol:
            point[i] = size
    return point


def near_segments(points, segments, tol):
    """Vectorised test: is each point within ``tol`` of any segment?

    ``points`` is the ``(3, n)`` array DOLFINx hands to a locator.
    """
    px, py = points[0], points[1]
    hit = np.zeros(px.shape, dtype=bool)
    for p, q in segments:
        d = q - p
        t = np.clip(((px - p[0]) * d[0] + (py - p[1]) * d[1]) / (d @ d), 0.0, 1.0)
        dx, dy = px - (p[0] + t * d[0]), py - (p[1] + t * d[1])
        hit |= dx * dx + dy * dy < tol * tol
    return hit


def network_tensor(segments):
    """The second-moment tensor of the network, ``sum_i length_i * t_i (x) t_i``.

    This is the only geometric information a first-order (no-tortuosity) estimate
    of the effective diffusivity needs: a boundary conducts along its own tangent
    and not at all across it, so each segment contributes a rank-1 tensor. Its
    trace is the total ridge length, and its anisotropy is the anisotropy the
    homogenised tensor would have if the network were perfectly connected.

    Returns:
        ``(2, 2)`` array with units of length (m).
    """
    tensor = np.zeros((2, 2))
    for p, q in segments:
        d = q - p
        length = np.linalg.norm(d)
        if length == 0.0:
            continue
        t = d / length
        tensor += length * np.outer(t, t)
    return tensor


def connected_components(segments, size):
    """Number of connected components of the network, by union-find over endpoints."""
    uf = UnionFind()
    scale = 1e-9 * size
    keys = []
    for p, q in segments:
        a, b = quantise(p, scale), quantise(q, scale)
        uf.union(a, b)
        keys += [a, b]
    return uf.n_components(keys)


def triple_junctions(segments, size):
    """Interior points where three or more ridges meet."""
    scale = 1e-9 * size
    ends = Counter(quantise(p, scale) for seg in segments for p in seg)
    tol = 1e-9 * size
    return [
        np.array(k, dtype=float) * scale
        for k, n in ends.items()
        if n >= 3
        and tol < k[0] * scale < size - tol
        and tol < k[1] * scale < size - tol
    ]
