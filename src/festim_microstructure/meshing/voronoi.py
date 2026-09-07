"""Periodic Voronoi microstructures and the meshes that conform to them.

2D: ridges are segments, clipped to the box by Liang-Barsky, and the mesh is
built with ``occ.fragment`` so its facets lie on every ridge
(:func:`voronoi_segments`, :func:`build_mesh`, :class:`VoronoiMicrostructure`).

3D: a ridge is a convex polygon, clipped by Sutherland-Hodgman against the six
half-spaces of the box, and the grain-boundary facets are tagged with a physical
group so they can be picked up from the facet tags
(:func:`voronoi_faces`, :func:`build_mesh_3d`).

The grain boundaries are *lines*, not thin bands: nothing here resolves the
physical boundary width ``delta``. That width only ever appears as a
coefficient, which is what lets a 1 nm boundary sit inside a 20 um cell. What
the mesh must do is put facets exactly on the ridges, so that a codim-1 subdomain
can be built from them.

Grains can be elongated (``aspect``): the seeds are Voronoi-tessellated in a
coordinate stretched along x, so the grains come out with an aspect ratio while
staying a proper periodic tessellation. An equiaxed tessellation homogenises to a
nearly isotropic tensor, so the elongation is what makes the anisotropy
identification worth doing.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Voronoi

# gmsh, dolfinx and mpi4py are imported inside the two mesh builders, so that the
# tessellation geometry above them stays importable (and testable) without them.

__all__ = [
    "GB_TAG_3D",
    "VoronoiMicrostructure",
    "build_mesh",
    "build_mesh_3d",
    "clip_to_box",
    "clip_polygon_to_box",
    "connected_components",
    "connected_components_3d",
    "near_faces",
    "near_segments",
    "network_tensor",
    "polygon_area",
    "snap_segments",
    "triple_junctions",
    "triple_lines",
    "voronoi_faces",
    "voronoi_segments",
]

GB_TAG_3D = 2  # physical group of the grain-boundary facets written by build_mesh_3d


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
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    scale = 1e-9 * size
    for p, q in segments:
        a = tuple(np.round(p / scale).astype(np.int64))
        b = tuple(np.round(q / scale).astype(np.int64))
        parent[find(a)] = find(b)
    return len({find(k) for k in parent})


def triple_junctions(segments, size):
    """Interior points where three or more ridges meet."""
    scale = 1e-9 * size
    ends = Counter(
        tuple(np.round(p / scale).astype(np.int64)) for seg in segments for p in seg
    )
    tol = 1e-9 * size
    return [
        np.array(k, dtype=float) * scale
        for k, n in ends.items()
        if n >= 3
        and tol < k[0] * scale < size - tol
        and tol < k[1] * scale < size - tol
    ]


def build_mesh(segments, size, h_gb, h_bulk, comm=None, msh_path=None):
    """A triangular mesh of ``[0, size]^2`` whose facets lie on every ridge.

    Returns ``(mesh, cell_tags, n_grains)``, the tags marking each Voronoi cell
    with its own id, numbered from 1. With ``msh_path`` the gmsh model is also
    written to disk (in units of ``size``, see below) before being handed to
    dolfinx.

    ``occ.fragment`` is what guarantees that: it splits the rectangle along every
    segment, so the segments become model edges the mesher is obliged to follow.
    The mesh is then graded from ``h_gb`` on the ridges to ``h_bulk`` in the grain
    interiors -- the concentration field varies fastest next to a short circuit.

    The model is built in units of ``size`` and the mesh scaled back afterwards.
    OpenCASCADE compares points against an absolute tolerance of about 1e-7, so a
    geometry stated in metres and tens of microns across is entirely below its
    resolution -- every point would be a duplicate of every other.
    """
    from mpi4py import MPI

    import gmsh
    from dolfinx.io.gmsh import model_to_mesh

    comm = MPI.COMM_WORLD if comm is None else comm
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("polycrystal")
    occ = gmsh.model.occ

    rect = occ.addRectangle(0, 0, 0, 1.0, 1.0)
    points = {}

    def point(xy):
        # one gmsh point per ridge endpoint, so that segments meeting at a
        # junction share it and fragment has nothing to reconcile
        key = tuple(np.round(xy / size, 9))
        if key not in points:
            points[key] = occ.addPoint(key[0], key[1], 0)
        return points[key]

    lines = [occ.addLine(point(p), point(q)) for p, q in segments]
    _, out_map = occ.fragment([(2, rect)], [(1, ln) for ln in lines])
    occ.synchronize()

    # one physical group per Voronoi cell, so that every grain becomes a volume
    # subdomain of its own. ``fragment`` splits the rectangle along every ridge
    # either way; tagging the pieces separately is the whole difference between a
    # polycrystal whose lattice field is continuous and one whose grains can each
    # hold their own concentration, orientation and traps.
    grain_surfaces = [tag for (dim, tag) in out_map[0] if dim == 2]
    for i, tag in enumerate(grain_surfaces):
        gmsh.model.addPhysicalGroup(2, [tag], i + 1)

    gb_curves = [tag for entry in out_map[1:] for (dim, tag) in entry if dim == 1]

    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "CurvesList", gb_curves)
    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", h_gb / size)
    field.setNumber(2, "SizeMax", h_bulk / size)
    field.setNumber(2, "DistMin", 2 * h_gb / size)
    field.setNumber(2, "DistMax", 12 * h_gb / size)
    field.setAsBackgroundMesh(2)
    for opt in (
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.MeshSizeExtendFromBoundary",
    ):
        gmsh.option.setNumber(opt, 0)

    gmsh.model.mesh.generate(2)
    if msh_path is not None:
        gmsh.write(str(msh_path))
    result = model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    mesh = result.mesh if hasattr(result, "mesh") else result[0]
    cell_tags = result.cell_tags if hasattr(result, "cell_tags") else result[1]
    mesh.geometry.x[:, :2] *= size
    return mesh, cell_tags, len(grain_surfaces)


# ----------------------------------------------------------------------- 3D


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
    needed for meshes that arrive without facet tags; :func:`build_mesh_3d` tags
    the grain-boundary facets, which is the safer route.
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
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edge_to_face = defaultdict(list)
    for k, poly in enumerate(faces):
        find(k)
        for e in face_edges(poly):
            edge_to_face[e].append(k)
    for shared in edge_to_face.values():
        for k in shared[1:]:
            parent[find(shared[0])] = find(k)
    return len({find(k) for k in range(len(faces))})


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


def build_mesh_3d(faces, size, h_gb, h_bulk, comm=None, msh_path=None):
    """A tet mesh of ``[0, size]^3`` whose facets conform to every polygon.

    Returns ``(mesh, facet_tags)``. Every fragment of the box is the same
    material and goes in one volume group (id 1); the grain-boundary facets are
    tagged :data:`GB_TAG_3D`, so the network is picked up from the facet tags by
    :class:`~festim_microstructure.subdomains.TaggedGrainBoundaryNetwork`.
    """
    from mpi4py import MPI

    import gmsh
    from dolfinx.io.gmsh import model_to_mesh

    comm = MPI.COMM_WORLD if comm is None else comm
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("polycrystal3d")
    occ = gmsh.model.occ

    box = occ.addBox(0, 0, 0, size, size, size)
    surfaces = []
    for poly in faces:
        pts = [occ.addPoint(*p) for p in poly]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        surfaces.append(occ.addPlaneSurface([occ.addCurveLoop(lines)]))
    # fragment forces the box to be split along every polygon, so the generated mesh
    # has facets lying exactly on the grain boundaries
    out, out_map = occ.fragment([(3, box)], [(2, s) for s in surfaces])
    occ.synchronize()

    gb_surfaces = sorted({t for entry in out_map[1:] for (d, t) in entry if d == 2})
    gmsh.model.addPhysicalGroup(3, [t for (d, t) in out if d == 3], 1)
    gmsh.model.addPhysicalGroup(2, gb_surfaces, GB_TAG_3D)

    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "SurfacesList", gb_surfaces)
    field.setNumber(1, "Sampling", 30)  # surface distance fields need sample points
    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", h_gb)
    field.setNumber(2, "SizeMax", h_bulk)
    field.setNumber(2, "DistMin", h_gb)
    field.setNumber(2, "DistMax", 5 * h_gb)
    field.setAsBackgroundMesh(2)
    for opt in (
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.MeshSizeExtendFromBoundary",
    ):
        gmsh.option.setNumber(opt, 0)

    gmsh.model.mesh.generate(3)
    if msh_path is not None:
        gmsh.write(str(msh_path))
    result = model_to_mesh(gmsh.model, comm, 0, gdim=3)
    gmsh.finalize()
    if hasattr(result, "mesh"):
        return result.mesh, result.facet_tags
    return result[0], result[2]


# --------------------------------------------------------------- 2D polycrystal


@dataclass
class VoronoiMicrostructure:
    """A Voronoi polycrystal, the mesh that conforms to it, and the grain tags.

    This is the reference implementation of the microstructure contract used by
    :mod:`festim_microstructure.models.resolved`: ``mesh``, ``cell_tags``,
    ``grain_ids``, ``orientations``, a network ``locator`` and its ``tolerance``.

    ``n_grains`` is generally larger than ``n_seeds``: the ridges of the
    periodically tiled seed set cut the box into more pieces than there are seeds
    inside it, and each piece is a grain in its own right.
    """

    size: float
    n_seeds: int
    aspect: float
    seed: int
    segments: list
    mesh: object
    cell_tags: object
    n_grains: int
    orientations: np.ndarray  # one angle per grain, radians
    h_gb: float

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
        """
        rng = np.random.default_rng(seed)
        segments = voronoi_segments(n_seeds, size, rng, aspect)
        # the short axis of a grain: equal-area grains of aspect ratio `aspect`
        short_axis = np.sqrt(size**2 / n_seeds / aspect)
        h_gb = short_axis / cells_per_grain
        segments = snap_segments(segments, 0.1 * h_gb, size)
        mesh, cell_tags, n_grains = build_mesh(
            segments, size, h_gb, bulk_coarsening * h_gb, comm=comm
        )
        # an untextured polycrystal: orientations uniform on [0, pi). Replace this
        # with the measured Euler angles to drive the model from EBSD or Neper.
        orientations = rng.uniform(0.0, np.pi, n_grains)
        return cls(
            size,
            n_seeds,
            aspect,
            seed,
            segments,
            mesh,
            cell_tags,
            n_grains,
            orientations,
            h_gb,
        )

    @property
    def grain_ids(self):
        return np.arange(1, self.n_grains + 1)

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

    def misorientation(self, grain_a, grain_b):
        """Misorientation angle between two grains, folded into ``[0, pi/2]``.

        The hook for misorientation-dependent boundary properties: a boundary's
        diffusivity or its exchange rate with the grains can be made a function of
        this instead of a constant.
        """
        delta = abs(self.orientations[grain_a - 1] - self.orientations[grain_b - 1])
        delta %= np.pi
        return min(delta, np.pi - delta)

    @property
    def area(self):
        return self.size**2

    @property
    def ridge_length(self):
        return float(np.trace(network_tensor(self.segments)))

    @property
    def tolerance(self):
        """Distance below which a point counts as lying on the network."""
        return 0.05 * self.h_gb

    def locator(self, points):
        return near_segments(points, self.segments, self.tolerance)

    def report(self):
        tensor = network_tensor(self.segments)
        evals, evecs = np.linalg.eigh(tensor)
        lines = [
            f"microstructure: {self.n_seeds} grains in a {1e6 * self.size:.1f} um box,"
            f" aspect {self.aspect:g}, seed {self.seed}",
            f"  ridge segments                 : {len(self.segments)}",
            f"  triple junctions (interior)    : "
            f"{len(triple_junctions(self.segments, self.size))}",
            f"  connected components           : "
            f"{connected_components(self.segments, self.size)}",
            f"  ridge length / area            : "
            f"{self.ridge_length / self.area:.4g} 1/m",
            f"  network tensor eigenvalues     : "
            f"{evals[1] / self.ridge_length:.3f}, {evals[0] / self.ridge_length:.3f}"
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
            f"  mesh                           : "
            f"{self.mesh.topology.index_map(2).size_global} cells,"
            f" h_gb = {1e9 * self.h_gb:.0f} nm",
        ]
        return "\n".join(lines)


def main(argv=None):
    """``fm-voronoi``: generate a 2D Voronoi polycrystal and report on it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-grains", type=int, default=48, help="Voronoi seeds")
    parser.add_argument("--domain-size", type=float, default=20e-6, help="box side (m)")
    parser.add_argument("--aspect", type=float, default=1.0, help="grain elongation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cells-per-grain", type=int, default=10)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="write the gmsh model here (.msh, in units of --domain-size)",
    )
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    segments = voronoi_segments(args.n_grains, args.domain_size, rng, args.aspect)
    short_axis = np.sqrt(args.domain_size**2 / args.n_grains / args.aspect)
    h_gb = short_axis / args.cells_per_grain
    segments = snap_segments(segments, 0.1 * h_gb, args.domain_size)
    mesh, cell_tags, n_grains = build_mesh(
        segments, args.domain_size, h_gb, 8.0 * h_gb, msh_path=args.out
    )
    micro = VoronoiMicrostructure(
        args.domain_size,
        args.n_grains,
        args.aspect,
        args.seed,
        segments,
        mesh,
        cell_tags,
        n_grains,
        rng.uniform(0.0, np.pi, n_grains),
        h_gb,
    )
    print(micro.report())
    if args.out:
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
