"""Conforming Gmsh meshes for a Voronoi tessellation, in 2D and 3D.

The only module in the subpackage that imports Gmsh or DOLFINx, and it does so
inside the functions, so that the geometry modules stay importable without a
FEniCS install.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GB_TAG_3D",
    "MeshSizing",
    "build_mesh",
    "build_mesh_3d",
    "grain_tags_from_seeds",
]

GB_TAG_3D = 2  # physical group of the grain-boundary facets written by build_mesh_3d


@dataclass(frozen=True)
class MeshSizing:
    """The graded element size both meshers apply around the boundaries.

    ``h_gb`` on the boundaries, growing to ``h_bulk`` in the grain interiors --
    the concentration field varies fastest next to a short circuit. The distance
    band over which it grows is expressed in multiples of ``h_gb`` because that
    is the only length the grading has to compare against.
    """

    h_gb: float
    h_bulk: float
    dist_min_factor: float = 2.0
    dist_max_factor: float = 12.0

    def apply(self, gmsh, entities, kind, scale=1.0):
        """Install the Distance/Threshold background field on ``entities``.

        ``kind`` is ``"CurvesList"`` in 2D and ``"SurfacesList"`` in 3D;
        ``scale`` is the length unit the model was built in.
        """
        field = gmsh.model.mesh.field
        field.add("Distance", 1)
        field.setNumbers(1, kind, entities)
        if kind == "SurfacesList":
            field.setNumber(1, "Sampling", 30)  # surface distance fields need samples
        field.add("Threshold", 2)
        field.setNumber(2, "InField", 1)
        field.setNumber(2, "SizeMin", self.h_gb / scale)
        field.setNumber(2, "SizeMax", self.h_bulk / scale)
        field.setNumber(2, "DistMin", self.dist_min_factor * self.h_gb / scale)
        field.setNumber(2, "DistMax", self.dist_max_factor * self.h_gb / scale)
        field.setAsBackgroundMesh(2)
        for opt in (
            "Mesh.MeshSizeFromPoints",
            "Mesh.MeshSizeFromCurvature",
            "Mesh.MeshSizeExtendFromBoundary",
        ):
            gmsh.option.setNumber(opt, 0)


def _unpack(result, *names):
    """dolfinx changed ``model_to_mesh`` from a tuple to a named result."""
    if hasattr(result, "mesh"):
        return tuple(getattr(result, n) for n in names)
    index = {"mesh": 0, "cell_tags": 1, "facet_tags": 2}
    return tuple(result[index[n]] for n in names)


def build_mesh(segments, size, sizing, comm=None, msh_path=None):
    """A triangular mesh of ``[0, size]^2`` whose facets lie on every ridge.

    Returns ``(mesh, cell_tags, n_grains)``, the tags marking each Voronoi cell
    with its own id, numbered from 1. With ``msh_path`` the gmsh model is also
    written to disk (in units of ``size``, see below) before being handed to
    dolfinx.

    ``occ.fragment`` is what guarantees that: it splits the rectangle along every
    segment, so the segments become model edges the mesher is obliged to follow.

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
    sizing.apply(gmsh, gb_curves, "CurvesList", scale=size)

    gmsh.model.mesh.generate(2)
    if msh_path is not None:
        gmsh.write(str(msh_path))
    result = model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    mesh, cell_tags = _unpack(result, "mesh", "cell_tags")
    mesh.geometry.x[:, :2] *= size
    return mesh, cell_tags, len(grain_surfaces)


def build_mesh_3d(faces, size, sizing, comm=None, msh_path=None):
    """A tet mesh of ``[0, size]^3`` whose facets conform to every polygon.

    Returns ``(mesh, facet_tags)``. Every fragment of the box is the same
    material and goes in one volume group (id 1); the grain-boundary facets are
    tagged :data:`GB_TAG_3D`, so the network is picked up from the facet tags by
    :class:`~festim_microstructure.fem.subdomains.TaggedGrainBoundaryNetwork`.
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

    # The 3D model is built in the same units the caller passed, so no rescaling.
    sizing.apply(gmsh, gb_surfaces, "SurfacesList", scale=1.0)

    gmsh.model.mesh.generate(3)
    if msh_path is not None:
        gmsh.write(str(msh_path))
    result = model_to_mesh(gmsh.model, comm, 0, gdim=3)
    gmsh.finalize()
    return _unpack(result, "mesh", "facet_tags")


def grain_tags_from_seeds(mesh, seeds, size):
    """Cell tags for a mesh that conforms to the periodic Voronoi of ``seeds``.

    Each cell takes the index of the nearest seed *image* (the seeds tiled over
    the 3x3x3 neighbourhood, as :func:`~.geometry3d.voronoi_faces` tiles them).
    Every facet of a conforming mesh lies on a bisector plane of two seeds, so a
    cell is never straddling and the classification is exact. Pieces cut off by
    image seeds get ids of their own, as the 2D mesher's ``fragment`` gives them.
    Returns ``(cell_tags, grain_ids)``; ids are 1-based and global across ranks.
    """
    import dolfinx
    from scipy.spatial import cKDTree

    seeds = np.asarray(seeds)
    dim = seeds.shape[1]
    shifts = np.stack(np.meshgrid(*[[-1, 0, 1]] * dim, indexing="ij"), -1)
    images = (seeds[None] + shifts.reshape(-1, dim)[:, None] * size).reshape(-1, dim)
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n = imap.size_local + imap.num_ghosts
    mid = dolfinx.mesh.compute_midpoints(mesh, tdim, np.arange(n, dtype=np.int32))
    _, idx = cKDTree(images).query(mid[:, :dim])
    values = (idx + 1).astype(np.int32)
    tags = dolfinx.mesh.meshtags(mesh, tdim, np.arange(n, dtype=np.int32), values)
    present = np.unique(np.concatenate(mesh.comm.allgather(np.unique(values))))
    return tags, present.astype(np.int32)
