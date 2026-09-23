"""Read Gmsh meshes and the accompanying Neper entity-statistics files."""

from __future__ import annotations

import numpy as np

__all__ = ["StatFile", "read_mesh", "read_msh4"]


class StatFile:
    """One Neper .st* file: scalar keys in columns, entities in id order.

    Ids are 1-based throughout Neper, so ``self.values[k]`` is entity ``k + 1``
    and :meth:`ids` converts a boolean mask back into ids.
    """

    def __init__(self, path, keys):
        raw = np.loadtxt(path, ndmin=2)
        if raw.shape[1] != len(keys):
            raise ValueError(
                f"{path} has {raw.shape[1]} columns but {len(keys)} keys were "
                f"expected ({', '.join(keys)}); the -stat option and the key "
                "tuple have drifted apart"
            )
        self.values = {k: raw[:, i] for i, k in enumerate(keys)}
        self.n = raw.shape[0]

    def __getitem__(self, key):
        return self.values[key]

    @staticmethod
    def ids(mask):
        return np.flatnonzero(mask).astype(np.int32) + 1


def read_mesh(base, gdim, comm=None, rank=0):
    """Read a Neper ``.msh4`` into dolfinx.

    ``cell_tags`` carry the polyhedron / raster-cell (grain) id and
    ``facet_tags`` the tessellation face (3D) or edge (2D) id, because Neper
    writes every tessellation entity as an element set and the facet physical
    ids run independently of the cell ones.

    Coordinates must already be in SI units (metres). No scaling is applied.
    """
    from mpi4py import MPI

    from dolfinx.io import gmsh as gmshio

    comm = MPI.COMM_WORLD if comm is None else comm
    result = gmshio.read_from_msh(str(base) + ".msh4", comm, rank, gdim=gdim)
    if hasattr(result, "mesh"):
        mesh, cell_tags, facet_tags = result.mesh, result.cell_tags, result.facet_tags
    else:
        mesh, cell_tags, facet_tags = result[0], result[1], result[2]
    if facet_tags is None or facet_tags.values.size == 0:
        raise RuntimeError(
            "no facet tags were read: the facet element sets did not survive "
            "the msh4 round trip (for a raster mesh, check that -dim all reached "
            "neper -M)"
        )
    return mesh, cell_tags, facet_tags


def read_msh4(path):
    """Nodes and tagged 1D / 2D elements of a Gmsh 4.1 ascii mesh.

    Returns ``(xyz, seg, tri)`` where ``xyz`` maps node tag -> coordinates,
    ``seg`` is a list of (edge id, [n1, n2]) and ``tri`` a list of
    (face id, [n1, n2, n3]); the ids are the entity tags Neper wrote, which for
    its own meshes equal the tessellation entity ids (edge#, face#).
    """
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = lines.index("$Nodes")
    nblocks = int(lines[i + 1].split()[0])
    p = i + 2
    xyz = {}
    for _ in range(nblocks):
        _dim, _tag, _par, n = map(int, lines[p].split())
        p += 1
        tags = [int(lines[p + k]) for k in range(n)]
        p += n
        for k in range(n):
            xyz[tags[k]] = np.array(lines[p + k].split(), dtype=float)
        p += n
    i = lines.index("$Elements")
    nblocks = int(lines[i + 1].split()[0])
    p = i + 2
    seg, tri = [], []
    for _ in range(nblocks):
        dim, tag, _typ, n = map(int, lines[p].split())
        p += 1
        for k in range(n):
            nodes = [int(v) for v in lines[p + k].split()[1:]]
            if dim == 1:
                seg.append((tag, nodes))
            elif dim == 2:
                tri.append((tag, nodes))
        p += n
    return xyz, seg, tri
