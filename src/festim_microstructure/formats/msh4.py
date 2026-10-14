"""Read tagged Gmsh meshes."""

from __future__ import annotations

import meshio

__all__ = ["read_mesh", "read_msh4"]


def read_mesh(base, gdim, comm=None, rank=0):
    """Read a tagged Gmsh ``.msh4`` into DOLFINx.

    ``cell_tags`` carry grain IDs and ``facet_tags`` carry boundary IDs.
    The physical IDs of cells and facets are independent. UPXO exports use
    the FESTIM tagging adapter.

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
            "the msh4 round trip. Ensure the mesh exporter writes physical "
            "groups for boundary curves (2D) or surfaces (3D)."
        )
    return mesh, cell_tags, facet_tags


def read_msh4(path):
    """Nodes and physically tagged lines / triangles of a Gmsh mesh.

    Returns ``(xyz, seg, tri)`` where ``xyz`` maps node tag -> coordinates,
    ``seg`` is a list of (edge id, [n1, n2]) and ``tri`` a list of
    (grain id, [n1, n2, n3]). Node indices are local to this read. Physical
    groups supply boundary and grain IDs, as they do in the DOLFINx reader;
    geometrical entity tags do not determine ownership.
    """
    mesh = meshio.read(path, file_format="gmsh")
    physical = mesh.cell_data.get("gmsh:physical", [])
    seg, tri = [], []
    for i, block in enumerate(mesh.cells):
        if block.dim in (1, 2) and block.type not in ("line", "triangle"):
            raise ValueError(f"{path}: mesh diagnostics require linear lines/triangles")
        if block.type not in ("line", "triangle"):
            continue
        if i >= len(physical) or (physical[i] <= 0).any():
            raise ValueError(f"{path}: every {block.type} needs a positive physical ID")
        target = seg if block.type == "line" else tri
        target.extend(
            (int(tag), nodes.tolist())
            for tag, nodes in zip(physical[i], block.data, strict=True)
        )
    return dict(enumerate(mesh.points)), seg, tri
