"""Integrals and topology checks for solved microstructure models."""

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

__all__ = [
    "component_count",
    "inventory",
    "junction_only_below",
    "submesh_measure",
]


def inventory(c_b, c_gb, delta):
    """Total hydrogen: the grains plus the boundary slabs (``delta`` times the
    network's length in 2D or area in 3D)."""
    dx_bulk = ufl.Measure("dx", domain=c_b.function_space.mesh)
    dx_gb = ufl.Measure("dx", domain=c_gb.function_space.mesh)
    total = dolfinx.fem.assemble_scalar(dolfinx.fem.form(c_b * dx_bulk))
    total += delta * dolfinx.fem.assemble_scalar(dolfinx.fem.form(c_gb * dx_gb))
    return c_b.function_space.mesh.comm.allreduce(total, op=MPI.SUM)


def submesh_measure(subdomain):
    """Total length (2D network) or area (3D network) of the submesh."""
    sub = subdomain.submesh
    geom = sub.geometry
    # Newer DOLFINx exposes one dofmap per coordinate element.
    dofmap = geom.dofmaps[0] if hasattr(geom, "dofmaps") else geom.dofmap
    tdim = sub.topology.dim
    n_local = sub.topology.index_map(tdim).size_local
    x = geom.x
    if tdim == 1:
        seg = dofmap.reshape(-1, 2)[:n_local]
        local = float(np.sum(np.linalg.norm(x[seg[:, 1]] - x[seg[:, 0]], axis=1)))
    elif tdim == 2:
        tri = dofmap.reshape(-1, 3)[:n_local]
        local = float(
            np.sum(
                0.5
                * np.linalg.norm(
                    np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]]),
                    axis=1,
                )
            )
        )
    else:
        raise ValueError(f"a network submesh has tdim 1 or 2, not {tdim}")
    return sub.comm.allreduce(local, op=MPI.SUM)


def component_count(subdomain):
    """Return the network component count, or ``None`` in parallel.

    Cells join through vertices in 2D and edges in 3D.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    sub = subdomain.submesh
    if sub.comm.size > 1:
        return None
    tdim = sub.topology.dim
    jdim = tdim - 1  # entities the cells are joined through
    sub.topology.create_connectivity(tdim, jdim)
    sub.topology.create_connectivity(jdim, tdim)
    j2c = sub.topology.connectivity(jdim, tdim)
    n = sub.topology.index_map(tdim).size_local
    rows, cols = [], []
    for j in range(sub.topology.index_map(jdim).size_local):
        cells = j2c.links(j)
        for a in cells:
            for b in cells:
                rows.append(a)
                cols.append(b)
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return connected_components(adj, directed=False)[0]


def junction_only_below(entities, axis, top, tol=1e-9):
    """Return the depth below which transport requires a network junction."""
    touching = [e for e in entities if np.asarray(e)[:, axis].max() > top - tol]
    return min((np.asarray(e)[:, axis].min() for e in touching), default=top)
