"""Integrals and topology checks on a solved short-circuit problem.

Everything here takes dolfinx / FESTIM objects and returns numbers; the
per-grain (resolved) model has its own versions in
:mod:`festim_microstructure.models.resolved` because they need the model's
tensors and window.
"""

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
    # dolfinx >= 0.9 exposes one dofmap per coordinate element; the scalar
    # attribute is deprecated there and absent in later releases
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
    """Connected components of the network submesh.

    In 2D the network is a graph of segments meeting at points, so connectivity
    runs through vertices; in 3D it is triangles meeting along edges. More than
    one component is expected once a disorientation threshold starts removing
    boundaries, and it is worth knowing: a component that does not touch the
    charged surface is never fed, and a fragmented network is no longer the
    single connected object the codim-1 formulation was chosen for.

    Serial only -- in parallel the adjacency is partitioned and this would
    count per-rank pieces, so ``None`` is returned.
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
    """Deepest coordinate reached by a boundary that touches the charged surface.

    Below this depth no boundary is fed directly, so whatever the network holds
    there has crossed at least one junction. ``entities`` is a list of point
    arrays (segments as ``(2, 2)``, polygons as ``(n, 3)``); ``axis`` is the
    depth coordinate and ``top`` the charged surface's position on it.
    """
    touching = [e for e in entities if np.asarray(e)[:, axis].max() > top - tol]
    return min((np.asarray(e)[:, axis].min() for e in touching), default=top)
