"""Averages, inventories, and field diagnostics for resolved transport models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

if TYPE_CHECKING:
    from festim_microstructure.models.resolved import MicroModel

__all__ = [
    "averages",
    "equilibrium_error",
    "grain_areas",
    "inventory",
    "mean_lattice_tensor",
    "parent_field",
]


def _window(mesh, window):
    """Return a UFL indicator for an axis-aligned window, or one for ``None``."""
    if window is None:
        return 1.0
    lo, hi = window
    x = ufl.SpatialCoordinate(mesh)
    inside = ufl.And(x[0] > lo[0], x[0] < hi[0])
    for i in range(1, len(lo)):
        inside = ufl.And(inside, ufl.And(x[i] > lo[i], x[i] < hi[i]))
    return ufl.conditional(inside, 1.0, 0.0)


def _assemble(expr):
    """Small helper for assembling weak formulation expressions."""
    form = dolfinx.fem.form(expr)
    local = dolfinx.fem.assemble_scalar(form)
    return form.mesh.comm.allreduce(local, op=MPI.SUM)


def averages(mm: MicroModel, window=None):
    """Return ``(flux, gradient, measure)`` over a window or the whole cell.

    Flux includes the GB term ``delta * D_gb * grad(c_gb)``.
    """
    delta, D_gb = mm.physics.delta, mm.physics.D_gb
    dim = mm.micro.mesh.geometry.dim
    scalar = dolfinx.default_scalar_type
    area = 0.0
    q = np.zeros(dim)
    grad_c = np.zeros(dim)

    for c, grain in zip(mm.grain_solutions, mm.grains, strict=True):
        mesh = c.function_space.mesh
        dx = ufl.Measure("dx", domain=mesh)
        w = _window(mesh, window)
        # A Constant avoids a separate compiled form per grain tensor.
        D = dolfinx.fem.Constant(mesh, np.asarray(mm.tensors[grain.id], dtype=scalar))
        flux = -D * ufl.grad(c)
        area += _assemble(w * dx)
        for i in range(dim):
            q[i] += _assemble(w * flux[i] * dx)
            grad_c[i] += _assemble(w * ufl.grad(c)[i] * dx)

    cgb = mm.network_solution
    mesh_g = cgb.function_space.mesh
    dx_g = ufl.Measure("dx", domain=mesh_g)
    w_g = _window(mesh_g, window)
    delta_D_gb = dolfinx.fem.Constant(mesh_g, scalar(delta * D_gb))
    for i in range(dim):
        q[i] -= _assemble(w_g * delta_D_gb * ufl.grad(cgb)[i] * dx_g)

    return q / area, grad_c / area, area


def inventory(mm: MicroModel, window=None):
    """Return inventory in grains plus the width-scaled GB network."""
    total = 0.0
    for c in mm.grain_solutions:
        mesh = c.function_space.mesh
        total += _assemble(_window(mesh, window) * c * ufl.Measure("dx", domain=mesh))
    cgb = mm.network_solution
    mesh_g = cgb.function_space.mesh

    # Convert the codimension-one integral to slab inventory.
    total += mm.physics.delta * _assemble(
        _window(mesh_g, window) * cgb * ufl.Measure("dx", domain=mesh_g)
    )
    return total


def grain_areas(micro):
    """Return grain areas from a single DG0 assembly."""
    mesh = micro.mesh
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    volumes = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(ufl.TestFunction(V) * ufl.Measure("dx", domain=mesh))
    ).array
    return {
        int(g): float(volumes[V.dofmap.list[micro.cell_tags.find(g)].reshape(-1)].sum())
        for g in micro.grain_ids
    }


def mean_lattice_tensor(mm: MicroModel):
    """Area-weighted average of the grains' lattice tensors."""
    areas = grain_areas(mm.micro)
    total = sum(areas.values())
    return sum(areas[g.id] * mm.tensors[g.id] for g in mm.grains) / total


def parent_field(mm: MicroModel, name="c"):
    """Gather all grain solutions into one discontinuous parent-mesh field."""
    V = dolfinx.fem.functionspace(mm.model.mesh.mesh, ("DG", 1))
    field = dolfinx.fem.Function(V, name=name)

    def update():
        for spe, grain in zip(mm.species, mm.grains, strict=True):
            parent_cells = mm.model.volume_meshtags.find(grain.id)
            sub_cells = grain.cell_map.sub_topology_to_topology(
                parent_cells, inverse=True
            )
            field.interpolate(
                spe.subdomain_to_post_processing_solution[grain],
                cells0=sub_cells,
                cells1=parent_cells,
            )

    return field, update


def equilibrium_error(mm: MicroModel):
    """Return the normalized grain/GB concentration mismatch."""
    import scipy.spatial

    cgb = mm.network_solution
    x_gb = cgb.function_space.tabulate_dof_coordinates()
    if x_gb.shape[0] == 0:
        return 0.0
    tree = scipy.spatial.cKDTree(x_gb)
    tol = mm.micro.tolerance
    worst, scale = 0.0, 1e-300
    for c in mm.grain_solutions:
        x = c.function_space.tabulate_dof_coordinates()
        scale = max(scale, float(np.abs(c.x.array).max()))
        distance, idx = tree.query(x)
        on_network = distance < tol
        if on_network.any():
            worst = max(
                worst,
                float(
                    np.abs(c.x.array[on_network] - cgb.x.array[idx[on_network]]).max()
                ),
            )
    return worst / scale
