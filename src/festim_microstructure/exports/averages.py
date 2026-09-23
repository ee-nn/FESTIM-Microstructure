"""Averages, inventories, and field diagnostics for grain/network problems.

Everything here reads plain FESTIM objects: the list of grain subdomains and
the list of species solved on them, aligned, plus the network subdomain and
its species. The solution of a species on a subdomain is
``species.subdomain_to_post_processing_solution[subdomain]``, which FESTIM
fills in ``initialise()`` and refreshes after every solve.

The physical width ``delta`` of the boundary slab is never meshed, so it comes
back here in every quantity the network contributes: the network holds
``delta * integral(c_gb)`` and carries ``delta * D_gb * grad(c_gb)``. Simple
totals of one species on one subdomain are FESTIM's own ``TotalVolume`` and
``AverageVolume``; what is kept here is what they cannot do -- sums over every
grain, the width-scaled network share, a window, and the tensor-valued flux.
"""

from __future__ import annotations

from typing import Any, cast

from mpi4py import MPI

import dolfinx
import dolfinx.io
import numpy as np
import ufl

__all__ = [
    "averages",
    "equilibrium_error",
    "grain_areas",
    "inventory",
    "lattice_mean_below",
    "mean_lattice_tensor",
    "parent_field",
    "solutions",
    "write_vtx",
]


def solutions(species, subdomains):
    """The post-processing solution of each species on its subdomain, aligned."""
    return [
        spe.subdomain_to_post_processing_solution[sub]
        for spe, sub in zip(species, subdomains, strict=True)
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
    form = cast(dolfinx.fem.Form, dolfinx.fem.form(expr))
    local = dolfinx.fem.assemble_scalar(form)
    return form.mesh.comm.allreduce(local, op=MPI.SUM)


def averages(grains, species, tensors, network, c_gb, delta, D_gb, window=None):
    """Return ``(flux, gradient, measure)`` over a window or the whole cell.

    ``tensors`` maps each grain id to its lattice tensor, as
    :func:`~festim_microstructure.materials.crystal_diffusivity_field` returns
    it. The flux includes the network's own ``delta * D_gb * grad(c_gb)``;
    dropping it is the commonest way to get a homogenisation wrong, since it is
    precisely the short circuit being measured.
    """
    scalar = dolfinx.default_scalar_type
    dim = grains[0].parent_mesh.geometry.dim
    area = 0.0
    q = np.zeros(dim)
    grad_c = np.zeros(dim)

    for c, grain in zip(solutions(species, grains), grains, strict=True):
        mesh = c.function_space.mesh
        dx = ufl.Measure("dx", domain=mesh)
        w = _window(mesh, window)
        # A Constant avoids a separate compiled form per grain tensor.
        D = dolfinx.fem.Constant(mesh, np.asarray(tensors[grain.id], dtype=scalar))
        # UFL installs expression operators dynamically; their return types
        # cannot currently be inferred from its Python implementation.
        gradient = cast(Any, ufl.grad(c))
        flux = -D * gradient
        area += _assemble(w * dx)
        for i in range(dim):
            q[i] += _assemble(w * flux[i] * dx)
            grad_c[i] += _assemble(w * gradient[i] * dx)

    cgb = c_gb.subdomain_to_post_processing_solution[network]
    mesh_g = cgb.function_space.mesh
    dx_g = ufl.Measure("dx", domain=mesh_g)
    w_g = _window(mesh_g, window)
    delta_D_gb = dolfinx.fem.Constant(mesh_g, scalar(delta * D_gb))
    gradient_gb = cast(Any, ufl.grad(cgb))
    for i in range(dim):
        q[i] -= _assemble(w_g * delta_D_gb * gradient_gb[i] * dx_g)

    return q / area, grad_c / area, area


def inventory(grains, species, network, c_gb, delta, window=None):
    """Return the inventory in the grains plus the width-scaled network."""
    total = 0.0
    for c in solutions(species, grains):
        mesh = c.function_space.mesh
        total += _assemble(_window(mesh, window) * c * ufl.Measure("dx", domain=mesh))
    cgb = c_gb.subdomain_to_post_processing_solution[network]
    mesh_g = cgb.function_space.mesh
    # Convert the codimension-one integral to slab inventory.
    total += delta * _assemble(
        _window(mesh_g, window) * cgb * ufl.Measure("dx", domain=mesh_g)
    )
    return total


def lattice_mean_below(grains, species, axis, depth):
    """Mean lattice concentration below ``depth`` along ``axis``.

    Taken over the dofs of every grain, which is how a single-field model would
    have read its one bulk array. A dof on a boundary belongs to each grain that
    touches it and is therefore counted once per grain. Rank-local, like the
    arrays it reads: a diagnostic, not an integral (use :func:`inventory`).
    """
    below = [
        c.x.array[c.function_space.tabulate_dof_coordinates()[:, axis] < depth]
        for c in solutions(species, grains)
    ]
    values = np.concatenate(below) if below else np.zeros(0)
    return float(values.mean()) if values.size else 0.0


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


def mean_lattice_tensor(micro, tensors):
    """Area-weighted average of the grains' lattice tensors."""
    areas = grain_areas(micro)
    total = sum(areas.values())
    return sum(areas[g] * tensors[g] for g in areas) / total


def parent_field(grains, species, name="c"):
    """Gather all grain solutions into one discontinuous parent-mesh field.

    Returns the DG1 field and an ``update()`` that refills it from the current
    solutions. FESTIM's own ``SpeciesExport`` writes one dataset per species
    instead; use this when a polycrystal should be one dataset whatever its
    grain count, with the jumps across the boundaries visible.
    """
    mesh = grains[0].parent_mesh
    V = dolfinx.fem.functionspace(mesh, ("DG", 1))
    field = cast(dolfinx.fem.Function, dolfinx.fem.Function(V, name=name))

    def update():
        for c, grain in zip(solutions(species, grains), grains, strict=True):
            parent_cells = grain.cell_tags.find(grain.id)
            sub_cells = grain.cell_map.sub_topology_to_topology(
                parent_cells, inverse=True
            )
            field.interpolate(c, cells0=sub_cells, cells1=parent_cells)

    return field, update


def write_vtx(grains, species, network, c_gb, prefix, time=0.0):
    """Write ``<prefix>_grains.bp`` and ``<prefix>_network.bp``.

    The grains go out as one discontinuous parent-mesh field (see
    :func:`parent_field`) and the network as its own dataset. This is a snapshot
    of the current state, so a transient run writes its last step; for a time
    series, hand FESTIM one ``festim.SpeciesExport(..., format="vtx")`` per
    subdomain through the problem's ``exports``.
    """
    field, update = parent_field(grains, species, name="c_lattice")
    update()
    comm = field.function_space.mesh.comm
    with dolfinx.io.VTXWriter(comm, f"{prefix}_grains.bp", [field], "BP5") as writer:
        writer.write(time)
    cgb = c_gb.subdomain_to_post_processing_solution[network]
    with dolfinx.io.VTXWriter(comm, f"{prefix}_network.bp", [cgb], "BP5") as writer:
        writer.write(time)


def equilibrium_error(grains, species, network, c_gb, tolerance):
    """Return the normalized grain/GB concentration mismatch.

    ``tolerance`` is the distance below which a grain dof counts as lying on
    the network: ``micro.tolerance`` for a microstructure of this package.
    """
    from scipy.spatial import KDTree

    cgb = c_gb.subdomain_to_post_processing_solution[network]
    x_gb = cgb.function_space.tabulate_dof_coordinates()
    if x_gb.shape[0] == 0:
        return 0.0
    tree = KDTree(x_gb, leafsize=16)
    worst, scale = 0.0, 1e-300
    for c in solutions(species, grains):
        x = c.function_space.tabulate_dof_coordinates()
        scale = max(scale, float(np.abs(c.x.array).max()))
        distance, idx = tree.query(x)
        on_network = distance < tolerance
        if on_network.any():
            worst = max(
                worst,
                float(
                    np.abs(c.x.array[on_network] - cgb.x.array[idx[on_network]]).max()
                ),
            )
    return worst / scale
