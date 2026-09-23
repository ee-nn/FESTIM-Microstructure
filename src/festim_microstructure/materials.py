"""Diffusivity fields for grains and boundaries, and short-circuit estimates.

Material parameters are FESTIM's business: a grain-boundary material is an
ordinary ``festim.Material(D_0=..., E_D=...)`` and a grain material wraps the
lattice tensor field built here as ``festim.Material(D=field)``. What is specific
to a microstructure -- an orientation-dependent lattice tensor per grain, and
the classical estimates that say what regime a boundary is in -- lives in this
module. A diffusivity that differs from one boundary to the next is handled by
the network subdomain itself, since only it knows which tessellation entity each
submesh cell came from: see ``diffusivity_by_entity`` on
:class:`~festim_microstructure.fem.subdomains.GrainBoundaryNetwork`.
"""

from __future__ import annotations

from typing import cast

import dolfinx
import numpy as np

__all__ = [
    "beta_parameter",
    "crystal_diffusivity_field",
    "crystal_tensor",
    "equilibration_length",
    "fill_crystal_diffusivity_field",
    "hart_bound",
    "interface_resistance_ratio",
]


def crystal_tensor(D_bulk, orientation, crystal_anisotropy=1.0):
    """Return the 2D lattice diffusivity tensor of one grain, in the specimen frame.

    ``crystal_anisotropy`` is the ratio of the two principal diffusivities. It is
    set by the metal's lattice: 1 for a cubic metal, where Neumann's principle
    forbids a second-rank property from depending on grain orientation, and
    possibly larger for HCP or tetragonal metals. ``orientation`` is the angle
    from the crystal frame to the specimen frame, in radians.
    """
    root = np.sqrt(crystal_anisotropy)
    principal = np.diag([D_bulk * root, D_bulk / root])
    c, s = np.cos(orientation), np.sin(orientation)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ principal @ rotation.T


def beta_parameter(delta, D_gb, D_b, t_end):
    """Return Le Claire's type-B parameter; short circuits need ``beta >> 1``.

    Below about one the boundary has no tail of its own to measure and the
    kinetics are type A or C instead. Le Claire (1963), Br. J. Appl. Phys. 14,
    351; Mishin et al. (1997), Def. Diff. Forum 143-147, 1359 survey the ranges.
    """
    return delta * (D_gb / D_b - 1) / (2 * np.sqrt(D_b * t_end))


def hart_bound(f_gb, D_gb, D_b):
    """Return the parallel (Hart) upper bound ``f D_gb + (1 - f) D_b``.

    Exact only when the two phases conduct in parallel along the flux, and an
    upper bound otherwise: Hart (1957), Acta Metall. 5, 597.
    """
    return f_gb * D_gb + (1 - f_gb) * D_b


def equilibration_length(delta, D_gb, k):
    """Return ``sqrt(delta D_gb / 2k)``, the length over which a boundary
    equilibrates with its grains; it should be well below the grain size."""
    return np.sqrt(delta * D_gb / (2 * k))


def interface_resistance_ratio(k, grain_size, D_bulk):
    """Return ``(2/k) / (grain_size / D_bulk)``: crossing one boundary against
    crossing one grain. Values well below one mean transparent boundaries;
    Fisher's local-equilibrium picture is the limit 0."""
    return (2.0 / k) / (grain_size / D_bulk)


def crystal_diffusivity_field(micro, D_bulk, crystal_anisotropy=1.0):
    """Create a parent-mesh DG0 tensor field with one lattice tensor per grain.

    ``micro`` is a :class:`~festim_microstructure.microstructure.MeshedMicrostructure`;
    its ``orientations`` set each grain's tensor through :func:`crystal_tensor`.
    Returns the field and the grain-id-to-tensor mapping. Hand the field to
    every grain through ``festim.Material(D=field)``.
    """
    mesh = micro.mesh
    gdim = mesh.geometry.dim
    V = dolfinx.fem.functionspace(mesh, ("DG", 0, (gdim, gdim)))
    # UFL's inherited __new__ also advertises Cofunction for dual spaces.
    D = cast(dolfinx.fem.Function, dolfinx.fem.Function(V, name="D_lattice"))
    return D, fill_crystal_diffusivity_field(D, micro, D_bulk, crystal_anisotropy)


def fill_crystal_diffusivity_field(D, micro, D_bulk, crystal_anisotropy=1.0):
    """Update an existing lattice field in place and return its grain-to-tensor
    mapping. Coefficient sweeps refill the field this way rather than rebuild
    it, so the compiled forms are reused."""
    V = D.function_space
    gdim = V.mesh.geometry.dim
    tensors = {}
    n = gdim * gdim
    for grain_id in micro.grain_ids:
        tensor = crystal_tensor(
            D_bulk, micro.orientations[grain_id - 1], crystal_anisotropy
        )
        if gdim == 3:
            full = np.eye(3) * D_bulk
            full[:2, :2] = tensor
            tensor = full
        tensors[grain_id] = tensor
        cells = micro.cell_tags.find(grain_id)
        dofs = V.dofmap.list[cells].reshape(-1)
        flat = tensor.reshape(-1)
        for component in range(n):
            D.x.array[n * dofs + component] = flat[component]
    D.x.scatter_forward()
    return tensors
