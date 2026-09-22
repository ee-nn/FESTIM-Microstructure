"""Material parameters and per-grain/per-boundary diffusivity fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import dolfinx
import festim as F
import numpy as np

__all__ = [
    "Physics",
    "beta_parameter",
    "crystal_diffusivity_field",
    "fill_crystal_diffusivity_field",
    "gb_diffusivity_field",
    "hart_bound",
]


@dataclass
class Physics:
    """Transport parameters; defaults are tungsten-like and GB-dominated."""

    T: float = 500.0
    D_0_bulk: float = 1.9e-7  # m2/s
    E_D_bulk: float = 0.39  # eV
    D_0_gb: float = 2.85e-7  # m2/s, 1.5x the lattice prefactor: 2D vs 3D random walk
    E_D_gb: float = 0.12  # eV
    delta: float = 1e-9  # m, boundary width
    k_exchange: float = 3.0  # m/s, grain <-> boundary transfer coefficient
    crystal_anisotropy: float = 1.0  # -, geometrically set by the metal's lattice
    # Cubic metals use 1; HCP/tetragonal metals may need a larger value.

    @property
    def D_bulk(self):
        """The (isotropic) lattice diffusivity."""
        return self.D_0_bulk * np.exp(-self.E_D_bulk / (F.k_B * self.T))

    @property
    def D_gb(self):
        """The (isotropic) grain boundary diffusivity."""
        return self.D_0_gb * np.exp(-self.E_D_gb / (F.k_B * self.T))

    @property
    def contrast(self):
        """Ratio of the grain boundary to bulk (isotropic) diffusivities."""
        return self.D_gb / self.D_bulk

    def crystal_tensor(self, orientation):
        """Return the 2D lattice diffusivity tensor for a grain orientation."""
        root = np.sqrt(self.crystal_anisotropy)
        principal = np.diag([self.D_bulk * root, self.D_bulk / root])
        # Rotate from the crystal frame to the specimen frame.
        c, s = np.cos(orientation), np.sin(orientation)
        rotation = np.array([[c, -s], [s, c]])
        return rotation @ principal @ rotation.T

    @property
    def equilibration_length(self):
        """Return the GB equilibration length; it should be below grain size."""
        return np.sqrt(self.delta * self.D_gb / (2 * self.k_exchange))

    def interface_resistance_ratio(self, grain_size):
        """Return ``(2/k)/(grain_size/D_bulk)``; values below one are transparent."""
        return (2.0 / self.k_exchange) / (grain_size / self.D_bulk)

    def report(self, grain_size=None):
        lines = [
            f"physics at T = {self.T:g} K",
            f"  D_bulk (orientation average)   : {self.D_bulk:.3e} m2/s",
            f"  D_gb                           : {self.D_gb:.3e} m2/s",
            f"  D_gb / D_bulk                  : {self.contrast:.4g}",
            f"  crystal anisotropy             : {self.crystal_anisotropy:g}",
            f"  boundary width, delta          : {1e9 * self.delta:g} nm",
            f"  exchange rate, k               : {self.k_exchange:.3e} m/s",
            f"  equilibration length           : "
            f"{1e9 * self.equilibration_length:.3g} nm",
        ]
        if grain_size is not None:
            lines.append(
                f"  interface / lattice resistance : "
                f"{self.interface_resistance_ratio(grain_size):.3e}"
            )
        return "\n".join(lines)


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


def crystal_diffusivity_field(micro, physics):
    """Create a parent-mesh DG0 tensor field with one lattice tensor per grain."""
    mesh = micro.mesh
    gdim = mesh.geometry.dim
    V = dolfinx.fem.functionspace(mesh, ("DG", 0, (gdim, gdim)))
    # UFL's inherited __new__ also advertises Cofunction for dual spaces.
    D = cast(dolfinx.fem.Function, dolfinx.fem.Function(V, name="D_lattice"))
    return D, fill_crystal_diffusivity_field(D, micro, physics)


def fill_crystal_diffusivity_field(D, micro, physics):
    """Update an existing lattice field and return its grain-to-tensor mapping."""
    V = D.function_space
    gdim = V.mesh.geometry.dim
    tensors = {}
    n = gdim * gdim
    for grain_id in micro.grain_ids:
        tensor = physics.crystal_tensor(micro.orientations[grain_id - 1])
        if gdim == 3:
            full = np.eye(3) * physics.D_bulk
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


def gb_diffusivity_field(network, theta, d_low, d_high, theta_c=15.0):
    """Create a DG0 GB field selected by disorientation threshold.

    The tagged network must be initialized so its parent-cell map exists.
    """
    submesh = network.submesh
    parent = None
    for name in ("submesh_to_mesh", "submesh_to_parent", "parent_to_submesh"):
        parent = getattr(network, name, None)
        if parent is not None:
            break
    ids_of_facets = getattr(network, "entity_ids_of_facets", None)
    if parent is None or ids_of_facets is None:
        raise NotImplementedError(
            "cannot map submesh cells back to tessellation entities: the "
            "subdomain does not expose its parent entity map under any of the "
            "expected names, or was located by a predicate rather than by "
            "facet tags, which leaves it with no ids to map back. Read it off "
            "dolfinx.mesh.create_submesh directly."
        )
    # Parent-cell positions index the aligned located-facet tag array.
    theta = np.asarray(theta)
    V = dolfinx.fem.functionspace(submesh, ("DG", 0))
    d = cast(dolfinx.fem.Function, dolfinx.fem.Function(V, name="D_gb"))
    tdim = submesh.topology.dim
    n_local = submesh.topology.index_map(tdim).size_local
    ids = ids_of_facets[np.asarray(parent)[:n_local]]
    d.x.array[:n_local] = np.where(theta[ids - 1] >= theta_c, d_high, d_low)
    d.x.scatter_forward()
    return d
