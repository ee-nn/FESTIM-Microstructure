"""Material data and the coefficient fields the models are built from.

:class:`Physics` holds the transport parameters of a polycrystal (lattice and
boundary Arrhenius laws, boundary width, exchange rate). The two field builders
put per-grain and per-boundary coefficients on the mesh:

* :func:`crystal_diffusivity_field` -- one lattice tensor per grain, as a DG0
  tensor field on the parent mesh;
* :func:`gb_diffusivity_field` -- one diffusivity per boundary, as a DG0 scalar
  field on the network submesh, chosen by the boundary's disorientation.
"""

from dataclasses import dataclass

import dolfinx
import festim as F
import numpy as np

__all__ = ["Physics", "crystal_diffusivity_field", "gb_diffusivity_field"]


@dataclass
class Physics:
    """Material data. Defaults are tungsten-like, in the short-circuit regime.

    ``E_D_bulk`` is Frauenfelder's lattice migration energy and ``E_D_gb`` a
    typical DFT boundary value; at 500 K that is a diffusivity contrast of ~800,
    which with a boundary area fraction of a few 1e-3 is what puts the material
    in the regime where the network decides the transport.
    """

    T: float = 500.0
    D_0_bulk: float = 1.9e-7  # m2/s
    E_D_bulk: float = 0.39  # eV
    D_0_gb: float = 2.85e-7  # m2/s, 1.5x the lattice prefactor: 2D vs 3D random walk
    E_D_gb: float = 0.12  # eV
    delta: float = 1e-9  # m, boundary width
    k_exchange: float = 3.0  # m/s, grain <-> boundary transfer coefficient
    crystal_anisotropy: float = 1.0  # -, geometrically set by the metal's lattice
    # this is usually 1.0 for most metals (cubic symmetry). Set to >1 for HCP/tetragonal

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
        """The lattice diffusivity of a grain whose fast axis is at ``orientation``.

        The two principal values are ``D_bulk * sqrt(a)`` and ``D_bulk / sqrt(a)``,
        so their geometric mean is ``D_bulk`` whatever the anisotropy ``a``, and
        an untextured aggregate of them still averages to the lattice value.
        """
        # NOTE this function should probably be called something more specific,
        # such as lattice_diffusivity_tensor
        root = np.sqrt(self.crystal_anisotropy)
        principal = np.diag([self.D_bulk * root, self.D_bulk / root])
        # coordinate transformation from the principal frame to the home frame
        c, s = np.cos(orientation), np.sin(orientation)
        rotation = np.array([[c, -s], [s, c]])
        return rotation @ principal @ rotation.T

    @property
    def equilibration_length(self):
        """Characteristic along a boundary over which it equilibrates with the grains.

        Much smaller than grain size implies local equilibrium, ``c_gb = c_grain``,
        which is the regime in which a single-field homogeneous model is valid.
        """
        return np.sqrt(self.delta * self.D_gb / (2 * self.k_exchange))

    def interface_resistance_ratio(self, grain_size):
        """Ratio of transfer resistance of one boundary crossing to the lattice
        resistance of one grain, ``(2/k) / (d/D_bulk)``.

        < 1 means GBs are transparent and the polycrystal behaves as if the
        lattice field were continuous.
        """
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


def crystal_diffusivity_field(micro, physics):
    """A DG0 tensor field on the parent mesh holding each grain's lattice tensor.

    ``festim.Material`` takes a ``fem.Function`` for ``D``, and the subdomain
    integrals are parent-mesh integrals indexed by the grain id, so one field
    shared by every grain is enough: each grain's term only ever reads the cells
    of that grain.

    ``micro`` must provide ``mesh``, ``cell_tags``, ``grain_ids`` and
    ``orientations`` (one angle per grain, radians).
    """
    mesh = micro.mesh
    V = dolfinx.fem.functionspace(mesh, ("DG", 0, (2, 2)))
    D = dolfinx.fem.Function(V, name="D_lattice")
    tensors = {}
    for grain_id in micro.grain_ids:
        tensor = physics.crystal_tensor(micro.orientations[grain_id - 1])
        tensors[grain_id] = tensor
        cells = micro.cell_tags.find(grain_id)
        dofs = V.dofmap.list[cells].reshape(-1)
        for component in range(4):
            D.x.array[4 * dofs + component] = tensor.reshape(-1)[component]
    D.x.scatter_forward()
    return D, tensors


def gb_diffusivity_field(network, theta, d_low, d_high, theta_c=15.0):
    """A per-boundary diffusivity as a DG0 field on the network submesh.

    Low-angle boundaries are not fast paths, and the disorientation is a
    property of the tessellation (Neper's face table) or of the specimen (EBSD),
    so the natural refinement of the model is a diffusivity that varies from
    boundary to boundary. It has to enter as a coefficient on the *one* submesh:
    splitting the network into a high-angle and a low-angle subdomain would give
    two disconnected submeshes and put the junction conditions straight back.

    Args:
        network: a :class:`~festim_microstructure.subdomains.TaggedGrainBoundaryNetwork`
            whose submesh exists, i.e. after ``model.initialise()``.
        theta: disorientation of every tessellation face / edge, in degrees, in
            id order (``theta[k]`` belongs to entity ``k + 1``). Both
            :class:`~festim_microstructure.meshing.neper.NeperMicrostructure` and
            :class:`~festim_microstructure.meshing.ebsd.pipeline.EbsdMicrostructure`
            expose it as ``micro.theta``.
        d_low, d_high: diffusivity below / at-or-above ``theta_c``.

    CHECK before relying on this: it depends on the submesh-to-parent map that
    the subdomain exposes, whose attribute name has moved between FESTIM
    versions, and on ``create_submesh`` preserving the order of the located
    facets.
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
            "expected names, or is not a TaggedGrainBoundaryNetwork. Read it off "
            "dolfinx.mesh.create_submesh directly."
        )
    # entity_ids_of_facets is aligned with the facet list returned by
    # locate_subdomain_entities, and create_submesh keeps that order, so the
    # parent map indexes straight into it: parent[c] is the position of submesh
    # cell c in the located facet list, hence its tessellation id.
    theta = np.asarray(theta)
    V = dolfinx.fem.functionspace(submesh, ("DG", 0))
    d = dolfinx.fem.Function(V, name="D_gb")
    tdim = submesh.topology.dim
    n_local = submesh.topology.index_map(tdim).size_local
    ids = ids_of_facets[np.asarray(parent)[:n_local]]
    d.x.array[:n_local] = np.where(theta[ids - 1] >= theta_c, d_high, d_low)
    d.x.scatter_forward()
    return d
