"""Fisher-type short-circuit diffusion: a lattice and a grain-boundary network.

A specimen is held at a constant surface concentration, and the grain-boundary
network carries hydrogen far deeper than lattice diffusion alone would, leaking
sideways into the grains as it goes. This is the classical Fisher (1951)
picture, with the network declared as **one** codim-1 subdomain carrying **one**
species, so that hydrogen crosses from one boundary to another at every
junction with no junction condition to write.

Because every grain is the same material, the grains are a *single* volume
subdomain, and the network sits inside it rather than between two subdomains.
FESTIM decides that the coupling is an interior-facet integral from the mesh
topology, not from the number of subdomains the manifold separates. (The
per-grain formulation, where the lattice concentration may jump across a
boundary, is :mod:`festim_microstructure.models.resolved`.)

UNITS -- the one thing to get right
-----------------------------------
The grain boundary is physically a slab of width ``delta`` that has been
collapsed onto a surface. ``c_gb`` is the concentration *inside* that slab
(H/m3, the same unit as the bulk), so the two halves of the exchange are not
the same number:

* the bulk loses a **flux** ``J = k (c_b - c_gb)`` (H/m2/s) through the grain
  boundary plane;
* the slab gains a **volumetric rate** ``2 J / delta`` (H/m3/s).

``1 / delta`` converts a per-area flux into a per-volume rate inside the slab,
and the factor ``2`` is because the full slab collects from both of its faces.
What must stay paired is that both halves are written from the *same* ``J``:
that is what makes the exchange conservative. Fisher's model assumes *local
equilibrium* between the boundary and the lattice in contact with it; FESTIM's
coupling is kinetic, so equilibrium is approached by making ``k`` large
compared with the bulk transport it competes with, ``sqrt(D_b / t_end)``.
"""

from dataclasses import dataclass, field

import festim as F
import numpy as np

__all__ = ["ShortCircuitParams", "ShortCircuitProblem", "beta_parameter", "hart_bound"]


@dataclass
class ShortCircuitParams:
    """Transport parameters and run settings for :class:`ShortCircuitProblem`."""

    D_b: float
    """Lattice diffusivity (m2/s)."""
    D_gb: float
    """Grain-boundary diffusivity (m2/s)."""
    delta: float
    """Grain-boundary width (m)."""
    k_exchange: float
    """Bulk <-> grain-boundary exchange coefficient (m/s); see the module note."""
    c0: float = 1.0
    """Concentration held at the charged surface."""
    T: float = 500.0
    """Temperature; only matters if a material carries a non-zero ``E_D``."""
    t_end: float = 1.0
    dt: float = 1e-3
    atol: float = 1e-14
    rtol: float = 1e-12
    petsc_options: dict | None = None
    extra: dict = field(default_factory=dict)
    """Anything else an example wants to carry around with the run."""


class ShortCircuitProblem:
    """The subdomains and species of a short-circuit problem, built once.

    The same ``grains`` / ``network`` / ``c_b`` / ``c_gb`` objects are reused
    across solves (for instance the fast case and the ``D_gb = D_b`` reference
    case), exactly as the original scripts did.

    Args:
        mesh: the dolfinx mesh.
        network: a :class:`~festim_microstructure.subdomains.GrainBoundaryNetwork`
            or :class:`~festim_microstructure.subdomains.TaggedGrainBoundaryNetwork`
            (its ``material`` is overwritten by :meth:`solve`).
        charged_surface: locator ``(3, n) -> bool`` for the surface held at
            ``c0``. It is applied twice: to the outer boundary of the grains,
            and -- evaluated on the network itself, so as a ``dim = tdim - 2``
            subdomain -- to the points/curves where a boundary meets that
            surface, the network's "mouths". Fixing only the grains would leave
            the mouths free and let the short circuits leak out of the cell.
        params: :class:`ShortCircuitParams`.
        bulk_material: overrides the default ``F.Material(D_0=D_b, E_D=0)``.
    """

    GRAINS_ID, NETWORK_ID, SURFACE_ID, MOUTHS_ID = 1, 2, 3, 4

    def __init__(self, mesh, network, charged_surface, params, bulk_material=None):
        self.mesh = mesh
        self.params = params
        tdim = mesh.topology.dim
        self.grains = F.VolumeSubdomain(
            id=self.GRAINS_ID,
            material=bulk_material or F.Material(D_0=params.D_b, E_D=0.0),
            locator=lambda x: np.full_like(x[0], True, dtype=bool),
        )
        self.network = network
        self.surface = F.SurfaceSubdomain(id=self.SURFACE_ID, locator=charged_surface)
        # every point (2D) or curve (3D) where a grain boundary meets the charged
        # surface, in one object: the locator runs on the network itself, so
        # dim = mesh dimension - 2
        self.mouths = F.SurfaceSubdomain(
            id=self.MOUTHS_ID, dim=tdim - 2, locator=charged_surface
        )
        self.c_b = F.Species("c_b", subdomains=[self.grains])
        self.c_gb = F.Species("c_gb", subdomains=[self.network])
        self.model = None

    def build(self, d_gb=None, exports=(), gb_material=None):
        """Assemble a :class:`festim.HydrogenTransportProblemDiscontinuous`.

        ``d_gb`` defaults to ``params.D_gb``; pass ``params.D_b`` for the
        reference case in which the boundaries are not short circuits at all.
        ``gb_material`` overrides the network material entirely (for a
        per-boundary ``D_gb`` field, see
        :func:`~festim_microstructure.models.properties.gb_diffusivity_field`).
        """
        p = self.params
        d_gb = p.D_gb if d_gb is None else d_gb
        self.network.material = gb_material or F.Material(D_0=d_gb, E_D=0.0)
        c_b, c_gb, network = self.c_b, self.c_gb, self.network
        delta, k = p.delta, p.k_exchange
        self.model = F.HydrogenTransportProblemDiscontinuous(
            mesh=F.Mesh(self.mesh),
            species=[c_b, c_gb],
            subdomains=[self.grains, network, self.surface, self.mouths],
            sources=[
                # ... and the slab gains it, as a volumetric rate (see UNITS)
                F.ParticleSource(
                    value=lambda cb, cg: (2.0 / delta) * k * (cb - cg),
                    species=c_gb,
                    volume=network,
                    species_dependent_value={"cb": c_b, "cg": c_gb},
                )
            ],
            boundary_conditions=[
                # the bulk loses the flux J = k (c_b - c_gb) through the grain
                # boundary. FESTIM's flux convention is the influx, hence the
                # reversed sign
                F.ParticleFluxBC(
                    subdomain=network,
                    species=c_b,
                    value=lambda cb, cg: k * (cg - cb),
                    species_dependent_value={"cb": c_b, "cg": c_gb},
                ),
                F.FixedConcentrationBC(subdomain=self.surface, value=p.c0, species=c_b),
                F.FixedConcentrationBC(subdomain=self.mouths, value=p.c0, species=c_gb),
            ],
            temperature=p.T,
            settings=F.Settings(
                atol=p.atol,
                rtol=p.rtol,
                transient=True,
                final_time=p.t_end,
                stepsize=F.Stepsize(initial_value=p.dt),
            ),
            exports=list(exports),
            **({"petsc_options": p.petsc_options} if p.petsc_options else {}),
        )
        return self.model

    def solve(self, d_gb=None, exports=(), gb_material=None):
        """Build, initialise and run.

        Returns ``(model, c_b_solution, c_gb_solution)``.
        """
        model = self.build(d_gb, exports, gb_material)
        model.initialise()
        model.run()
        return model, self.bulk_solution, self.network_solution

    @property
    def bulk_solution(self):
        return self.c_b.subdomain_to_post_processing_solution[self.grains]

    @property
    def network_solution(self):
        return self.c_gb.subdomain_to_post_processing_solution[self.network]

    def vtx_exports(self, prefix):
        """VTX exports ``<prefix>_grains.bp`` and ``<prefix>_network.bp``."""
        return [
            F.VTXSpeciesExport(
                f"{prefix}_grains.bp", field=self.c_b, subdomain=self.grains
            ),
            F.VTXSpeciesExport(
                f"{prefix}_network.bp", field=self.c_gb, subdomain=self.network
            ),
        ]


def beta_parameter(delta, D_gb, D_b, t_end):
    """Le Claire's type-B parameter ``delta (D_gb/D_b - 1) / (2 sqrt(D_b t))``.

    A short circuit needs ``beta >> 1``.
    """
    return delta * (D_gb / D_b - 1) / (2 * np.sqrt(D_b * t_end))


def hart_bound(f_gb, D_gb, D_b):
    """``f D_gb + (1 - f) D_b``: the upper bound if every boundary ran straight
    along the gradient. A real network is tortuous and only partly connected to
    the source, so the observed enhancement is well below it."""
    return f_gb * D_gb + (1 - f_gb) * D_b
