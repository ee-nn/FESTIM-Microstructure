"""Fisher short-circuit diffusion with one bulk and one GB-network field.

The collapsed GB slab receives ``2 k (c_b - c_gb) / delta`` while the bulk
loses the matching flux. The resolved model instead has one bulk field per grain.
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
    """Build and reuse a Fisher model for a mesh and GB network.

    The charged-surface locator is also applied to network mouths.
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
        # Network mouths are codimension two in the parent mesh.
        self.mouths = F.SurfaceSubdomain(
            id=self.MOUTHS_ID, dim=tdim - 2, locator=charged_surface
        )
        self.c_b = F.Species("c_b", subdomains=[self.grains])
        self.c_gb = F.Species("c_gb", subdomains=[self.network])
        self.model = None

    def build(self, d_gb=None, exports=(), gb_material=None):
        """Assemble the problem; override the GB diffusivity or material if needed."""
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
                # The slab gains the two-sided exchange as a volumetric rate.
                F.ParticleSource(
                    value=lambda cb, cg: (2.0 / delta) * k * (cb - cg),
                    species=c_gb,
                    volume=network,
                    species_dependent_value={"cb": c_b, "cg": c_gb},
                )
            ],
            boundary_conditions=[
                # FESTIM expresses influx, hence ``k * (c_gb - c_b)``.
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
    """Return Le Claire's type-B parameter; short circuits need ``beta >> 1``."""
    return delta * (D_gb / D_b - 1) / (2 * np.sqrt(D_b * t_end))


def hart_bound(f_gb, D_gb, D_b):
    """Return the parallel (Hart) upper bound ``f D_gb + (1 - f) D_b``."""
    return f_gb * D_gb + (1 - f_gb) * D_b
