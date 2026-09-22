"""Hydrogen transport through grains and a codimension-one GB network.

Each grain has its own lattice concentration field and exchanges hydrogen with
the collapsed boundary slab at a rate ``k (c_grain - c_gb)``. The network
equation uses ``k / delta`` because the physical boundary width ``delta`` is a
model coefficient rather than a meshed dimension.

An interior facet contributes one exchange term for each adjacent grain. For
each side, the network source ``k (c_grain - c_gb) / delta`` balances the grain
flux ``k (c_gb - c_grain)`` after accounting for the slab width. This preserves
mass while allowing the lattice concentration to jump across a boundary.

Sweepable coefficients remain DOLFINx constants or functions so changing the
physics does not require FFCx to recompile the forms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import dolfinx
import festim as F

from festim_microstructure.fem.solvers import (
    ATOL,
    DIRECT_SOLVER_OPTIONS,
    tune_direct_solver,
)
from festim_microstructure.fem.subdomains import (
    Grain,
    GrainBoundaryNetwork,
    GrainSurface,
)
from festim_microstructure.materials import (
    Physics,
    crystal_diffusivity_field,
    fill_crystal_diffusivity_field,
)
from festim_microstructure.microstructure import MeshedMicrostructure, require

__all__ = ["MicroModel", "SolveOptions", "build"]


class _MutableScalarDiffusivity(F.Material):
    """Retain one mutable DOLFINx diffusivity constant per mesh."""

    def __init__(self, D):
        super().__init__(D_0=float(D), E_D=0.0)
        self._constants = {}

    @property
    def D_value(self):
        return self.D_0

    @D_value.setter
    def D_value(self, value):
        self.D_0 = float(value)
        for constant in self._constants.values():
            constant.value = dolfinx.default_scalar_type(self.D_0)

    def get_diffusion_coefficient(self, mesh=None, temperature=None, species=None):
        # Parent mesh and each submesh need distinct constants.
        key = id(mesh)
        if key not in self._constants:
            self._constants[key] = dolfinx.fem.Constant(
                mesh, dolfinx.default_scalar_type(self.D_0)
            )
        return self._constants[key]


@dataclass
class SolveOptions:
    """Everything :func:`build` hands to the solver rather than to the physics.

    Split out because these six travelled together through a twelve-argument
    signature while saying nothing about the microstructure or the material:
    they describe how the resulting problem is stepped and solved.
    """

    transient: bool = False
    final_time: float | None = None
    stepsize: float | F.Stepsize | None = None
    atol: float = ATOL
    rtol: float = 1e-10
    petsc_options: dict | None = None

    def festim_settings(self):
        return F.Settings(
            atol=self.atol,
            rtol=self.rtol,
            transient=self.transient,
            final_time=self.final_time,
            stepsize=self.stepsize,
        )

    def petsc(self):
        """The PETSc options, defaulting to the direct solver these need."""
        if self.petsc_options is None:
            return dict(DIRECT_SOLVER_OPTIONS)
        return self.petsc_options


NETWORK_ID = 1_000_000  # above every grain id
SURFACE_ID_0 = 2_000_000  # the per-grain boundary patches are numbered from here


@dataclass
class MicroModel:
    """A built grain/network problem and the handles used to solve and inspect it."""

    model: F.HydrogenTransportProblemDiscontinuous
    micro: MeshedMicrostructure
    physics: Physics
    grains: list
    network: GrainBoundaryNetwork
    species: list
    c_gb: F.Species
    tensors: dict
    surfaces: dict = field(default_factory=dict)
    lattice_field: dolfinx.fem.Function | None = None
    k_constants: dict = field(default_factory=dict)  # grain id -> exchange rate
    delta_constant: dolfinx.fem.Constant | None = None
    gb_material: _MutableScalarDiffusivity | None = None
    _initialised: bool = False
    _initial_guess: list | None = None  # the unknowns as initialise() left them

    def initialise(self, force=False):
        """Build the function spaces, forms and solver.

        Idempotent unless ``force``, which rebuilds even if this model has
        already been initialised: the hook for swapping in a coefficient that
        can only be built once a submesh exists (see
        :func:`~festim_microstructure.materials.gb_diffusivity_field`).
        """
        if force or not self._initialised:
            self.model.initialise()
            tune_direct_solver(self.model)
            # Preserve the cold-start state for reproducible sweep points.
            self._initial_guess = [u.x.array.copy() for u in self._unknowns]
            self._initialised = True
        return self

    @property
    def _unknowns(self):
        """The functions the SNES solves for, in the solver's own order."""
        u = self.model.solver.u
        return list(u) if isinstance(u, Sequence) else [u]

    def reset_initial_guess(self):
        """Put the unknowns back to the state :meth:`initialise` left them in."""
        if self._initial_guess is None:
            raise RuntimeError("initialise() the model before solving it")
        for u, guess in zip(self._unknowns, self._initial_guess, strict=True):
            u.x.array[:] = guess
            u.x.scatter_forward()
        return self

    def solve(self, warm_start=False):
        """Solve from current coefficients, cold-starting unless requested.

        Warm starts are unsafe for these unscaled problems because FESTIM's
        relative residual test can reject a converged round-off floor.
        """
        if not warm_start:
            self.reset_initial_guess()
        self.model.run()
        snes = getattr(getattr(self.model, "solver", None), "solver", None)
        if snes is not None:
            reason = snes.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(
                    f"the Newton solve did not converge (SNES reason {reason}). "
                    f"With delta = {self.physics.delta:.3e} m and k = "
                    f"{self.physics.k_exchange:.3e} m/s the exchange block is "
                    f"{self.physics.k_exchange / self.physics.delta:.3e} against a "
                    f"lattice block of order {self.physics.D_bulk:.3e}; see "
                    "tune_direct_solver for the MUMPS fill-in this costs."
                )
        return self

    def run(self):
        return self.initialise().solve()

    def set_physics(self, physics: Physics, exchange_rate=None):
        """Update coefficients without rebuilding forms, spaces, or geometry."""
        if (
            not self.k_constants
            or self.delta_constant is None
            or self.gb_material is None
            or self.lattice_field is None
        ):
            raise RuntimeError(
                "this MicroModel has no coefficient handles, so it cannot be "
                "retuned; it was not produced by build()"
            )
        self.physics = physics
        self.delta_constant.value[...] = dolfinx.default_scalar_type(physics.delta)
        for grain in self.grains:
            self.k_constants[grain.id].value = dolfinx.default_scalar_type(
                float(
                    physics.k_exchange
                    if exchange_rate is None
                    else exchange_rate(grain.id)
                )
            )
        self.gb_material.D_value = physics.D_gb
        # Temperature also changes the lattice tensor.
        self.tensors = fill_crystal_diffusivity_field(
            self.lattice_field, self.micro, physics
        )
        return self

    @property
    def grain_solutions(self):
        return [
            spe.subdomain_to_post_processing_solution[grain]
            for spe, grain in zip(self.species, self.grains, strict=True)
        ]

    @property
    def network_solution(self):
        return self.c_gb.subdomain_to_post_processing_solution[self.network]


def build(
    micro: MeshedMicrostructure,
    physics: Physics,
    bcs,
    *,
    exchange_rate=None,
    initial_conditions=(),
    exports=(),
    solve: SolveOptions | None = None,
):
    """Assemble the transport model for a microstructure and grain-surface BCs.

    ``bcs`` contains ``(name, locator, value)`` tuples. ``exchange_rate`` maps a
    grain id to ``k`` and defaults to ``physics.k_exchange``. Stepping and
    solver settings live in ``solve``; see :class:`SolveOptions`.

    The network is read from ``micro.facet_tags`` when there are any and
    located by ``micro.locator`` otherwise; ``micro.gb_tag`` may be one tag or a
    sequence of them. See
    :class:`~festim_microstructure.fem.subdomains.GrainBoundaryNetwork`.

    ``micro`` must satisfy
    :class:`~festim_microstructure.microstructure.MeshedMicrostructure`. That is
    checked here, so an incompatible microstructure is named at the call site
    rather than raising an ``AttributeError`` part-way through form assembly.
    """
    require(micro, MeshedMicrostructure, context="in build()")
    solve = solve or SolveOptions()

    # Material fields and one lattice species per grain.
    D_field, tensors = crystal_diffusivity_field(micro, physics)
    grains = [
        Grain(id=int(g), material=F.Material(D=D_field), cell_tags=micro.cell_tags)
        for g in micro.grain_ids
    ]
    tdim = micro.mesh.topology.dim
    gb_material = _MutableScalarDiffusivity(physics.D_gb)
    # Tagged facets avoid geometric network detection.
    located = (
        {"facet_tags": micro.facet_tags, "entity_ids": micro.gb_tag}
        if micro.facet_tags is not None
        else {"locator": micro.locator}
    )
    network = GrainBoundaryNetwork(NETWORK_ID, gb_material, tdim - 1, **located)
    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])

    # Parent-mesh Constants make coefficient sweeps reuse compiled forms.
    scalar = dolfinx.default_scalar_type
    delta_constant = dolfinx.fem.Constant(micro.mesh, scalar(physics.delta))
    k_constants = {
        grain.id: dolfinx.fem.Constant(
            micro.mesh,
            scalar(
                float(
                    physics.k_exchange
                    if exchange_rate is None
                    else exchange_rate(grain.id)
                )
            ),
        )
        for grain in grains
    }

    # Each grain contributes its own side of the network exchange.
    sources = [
        F.ParticleSource(
            # FESTIM inspects arguments, so close over ``delta_constant``.
            value=lambda c_g, c_b, k=k_constants[grain.id]: (
                (k / delta_constant) * (c_b - c_g)
            ),
            species=c_gb,
            volume=network,
            species_dependent_value={"c_b": spe, "c_g": c_gb},
        )
        for spe, grain in zip(species, grains, strict=True)
    ]
    boundary_conditions: list[F.ParticleFluxBC | F.FixedConcentrationBC] = [
        F.ParticleFluxBC(
            subdomain=network,
            species=spe,
            value=lambda c_g, c_b, k=k_constants[grain.id]: k * (c_g - c_b),
            species_dependent_value={"c_b": spe, "c_g": c_gb},
        )
        for spe, grain in zip(species, grains, strict=True)
    ]

    # Grain patches and GB mouths for each prescribed outer surface.
    subdomains = [*grains, network]
    surfaces = {}
    next_id = SURFACE_ID_0
    for name, locator, value in bcs:
        patches = []
        for spe, grain in zip(species, grains, strict=True):
            patch = GrainSurface(next_id, grain.id, micro.cell_tags, locator)
            next_id += 1
            if len(patch.locate_boundary_facet_indices(micro.mesh)) == 0:
                continue  # this grain does not touch that part of the surface
            patches.append(patch)
            subdomains.append(patch)
            boundary_conditions.append(
                F.FixedConcentrationBC(subdomain=patch, value=value, species=spe)
            )
        # GB mouths are codimension two in the parent mesh.
        mouths = F.SurfaceSubdomain(id=next_id, dim=tdim - 2, locator=locator)
        next_id += 1
        subdomains.append(mouths)
        boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=mouths, value=value, species=c_gb)
        )
        surfaces[name] = (patches, mouths)

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(micro.mesh),
        species=[*species, c_gb],
        subdomains=subdomains,
        sources=sources,
        boundary_conditions=boundary_conditions,
        initial_conditions=list(initial_conditions),
        temperature=physics.T,
        settings=solve.festim_settings(),
        exports=list(exports),
        petsc_options=solve.petsc(),
    )
    model.show_progress_bar = solve.transient
    return MicroModel(
        model,
        micro,
        physics,
        grains,
        network,
        species,
        c_gb,
        tensors,
        surfaces,
        lattice_field=D_field,
        k_constants=k_constants,
        delta_constant=delta_constant,
        gb_material=gb_material,
    )
