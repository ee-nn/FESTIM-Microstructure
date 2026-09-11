"""Per-grain lattice fields coupled through one codimension-one GB network.

Each grain exchanges ``k (c_grain - c_gb)`` with the collapsed boundary slab;
the network equation uses ``k / delta``. Sweepable coefficients remain DOLFINx
constants or functions to avoid FFCx recompilation.
"""

from dataclasses import dataclass, field

from mpi4py import MPI

import dolfinx
import festim as F
import numpy as np
import ufl

from ..solvers import ATOL, DIRECT_SOLVER_OPTIONS, tune_direct_solver
from ..subdomains import (
    Grain,
    GrainBoundaryNetwork,
    GrainSurface,
    TaggedGrainBoundaryNetwork,
    facet_midpoints,
)
from .properties import (
    Physics,
    crystal_diffusivity_field,
    fill_crystal_diffusivity_field,
)

__all__ = [
    "ATOL",
    "ConstantDiffusivity",
    "Grain",
    "GrainBoundaryNetwork",
    "GrainSurface",
    "MicroModel",
    "Physics",
    "averages",
    "build",
    "check_network_covers_grain_boundaries",
    "crystal_diffusivity_field",
    "equilibrium_error",
    "fill_crystal_diffusivity_field",
    "grain_areas",
    "inventory",
    "mean_lattice_tensor",
    "parent_field",
    "tune_direct_solver",
]

NETWORK_ID = 1_000_000  # above every grain id
SURFACE_ID_0 = 2_000_000  # the per-grain boundary patches are numbered from here


class ConstantDiffusivity(F.Material):
    """A material retaining one mutable DOLFINx diffusivity constant per mesh."""

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


def check_network_covers_grain_boundaries(micro):
    """Return ``(all boundaries, missing boundaries)`` for a microstructure."""
    mesh, tags = micro.mesh, micro.cell_tags
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    index_map = tags.topology.index_map(tdim)
    values = np.zeros(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    values[tags.indices] = tags.values

    # One vectorized locator call avoids an expensive facet-by-facet search.
    facets = np.arange(mesh.topology.index_map(tdim - 1).size_local, dtype=np.int32)
    offsets = facet_to_cell.offsets
    interior = facets[(offsets[facets + 1] - offsets[facets]) == 2]
    pair = facet_to_cell.array[offsets[interior][:, None] + np.arange(2)]
    boundaries = interior[values[pair[:, 0]] != values[pair[:, 1]]]
    if boundaries.size == 0:
        return 0, 0
    on_network = micro.locator(facet_midpoints(mesh, boundaries).T)
    return int(boundaries.size), int(np.count_nonzero(~on_network))


@dataclass
class MicroModel:
    """A built (not yet solved) resolved problem, and the handles to read it."""

    model: F.HydrogenTransportProblemDiscontinuous
    micro: object
    physics: Physics
    grains: list
    network: GrainBoundaryNetwork
    species: list
    c_gb: F.Species
    tensors: dict
    surfaces: dict = field(default_factory=dict)
    lattice_field: object = None  # the DG0 tensor field holding D_m per grain
    k_constants: dict = field(default_factory=dict)  # grain id -> exchange rate
    delta_constant: object = None
    gb_material: ConstantDiffusivity | None = None
    _initialised: bool = False
    _initial_guess: list | None = None  # the unknowns as initialise() left them

    def initialise(self):
        """Build the function spaces, forms and solver. Idempotent."""
        if not self._initialised:
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
        return list(u) if isinstance(u, list | tuple) else [u]

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
        if not self.k_constants:
            raise RuntimeError(
                "this MicroModel has no coefficient handles, so it cannot be "
                "retuned; it was not produced by build()"
            )
        if exchange_rate is None:

            def exchange_rate(grain_id):
                return physics.k_exchange

        self.physics = physics
        self.delta_constant.value = dolfinx.default_scalar_type(physics.delta)
        for grain in self.grains:
            self.k_constants[grain.id].value = dolfinx.default_scalar_type(
                float(exchange_rate(grain.id))
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


def build(
    micro,
    physics: Physics,
    bcs,
    transient=False,
    final_time=None,
    stepsize=None,
    exchange_rate=None,
    initial_conditions=(),
    exports=(),
    atol=ATOL,
    rtol=1e-10,
    petsc_options=None,
):
    """Assemble a resolved model for a microstructure and grain-surface BCs.

    ``bcs`` contains ``(name, locator, value)`` tuples. ``exchange_rate`` maps a
    grain id to ``k`` and defaults to ``physics.k_exchange``.
    """
    if petsc_options is None:
        petsc_options = dict(DIRECT_SOLVER_OPTIONS)

    if exchange_rate is None:

        def exchange_rate(grain_id):
            return physics.k_exchange

    # Material fields and one lattice species per grain.
    D_field, tensors = crystal_diffusivity_field(micro, physics)
    grains = [
        Grain(id=int(g), material=F.Material(D=D_field), cell_tags=micro.cell_tags)
        for g in micro.grain_ids
    ]
    tdim = micro.mesh.topology.dim
    gb_material = ConstantDiffusivity(physics.D_gb)
    if getattr(micro, "facet_tags", None) is not None:
        # Tagged facets avoid geometric network detection.
        network = TaggedGrainBoundaryNetwork(
            NETWORK_ID, gb_material, micro.facet_tags, [micro.gb_tag], dim=tdim - 1
        )
    else:
        network = GrainBoundaryNetwork(
            id=NETWORK_ID, material=gb_material, locator=micro.locator, dim=tdim - 1
        )
    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])

    # Parent-mesh Constants make coefficient sweeps reuse compiled forms.
    scalar = dolfinx.default_scalar_type
    delta_constant = dolfinx.fem.Constant(micro.mesh, scalar(physics.delta))
    k_constants = {
        grain.id: dolfinx.fem.Constant(
            micro.mesh, scalar(float(exchange_rate(grain.id)))
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
    boundary_conditions = [
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
        settings=F.Settings(
            atol=atol,
            rtol=rtol,
            transient=transient,
            final_time=final_time,
            stepsize=stepsize,
        ),
        exports=list(exports),
        petsc_options=petsc_options,
    )
    model.show_progress_bar = transient
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
