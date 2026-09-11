"""The resolved model: one subdomain per grain, coupled through a codim-1 network.

Every Voronoi cell is its own :class:`festim.VolumeSubdomain` carrying its own
:class:`festim.Species`, so the lattice concentration is allowed to *jump* from
one grain to the next. That is what makes a grain boundary able to resist as well
as to conduct: the only path from one grain to another is through the network,
and the exchange rate ``k`` of a grain with the network is a real transfer
resistance in series with the lattice. A single bulk subdomain cannot represent
that -- there the lattice field is continuous across every boundary, so a
boundary can only ever be a short circuit in parallel with the lattice.

The network itself stays **one** codim-1 subdomain with one species. That is what
makes triple junctions work with no junction condition to write: the submesh
built from all the grain-boundary facets is connected, so a single continuous
field lives on it. FESTIM gives each adjacent grain an interior-facet integral of
its own on which that grain is the ``"+"`` side, so a facet shared by two grains
is integrated once per grain -- which is exactly the two-sided feeding the
physics asks for.

Units, and where ``delta`` goes
-------------------------------
The physical boundary is a slab of width ``delta`` with tangential diffusivity
``D_gb``, fed from the grain on each side. Per unit boundary *area*::

    delta dc_gb/dt = delta D_gb lap_s(c_gb) + sum_(sides s) k_s (c_s - c_gb)

FESTIM assembles the manifold equation without any thickness factor, so it is
divided by ``delta``: the manifold material carries ``D_gb`` itself and each side
contributes a source ``(k_s / delta) (c_s - c_gb)`` -- one term per grain, no
factor of two, because the sum over the two adjacent grains supplies it. Each
grain loses what it gives, ``k_s (c_gb - c_s)`` per unit area, on its own side.
``delta`` reappears in every post-processed quantity: the network holds
``delta * integral(c_gb)`` and carries ``delta * D_gb * grad_s(c_gb)``.
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
from .properties import Physics, crystal_diffusivity_field

__all__ = [
    "ATOL",
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
    "grain_areas",
    "inventory",
    "mean_lattice_tensor",
    "parent_field",
    "tune_direct_solver",
]

NETWORK_ID = 1_000_000  # above every grain id
SURFACE_ID_0 = 2_000_000  # the per-grain boundary patches are numbered from here


def check_network_covers_grain_boundaries(micro):
    """Every interior facet separating two grains must be in the network.

    If one were missed it would carry no coupling at all, and since the grains are
    now separate subdomains the model would have a perfectly sealed wall where the
    microstructure has a grain boundary.
    """
    mesh, tags = micro.mesh, micro.cell_tags
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    index_map = tags.topology.index_map(tdim)
    values = np.zeros(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    values[tags.indices] = tags.values

    # Vectorised, and with one locator call rather than one per facet: a locator
    # such as near_segments loops over every ridge internally, so calling it per
    # facet costs n_facets x n_segments and dominates any mesh worth checking.
    # The facets are selected the same way GrainBoundaryNetwork selects its
    # candidates, and both share facet_midpoints.
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

    def run(self):
        self.model.initialise()
        tune_direct_solver(self.model)
        self.model.run()
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
    """Indicator of an axis-aligned box, as a UFL expression on ``mesh``.

    ``window`` is ``((x0, y0), (x1, y1))`` in 2D or ``((x0, y0, z0), (x1, y1, z1))``
    in 3D; ``None`` is the whole cell.
    """
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
    """Volume averages of flux and gradient over ``window`` (the whole cell if None).

    Returns ``(q, grad_c, area)``. The flux sums every grain's lattice flux and
    the network's own ``delta D_gb grad_s c_gb``. Note that ``grad_c`` is
    the average over the grains of a field that is *discontinuous* between them
    (the jumps are carried by the network term).
    """
    delta, D_gb = mm.physics.delta, mm.physics.D_gb
    dim = mm.micro.mesh.geometry.dim
    area = 0.0
    q = np.zeros(dim)
    grad_c = np.zeros(dim)

    # NOTE I wonder if there is a way to use FESTIM's exports here instead of needing to
    # assemble everything from scratch
    for c, grain in zip(mm.grain_solutions, mm.grains, strict=True):
        mesh = c.function_space.mesh
        dx = ufl.Measure("dx", domain=mesh)
        w = _window(mesh, window)
        D = ufl.as_matrix(mm.tensors[grain.id].tolist())
        flux = -D * ufl.grad(c)
        area += _assemble(w * dx)
        for i in range(dim):
            q[i] += _assemble(w * flux[i] * dx)
            grad_c[i] += _assemble(w * ufl.grad(c)[i] * dx)

    cgb = mm.network_solution
    mesh_g = cgb.function_space.mesh
    dx_g = ufl.Measure("dx", domain=mesh_g)
    w_g = _window(mesh_g, window)
    for i in range(dim):
        q[i] -= _assemble(w_g * delta * D_gb * ufl.grad(cgb)[i] * dx_g)

    return q / area, grad_c / area, area


def inventory(mm: MicroModel, window=None):
    """Helper function to calculate the hydrogen per unit area
    of the cell, including both grains and grain boundaries."""
    total = 0.0
    for c in mm.grain_solutions:
        mesh = c.function_space.mesh
        total += _assemble(_window(mesh, window) * c * ufl.Measure("dx", domain=mesh))
    cgb = mm.network_solution
    mesh_g = cgb.function_space.mesh

    # multiply by the GB width for dimensional consistency
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
    """Assemble the resolved problem on ``micro``.

    Args:
        micro: a :class:`~festim_microstructure.meshing.voronoi.VoronoiMicrostructure`
            or :class:`~festim_microstructure.meshing.voronoi.VoronoiMicrostructure3D`
            (or anything with the same ``mesh`` / ``cell_tags`` / ``grain_ids`` /
            ``orientations`` / ``locator`` / ``tolerance`` attributes; with
            ``facet_tags`` and ``gb_tag`` the network is read from the tags).
        physics: the material data.
        bcs: an iterable of ``(name, locator, value)``. Each fixes the
            concentration on the part of the outer boundary picked by ``locator``
            including every adjacent grain and point where a boundary
            meets that part of the surface (a codim-2 subdomain). ``value``
            is a float or a callable of ``x``.
        petsc_options: solver configuration. Defaults to FESTIM's direct solve
            with a larger MUMPS working array. See ``tune_direct_solver`` for details.
        exchange_rate: ``grain_id -> k``, the transfer coefficient of that grain
            with the network. Defaults to ``physics.k_exchange`` everywhere. The
            exchange is a property of the (network, grain) pair, so it can be made
            to depend on the grain it faces.
        transient: if False, a steady solve.

    Returns:
        MicroModel
    """
    if petsc_options is None:
        petsc_options = dict(DIRECT_SOLVER_OPTIONS)

    if exchange_rate is None:

        def exchange_rate(grain_id):
            return physics.k_exchange

    # Assemble the material properties
    D_field, tensors = crystal_diffusivity_field(micro, physics)
    grains = [
        Grain(id=int(g), material=F.Material(D=D_field), cell_tags=micro.cell_tags)
        for g in micro.grain_ids
    ]
    tdim = micro.mesh.topology.dim
    gb_material = F.Material(D_0=physics.D_gb, E_D=0.0)
    if getattr(micro, "facet_tags", None) is not None:
        # the mesher tagged the boundary facets: nothing geometric to get wrong
        network = TaggedGrainBoundaryNetwork(
            NETWORK_ID, gb_material, micro.facet_tags, [micro.gb_tag], dim=tdim - 1
        )
    else:
        network = GrainBoundaryNetwork(
            id=NETWORK_ID, material=gb_material, locator=micro.locator, dim=tdim - 1
        )
    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])

    delta = physics.delta
    # one exchange per grain, each naming only that grain's species. This is how
    # FESTIM works out which side of the network the term belongs to.
    # The two grains adjacent to a facet each contribute their own term.
    sources = [
        F.ParticleSource(
            value=lambda c_g, c_b, k=exchange_rate(grain.id): (k / delta) * (c_b - c_g),
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
            value=lambda c_g, c_b, k=exchange_rate(grain.id): k * (c_g - c_b),
            species_dependent_value={"c_b": spe, "c_g": c_gb},
        )
        for spe, grain in zip(species, grains, strict=True)
    ]

    # Assign boundary conditions
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
        # where the network meets that part of the surface: codim-2 of the parent,
        # points in 2D and curves in 3D
        mouths = F.SurfaceSubdomain(id=next_id, dim=tdim - 2, locator=locator)
        next_id += 1
        subdomains.append(mouths)
        boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=mouths, value=value, species=c_gb)
        )
        surfaces[name] = (patches, mouths)

    # Build the codim problem
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
        model, micro, physics, grains, network, species, c_gb, tensors, surfaces
    )


def grain_areas(micro):
    """Area of every grain, by assembling the DG0 test function against ``dx``.

    That assembly gives one entry per cell holding the cell's own area, so the
    grains are summed from it without a form per grain.
    """
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
    """Every grain's solution gathered into one DG1 field on the parent mesh.

    Each grain lives on its own submesh, so a plain export writes one file per
    grain. Interpolating them all into a single discontinuous field instead gives
    one ParaView dataset in which the jump across each boundary is visible.
    """
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
    """max ``|c_grain - c_gb|`` / max ``c_grain``. If ~0: the GBs are transparent
    and a single-field homogeneous model can exist. If ~O(1): grains are throttled
    off from the network and no single D_eff can reproduce the microstructure.
    """
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
