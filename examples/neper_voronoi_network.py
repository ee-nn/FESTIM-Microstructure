"""Short-circuit diffusion through the grain-boundary network of a Neper
polycrystal.

Same formulation as the in-situ Voronoi version: the whole network is **one**
codim-1 subdomain carrying **one** species. What changes is only where the
microstructure comes from -- see :mod:`festim_microstructure.meshing.neper` for
what Neper adds (``domface``, ``theta``, exact junction topology) and for how
the binaries are found (``FM_NEPER_BIN`` / ``FM_GMSH_BIN``, or ``PATH``).

Run::

    python examples/neper_voronoi_network.py
    mpirun -n 4 python examples/neper_voronoi_network.py
"""

from dataclasses import dataclass, field
from pathlib import Path

import festim as F
import numpy as np

from festim_microstructure.meshing.neper import (
    NeperMicrostructure,
    NeperOptions,
    read_mesh,
    run_neper,
)
from festim_microstructure.models.fisher import (
    ShortCircuitParams,
    ShortCircuitProblem,
    beta_parameter,
    hart_bound,
)
from festim_microstructure.models.properties import gb_diffusivity_field
from festim_microstructure.postprocessing.measures import (
    component_count,
    inventory,
    submesh_measure,
)
from festim_microstructure.subdomains import TaggedGrainBoundaryNetwork


@dataclass
class Setup:
    L: float = 1.0  # specimen size (Neper's default domain is cube(1,1,1))
    n_cells: int = 100  # number of grains; Neper is happy into the 1e5 range
    seed: int = 1  # -id, the rng seed, so the microstructure is reproducible

    D_B: float = 1e-3
    D_GB: float = 30.0
    delta: float = 1e-3
    k_exchange: float = 1.0
    c0: float = 1.0
    t_end: float = 3.0
    dt: float = 0.05

    theta_min: float = 0.0  # keep only boundaries above this disorientation (deg)
    theta_dependent_D: bool = False  # see gb_diffusivity_field, CHECK before enabling
    neper: NeperOptions = field(default_factory=NeperOptions)
    stem: str = "poly"
    workdir: Path = Path(__file__).resolve().parent / "results"
    force: bool = False


def main(s=Setup()):
    L, D_B, D_GB = s.L, s.D_B, s.D_GB

    base = run_neper(
        s.n_cells, s.seed, s.stem, s.workdir, options=s.neper, force=s.force
    )
    micro = NeperMicrostructure(base, theta_min=s.theta_min, options=s.neper)
    mesh, cell_tags, facet_tags = read_mesh(base, gdim=3)

    network = TaggedGrainBoundaryNetwork(
        id=ShortCircuitProblem.NETWORK_ID,
        material=F.Material(D_0=D_GB, E_D=0.0),
        facet_tags=facet_tags,
        entity_ids=micro.network_face_ids,
        dim=2,
    )
    params = ShortCircuitParams(
        D_b=D_B,
        D_gb=D_GB,
        delta=s.delta,
        k_exchange=s.k_exchange,
        c0=s.c0,
        t_end=s.t_end,
        dt=s.dt,
        atol=1e-14,
        rtol=1e-12,
    )
    problem = ShortCircuitProblem(
        mesh, network, charged_surface=lambda x: np.isclose(x[2], L), params=params
    )

    exports = problem.vtx_exports(str(base.parent / "neper"))
    if s.theta_dependent_D:
        # the submesh has to exist before a field can live on it, so build and
        # initialise once, then swap the material in and initialise again
        problem.build(D_GB, exports).initialise()
        network.material = F.Material(
            D_0=gb_diffusivity_field(network, micro.theta, D_B, D_GB), E_D=0.0
        )
    model, cb_fast, cgb_fast = problem.solve(D_GB, exports)

    # what we built
    print(micro.report(n_cells=s.n_cells))
    area_mesh, area_tess = submesh_measure(network), micro.network_area
    n_comp = component_count(network)
    n_cells = mesh.topology.index_map(3).size_global
    print(f"  mesh                            : {n_cells} cells")
    print(
        f"  network captured by the submesh : {area_mesh:.4f} of {area_tess:.4f}"
        f" ({100 * area_mesh / area_tess:.2f} %)"
    )
    if n_comp is not None:
        print(f"  connected components            : {n_comp}")
    print(f"  interior facets                 : {model.manifold_is_interior(network)}")

    # effect of the network
    fast = inventory(cb_fast, cgb_fast, s.delta)
    depth = micro.junction_only_below(L)
    gb_z = cgb_fast.function_space.tabulate_dof_coordinates()[:, 2]
    deep = gb_z < depth
    c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

    _, cb_ref, cgb_ref = problem.solve(D_B)
    ref = inventory(cb_ref, cgb_ref, s.delta)

    print(
        f"\nafter t = {s.t_end} (lattice diffusion alone reaches "
        f"~{2 * np.sqrt(D_B * s.t_end):.3g})"
    )
    print(f"  inventory with fast boundaries : {fast:.4e}")
    print(f"  inventory with D_gb = D_b      : {ref:.4e}")
    print(f"  enhancement                    : x {fast / ref:.1f}")
    f_gb = s.delta * area_tess / L**3
    print(f"  boundary volume fraction f     : {f_gb:.3e}")
    print(
        f"  Hart bound f D_gb + (1-f) D_b  : {hart_bound(f_gb, D_GB, D_B):.3e}"
        f"  (vs D_b = {D_B:.3e})"
    )
    beta = beta_parameter(s.delta, D_GB, D_B, s.t_end)
    print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")

    bulk_z = cb_fast.function_space.tabulate_dof_coordinates()[:, 2]
    c_grain_deep = cb_fast.x.array[bulk_z < depth].mean()
    print("\njunction transport: no boundary touching the charged face reaches below")
    print(f"z = {depth:.3f}, so everything the network holds there has")
    print("crossed at least one triple line.")
    print(f"  max c on the network there     : {c_deep:.4e}")
    print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
    print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")


if __name__ == "__main__":
    main()
