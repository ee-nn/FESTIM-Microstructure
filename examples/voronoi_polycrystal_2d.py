"""Short-circuit diffusion through the grain-boundary network of a Voronoi polycrystal.

The point of the example is that the entire network is declared as **one** codim-1
subdomain with **one** species. That is what makes triple junctions work: the submesh
built from all the grain-boundary facets is topologically connected, so a single
continuous field lives on it and hydrogen crosses from one boundary to another with no
junction condition to write. See :mod:`festim_microstructure.models.fisher`.

Run::

    python examples/voronoi_polycrystal_2d.py
"""

from dataclasses import dataclass

import festim as F
import numpy as np

from festim_microstructure.meshing.voronoi import (
    build_mesh,
    connected_components,
    near_segments,
    triple_junctions,
    voronoi_segments,
)
from festim_microstructure.models.fisher import (
    ShortCircuitParams,
    ShortCircuitProblem,
    beta_parameter,
    hart_bound,
)
from festim_microstructure.postprocessing.measures import (
    component_count,
    inventory,
    junction_only_below,
    submesh_measure,
)
from festim_microstructure.subdomains import GrainBoundaryNetwork


@dataclass
class Setup:
    L: float = 1.0e-4  # specimen size, 100 microns
    n_seeds: int = 12  # number of grains
    seed: int = 3  # rng seed, so the microstructure is reproducible

    E_M_BULK: float = 0.20  # eV, DFT, Diaz-Rodriguez Fig. 3
    E_M_GB: float = 0.12  # eV, DFT, Diaz-Rodriguez Fig. 3
    D0_B: float = 1.9e-7  # m^2/s, assumed
    T: float = 500.0  # K
    delta: float = 1e-9  # grain-boundary width
    k_exchange: float = 3.0  # bulk <-> grain-boundary exchange, m/s
    c0: float = 5.6e27  # surface concentration
    t_end: float = 1.0
    dt: float = 1e-3

    @property
    def D0_GB(self):
        return 1.5 * self.D0_B  # 2D vs 3D random-walk prefactor

    @property
    def D_B(self):
        return self.D0_B * np.exp(-self.E_M_BULK / (F.k_B * self.T))

    @property
    def D_GB(self):
        return self.D0_GB * np.exp(-self.E_M_GB / (F.k_B * self.T))

    @property
    def h_gb(self):
        return 0.004 * self.L  # mesh size at the boundaries

    @property
    def h_bulk(self):
        return 0.04 * self.L  # mesh size in the grain interiors


def main(s=Setup()):
    L, D_B, D_GB = s.L, s.D_B, s.D_GB

    # microstructure
    segments = voronoi_segments(s.n_seeds, L, np.random.default_rng(s.seed))
    mesh, cell_tags, n_grains = build_mesh(segments, L, s.h_gb, s.h_bulk)
    tol = 1e-7  # distance below which a point counts as lying on a ridge
    network = GrainBoundaryNetwork(
        id=ShortCircuitProblem.NETWORK_ID,
        material=F.Material(D_0=D_GB, E_D=0.0),
        locator=lambda x: near_segments(x, segments, tol),
        dim=1,
    )
    params = ShortCircuitParams(
        D_b=D_B,
        D_gb=D_GB,
        delta=s.delta,
        k_exchange=s.k_exchange,
        c0=s.c0,
        T=s.T,
        t_end=s.t_end,
        dt=s.dt,
        atol=1e-8,
        rtol=1e-6,
    )
    problem = ShortCircuitProblem(
        mesh, network, charged_surface=lambda x: np.isclose(x[1], L), params=params
    )

    # build and solve
    model, cb_fast, cgb_fast = problem.solve(
        D_GB, exports=problem.vtx_exports("voronoi")
    )

    # what we built
    ridge_length = sum(float(np.linalg.norm(q - p)) for p, q in segments)
    facet_length = submesh_measure(network)
    print(
        f"microstructure: {s.n_seeds} seeds, {n_grains} grains, "
        f"{len(segments)} boundary segments"
    )
    print(f"  triple junctions inside the box : {len(triple_junctions(segments, L))}")
    print(f"  connected components (ridges)   : {connected_components(segments, L)}")
    print(f"  connected components (submesh)  : {component_count(network)}")
    print(f"  facets dropped on the outer box : {network.n_dropped}")
    n_cells = mesh.topology.index_map(2).size_global
    print(f"  mesh                            : {n_cells} cells")
    print(
        f"  network captured by the submesh : {facet_length:.4e} of {ridge_length:.4e}"
        f" ({100 * facet_length / ridge_length:.2f} %)"
    )
    print(f"  interior facets                 : {model.manifold_is_interior(network)}")

    # effect of the network
    fast = inventory(cb_fast, cgb_fast, s.delta)
    depth = junction_only_below([np.array(seg) for seg in segments], axis=1, top=L)
    gb_y = cgb_fast.function_space.tabulate_dof_coordinates()[:, 1]
    deep = gb_y < depth
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
    f_gb = s.delta * ridge_length / L**2
    print(f"  boundary area fraction f       : {f_gb:.3e}")
    print(
        f"  Hart bound f D_gb + (1-f) D_b  : {hart_bound(f_gb, D_GB, D_B):.3e}"
        f"  (vs D_b = {D_B:.3e})"
    )
    beta = beta_parameter(s.delta, D_GB, D_B, s.t_end)
    print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")

    bulk_y = cb_fast.function_space.tabulate_dof_coordinates()[:, 1]
    c_grain_deep = cb_fast.x.array[bulk_y < depth].mean()
    print(
        "\njunction transport: no boundary touching the charged surface reaches below"
    )
    print(f"y = {depth:.3e}, so everything the network holds there has crossed")
    print("at least one triple junction.")
    print(f"  max c on the network there     : {c_deep:.4e}")
    print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
    if c_grain_deep > 1e-12 * s.c0:
        print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
    else:
        print("  ratio                          : n/a - nothing has reached this depth")


if __name__ == "__main__":
    main()
