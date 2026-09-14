"""Compare a codimension-one Fisher GB model with the Whipple/Le Claire result.

One boundary at ``x = 0`` between two grains, with a charged surface at
``y = 0``. The cell is symmetric about the boundary, so the right-hand grain is
exactly the half-cell the Whipple/Le Claire analysis is written for, and the
left-hand grain is what supplies the boundary slab's second face. The slab
therefore receives ``k (c_1 - c_gb) + k (c_2 - c_gb)``, which is the ``2 k`` of
Fisher's equation, without either side of it being assumed.
"""

from dataclasses import dataclass

from mpi4py import MPI

import dolfinx
import festim as F
import numpy as np

import festim_microstructure as fm

LEFT, RIGHT = 1, 2  # grain tags either side of the boundary at x = 0


@dataclass
class Setup:
    E_M_BULK: float = 0.20  # eV, DFT, Diaz-Rodriguez Fig. 3
    E_M_GB: float = 0.12  # eV, DFT, Diaz-Rodriguez Fig. 3
    D0_B: float = 1.9e-7  # m^2/s, assumed
    T: float = 500.0  # K
    delta: float = 1e-9  # grain-boundary width
    k_exchange: float = 1.0  # bulk <-> grain-boundary exchange coefficient
    c0: float = 5.6e27  # surface concentration
    LX: float = 0.1 / 1e4  # half grain width
    LY: float = 1.5 / 1e4  # depth
    NX: int = 100
    NY: int = 150
    t_end: float = 1.0
    dt: float = 1e-3

    @property
    def D_B(self):
        return self.D0_B * np.exp(-self.E_M_BULK / (F.k_B * self.T))

    @property
    def D_GB(self):
        return 1.5 * self.D0_B * np.exp(-self.E_M_GB / (F.k_B * self.T))


def eval_on(fn, points):
    """Evaluate a fenics Function at a list of (x, y, 0) points."""
    fn_mesh = fn.function_space.mesh
    tree = dolfinx.geometry.bb_tree(fn_mesh, fn_mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(fn_mesh, candidates, points)
    cells = [colliding.links(i)[0] for i in range(len(points))]
    return fn.eval(points, cells).reshape(-1)


def two_grains(mesh):
    """Tag the cells left of the boundary ``LEFT`` and those right of it
    ``RIGHT``. Every cell lies wholly on one side: ``x = 0`` is a mesh line."""
    tdim = mesh.topology.dim
    index_map = mesh.topology.index_map(tdim)
    cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    values = np.where(midpoints[:, 0] < 0.0, LEFT, RIGHT).astype(np.int32)
    return dolfinx.mesh.meshtags(mesh, tdim, cells, values)


def main(s=Setup()):
    D_B, D_GB, LX, LY = s.D_B, s.D_GB, s.LX, s.LY

    # The full cell, [-LX, LX] x [0, LY]. Its right half is meshed exactly as
    # the half-cell was, so the profiles below are the same numbers.
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([-LX, 0.0]), np.array([LX, LY])], [2 * s.NX, s.NY]
    )
    micro = fm.TaggedPolycrystal(
        mesh=mesh,
        cell_tags=two_grains(mesh),
        grain_ids=np.array([LEFT, RIGHT]),
        tolerance=0.05 * LX / s.NX,
        # A line subdomain carries the GB transport equation; here it is the
        # single interior mesh line at x = 0.
        network_locator=lambda x: np.isclose(x[0], 0.0, atol=1e-11),
        name="one boundary between two grains",
    )
    # E_D = 0 on both phases: the setup has already done the Arrhenius part.
    physics = fm.Physics(
        T=s.T,
        D_0_bulk=D_B,
        E_D_bulk=0.0,
        D_0_gb=D_GB,
        E_D_gb=0.0,
        delta=s.delta,
        k_exchange=s.k_exchange,
        crystal_anisotropy=1.0,
    )
    # Large ``k`` approaches Fisher's local-equilibrium assumption.
    model = fm.build(
        micro,
        physics,
        bcs=[("charged", lambda x: np.isclose(x[1], 0.0, atol=1e-11), s.c0)],
        solve=fm.SolveOptions(
            transient=True, final_time=s.t_end, stepsize=s.dt, atol=1e-8, rtol=1e-6
        ),
    ).run()
    fm.exports.averages.write_vtx(model, "fisher", time=s.t_end)

    grains = dict(zip([g.id for g in model.grains], model.grain_solutions, strict=True))
    cb_fn, cg_fn = grains[RIGHT], model.network_solution

    # analysis
    order = np.argsort(cg_fn.function_space.tabulate_dof_coordinates()[:, 1])
    y_gb = cg_fn.function_space.tabulate_dof_coordinates()[order, 1]
    c_gb_vals = cg_fn.x.array[order]

    bulk_only = 2 * np.sqrt(D_B * s.t_end)
    print(f"lattice diffusion alone would reach ~{bulk_only:.3g}")
    print("grain boundary profile:")
    for y in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"   y = {y:.2f}   c_gb = {np.interp(y, y_gb, c_gb_vals):.4e}")

    # Check the local-equilibrium approximation.
    probe = np.linspace(0.05 / 1e4, 1.0 / 1e4, 20)
    pts = np.column_stack([np.zeros_like(probe), probe, np.zeros_like(probe)])
    ratio = eval_on(cb_fn, pts) / np.interp(probe, y_gb, c_gb_vals)
    print(
        f"\nlocal equilibrium c_b(0,y)/c_gb(y): {ratio.min():.4f} .. {ratio.max():.4f}"
    )

    # Whipple/Le Claire type-B fit: ln(cbar) vs. y**(6/5). One grain and half
    # the slab, which is the half-cell the closed form is written for.
    depths = np.linspace(0.15, 1.0, 25)
    xs = np.linspace(0.0, LX, 200)
    cbar = np.array(
        [
            (
                np.trapezoid(
                    eval_on(
                        cb_fn,
                        np.column_stack([xs, np.full_like(xs, y), np.zeros_like(xs)]),
                    ),
                    xs,
                )
                + (s.delta / 2) * np.interp(y, y_gb, c_gb_vals)
            )
            / (LX + s.delta / 2)
            for y in depths
        ]
    )

    slope, intercept = np.polyfit(depths**1.2, np.log(cbar), 1)
    residual = np.abs(np.log(cbar) - (slope * depths**1.2 + intercept)).max()
    recovered = 1.322 * np.sqrt(D_B / s.t_end) * (-slope) ** (-5.0 / 3.0)

    alpha = s.delta / (2 * np.sqrt(D_B * s.t_end))
    beta = fm.materials.beta_parameter(s.delta, D_GB, D_B, s.t_end)
    print(
        f"\nLe Claire analysis (valid for alpha << 1, beta >> 1: "
        f"alpha = {alpha:.3f}, beta = {beta:.0f})"
    )
    print(
        f"   d ln(cbar) / d y**(6/5) = {slope:.4f}  (max fit residual {residual:.3f})"
    )
    print(f"   recovered delta*D_gb    = {recovered:.4e}")
    print(f"   input     delta*D_gb    = {s.delta * D_GB:.4e}")
    error = 100 * abs(recovered / (s.delta * D_GB) - 1)
    print(f"   error                   = {error:.1f} %")


if __name__ == "__main__":
    main()
