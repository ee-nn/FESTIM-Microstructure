"""Fisher grain-boundary diffusion, as a codimension-1 subdomain.

A short-circuit diffusion problem: a specimen is held at a constant surface
concentration, and a grain boundary perpendicular to that surface carries hydrogen far
deeper than lattice diffusion alone would, leaking sideways into the grain as it goes.
This is the classical Fisher (1951) model; Whipple and Le Claire solved it analytically,
and the script checks the simulation against that solution at the end.

Geometry -- the problem is symmetric about the grain boundary plane, so only half of it
is modelled. ``x`` runs across the grain, ``y`` runs into the depth::

      y=0   ┌───────────────┐   surface, c = C0
            │Γ              │
            │Γ    grain     │   Γ = the grain boundary, a codim-1 subdomain at x=0
            │Γ    (D_B)     │       carrying its own equation with D_GB >> D_B
            │Γ              │
      y=LY  └───────────────┘
           x=0            x=LX  <- mid-grain symmetry plane, zero flux

Requires the codim-1 machinery *and* Dirichlet conditions on the boundary of a manifold
(the ``dim`` argument of ``SurfaceSubdomain``), which is what pins the grain boundary to
the surface concentration at its mouth. The exchange units are explained in
:mod:`festim_microstructure.models.fisher`; in this symmetric half-cell the half-slab
of width ``delta / 2`` collects from one face only, ``J / (delta / 2)``, which is the
same coefficient ``2 J / delta`` the full-slab convention gives.

Run::

    python examples/fisher_grain_boundary.py
"""

from dataclasses import dataclass

from mpi4py import MPI

import dolfinx
import festim as F
import numpy as np

from festim_microstructure.models.fisher import ShortCircuitParams, ShortCircuitProblem


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


def main(s=Setup()):
    D_B, D_GB, LX, LY = s.D_B, s.D_GB, s.LX, s.LY

    # Fisher's model assumes *local equilibrium* between the grain boundary and the
    # lattice in contact with it, rather than a finite exchange rate. FESTIM's
    # coupling is kinetic, so equilibrium is approached by making k large compared
    # with the bulk transport it competes with, sqrt(D_B / T_END). The script
    # reports how well that holds.
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([LX, LY])], [s.NX, s.NY]
    )
    # the grain boundary: a line inside a 2D mesh, with its own transport equation
    gb = F.VolumeSubdomain(
        id=ShortCircuitProblem.NETWORK_ID,
        material=F.Material(D_0=D_GB, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.0, atol=1e-11),
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
        mesh,
        gb,
        charged_surface=lambda x: np.isclose(x[1], 0.0, atol=1e-11),
        params=params,
    )
    _, cb_fn, cg_fn = problem.solve(D_GB, exports=problem.vtx_exports("fisher"))

    # analysis
    order = np.argsort(cg_fn.function_space.tabulate_dof_coordinates()[:, 1])
    y_gb = cg_fn.function_space.tabulate_dof_coordinates()[order, 1]
    c_gb_vals = cg_fn.x.array[order]

    bulk_only = 2 * np.sqrt(D_B * s.t_end)
    print(f"lattice diffusion alone would reach ~{bulk_only:.3g}")
    print("grain boundary profile:")
    for y in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"   y = {y:.2f}   c_gb = {np.interp(y, y_gb, c_gb_vals):.4e}")

    # how close the kinetic exchange gets to Fisher's local-equilibrium assumption
    probe = np.linspace(0.05 / 1e4, 1.0 / 1e4, 20)
    pts = np.column_stack([np.zeros_like(probe), probe, np.zeros_like(probe)])
    ratio = eval_on(cb_fn, pts) / np.interp(probe, y_gb, c_gb_vals)
    print(
        f"\nlocal equilibrium c_b(0,y)/c_gb(y): {ratio.min():.4f} .. {ratio.max():.4f}"
    )

    # Whipple / Le Claire: in type-B kinetics the section-averaged concentration obeys
    #     ln(cbar) linear in y**(6/5), and
    #     delta*D_gb = 1.322 sqrt(D_B/t) (-slope)**(-5/3)
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
    beta = s.delta * (D_GB / D_B - 1) / (2 * np.sqrt(D_B * s.t_end))
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
