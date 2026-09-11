"""Li et al. (2022), Front. Mater. 9:935129, Fig. 4, as a codim-1 problem,
using the repo's resolved model.

The boundary is a zero-thickness manifold *given* a thickness ``delta``:
tangential conductance ``delta D_gb`` along it (the network's own equation)
and a transverse resistance ``delta / D_gb`` across it. The second is what the
resolved model's per-grain subdomains are for: the lattice field may jump
across a boundary, and each grain exchanges with the network at ``k`` per
unit area on its own side. With no segregation (Li et al. have none) the
two-sided exchange gives a steady transverse flux ``(k/2)(c1 - c2)``, and a
slab gives ``D_gb (c1 - c2)/delta``, so

    k = 2 D_gb / delta

is the whole of the new physics. Everything else is the repo:
``VoronoiMicrostructure`` / ``VoronoiMicrostructure3D`` for the geometry and
mesh, ``Physics`` for the coefficients, ``resolved.build`` / ``MicroModel.run``
for the problem, ``resolved.averages`` for their Eq. 21 and
``submesh_measure`` for the boundary area.

WHAT THE THIN-BOUNDARY MODEL IS, AND WHERE IT STOPS
---------------------------------------------------
A manifold occupies no volume, so the lattice fills the whole box and the
model lacks the ``(1 - f_GB)`` factor that a band of volume fraction
``f_GB = delta S_v`` removes from the lattice. In parallel that is

    codim:       D_eff/D_m = 1 + f_GB r            (exact)
    volumetric:  D_eff/D_m = 1 + f_GB (r - 1)      (their Hart line, Eq. 28)

so the codim result is the volumetric one to leading order in ``f_GB``, and
the neglected term is ``f_GB D_m``: relative error ``f_GB / (1 + f_GB r)``,
which is ~1 % at r = 100 even at f_GB = 0.3, but is the *entire* effect when
r < 1. The thin-boundary model is an expansion in ``delta S_v``; it breaks
down when the boundary width stops being negligible against the grain size
-- for a 1 nm boundary and a foam, f_GB = 0.1 at 30 nm grains, 0.3 at 10 nm.
Their sweep, f_GB = 0.3-0.7, is grains of 4-10 nm: a boundary phase with
grains in it. Real 10-1000 nm grains sit at f_GB = 0.003-0.3, where the
neglected terms are smaller than the error of calling the boundary a slab in
the first place, and where the codim model is the right tool: one mesh per
microstructure for every ``delta``, a boundary concentration of its own, and
room for segregation and trapping.

So two sweeps run on the same meshes: their axis, drawn on their four panels
with their references so the O(f_GB) departure is visible, and the thin range.

STRUCTURES
----------
  iso:    ``VoronoiMicrostructure3D``, a periodic Poisson-Voronoi foam meshed
          conformingly by gmsh in their 96 nm box, one cell tag per grain.
  col_x:  ``VoronoiMicrostructure``, the 2D cross-section of prismatic
          columns; flux across the columns.
  col_z:  flux along the columns. In the codim model this is analytic,
          ``1 + f_GB r``: with a uniform axial gradient every grain and the
          network carry the same linear profile, the exchange terms vanish and
          the currents add. It is drawn, not simulated.

D_m, T, the imposed concentrations and the references (Hart, Eq. 28; HS,
Eq. 33) are as in li2022_fig4_voronoi.py. Each target ``f_GB`` sets
``delta = f_GB / S_v`` and ``k = 2 D_gb / delta``; the volumetric file's
warning about locator tolerances applies (coordinates are ~1e-9 m).

OUTPUTS
-------
li2022-fig4-codim.png       their range, their four panels, with Hart and HS
li2022-fig4-codim-thin.png  f_GB from 0.003 to 0.7 on a log axis
li2022-fig4-codim.csv       every run
"""

import csv

from mpi4py import MPI

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from festim_microstructure.meshing.voronoi import (
    VoronoiMicrostructure,
    VoronoiMicrostructure3D,
)
from festim_microstructure.models.resolved import Physics, averages, build
from festim_microstructure.postprocessing.measures import submesh_measure

# physics (SI)
# ----------------------------------------------------------------------------
T = 1073.0  # K
D0_M = 5.13e-8  # m^2/s, Liu et al. 2014, cited in their Sect. 2.3
E_M = 0.21  # eV
RATIOS = [100.0, 10.0, 0.2, 0.1]  # D_GB/D_m of their panels D, E, G, H
C_IN = 0.4e24  # m^-3
C_OUT = 0.1e24

# microstructures
# ----------------------------------------------------------------------------
B = 96e-9  # m, their 3D box
SEED = 0
ISO_SEEDS = 12  # the mesher cuts more pieces than this; raise once it runs
ISO_CELLS_PER_GRAIN = 8
COL_SEEDS = 25  # ~5 per edge, as in their Fig. 4B
COL_CELLS_PER_GRAIN = 12
BULK_COARSENING = 4.0

F_GB_THEIRS = [0.3, 0.4, 0.5, 0.6, 0.7]
F_GB_THIN = [0.003, 0.01, 0.03, 0.1]

# (structure, flux axis); col_z is analytic and not in this list
SIMULATED = [("col_x", 0), ("iso", 2)]
LABEL = {"col_z": "Col_I(Z)", "col_x": "Col_I(X)", "iso": "Iso"}
K_B = 8.617333e-5  # eV/K


def D_m(T):
    return D0_M * np.exp(-E_M / (K_B * T))


def physics_for(ratio, f_gb, S_v):
    """The coefficients of one run: D_gb/D_m = ratio at every T, delta from the
    target volume fraction, k from the slab."""
    delta = f_gb / S_v
    D_gb = ratio * D_m(T)
    return Physics(
        T=T,
        D_0_bulk=D0_M,
        E_D_bulk=E_M,
        D_0_gb=ratio * D0_M,
        E_D_gb=E_M,
        delta=delta,
        k_exchange=2.0 * D_gb / delta,
        crystal_anisotropy=1.0,
    )


def boundary_conditions(axis, length, tol):
    """Their 0.4 / 0.1 on the two faces normal to the flux; the other faces are
    no-flux. ``build`` applies each to every grain touching the face and to the
    network's mouths on it."""
    return [
        ("inlet", lambda x: np.abs(x[axis]) < tol, C_IN),
        ("outlet", lambda x: np.abs(x[axis] - length) < tol, C_OUT),
    ]


def make(structure, comm):
    if structure == "iso":
        return VoronoiMicrostructure3D.create(
            size=B,
            n_seeds=ISO_SEEDS,
            seed=SEED,
            cells_per_grain=ISO_CELLS_PER_GRAIN,
            bulk_coarsening=BULK_COARSENING,
            comm=comm,
        )
    return VoronoiMicrostructure.create(
        size=B,
        n_seeds=COL_SEEDS,
        seed=SEED,
        cells_per_grain=COL_CELLS_PER_GRAIN,
        bulk_coarsening=BULK_COARSENING,
        comm=comm,
    )


def boundary_area_per_volume(micro, bcs):
    """S_v from the network submesh, which exists once a problem is initialised."""
    mm = build(micro, physics_for(1.0, 0.01, 1.0 / B), bcs)
    mm.model.initialise()
    dim = micro.mesh.geometry.dim
    return submesh_measure(mm.network) / B**dim


def run(micro, axis, bcs, S_v, ratio, f_gb):
    physics = physics_for(ratio, f_gb, S_v)
    mm = build(micro, physics, bcs).run()
    q, _, _ = averages(mm)  # their Eq. 21: volume-averaged flux
    D_eff = q[axis] * B / (C_IN - C_OUT)
    return dict(
        f_gb=f_gb,
        delta_nm=physics.delta * 1e9,
        S_v_per_nm=S_v * 1e-9,
        k_m_per_s=physics.k_exchange,
        equilibration_length_nm=physics.equilibration_length * 1e9,
        n_grains=len(mm.grains),
        ratio=ratio,
        D_m=physics.D_bulk,
        D_eff=D_eff,
        D_eff_over_D_m=D_eff / physics.D_bulk,
    )


# ----------------------------------------------------------------------------
# references and figures
# ----------------------------------------------------------------------------
def hart(f, r):
    """Their Eq. 28 (volumetric, parallel)."""
    return 1.0 + f * (r - 1.0)


def codim_parallel(f, r):
    """The codim model with the network parallel to the flux: exact."""
    return 1.0 + f * r


def hashin_shtrikman(f, r):
    """Their Eq. 33."""
    return r + (1.0 - f) / (1.0 / (1.0 - r) + f / (3.0 * r))


def draw(rows, fname, f_lo, f_hi, logx):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    f = np.geomspace(f_lo, f_hi, 200) if logx else np.linspace(f_lo, f_hi, 200)
    style = {
        "col_x": dict(
            marker="*",
            ms=9,
            ls="none",
            color="tab:red",
            mfc="none",
            label="Col_I(X), codim",
        ),
        "iso": dict(
            marker="o",
            ms=6,
            ls="none",
            color="tab:purple",
            mfc="none",
            label="Iso, codim",
        ),
    }
    for axp, r in zip(axes.flat, RATIOS, strict=True):
        axp.plot(
            f, hart(f, r), "-", color="0.6", label="Hart, their Eq. 28 (volumetric)"
        )
        axp.plot(
            f,
            codim_parallel(f, r),
            "--",
            color="tab:red",
            lw=0.9,
            label="Col_I(Z), codim: 1 + f r (exact)",
        )
        axp.plot(
            f,
            hashin_shtrikman(f, r),
            ":",
            color="tab:blue",
            label="HS, their Eq. 33 (volumetric)",
        )
        for key, st in style.items():
            sub = sorted(
                (q for q in rows if q["structure"] == key and q["ratio"] == r),
                key=lambda q: q["f_gb"],
            )
            if sub:
                axp.plot(
                    [q["f_gb"] for q in sub], [q["D_eff_over_D_m"] for q in sub], **st
                )
        axp.text(0.05, 0.9, f"$D_{{GB}}/D_m$ = {r:g}", transform=axp.transAxes)
        if logx:
            axp.set_xscale("log")
            if r > 1:
                axp.set_yscale("log")
        axp.set_xlim(f_lo, f_hi)
        axp.legend(fontsize=6, loc="lower right" if r > 1 else "upper right")
    for axp in axes[1]:
        axp.set_xlabel("boundary volume fraction $f_{GB} = \\delta\\,S_v$")
    for axp in axes[:, 0]:
        axp.set_ylabel(r"$D^{eff}/D_m$")
    fig.suptitle(
        "Li et al. 2022 Fig. 4, codim-1 boundary with thickness δ (k = 2 D_GB/δ)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


# ----------------------------------------------------------------------------
def main():
    mpl.use("Agg")
    comm = MPI.COMM_WORLD
    rows = []
    for structure, axis in SIMULATED:
        micro = make(structure, comm)
        bcs = boundary_conditions(axis, B, 1e-6 * B)
        S_v = boundary_area_per_volume(micro, bcs)
        if comm.rank == 0:
            print(
                f"{LABEL[structure]}: {micro.n_grains} grains, S_v = {S_v * 1e-9:.4f} /nm; "
                f"delta = {0.3 / S_v * 1e9:.2f} nm at f_GB = 0.3, {0.01 / S_v * 1e9:.3f} nm at 0.01"
            )
        for f_gb in [*F_GB_THIN, *F_GB_THEIRS]:
            for ratio in RATIOS:
                r = run(micro, axis, bcs, S_v, ratio, f_gb)
                r["structure"] = structure
                rows.append(r)
                if comm.rank == 0:
                    print(
                        f"  f_GB={f_gb:5.3f} delta={r['delta_nm']:6.2f} nm r={ratio:6g}"
                        f"D_eff/D_m={r['D_eff_over_D_m']:9.4f}  HS={hashin_shtrikman(f_gb, ratio):9.4f}  "
                        f"(equilibration length {r['equilibration_length_nm']:.2f} nm)"
                    )
    if comm.rank != 0:
        return
    with open("li2022-fig4-codim.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    draw(rows, "li2022-fig4-codim.png", 0.25, 0.75, logx=False)
    draw(rows, "li2022-fig4-codim-thin.png", 0.002, 0.8, logx=True)


if __name__ == "__main__":
    main()
