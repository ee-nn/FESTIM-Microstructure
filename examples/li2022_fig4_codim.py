"""Reproduce Li et al. (2022), Fig. 4, with the codimension-one GB model.

The resolved model gives a GB tangential conductance ``delta * D_gb`` and uses
``k = 2 * D_gb / delta`` to match transverse slab resistance. It compares the
thin-boundary result with volumetric Hart and Hashin-Shtrikman references.
"""

from mpi4py import MPI

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from festim_microstructure.meshing.voronoi import (
    VoronoiMicrostructure,
    VoronoiMicrostructure3D,
)
from festim_microstructure.models.resolved import Physics, averages, build
from festim_microstructure.postprocessing.measures import submesh_measure

# Physics (SI).
T = 1073.0  # K
D0_M = 5.13e-8  # m^2/s, Liu et al. 2014, cited in their Sect. 2.3
E_M = 0.21  # eV
RATIOS = [100.0, 10.0, 0.2, 0.1]  # D_GB/D_m of their panels D, E, G, H
C_IN = 0.4e24  # m^-3
C_OUT = 0.1e24

# Microstructures.
B = 96e-9  # m, their 3D box
SEED = 0
ISO_SEEDS = 12  # the mesher cuts more pieces than this; raise once it runs
ISO_CELLS_PER_GRAIN = 8
COL_SEEDS = 25  # ~5 per edge, as in their Fig. 4B
COL_CELLS_PER_GRAIN = 12
BULK_COARSENING = 4.0

F_GB_THEIRS = [0.3, 0.4, 0.5, 0.6, 0.7]
F_GB_THIN = [0.003, 0.01, 0.03, 0.1]

# ``col_z`` is analytic and therefore not simulated.
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


def prepare(micro, bcs):
    """Build once and return the model plus its network area density ``S_v``."""
    mm = build(micro, physics_for(1.0, 0.01, 1.0 / B), bcs).initialise()
    dim = micro.mesh.geometry.dim
    return mm, submesh_measure(mm.network) / B**dim


def run(mm, axis, S_v, ratio, f_gb):
    physics = physics_for(ratio, f_gb, S_v)
    mm.set_physics(physics).solve()
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


# References and figures.
def hart(f, r):
    """Their Eq. 28 (volumetric, parallel)."""
    return 1.0 + f * (r - 1.0)


def codim_parallel(f, r):
    """The codim model with the network parallel to the flux: exact."""
    return 1.0 + f * r


def hashin_shtrikman(f, r):
    """Their Eq. 33."""
    return r + (1.0 - f) / (1.0 / (1.0 - r) + f / (3.0 * r))


GRAIN_COLOR = "#f5e63b"
GB_COLOR = "#2166d1"


def draw_isometric(ax, micro, title):
    """Draw the codimension-one network inside a grain-coloured cube."""
    B_nm = micro.size * 1e9
    t = np.linspace(0.0, B_nm, 2)
    U, V = np.meshgrid(t, t)
    for X, Y, Z in ((B_nm + 0 * U, U, V), (U, 0 * U, V), (U, V, B_nm + 0 * U)):
        ax.plot_surface(X, Y, Z, color=GRAIN_COLOR, alpha=0.55, shade=False)

    if micro.mesh.geometry.dim == 3:
        polygons = [face * 1e9 for face in micro.faces]
    else:
        polygons = [
            np.array(
                [
                    [p[0], p[1], 0.0],
                    [q[0], q[1], 0.0],
                    [q[0], q[1], micro.size],
                    [p[0], p[1], micro.size],
                ]
            )
            * 1e9
            for p, q in micro.segments
        ]
    ax.add_collection3d(
        Poly3DCollection(polygons, facecolor=GB_COLOR, edgecolor=GB_COLOR, alpha=0.72)
    )
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((1, 1, 1))
    ticks = np.linspace(0, B_nm, 4)
    ax.set(
        xticks=ticks,
        yticks=ticks,
        zticks=ticks,
        xlim=(0, B_nm),
        ylim=(0, B_nm),
        zlim=(0, B_nm),
    )
    ax.set_xlabel("x (nm)", labelpad=4)
    ax.set_ylabel("y (nm)", labelpad=4)
    ax.set_zlabel("z (nm)", labelpad=4)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)


def draw_fig4ab(iso, columns, f_gb):
    """Write the codim isometric and columnar microstructure figure."""
    fig = plt.figure(figsize=(9, 4.5))
    for i, (micro, label) in enumerate(((iso, "Iso"), (columns, "Col-I")), start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        draw_isometric(
            ax,
            micro,
            f"{label}: {micro.n_grains} grains, f_GB = {f_gb:.2f} (codim)",
        )
    fig.tight_layout()
    fig.savefig("li2022-fig4ab-codim.png", dpi=300)
    plt.close(fig)


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
    sampled_f = sorted({row["f_gb"] for row in rows if f_lo <= row["f_gb"] <= f_hi})
    for axp, r in zip(axes.flat, RATIOS, strict=True):
        axp.plot(f, hart(f, r), "-", color="0.6", label="Hart bound (volumetric)")
        axp.plot(
            sampled_f,
            codim_parallel(np.asarray(sampled_f), r),
            marker="*",
            ms=8,
            ls="none",
            color="tab:red",
            label="Col_I(Z), codim: 1 + f r (exact)",
        )
        axp.plot(
            f,
            hashin_shtrikman(f, r),
            ":",
            color="tab:blue",
            label="Hashin-Shtrikman (volumetric)",
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
        axp.legend(fontsize=10, loc="lower right" if r > 1 else "upper right")
    for axp in axes[1]:
        axp.set_xlabel("boundary volume fraction $f_{GB} = \\delta\\,S_v$")
    for axp in axes[:, 0]:
        axp.set_ylabel(r"$D^{eff}/D_m$")
    fig.suptitle(
        "Li et al. 2022 Fig. 4, codim-1 boundary with thickness δ (k = 2 D_GB/δ)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=300)


# ----------------------------------------------------------------------------
def main():
    mpl.use("Agg")
    comm = MPI.COMM_WORLD
    rows = []
    micros = {}
    for structure, axis in SIMULATED:
        micro = make(structure, comm)
        micros[structure] = micro
        bcs = boundary_conditions(axis, B, 1e-6 * B)
        mm, S_v = prepare(micro, bcs)
        if comm.rank == 0:
            print(
                f"{LABEL[structure]}: {micro.n_grains} grains, S_v = {S_v * 1e-9:.4f} /nm; "  # noqa: E501
                f"delta = {0.3 / S_v * 1e9:.2f} nm at f_GB = 0.3, {0.01 / S_v * 1e9:.3f} nm at 0.01"  # noqa: E501
            )
        for f_gb in [*F_GB_THIN, *F_GB_THEIRS]:
            for ratio in RATIOS:
                r = run(mm, axis, S_v, ratio, f_gb)
                r["structure"] = structure
                rows.append(r)
                if comm.rank == 0:
                    print(
                        f"  f_GB={f_gb:5.3f} delta={r['delta_nm']:6.2f} nm r={ratio:6g}"
                        f"D_eff/D_m={r['D_eff_over_D_m']:9.4f}  HS={hashin_shtrikman(f_gb, ratio):9.4f}  "  # noqa: E501
                        f"(equilibration length {r['equilibration_length_nm']:.2f} nm)"
                    )
    if comm.rank != 0:
        return
    draw_fig4ab(micros["iso"], micros["col_x"], F_GB_THEIRS[0])
    draw(rows, "li2022-fig4-codim-thin.png", 0.002, 0.8, logx=True)


if __name__ == "__main__":
    main()
