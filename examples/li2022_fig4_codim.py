"""Reproduce Li et al. (2022), Fig. 4, with the codimension-one GB model.

The model gives a GB tangential conductance ``delta * D_gb`` and uses
``k = 2 * D_gb / delta`` to match transverse slab resistance. It compares the
thin-boundary result with volumetric Hart and Hashin-Shtrikman references.

The sweep over the boundary volume fraction changes only ``delta`` and ``k``,
which are DOLFINx constants of the declared FESTIM problem, so one problem per
(microstructure, ``D_gb/D_m``) is assembled and re-solved from a cold start at
every point; the boundary material itself is an ordinary ``festim.Material``
and needs a fresh problem when ``D_gb`` changes.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import dolfinx
import festim as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem

NETWORK_ID = 1_000_000  # above every grain id; a manifold shares the surface ids
SURFACE_ID_0 = 2_000_000  # the per-grain surface patches are numbered from here

# Physics (SI).
T = 1073.0  # K
D0_M = 5.13e-8  # m^2/s, Liu et al. 2014, cited in their Sect. 2.3
E_M = 0.21  # eV
RATIOS = [10.0, 0.1]  # D_GB/D_m of their panels E, H
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


def boundary_conditions(axis, length, tol):
    """Their 0.4 / 0.1 on the two faces normal to the flux; the other faces are
    no-flux. Each is applied to every grain touching the face and to the
    network's mouths on it."""
    return [
        (lambda x: np.abs(x[axis]) < tol, C_IN),
        (lambda x: np.abs(x[axis] - length) < tol, C_OUT),
    ]


def make(structure, comm):
    if structure == "iso":
        return fm.VoronoiMicrostructure.create(
            size=B,
            n_seeds=ISO_SEEDS,
            dim=3,
            seed=SEED,
            cells_per_grain=ISO_CELLS_PER_GRAIN,
            bulk_coarsening=BULK_COARSENING,
            comm=comm,
        )
    return fm.VoronoiMicrostructure.create(
        size=B,
        n_seeds=COL_SEEDS,
        seed=SEED,
        cells_per_grain=COL_CELLS_PER_GRAIN,
        bulk_coarsening=BULK_COARSENING,
        comm=comm,
    )


@dataclass
class CellProblem:
    """One declared problem and the two constants the sweep changes."""

    model: F.HydrogenTransportProblemDiscontinuous
    grains: list
    species: list
    tensors: dict
    network: fm.GrainBoundaryNetwork
    c_gb: F.Species
    D_gb: float
    k: dolfinx.fem.Constant
    width: dolfinx.fem.Constant

    def solve(self, delta, k):
        """Solve for a new slab width and exchange rate, from a cold start.

        A warm start is unsafe for this unscaled problem: FESTIM's relative
        residual test compares against the first residual, which a warm start
        makes a round-off floor.
        """
        self.width.value = delta
        self.k.value = k
        for subdomain in self.model.volume_subdomains:
            subdomain.u.x.array[:] = 0.0
            subdomain.u.x.scatter_forward()
        self.model.run()
        return self


def cell_problem(micro, D_gb, bcs):
    """The grain/network problem as FESTIM declarations; see the Voronoi
    examples. ``delta`` and ``k`` start at placeholders and are set per point."""
    mesh = micro.mesh
    scalar = dolfinx.default_scalar_type
    D_lattice, tensors = fm.materials.crystal_diffusivity_field(micro, D_m(T))
    grains = fm.fem.subdomains.grain_subdomains(micro, F.Material(D=D_lattice))
    network = fm.fem.subdomains.grain_boundary_network(
        NETWORK_ID, micro, F.Material(D_0=D_gb, E_D=0.0)
    )
    grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    species_of = dict(zip((g.id for g in grains), grain_species, strict=True))

    # Keeping these as DOLFINx constants allows the sweep to change them without
    # re-assembling (it's only needed for efficiency)
    k = dolfinx.fem.Constant(mesh, scalar(1.0))
    width = dolfinx.fem.Constant(mesh, scalar(1.0))
    sources, boundary_conditions = [], []
    for c_grain in grain_species:
        exchange = {"c_g": c_grain, "c_n": c_gb}
        sources.append(
            F.ParticleSource(
                value=lambda c_g, c_n: (k / width) * (c_g - c_n),
                species=c_gb,
                volume=network,
                species_dependent_value=exchange,
            )
        )
        boundary_conditions.append(
            F.ParticleFluxBC(
                subdomain=network,
                species=c_grain,
                value=lambda c_g, c_n: k * (c_n - c_g),
                species_dependent_value=exchange,
            )
        )

    subdomains = [*grains, network]
    next_id = SURFACE_ID_0
    for locator, value in bcs:
        patches, mouths = fm.fem.subdomains.grain_surfaces(
            mesh, grains, locator, next_id
        )
        next_id = mouths.id + 1
        subdomains += [*patches, mouths]
        boundary_conditions += [
            F.FixedConcentrationBC(
                subdomain=p, value=value, species=species_of[p.grain_id]
            )
            for p in patches
        ]
        boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=mouths, value=value, species=c_gb)
        )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        subdomains=subdomains,
        species=[*grain_species, c_gb],
        sources=sources,
        boundary_conditions=boundary_conditions,
        temperature=T,
        settings=F.Settings(atol=1e-25, rtol=1e-10, transient=False),
    )
    model.show_progress_bar = False
    model.initialise()
    fm.fem.solvers.tune_direct_solver(model)
    return CellProblem(
        model, grains, grain_species, tensors, network, c_gb, D_gb, k, width
    )


def area_density(cp, micro):
    """The network area per unit volume ``S_v`` of the meshed network."""
    dim = micro.mesh.geometry.dim
    return fm.exports.measures.submesh_measure(cp.network) / B**dim


def run(cp, axis, S_v, ratio, f_gb) -> dict[str, float | str]:
    delta = f_gb / S_v
    k = 2.0 * cp.D_gb / delta
    cp.solve(delta, k)
    # their Eq. 21: volume-averaged flux
    q, _, _ = fm.exports.averages.averages(
        cp.grains, cp.species, cp.tensors, cp.network, cp.c_gb, delta, cp.D_gb
    )
    D_eff = q[axis] * B / (C_IN - C_OUT)
    return dict(
        f_gb=f_gb,
        delta_nm=delta * 1e9,
        S_v_per_nm=S_v * 1e-9,
        k_m_per_s=k,
        equilibration_length_nm=fm.materials.equilibration_length(delta, cp.D_gb, k)
        * 1e9,
        n_grains=len(cp.grains),
        ratio=ratio,
        D_m=D_m(T),
        D_eff=D_eff,
        D_eff_over_D_m=D_eff / D_m(T),
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
        polygons = [face * 1e9 for face in micro.boundaries]
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
            for p, q in micro.boundaries
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
    fig.savefig(OUTPUT_DIR / "li2022-fig4ab-codim.png", dpi=300)
    plt.close(fig)


def draw(rows, fname, f_lo, f_hi, logx):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
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
        axp.text(
            0.05, 0.9, f"$D_{{GB}}/D_m$ = {r:g}", transform=axp.transAxes, fontsize=18
        )
        if logx:
            axp.set_xscale("log")
            if r > 1:
                axp.set_yscale("log")
        axp.set_xlim(f_lo, f_hi)
        axp.legend(fontsize=13, loc="lower right" if r > 1 else "upper right")
    for axp in axes:
        axp.set_box_aspect(1.1)
        axp.tick_params(axis="both", which="both", labelsize=16)
        axp.set_xlabel("Boundary volume fraction $f_{GB}$", fontsize=18)
    for axp in axes[:1]:
        axp.set_ylabel(r"$D^{eff}/D_m$", fontsize=20)
    fig.suptitle(
        "Li et al. 2022 Fig. 4, codim-1 boundary\nThickness δ (k = 2 D_GB/δ)",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=300)


# ----------------------------------------------------------------------------
def write_csv(rows, fname):
    """Export every simulation field and reference values at each sample."""
    if not rows:
        return
    records = []
    for row in rows:
        record = dict(row)
        f, ratio = row["f_gb"], row["ratio"]
        record["hart_D_eff_over_D_m"] = hart(f, ratio)
        record["hs_D_eff_over_D_m"] = hashin_shtrikman(f, ratio)
        record["col_z_exact_D_eff_over_D_m"] = codim_parallel(f, ratio)
        records.append(record)
    with fname.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.use("Agg")
    comm = MPI.COMM_WORLD
    rows = []
    micros = {}
    for structure, axis in SIMULATED:
        micro = make(structure, comm)
        micros[structure] = micro
        bcs = boundary_conditions(axis, B, 1e-6 * B)
        S_v = None
        for ratio in RATIOS:
            # one problem per D_gb; the f_GB sweep below only moves constants
            cp = cell_problem(micro, ratio * D_m(T), bcs)
            if S_v is None:
                S_v = area_density(cp, micro)
                if comm.rank == 0:
                    print(
                        f"{LABEL[structure]}: {micro.n_grains} grains, S_v = {S_v * 1e-9:.4f} /nm; "  # noqa: E501
                        f"delta = {0.3 / S_v * 1e9:.2f} nm at f_GB = 0.3, {0.01 / S_v * 1e9:.3f} nm at 0.01"  # noqa: E501
                    )
            for f_gb in [*F_GB_THIN, *F_GB_THEIRS]:
                r = run(cp, axis, S_v, ratio, f_gb)
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
    write_csv(rows, OUTPUT_DIR / "li2022-fig4-codim-thin.csv")
    draw_fig4ab(micros["iso"], micros["col_x"], F_GB_THEIRS[0])
    draw(rows, OUTPUT_DIR / "li2022-fig4-codim-thin.png", 0.002, 0.8, logx=True)


if __name__ == "__main__":
    main()
