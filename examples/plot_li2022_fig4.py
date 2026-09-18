"""Recreate the Li 2022 Fig. 4 PNGs from simulation CSV outputs.

Run this file to read both CSVs from results/li2022_fig4/ and
results/li2022_fig4_codim/, relative to this script, and save each PNG beside
its CSV. Additional -linear.png figures show only the original 0.3--0.7
volume-fraction samples on linear axes. No command-line arguments are needed.

Requires only NumPy and Matplotlib; no simulation dependencies are imported.
Analytical reference curves are evaluated on the same grid as the simulations'
plotting routines. Simulation markers come from the CSV records.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RATIOS = [10.0, 0.1]
PAPER_F_GB = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def read_csv(path):
    """Load plotted fields while allowing additional diagnostic columns."""
    required = {"structure", "ratio", "f_gb", "D_eff_over_D_m"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = []
        for line, row in enumerate(reader, start=2):
            try:
                rows.append(
                    dict(
                        structure=row["structure"],
                        ratio=float(row["ratio"]),
                        f_gb=float(row["f_gb"]),
                        D_eff_over_D_m=float(row["D_eff_over_D_m"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid numeric data") from exc
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def hart(f, r):
    """Their Eq. 28: phases in parallel. Exact for Col_I along the columns."""
    return 1.0 + f * (r - 1.0)


def hashin_shtrikman(f, r):
    """Their Eq. 33 (Chen & Schuh 2007): D_eff/D_m for an isometric polycrystal
    with the boundary as the connected phase."""
    return r + (1.0 - f) / (1.0 / (1.0 - r) + f / (3.0 * r))


def draw_volumetric(rows, fname, f_lo=0.002, f_hi=0.8, logx=True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), layout="constrained")
    f = np.geomspace(f_lo, f_hi, 200) if logx else np.linspace(f_lo, f_hi, 200)
    style = {
        "col_z": dict(marker="*", ms=9, ls="none", color="tab:red", label="Col_I(Z)"),
        "col_x": dict(
            marker="*", ms=9, ls="none", color="tab:red", mfc="none", label="Col_I(X)"
        ),
        "iso": dict(
            marker="o", ms=6, ls="none", color="tab:purple", mfc="none", label="Iso"
        ),
    }
    for axp, r in zip(axes.flat, RATIOS, strict=True):
        axp.plot(f, hart(f, r), "-", color="0.6", label="Hart, their Eq. 28")
        axp.plot(
            f, hashin_shtrikman(f, r), ":", color="tab:blue", label="HS, their Eq. 33"
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
        axp.set_title(f"$D_{{GB}}/D_m$ = {r:g}", fontsize=18, pad=12)
        if logx:
            axp.set_xscale("log")
            if r > 1:
                axp.set_yscale("log")
        axp.set_xlim(f_lo, f_hi)
        if not logx:
            axp.set_xticks(PAPER_F_GB)
        axp.legend(fontsize=13, loc="upper left" if r > 1 else "lower left")
    for axp in axes:
        axp.set_box_aspect(1.1)
        axp.tick_params(axis="both", which="both", labelsize=16)
        axp.set_xlabel("Boundary volume fraction $f_{GB}$", fontsize=18)
    for axp in axes[:1]:
        axp.set_ylabel(r"$D^{eff}/D_m$", fontsize=20)
    fig.suptitle(
        "Li et al. 2022 Fig. 4 E,H\nVoronoi foam / columns, volumetric GB band",
        fontsize=16,
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def codim_parallel(f, r):
    """The codim model with the network parallel to the flux: exact."""
    return 1.0 + f * r


def draw_codim(rows, fname, f_lo=0.002, f_hi=0.8, logx=True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), layout="constrained")
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
        axp.set_title(f"$D_{{GB}}/D_m$ = {r:g}", fontsize=18, pad=12)
        if logx:
            axp.set_xscale("log")
            if r > 1:
                axp.set_yscale("log")
        axp.set_xlim(f_lo, f_hi)
        if not logx:
            axp.set_xticks(PAPER_F_GB)
        axp.legend(fontsize=13, loc="upper left" if r > 1 else "lower left")
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
    fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    datasets = (
        (RESULTS_DIR / "li2022_fig4" / "li2022-fig4.csv", draw_volumetric),
        (RESULTS_DIR / "li2022_fig4_codim" / "li2022-fig4-codim-thin.csv", draw_codim),
    )
    for path, draw in datasets:
        rows = read_csv(path)
        output = path.with_suffix(".png")
        draw(rows, output)
        print(f"Saved {output}")
        # Volumetric fractions differ slightly from targets due to mesh resolution.
        paper_rows = [
            row for row in rows
            if np.any(np.isclose(row["f_gb"], PAPER_F_GB, rtol=0, atol=1e-3))
        ]
        if not paper_rows:
            raise ValueError(f"{path}: no samples matching the original fractions")
        linear_output = path.with_name(f"{path.stem}-linear.png")
        draw(paper_rows, linear_output, f_lo=0.25, f_hi=0.75, logx=False)
        print(f"Saved {linear_output}")


if __name__ == "__main__":
    main()
