"""Measure segmentation error and verify raster orientation readback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from festim_microstructure.ebsd.orientation import (
    ROT_X_180,
    cubic_disorientation_angle,
    qconj,
    qmul,
)
from festim_microstructure.ebsd.settings import Settings
from festim_microstructure.formats.ebsd import read_ebsd
from festim_microstructure.plotting import (
    scale_bar_ax,
    use_agg,
)

__all__ = [
    "L2Stats",
    "QualityPanels",
    "SegmentationError",
    "format_report",
    "l2_stats",
    "per_grain_rms",
    "segmentation_error",
    "theta_field",
    "verify_readback",
    "write_csv",
    "write_orientation_png",
    "write_png",
    "write_quality_png",
]


def theta_field(qgrid, qref):
    """Per-pixel disorientation (degrees) between two (ny, nx, 4) quat fields."""
    a = qgrid.reshape(-1, 4)
    b = qref.reshape(-1, 4)
    return cubic_disorientation_angle(qmul(qconj(a), b)).reshape(qgrid.shape[:2])


@dataclass(frozen=True)
class L2Stats:
    """Order statistics of a disorientation population.

    ``n == 0`` is a real outcome (a grain set with no voxel in it), so the
    remaining fields are NaN rather than absent; ``bool(stats)`` is False then.
    """

    n: int
    area: float = float("nan")
    l2: float = float("nan")  #: deg * length
    rms: float = float("nan")  #: deg, = l2 / sqrt(area)
    mean: float = float("nan")
    median: float = float("nan")
    p99: float = float("nan")
    max: float = float("nan")
    threshold: float | None = None
    frac_over: float | None = None

    def __bool__(self) -> bool:
        """Return whether this error distribution has sampled pixels."""
        return self.n > 0


@dataclass
class SegmentationError:
    """What representing each pixel by its grain's orientation costs.

    ``theta`` is the per-pixel disorientation with NaN off the grains, and the
    three populations are disjoint views of it: ``indexed`` (a voxel in a grain
    of its own), ``backfilled`` (a voxel handed to its nearest grain), and
    ``all`` (both together).
    """

    theta: Any  #: (ny, nx), NaN where no grain
    good: Any  #: (ny, nx) bool, the indexed population
    filled: Any  #: (ny, nx) bool, the back-filled population
    ncells: int
    vox: tuple
    all: L2Stats
    indexed: L2Stats
    backfilled: L2Stats
    grain_rms: Any = None  #: per-grain RMS theta, id order
    grain_npx: Any = None  #: per-grain voxel count, id order
    voxel: L2Stats | None = None  #: pixel orientation transcription check, if available


def l2_stats(theta, mask, vox, threshold=None):
    """Return L2 and order statistics for selected disorientation values."""
    t = np.asarray(theta)[mask]
    a = vox[0] * vox[1]
    n = t.size
    if n == 0:
        return L2Stats(n=0)
    sq = float(np.sum(t**2))
    over = None if threshold is None else float((t > threshold).mean())
    return L2Stats(
        n=int(n),
        area=n * a,
        l2=float(np.sqrt(sq * a)),
        rms=float(np.sqrt(sq / n)),
        mean=float(t.mean()),
        median=float(np.median(t)),
        p99=float(np.percentile(t, 99)),
        max=float(t.max()),
        threshold=None if threshold is None else float(threshold),
        frac_over=over,
    )


def per_grain_rms(theta, cellids, ncells):
    """(rms, count) per cell id 1..ncells, over the voxels assigned to it."""
    flat_id = cellids.ravel()
    flat_t = np.asarray(theta).ravel()
    keep = flat_id > 0
    cnt = np.bincount(flat_id[keep], minlength=ncells + 1)[1:]
    ssq = np.bincount(flat_id[keep], weights=flat_t[keep] ** 2, minlength=ncells + 1)[
        1:
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(ssq / np.where(cnt > 0, cnt, np.nan))
    return rms, cnt


def segmentation_error(
    qgrid, cellids, qcell, vox, ok=None, threshold=None, qvox=None, backfilled=None
):
    """Return segmentation and optional transcription-error diagnostics.

    ``qgrid`` and ``qvox`` are source and imported pixel quaternions;
    ``qcell`` contains the symmetry-aware mean per grain. Zero labels are
    unassigned. ``ok`` is the quality mask. ``backfilled`` marks pixels
    without original DefDAP membership, including rejected and pruned pixels;
    these contribute only to the filled population, never indexed error.
    """
    ncells = int(cellids.max())
    assigned = cellids > 0
    ok = np.ones_like(assigned) if ok is None else np.asarray(ok, dtype=bool)
    filled_in = ~ok if backfilled is None else np.asarray(backfilled, dtype=bool)

    qref = qcell[np.maximum(cellids - 1, 0)]  # id 0 -> cell 1, masked out below
    theta = theta_field(qgrid, qref)

    good = assigned & ~filled_in & ok  # in a grain of its own: the honest set
    filled = assigned & filled_in
    rms, cnt = per_grain_rms(
        np.where(good, theta, 0.0), np.where(good, cellids, 0), ncells
    )
    return SegmentationError(
        theta=np.where(assigned, theta, np.nan),
        good=good,
        filled=filled,
        ncells=ncells,
        vox=tuple(vox),
        all=l2_stats(theta, assigned, vox, threshold),
        indexed=l2_stats(theta, good, vox, threshold),
        backfilled=l2_stats(theta, filled, vox, threshold),
        grain_rms=rms,
        grain_npx=cnt,
        voxel=None if qvox is None else l2_stats(theta_field(qgrid, qvox), ok, vox),
    )


def format_report(res: SegmentationError, label="segmentation"):
    """Format segmentation and readback error statistics."""
    lines = []
    a = res.indexed
    if not a:
        return [f"{label}: no indexed voxel is assigned to a grain"]
    lines.append(
        f"{label} L2: RMS {a.rms:.3f} deg over {a.n} indexed voxels "
        f"(||theta||_2 = {a.l2:.4g} deg*len over {a.area:.4g} len^2)"
    )
    lines.append(
        f"{label}   : mean {a.mean:.3f}, median {a.median:.3f}, "
        f"p99 {a.p99:.3f}, max {a.max:.3f} deg"
    )
    if a.frac_over is not None:
        lines.append(
            f"{label}   : {100 * a.frac_over:.2f} % of voxels sit further "
            f"than the {a.threshold:g} deg segmentation threshold from "
            "their own grain's orientation"
        )
    b, c = res.backfilled, res.all
    if b:
        lines.append(
            f"{label}   : back-filled voxels ({b.n}, excluded above): "
            f"RMS {b.rms:.3f} deg, max {b.max:.3f} deg"
        )
        lines.append(
            f"{label}   : over every voxel that has a grain ({c.n}, the two "
            f"populations together): RMS {c.rms:.3f} deg, "
            f"||theta||_2 = {c.l2:.4g} deg*len"
        )
    rms, cnt = res.grain_rms, res.grain_npx
    fin = np.flatnonzero(np.isfinite(rms))
    if fin.size:
        worst = fin[np.argsort(rms[fin])[::-1][:5]]
        lines.append(
            f"{label}   : worst grains (id: RMS deg, px) "
            + ", ".join(f"{i + 1}: {rms[i]:.2f}, {cnt[i]}" for i in worst)
        )
    if res.voxel is not None:
        v = res.voxel
        verdict = "" if v.max < 1e-3 else "  <-- NOT a round trip"
        lines.append(
            f"transcription L2: RMS {v.rms:.2e} deg, max {v.max:.2e} deg "
            f"(imported pixels vs source CTF){verdict}"
        )
    return lines


def write_png(
    path, res: SegmentationError, cellids, unit="um", dpi=150, log=print, flip_y=False
):
    """theta map, its distribution, and the per-grain RMS on the same map.

    The segmentation threshold is read off ``res``; it was already recorded
    there when the statistics were taken.
    """
    threshold = res.indexed.threshold
    if flip_y:
        cellids = cellids[::-1]
    plt = use_agg()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15})

    theta = res.theta[::-1] if flip_y else res.theta
    ny, nx = theta.shape
    vx, vy = res.vox
    kw = dict(interpolation="nearest", origin="lower", extent=(0, nx * vx, 0, ny * vy))
    a = res.indexed
    vmax = max(a.p99, 1e-3)
    aspect = ny * vy / (nx * vx)

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 3.0 + 5.5 * aspect), layout="constrained"
    )

    ax = axes[0]
    im = ax.imshow(theta, cmap="inferno", vmin=0, vmax=vmax, **kw)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("theta (deg)")
    ax.set_title(f"per-pixel segmentation error: RMS {a.rms:.2f} deg")
    ax.set_xlabel(f"clipped at p99 = {vmax:.2f} deg")
    ax.set_xticks([])
    ax.set_yticks([])
    scale_bar_ax(ax, nx * vx, unit)

    ax = axes[1]
    good = res.theta[res.good]
    filled = res.theta[res.filled]
    hi = max(np.percentile(good, 99.9) * 1.5, threshold or 0, 1e-3)
    bins = np.linspace(0, hi, 80)
    ax.hist(good, bins=bins, color="0.35", label=f"indexed ({good.size})")
    if filled.size:
        ax.hist(
            np.clip(filled, 0, hi),
            bins=bins,
            color="tab:orange",
            alpha=0.8,
            label=f"back-filled ({filled.size}), clipped",
        )
    ax.set_yscale("log")
    ax.axvline(a.rms, color="tab:red", lw=2, label=f"RMS {a.rms:.2f} deg")
    ax.axvline(
        a.median, color="tab:blue", lw=2, ls="--", label=f"median {a.median:.2f}"
    )
    if threshold and threshold <= hi:
        ax.axvline(
            threshold, color="k", lw=2, ls=":", label=f"threshold {threshold:g} deg"
        )
    ax.set_xlabel("theta to the grain orientation (deg)")
    ax.set_ylabel("voxels")
    ax.set_title("distribution")
    ax.legend(fontsize=12)

    ax = axes[2]
    rms = res.grain_rms
    painted = np.full(theta.shape, np.nan)
    m = cellids > 0
    painted[m] = rms[cellids[m] - 1]
    im = ax.imshow(painted, cmap="viridis", **kw)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("RMS theta (deg)")
    ax.set_title(f"{res.ncells} grains, worst {np.nanmax(rms):.2f} deg")
    ax.set_xticks([])
    ax.set_yticks([])
    scale_bar_ax(ax, nx * vx, unit)

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    if log:
        log(f"  wrote {path}")
    return path


def write_csv(path, res: SegmentationError, log=print):
    """Write the per-grain segmentation error table."""
    rms, cnt = res.grain_rms, res.grain_npx
    with open(path, "w") as fh:
        fh.write("cell_id,n_voxels,rms_theta_deg\n")
        for k, (r, c) in enumerate(zip(rms, cnt), start=1):
            fh.write(f"{k},{int(c)},{'' if not np.isfinite(r) else f'{r:.6g}'}\n")
    if log:
        log(f"  wrote {path}")
    return path


@dataclass
class QualityPanels:
    """The arrays and settings the three quality panels are drawn from.

    One record rather than eight positional arguments: they all come from the
    same conversion, and a caller that swapped ``ok`` for ``unassigned`` would
    previously have got a plausible-looking and wrong figure.
    """

    diag: dict  #: per-pixel quality columns, as build_grid returned them
    ok: Any  #: (ny, nx) bool quality mask
    unassigned: Any  #: (ny, nx) bool, empty before the back-fill
    cellids: Any  #: (ny, nx) grain id per pixel
    settings: Settings
    vox: tuple  #: (dx, dy)
    unit: str
    flip_y: bool = False


def verify_readback(path, qgrid, ok, cellids, flip_y):
    """Require identical labels, quality masks and measured pixel rotations."""
    back = read_ebsd(path)
    expected_cells = cellids[::-1] if flip_y else cellids
    expected_ok = ok[::-1] if flip_y else ok
    expected_q = qmul(ROT_X_180, qgrid[::-1]) if flip_y else qgrid
    if not np.array_equal(back.labels, expected_cells):
        raise ValueError("read-back grain labels differ")
    if not np.array_equal(back.indexed, expected_ok):
        raise ValueError("read-back indexed mask differs")
    dis = theta_field(expected_q, back.pixel_quats)
    if np.max(dis[expected_ok]) > 1e-3:
        raise ValueError("read-back measured orientations differ")
    return [
        "read-back: grain labels and indexed mask identical",
        f"read-back: indexed pixel orientations max {dis[expected_ok].max():.2e} deg",
    ]


def write_orientation_png(path, data, log=print, unit="um"):
    """Native grain and cubic IPF-Z maps with a matching crystallographic key.

    Rotate sample Z into the crystal frame, fold by cubic symmetry into
    0 <= y <= x <= z, and color the [001], [101], [111] vertices red,
    green, blue. The same mapping draws the key and both orientation maps.
    """
    from festim_microstructure.ebsd.orientation import qconj, qmul
    from festim_microstructure.plotting import draw_raster

    def colors(q):
        direction = np.zeros_like(q)
        direction[..., 3] = 1
        v = np.sort(np.abs(qmul(qmul(qconj(q), direction), q)[..., 1:]), axis=-1)
        y, x, z = v[..., 0], v[..., 1], v[..., 2]
        rgb = np.stack((z - x, x - y, y), axis=-1)
        rgb = np.sqrt(np.maximum(rgb, 0))
        return rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1e-15)

    plt = use_agg()
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
    extent = (0, data.extent[0], 0, data.extent[1])
    draw_raster(axes[0], data.labels, data.vox)
    axes[0].set_title("UPXO/DefDAP grains")
    for ax, quats, mask, title in (
        (
            axes[1],
            data.pixel_quats,
            data.indexed,
            "measured IPF-Z (rejected pixels grey)",
        ),
        (
            axes[2],
            data.grain_quats[np.maximum(data.labels - 1, 0)],
            data.labels > 0,
            "grain-mean IPF-Z",
        ),
    ):
        rgb = colors(quats)
        rgb[~mask] = 0.6
        ax.imshow(rgb, origin="lower", interpolation="nearest", extent=extent)
        ax.set_title(title)
    from matplotlib.tri import Triangulation

    # Barycentric directions give a key consistent with the map coloring.
    n = 40
    coords, directions = [], []
    vertices = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            weights = np.array([1 - (i + j) / n, i / n, j / n])
            coords.append((i / n + j / (2 * n), j / n))
            directions.append(weights @ vertices)
    coords, directions = np.asarray(coords), np.asarray(directions)
    v = np.sort(directions, axis=-1)
    rgb = np.sqrt(
        np.maximum(np.column_stack((v[:, 2] - v[:, 1], v[:, 1] - v[:, 0], v[:, 0])), 0)
    )
    rgb /= rgb.max(axis=1, keepdims=True)
    tri = Triangulation(coords[:, 0], coords[:, 1])
    from matplotlib.collections import PolyCollection

    axes[3].add_collection(
        PolyCollection(
            coords[tri.triangles],
            facecolors=rgb[tri.triangles].mean(axis=1),
            edgecolors="none",
        )
    )
    axes[3].autoscale_view()
    for xy, label in (
        ((0, 0), "[001] red"),
        ((1, 0), "[101] green"),
        ((0.5, 1), "[111] blue"),
    ):
        axes[3].text(*xy, label, ha="center")
    axes[3].set_title("cubic IPF-Z key")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[:3]:
        scale_bar_ax(ax, data.extent[0], unit)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if log:
        log(f"  wrote {path}")
    return path


def write_quality_png(path, panels: QualityPanels, log=print):
    """MAD, quality rejection reasons, and filled/pruned grain pixels."""
    diag, ok, cellids = panels.diag, panels.ok, panels.cellids
    unassigned, opt, vox = panels.unassigned, panels.settings, panels.vox
    unit, flip_y = panels.unit, panels.flip_y

    plt = use_agg()
    from matplotlib.colors import ListedColormap

    def orient(a):
        """Restore CTF row order when the output raster was flipped."""
        return a[::-1] if flip_y else a

    ny, nx = ok.shape
    mad, err, bands, phase = (diag.get(k) for k in ("MAD", "Error", "Bands", "Phase"))

    # rejection reason, first failing test wins
    reason = np.zeros((ny, nx), dtype=int)  # 0 kept
    if phase is not None:
        reason[(reason == 0) & (phase != opt.phase)] = 4
    if err is not None and not opt.allow_error:
        reason[(reason == 0) & (err != 0)] = 1
    if mad is not None:
        reason[(reason == 0) & (mad > opt.max_mad)] = 2
    if bands is not None and opt.min_bands:
        reason[(reason == 0) & (bands < opt.min_bands)] = 3
    reason[ok] = 0
    labels = ["kept", "Error != 0", f"MAD > {opt.max_mad:g}", "Bands < min", "phase"]
    counts = np.bincount(reason.ravel(), minlength=5)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 2.4 + 5.2 * ny / nx), layout="constrained"
    )
    kw = dict(
        interpolation="nearest", origin="lower", extent=(0, nx * vox[0], 0, ny * vox[1])
    )

    ax = axes[0]
    if mad is not None:
        im = ax.imshow(
            orient(mad), cmap="viridis", vmin=0, vmax=max(opt.max_mad * 1.5, 1.0), **kw
        )
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("MAD (deg)")
        ax.set_title(f"MAD: {int((mad > opt.max_mad).sum())} px over max_mad")
    else:
        ax.set_title("no MAD column")

    ax = axes[1]
    cmap = ListedColormap(["white", "tab:red", "tab:orange", "tab:purple", "tab:brown"])
    im = ax.imshow(orient(reason), cmap=cmap, vmin=-0.5, vmax=4.5, **kw)
    fig.colorbar(im, ax=ax, ticks=range(5), fraction=0.046).set_ticklabels(labels)
    ax.set_title(
        f"quality rejected: {int((~ok).sum())}/{ok.size} px "
        f"({100 * (~ok).mean():.1f} %)"
    )

    ax = axes[2]
    ncell = int(cellids.max())
    perm = np.random.default_rng(0).permutation(ncell) + 1
    shown = np.where(cellids > 0, perm[np.maximum(cellids - 1, 0)], np.nan)
    ax.imshow(orient(shown), cmap="tab20", **kw)
    filled = unassigned & (cellids > 0)
    ax.imshow(
        orient(np.where(filled, 1.0, np.nan)),
        cmap=ListedColormap(["black"]),
        alpha=0.45,
        **kw,
    )
    ax.set_title(
        f"{ncell} grains\n{int(filled.sum())} filled pixels (dark), "
        f"{int((cellids == 0).sum())} empty"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        scale_bar_ax(ax, nx * vox[0], unit)
    fig.suptitle(
        f"{Path(opt.ctf).name}: "
        + ", ".join(f"{labels[k]} {counts[k]}" for k in range(5) if counts[k])
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if log:
        log(f"  wrote {path}")
    return path
