"""Measure segmentation error and verify raster orientation readback."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from festim_microstructure._binaries import find_binary, subprocess_env
from festim_microstructure.ebsd.orientation import (
    cubic_disorientation_angle,
    qconj,
    qmul,
    rodrigues_to_quat,
)
from festim_microstructure.ebsd.settings import Settings
from festim_microstructure.formats.tesr import read_tesr_full
from festim_microstructure.plotting import (
    annotate_png,
    append_key,
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
    "render_checks",
    "segmentation_error",
    "theta_field",
    "verify_readback",
    "write_csv",
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
    voxel: L2Stats | None = None  #: **oridata transcription check, if available


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

    qgrid   (ny, nx, 4) per-pixel quaternions straight from the .ctf
    cellids (ny, nx)    grain id per pixel, 0 = unassigned, as in the tesr
    qcell   (ncell, 4)  one quaternion per grain, as in the tesr's **cell/*ori
    ok      (ny, nx)    quality mask, i.e. the tesr's **oridef; failed voxels
                        carry a meaningless orientation and are reported apart
    qvox    (ny, nx, 4) the tesr's **oridata, for the transcription check
    backfilled (ny, nx) voxels fill_holes gave to their nearest cell rather
                        than to a grain of their own -- quality rejections
                        *and* grains the min_pixels prune removed. Their theta
                        is the price of filling, not a segmentation error, so
                        they are counted apart. Defaults to ~ok, which catches
                        only the first kind.
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
            f"(**oridata vs the raw Euler angles){verdict}"
        )
    return lines


def write_png(path, res: SegmentationError, cellids, unit="um", dpi=150, log=print):
    """theta map, its distribution, and the per-grain RMS on the same map.

    The segmentation threshold is read off ``res``; it was already recorded
    there when the statistics were taken.
    """
    threshold = res.indexed.threshold
    plt = use_agg()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15})

    theta = res.theta
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
    good = theta[res.good]
    filled = theta[res.filled]
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


def _run(cmd, cwd, log):
    """Run a Neper command, returning True on success and reporting on failure.

    The directories of ``cmd[0]`` (neper) and of any ``-povray`` argument are
    put on the child's PATH, since Neper may look the renderer up by name.
    """
    extra = [cmd[cmd.index("-povray") + 1]] if "-povray" in cmd else []
    env = subprocess_env(cmd[0], *extra)
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if out.returncode:
        tail = (out.stderr or out.stdout).strip().splitlines()[-3:]
        log(f"  WARNING: {Path(cmd[0]).name} {cmd[1]} failed: {' / '.join(tail)}")
    return out.returncode == 0


def render_checks(tesr, width, unit="um", neper="neper", povray="povray", log=print):
    """Render the written raster with neper -V. Returns the paths written.

    Two images, both of the .tesr this module just wrote and of nothing else,
    which is why they belong here rather than in the meshing stage:

      <stem>-ori.png     per-voxel orientation, IPF-Z, with the colour key
      <stem>-grains.png  cell ids in Neper's integer palette

    Look at these before trusting anything downstream: an inverted orientation
    convention shows up as IPF colours that disagree with AZtec or MTEX, and a
    bad segmentation as speckle or as obviously back-filled grains.

    -V colours by orientation but does not print the key
    (neper.info/tutorials/orientation_color_key.html), so the key is built the
    way that page documents -- tessellate the standard stereographic triangle,
    mesh it, read the node colours out with `-statnode col_stdtriangle` -- then
    pasted beside the map and deleted.

    Needs the neper binary and POV-Ray. A missing one is reported and skipped,
    since everything else the conversion produces is pure Python.
    """
    tesr = Path(tesr)
    work, stem = tesr.parent, tesr.stem
    neper = (
        find_binary("neper", neper, "FM_NEPER_BIN", required=False) if neper else None
    )
    if neper is None:
        log("  note: neper not found, skipping the rendered check images")
        return []
    povray = find_binary("povray", povray, "FM_POVRAY_BIN", required=False) or povray

    written = []
    for name, opts in (
        ("ori", ["-datavoxcol", "ori", "-datavoxcolscheme", "ipf"]),
        ("grains", []),
    ):
        png = work / f"{stem}-{name}.png"
        cmd = [neper, "-V", tesr.name, "-povray", povray, *opts, "-print", png.stem]
        if not _run(cmd, work, log):
            continue
        # neper -V frames the flat map in the middle of a 3D canvas, so the
        # border comes off first; after that the image width *is* `width`
        annotate_png(png, width, unit, trim_border=True, log=log)
        written.append(png)

    ori = work / f"{stem}-ori.png"
    if ori in written:
        tri, key = f"{stem}-stdtriangle", work / f"{stem}-ipfkey.png"
        if (
            _run(
                [
                    neper,
                    "-T",
                    "-n",
                    "1",
                    "-domain",
                    "stdtriangle(20)",
                    "-dim",
                    "2",
                    "-o",
                    tri,
                ],
                work,
                log,
            )
            and _run(
                [
                    neper,
                    "-M",
                    f"{tri}.tess",
                    "-cl",
                    "0.02",
                    "-statnode",
                    "col_stdtriangle",
                ],
                work,
                log,
            )
            and _run(
                [
                    neper,
                    "-V",
                    f"{tri}.msh",
                    "-povray",
                    povray,
                    "-datanodecol",
                    f"col:file({tri}.stnode)",
                    "-dataeltcol",
                    "from_nodes",
                    "-dataelt2dedgerad",
                    "0",
                    "-dataelt1drad",
                    "0.001",
                    "-showelt1d",
                    "all",
                    "-imagesize",
                    "800:400",
                    "-print",
                    key.stem,
                ],
                work,
                log,
            )
        ):
            append_key(ori, key, log=log)
            key.unlink()
        for ext in (".tess", ".msh", ".stnode"):
            (work / (tri + ext)).unlink(missing_ok=True)
    return written


def verify_readback(path, qgrid, ok, cellids, flip_y):
    """Re-read the written tesr, compare with what was meant, return a report.

    Three checks: the cell map is identical, `**oridef` is the quality mask,
    and every voxel orientation read back is the same rotation as the raw Euler
    triple. The file holds a fundamental-zone representative, so equality is
    only expected up to the symmetry group -- which is what a disorientation
    measures, hence ~0 rather than exactly 0.
    """
    back = read_tesr_full(path)
    exp_cells = cellids[::-1] if flip_y else cellids
    exp_ok = ok[::-1] if flip_y else ok
    exp_q = qgrid[::-1] if flip_y else qgrid
    same_cells = np.array_equal(back["cells"], exp_cells)
    report = [
        f"read-back: cell ids {'identical' if same_cells else 'DIFFER'} "
        f"({back['nx']} x {back['ny']} voxels)"
    ]
    if "vox_ori" in back:
        same_def = np.array_equal(back["oridef"], exp_ok)
        report.append(
            f"read-back: **oridef {'identical' if same_def else 'DIFFERS'} "
            "to the quality mask"
        )
        dis = cubic_disorientation_angle(
            qmul(qconj(exp_q.reshape(-1, 4)), rodrigues_to_quat(back["vox_ori"]))
        )
        report.append(
            f"read-back: voxel orientations vs raw Euler angles: max "
            f"{dis.max():.2e} deg, mean {dis.mean():.2e} deg over {dis.size} voxels"
            + ("" if dis.max() < 1e-3 else "  <-- NOT a round trip")
        )
    return report


def write_quality_png(path, panels: QualityPanels, log=print):
    """Three panels tracing every grey pixel of `neper -V ... -datavoxcol ori`.

    Neper paints a voxel grey where `**oridef` is 0, and that is written from
    the quality mask, so a grey pixel is one the .ctf's own columns failed. The
    panels are the MAD column (which is what the max_mad cut has to be chosen
    against), which test rejected each pixel, and which pixels the cell map
    back-filled because they were rejected or belonged to a pruned grain.
    """
    diag, ok, cellids = panels.diag, panels.ok, panels.cellids
    unassigned, opt, vox = panels.unassigned, panels.settings, panels.vox
    unit, flip_y = panels.unit, panels.flip_y

    plt = use_agg()
    from matplotlib.colors import ListedColormap

    def orient(a):
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
        f"rejected (grey in -V): {int((~ok).sum())}/{ok.size} px "
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
        f"{ncell} cells, {int(filled.sum())} px back-filled (dark), "
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
