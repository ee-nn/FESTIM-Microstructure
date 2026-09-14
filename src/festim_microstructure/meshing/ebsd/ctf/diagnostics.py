"""Check images and read-back verification for a written ``.tesr``.

Everything here is optional: the conversion itself is pure NumPy/SciPy, while
the rendered checks shell out to Neper and POV-Ray and the panels need
matplotlib. Both are imported lazily so that importing the converter does not
require either.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...._binaries import find_binary, subprocess_env
from ..mesh_overlay import use_agg
from ..micrograph import annotate_png, append_key, scale_bar_ax
from ..orientation import (
    cubic_disorientation_angle,
    qconj,
    qmul,
    rodrigues_to_quat,
)
from ..segmentation_error import read_tesr_full
from .settings import Settings

__all__ = [
    "QualityPanels",
    "render_checks",
    "verify_readback",
    "write_quality_png",
]


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
