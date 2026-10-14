"""Shared raster colours, Matplotlib defaults, and scale bars."""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "RCPARAMS",
    "draw_raster",
    "format_length",
    "nice_length",
    "scale_bar_ax",
    "use_agg",
]


def nice_length(width, fraction=0.25):
    """Largest 1/2/5 x 10^k not exceeding `fraction` of `width`."""
    target = width * fraction
    k = np.floor(np.log10(target))
    for m in (5.0, 2.0, 1.0):
        if m * 10**k <= target:
            return m * 10**k
    return 10 ** (k - 1) * 5.0


def format_length(value, unit):
    """Format a length with a compact value and display unit."""
    return f"{value:g} {'um' if unit in ('micron', 'microns') else unit}"


def scale_bar_ax(ax, width_units, unit="um", length=None, color="white"):
    """Scale bar in the lower right of a matplotlib axes in data units."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    length = nice_length(width_units) if length is None else length
    bar = AnchoredSizeBar(
        ax.transData,
        length,
        format_length(length, unit),
        "lower right",
        pad=0.5,
        borderpad=0.8,
        sep=4,
        color=color,
        frameon=True,
        size_vertical=0.012 * width_units,
        fontproperties={"size": 16},
    )
    bar.patch.set_facecolor("black")
    bar.patch.set_alpha(0.45)
    bar.patch.set_edgecolor("none")
    ax.add_artist(bar)
    return bar


RCPARAMS: dict[Literal["font.size", "axes.titlesize", "figure.titlesize"], int] = {
    "font.size": 16,
    "axes.titlesize": 16,
    "figure.titlesize": 16,
}


def use_agg():
    """Agg backend + the pipeline's font sizes, for functions that write a PNG.

    Kept out of module scope so importing this file leaves a caller's
    matplotlib alone.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    for key, value in RCPARAMS.items():
        plt.rcParams[key] = value
    return plt


def draw_raster(ax, cells, vox, alpha=1.0):
    """Raster coloured by cell id with Matplotlib's tab20; returns the RGBA.

    Colours repeat every 20 grains and empty voxels (id 0) stay transparent.
    `alpha` below 1 fades the background to keep boundary overlays readable.
    """
    from matplotlib import colormaps

    ny, nx = cells.shape
    ncell = int(cells.max())
    lut = np.zeros((ncell + 1, 4), dtype=np.uint8)  # row 0 = empty = transparent
    palette = colormaps["tab20"]
    lut[1:] = palette(np.arange(ncell) % palette.N, bytes=True)
    lut[1:, 3] = round(255 * alpha)
    rgba = lut[cells]
    ax.imshow(
        rgba,
        interpolation="nearest",
        origin="lower",
        extent=(0, nx * vox[0], 0, ny * vox[1]),
    )
    ax.set_xlim(0, nx * vox[0])
    ax.set_ylim(0, ny * vox[1])
    return rgba
