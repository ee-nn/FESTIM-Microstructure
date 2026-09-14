"""Compare meshed grain areas and boundaries with the source raster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from festim_microstructure.formats.msh4 import read_msh4
from festim_microstructure.formats.tesr import read_tesr, tesr_origin
from festim_microstructure.plotting import (
    draw_raster,
    scale_bar_ax,
    use_agg,
)

__all__ = [
    "AreaChange",
    "AreaReportOptions",
    "ChangeStats",
    "area_change",
    "edge_sides",
    "format_report",
    "measure",
    "overlay",
    "raster_areas",
    "summarise",
    "triangle_areas",
    "write_csv",
    "write_png",
]


@dataclass(frozen=True)
class ChangeStats:
    """Distribution of a per-grain percentage change."""

    n: int
    min: float = float("nan")
    mean: float = float("nan")
    median: float = float("nan")
    max: float = float("nan")
    abs_mean: float = float("nan")
    rms: float = float("nan")
    area_weighted_mean: float | None = None

    def __bool__(self) -> bool:
        return self.n > 0


@dataclass
class AreaChange:
    """Raster grain areas against their meshed counterparts.

    ``delta``/``area``/``ecd`` stay ``None`` when the face-to-cell mapping is
    not the identity and ``allow_mismatch`` was not given: every per-grain
    quantity downstream is indexed by face id, so the areas would be attached
    to the wrong grains. :func:`measure` turns that into an exception.
    """

    ncell: int
    nface: int
    vox: tuple
    cells: Any
    npx: Any
    identity: bool
    allow_mismatch: bool
    mapping: Any
    face_area: Any
    face_matched: Any
    displaced: float
    a_raster: Any
    a_mesh: Any
    mesh_total: float
    raster_total: float
    delta: Any = None  #: per-grain area change, %
    delta_ecd: Any = None  #: the same as an equivalent-diameter change, %
    area: ChangeStats | None = None
    ecd: ChangeStats | None = None

    @property
    def comparable(self) -> bool:
        """True when the per-grain comparison was actually made."""
        return self.delta is not None


@dataclass
class AreaReportOptions:
    """Where :func:`measure` writes, and how tolerant it is."""

    csv: str | None = None
    png: str | None = None
    unit: str = "um"
    dpi: int = 150
    allow_mismatch: bool = False


def raster_areas(cells, vox, ncell=None):
    """Area of every cell id 1..ncell, as (voxel count) x (voxel area)."""
    ncell = int(cells.max()) if ncell is None else ncell
    counts = np.bincount(cells.ravel(), minlength=ncell + 1)[1 : ncell + 1]
    return counts * vox[0] * vox[1], counts


def triangle_areas(xyz, tri):
    """Return ``(face tag, area, centroid)`` for every triangle."""
    tags = np.fromiter((t for t, _ in tri), dtype=np.int64, count=len(tri))
    nodes = np.array([v[:3] for _t, v in tri], dtype=np.int64)
    coord = np.zeros((max(xyz) + 1, 2))
    for t, c in xyz.items():
        coord[t] = c[:2]
    p = coord[nodes]  # (ntri, 3, 2)
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1])
        + b[:, 0] * (c[:, 1] - a[:, 1])
        + c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return tags, area, p.mean(axis=1)


def face_to_cell(cells, vox, origin, tags, area, centroid, ncell):
    """Map mesh faces to raster cells and measure unsigned displaced area."""
    ny, nx = cells.shape
    ix = np.clip(((centroid[:, 0] - origin[0]) / vox[0]).astype(int), 0, nx - 1)
    iy = np.clip(((centroid[:, 1] - origin[1]) / vox[1]).astype(int), 0, ny - 1)
    under = cells[iy, ix].astype(np.int64)  # raster cell each triangle sits on
    ntag = int(tags.max())

    key = tags * (ncell + 1) + under
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=area)
    utag, ucell = uk // (ncell + 1), uk % (ncell + 1)

    # Sort each face's candidate cells by accumulated area.
    order = np.lexsort((w, utag))
    ut, uc, uw = utag[order], ucell[order], w[order]
    last = np.flatnonzero(np.r_[ut[1:] != ut[:-1], True])

    mapping = np.zeros(ntag + 1, dtype=np.int64)
    best = np.zeros(ntag + 1)
    mapping[ut[last]] = uc[last]
    best[ut[last]] = uw[last]

    total = np.bincount(tags, weights=area, minlength=ntag + 1)
    same = utag == ucell  # area lying over the cell of its own id
    displaced = 1.0 - w[same].sum() / total.sum() if total.sum() else 0.0
    return mapping, total, best, displaced


def summarise(delta, weights=None) -> ChangeStats:
    """min / mean / median / max, plus the sign-blind and weighted versions."""
    d = np.asarray(delta, dtype=float)
    finite = np.isfinite(d)
    d = d[finite]
    if d.size == 0:
        return ChangeStats(n=0)
    weighted = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)[finite]
        weighted = float(np.sum(d * w) / np.sum(w))
    return ChangeStats(
        n=int(d.size),
        min=float(d.min()),
        mean=float(d.mean()),
        median=float(np.median(d)),
        max=float(d.max()),
        abs_mean=float(np.abs(d).mean()),
        rms=float(np.sqrt(np.mean(d**2))),
        area_weighted_mean=weighted,
    )


def area_change(tesr_path, msh4_path, allow_mismatch=False) -> AreaChange:
    """The whole diagnostic, as an :class:`AreaChange`."""
    cells, vox = read_tesr(tesr_path)
    origin = tesr_origin(tesr_path)
    xyz, _seg, tri = read_msh4(msh4_path)
    if not tri:
        raise ValueError(f"{msh4_path}: no 2D elements. Mesh with -dim all.")

    ncell = int(cells.max())
    a_ras, npx = raster_areas(cells, vox, ncell)
    tags, tarea, cent = triangle_areas(xyz, tri)
    mapping, f_total, f_best, displaced = face_to_cell(
        cells, vox, origin, tags, tarea, cent, ncell
    )

    ntag = int(tags.max())
    faces = np.unique(tags)
    identity = bool(np.array_equal(mapping[faces], faces))

    a_mesh = (
        np.bincount(tags, weights=tarea, minlength=ncell + 1)[1 : ncell + 1]
        if ntag <= ncell
        else None
    )

    res = AreaChange(
        ncell=ncell,
        nface=int(faces.size),
        vox=vox,
        cells=cells,
        npx=npx,
        identity=bool(identity),
        allow_mismatch=bool(allow_mismatch),
        mapping=mapping,
        face_area=f_total,
        face_matched=f_best,
        displaced=float(displaced),
        a_raster=a_ras,
        a_mesh=a_mesh,
        mesh_total=float(tarea.sum()),
        raster_total=float(a_ras.sum()),
    )
    if a_mesh is None or not (identity or allow_mismatch):
        return res

    with np.errstate(invalid="ignore", divide="ignore"):
        delta = 100.0 * (a_mesh - a_ras) / np.where(a_ras > 0, a_ras, np.nan)
        # Equivalent circular diameter converts area change to a length scale.
        d_ecd = 100.0 * (np.sqrt(a_mesh / np.where(a_ras > 0, a_ras, np.nan)) - 1.0)
    res.delta = delta
    res.delta_ecd = d_ecd
    res.area = summarise(delta, weights=a_ras)
    res.ecd = summarise(d_ecd, weights=a_ras)
    return res


def format_report(res: AreaChange):
    lines = []
    if res.nface != res.ncell:
        lines.append(
            f"WARNING: the raster has {res.ncell} cells but the mesh has "
            f"{res.nface} faces. Grains were lost or split in the "
            "reconstruction; the area comparison below is not one-to-one."
        )
    if not res.identity:
        tag = "WARNING" if res.allow_mismatch else "ERROR"
        m, tot, matched = res.mapping, res.face_area, res.face_matched
        bad = [
            (f, int(m[f]), matched[f] / tot[f] if tot[f] else np.nan)
            for f in range(1, len(m))
            if int(m[f]) not in (0, f) and tot[f] > 0
        ]
        lines.append(
            f"{tag}: mesh face k is not raster cell k. Faces sitting mostly on "
            "another cell (face -> cell, fraction of the face's area there): "
            + ", ".join(f"{f} -> {k} ({100 * q:.0f} %)" for f, k, q in bad[:10])
            + (" ..." if len(bad) > 10 else "")
        )
        lines.append(
            f"  {len(bad)} of {res.nface} faces. Every per-grain quantity "
            "downstream, theta included, is indexed by face id, so this has to "
            "be resolved before the mesh is used. A face whose fraction is "
            "barely over half is an ambiguous small grain rather than a "
            "mis-tagged one; --allow-mismatch carries on with the identity "
            "mapping and reports the areas anyway."
        )
        if not res.allow_mismatch:
            return lines
    if not res.comparable:
        return lines

    a, e = res.area, res.ecd
    lines.append(
        f"grain area change (smoothing + meshing, {a.n} grains): "
        f"min {a.min:+.2f} %, mean {a.mean:+.2f} %, "
        f"median {a.median:+.2f} %, max {a.max:+.2f} %"
    )
    lines.append(
        f"  |change|: mean {a.abs_mean:.2f} %, rms {a.rms:.2f} %; "
        f"area-weighted mean {a.area_weighted_mean:+.2f} %"
    )
    lines.append(
        f"  as a size (ECD ~ sqrt(area)): min {e.min:+.2f} %, "
        f"mean {e.mean:+.2f} %, median {e.median:+.2f} %, max {e.max:+.2f} %"
    )
    tot = 100.0 * (res.mesh_total - res.raster_total) / res.raster_total
    lines.append(
        f"  total meshed area {res.mesh_total:.6g} vs rastered "
        f"{res.raster_total:.6g} ({tot:+.3f} %); "
        f"displaced area {100 * res.displaced:.2f} % "
        "(meshed area lying over a different raster cell)"
    )
    return lines


def write_csv(path, res: AreaChange, log=print):
    with open(path, "w") as fh:
        fh.write(
            "cell_id,n_voxels,area_raster,area_mesh,delta_area_pct,delta_ecd_pct\n"
        )
        for k in range(len(res.a_raster)):
            fh.write(
                f"{k + 1},{int(res.npx[k])},{res.a_raster[k]:.8g},"
                f"{res.a_mesh[k]:.8g},{res.delta[k]:.6g},"
                f"{res.delta_ecd[k]:.6g}\n"
            )
    if log:
        log(f"  wrote {path}")
    return path


def write_png(path, res: AreaChange, unit="um", dpi=150, log=print):
    """The grains coloured by their area change, next to its distribution."""
    plt = use_agg()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15})

    cells, (vx, vy) = res.cells, res.vox
    ny, nx = cells.shape
    delta = res.delta
    painted = np.full(cells.shape, np.nan)
    m = cells > 0
    painted[m] = delta[cells[m] - 1]
    lim = max(np.nanpercentile(np.abs(delta), 98), 1e-3)

    aspect = ny * vy / (nx * vx)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 3.0 + 5.5 * aspect), layout="constrained"
    )

    ax = axes[0]
    im = ax.imshow(
        painted,
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
        origin="lower",
        extent=(0, nx * vx, 0, ny * vy),
    )
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("area change (%)")
    ax.set_title("mesh vs raster, per grain")
    ax.set_xlabel(f"blue = shrunk, red = grew; colour limit {lim:.2g} %")
    ax.set_xticks([])
    ax.set_yticks([])
    scale_bar_ax(ax, nx * vx, unit)

    ax = axes[1]
    a = res.area
    ax.hist(delta[np.isfinite(delta)], bins=40, color="0.35")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(
        a["median"],
        color="tab:blue",
        ls="--",
        lw=2,
        label=f"median {a.median:+.2f} %",
    )
    ax.axvline(a["mean"], color="tab:red", lw=2, label=f"mean {a.mean:+.2f} %")
    ax.set_xlabel("area change (%)")
    ax.set_ylabel("grains")
    ax.set_title(f"min {a.min:+.2f} %, max {a.max:+.2f} % over {a.n} grains")
    ax.legend(fontsize=12)

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    if log:
        log(f"  wrote {path}")
    return path


def measure(tesr, msh4, options: AreaReportOptions | None = None, log=print):
    """Measure, report, and optionally write the table and the picture.

    Returns the :class:`AreaChange`. Raises ValueError when a mesh face does not
    sit on the raster cell of the same id, unless ``options.allow_mismatch``:
    every per-grain quantity downstream is indexed by face id, so continuing
    past that would attach the areas -- and the transport driver's theta -- to
    the wrong grains.
    """
    opt = options or AreaReportOptions()
    res = area_change(tesr, msh4, allow_mismatch=opt.allow_mismatch)
    lines = format_report(res)
    if log:
        for line in lines:
            log("  " + line)
    if not res.comparable:
        raise ValueError("\n".join(lines))
    if opt.csv:
        write_csv(opt.csv, res, log=log)
    if opt.png:
        write_png(opt.png, res, unit=opt.unit, dpi=opt.dpi, log=log)
    return res


def edge_sides(seg, tri):
    """edge id -> set of face ids its segments belong to (1: surface, 2: interior)."""
    tri_of = {}
    for face, v in tri:
        for a, b in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            tri_of.setdefault(frozenset((a, b)), set()).add(face)
    sides = {}
    for edge, v in seg:
        sides.setdefault(edge, set()).update(tri_of.get(frozenset(v), ()))
    return sides


def overlay(tesr, msh4, output="check-mesh.png", dpi=150, unit="um", log=print):
    """Write the overlay PNG. Returns the output path.

    `tesr` and `msh4` are paths; everything else is cosmetic. The counts are
    printed through `log`, which can be set to None to silence them.
    """
    plt = use_agg()

    cells, vox = read_tesr(tesr)
    xyz, seg, tri = read_msh4(msh4)
    sides = edge_sides(seg, tri)
    faces = {t for t, _ in tri}
    if log and len(faces) != int(cells.max()):
        log(f"note: raster has {int(cells.max())} cells, mesh has {len(faces)} faces")

    ny, nx = cells.shape
    fig, ax = plt.subplots(figsize=(7, 7 * ny * vox[1] / (nx * vox[0])))
    draw_raster(ax, cells, vox)
    n_int = n_surf = 0
    for edge, v in seg:
        interior = len(sides[edge]) == 2
        n_int += interior
        n_surf += not interior
        a, b = xyz[v[0]], xyz[v[1]]
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="black" if interior else "0.55",
            lw=1.0 if interior else 0.7,
        )
    ax.set_title(
        f"{len(sides)} edges ({sum(len(s) == 2 for s in sides.values())} interior), "
        f"{n_int + n_surf} segments over {int(cells.max())} raster cells"
    )
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    scale_bar_ax(ax, nx * vox[0], unit)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    if log:
        log(f"  wrote {output}")
    return output
