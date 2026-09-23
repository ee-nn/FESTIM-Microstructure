"""Mesh EBSD rasters and rebuild the 2D GB topology required by FESTIM.

The pipeline converts ``.tesr`` to ``.msh4``, then derives edge connectivity,
disorientation, lengths, and junctions from Neper's element sets.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

from festim_microstructure.ebsd.orientation import (
    cubic_disorientation_angle,
    qconj,
    qmul,
    rodrigues_to_quat,
)
from festim_microstructure.formats.tesr import read_tesr
from festim_microstructure.meshing.diagnostics import (
    AreaReportOptions,
    measure,
    overlay,
)
from festim_microstructure.meshing.neper import (
    TesrMeshOptions,
    mesh_tesr,
    run_interruptible,
)
from festim_microstructure.plotting import draw_raster, scale_bar_ax, use_agg

__all__ = [
    "EbsdMicrostructure",
    "EbsdOptions",
    "EdgeTable",
    "cleanup_unscaled_files",
    "mesh_diagnostics",
    "read_extent",
    "run_ebsd_pipeline",
    "unit_name",
    "write_network_png",
]


UNIT_NAMES = {1e-9: "nm", 1e-6: "um", 1e-3: "mm", 1.0: "m"}


def unit_name(unit):
    return UNIT_NAMES.get(unit, "tesr units")


@dataclass
class EbsdOptions:
    """Inputs for EBSD raster meshing."""

    tesr: str
    """Raster tessellation; convert source EBSD data first."""
    unit: float = 1e-6
    """Metres per TESR unit; the final mesh and extent are saved in SI units."""
    theta_min: float = 10.0
    """Minimum GB disorientation in degrees."""
    mesh: TesrMeshOptions | None = None
    """Neper meshing options; defaults to :class:`TesrMeshOptions`."""
    stem: str = "poly"
    check_images: bool = True

    def __post_init__(self):
        if self.mesh is None:
            self.mesh = TesrMeshOptions()


def run_ebsd_pipeline(
    options, workdir: str | Path = "results", neper_bin=None, gmsh_bin=None, force=True
):
    """Write an SI mesh and metadata; return the extension-free output path.

    Meshing and raster diagnostics run in the input units under ``workdir``,
    with an ``-unscaled`` stem to distinguish them from the SI outputs.
    The final mesh is always exported from that raw mesh, so cached runs do
    not compound the conversion to metres.
    """
    if not np.isfinite(options.unit) or options.unit <= 0:
        raise ValueError("unit must be a finite, positive metres-per-TESR-unit factor")
    meshing = replace(
        options.mesh,
        stem=f"{options.stem}-unscaled",
        workdir=Path(workdir),
        neper_bin=neper_bin,
        gmsh_bin=gmsh_bin,
        force=force,
    )
    raw_base = mesh_tesr(options.tesr, meshing)
    base = raw_base.parent / options.stem
    mesh_diagnostics(
        raw_base,
        unit_name(options.unit),
        check_images=options.check_images,
        report_base=base,
    )
    _, gmsh, env = meshing.binaries()
    run_interruptible(
        [
            gmsh,
            str(raw_base.with_suffix(".msh4")),
            "-save",
            "-setnumber",
            "Mesh.ScalingFactor",
            str(options.unit),
            "-format",
            "msh41",
            "-o",
            str(base.with_suffix(".msh4")),
        ],
        env=env,
    )
    stats = np.loadtxt(str(raw_base) + ".sttesr", ndmin=2)
    stats[:, 1:] *= options.unit
    np.savetxt(str(base) + ".sttesr", stats)
    shutil.copyfile(str(raw_base) + "-grainori.txt", str(base) + "-grainori.txt")
    return base


def read_extent(base):
    """``(LX, LY)`` in metres, from the pipeline's SI ``.sttesr`` file."""
    cols = np.loadtxt(str(base) + ".sttesr", ndmin=2)[0]
    return float(cols[1]), float(cols[2])


def cleanup_unscaled_files(base):
    """Delete the temporary raster-unit files used to produce an SI mesh.

    Call this only after diagnostics that need ``<stem>-unscaled-raw.tesr``
    have been written. The final ``.msh4``, SI metadata, and diagnostics do not
    use these files.

    Returns the paths that were removed.
    """
    base = Path(base)
    prefix = f"{base.name}-unscaled"
    removed = []
    for path in base.parent.glob(f"{prefix}*"):
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def mesh_diagnostics(base, unit_name="um", check_images=True, report_base=None):
    """Write mesh-overlay and grain-area diagnostics.

    The area check also verifies that mesh face ids still match raster cell ids.
    ``report_base`` controls the CSV name independently of the unscaled mesh.
    """
    base = Path(base)
    report_base = base if report_base is None else Path(report_base)
    work = base.parent
    tesr, msh4 = f"{base}-raw.tesr", f"{base}.msh4"

    if check_images:
        overlay(tesr, msh4, output=str(work / "check-mesh.png"), unit=unit_name)
    measure(
        tesr,
        msh4,
        AreaReportOptions(
            csv=f"{report_base}-areachange.csv",
            png=str(work / "check-area.png") if check_images else None,
            unit=unit_name,
        ),
    )


def _ids(mask):
    """Boolean mask over entities in id order -> 1-based ids."""
    return np.flatnonzero(mask).astype(np.int32) + 1


def _allreduce(comm, arr, op):
    out = np.empty_like(arr)
    comm.Allreduce(np.ascontiguousarray(arr), out, op=op)
    return out


class EdgeTable:
    """Per-edge scalars in id order: ``values[k]`` belongs to edge ``k + 1``."""

    def __init__(self, values):
        self.values = values
        self.n = len(next(iter(values.values())))

    def __getitem__(self, key):
        return self.values[key]


def _edge_grain_pairs(comm, facet_edge, cell_grain, f2c, n_edge):
    """Which grains sit either side of each ``edge#`` set.

    Gathered across ranks: an edge can be cut by the partition, so no single
    rank sees both of its grains. Returns ``(pair, sides, domtype)`` in edge-id
    order, where ``domtype < 0`` marks an interior boundary.
    """
    local = {}
    for f in np.flatnonzero(facet_edge):
        local.setdefault(int(facet_edge[f]), set()).update(
            int(cell_grain[c]) for c in f2c.links(f)
        )
    grains = [set() for _ in range(n_edge + 1)]
    for part in comm.allgather({k: sorted(v) for k, v in local.items()}):
        for k, v in part.items():
            grains[k].update(v)
    sides = np.array([len(g) for g in grains[1:]])
    if sides.min() < 1 or sides.max() > 2 or 0 in set().union(*grains[1:]):
        bad = _ids((sides < 1) | (sides > 2))
        raise RuntimeError(
            f"edges {bad[:10]} touch {sides[bad[:10] - 1]} grains; every "
            "edge# set must lie between two face# sets or between one "
            "face# set and the domain. The msh4 is not a neper -M raster "
            "mesh, or the cell tags did not survive the read."
        )
    pair = np.array(
        [sorted(g) + [0] * (2 - len(g)) for g in grains[1:]], dtype=np.int32
    )
    return pair, sides, np.where(sides == 2, -1.0, 1.0)


def _edge_extents(comm, mesh, facet_edge, fdim, fmap, n_edge):
    """Length and y-range of every edge set, summed/reduced over owned facets."""
    owned = np.arange(fmap.size_local, dtype=np.int32)
    owned = owned[facet_edge[owned] > 0]
    nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, owned, False)
    x = mesh.geometry.x[nodes]  # (nf, 2, 3)
    e = facet_edge[owned]
    length = np.bincount(
        e, weights=np.linalg.norm(x[:, 1] - x[:, 0], axis=1), minlength=n_edge + 1
    )
    ymin = np.full(n_edge + 1, np.inf)
    ymax = np.full(n_edge + 1, -np.inf)
    np.minimum.at(ymin, e, x[:, :, 1].min(axis=1))
    np.maximum.at(ymax, e, x[:, :, 1].max(axis=1))
    return (
        _allreduce(comm, length[1:], MPI.SUM),
        _allreduce(comm, ymin[1:], MPI.MIN),
        _allreduce(comm, ymax[1:], MPI.MAX),
    )


def _edge_theta(base, n_grain, n_edge, pair, sides):
    """Disorientation per edge from the grain orientations Neper wrote.

    Line k of ``-grainori.txt`` is face k, so the file and the ``face#`` sets
    must describe the same raster; that is checked rather than assumed.
    """
    ori = np.loadtxt(str(base) + "-grainori.txt", ndmin=2)
    if ori.shape[0] != n_grain:
        raise RuntimeError(
            f"{base}-grainori.txt has {ori.shape[0]} lines but the mesh "
            f"has {n_grain} face# sets; they must be the same raster"
        )
    q = rodrigues_to_quat(ori)
    theta = np.zeros(n_edge)
    inner = sides == 2
    a, b = pair[inner, 0] - 1, pair[inner, 1] - 1
    theta[inner] = cubic_disorientation_angle(qmul(qconj(q[a]), q[b]))
    return ori, theta


def _count_triple_junctions(comm, mesh, facet_edge, v2f, vmap, extent):
    """Owned vertices where 3+ distinct edge ids meet, off the box boundary."""
    n_v = vmap.size_local
    verts = np.arange(n_v, dtype=np.int32)
    xv = dolfinx.mesh.compute_midpoints(mesh, 0, verts)
    lx, ly = extent
    tol = 1e-9 * max(lx, ly)
    on_box = (
        (np.abs(xv[:, 0]) < tol)
        | (np.abs(xv[:, 0] - lx) < tol)
        | (np.abs(xv[:, 1]) < tol)
        | (np.abs(xv[:, 1] - ly) < tol)
    )
    edgenb = np.fromiter(
        (len(set(facet_edge[v2f.links(v)].tolist()) - {0}) for v in verts),
        dtype=int,
        count=n_v,
    )
    return comm.allreduce(int(((edgenb >= 3) & ~on_box).sum()), op=MPI.SUM)


@dataclass
class EbsdMicrostructure:
    """Rebuild GB topology and disorientations from Neper's EBSD mesh.

    Mesh ``edge#`` sets provide boundaries and ``face#`` sets retain raster cell
    ids. Junction counts are exact in serial and lower bounds in parallel.
    Implements :class:`~festim_microstructure.microstructure.BoundaryNetwork`.

    Build it with :meth:`from_mesh`; the constructor only stores the arrays that
    reconstruction produced, so the class can be built in a test without a mesh,
    a communicator or an orientation file.
    """

    base: Path
    extent: tuple
    n_grains: int
    ori: np.ndarray  #: (n_grain, 3) Rodrigues, one per raster cell
    facet_edge: np.ndarray  #: edge id per local facet, 0 off the network
    edges: EdgeTable
    triple_junctions: int
    surface_edges: int
    theta_min: float = 10.0

    @classmethod
    def from_mesh(
        cls, base, mesh, cell_tags, facet_tags, extent, theta_min=10.0, crysym="cubic"
    ):
        """Reconstruct the topology from a mesh ``neper -M`` wrote."""
        if crysym != "cubic":
            raise NotImplementedError(
                "theta is computed with the closed-form cubic disorientation "
                f"from orientation.py; crysym = {crysym!r} needs a general "
                "symmetry-operator search"
            )
        comm = mesh.comm
        top = mesh.topology
        tdim = top.dim
        fdim = tdim - 1
        top.create_connectivity(fdim, tdim)
        top.create_connectivity(0, fdim)
        f2c = top.connectivity(fdim, tdim)
        v2f = top.connectivity(0, fdim)
        fmap, cmap, vmap = top.index_map(fdim), top.index_map(tdim), top.index_map(0)

        facet_edge = np.zeros(fmap.size_local + fmap.num_ghosts, dtype=np.int32)
        facet_edge[facet_tags.indices] = facet_tags.values
        cell_grain = np.zeros(cmap.size_local + cmap.num_ghosts, dtype=np.int32)
        cell_grain[cell_tags.indices] = cell_tags.values
        n_edge = comm.allreduce(int(facet_edge.max(initial=0)), op=MPI.MAX)
        n_grain = comm.allreduce(int(cell_grain.max(initial=0)), op=MPI.MAX)
        if n_edge == 0 or n_grain == 0:
            raise RuntimeError("the mesh carries no edge# / face# element sets")

        pair, sides, domtype = _edge_grain_pairs(
            comm, facet_edge, cell_grain, f2c, n_edge
        )
        length, ymin, ymax = _edge_extents(comm, mesh, facet_edge, fdim, fmap, n_edge)
        ori, theta = _edge_theta(base, n_grain, n_edge, pair, sides)

        return cls(
            base=Path(base),
            extent=extent,
            n_grains=n_grain,
            ori=ori,
            facet_edge=facet_edge,
            edges=EdgeTable(
                {
                    "domtype": domtype,
                    "theta": theta,
                    "length": length,
                    "ymin": ymin,
                    "ymax": ymax,
                    "grain_a": pair[:, 0].astype(float),
                    "grain_b": pair[:, 1].astype(float),
                }
            ),
            triple_junctions=_count_triple_junctions(
                comm, mesh, facet_edge, v2f, vmap, extent
            ),
            surface_edges=int((sides == 1).sum()),
            theta_min=theta_min,
        )

    @property
    def interior_mask(self):
        return self.edges["domtype"] < 0

    @property
    def network_mask(self):
        return self.interior_mask & (self.edges["theta"] > self.theta_min)

    @property
    def network_ids(self):
        """1-based ids of the edges in the network."""
        return _ids(self.network_mask)

    #: Dimension-specific spellings, kept so existing callers keep working.
    network_edge_ids = network_ids
    network_entity_ids = network_ids

    @property
    def theta(self):
        """Disorientation of every edge, degrees, in id order."""
        return self.edges["theta"]

    @property
    def network_measure(self):
        """Total length of the boundaries in the network."""
        return float(self.edges["length"][self.network_mask].sum())

    #: The 2D spelling of :attr:`network_measure`.
    network_length = network_measure

    def junction_only_below(self, y_top, tol=1e-12):
        """Deepest point reached by a boundary that touches the charged edge.

        Below this depth no boundary is fed directly, so whatever the network
        holds there has crossed at least one triple junction.
        """
        touching = self.network_mask & (self.edges["ymax"] > y_top - tol)
        if not touching.any():
            return y_top
        return float(self.edges["ymin"][touching].min())

    def check_orientations(self):
        """Fail loudly if theta is not a real disorientation distribution.

        With theta computed here rather than by Neper the failure modes move:
        an all-zero orientation file, or one that does not belong to this
        raster, are what would make every boundary look alike.
        """
        theta = self.edges["theta"][self.interior_mask]
        problems = []
        if np.allclose(self.ori, 0.0):
            problems.append("every grain orientation in the tesr readout is zero")
        if theta.size and np.allclose(theta, 0.0):
            problems.append("every interior edge has theta = 0")
        if theta.size and theta.max() > 63.0:
            # the maximum disorientation is ~62.8 deg for cubic symmetry
            problems.append(
                f"max theta = {theta.max():.1f} deg exceeds the cubic bound"
            )
        if problems:
            raise RuntimeError(
                "the orientations are not usable: "
                + "; ".join(problems)
                + ". Check that -grainori.txt was written from the same tesr "
                "that was meshed."
            )
        return self.n_grains

    def report(self):
        kept, total = int(self.network_mask.sum()), int(self.interior_mask.sum())
        theta = self.edges["theta"][self.network_mask]
        lx, ly = self.extent
        lines = [
            f"microstructure: {self.n_grains} grains from {self.base.name} "
            "(2D, raster mesh)",
            f"  domain                          : {lx:g} x {ly:g}",
            f"  edges                           : {self.edges.n}",
            f"  on the specimen surface         : {self.surface_edges}",
            f"  grain boundaries (interior)     : {total}",
            f"  kept above {self.theta_min:g} deg    : {kept}",
        ]
        if kept:
            lines.append(
                f"  disorientation                  : "
                f"{theta.min():.1f} - {theta.max():.1f} deg (mean {theta.mean():.1f})"
            )
        lines += [
            f"  triple junctions                : {self.triple_junctions}",
            f"  boundary length                 : {self.network_measure:.4g}",
        ]
        return "\n".join(lines)


def write_network_png(base, mesh, micro, tesr_path, unit=1e-6, unit_name="um"):
    """check-network.png: the raster with the boundaries as FESTIM will use them.

    Same background as check-mesh.png (meshing.diagnostics), but the edges are the
    driver's: those above ``micro.theta_min`` coloured by theta, interior edges below it
    dashed white, specimen-surface edges grey. Compare with check-mesh.png to
    see what the disorientation filter removed, and with check-grains.png to
    see how far interface smoothing moved the boundaries off the pixels.
    """
    if mesh.comm.size > 1:
        return
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    use_agg()
    base = Path(base)
    cells, vox = read_tesr(tesr_path)
    fdim = mesh.topology.dim - 1
    owned = np.arange(mesh.topology.index_map(fdim).size_local, dtype=np.int32)
    owned = owned[micro.facet_edge[owned] > 0]
    nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, owned, False)
    segs = mesh.geometry.x[nodes][:, :, :2] / unit  # back to tesr units
    e = micro.facet_edge[owned] - 1
    surface = micro.edges["domtype"][e] > 0
    kept = micro.network_mask[e]
    dropped = ~surface & ~kept

    ny, nx = cells.shape
    fig, ax = plt.subplots(figsize=(8, 8 * ny * vox[1] / (nx * vox[0])))
    draw_raster(ax, cells, vox)
    ax.add_collection(LineCollection(segs[surface], colors="0.6", lw=0.7))
    ax.add_collection(
        LineCollection(segs[dropped], colors="white", lw=1.3, linestyles="--")
    )
    net = LineCollection(segs[kept], cmap="inferno", lw=1.8)
    net.set_array(micro.edges["theta"][e[kept]])
    net.set_clim(micro.theta_min, 62.8)
    ax.add_collection(net)
    fig.colorbar(net, ax=ax, fraction=0.046).set_label("theta (deg)")
    n_kept, n_int = int(micro.network_mask.sum()), int(micro.interior_mask.sum())
    ax.set_title(
        f"network used by FESTIM: {n_kept} of {n_int} boundaries above "
        f"{micro.theta_min:g} deg\n(dashed white = dropped, grey = specimen surface)"
    )
    ax.set_xlabel(f"x ({unit_name})")
    ax.set_ylabel(f"y ({unit_name})")
    scale_bar_ax(ax, nx * vox[0], unit_name)
    fig.tight_layout()
    out = base.parent / "check-network.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")
