"""Mesh EBSD rasters and rebuild the 2D GB topology required by FESTIM.

The pipeline imports CTF grains through UPXO/DefDAP, meshes with UPXO, and derives
edge connectivity, disorientation, lengths, and junctions from the mesh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

from festim_microstructure.ebsd.convert import convert
from festim_microstructure.ebsd.orientation import (
    cubic_disorientation_angle,
    qconj,
    qmul,
)
from festim_microstructure.ebsd.settings import Settings
from festim_microstructure.formats.ebsd import read_ebsd
from festim_microstructure.meshing.diagnostics import (
    AreaReportOptions,
    measure,
    overlay,
)
from festim_microstructure.meshing.upxo import UpxoMeshOptions, mesh_ebsd
from festim_microstructure.plotting import draw_raster, scale_bar_ax, use_agg

__all__ = [
    "EbsdMicrostructure",
    "EbsdOptions",
    "EdgeTable",
    "mesh_diagnostics",
    "read_extent",
    "run_ebsd_pipeline",
    "unit_name",
    "write_network_png",
]


UNIT_NAMES = {1e-9: "nm", 1e-6: "um", 1e-3: "mm", 1.0: "m"}


def unit_name(unit):
    """Display label for a metres-per-source-coordinate scale."""
    return UNIT_NAMES.get(unit, "source units")


@dataclass
class EbsdOptions:
    """Full CTF → UPXO/DefDAP grains → UPXO mesh → FESTIM checks."""

    ctf: str
    unit: float = 1e-6
    theta_min: float = 10.0
    import_settings: Settings | None = None
    mesh: UpxoMeshOptions | None = None
    stem: str = "poly"
    check_images: bool = True

    def __post_init__(self):
        if self.mesh is None:
            self.mesh = UpxoMeshOptions()
        if self.import_settings is None:
            self.import_settings = Settings(ctf=str(self.ctf))
        if Path(self.import_settings.ctf).resolve() != Path(self.ctf).resolve():
            raise ValueError(
                "ctf and import_settings.ctf must refer to the same source"
            )
        if not np.isfinite(self.unit) or self.unit <= 0:
            raise ValueError("unit must be finite and positive")
        if not np.isfinite(self.theta_min) or not 0 <= self.theta_min < 63:
            raise ValueError("theta_min must be between 0 and 63 degrees")
        if (
            not self.stem
            or Path(self.stem).name != self.stem
            or self.stem in (".", "..")
        ):
            raise ValueError("stem must be a filename without directory components")


def run_ebsd_pipeline(options, workdir: str | Path = "results", force=True):
    """Import, mesh, check source geometry/orientations and load FESTIM tags.

    Accepted meshes carry their native EBSD archive and SI metadata. Grain and
    boundary tags are checked through the actual DOLFINx importer, and the
    reconstructed FESTIM network is reported alongside geometry diagnostics.
    """
    options.__post_init__()
    if not isinstance(options.mesh, UpxoMeshOptions):
        raise TypeError("EbsdOptions.mesh must be UpxoMeshOptions")
    # Only one rank imports/publishes files; all ranks load the accepted mesh.
    # Propagate preparation failures before entering collective mesh import.
    comm = MPI.COMM_WORLD
    base, failure = None, None
    if comm.rank == 0:
        try:
            workdir = Path(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            imported = convert(
                settings=replace(
                    options.import_settings,
                    python=options.import_settings.python or options.mesh.python,
                    diagnostics=options.import_settings.diagnostics
                    or options.check_images,
                ),
                output=workdir / f"{options.stem}-import.npz",
                force=force,
            )
            base = mesh_ebsd(
                imported.archive,
                replace(
                    options.mesh,
                    python=options.mesh.python or options.import_settings.python,
                ),
                workdir=workdir,
                stem=options.stem,
                unit=options.unit,
                force=force,
            )
            mesh_diagnostics(
                base, unit_name(options.unit), check_images=options.check_images
            )
        except Exception as exc:
            failure = exc
    base, failure = comm.bcast((base, failure), root=0)
    if failure is not None:
        raise failure
    from festim_microstructure.formats.msh4 import read_mesh

    mesh, cells, facets = read_mesh(base, gdim=2)
    micro = EbsdMicrostructure.from_mesh(
        base, mesh, cells, facets, read_extent(base), theta_min=options.theta_min
    )
    micro.check_orientations()
    if mesh.comm.rank == 0:
        report = dict(
            n_grains=micro.n_grains,
            n_edges=micro.edges.n,
            interior_edges=int(micro.interior_mask.sum()),
            surface_edges=micro.surface_edges,
            triple_junctions=micro.triple_junctions,
            theta_min=options.theta_min,
            orientation_source=f"{base.name}-ebsd.npz",
            extent_m=list(read_extent(base)),
        )
        Path(f"{base}-festim.json").write_text(json.dumps(report, indent=2) + "\n")
    if options.check_images:
        write_network_png(
            base, mesh, micro, f"{base}-ebsd.npz", options.unit, unit_name(options.unit)
        )
    return base


def read_extent(base):
    """Domain width and height in metres from native mesh metadata."""
    return tuple(json.loads(Path(f"{base}-metadata.json").read_text())["extent_m"])


def mesh_diagnostics(base, unit_name="um", check_images=True):
    """Compare final SI mesh with imported labels, in source display units."""
    base = Path(base)
    archive, msh4 = f"{base}-ebsd.npz", f"{base}.msh4"
    scale = json.loads(Path(f"{base}-metadata.json").read_text())["unit"]
    if check_images:
        overlay(
            archive,
            msh4,
            output=f"{base}-check-mesh.png",
            unit=unit_name,
            mesh_unit=scale,
        )
    return measure(
        archive,
        msh4,
        AreaReportOptions(
            csv=f"{base}-areachange.csv",
            png=f"{base}-check-area.png" if check_images else None,
            unit=unit_name,
            mesh_unit=scale,
        ),
    )


def _ids(mask):
    """Boolean mask over entities in id order -> 1-based ids."""
    return np.flatnonzero(mask).astype(np.int32) + 1


def _allreduce(comm, arr, op):
    """Reduce an array over MPI ranks into a new array."""
    out = np.empty_like(arr)
    comm.Allreduce(np.ascontiguousarray(arr), out, op=op)
    return out


class EdgeTable:
    """Per-edge scalars in id order: ``values[k]`` belongs to edge ``k + 1``."""

    def __init__(self, values):
        """Store per-edge columns and their common row count."""
        self.values = values
        self.n = len(next(iter(values.values())))

    def __getitem__(self, key):
        """Return the per-edge column named by ``key``."""
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
            "face# set and the domain. Check the grain and boundary tags "
            "in the mesh."
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
    """Disorientations from the native archive associated with this mesh."""
    data = read_ebsd(f"{base}-ebsd.npz")
    if data.ncells != n_grain:
        raise RuntimeError(
            f"native EBSD archive has {data.ncells} grains "
            f"but mesh has {n_grain} face sets"
        )
    from festim_microstructure.ebsd.orientation import (
        quat_to_rodrigues,
        to_fundamental_zone,
    )

    q = data.grain_quats
    ori = quat_to_rodrigues(to_fundamental_zone(q))
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
    """Rebuild GB topology and disorientations from the tagged UPXO EBSD mesh.

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
        """Reconstruct topology from the pipeline's grain and boundary tags."""
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
        top.create_connectivity(0, tdim)
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
        """Mark edges shared by two grains rather than the exterior."""
        return self.edges["domtype"] < 0

    @property
    def network_mask(self):
        """Mark interior edges above the disorientation threshold."""
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
        """Check finite orientations and cubic boundary-angle bounds.

        Identity orientations and zero-angle boundaries can be physically valid;
        source readback checks and grain IDs establish their provenance.
        """
        theta = self.edges["theta"][self.interior_mask]
        if (
            not np.isfinite(self.ori).all()
            or not np.isfinite(theta).all()
            or np.any(theta < 0)
            or np.any(theta > 63)
        ):
            raise RuntimeError(
                "grain orientations or cubic disorientations are invalid"
            )
        return self.n_grains

    def report(self):
        """Summarize the EBSD mesh geometry and selected boundary network."""
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


def write_network_png(base, mesh, micro, archive_path, unit=1e-6, unit_name="um"):
    """Write ``<stem>-check-network.png`` with the boundaries FESTIM selects.

    The background matches the mesh overlay. Retained interior boundaries are
    coloured by disorientation, rejected boundaries are dashed white, and
    specimen-surface edges are grey. Compare with ``<stem>-check-mesh.png`` to
    inspect the disorientation filter and displacement from the source pixels.
    """
    if mesh.comm.size > 1:
        return
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    use_agg()
    base = Path(base)
    data = read_ebsd(archive_path)
    cells, vox = data.labels, data.vox
    fdim = mesh.topology.dim - 1
    owned = np.arange(mesh.topology.index_map(fdim).size_local, dtype=np.int32)
    owned = owned[micro.facet_edge[owned] > 0]
    nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, owned, False)
    segs = mesh.geometry.x[nodes][:, :, :2] / unit  # back to source units
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
    out = Path(f"{base}-check-network.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")
