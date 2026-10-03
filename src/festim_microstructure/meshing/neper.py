"""Run Neper tessellation/meshing workflows and read their topology data.

Supports generated polycrystals. :class:`NeperMesh` is the
whole 3D path in one object: it generates the tessellation, meshes it, reads the
mesh, and interprets the topology statistics read by formats.msh4.
"""

import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .._binaries import find_binary, subprocess_env
from ..formats.msh4 import StatFile, read_mesh

# Keep DOLFINx imports local so Neper/stat tools remain lightweight.

__all__ = [
    "EDGE_KEYS",
    "FACE_KEYS",
    "VER_KEYS",
    "NeperMesh",
    "NeperSettings",
    "find_binary",
    "run_interruptible",
]

# Stat columns are scalar, one per entity, in id order.
FACE_KEYS = ("domface", "theta", "area", "zmin", "zmax")
EDGE_KEYS = ("domtype", "facenb", "length")
VER_KEYS = ("domtype", "edgenb")

# Binaries.


@dataclass
class _Outputs:
    """Where a generated Neper polycrystal writes, and which binaries it uses.

    Attributes:
        stem (str): Basename Neper is told to write, and so the name of every
            output file.
        workdir (str | Path): Directory those files go in; created if missing.
        neper_bin (str | None): Path to the neper binary; ``None`` looks at
            ``FM_NEPER_BIN`` and then ``PATH``.
        gmsh_bin (str | None): The same for gmsh.
        force (bool): Regenerate even when the outputs are already there. Each
            stage is otherwise cached on its own output file, so an interrupted
            run resumes rather than restarting.
    """

    stem: str = "poly"
    workdir: str | Path = "results"
    neper_bin: str | None = None
    gmsh_bin: str | None = None
    force: bool = False

    @property
    def base(self) -> Path:
        """The extension-free output path, with its directory created."""
        base = (Path(self.workdir) / self.stem).resolve()
        base.parent.mkdir(parents=True, exist_ok=True)
        return base

    def binaries(self):
        """Resolve ``(neper, gmsh)`` and the environment their children need."""
        neper = find_binary("neper", self.neper_bin, "FM_NEPER_BIN")
        gmsh = find_binary("gmsh", self.gmsh_bin, "FM_GMSH_BIN")
        return neper, gmsh, subprocess_env(neper, gmsh)


def run_interruptible(cmd, cwd=None, env=None):
    """Run Neper; let its first Ctrl+C finish writing the current output."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        print(
            "\ninterrupted: waiting for neper to write its output "
            "(Ctrl+C again to abandon it)"
        )
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            raise
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


# 3D tessellation and mesh.


@dataclass
class NeperSettings(_Outputs):
    """Everything :class:`NeperMesh` needs beyond a cell count and a seed.

    Mostly options forwarded to ``neper -T`` and ``neper -M``: ``rclface``
    controls GB refinement and ``rcl`` the cell interior. ``theta_min`` is the
    exception, read back out of the statistics after the run rather than passed
    to Neper. Output paths and binaries come from :class:`_Outputs`.

    Attributes:
        theta_min (float): Keep only boundaries whose disorientation exceeds
            this, in degrees. Read after meshing, so changing it costs nothing:
            the mesh still carries every face, and only which of them count as
            the network changes.
        rcl (float): Element size in the grain interiors, relative to average cell size.
        rclface (float): Element size on the grain boundaries.
        rcledge (float | None): Element size on the triple lines; None = same as the
            faces.
        pl (float): Progression factor: max length ratio between adjacent 1D elements.
        mesh_qual_min (float | None): Multimeshing retries each face and polyhedron with
            several algorithms until this is reached, so quality target and meshing time
            trade off directly; 0.9 is Neper's default and 0.7 is reasonable while
            iterating.
        mesh_max_time (float | None): Seconds per face/polyhedron; the default is 1000
            s, long enough for one pathological cell to stall a run without saying so.
            Try 30 when diagnosing.
        regularize (bool): ``-reg 1``: removes the small edges and faces that otherwise
            produce unusable tets at the triple lines -- which is where this whole
            problem lives. It makes internal faces slightly non-planar, which is the
            reason to let Neper mesh rather than rebuilding the geometry with OCC.
        reg_rsel (float | None): ``-rsel``, the small-edge length used by
            regularization. Neper's default is 1, picked to suit the *default* ``-rcl``;
            it should track whatever ``rcl`` you actually use.
        periodicity (str | None): e.g. ``"x,y"``. MUTUALLY EXCLUSIVE with
            ``regularize``: Neper rejects ``-reg 1`` on a periodic tessellation. To have
            both, set ``rsel`` and leave ``regularize`` off, so the small edges are
            suppressed at the tessellation stage instead.
        rsel (float | None): Small-edge control when periodic; try ``rcl`` as a first
            value.
        morpho (str): ``voronoi`` is a Poisson-Voronoi tessellation: uniform random
            seeds, no optimization stage, effectively instant. ``gg`` is the
            grain-growth morphology (lognormal equivalent diameter and sphericity); more
            realistic, but fitting it can run for tens of thousands of iterations.
        morpho_stop (str | None): Only used when ``morpho`` involves an optimization.
            Neper's default stopping rule has an unreachable second clause, so the run
            grinds on a plateau; a value or iteration cap makes the cost bounded.
        face_keys (tuple): Columns requested in the face statistics output.
        edge_keys (tuple): Columns requested in the edge statistics output.
        ver_keys (tuple): Columns requested in the vertex statistics output.
    """

    theta_min: float = 0.0

    rcl: float = 0.8
    rclface: float = 0.2
    rcledge: float | None = None
    pl: float = 2.5
    mesh_qual_min: float | None = 0.7
    mesh_max_time: float | None = None

    regularize: bool = True
    reg_rsel: float | None = 0.8
    periodicity: str | None = None
    rsel: float | None = None

    morpho: str = "voronoi"
    morpho_stop: str | None = "eps<1e-6||val<1e-3||iter>=20000"

    face_keys: tuple = FACE_KEYS
    edge_keys: tuple = EDGE_KEYS
    ver_keys: tuple = VER_KEYS

    def validate(self):
        """Reject incompatible Neper regularization and periodicity settings."""
        if self.regularize and self.periodicity:
            raise ValueError(
                "Neper does not allow -regularization on a periodic tessellation. "
                "Either drop periodicity (recommended: element quality at the "
                "triple lines matters more here than the x/y wall artefacts), or "
                "set regularize=False and give rsel a value so the small edges "
                "are suppressed at the tessellation stage instead."
            )


def _generate(n, seed, opt):
    """Generate and mesh a polycrystal. Returns the base path (no extension).

    ``-T`` and ``-M`` are two calls rather than one so that the ``.tess`` is a
    persistent object: the face ids the ``.stface`` lines refer to are the same
    ids ``-M`` turns into 2D element sets. Keeping the file makes the numbering
    auditable, and ``-T`` -- the expensive half when the morphology needs
    optimization -- is a pure function of ``(n, seed, morpho)``, so it is cached
    on its own: a failure in ``-M`` should not cost it again.
    """
    stem, force = opt.stem, opt.force
    base = opt.base
    print(f"neper outputs -> {base.parent}")
    if base.with_suffix(".msh4").exists() and not force:
        print(f"  reusing {base.name}.msh4")
        return base

    neper_bin, gmsh_bin, env = opt.binaries()

    morpho = opt.morpho
    if opt.rsel is not None:
        # ``rsel`` removes short edges while preserving periodicity.
        morpho = f"{morpho},rsel:{opt.rsel}"

    tess = [neper_bin, "-T", "-n", str(n), "-id", str(seed), "-morpho", morpho]
    if opt.morpho != "voronoi" and opt.morpho_stop:
        tess += ["-morphooptistop", opt.morpho_stop]
    if opt.regularize:
        tess += ["-reg", "1"]
        if opt.reg_rsel is not None:
            tess += ["-rsel", str(opt.reg_rsel)]
    if opt.periodicity:
        tess += ["-periodicity", opt.periodicity]
    tess += [
        "-statface",
        ",".join(opt.face_keys),
        "-statedge",
        ",".join(opt.edge_keys),
        "-statver",
        ",".join(opt.ver_keys),
        "-o",
        stem,
    ]
    wd = str(base.parent)
    # Isolate Neper's per-face Gmsh scratch files.
    tmp = base.parent / "tmp"
    tmp.mkdir(exist_ok=True)
    if not base.with_suffix(".tess").exists() or force:
        run_interruptible(tess, cwd=wd, env=env)
    else:
        print(f"  reusing {base.name}.tess")
    # FESTIM needs first-order Gmsh v4 element sets.
    run_interruptible(
        [
            neper_bin,
            "-M",
            stem + ".tess",
            "-gmsh",
            gmsh_bin,
            "-order",
            "1",
            "-elttype",
            "tet",
            "-rcl",
            str(opt.rcl),
            "-rclface",
            str(opt.rclface),
            *(["-rcledge", str(opt.rcledge)] if opt.rcledge is not None else []),
            "-pl",
            str(opt.pl),
            *(["-meshqualmin", str(opt.mesh_qual_min)] if opt.mesh_qual_min else []),
            *(
                [
                    "-mesh2dmaxtime",
                    str(opt.mesh_max_time),
                    "-mesh3dmaxtime",
                    str(opt.mesh_max_time),
                ]
                if opt.mesh_max_time
                else []
            ),
            "-tmp",
            str(tmp),
            "-format",
            "msh4",
            "-o",
            stem,
        ],
        cwd=wd,
        env=env,
    )
    leftovers = list(tmp.glob("*"))
    if leftovers:
        # A successful run leaves only stale files from prior failures.
        for f in leftovers:
            f.unlink()
        print(f"  cleared {len(leftovers)} stale gmsh scratch files")
    return base


class NeperMesh:
    """A Neper polycrystal: generated, meshed, read back, and described.

    Constructing one runs ``neper -T`` and ``neper -M`` -- or reuses whatever an
    earlier run left in ``settings.workdir`` -- and then reads the tessellation's
    own statistics. It unpacks into what a model is built from::

        mesh, cell_tags, facet_tags, network_ids = fm.NeperMesh(100, 1)

    ``network_ids`` is a list of face ids rather than one marker because Neper
    tags every tessellation face separately; pass it to
    :class:`~festim_microstructure.microstructure.TaggedPolycrystal` as
    ``gb_tag``.

    Every number reported here is Neper's, computed on the exact topology rather
    than reconstructed from the geometry. Implements
    :class:`~festim_microstructure.microstructure.BoundaryNetwork`.

    The DOLFINx mesh is read on first use rather than in the constructor, so a
    tessellation can be generated and inspected -- disorientations, junction
    counts, boundary area -- on a machine with no solver stack installed.
    Reading is collective, and stays so: every rank runs the same statements.

    :meth:`from_base` builds one from files an earlier run wrote, without
    invoking Neper at all.
    """

    def __init__(self, n_cells=None, seed=1, settings=None, *, base=None):
        """Generate or load a Neper tessellation and its statistics.

        Args:
            n_cells: Number of cells to generate; omitted when loading ``base``.
            seed: Random seed used for generation.
            settings: Neper options and statistic column names.
            base: Existing extension-free output path to load instead.
        """
        self.settings = settings or NeperSettings()
        self.n_cells = n_cells
        self.seed = seed
        if base is None:
            if n_cells is None:
                raise TypeError(
                    "NeperMesh(n_cells, seed, settings) generates a tessellation "
                    "and NeperMesh.from_base(base) reads one an earlier run left "
                    "on disk; this call gave neither a cell count nor a base path"
                )
            self.settings.validate()
            base = _generate(n_cells, seed, self.settings)
        self.base = Path(base)
        self.faces = StatFile(f"{self.base}.stface", self.settings.face_keys)
        self.edges = StatFile(f"{self.base}.stedge", self.settings.edge_keys)
        self.vertices = StatFile(f"{self.base}.stver", self.settings.ver_keys)

    @classmethod
    def from_base(cls, base, settings=None):
        """Read the three stat files an earlier run wrote, without running Neper.

        Args:
            base: extension-free output path.
            settings: the :class:`NeperSettings` the stats were written with, for
                the key tuples and ``theta_min``.
        """
        return cls(base=base, settings=settings)

    def __iter__(self):
        """``mesh, cell_tags, facet_tags, network_ids``, in that order.

        The four things a model is built from, so that one statement takes the
        caller from a cell count to a microstructure.
        """
        yield self.mesh
        yield self.cell_tags
        yield self.facet_tags
        yield self.network_ids

    # The DOLFINx half, read once on demand. Three properties over one cached
    # read, so that touching any of them does not re-read the other two.
    @cached_property
    def _read(self):
        """Read and cache the 3D mesh and its cell and facet tags."""
        return read_mesh(self.base, gdim=3)

    @property
    def mesh(self):
        """``dolfinx.mesh.Mesh`` of the tessellation."""
        return self._read[0]

    @property
    def cell_tags(self):
        """Polyhedron (grain) id of every cell."""
        return self._read[1]

    @property
    def facet_tags(self):
        """Tessellation face id of every tagged facet."""
        return self._read[2]

    @property
    def theta_min(self):
        """Read from the settings, so there is one place to change it."""
        return self.settings.theta_min

    # `domface` is the id of the domain face a tessellation face lies on, and
    # -1 when it lies on none. A face can only meet the boundary by lying on a
    # domain face, so domface < 0 is exactly "interior", i.e. a real grain
    # boundary rather than a piece of free surface.
    @property
    def interior_mask(self):
        """Mark faces lying inside the domain rather than on its surface."""
        return self.faces["domface"] < 0

    @property
    def network_mask(self):
        """Mark interior faces above the disorientation threshold."""
        return self.interior_mask & (self.faces["theta"] > self.theta_min)

    @property
    def network_ids(self):
        """1-based ids of the faces in the network."""
        return StatFile.ids(self.network_mask)

    #: Dimension-specific spellings, kept so existing callers keep working.
    network_face_ids = network_ids
    network_entity_ids = network_ids

    @property
    def theta(self):
        """Disorientation of every face, degrees, in id order."""
        return self.faces["theta"]

    @property
    def network_measure(self):
        """Total area of the boundaries in the network."""
        return float(self.faces["area"][self.network_mask].sum())

    #: The 3D spelling of :attr:`network_measure`.
    network_area = network_measure

    def junction_only_below(self, z_top, tol=1e-9):
        """Deepest point reached by a boundary that touches the charged face.

        Below this depth no boundary is fed directly, so whatever the network
        holds there has crossed at least one triple line.
        """
        touching = self.network_mask & (self.faces["zmax"] > z_top - tol)
        if not touching.any():
            return z_top
        return float(self.faces["zmin"][touching].min())

    # For edges and vertices, `domtype` is 0/1/2 when the entity sits on a
    # domain vertex / edge / face, so a negative value means it is interior.
    # A triple line is an interior edge shared by three or more faces; a
    # quadruple point is an interior vertex meeting four or more edges.
    @property
    def triple_lines(self):
        """Return the count and total length of interior triple lines."""
        m = (self.edges["domtype"] < 0) & (self.edges["facenb"] >= 3)
        return int(m.sum()), float(self.edges["length"][m].sum())

    @property
    def quadruple_points(self):
        """Count interior vertices meeting at least four edges."""
        m = (self.vertices["domtype"] < 0) & (self.vertices["edgenb"] >= 4)
        return int(m.sum())

    def report(self):
        """Summarize the 3D tessellation and selected boundary network."""
        n_tl, len_tl = self.triple_lines
        kept, total = int(self.network_mask.sum()), int(self.interior_mask.sum())
        theta = self.faces["theta"][self.network_mask]
        head = f"{self.n_cells} cells, " if self.n_cells is not None else ""
        lines = [
            f"microstructure: {head}{self.faces.n} faces",
            f"  grain boundaries (interior)     : {total}",
            f"  kept above {self.theta_min:g} deg          : {kept}",
        ]
        if kept:
            lines.append(
                f"  disorientation                  : "
                f"{theta.min():.1f} - {theta.max():.1f} deg (mean {theta.mean():.1f})"
            )
        lines += [
            f"  triple lines                    : {n_tl} (total length {len_tl:.3f})",
            f"  quadruple points                : {self.quadruple_points}",
            f"  boundary area                   : {self.network_measure:.4f}",
        ]
        return "\n".join(lines)
