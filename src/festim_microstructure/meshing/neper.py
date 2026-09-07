"""Neper as a mesh generator: tessellations, raster (EBSD) meshing, readers.

Neper is a command-line program -- there is no library to link against and no
bindings to write -- so the integration is ``subprocess.run`` plus a mesh
reader:

* ``neper -T`` writes a tessellation (``.tess``) and, with the ``-stat*``
  options, one line of data per tessellation vertex / edge / face. Those files
  are the lookup tables that turn mesh tags back into geometry and physics
  (:func:`run_neper`, :class:`StatFile`, :class:`NeperMicrostructure`).
* ``neper -M`` meshes it. The mesh conforms to every grain boundary by
  construction, so ``occ.fragment`` and the clipping of the Voronoi ridges both
  disappear, and so does the Distance/Threshold background field: ``-rclface``
  sets the element size on the faces and ``-rcl`` the size in the cells.
* ``neper -M map.tesr`` meshes an EBSD raster directly (2D only), reconstructing
  the boundaries as it goes (:func:`mesh_tesr`).
* The element sets come out of ``gmshio`` as ``facet_tags`` whose value is the
  tessellation face (3D) or edge (2D) id, so the network is picked up from tags
  exactly as before -- no geometric locator, no midpoint test
  (:func:`read_mesh`).

Three things the scipy construction could not give at all:

* ``domface``, which separates true grain boundaries from the tessellation
  faces lying on the walls of the specimen (without it the network includes the
  free surface and hydrogen short circuits around the outside);
* ``theta``, the disorientation angle between the two grains, which is what
  actually decides whether a boundary is a fast path;
* exact topology of the triple lines and quadruple points, instead of the
  Counter/union-find reconstruction from rounded vertex coordinates.

Binaries
--------
Neper does not have to live in the same conda environment as FESTIM -- it only
has to be a path. Keeping it in its own environment avoids letting the solver
rearrange a working dolfinx install over a dependency (GSL, scotch) that has
nothing to do with FESTIM. :func:`find_binary` resolves each program from, in
order, an explicit argument, an environment variable (``FM_NEPER_BIN``,
``FM_GMSH_BIN``, ``FM_POVRAY_BIN``) and ``PATH``. ``GMSH_BIN`` is the
*executable* that Neper calls for meshing: the conda-forge package providing
it is ``gmsh``; ``python-gmsh`` is only the bindings, so a dolfinx environment
may have the API without the command.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._binaries import find_binary

# dolfinx and mpi4py are imported inside read_mesh, so that running Neper and
# reading its stat files stays possible without them.

__all__ = [
    "EDGE_KEYS",
    "FACE_KEYS",
    "VER_KEYS",
    "NeperMicrostructure",
    "NeperOptions",
    "StatFile",
    "TesrMeshOptions",
    "find_binary",
    "mesh_tesr",
    "read_mesh",
    "run_interruptible",
    "run_neper",
]

# The stat keys are all scalars, one column each, one line per entity in id
# order. Keep these tuples and the -stat* arguments in run_neper in step.
FACE_KEYS = ("domface", "theta", "area", "zmin", "zmax")
EDGE_KEYS = ("domtype", "facenb", "length")
VER_KEYS = ("domtype", "edgenb")


# ------------------------------------------------------------------ binaries


def run_interruptible(cmd, cwd=None, env=None):
    """Run a Neper command, surviving a Ctrl+C long enough for it to finish.

    ``cwd`` is used to keep every path Neper sees short and relative -- see the
    note in :func:`run_neper` about whitespace in directory names.

    Neper treats SIGINT as "stop optimizing, keep the current solution and
    write the output". But Ctrl+C goes to the whole foreground process group,
    so the Python parent gets it too -- and if the parent exits immediately,
    the child is killed part-way through writing and you are left with partial
    files, or none. Catching it here and waiting lets Neper land its output;
    a second Ctrl+C still gets you out.
    """
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


# ----------------------------------------------------- tessellation + mesh (3D)


@dataclass
class NeperOptions:
    """Everything ``run_neper`` passes to ``neper -T`` and ``neper -M``.

    Meshing has two competing costs. The element *count* is dominated by
    ``rcl``, since the faces are 2D and the interiors 3D. But the meshing
    *time* is driven by the ratio ``rcl / rclface``: a steep size jump means
    Netgen grades hard, fails the quality target more often, and multimeshing
    retries. Neper meshes each face at ``rclface`` and then fills each
    polyhedron from that boundary mesh at ``rcl``, grading between the two, so
    refinement near the boundaries is preserved however coarse the interior
    gets.
    """

    rcl: float = 0.8
    """Element size in the grain interiors, relative to average cell size."""
    rclface: float = 0.2
    """Element size on the grain boundaries."""
    rcledge: float | None = None
    """Element size on the triple lines; None = same as the faces."""
    pl: float = 2.5
    """Progression factor: max length ratio between adjacent 1D elements."""
    mesh_qual_min: float | None = 0.7
    """Multimeshing retries each face and polyhedron with several algorithms
    until this is reached, so quality target and meshing time trade off
    directly; 0.9 is Neper's default and 0.7 is reasonable while iterating."""
    mesh_max_time: float | None = None
    """Seconds per face/polyhedron; the default is 1000 s, long enough for one
    pathological cell to stall a run without saying so. Try 30 when
    diagnosing."""

    regularize: bool = True
    """``-reg 1``: removes the small edges and faces that otherwise produce
    unusable tets at the triple lines -- which is where this whole problem
    lives. It makes internal faces slightly non-planar, which is the reason to
    let Neper mesh rather than rebuilding the geometry with OCC."""
    reg_rsel: float | None = 0.8
    """``-rsel``, the small-edge length used by regularization. Neper's default
    is 1, picked to suit the *default* ``-rcl``; it should track whatever
    ``rcl`` you actually use."""
    periodicity: str | None = None
    """e.g. ``"x,y"``. MUTUALLY EXCLUSIVE with ``regularize``: Neper rejects
    ``-reg 1`` on a periodic tessellation. To have both, set ``rsel`` and leave
    ``regularize`` off, so the small edges are suppressed at the tessellation
    stage instead."""
    rsel: float | None = None
    """Small-edge control when periodic; try ``rcl`` as a first value."""

    morpho: str = "voronoi"
    """``voronoi`` is a Poisson-Voronoi tessellation: uniform random seeds, no
    optimization stage, effectively instant. ``gg`` is the grain-growth
    morphology (lognormal equivalent diameter and sphericity); more realistic,
    but fitting it can run for tens of thousands of iterations."""
    morpho_stop: str | None = "eps<1e-6||val<1e-3||iter>=20000"
    """Only used when ``morpho`` involves an optimization. Neper's default
    stopping rule has an unreachable second clause, so the run grinds on a
    plateau; a value or iteration cap makes the cost bounded."""

    face_keys: tuple = FACE_KEYS
    edge_keys: tuple = EDGE_KEYS
    ver_keys: tuple = VER_KEYS

    def validate(self):
        if self.regularize and self.periodicity:
            raise ValueError(
                "Neper does not allow -regularization on a periodic tessellation. "
                "Either drop periodicity (recommended: element quality at the "
                "triple lines matters more here than the x/y wall artefacts), or "
                "set regularize=False and give rsel a value so the small edges "
                "are suppressed at the tessellation stage instead."
            )


def run_neper(
    n,
    seed=1,
    stem="poly",
    workdir="results",
    options=None,
    neper_bin=None,
    gmsh_bin=None,
    force=False,
):
    """Generate and mesh a polycrystal. Returns the base path (no extension).

    ``-T`` and ``-M`` are two calls rather than one so that the ``.tess`` is a
    persistent object: the face ids the ``.stface`` lines refer to are the same
    ids ``-M`` turns into 2D element sets. Keeping the file makes the numbering
    auditable, and ``-T`` -- the expensive half when the morphology needs
    optimization -- is a pure function of ``(n, seed, morpho)``, so it is cached
    on its own: a failure in ``-M`` should not cost it again.

    Neper re-parses its input-file argument -- it is a structured field
    supporting comma-separated files and colon-separated transformations -- and
    splits it on whitespace, so an absolute path through a directory like
    "mwes + examples" arrives as several unusable fragments. Running with cwd set
    to the output directory and passing bare names sidesteps it entirely.
    """
    opt = options or NeperOptions()
    opt.validate()
    base = (Path(workdir) / stem).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    print(f"neper outputs -> {base.parent}")
    if base.with_suffix(".msh4").exists() and not force:
        print(f"  reusing {base.name}.msh4")
        return base

    neper_bin = find_binary("neper", neper_bin, "FM_NEPER_BIN")
    gmsh_bin = find_binary("gmsh", gmsh_bin, "FM_GMSH_BIN")

    morpho = opt.morpho
    if opt.rsel is not None:
        # `rsel` avoids small edges like -reg does, but during tessellation, so
        # it is compatible with periodicity. It combines with the `gg` alias;
        # combining it with the `voronoi` keyword may not be accepted.
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
    # Gmsh scratch: Neper writes one .geo per tessellation face and polyhedron,
    # meshes it, reads the .msh back and deletes the pair. Giving them their own
    # directory keeps results/ readable and makes leftovers safe to delete.
    tmp = base.parent / "tmp"
    tmp.mkdir(exist_ok=True)
    if not base.with_suffix(".tess").exists() or force:
        run_interruptible(tess, cwd=wd)
    else:
        print(f"  reusing {base.name}.tess")
    # -order 1: the default is 2. -format msh4: Gmsh v4; the default `msh` is
    # Neper's own dialect of Gmsh 2.2 with extra sections generic readers do
    # not expect.
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
    )
    leftovers = list(tmp.glob("*"))
    if leftovers:
        # -M succeeded, so anything still here is from an earlier failed run
        for f in leftovers:
            f.unlink()
        print(f"  cleared {len(leftovers)} stale gmsh scratch files")
    return base


class StatFile:
    """One Neper .st* file: scalar keys in columns, entities in id order.

    Ids are 1-based throughout Neper, so ``self.values[k]`` is entity ``k + 1``
    and :meth:`ids` converts a boolean mask back into ids.
    """

    def __init__(self, path, keys):
        raw = np.loadtxt(path, ndmin=2)
        if raw.shape[1] != len(keys):
            raise ValueError(
                f"{path} has {raw.shape[1]} columns but {len(keys)} keys were "
                f"expected ({', '.join(keys)}); the -stat option and the key "
                "tuple have drifted apart"
            )
        self.values = {k: raw[:, i] for i, k in enumerate(keys)}
        self.n = raw.shape[0]

    def __getitem__(self, key):
        return self.values[key]

    @staticmethod
    def ids(mask):
        return np.flatnonzero(mask).astype(np.int32) + 1


class NeperMicrostructure:
    """The tessellation's own description of itself, read back from the stats.

    Every number here is Neper's, computed on the exact topology rather than
    reconstructed from the geometry.

    Args:
        base: output of :func:`run_neper` (path without extension).
        theta_min: keep only boundaries above this disorientation (degrees).
        options: the :class:`NeperOptions` the stats were written with (for the
            key tuples).
    """

    def __init__(self, base, theta_min=0.0, options=None):
        opt = options or NeperOptions()
        self.base = Path(base)
        self.theta_min = theta_min
        self.faces = StatFile(str(base) + ".stface", opt.face_keys)
        self.edges = StatFile(str(base) + ".stedge", opt.edge_keys)
        self.vertices = StatFile(str(base) + ".stver", opt.ver_keys)

    # `domface` is the id of the domain face a tessellation face lies on, and
    # -1 when it lies on none. A face can only meet the boundary by lying on a
    # domain face, so domface < 0 is exactly "interior", i.e. a real grain
    # boundary rather than a piece of free surface.
    @property
    def interior_mask(self):
        return self.faces["domface"] < 0

    @property
    def network_mask(self):
        return self.interior_mask & (self.faces["theta"] > self.theta_min)

    @property
    def network_face_ids(self):
        return StatFile.ids(self.network_mask)

    network_entity_ids = network_face_ids

    @property
    def theta(self):
        """Disorientation of every face, degrees, in id order."""
        return self.faces["theta"]

    @property
    def network_area(self):
        return float(self.faces["area"][self.network_mask].sum())

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
        m = (self.edges["domtype"] < 0) & (self.edges["facenb"] >= 3)
        return int(m.sum()), float(self.edges["length"][m].sum())

    @property
    def quadruple_points(self):
        m = (self.vertices["domtype"] < 0) & (self.vertices["edgenb"] >= 4)
        return int(m.sum())

    def report(self, n_cells=None):
        n_tl, len_tl = self.triple_lines
        kept, total = int(self.network_mask.sum()), int(self.interior_mask.sum())
        theta = self.faces["theta"][self.network_mask]
        head = f"{n_cells} cells, " if n_cells is not None else ""
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
            f"  boundary area                   : {self.network_area:.4f}",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------- raster (EBSD, 2D)


@dataclass
class TesrMeshOptions:
    """Everything :func:`mesh_tesr` passes to Neper for a raster input."""

    crysym: str = "cubic"
    orides: str = "rodrigues:passive"
    """Orientation descriptor; must match how the tesr stores them."""
    tesr_transform: str | None = None
    """Optional Neper transformation chain applied to the input raster. Empty by
    default: the converter is expected to have done the cropping and cleanup,
    and skipping this avoids Neper's tesr write path."""
    tesr_smooth: str = "laplacian"
    """Interface smoothing before meshing. The reconstructed boundaries are
    pixel staircases; Laplacian smoothing rounds them off. ``"none"`` keeps the
    staircase."""
    tesr_smooth_fact: float = 0.5
    tesr_smooth_iter: int = 5
    rcl: float = 0.25
    """Relative characteristic length. Only ``-rcl`` acts on a raster input:
    Neper derives the edge and vertex lengths from the face value and never
    consults ``-rcledge`` / ``-rclver``."""
    mesh_qual_min: float | None = 0.7
    mesh_max_time: float | None = None
    extra_stat_keys: tuple = field(default_factory=tuple)


def _need(path, force):
    if force or not Path(path).is_file():
        return True
    print(f"  reusing {Path(path).name}")
    return False


def mesh_tesr(
    tesr,
    stem="poly",
    workdir="results",
    options=None,
    neper_bin=None,
    gmsh_bin=None,
    force=False,
):
    """A single EBSD map -> triangular mesh conforming to the raster's own
    grain boundaries. Returns the base path (no extension).

    ``neper -M map.tesr`` meshes the raster directly, which is supported in 2D
    only. The msh4 carries the reconstructed topology as physical groups
    ``ver#``, ``edge#``, ``face#``, and face k is raster cell k, so:

    * grain boundary: 1D element set ``edge#``, touching two ``face#`` sets;
    * specimen surface: 1D element set touching one face;
    * theta: disorientation of the two grains' orientations
      (``<stem>-grainori.txt``), computed by the Python side;
    * triple junction: mesh vertex where 3+ distinct edge ids meet.

    Outputs ``<stem>.msh4`` (Gmsh v4, linear triangles, all dimensions),
    ``<stem>.sttesr`` (raster geometry: ``dim, rastersizex, rastersizey,
    voxsizex, voxsizey``, in the raster's unit) and ``<stem>-grainori.txt``
    (one orientation per grain). Everything runs in the raster's unit; the
    caller converts the mesh to metres after reading it.

    This replaces the former ``ebsd_to_mesh.sh``; each stage is cached on its
    output file unless ``force``.
    """
    opt = options or TesrMeshOptions()
    base = (Path(workdir) / stem).resolve()
    wd = base.parent
    wd.mkdir(parents=True, exist_ok=True)
    tesr = Path(tesr).resolve()
    if not tesr.is_file():
        raise FileNotFoundError(
            f"no EBSD map .tesr file at {tesr}. The map must be written as a "
            "raster tessellation first (festim_microstructure.meshing.ebsd.ctf)."
        )
    if any(c.isspace() for c in str(tesr)):
        raise ValueError(
            f"the path {str(tesr)!r} contains whitespace. Neper's input-file "
            "argument is a structured field, so a path with whitespace arrives "
            "as several unusable fragments."
        )
    neper_bin = find_binary("neper", neper_bin, "FM_NEPER_BIN")
    gmsh_bin = find_binary("gmsh", gmsh_bin, "FM_GMSH_BIN")
    # the equivalent of activating neper's environment for the child processes
    # only: neper -M spawns gmsh by name unless given a path, and other helpers
    # are looked up on PATH
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(Path(neper_bin).parent), str(Path(gmsh_bin).parent), env.get("PATH", "")]
    )
    tmp = wd / "tmp"
    tmp.mkdir(exist_ok=True)

    # 0. stage the raster. By default nothing is done to it: the converter
    #    already crops, fills holes and numbers cells from 1 with the origin at
    #    (0,0), and its grains are connected components so `rmsat` has nothing
    #    to remove.
    raw = wd / f"{stem}-raw.tesr"
    if _need(raw, force):
        if opt.tesr_transform:
            print(f"  transforming: {opt.tesr_transform}")
            run_interruptible(
                [
                    neper_bin,
                    "-T",
                    "-loadtesr",
                    str(tesr),
                    "-transform",
                    opt.tesr_transform,
                    "-o",
                    f"{stem}-raw",
                ],
                cwd=str(wd),
                env=env,
            )
            if "**oridata" in raw.read_text(errors="replace"):
                print(
                    f"  WARNING: {raw.name} was written by neper -T and contains "
                    "**oridata. Neper 5.0.0 may not be able to read it back; if "
                    "the next command stalls, regenerate the input without "
                    "per-voxel orientations."
                )
        else:
            shutil.copyfile(tesr, raw)

    # geometry of the cleaned raster, one line, columns in the order given
    run_interruptible(
        [
            neper_bin,
            "-T",
            "-loadtesr",
            raw.name,
            "-stattesr",
            "dim,rastersizex,rastersizey,voxsizex,voxsizey",
            "-o",
            stem,
        ],
        cwd=str(wd),
        env=env,
    )
    dim, lx, ly, vsx, vsy = np.loadtxt(base.with_suffix(".sttesr"), ndmin=2)[0]
    print(f"  raster: dim={int(dim)}  extent={lx:g} x {ly:g}  pixel={vsx:g} x {vsy:g}")
    if int(dim) != 2:
        raise RuntimeError(
            f"this pipeline expects a 2D EBSD map, got a {int(dim)}D tesr. To take "
            "a single slice out of a 3D map, crop it to one voxel along z and "
            "apply the '2d' transform: neper -T -loadtesr map.tesr -transform "
            "'crop(cube(...,zmin,zmin+voxsizez)),2d' -o slice"
        )

    # per-grain orientations. For a raster tessellation the orientation key is
    # the descriptor itself (`rodrigues`, `euler-bunge`, ...) -- `ori` is a
    # simulation result key and is not valid here.
    ori = wd / f"{stem}-grainori.txt"
    if _need(ori, force):
        run_interruptible(
            [
                neper_bin,
                "-T",
                "-loadtesr",
                raw.name,
                "-oridescriptor",
                opt.orides,
                "-statcell",
                opt.orides.split(":")[0],
                "-o",
                f"{stem}-grainori",
            ],
            cwd=str(wd),
            env=env,
        )
        (wd / f"{stem}-grainori.stcell").replace(ori)
    n_cells = sum(1 for _ in open(ori))
    print(f"  grains: {n_cells}")

    # 1. mesh the raster. Gmsh v4 because FESTIM reads it with
    #    dolfinx.io.gmshio and needs the 1D element sets, which carry the
    #    reconstructed edge ids. -tmp must exist beforehand.
    msh = base.with_suffix(".msh4")
    if _need(msh, force):
        run_interruptible(
            [
                neper_bin,
                "-M",
                raw.name,
                "-gmsh",
                gmsh_bin,
                "-dim",
                "all",
                "-order",
                "1",
                "-elttype",
                "tri",
                "-rcl",
                str(opt.rcl),
                "-tesrsmooth",
                opt.tesr_smooth,
                "-tesrsmoothfact",
                str(opt.tesr_smooth_fact),
                "-tesrsmoothitermax",
                str(opt.tesr_smooth_iter),
                *(
                    ["-meshqualmin", str(opt.mesh_qual_min)]
                    if opt.mesh_qual_min
                    else []
                ),
                *(
                    ["-mesh2dmaxtime", str(opt.mesh_max_time)]
                    if opt.mesh_max_time
                    else []
                ),
                "-tmp",
                str(tmp),
                "-format",
                "msh4",
                "-statmesh",
                "nodenb,eltnb",
                "-o",
                stem,
            ],
            cwd=str(wd),
            env=env,
        )
    try:
        tmp.rmdir()
    except OSError:
        print(f"  note: {tmp} is not empty (stale gmsh scratch)")
    print(f"ok: {msh.name}  ({n_cells} grains, domain {lx:g} x {ly:g})")
    return base


# -------------------------------------------------------------------- reader


def read_mesh(base, gdim, unit=1.0, comm=None, rank=0):
    """Read a Neper ``.msh4`` into dolfinx.

    ``cell_tags`` carry the polyhedron / raster-cell (grain) id and
    ``facet_tags`` the tessellation face (3D) or edge (2D) id, because Neper
    writes every tessellation entity as an element set and the facet physical
    ids run independently of the cell ones.

    ``unit`` is metres per mesh length unit; the geometry is scaled by it
    before any locator, submesh or dof coordinate is derived from it (an EBSD
    raster is meshed in microns so that Gmsh's absolute tolerances are
    exercised at O(1-100) rather than O(1e-6)).
    """
    from mpi4py import MPI

    from dolfinx.io import gmsh as gmshio

    comm = MPI.COMM_WORLD if comm is None else comm
    result = gmshio.read_from_msh(str(base) + ".msh4", comm, rank, gdim=gdim)
    if hasattr(result, "mesh"):
        mesh, cell_tags, facet_tags = result.mesh, result.cell_tags, result.facet_tags
    else:
        mesh, cell_tags, facet_tags = result[0], result[1], result[2]
    if facet_tags is None or facet_tags.values.size == 0:
        raise RuntimeError(
            "no facet tags were read: the facet element sets did not survive "
            "the msh4 round trip (for a raster mesh, check that -dim all reached "
            "neper -M)"
        )
    if unit != 1.0:
        mesh.geometry.x[:] *= unit
    return mesh, cell_tags, facet_tags
