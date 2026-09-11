"""Reproduce Li et al. (2022), Front. Mater. 9:935129, Fig. 4: the effective diffusion
coefficient of H in polycrystalline W with a *volumetric* grain-boundary
phase, on Voronoi microstructures.

Their model (Sect. 2.2, Eqs. 16-17, 34) is the steady diffusion equation with
a piecewise-constant diffusivity: D_m in the grains, D_GB in a grain-boundary
band of finite width, one continuous concentration field, no trapping and no
interface kinetics. The effective diffusivity is the ratio of the average flux
to the average gradient (Eq. 21). In FESTIM that is a HydrogenTransportProblem
with two volume subdomains carrying two materials, solved with
transient=False. This file reproduces their Fig. 4 panels A, B (the two
microstructures, drawn as isometric cubes) and D, E, G, H (D_eff/D_m against
grain-boundary density for D_GB/D_m = 100, 10, 0.2 and 0.1).

WHY THE BOUNDARY IS A VOLUME AND NOT A MANIFOLD
-----------------------------------------------
Panels G and H put D_eff/D_m well below 1 and falling with boundary density.
A codim-1 (zero-thickness) boundary cannot do that: it adds conductance in
parallel with the lattice and removes no lattice volume, so at D_GB/D_m = 0.1
it would give Col_I(Z) = 1 + 0.1 f, not the 1 - 0.9 f of their Hart line
(Eq. 28), which is exact for phases in parallel. The reduction is a volumetric
effect of a slow band occupying 30-70 % of the volume. The boundary here is
therefore a band of width w around every Voronoi face, assigned D_GB, on a
structured voxel mesh whose cells are classified by their midpoint's distance
to the nearest face. For D_GB/D_m > 1 a codim-1 network on the same Voronoi
skeleton is the w -> 0 limit of this and would agree with panels D and E.

THE MICROSTRUCTURES
-------------------
  "iso":  a 3D Voronoi foam from seeds placed uniformly at random within the
          cells of a cubic lattice (stratified: a narrower size distribution
          than Poisson seeds, like phase-field grain growth), periodic through
          seed images, in a cube of side B. Their Fig. 4A. The jitter must be
          the full half-cell: with less, the Voronoi faces stay near the
          lattice mid-planes, which are also the box faces, and those faces
          (the no-flux sides and the rendered ones) carry far more band than
          the volume fraction -- 47 % against 28 % at a quarter-cell jitter.
          At the full half-cell the face fractions match the volume fraction,
          as Delesse's principle requires of a representative section.
  "col":  a 2D Voronoi tessellation in the x-y plane from a jittered square
          lattice, extruded along z into prismatic columns. Their Fig. 4B.
          Col_I(Z) is the flux along the columns, where the phases are in
          parallel and Hart's Eq. 28 is exact whatever the cross-section --
          it is solved on a two-cell-thick slab and serves as the code check.
          Col_I(X) is the flux across the columns, a 2D problem.

The box is their 96 nm cube on their 1 nm grid (Sect. 2.3), with about six
grains per edge as in their Fig. 4A and five columns per edge as in Fig. 4B.
The Dirichlet faces carry the paper's 0.4 and 0.1 (here in m^-3 x 1e24) and
the lateral faces are no-flux, as a representative volume; the Voronoi is
periodic, the lateral condition is not, and 216 grains are enough for the
edge to be forgotten.

HOW THICK THE BANDS ARE
-----------------------
The band-to-grain ratio is fixed by the volume fraction: f_GB ~ S_v w with
S_v ~ 2.8/g for a 3D foam and 2.0/g for columns (g the seed spacing), so
f_GB = 0.3 means w/g ~ 0.11 and f_GB = 0.5 means w/g ~ 0.18. Their Fig. 4A
shows bands ~0.12-0.15 of the grain size, i.e. f_GB ~ 0.35-0.4, and the
same ratio here looks thin only when the box holds as many grains as theirs.
The renders (panels A-B) carry nm axes so the sizes can be read off; the
band drawn is the one of index RENDER_INDEX in the sweep, with its f_GB in
the title. Their blue is the diffuse phase-field interface (a 4 nm tanh
profile, green halo included); the sharp band here is the eta <= 0.9 phase
that their diffusion calculation actually uses.

THE X-AXIS, AND THE nm^-1
-------------------------
Their "grain boundary density" is the boundary area per unit volume, S_v,
which has units of 1/length, and in their phase-field grid the boundary band
(eta <= 0.9) is one grid spacing wide, dx = 1 nm. The band's volume fraction
is f_GB = S_v x dx, so the number on their axis IS the volume fraction when
dx = 1 nm. Check: their HS curve (Eq. 33) at "0.3 nm^-1" with D_GB/D_m = 100
reads ~23, and Eq. 33 with f_GB = 0.3 gives 23.1. D_eff/D_m depends on the
morphology only through f_GB (the problem is scale-free), so this file keeps
the seeds fixed and sweeps the band width w, which is the same family of
structures as sweeping the grain size at fixed w = 1 nm. The measured f_GB is
plotted; S_v = f_GB/w is reported alongside.

DIMENSIONS
----------
Everything is SI. D_m = 5.13e-8 exp(-0.21 eV/kT) m^2/s is the bulk value
they cite (Liu et al. 2014, Sect. 2.3); the band carries the same activation
energy scaled by the ratio, so D_GB/D_m is temperature-independent as in
their sweep. T = 1073 K (their Sect. 3.4) affects nothing plotted.

Coordinates are of order 1e-9 m. Every locator therefore uses an explicit
tolerance scaled to the mesh: numpy's isclose default atol = 1e-8 is larger
than the whole domain and marks every boundary point as "on the inlet".

MESH, SOLVER AND EXTRACTION
---------------------------
Structured tetrahedral meshes, N cells per side; each cell is grain or band
by its midpoint. D_eff is taken from the outlet flux, J_out L/(A dC), and
cross-checked against the volume average of their Eq. 21 and the inlet flux.
The 3D foam has ~5e6 tetrahedra and ~9e5 unknowns at their resolution and is
solved with CG + GAMG (PETSC_OPTIONS_3D); the 2D and slab problems use
FESTIM's default direct solver. ISO_N can be lowered for a quick look, but
below ~64 the thinnest band is under two cells wide.

OUTPUTS
-------
li2022-fig4ab.png   isometric cubes of the two microstructures, panels A-B
li2022-fig4.png     D_eff/D_m against f_GB for the four ratios, panels D,E,G,H,
                    with Hart (Eq. 28) and HS (Eq. 33)
"""

from dataclasses import dataclass
from functools import cached_property

from mpi4py import MPI

import dolfinx
import festim as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ufl
from scipy.spatial import cKDTree

# physics (SI)
# ----------------------------------------------------------------------------
T = 1073.0  # K; only scales D_m (their Sect. 3.4)
D0_M = 5.13e-8  # m^2/s, Liu et al. 2014, cited in their Sect. 2.3
E_M = 0.21  # eV
RATIOS = [100.0, 10.0, 0.2, 0.1]  # D_GB/D_m of their panels D, E, G, H
C_IN = 0.4e24  # m^-3, their 0.4 and 0.1 (arbitrary units)
C_OUT = 0.1e24

# microstructures
# ----------------------------------------------------------------------------
B = 96e-9  # m, cube side: their 3D box (Sect. 2.3)
DX = 1e-9  # m, their grid spacing (Sect. 2.3)
SEED = 0
JITTER = 0.5  # seed displacement, fraction of g: 0.5 = uniform within its lattice cell

ISO_N = round(B / DX)  # 96 cells per side, their grid
ISO_SEEDS = (
    6  # seeds per side -> 216 grains, g = 16 nm, ~6 per edge as in their Fig. 4A
)
# band width / grain spacing; f_GB ~ 0.3-0.7
ISO_W_OVER_G = [0.11, 0.15, 0.20, 0.26, 0.33]

COL_N = round(B / DX)  # 96 cells per side in x-y
COL_SEEDS = (
    5  # seeds per side -> 25 columns, g = 19 nm, ~5 per edge as in their Fig. 4B
)
COL_W_OVER_G = [0.16, 0.22, 0.29, 0.37, 0.46]  # f_GB ~ 0.3-0.7

RENDER_INDEX = 0  # which band width of the sweep is drawn in panels A-B (0 = thinnest)

STRUCTURES = ["col_z", "col_x", "iso"]  # their Col_I(Z), Col_I(X), Iso
LABEL = {"col_z": "Col_I(Z)", "col_x": "Col_I(X)", "iso": "Iso"}
K_NEIGHBOURS = 16  # seeds consulted for the distance to the nearest face

PETSC_OPTIONS_3D = {
    "ksp_type": "cg",
    "pc_type": "gamg",
    "ksp_rtol": 1e-12,
    "ksp_atol": 0.0,
    "ksp_max_it": 2000,
}


def D_m(T):
    return D0_M * np.exp(-E_M / (F.k_B * T))


# ----------------------------------------------------------------------------
# Voronoi foams as point classifiers
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Foam:
    """A periodic Voronoi tessellation of a cube (dim=3) or of the x-y plane
    extruded along z (dim=2), from seeds on a jittered lattice."""

    dim: int
    n_seeds: int
    B: float = B
    seed: int = SEED

    @property
    def g(self):
        """Lattice spacing of the seeds, the grain size."""
        return self.B / self.n_seeds

    @cached_property
    def seeds(self):
        rng = np.random.default_rng(self.seed + self.dim)
        axes = np.arange(self.n_seeds)
        grid = np.stack(np.meshgrid(*[axes] * self.dim, indexing="ij"), -1).reshape(
            -1, self.dim
        )
        pts = (grid + 0.5) * self.g
        pts += rng.uniform(-JITTER, JITTER, size=pts.shape) * self.g
        return pts

    @cached_property
    def _tree(self):
        shifts = np.stack(
            np.meshgrid(*[[-1, 0, 1]] * self.dim, indexing="ij"), -1
        ).reshape(-1, self.dim)
        images = (self.seeds[None, :, :] + shifts[:, None, :] * self.B).reshape(
            -1, self.dim
        )
        return cKDTree(images)

    def face_distance(self, points):
        """Distance from each point (dim x n) to the nearest Voronoi face: the
        minimum over other seeds j of the distance to the bisector plane of
        the nearest seed i and j, which is exactly the distance to the
        boundary of i's cell since the cell is the intersection of those
        half-spaces."""
        p = np.ascontiguousarray(points[: self.dim].T)
        k = min(K_NEIGHBOURS, self._tree.n)
        d, idx = self._tree.query(p, k=k)
        s = self._tree.data[idx]  # (n, k, dim)
        s_i = s[:, :1, :]
        sep = np.linalg.norm(s[:, 1:, :] - s_i, axis=-1)
        dist = (d[:, 1:] ** 2 - d[:, :1] ** 2) / (2.0 * sep)
        return dist.min(axis=1)

    def is_grain(self, points, w):
        return self.face_distance(points) > 0.5 * w


# ----------------------------------------------------------------------------
# one simulation
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    structure: str
    foam: Foam
    w: float

    @property
    def dim(self):
        return 2 if self.structure == "col_x" else 3

    @property
    def axis(self):
        return 0 if self.structure == "col_x" else 2

    @property
    def n(self):
        return ISO_N if self.structure == "iso" else COL_N

    @property
    def h(self):
        return self.foam.B / self.n

    @property
    def extent(self):
        b = self.foam.B
        return {"col_x": (b, b), "col_z": (b, b, 2 * self.h), "iso": (b, b, b)}[
            self.structure
        ]

    @property
    def n_cells(self):
        n = self.n
        return {"col_x": (n, n), "col_z": (n, n, 2), "iso": (n, n, n)}[self.structure]

    @property
    def length(self):
        return self.extent[self.axis]

    @property
    def cross_section(self):
        ext = list(self.extent)
        del ext[self.axis]
        return float(np.prod(ext))

    @property
    def tol(self):
        """Locator tolerance: a small fraction of the cell size, never numpy's
        default (see the header)."""
        return 1e-3 * self.h


class PhaseRegion(F.VolumeSubdomain):
    """A volume subdomain given by an explicit list of cells (classified by
    midpoint), instead of FESTIM's all-vertices predicate."""

    def __init__(self, id, material, cells):
        super().__init__(id=id, material=material)
        self._cells = np.asarray(cells, dtype=np.int32)

    def locate_subdomain_entities(self, mesh):
        return self._cells


def make_mesh(case, comm):
    lo = np.zeros(case.dim)
    hi = np.array(case.extent)
    if case.dim == 2:
        return dolfinx.mesh.create_rectangle(
            comm, [lo, hi], list(case.n_cells), cell_type=dolfinx.mesh.CellType.triangle
        )
    return dolfinx.mesh.create_box(
        comm, [lo, hi], list(case.n_cells), cell_type=dolfinx.mesh.CellType.tetrahedron
    )


def classify_cells(mesh, case):
    """Grain and band cell indices (local + ghost) by midpoint, and f_GB."""
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n_all = imap.size_local + imap.num_ghosts
    mid = dolfinx.mesh.compute_midpoints(mesh, tdim, np.arange(n_all, dtype=np.int32))
    grain = case.foam.is_grain(mid.T, case.w)
    cells_grain = np.flatnonzero(grain).astype(np.int32)
    cells_gb = np.flatnonzero(~grain).astype(np.int32)
    # cells of a structured box mesh have equal volume, so counts are fractions
    n_local = imap.size_local
    n_gb_local = int(np.sum(~grain[:n_local]))
    tot = mesh.comm.allreduce(n_local, op=MPI.SUM)
    gb = mesh.comm.allreduce(n_gb_local, op=MPI.SUM)
    return cells_grain, cells_gb, gb / tot


def _face_measure(mesh, axis, value, tol):
    fdim = mesh.topology.dim - 1
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.abs(x[axis] - value) < tol
    )
    tags = dolfinx.mesh.meshtags(
        mesh, fdim, np.sort(facets), np.ones(facets.size, dtype=np.int32)
    )
    return ufl.Measure("ds", domain=mesh, subdomain_data=tags)(1)


def _dg0_dofs(V0, cells):
    """The DG0 dof of each cell (not assumed equal to the cell index)."""
    dm = V0.dofmap
    try:
        return np.asarray(dm.list)[cells, 0]
    except (AttributeError, TypeError):
        return np.array([dm.cell_dofs(c)[0] for c in cells], dtype=np.int32)


def _assemble(form, comm):
    return comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM
    )


def run(case, ratio, mesh, cells_grain, cells_gb, f_gb, comm):
    """Steady solve; returns D_eff/D_m and the checks."""
    Dm = D_m(T)
    Dgb = ratio * Dm
    ax, tol = case.axis, case.tol

    grain = PhaseRegion(id=1, material=F.Material(D_0=D0_M, E_D=E_M), cells=cells_grain)
    band = PhaseRegion(
        id=2, material=F.Material(D_0=ratio * D0_M, E_D=E_M), cells=cells_gb
    )
    inlet = F.SurfaceSubdomain(id=3, locator=lambda x: np.abs(x[ax]) < tol)
    outlet = F.SurfaceSubdomain(
        id=4, locator=lambda x: np.abs(x[ax] - case.length) < tol
    )

    H = F.Species("H")
    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[grain, band, inlet, outlet],
        species=[H],
        temperature=T,
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=inlet, value=C_IN, species=H),
            F.FixedConcentrationBC(subdomain=outlet, value=C_OUT, species=H),
        ],
        # the residual scales with D C h ~ 1e5 (3D) to D C ~ 1e15 (2D) in these
        # units, far above atol, so convergence is decided by rtol and the
        # solver cannot stop at iteration 0 on a zero initial guess
        settings=F.Settings(atol=1e-8, rtol=1e-10, transient=False),
        exports=[],
        petsc_options=PETSC_OPTIONS_3D if case.structure == "iso" else None,
    )
    model.initialise()
    model.run()
    c = H.post_processing_solution

    # D as a DG0 field from the same classification, for the flux forms
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    D = dolfinx.fem.Function(V0, name="D")
    D.x.array[:] = Dgb
    D.x.array[_dg0_dofs(V0, cells_grain)] = Dm
    D.x.scatter_forward()

    dC = C_IN - C_OUT
    g = ufl.grad(c)[ax]
    j_out = _assemble(-D * g * _face_measure(mesh, ax, case.length, tol), comm)
    j_in = _assemble(-D * g * _face_measure(mesh, ax, 0.0, tol), comm)
    vol = _assemble(1.0 * ufl.dx(domain=mesh), comm)
    j_bar = _assemble(-D * g * ufl.dx, comm) / vol  # their Eq. 21
    g_bar = _assemble(g * ufl.dx, comm) / vol

    D_eff = j_out * case.length / (case.cross_section * dC)
    D_eff_avg = -j_bar / g_bar

    return dict(
        structure=case.structure,
        f_gb=f_gb,
        w_nm=case.w * 1e9,
        g_nm=case.foam.g * 1e9,
        S_v_per_nm=f_gb / (case.w * 1e9),
        n=case.n,
        ratio=ratio,
        D_m=Dm,
        D_eff=D_eff,
        D_eff_over_D_m=D_eff / Dm,
        D_eff_avg_over_D_m=D_eff_avg / Dm,
        flux_balance=(j_in - j_out) / j_in if j_in else float("nan"),
    )


# ----------------------------------------------------------------------------
# reference formulas from the paper
# ----------------------------------------------------------------------------
def hart(f, r):
    """Their Eq. 28: phases in parallel. Exact for Col_I along the columns."""
    return 1.0 + f * (r - 1.0)


def hashin_shtrikman(f, r):
    """Their Eq. 33 (Chen & Schuh 2007): D_eff/D_m for an isometric polycrystal
    with the boundary as the connected phase."""
    return r + (1.0 - f) / (1.0 / (1.0 - r) + f / (3.0 * r))


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
GRAIN_RGB = (1.0, 0.93, 0.15)  # their yellow grains
BAND_RGB = (0.13, 0.40, 0.85)  # their blue boundaries


def draw_isometric(ax, foam, w, n=220, title=""):
    """The cube of side B seen from (+x, -y, +z), its three visible faces
    coloured by the same classifier the mesh uses."""
    b = foam.B
    t = (np.arange(n) + 0.5) / n * b
    U, V = np.meshgrid(t, t, indexing="ij")
    full = np.full_like(U, b)
    zero = np.zeros_like(U)
    faces = [(full, U, V), (U, zero, V), (U, V, full)]  # x = B, y = 0, z = B
    for X, Y, Z in faces:
        pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        grain = foam.is_grain(pts, w).reshape(U.shape)
        rgb = np.where(grain[..., None], GRAIN_RGB, BAND_RGB)
        ax.plot_surface(
            X * 1e9,
            Y * 1e9,
            Z * 1e9,
            facecolors=rgb,
            rstride=1,
            cstride=1,
            shade=False,
            linewidth=0,
            antialiased=False,
        )
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((1, 1, 1))
    ticks = np.linspace(0, b * 1e9, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xlabel("x (nm)", labelpad=4)
    ax.set_ylabel("y (nm)", labelpad=4)
    ax.set_zlabel("z (nm)", labelpad=4)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, b * 1e9)
    ax.set_ylim(0, b * 1e9)
    ax.set_zlim(0, b * 1e9)
    ax.set_title(title, fontsize=10)


def draw_fig4ab(iso_case, col_case, f_iso=None, f_col=None):
    fig = plt.figure(figsize=(9, 4.5))
    for i, (case, name, f) in enumerate(
        [(iso_case, "Iso", f_iso), (col_case, "Col-I", f_col)], start=1
    ):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ftxt = "" if f is None else f", f_GB = {f:.2f}"
        draw_isometric(
            ax,
            case.foam,
            case.w,
            title=f"{name}: {case.foam.n_seeds**case.foam.dim} grains, g = "
            f"{case.foam.g * 1e9:.0f} nm, w = {case.w * 1e9:.1f} nm{ftxt}",
        )
    fig.tight_layout()
    fig.savefig("li2022-fig4ab.png", dpi=150)


def draw_fig4(rows):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    f = np.linspace(0.25, 0.75, 100)
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
        axp.text(0.05, 0.9, f"$D_{{GB}}/D_m$ = {r:g}", transform=axp.transAxes)
        axp.set_xlim(0.25, 0.75)
        axp.legend(fontsize=7, loc="lower right" if r > 1 else "upper right")
    for axp in axes[1]:
        axp.set_xlabel("grain-boundary volume fraction $f_{GB}$ = $S_v$ x 1 nm")
    for axp in axes[:, 0]:
        axp.set_ylabel(r"$D^{eff}/D_m$")
    fig.suptitle(
        "Li et al. 2022 Fig. 4 D,E,G,H — Voronoi foam / extruded Voronoi columns, "
        "volumetric GB band",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig("li2022-fig4.png", dpi=150)


# ----------------------------------------------------------------------------
def cases_for(structure):
    if structure == "iso":
        foam = Foam(dim=3, n_seeds=ISO_SEEDS)
        return [Case(structure, foam, wg * foam.g) for wg in ISO_W_OVER_G]
    foam = Foam(dim=2, n_seeds=COL_SEEDS)
    return [Case(structure, foam, wg * foam.g) for wg in COL_W_OVER_G]


def main():
    mpl.use("Agg")
    comm = MPI.COMM_WORLD
    rows = []
    for structure in STRUCTURES:
        cases = cases_for(structure)
        mesh = make_mesh(cases[0], comm)  # the mesh does not depend on w
        for case in cases:
            cells_grain, cells_gb, f_gb = classify_cells(mesh, case)
            if comm.rank == 0:
                print(
                    f"{LABEL[structure]:9s} w/g={case.w / case.foam.g:.2f}  "
                    f"w={case.w * 1e9:.2f} nm ({case.w / case.h:.1f} cells)  "
                    f"f_GB={f_gb:.3f}  S_v={f_gb / (case.w * 1e9):.3f}/nm  "
                    f"mesh {'x'.join(map(str, case.n_cells))}"
                )
            for ratio in RATIOS:
                r = run(case, ratio, mesh, cells_grain, cells_gb, f_gb, comm)
                rows.append(r)
                if comm.rank == 0:
                    ref = (
                        hart(f_gb, ratio)
                        if structure == "col_z"
                        else hashin_shtrikman(f_gb, ratio)
                    )
                    print(
                        f"    r={ratio:6g}  D_eff/D_m={r['D_eff_over_D_m']:8.4f}  "
                        f"(Eq. 21 avg {r['D_eff_avg_over_D_m']:8.4f}, in/out {r['flux_balance']:+.1e})  "
                        f"{'Hart' if structure == 'col_z' else 'HS'}={ref:8.4f}"
                    )
    if comm.rank != 0:
        return
    # the measured f_GB of each sweep member (one entry per band width)
    fi = [q["f_gb"] for q in rows if q["structure"] == "iso"][:: len(RATIOS)]
    fc = [q["f_gb"] for q in rows if q["structure"] == "col_z"][:: len(RATIOS)]
    draw_fig4ab(
        cases_for("iso")[RENDER_INDEX],
        cases_for("col_z")[RENDER_INDEX],
        fi[RENDER_INDEX] if len(fi) > RENDER_INDEX else None,
        fc[RENDER_INDEX] if len(fc) > RENDER_INDEX else None,
    )
    draw_fig4(rows)


if __name__ == "__main__":
    main()
