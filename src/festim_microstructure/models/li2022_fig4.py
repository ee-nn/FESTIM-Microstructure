"""Li et al. (2022), Front. Mater. 9:935129, Fig. 4: the effective diffusion
coefficient of H in polycrystalline W with a *volumetric* grain boundary phase.

Their model (Sect. 2.2, Eqs. 16-17, 34) is the steady diffusion equation with
a piecewise-constant diffusivity: D_m in the grains, D_GB in a grain-boundary
phase of finite width, and one continuous concentration field. There is no
trapping, no segregation and no interface kinetics -- they state the method
applies where "there is no interaction between the diffusion component and
microstructure" (Sect. 1). The effective diffusivity is the ratio of the
average flux to the average gradient (Eq. 21). In FESTIM that is a plain
HydrogenTransportProblem with two volume subdomains carrying two materials,
solved with transient=False. This file reproduces their Fig. 4 panels A, B
(the two microstructures) and D, E, G, H (D_eff/D_m against grain-boundary
density for D_GB/D_m = 100, 10, 0.2 and 0.1).

THE TWO MICROSTRUCTURES, IDEALISED
----------------------------------
Their grains come from phase-field grain growth (Voronoi-like foams, Fig. 4A;
columns, Fig. 4B). Here both are periodic lattices, solved on one unit cell
with no-flux lateral faces, which is the exact periodic solution because the
cell's outer faces are mirror planes through the middle of the boundaries:

  "col1":  square columns of side L along z, separated by boundaries of
           width w. The unit cell is (L+w) x (L+w) in the x-y plane. Along z
           (their Col_I(Z)) the phases are in parallel and Hart's rule, their
           Eq. 28, is exact whatever the cross-section; across the columns
           (Col_I(X)) the cross-section is the "perpendicular slabs" cross of
           their Fig. 1B, and their Eqs. 30-31 (Jiang et al.) bound it.
  "iso":   cubic grains of side L on a simple-cubic lattice, boundaries of
           width w, unit cell (L+w)^3. Their Iso is compared with the
           Hashin-Shtrikman formula, Eq. 33, and so is this one; a cubic
           lattice is not the HS coated-sphere assemblage, so a residual
           against HS is expected and is not an error.

The grain-boundary *volume fraction* is the only geometric parameter that
matters for D_eff/D_m: 1 - (L/(L+w))^2 for columns, 1 - (L/(L+w))^3 for
cubes. The absolute width w sets nothing but the length unit.

THE X-AXIS
----------
Their Fig. 4 is plotted against "grain boundary density (nm^-1)" from 0.3 to
0.7. Their HS curve at 0.3 with D_GB/D_m = 100 reads ~22-23, and Eq. 33 gives
23.0 at a volume fraction f_GB = 0.3, so the axis is numerically the GB volume
fraction: their GB phase is the eta <= 0.9 band of a phase-field interface on
a 1 nm grid, one grid cell wide, and area per volume x 1 nm = volume fraction.
This file sweeps f_GB directly and labels the axis accordingly.

DIMENSIONS
----------
Everything is SI. D_m = 5.13e-8 exp(-0.21 eV/kT) m^2/s is the bulk value
they cite (Liu et al. 2014, their Sect. 2.3); the boundary carries the same
activation energy scaled by the ratio, so D_GB/D_m is temperature-independent
as in their sweep. T = 1073 K is the temperature of their Sect. 3.4 and
affects nothing plotted. The imposed concentrations are 0.4 and 0.1 in their
units (Sect. 2.3); here 0.4e24 and 0.1e24 m^-3, which likewise affect nothing
since the problem is linear.

MESH AND EXTRACTION
-------------------
Structured meshes with the cell size h = w / CELLS_PER_GB, so that every
grain/boundary interface lies on a node plane and the grain side L is snapped
to a multiple of h (the realised f_GB is reported, not the target). D_eff is
taken from the outlet flux, J_out a / (A dC), and cross-checked against the
volume average of their Eq. 21 and against the inlet flux; at steady state
the three agree to solver tolerance. Col_I(Z) is linear in z and is solved on
a two-cell-thick slab; Col_I(X) is a 2D cross-section; Iso is the full cube.

OUTPUTS
-------
li2022-fig4ab.png   the two unit cells (cross-sections), panels A-B
li2022-fig4.png     D_eff/D_m against f_GB for the four ratios, panels D,E,G,H,
                    with Hart, HS and the two Jiang bounds
li2022-<structure>-f<frac>.xdmf, if WRITE_XDMF, the phase tags for ParaView
"""

from dataclasses import dataclass

from mpi4py import MPI

import dolfinx
import festim as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ufl

# physics (SI)
# ----------------------------------------------------------------------------
T = 1073.0  # K; only scales D_m (their Sect. 3.4)
D0_M = 5.13e-8  # m^2/s, Liu et al. 2014, cited in their Sect. 2.3
E_M = 0.21  # eV
RATIOS = [100.0, 10.0, 0.2, 0.1]  # D_GB/D_m of their panels D, E, G, H
C_IN = 0.4e24  # m^-3, their 0.4 and 0.1 (arbitrary units)
C_OUT = 0.1e24

# geometry and mesh
# ----------------------------------------------------------------------------
W_GB = 1.0e-9  # m, boundary width; the length unit, nothing more
CELLS_PER_GB = 4  # cells across the boundary width; sets the f_GB snapping and the mesh
F_GB_TARGETS = [0.3, 0.4, 0.5, 0.6, 0.7]  # their x-range; snapped to the mesh
STRUCTURES = ["col1_z", "col1_x", "iso"]  # their Col_I(Z), Col_I(X), Iso
WRITE_XDMF = False

STRUCTURE_LABEL = {"col1_z": "Col_I(Z)", "col1_x": "Col_I(X)", "iso": "Iso"}


def D_m(T):
    return D0_M * np.exp(-E_M / (F.k_B * T))


@dataclass(frozen=True)
class Cell:
    """One periodic unit cell: grain side L (m), boundary width w (m), and the
    structure it belongs to. Everything else is derived."""

    structure: str
    L: float
    w: float

    @property
    def a(self):
        """Unit-cell side."""
        return self.L + self.w

    @property
    def h(self):
        return self.w / CELLS_PER_GB

    @property
    def n_side(self):
        return round(self.a / self.h)

    @property
    def dim(self):
        return 2 if self.structure == "col1_x" else 3

    @property
    def axis(self):
        """Index of the imposed-gradient direction."""
        return 0 if self.structure == "col1_x" else 2

    @property
    def grain_axes(self):
        """Axes along which the grain is bounded by boundary phase."""
        return (0, 1, 2) if self.structure == "iso" else (0, 1)

    @property
    def f_gb(self):
        return 1.0 - (self.L / self.a) ** len(self.grain_axes)

    @property
    def extent(self):
        """Domain size along each axis."""
        if self.structure == "col1_x":
            return (self.a, self.a)
        if self.structure == "col1_z":
            return (self.a, self.a, 2 * self.h)  # linear in z: two cells suffice
        return (self.a, self.a, self.a)

    @property
    def n_cells(self):
        n = self.n_side
        if self.structure == "col1_x":
            return (n, n)
        if self.structure == "col1_z":
            return (n, n, 2)
        return (n, n, n)

    @property
    def cross_section(self):
        """Area (3D) or width (2D) of the inlet/outlet face."""
        ext = list(self.extent)
        del ext[self.axis]
        return float(np.prod(ext))

    @property
    def length(self):
        return self.extent[self.axis]

    @property
    def label(self):
        return f"{STRUCTURE_LABEL[self.structure]} f_GB={self.f_gb:.3f}"

    def in_grain(self, x):
        """Vectorised point test: inside the grain block (open set)."""
        lo, hi = self.w / 2, self.L + self.w / 2
        ok = np.ones(x.shape[1], dtype=bool)
        for ax in self.grain_axes:
            ok &= (x[ax] > lo) & (x[ax] < hi)
        return ok


def snap_cell(structure, f_target, w):
    """The unit cell whose grain side is a multiple of h and whose f_GB is
    closest to f_target. The grain must be at least one cell wide."""
    exponent = 3 if structure == "iso" else 2
    ratio = (1.0 - f_target) ** (1.0 / exponent)  # L / (L + w)
    L_exact = w * ratio / (1.0 - ratio)
    h = w / CELLS_PER_GB
    L = max(1, round(L_exact / h)) * h
    return Cell(structure=structure, L=L, w=w)


# ----------------------------------------------------------------------------
# reference formulas from the paper
# ----------------------------------------------------------------------------
def hart(f, r):
    """Their Eq. 28: phases in parallel. Exact for Col_I along the columns."""
    return 1.0 + f * (r - 1.0)


def hashin_shtrikman(f, r):
    """Their Eq. 33 (Chen & Schuh 2007), D_eff/D_m for an isometric polycrystal
    with the boundary as the connected phase."""
    D_m_, D_gb = 1.0, r
    return D_gb + (1.0 - f) / (1.0 / (D_m_ - D_gb) + f / (3.0 * D_gb))


def jiang_bounds(f, r):
    """Their Eqs. 30-31 (Jiang et al. 2021) for the perpendicular-slab cross of
    their Fig. 1B, which is the Col_I cross-section. Written here from the
    series/parallel constructions they come from, in terms of the width
    fraction s = w/a (f = 1 - (1-s)^2):
      Eq. 30: strips along the flux; the boundary strip in parallel with
              strips that are grain and boundary in series
      Eq. 31: strips across the flux; the boundary strip in series with
              strips that are grain and boundary in parallel
    Eq. 31 as printed in the paper is reproduced exactly by this form (checked
    at D_b = 1, D_slab = 15, L_x = 64, w = 32: 10.4); Eq. 30 is printed with a
    typeset error in its last term and is taken from the construction. Which
    of the two is the upper bound depends on the sign of r - 1.
    """
    s = 1.0 - np.sqrt(1.0 - f)
    D_b, D_s = 1.0, r
    series = 1.0 / ((1.0 - s) / D_b + s / D_s)
    lower = s * D_s + (1.0 - s) * series
    parallel = s * D_s + (1.0 - s) * D_b
    upper = 1.0 / (s / D_s + (1.0 - s) / parallel)
    return lower, upper


# ----------------------------------------------------------------------------
# the FESTIM problem
# ----------------------------------------------------------------------------
class PhaseRegion(F.VolumeSubdomain):
    """Cells classified by midpoint, so that the grain/boundary split is exact
    on a mesh whose node planes coincide with the interfaces. FESTIM's default
    locator marks a cell when all its vertices satisfy the predicate, which
    would leave the boundary phase's interface-touching cells ambiguous."""

    def __init__(self, id, material, cell, grain):
        super().__init__(id=id, material=material)
        self.cell = cell
        self.grain = grain

    def locate_subdomain_entities(self, mesh):
        tdim = mesh.topology.dim
        n = (
            mesh.topology.index_map(tdim).size_local
            + mesh.topology.index_map(tdim).num_ghosts
        )
        mid = dolfinx.mesh.compute_midpoints(mesh, tdim, np.arange(n, dtype=np.int32))
        # compute_midpoints returns (n, 3) whatever the geometric dimension
        inside = self.cell.in_grain(mid.T)
        return np.flatnonzero(inside if self.grain else ~inside).astype(np.int32)


def make_mesh(cell, comm):
    lo = np.zeros(cell.dim)
    hi = np.array(cell.extent)
    if cell.dim == 2:
        return dolfinx.mesh.create_rectangle(
            comm, [lo, hi], list(cell.n_cells), cell_type=dolfinx.mesh.CellType.triangle
        )
    return dolfinx.mesh.create_box(
        comm, [lo, hi], list(cell.n_cells), cell_type=dolfinx.mesh.CellType.tetrahedron
    )


def _face_measure(mesh, axis, value):
    fdim = mesh.topology.dim - 1
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[axis], value, atol=1e-12)
    )
    tags = dolfinx.mesh.meshtags(
        mesh, fdim, np.sort(facets), np.ones(facets.size, dtype=np.int32)
    )
    return ufl.Measure("ds", domain=mesh, subdomain_data=tags)(1)


def _assemble(form, comm):
    return comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM
    )


def run(cell, ratio, comm=None):
    """Steady solve on one unit cell; returns D_eff/D_m and the checks."""
    comm = MPI.COMM_WORLD if comm is None else comm
    mesh = make_mesh(cell, comm)
    Dm = D_m(T)
    Dgb = ratio * Dm

    grain = PhaseRegion(
        id=1, material=F.Material(D_0=D0_M, E_D=E_M), cell=cell, grain=True
    )
    gb = PhaseRegion(
        id=2, material=F.Material(D_0=ratio * D0_M, E_D=E_M), cell=cell, grain=False
    )
    ax = cell.axis
    inlet = F.SurfaceSubdomain(
        id=3,
        locator=lambda x: np.isclose(x[ax], 0.0, atol=1e-12),
    )
    outlet = F.SurfaceSubdomain(
        id=4, locator=lambda x: np.isclose(x[ax], cell.length, atol=1e-12)
    )

    H = F.Species("H")
    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[grain, gb, inlet, outlet],
        species=[H],
        temperature=T,
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=inlet, value=C_IN, species=H),
            F.FixedConcentrationBC(subdomain=outlet, value=C_OUT, species=H),
        ],
        settings=F.Settings(atol=1e-10, rtol=1e-12, transient=False),
        exports=[],
    )
    model.initialise()
    model.run()
    c = (
        H.post_processing_solution
        if H.post_processing_solution is not None
        else H.solution
    )

    # D as a DG0 field from the same classification, for the flux forms
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    D = dolfinx.fem.Function(V0, name="D")
    D.x.array[:] = Dgb
    D.x.array[grain.locate_subdomain_entities(mesh)] = Dm
    D.x.scatter_forward()

    dC = C_IN - C_OUT
    g = ufl.grad(c)[ax]
    # outlet and inlet fluxes in the +axis direction (atoms/s, per unit depth in 2D)
    j_out = _assemble(-D * g * _face_measure(mesh, ax, cell.length), comm)
    j_in = _assemble(-D * g * _face_measure(mesh, ax, 0.0), comm)
    # their Eq. 21: volume-averaged flux over volume-averaged gradient
    vol = _assemble(1.0 * ufl.dx(domain=mesh), comm)
    j_bar = _assemble(-D * g * ufl.dx, comm) / vol
    g_bar = _assemble(g * ufl.dx, comm) / vol

    D_eff = j_out * cell.length / (cell.cross_section * dC)
    D_eff_avg = -j_bar / g_bar

    if WRITE_XDMF and comm.rank == 0:
        _write_xdmf(mesh, grain, gb, cell, ratio)

    return dict(
        structure=cell.structure,
        f_gb=cell.f_gb,
        L_nm=cell.L * 1e9,
        w_nm=cell.w * 1e9,
        n_side=cell.n_side,
        ratio=ratio,
        D_m=Dm,
        D_eff=D_eff,
        D_eff_over_D_m=D_eff / Dm,
        D_eff_avg_over_D_m=D_eff_avg / Dm,
        flux_balance=(j_in - j_out) / j_in if j_in else float("nan"),
    )


def _write_xdmf(mesh, grain, gb, cell, ratio):
    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local
    tags = np.full(n, gb.id, dtype=np.int32)
    tags[grain.locate_subdomain_entities(mesh)[:n]] = grain.id
    mt = dolfinx.mesh.meshtags(mesh, tdim, np.arange(n, dtype=np.int32), tags)
    mesh.topology.create_connectivity(tdim, tdim)
    name = f"li2022-{cell.structure}-f{cell.f_gb:.2f}-r{ratio:g}.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, name, "w") as xf:
        xf.write_mesh(mesh)
        xf.write_meshtags(mt, mesh.geometry)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def draw_unit_cells(cells_by_structure):
    """Panels A-B: cross-sections of the two unit cells at one f_GB, tiled
    3 x 3 so the periodic structure is visible."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for axp, key, title in zip(
        axes,
        ("iso", "col1_x"),
        ("Iso (cubic lattice, x-z section)", "Col-I (columns, x-y section)"),
        strict=True,
    ):
        cell = cells_by_structure[key]
        a, w, L = cell.a, cell.w, cell.L
        n = 200
        xs = (np.arange(n) + 0.5) / n * 3 * a
        X, Y = np.meshgrid(xs, xs)
        u, v = np.remainder(X, a), np.remainder(Y, a)
        grain = (u > w / 2) & (u < L + w / 2) & (v > w / 2) & (v < L + w / 2)
        axp.imshow(
            grain,
            origin="lower",
            extent=[0, 3 * a * 1e9, 0, 3 * a * 1e9],
            cmap="viridis",
        )
        axp.set_title(
            f"{title}\nf_GB = {cell.f_gb:.2f}, w = {w * 1e9:g} nm", fontsize=9
        )
        axp.set_xlabel("nm")
    axes[0].set_ylabel("nm")
    fig.tight_layout()
    fig.savefig("li2022-fig4ab.png", dpi=150)


def draw_fig4(rows):
    """Panels D, E, G, H in their layout: D_eff/D_m against f_GB, one panel per
    ratio, with the paper's reference curves."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    f = np.linspace(0.28, 0.72, 100)
    style = {
        "col1_z": dict(marker="*", ms=9, ls="none", color="tab:red", label="Col_I(Z)"),
        "col1_x": dict(
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
        lo, hi = jiang_bounds(f, r)
        axp.plot(
            f, lo, "--", color="tab:orange", lw=0.8, label="Jiang bounds, Eqs. 30-31"
        )
        axp.plot(f, hi, "--", color="tab:orange", lw=0.8)
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
        axp.set_xlabel("grain-boundary volume fraction $f_{GB}$ (their density x 1 nm)")
    for axp in axes[:, 0]:
        axp.set_ylabel(r"$D^{eff}/D_m$")
    fig.suptitle(
        f"Li et al. 2022 Fig. 4 D,E,G,H — volumetric GB, continuous C, "
        f"{CELLS_PER_GB} cells per GB width",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig("li2022-fig4.png", dpi=150)


# ----------------------------------------------------------------------------
def main():
    mpl.use("Agg")
    comm = MPI.COMM_WORLD
    rows = []
    cells_for_figure = {}
    for structure in STRUCTURES:
        for f_target in F_GB_TARGETS:
            cell = snap_cell(structure, f_target, W_GB)
            cells_for_figure.setdefault(structure, cell) if np.isclose(
                f_target, 0.5, atol=1e-12
            ) else None
            for ratio in RATIOS:
                r = run(cell, ratio, comm)
                rows.append(r)
                if comm.rank == 0:
                    ref = {
                        "col1_z": hart(cell.f_gb, ratio),
                        "iso": hashin_shtrikman(cell.f_gb, ratio),
                        "col1_x": None,
                    }[structure]
                    bounds = jiang_bounds(cell.f_gb, ratio)
                    print(
                        f"{cell.label:26s} r={ratio:6g}  n={cell.n_side:3d}^{cell.dim} "
                        f"D_eff/D_m={r['D_eff_over_D_m']:8.4f}  "
                        f"(vol.avg {r['D_eff_avg_over_D_m']:8.4f},   "
                        f"in/out {r['flux_balance']:+.1e})"
                        + (
                            f"ref={ref:8.4f}"
                            if ref is not None
                            else f"bounds=[{bounds[0]:.4f}, {bounds[1]:.4f}]"
                        )
                    )
    if comm.rank != 0:
        return

    # the section figure wants a 2D column cell and the cube at the same f
    cells_for_figure.setdefault("col1_x", snap_cell("col1_x", 0.5, W_GB))
    cells_for_figure.setdefault("iso", snap_cell("iso", 0.5, W_GB))
    draw_unit_cells(cells_for_figure)
    draw_fig4(rows)


if __name__ == "__main__":
    main()
