"""Identify, validate and plot effective grain-boundary diffusivity.

Run this file for the complete study, or use --skip-validation / --skip-figures.
--plot-only regenerates summary figures from the saved JSON without solving.
All outputs live in examples/results/gb_homogenisation/.
Geometry, grain/network subdomains, lattice tensors, averages and field exports
come from festim_microstructure; the FESTIM problem and figures are study-specific.
"""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from mpi4py import MPI

import dolfinx
import festim as F
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.projections.polar import PolarAxes

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem

NETWORK_ID = 1_000_000  # above every grain id; a manifold shares the surface ids
SURFACE_ID_0 = 2_000_000  # the per-grain surface patches are numbered from here

# The problem is unscaled (D ~ 1e-11 m2/s, cell area ~ 1e-11 m2), so the Newton
# residual of a converged step is itself ~1e-12 and FESTIM's default atol stalls
# silently; see docs/gb_homogenisation.md. Nondimensionalising is the better fix.
ATOL = 1e-25
RTOL = 1e-10


@dataclass(frozen=True)
class Transport:
    """The coefficients of one run, evaluated at its temperature.

    FESTIM would evaluate an Arrhenius law itself from
    ``Material(D_0=..., E_D=...)``, but the numbers are needed here anyway: the
    lattice tensor field is built from ``D_bulk`` and the network flux
    ``delta * D_gb * grad(c_gb)`` is summed into every average.
    """

    T: float  # K
    D_bulk: float  # m2/s, the (isotropic) lattice diffusivity
    D_gb: float  # m2/s, the boundary diffusivity
    delta: float  # m, boundary width
    k: float  # m/s, grain <-> boundary transfer coefficient
    crystal_anisotropy: float = 1.0  # set by the lattice; 1 for cubic metals

    @classmethod
    def tungsten(cls, T=500.0, k=3.0, crystal_anisotropy=1.0):
        """Tungsten-like, GB-dominated: lattice ``D_0 = 1.9e-7 m2/s``,
        ``E_D = 0.39 eV``; boundaries ``D_0 = 2.85e-7`` (1.5x, a 2D against a
        3D random walk), ``E_D = 0.12 eV``; ``delta = 1 nm``."""
        return cls(
            T=T,
            D_bulk=1.9e-7 * np.exp(-0.39 / (F.k_B * T)),
            D_gb=2.85e-7 * np.exp(-0.12 / (F.k_B * T)),
            delta=1e-9,
            k=k,
            crystal_anisotropy=crystal_anisotropy,
        )

    @property
    def contrast(self):
        return self.D_gb / self.D_bulk

    def report(self, grain_size=None):
        length = fm.materials.equilibration_length(self.delta, self.D_gb, self.k)
        lines = [
            f"transport at T = {self.T:g} K",
            f"  D_bulk (orientation average)   : {self.D_bulk:.3e} m2/s",
            f"  D_gb                           : {self.D_gb:.3e} m2/s",
            f"  D_gb / D_bulk                  : {self.contrast:.4g}",
            f"  crystal anisotropy             : {self.crystal_anisotropy:g}",
            f"  boundary width, delta          : {1e9 * self.delta:g} nm",
            f"  exchange rate, k               : {self.k:.3e} m/s",
            f"  equilibration length           : {1e9 * length:.3g} nm",
        ]
        if grain_size is not None:
            ratio = fm.materials.interface_resistance_ratio(
                self.k, grain_size, self.D_bulk
            )
            lines.append(f"  interface / lattice resistance : {ratio:.3e}")
        return "\n".join(lines)


@dataclass
class CellProblem:
    """A declared cell problem and the handles its post-processing reads."""

    model: F.HydrogenTransportProblemDiscontinuous
    micro: fm.VoronoiMicrostructure
    transport: Transport
    grains: list
    species: list  # one per grain, aligned
    tensors: dict  # grain id -> lattice tensor
    network: fm.GrainBoundaryNetwork
    c_gb: F.Species

    def averages(self, window=None):
        """Flux, gradient and measure, delegated to the package."""
        t = self.transport
        return fm.exports.averages.averages(
            self.grains,
            self.species,
            self.tensors,
            self.network,
            self.c_gb,
            t.delta,
            t.D_gb,
            window,
        )

    def inventory(self):
        return fm.exports.averages.inventory(
            self.grains, self.species, self.network, self.c_gb, self.transport.delta
        )

    def solve(self):
        self.model.initialise()
        fm.fem.solvers.tune_direct_solver(self.model)  # MUMPS workspace
        self.model.run()
        return self


def cell_problem(micro, transport, bcs, settings=None):
    """Declare the grain/network problem for ``micro`` under ``bcs``.

    ``bcs`` is a list of ``(locator, value)`` pairs; each is applied to every
    grain touching that surface and to the network's mouths on it. Every grain
    is a ``VolumeSubdomain`` with a ``Species`` of its own, the network is one
    codim-1 subdomain with one species, and each grain exchanges
    ``k (c_grain - c_gb)`` with the boundary slab: a ``ParticleFluxBC`` on the
    grain and, since the network equation is written per unit slab width, a
    ``ParticleSource`` of ``k / delta`` times the same jump on the network.
    Steady by default; pass ``settings`` for a transient.
    """
    mesh = micro.mesh
    D_lattice, tensors = fm.materials.crystal_diffusivity_field(
        micro, transport.D_bulk, transport.crystal_anisotropy
    )
    grains = fm.fem.subdomains.grain_subdomains(micro, F.Material(D=D_lattice))
    network = fm.fem.subdomains.grain_boundary_network(
        NETWORK_ID, micro, F.Material(D_0=transport.D_gb, E_D=0.0)
    )
    grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    species_of = dict(zip((g.id for g in grains), grain_species, strict=True))

    k = dolfinx.fem.Constant(mesh, transport.k)
    width = dolfinx.fem.Constant(mesh, transport.delta)
    sources, boundary_conditions = [], []
    for c_grain in grain_species:
        exchange = {"c_g": c_grain, "c_n": c_gb}
        sources.append(
            F.ParticleSource(
                value=lambda c_g, c_n: (k / width) * (c_g - c_n),
                species=c_gb,
                volume=network,
                species_dependent_value=exchange,
            )
        )
        boundary_conditions.append(
            F.ParticleFluxBC(
                subdomain=network,
                species=c_grain,
                value=lambda c_g, c_n: k * (c_n - c_g),
                species_dependent_value=exchange,
            )
        )

    subdomains = [*grains, network]
    next_id = SURFACE_ID_0
    for locator, value in bcs:
        patches, mouths = fm.fem.subdomains.grain_surfaces(
            mesh, grains, locator, next_id
        )
        next_id = mouths.id + 1
        subdomains += [*patches, mouths]
        boundary_conditions += [
            F.FixedConcentrationBC(
                subdomain=p, value=value, species=species_of[p.grain_id]
            )
            for p in patches
        ]
        boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=mouths, value=value, species=c_gb)
        )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        subdomains=subdomains,
        species=[*grain_species, c_gb],
        sources=sources,
        boundary_conditions=boundary_conditions,
        temperature=transport.T,
        settings=settings or F.Settings(atol=ATOL, rtol=RTOL, transient=False),
    )
    model.show_progress_bar = False
    return CellProblem(
        model, micro, transport, grains, grain_species, tensors, network, c_gb
    )


def make_microstructure(size, grain_size, aspect=1.0, seed=0, cells_per_grain=10):
    """A microstructure whose Voronoi seeds are spaced by ``grain_size``."""
    n_seeds = max(2, round((size / grain_size) ** 2))
    return fm.VoronoiMicrostructure.create(
        size=size,
        n_seeds=n_seeds,
        aspect=aspect,
        seed=seed,
        cells_per_grain=cells_per_grain,
    )


def hart_bound(cp: CellProblem):
    """Return the parallel Hart/Voigt bound for the modelled microstructure."""
    micro, transport = cp.micro, cp.transport
    if not isinstance(micro, fm.VoronoiMicrostructure) or micro.dim != 2:
        raise TypeError("this Hart bound requires a 2D VoronoiMicrostructure")
    tensor = fm.voronoi.network_tensor(micro.boundaries, micro.dim)
    return (
        fm.exports.averages.mean_lattice_tensor(micro, cp.tensors)
        + transport.delta * transport.D_gb / micro.domain_measure * tensor
    )


@dataclass
class Identification:
    """The outcome of one identification."""

    D_cell: list  # 2x2, averaged over the whole cell (upper estimate)
    D_window: list  # 2x2, averaged over an interior window
    D_hart: list  # 2x2, the parallel bound
    D_bulk: float
    size: float
    grain_size: float
    aspect: float
    seed: int
    n_seeds: int
    n_grains: int
    n_cells: int
    k_exchange: float
    equilibrium_error: float

    @staticmethod
    def _principal(D):
        sym = 0.5 * (np.asarray(D) + np.asarray(D).T)
        evals, evecs = np.linalg.eigh(sym)
        order = np.argsort(evals)[::-1]
        return evals[order], evecs[:, order]

    def report(self, name="D_eff"):
        lines = []
        for label, D in (("whole cell", self.D_cell), ("window", self.D_window)):
            D = np.asarray(D)
            evals, evecs = self._principal(D)
            asym = abs(D[0, 1] - D[1, 0]) / max(abs(D).max(), 1e-300)
            lines += [
                f"  {name} ({label}), m2/s",
                f"    [[{D[0, 0]:.4e}, {D[0, 1]:+.4e}],",
                f"     [{D[1, 0]:+.4e}, {D[1, 1]:.4e}]]",
                f"    principal values / D_bulk   : "
                f"{evals[0] / self.D_bulk:.3f}, {evals[1] / self.D_bulk:.3f}",
                f"    anisotropy ratio            : {evals[0] / evals[1]:.3f}",
                f"    strong axis                 : "
                f"({evecs[0, 0]:+.3f}, {evecs[1, 0]:+.3f})",
                f"    non-symmetry of the estimate: {asym:.2e}",
            ]
        hart = np.asarray(self.D_hart)
        lines += [
            f"  Hart bound / D_bulk           : "
            f"{hart[0, 0] / self.D_bulk:.3f} (xx), {hart[1, 1] / self.D_bulk:.3f} (yy)",
            f"  fraction of the bound reached : "
            f"{np.asarray(self.D_cell)[0, 0] / hart[0, 0]:.3f} (xx), "
            f"{np.asarray(self.D_cell)[1, 1] / hart[1, 1]:.3f} (yy)"
            "   [whole-cell estimate, the one the bound applies to]",
            f"  local equilibrium error       : {self.equilibrium_error:.2e}",
        ]
        return "\n".join(lines)


def identify(
    micro,
    transport,
    window_fraction=0.5,
    export_prefix=None,
    verbose=True,
    field_cases=None,
):
    """Solve the two cell problems and assemble the effective tensor."""
    half = 0.5 * (1.0 - window_fraction) * micro.size
    window = ((half, half), (micro.size - half, micro.size - half))
    Q_cell, H_cell, Q_win, H_win = (np.zeros((2, 2)) for _ in range(4))
    eq_error = 0.0
    hart = None
    for j, G in enumerate((np.array([1.0, 0.0]), np.array([0.0, 1.0]))):
        # the uniform-gradient (Taylor) condition c = G.x on the whole boundary
        cp = cell_problem(
            micro,
            transport,
            bcs=[
                (
                    lambda x: np.full_like(x[0], True, dtype=bool),
                    (lambda x, G=G: G[0] * x[0] + G[1] * x[1]),
                )
            ],
        ).solve()

        q, g, _ = cp.averages()
        Q_cell[:, j], H_cell[:, j] = q, g
        q_w, g_w, _ = cp.averages(window=window)
        Q_win[:, j], H_win[:, j] = q_w, g_w
        eq_error = max(
            eq_error,
            fm.exports.averages.equilibrium_error(
                cp.grains, cp.species, cp.network, cp.c_gb, micro.tolerance
            ),
        )
        hart = hart_bound(cp)
        if field_cases is not None:
            field_cases.append(
                ("along x" if j == 0 else "across x", *network_flux_segments(cp, 1e6))
            )

        if export_prefix is not None:
            # the grains as one discontinuous parent field (so the jumps show),
            # and the network as a line dataset
            fm.exports.averages.write_vtx(
                cp.grains, cp.species, cp.network, cp.c_gb, f"{export_prefix}_{'xy'[j]}"
            )
        if verbose:
            print(f"    solved cell problem G = e_{'xy'[j]}", flush=True)

    assert hart is not None
    return Identification(
        D_cell=(-Q_cell @ np.linalg.inv(H_cell)).tolist(),
        D_window=(-Q_win @ np.linalg.inv(H_win)).tolist(),
        D_hart=hart.tolist(),
        D_bulk=transport.D_bulk,
        size=micro.size,
        grain_size=np.sqrt(micro.domain_measure / micro.n_grains),
        aspect=micro.aspect,
        seed=micro.seed,
        n_seeds=micro.n_seeds,
        n_grains=micro.n_grains,
        n_cells=micro.mesh.topology.index_map(2).size_global,
        k_exchange=transport.k,
        equilibrium_error=eq_error,
    )


def permeation_bcs(size, direction, c_in=1.0, c_out=0.0):
    """Fix concentration on opposite faces normal to ``direction``."""
    axis = "xy".index(direction)
    return [
        (lambda x, a=axis: np.isclose(x[a], 0.0), c_in),
        (lambda x, a=axis: np.isclose(x[a], size), c_out),
    ]


def steady_consistency(micro, transport, candidates, verbose=True):
    """Score candidate tensors against a permeation boundary condition."""
    rows = {}
    for direction in ("x", "y"):
        cp = cell_problem(
            micro, transport, bcs=permeation_bcs(micro.size, direction)
        ).solve()
        q, grad_c, _ = cp.averages()
        # Relative transverse errors are meaningless near zero flux.
        i = "xy".index(direction)
        if verbose:
            print(f"  driven along {direction}: q_{direction} = {q[i]:+.5e}")
        for label, D in candidates.items():
            predicted = -np.asarray(D) @ grad_c
            error = (predicted[i] - q[i]) / max(abs(q[i]), 1e-300)
            rows[(direction, label)] = error
            if verbose:
                print(
                    f"    {label:<11s} predicts {predicted[i]:+.5e}"
                    f"   ({100 * error:+6.2f} %)",
                    flush=True,
                )
    return rows


def homogeneous_model(
    size,
    D_eff,
    transport,
    bcs,
    n=48,
    transient=False,
    final_time=None,
    stepsize=None,
    atol=ATOL,
):
    """Build a homogeneous rectangle carrying the anisotropic tensor."""
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([size, size])], [n, n]
    )
    diffusivity_space = dolfinx.fem.functionspace(mesh, ("DG", 0, (2, 2)))
    diffusivity = cast(
        dolfinx.fem.Function,
        dolfinx.fem.Function(diffusivity_space, name="D_eff"),
    )
    diffusivity.x.array.reshape(-1, 4)[:] = np.asarray(D_eff).reshape(4)
    diffusivity.x.scatter_forward()
    volume = F.VolumeSubdomain(
        id=1,
        material=F.Material(D=diffusivity),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    c = F.Species("c", subdomains=[volume])
    subdomains: list[F.VolumeSubdomain | F.SurfaceSubdomain] = [volume]
    boundary_conditions = []
    for i, (locator, value) in enumerate(bcs):
        surface = F.SurfaceSubdomain(id=10 + i, locator=locator)
        subdomains.append(surface)
        boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=surface, value=value, species=c)
        )
    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        species=[c],
        subdomains=subdomains,
        boundary_conditions=boundary_conditions,
        temperature=transport.T,
        settings=F.Settings(
            atol=atol,
            rtol=RTOL,
            transient=transient,
            final_time=final_time,
            stepsize=stepsize,
        ),
    )
    model.show_progress_bar = False
    return model, c, volume, mesh


def uptake(micro, transport, D_eff, n_steps=60, verbose=True):
    """Compare microstructure and homogeneous uptake over one crossing time."""
    size = micro.size
    slow = min(np.linalg.eigvalsh(0.5 * (D_eff + D_eff.T)))
    final_time = 0.35 * size**2 / slow
    dt = final_time / n_steps
    bcs = [(lambda x: np.isclose(x[1], size), 1.0)]

    cp = cell_problem(
        micro,
        transport,
        bcs=bcs,
        settings=F.Settings(
            atol=ATOL,
            rtol=RTOL,
            transient=True,
            final_time=final_time,
            stepsize=F.Stepsize(initial_value=dt),
        ),
    )
    cp.model.initialise()  # the stepping is driven here rather than by run()
    fm.fem.solvers.tune_direct_solver(cp.model)

    homogeneous, c, volume, _ = homogeneous_model(
        size,
        D_eff,
        transport,
        bcs,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(initial_value=dt),
    )
    total = F.TotalVolume(field=c, volume=volume)
    homogeneous.exports = [total]
    homogeneous.initialise()
    times, micro_inventory, model_inventory = [0.0], [0.0], [0.0]
    while cp.model.t.value < final_time - 0.5 * dt:
        cp.model.iterate()
        homogeneous.iterate()
        times.append(float(cp.model.t))
        micro_inventory.append(cp.inventory())
        model_inventory.append(float(total.value))
        if verbose and len(times) % 10 == 0:
            print(
                f"    t = {times[-1]:.3e} s  microstructure {micro_inventory[-1]:.4e}"
                f"  homogeneous {model_inventory[-1]:.4e}",
                flush=True,
            )

    return (
        np.array(times),
        np.array(micro_inventory),
        np.array(model_inventory),
        final_time,
    )


# --- design tokens -------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQUENTIAL = [
    "#cde2fb",
    "#9ec5f4",
    "#5598e7",
    "#2a78d6",
    "#256abf",
    "#184f95",
    "#0d366b",
]
BLUES = LinearSegmentedColormap.from_list("blues", SEQUENTIAL)

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
}


def tidy(ax, grid_axis="y"):
    """Recessive chrome: no top/right spines, a hairline grid on one axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def label_end(ax, x, y, text, color, dx=6, dy=0, ha="left", va="center"):
    """Direct label at the end of a series, so identity never rests on colour."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        fontweight="bold",
        ha=ha,
        va=va,
    )


def place_labels(ax, x, items, dx=8, min_gap_px=13):
    """Direct-label several series at the same x without letting them collide.

    Labels are nudged apart in display space and the block re-centred, so their
    order still matches the order of the curves.
    """
    fig = ax.figure
    fig.canvas.draw()
    items = sorted(items, key=lambda t: t[0])
    original = [ax.transData.transform((x, y))[1] for y, _, _ in items]
    adjusted = list(original)
    for i in range(1, len(adjusted)):
        adjusted[i] = max(adjusted[i], adjusted[i - 1] + min_gap_px)
    shift = (sum(original) - sum(adjusted)) / len(adjusted)
    for (y, text, color), y0, y1 in zip(items, original, adjusted, strict=True):
        label_end(ax, x, y, text, color, dx=dx, dy=(y1 + shift - y0) * 72.0 / fig.dpi)


def title(ax, headline, sub=None):
    ax.set_title(headline, loc="left", color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(
            sub,
            xy=(0, 1),
            xytext=(0, 6),
            xycoords="axes fraction",
            textcoords="offset points",
            color=INK_2,
            fontsize=8.5,
            va="bottom",
        )


# --- figure 1: the microstructure and the corrector fields ---------------
def draw_network(ax, segments, color=INK, lw=0.9, alpha=1.0):
    for p, q in segments:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, alpha=alpha, zorder=3)


def network_flux_segments(cp, scale):
    """Each boundary segment, and the tangential flux it carries.

    The flux is reported as the *width of lattice carrying the same flux*,
    ``delta D_gb |ds c_gb| / (D_bulk |G|)``. That turns a quantity with awkward
    units into a length that can be held against the grain size: a boundary whose
    number exceeds the grain width is moving more hydrogen than a whole grain of
    lattice beside it.
    """
    c = cp.c_gb.subdomain_to_post_processing_solution[cp.network]
    V = c.function_space
    coords = V.tabulate_dof_coordinates()
    cells = V.dofmap.list
    p0, p1 = coords[cells[:, 0], :2], coords[cells[:, 1], :2]
    length = np.linalg.norm(p1 - p0, axis=1)
    keep = length > 0
    slope = np.abs(c.x.array[cells[:, 1]] - c.x.array[cells[:, 0]])[keep] / length[keep]
    t = cp.transport
    width = t.delta * t.D_gb * slope / t.D_bulk
    return np.stack([p0[keep] * scale, p1[keep] * scale], axis=1), width * scale


def colourbar(fig, mappable, ax):
    bar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7.5, color=AXIS)
    return bar


def figure_microstructure(micro, cases, path):
    """The polycrystal, and the flux its boundaries carry under each gradient.

    The two flux panels share one colour scale, which is the point: driven along
    the grain elongation the network lights up, driven across it far fewer
    boundaries are usefully oriented and the same colour scale stays dim. That
    difference *is* the anisotropy the identification puts a number on.

    There is deliberately no map of the bulk fluctuation here. It is very nearly
    zero: the network is a set of measure-zero lines, so it carries its extra flux
    without needing to disturb the lattice field much, and a map of it is a blank
    sheet of paper.
    """

    import matplotlib.pyplot as plt

    # cases are captured during identify(), so plotting never repeats a solve.
    size_um = 1e6 * micro.size
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    scale = 1e6  # metres -> microns
    segments_um = [(p * scale, q * scale) for p, q in micro.boundaries]
    grain_um = scale * np.sqrt(micro.domain_measure / micro.n_grains)

    ax = axes[0]
    ax.add_patch(
        Rectangle(
            (0, 0), size_um, size_um, facecolor="#f0efec", edgecolor="none", zorder=1
        )
    )
    draw_network(ax, segments_um, lw=1.0)
    title(
        ax,
        "The microstructure",
        f"{micro.n_grains} grains, elongated {micro.aspect:g}:1 along x",
    )

    ceiling = max(width.max() for _, _, width in cases)
    for ax, (name, lines, width) in zip(axes[1:], cases, strict=True):
        order = np.argsort(width)
        collection = LineCollection(
            lines[order],
            array=width[order],
            cmap=BLUES,
            linewidths=0.8 + 3.2 * width[order] / ceiling,
            zorder=3,
        )
        collection.set_clim(0.0, ceiling)
        ax.add_collection(collection)
        colourbar(fig, collection, ax)
        title(
            ax,
            f"Flux along the boundaries, driven {name}",
            "width of lattice carrying the same flux (um)",
        )
        ax.annotate(
            f"one grain is {grain_um:.2f} um across",
            xy=(0.5, 0.015),
            xycoords="axes fraction",
            ha="center",
            color=INK_2,
            fontsize=8,
        )

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, size_um)
        ax.set_ylim(0, size_um)
        ax.set_xlabel("x (um)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("y (um)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figure 2: the identified anisotropy ---------------------------------
def directional(D, theta):
    """``n . D . n``: the diffusivity felt along the direction ``theta``."""
    n = np.stack([np.cos(theta), np.sin(theta)])
    return np.einsum("in,ij,jn->n", n, D, n)


def figure_anisotropy(ident, path):
    import matplotlib.pyplot as plt

    theta = np.linspace(0, 2 * np.pi, 721)
    D_b = ident["D_bulk"]
    curves = [
        (np.asarray(ident["D_hart"]), SERIES[2], "Hart bound"),
        (np.asarray(ident["D_window"]), SERIES[0], "identified"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 5.2), subplot_kw={"projection": "polar"})
    ax.plot(
        theta,
        np.ones_like(theta),
        color=MUTED,
        lw=1.6,
        ls=(0, (4, 3)),
        zorder=2,
    )
    label_end(ax, np.deg2rad(70), 1.0, "lattice alone", MUTED, dx=4, dy=8)

    for D, color, name in curves:
        radius = directional(D, theta) / D_b
        ax.plot(theta, radius, color=color, lw=2.0, zorder=3)
        # labelled off the shoulder of each lobe rather than at 0 deg, where the
        # two curves are a hair apart and the angular tick already sits
        angle = np.deg2rad(20 if name == "identified" else -20)
        label_end(
            ax,
            angle,
            directional(D, np.array([angle]))[0] / D_b,
            name,
            color,
            dx=10,
            dy=6 if name == "identified" else -6,
        )

    assert isinstance(ax, PolarAxes)
    ax.set_theta_zero_location("E")
    ax.set_rlabel_position(103)
    ax.set_rticks([1, 2, 3, 4])
    ax.set_yticklabels(["1", "2", "3", "4"], fontsize=8)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax.grid(color=GRID, lw=0.8)
    ax.spines["polar"].set_color(AXIS)
    fig.text(
        0.01,
        0.985,
        "Diffusivity by direction, in units of the lattice value",
        color=INK,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.01,
        0.935,
        f"n . D_eff . n / D_bulk, {1e6 * ident['size']:.0f} um cell, "
        f"grains elongated {ident['aspect']:g}:1 along x",
        color=INK_2,
        fontsize=8.5,
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figure 3: does the cell contain enough grains? ----------------------
def figure_rve(identifications, path):
    import matplotlib.pyplot as plt

    sizes = np.array([1e6 * i["size"] for i in identifications])
    D_b = identifications[0]["D_bulk"]
    order = np.argsort(sizes)
    sizes = sizes[order]
    grains = [identifications[k]["n_grains"] for k in order]

    def component(key, i, j):
        return np.array(
            [np.asarray(identifications[k][key])[i, j] / D_b for k in order]
        )

    # the two components differ by a factor three, so a shared scale would squash
    # the weak axis into the bottom tenth of the frame and hide the convergence
    # that is the whole point of the figure. Each panel gets its own.
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for ax, (i, j), name in zip(
        axes, ((0, 0), (1, 1)), ("along x", "across x"), strict=True
    ):
        labels = []
        for key, color, label in (
            ("D_hart", SERIES[2], "Hart bound"),
            ("D_cell", SERIES[1], "whole cell"),
            ("D_window", SERIES[0], "window"),
        ):
            y = component(key, i, j)
            ax.plot(sizes, y, color=color, marker="o", markersize=7, zorder=3)
            labels.append((y[-1], label, color))
        tidy(ax)
        ax.set_xlabel("cell side (um)")
        ax.set_xticks(sizes)
        ax.set_xticklabels(
            [f"{s:g}\n{n} grains" for s, n in zip(sizes, grains, strict=True)]
        )
        ax.set_xlim(sizes[0] - 0.3, sizes[-1] + 1.9)
        ax.set_ylabel("D_eff / D_bulk")
        title(ax, f"D_eff {name}", None)
        place_labels(ax, sizes[-1], labels)
    fig.suptitle(
        "How far the estimate can be trusted, against cell size",
        x=0.008,
        ha="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.008,
        0.905,
        "compare whole-cell and window estimates as cell size increases; "
        "repeat seeds to assess sampling scatter",
        color=INK_2,
        fontsize=8.5,
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figure 4: where a single effective diffusivity stops existing -------
def figure_sweep(sweep, path, threshold=0.05):
    import matplotlib.pyplot as plt

    k = np.array([s["k_exchange"] for s in sweep])
    order = np.argsort(k)
    k = k[order]
    D_b = sweep[0]["D_bulk"]
    dxx = np.array([np.asarray(sweep[i]["D_window"])[0, 0] / D_b for i in order])
    dyy = np.array([np.asarray(sweep[i]["D_window"])[1, 1] / D_b for i in order])
    err = np.array([sweep[i]["equilibrium_error"] for i in order])

    broken = err > threshold
    edge = None
    if broken.any() and (~broken).any():
        # the shading ends midway (in log) between the last failing run and the
        # first passing one -- the transition was not resolved more finely
        edge = np.sqrt(k[broken].max() * k[~broken].min())

    fig, axes = plt.subplots(
        2, 1, figsize=(7.6, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]}
    )
    for ax in axes:
        if edge is not None:
            ax.axvspan(k.min() / 2, edge, color="#f0efec", zorder=0)

    ax = axes[0]
    series = ((dxx, SERIES[0], "D_eff along x"), (dyy, SERIES[1], "D_eff across x"))
    for y, color, label in series:
        ax.plot(k, y, color=color, marker="o", markersize=7, zorder=3)
        label_end(ax, k[-1], y[-1], label, color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("D_eff / D_bulk")
    tidy(ax)
    title(
        ax,
        "The plateau is the answer; the runaway is the diagnosis",
        "identified tensor against the grain-to-boundary exchange rate",
    )
    if edge is not None:
        ax.annotate(
            "grains decoupled:\nno single D_eff exists",
            xy=(k.min() * 1.6, max(dxx.max(), dyy.max())),
            color=INK_2,
            fontsize=8.5,
            va="top",
        )

    ax = axes[1]
    ax.plot(k, err, color=SERIES[3], marker="o", markersize=7, zorder=3)
    label_end(ax, k[-1], err[-1], "equilibrium\nerror", SERIES[3])
    ax.axhline(threshold, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"{threshold:g} -- above this the grains no longer sit at the boundary value",
        xy=(k.max(), threshold * 1.35),
        color=MUTED,
        fontsize=8,
        va="bottom",
        ha="right",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("exchange rate k (m/s)")
    ax.set_ylabel("max |c_grain - c_gb| / max c")
    tidy(ax)
    ax.set_xlim(k.min() / 2, k.max() * 6)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figure 5: does the tensor predict what it was not fitted to? --------
def figure_validation(validation, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    uptake = validation["uptake"]
    t = np.array(uptake["times"])
    # normalised by the inventory of a fully charged cell, so the axis is a
    # fraction rather than an integral in units nobody carries in their head
    full = validation["identification"]["size"] ** 2
    labels = []
    for key, color, label in (
        ("microstructure", SERIES[0], "microstructure model"),
        ("homogeneous", SERIES[1], "homogeneous D_eff"),
    ):
        y = np.array(uptake[key]) / full
        ax.plot(t, y, color=color, zorder=3)
        labels.append((y[-1], label, color))
    tidy(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("inventory, as a fraction of a fully charged cell")
    ax.set_xlim(0, t[-1] * 1.3)
    ax.set_ylim(0, None)
    place_labels(ax, t[-1], labels)
    title(
        ax,
        "Uptake from one face",
        "a transient the steady identification never saw",
    )

    ax = axes[1]
    labels = ["whole cell", "window", "Hart bound", "D_bulk only"]
    directions = ["x", "y"]
    height = 0.36
    positions = np.arange(len(labels))
    for d, (direction, hatch) in enumerate(zip(directions, ("", "///"), strict=True)):
        values = [
            100 * validation["permeation"][f"{direction}|{label}"] for label in labels
        ]
        offset = (d - 0.5) * (height + 0.03)
        bars = ax.barh(
            positions + offset,
            values,
            height=height,
            color=SERIES[0],
            hatch=hatch,
            edgecolor=SURFACE,
            linewidth=2.0,
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(
                f"{value:+.1f}%",
                xy=(value, bar.get_y() + bar.get_height() / 2),
                xytext=(5 if value >= 0 else -5, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
                color=INK_2,
            )
    ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelcolor=INK_2)
    ax.set_xlabel("error in the predicted permeation flux (%)")
    tidy(ax, grid_axis="x")
    ax.annotate(
        "solid: driven along x      hatched: driven across x",
        xy=(0, 1),
        xytext=(0, 6),
        xycoords="axes fraction",
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
        va="bottom",
    )
    ax.set_title("Permeation, with no-flux sides", loc="left", color=INK, pad=22)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_json(path, record):
    """Checkpoint each completed stage; only rank zero writes shared files."""
    if MPI.COMM_WORLD.rank == 0:
        with path.open("w") as stream:
            json.dump(record, stream, indent=2)


def validate(micro, transport, ident, path, n_steps=60, skip_transient=False):
    """Test an existing identification without fitting its tensor again."""
    record = {
        "identification": asdict(ident),
        "capacity": 1.0
        + transport.delta * micro.network_measure / micro.domain_measure,
    }
    write_json(path, record)
    print(f"effective capacity: {record['capacity']:.5f}")
    print("test A -- permeation with no-flux sides")
    errors = steady_consistency(
        micro,
        transport,
        {
            "whole cell": ident.D_cell,
            "window": ident.D_window,
            "Hart bound": ident.D_hart,
            "D_bulk only": ident.D_bulk * np.eye(2),
        },
    )
    record["permeation"] = {f"{d}|{label}": e for (d, label), e in errors.items()}
    write_json(path, record)
    if skip_transient:
        return record

    print("test B -- transient uptake")
    times, detailed, homogeneous, final_time = uptake(
        micro, transport, np.asarray(ident.D_window), n_steps=n_steps
    )
    record["uptake"] = {
        "times": times.tolist(),
        "microstructure": detailed.tolist(),
        "homogeneous": homogeneous.tolist(),
        "final_time": final_time,
    }
    final_inventory = detailed[-1]
    deviation = np.abs(detailed - homogeneous) / max(final_inventory, 1e-300)
    print(f"max inventory deviation: {100 * deviation.max():.2f}% of final inventory")
    print(f"final inventory deviation: {100 * deviation[-1]:.2f}%")
    half = 0.5 * final_inventory
    record["half_saturation"] = {
        "microstructure": float(np.interp(half, detailed, times)),
        "homogeneous": float(np.interp(half, homogeneous, times)),
    }
    write_json(path, record)
    return record


def render_figures(record, validation=None, field_case=None):
    """Render saved summaries and, optionally, fields captured during fitting."""
    plt = fm.plotting.use_agg()
    written = []
    with plt.rc_context(STYLE):
        identifications = record["identifications"]
        if identifications:
            biggest = max(identifications, key=lambda i: i["size"])
            written.append(
                figure_anisotropy(biggest, OUTPUT_DIR / "fig_anisotropy.png")
            )
            if len(identifications) > 1:
                written.append(figure_rve(identifications, OUTPUT_DIR / "fig_rve.png"))
        if record.get("k_sweep"):
            written.append(
                figure_sweep(record["k_sweep"], OUTPUT_DIR / "fig_sweep.png")
            )
        if validation and "uptake" in validation and "permeation" in validation:
            written.append(
                figure_validation(validation, OUTPUT_DIR / "fig_validation.png")
            )
        if field_case is not None:
            micro, cases = field_case
            written.append(
                figure_microstructure(
                    micro, cases, OUTPUT_DIR / "fig_microstructure.png"
                )
            )
    for path in written:
        print(f"  -> {path}")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--grain-size", type=float, default=0.6e-6)
    parser.add_argument("--sizes", type=float, nargs="+", default=[3e-6, 5e-6, 8e-6])
    parser.add_argument("--aspect", type=float, default=4.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--temperature", type=float, default=500.0)
    parser.add_argument("--crystal-anisotropy", type=float, default=1.0)
    parser.add_argument(
        "--k-sweep",
        type=float,
        nargs="+",
        help="exchange rates (m/s), evaluated on the first size and seed",
    )
    parser.add_argument("--cells-per-grain", type=int, default=8)
    parser.add_argument("--window-fraction", type=float, default=0.5)
    parser.add_argument(
        "--export", action="store_true", help="write VTX corrector fields"
    )
    parser.add_argument(
        "--steps", type=int, default=60, help="transient validation steps"
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-transient", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--skip-fields", action="store_true", help="omit the microstructure/flux figure"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="plot saved JSON; no solves or field maps",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("identified.json"),
        help="identification JSON, relative to examples/results/gb_homogenisation/",
    )
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=Path("validation.json"),
        help="validation JSON in the same results folder",
    )
    args = parser.parse_args(argv)
    for name in ("out", "validation_out"):
        path = (OUTPUT_DIR / getattr(args, name)).resolve()
        if not path.is_relative_to(OUTPUT_DIR):
            parser.error(f"--{name.replace('_', '-')} must stay within {OUTPUT_DIR}")
        setattr(args, name, path)
    if args.out == args.validation_out:
        parser.error("--out and --validation-out must be different files")
    if args.plot_only and args.skip_figures:
        parser.error("--plot-only cannot be combined with --skip-figures")
    if not 0 < args.window_fraction <= 1:
        parser.error("--window-fraction must be in (0, 1]")
    if (
        min(args.sizes) <= 0
        or min(
            args.grain_size,
            args.aspect,
            args.temperature,
            args.crystal_anisotropy,
            args.cells_per_grain,
            args.steps,
        )
        <= 0
    ):
        parser.error(
            "sizes, material parameters, mesh density and steps must be positive"
        )
    if args.k_sweep and min(args.k_sweep) <= 0:
        parser.error("--k-sweep rates must be positive")

    if MPI.COMM_WORLD.rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.barrier()
    if args.plot_only:
        if MPI.COMM_WORLD.rank == 0:
            with args.out.open() as stream:
                record = json.load(stream)
            validation = None
            if not args.skip_validation and args.validation_out.exists():
                with args.validation_out.open() as stream:
                    validation = json.load(stream)
            render_figures(record, validation)
        return

    transport = Transport.tungsten(
        T=args.temperature, crystal_anisotropy=args.crystal_anisotropy
    )
    record = {"identifications": [], "k_sweep": [], "transport": asdict(transport)}
    first_case = None
    field_case = None
    for size in args.sizes:
        for seed in args.seeds:
            micro = make_microstructure(
                size, args.grain_size, args.aspect, seed, args.cells_per_grain
            )
            print(
                transport.report(
                    grain_size=np.sqrt(micro.domain_measure / micro.n_grains)
                )
            )
            print(micro.report())
            fields = (
                []
                if first_case is None and not (args.skip_figures or args.skip_fields)
                else None
            )
            ident = identify(
                micro,
                transport,
                window_fraction=args.window_fraction,
                export_prefix=(
                    OUTPUT_DIR / f"corrector_size_{size}_seed_{seed}"
                    if args.export
                    else None
                ),
                field_cases=fields,
            )
            print(ident.report())
            record["identifications"].append(asdict(ident))
            write_json(args.out, record)
            if first_case is None:
                first_case = (micro, ident)
                if fields is not None:
                    # All ranks participate; rank zero plots the complete network.
                    gathered = micro.mesh.comm.gather(fields, root=0)
                    if MPI.COMM_WORLD.rank == 0:
                        cases = [
                            (
                                fields[j][0],
                                np.concatenate([part[j][1] for part in gathered]),
                                np.concatenate([part[j][2] for part in gathered]),
                            )
                            for j in range(2)
                        ]
                        field_case = (micro, cases)

    micro, first_ident = first_case
    for k in args.k_sweep or []:
        swept_transport = replace(transport, k=k)
        ident = (
            first_ident
            if k == transport.k
            else identify(
                micro,
                swept_transport,
                window_fraction=args.window_fraction,
                verbose=False,
            )
        )
        verdict = "ok" if ident.equilibrium_error < 0.05 else "NO SINGLE D_eff"
        print(
            f"k = {k:.3e} m/s: equilibrium error "
            f"{ident.equilibrium_error:.2e}; {verdict}"
        )
        record["k_sweep"].append(asdict(ident))
        write_json(args.out, record)

    validation = None
    if not args.skip_validation:
        validation = validate(
            micro,
            transport,
            first_ident,
            args.validation_out,
            n_steps=args.steps,
            skip_transient=args.skip_transient,
        )
    if not args.skip_figures and MPI.COMM_WORLD.rank == 0:
        render_figures(record, validation, field_case)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
