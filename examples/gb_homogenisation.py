"""Identify an anisotropic effective diffusivity from microstructure cell problems.

Whole-cell Taylor estimates are upper bounds; interior-window estimates reduce
boundary clamping. A large grain/GB mismatch means no single-field ``D_eff``.

The cell problem is FESTIM's ``HydrogenTransportProblemDiscontinuous``, declared
in :func:`cell_problem` the way FESTIM's manifold documentation shows; this
package supplies the tagged grains, the network, the per-grain surface patches,
the lattice tensor field and the averages. ``gb_validation.py`` and
``gb_figures.py`` import that declaration rather than repeat it.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import dolfinx
import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem

__all__ = [
    "ATOL",
    "RTOL",
    "CellProblem",
    "Identification",
    "Transport",
    "cell_averages",
    "cell_inventory",
    "cell_problem",
    "hart_bound",
    "identify",
    "make_microstructure",
]

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


def cell_averages(cp, window=None):
    """``(flux, gradient, measure)`` of a solved cell problem; see
    :func:`festim_microstructure.exports.averages.averages`."""
    t = cp.transport
    return fm.exports.averages.averages(
        cp.grains, cp.species, cp.tensors, cp.network, cp.c_gb, t.delta, t.D_gb, window
    )


def cell_inventory(cp, window=None):
    return fm.exports.averages.inventory(
        cp.grains, cp.species, cp.network, cp.c_gb, cp.transport.delta, window
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


def identify(micro, transport, window_fraction=0.5, export_prefix=None, verbose=True):
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

        q, g, _ = cell_averages(cp)
        Q_cell[:, j], H_cell[:, j] = q, g
        q_w, g_w, _ = cell_averages(cp, window=window)
        Q_win[:, j], H_win[:, j] = q_w, g_w
        eq_error = max(
            eq_error,
            fm.exports.averages.equilibrium_error(
                cp.grains, cp.species, cp.network, cp.c_gb, micro.tolerance
            ),
        )
        hart = hart_bound(cp)

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


def _dump(path, results, sweep):
    """Rewrite the results file. Called after every solve, so a run that dies
    late still leaves everything that had already been computed."""
    with open(path, "w") as f:
        json.dump({"identifications": results, "k_sweep": sweep}, f, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--grain-size", type=float, default=0.6e-6)
    parser.add_argument(
        "--sizes",
        type=float,
        nargs="+",
        default=[3e-6, 5e-6, 8e-6],
        help="cell sides to run, in metres: the RVE convergence study",
    )
    parser.add_argument("--aspect", type=float, default=4.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--temperature", type=float, default=500.0)
    parser.add_argument("--crystal-anisotropy", type=float, default=1.0)
    parser.add_argument(
        "--k-sweep",
        type=float,
        nargs="+",
        default=None,
        help="exchange rates (m/s) to run on the first cell size, to show where a "
        "single effective diffusivity stops existing",
    )
    parser.add_argument("--cells-per-grain", type=int, default=8)
    parser.add_argument("--window-fraction", type=float, default=0.5)
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("identified.json"),
        help="output filename, relative to examples/results/gb_homogenisation/",
    )
    args = parser.parse_args(argv)
    args.out = (OUTPUT_DIR / args.out).resolve()
    if not args.out.is_relative_to(OUTPUT_DIR):
        parser.error("--out must stay within " + str(OUTPUT_DIR))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    results, sweep = [], []
    for size in args.sizes:
        for seed in args.seeds:
            micro = make_microstructure(
                size, args.grain_size, args.aspect, seed, args.cells_per_grain
            )
            grain_size = np.sqrt(micro.domain_measure / micro.n_grains)
            transport = Transport.tungsten(
                T=args.temperature, crystal_anisotropy=args.crystal_anisotropy
            )
            print(transport.report(grain_size=grain_size))
            print(micro.report())
            ident = identify(
                micro,
                transport,
                window_fraction=args.window_fraction,
                export_prefix=(
                    OUTPUT_DIR / f"corrector_size_{size}_seed_{seed}"
                    if args.export
                    else None
                ),
            )
            print(ident.report())
            print(flush=True)
            results.append(asdict(ident))
            _dump(args.out, results, sweep)

    if args.k_sweep:
        micro = make_microstructure(
            args.sizes[0],
            args.grain_size,
            args.aspect,
            args.seeds[0],
            args.cells_per_grain,
        )
        grain_size = np.sqrt(micro.domain_measure / micro.n_grains)
        print(f"exchange-rate sweep on the {1e6 * args.sizes[0]:.1f} um cell")
        for k in args.k_sweep:
            transport = Transport.tungsten(
                T=args.temperature, k=k, crystal_anisotropy=args.crystal_anisotropy
            )
            ident = identify(
                micro, transport, window_fraction=args.window_fraction, verbose=False
            )
            D = np.asarray(ident.D_window)
            # A large mismatch signals dual-porosity, not a usable ``D_eff``.
            verdict = "ok" if ident.equilibrium_error < 0.05 else "NO SINGLE D_eff"
            print(
                f"  k = {k:9.3e} m/s  R_int/R_grain = "
                f"{fm.materials.interface_resistance_ratio(k, grain_size, transport.D_bulk):9.3e}  "  # noqa: E501
                f"Dxx/D_b = {D[0, 0] / ident.D_bulk:8.3f}  "
                f"Dyy/D_b = {D[1, 1] / ident.D_bulk:8.3f}  "
                f"eq.err = {ident.equilibrium_error:.2e}  {verdict}",
                flush=True,
            )
            sweep.append(asdict(ident))
            _dump(args.out, results, sweep)

    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
