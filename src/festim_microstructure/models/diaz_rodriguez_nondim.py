"""Díaz-Rodríguez et al. (2022) OKMC permeation setup, as a FESTIM continuum model.

Their box: a single columnar grain, L x L lateral, d deep, with the four
*lateral* faces defined as grain boundary. Permeation runs along the column
axis (z), from a charged surface at z=0 to a sink at z=d. L = 50, 100, 150 nm;
d = 50, 100, 200 nm. They could not reach the experimental d = 2 um because
the OKMC times were prohibitive (their Sect. 3.3); the continuum model has no
such cost, so the same script can be pointed at the experimental thickness
once it reproduces their boxes.

WHICH FIGURE, WHICH INLET
-------------------------
The paper's two OKMC studies use different inlet conditions and different
GB capacities, and the script reproduces the one that INLET_MODE selects:

  INLET_MODE = "concentration"  ->  their Fig. 7
      constant 5.6 H/nm^3 in the 0.4 nm near-surface region (Fig. 7 caption),
      maximum GB occupation 6.25 H/nm^2, boxes d x L x L = 50/100/200 x 150 x
      150 nm, temperature swept over 1000/T = 1.3-1.75. Output: permeability
      against 1000/T, with the GB-free (monocrystalline) reference and their
      experimental fit, Eq. (6).

  INLET_MODE = "flux"           ->  their Fig. 8
      constant flux of 1e23 H2 m^-2 s^-1 at 705 K (Fig. 8 caption), all nine
      boxes d in {50, 100, 200} x L in {50, 100, 150} nm, and the maximum GB
      occupation swept over 0.25, 1 and 4 H/nm^2 (their legend). Output:
      J_GB/J against L/d, one marker series per occupation limit.

Each run is a Box (L, d, Gamma_max); nothing about the geometry or the GB
capacity is a module constant any more.

GRAIN BOUNDARY MANIFOLDS
------------------------
Two codim-1 manifolds carry the GB network: the *interior* walls between
columns (present only when NX > 1 or NY > 1) and the *outer* lateral faces of
the box (OUTER_WALLS). With only the interior walls, the GB length per unit
area is (2 - 1/NX - 1/NY)/L, so a 1x2 array has 1/4 of the tiling's GB
density and a 2x2 array 1/2; the edge columns see fewer walls than the
interior ones and NX-invariance does NOT hold. With the outer faces declared
as half walls (OUTER_WALLS = "half") every column has two full-wall
equivalents, the density is exactly 2/L for any NX, NY, and the regular
tiling is a genuine *verification* run: if the result moves when NX changes,
something is wrong with the coupling or the facet location. OUTER_WALLS =
"full" is the paper's box: four one-sided faces of full capacity, density
4/L, meant for NX = NY = 1, and is what the two figure studies use. The
outer faces are always grain boundary; there is no option to leave them
reflecting, since that geometry corresponds to nothing in the paper.

The two manifolds are not joined at the lines where an interior wall meets
the outer face; FESTIM refuses a manifold that mixes interior and exterior
facets, and a junction condition is not written. For the regular tiling this
is exact: every wall has the same theta(z) by symmetry, so no current would
cross the junction anyway. With DISORDER=1 it is an approximation.

Set DISORDER=1 to make an array non-trivial -- it draws a per-boundary GB
migration energy from the 0.191-0.547 eV range Wei et al. (2026,
doi:10.1016/j.ijhydene.2025.152835) report across eight tungsten GBs, which
spans a factor of ~4e3 in D_gb at 500 K. That is the regime no single-column
OKMC box can reach and is the actual reason to run 10,000 grains.

PARAMETERS AND WHERE THEY COME FROM
-----------------------------------
From the paper:
  E_m,bulk = 0.20 eV, E_m,GB = 0.12 eV, E_bind = 1.05 eV   (DFT; Sect. 3.2, Fig. 3)
  max GB occupation 0.25-6.25 H/nm^2; 6.25 in Fig. 7, 0.25/1/4 in Fig. 8
                                                            (Sect. 2.4, captions)
  charged surface 5.6 H/nm^3 = 5.6e27 m^-3 over a 0.4 nm near-surface region
                                                            (Fig. 7 caption)
  constant-flux runs at 1e23 H2 m^-2 s^-1 at 705 K            (Fig. 8, Fig. 9)
  lattice constant 3.165 A                                  (Sect. 2.3)
  experimental permeability fit, Eq. (6):
      phi_NW = 1.74e15 exp(-0.569 eV/kT)  H2 m^-1 s^-1 Pa^-1/2
  temperatures 520-705 K (experiment); OKMC shown for 1000/T = 1.3-1.75

Esteban et al. (2001) solubility, used to convert the surface concentration
to a pressure (their Sect. 3.3). This is Esteban's *lattice* Sieverts'
constant for protium (Esteban Sect. 5, Table 2):
  K_s = 2.90e-2 mol m^-3 Pa^-1/2 exp(-26.9 kJ/mol / RT)
      = 1.75e22 exp(-0.279 eV/kT)  H m^-3 Pa^-1/2
Diaz-Rodriguez say they used it "considering traps", i.e. Esteban's
K_s,eff = K_s [1 + (N_t/N_l) exp(E_t/RT)] with N_t = 2.0e23 m^-3 and
E_t = 86.6 kJ/mol. That cannot be what produced their Fig. 7: K_s,eff has a
negative apparent activation energy below ~870 K (the 0.90 eV trap term
beats the 0.28 eV lattice term) and would make the GB-free permeability
rise toward low temperature, whereas their monocrystalline line falls with
a slope consistent with E_s + E_D = 0.48 eV. Neither constant reproduces
the 0.1-1 GPa they quote for 5.6 H/nm^3 (lattice: ~1e15 Pa; effective:
1e11-1e14 Pa). The lattice constant is also the physically consistent one
for this model, whose C is a mobile lattice concentration with the GB
inventory carried explicitly by Gamma.

NOT in the paper, assumed here and flagged:
  * Arrhenius prefactors. The paper says the parametrisation is that of its
    ref. [9] and does not restate it. D0_BULK = 1.9e-7 m^2/s is Fernandez et
    al. 2015 (their ref. [53]), whose barrier (0.20 eV) is exactly their
    E_m,bulk. D0_GB is taken as 1.5 x D0_BULK, the ratio of the 2D to the 3D
    random-walk prefactor (1/4 vs 1/6) at equal jump distance and attempt
    frequency -- they state explicitly that GB migration is 2D and bulk
    migration 3D. See PERMEABILITY for what the Fig. 7 monocrystalline line
    says about D0_BULK.
  * E_FORWARD, the bulk->GB approach barrier: Zhou et al. 2010 (their ref.
    [18]) give 0.13-0.16 eV.
  * The bulk site density N_b in the isotherm. bcc has 2 atoms per cubic cell
    and 6 tetrahedral sites per atom, i.e. 12 per cell, so N_b = 12/a^3 =
    3.8e29 m^-3. H occupies tetrahedral sites in W (Heinola & Ahlgren 2010,
    their ref. [54]).

  39 H/nm^2 is NOT the trap density. That is the 0 K DFT filling limit of
  their Fig. 6, which they say explicitly is not reached at temperature and is
  not what the OKMC used.

THE GB VARIABLE IS AREAL; THERE IS NO SLAB WIDTH
------------------------------------------------
The GB species Gamma is an areal concentration (m^-2). That matches both the
paper's GB (a discrete plane with an areal site limit, Sect. 2.4) and the
codim-1 formulation on FESTIM PR #1216, in which a ParticleSource on a
manifold is integrated over the manifold measure with no thickness factor and
is the same quantity as the normal flux leaving the bulk (the PR's reference
test: K_left=2, K_right=3, D=1.5 -> c = 14/9, 8/9, 4/9 only closes with that
convention). The capacity is the paper's areal limit directly and the GB
current per unit length is D_gb * grad(Gamma). No slab width appears anywhere.

THE COUPLING LAW
----------------
One reversible channel, H_b + V_gb <=> H_gb + V_b, with mass-action rates
on both sites. Per unit GB area and per side,

    J = K c_b (1 - Gamma/Gamma_max)  -  k_r Gamma (1 - c_b/N_b)

K [m/s] is kinetic, lambda nu exp(-E_forward/kT). k_r [1/s] is fixed by
detailed balance (the ratio K/k_r is thermodynamic, the magnitude is not;
Delaporte-Mathurin & Dark 2026, Eq. 8) so that J = 0 is the McLean/Oriani
site-fraction isotherm (McLean 1957; Oriani, Acta Metall. 18, 147, 1970)

    theta/(1-theta) = [theta_b/(1-theta_b)] exp(E_bind/kT),   theta_b = c_b/N_b,

which gives k_r = K/ell_s with ell_s = (Gamma_max/N_b) exp(E_bind/kT), a
segregation length [m]. The (1 - theta_b) factor is the empty-lattice-site
activity; dropping it (the dilute-lattice limit) shifts the equilibrium
theta/(1-theta) by a relative theta_b, which is <= 0.015 at the inlet in
"concentration" mode and ~1e-8 to 1e-5 in the other two. It is kept because
it costs nothing and makes the code the isotherm the header names. N_b is
the tetrahedral-site density (12 per bcc cell); the interstitial site count
is what theta_b is a fraction of, so this is the same N_b as in ell_s.

An interior wall receives 2J (two faces) and the bulk loses the same; FESTIM
applies the wall flux BC once per facet (restriction_of() returns "+" for a
manifold inside a single volume subdomain), hence N_FLUX_SIDES = 2 there,
confirmed by the mass-balance check ((in-out)/in = -3e-5). An outer face has
one grain side in the domain, so its bulk BC carries a factor 1 and is
integrated with ds (restriction None). What the outer GB does with that J
depends on OUTER_WALLS:

  "half": the face is a wall shared with a mirror column, capacity Gamma_max/2
          and fed with J from this side. Its site fraction then obeys
          d(theta)/dt = J/(Gamma_max/2) = 2J/Gamma_max -- the SAME equation
          as an interior wall -- and its along-GB current is half a wall's,
          so it is credited to the column with weight 1/2.
  "full": the paper's box face, capacity Gamma_max fed from one side:
          d(theta)/dt = J/Gamma_max, source factor 1, current weight 1.

Either way the blocking factors, the mouth value and D_gb are site-fraction
quantities and are unchanged.

K IS CAPPED
-----------
K = lambda nu exp(-E_forward/kT) is the bulk->GB approach hop with its own
barrier. Delaporte-Mathurin & Dark (Sect. 2.3) note that k+ = D/lambda when
the interface plane is just another lattice plane; a distinct barrier is
the general case their Appendix A describes (k+- built from the individual
hop rates of a distinct site), and the two agree within a factor of a few
here (D_b/lambda = 63 m/s, this form 103 m/s, at 705 K). Detailed balance
is unaffected either way: K/k_r = ell_s whatever K is.

The physical K (~60 m/s at 570 K) exceeds the rate at which diffusion can feed
the wall, D_b/h, by orders of magnitude. K is capped at DA_H_MAX * D_b/h with
h = L/CPG the wall-normal cell size. In "concentration" mode the forward
term saturates the GB within a fraction of L whatever the cap and the cap is
immaterial; in "flux" mode the GB is dilute, the bulk-to-GB transfer is
capture-limited, and the answer moves with K (roughly as its square root)
until Da = K L/D_b is large. Sweep DA_H_MAX and report Da alongside any
Fig. 8 comparison.

THE SOLVE IS DIMENSIONLESS
--------------------------
    x = L xhat,   t = (L^2/D_b) that,   c_b = C u,   Gamma = Gamma_max theta

    bulk:  du/dt = lap(u)
           flux BC on the walls, per side:
               -(Da_f u (1-theta) - Da_r theta (1 - eps u))
    GB:    dtheta/dt = (D_gb/D_b) lap(theta)
               + n_s (Ph_f u (1-theta) - Ph_r theta (1 - eps u))
           n_s = 2 on interior walls and "half" outer walls, 1 on "full"
           outer walls (see THE COUPLING LAW)

    Da_f = K L / D_b                  Ph_f = K C L^2 / (D_b Gamma_max)
    Da_r = k_r Gamma_max L / (D_b C)  Ph_r = k_r L^2 / D_b
    eps  = C / N_b                    (lattice site fraction at u = 1)

Da_f/Da_r = Ph_f/Ph_r = beta, so at u = 1 the wall is in equilibrium at
theta/(1-theta) = beta/(1-eps), and that is the value imposed at the mouths.

The independent groups are {Da_f, beta, eps, Lambda, D_gb/D_b, d/L} with
beta = eps exp(E_bind/kT) (inlet GB occupancy theta/(1-theta), dilute form)
and Lambda = C L/Gamma_max (bulk inventory across one column per unit GB
capacity). Lambda decides whether the GB can dominate: once saturated the
walls carry at most (GB length/area)(D_gb/D_b)(Gamma_max/C) relative to
the bulk, i.e. 4 (D_gb/D_b)/Lambda for the paper's box. The concentration
scale C depends on INLET_MODE. Currents convert back with j_b = D_b C L
jhat_b and j_gb = w D_b Gamma_max jhat_gb (atoms/s), with w = 1 for interior
walls and w = OUTER_CURRENT_WEIGHT for outer walls.

FLUX SPLIT
----------
The exit-plane J_GB/J is a last-cell quantity: with the drains held at
theta = 0 next to a loaded GB, the drop from theta ~ 1 to 0 happens in the
last cell and the GB "current" there is D_gb Gamma_max/dz, i.e. bulk hydrogen
captured in the last cell. Their Fig. 8 is defined the same way -- "the flux
that escapes from the GB at the end of the simulation box" -- and they warn
that it "may lead to misguiding interpretations" (Sect. 3.3, p. 1085), so it
is the quantity the Fig. 8 study plots, with the axial cell size DZ held
fixed in nm across boxes so the layer it measures is the same physical
thickness everywhere. The split is also evaluated on interior planes
(F_GB_PLANES, snapped to mesh node planes) and printed per run; those are
the transport measure. They are not drawn on the Fig. 8 reproduction, which
shows only the quantity the paper defines.

INLET CONDITION (INLET_MODE)
----------------------------
"concentration": u = 1 with C = 5.6e27 m^-3. Their Fig. 7 method (constant
    concentration in the near-surface region). Lambda ~ 130 for L = 150 nm:
    the GB saturates and the bulk carries most of the transport, so the exit
    fraction f_GB is high while D_eff/D_b is small -- the interpretation trap
    they describe for their 50x150x150 box (Sect. 3.3, p. 1085).

"flux": Neumann inlet, phi_imp into the bulk face, GB mouths no-flux,
    reflecting front face, Dirichlet 0 at the back. This is the OKMC
    constant-flux setup: no surface model (Sect. 3.3), so the whole phi_imp
    permeates, and Fig. 9's in-grain surface values (~0.005-0.01 H/nm^3 for
    d = 200 nm at 705 K) are of order phi_imp d/D_b. C = phi_imp d/D_b (the
    pure-Fick surface value), so u(0) <= 1 and the imposed dimensionless
    flux is 1/D_HAT. The inlet concentration is read off the solution.

PERMEABILITY
------------
Their Eq. (1): J = phi P^(1/2)/d, with P the pressure in equilibrium with the
inlet concentration through Esteban's solubility, P = (C_in/S)^2. Fig. 7 is
in H2 m^-1 s^-1 Pa^-1/2, so J_H2 = J_H/2 and

    phi = J_H2 d / sqrt(P) = S D_eff / 2,    phi_mono = S D_b / 2.

The absolute level is therefore S x D0_BULK and the GB effect is D_eff/D_b.
phi_mono is the pure-Fick reference for their dashed monocrystalline line;
with Esteban's lattice K_s and D0_BULK = 1.9e-7 it lies ~15x above that line,
whereas D0 = lambda^2 nu/6 = 2.1e-8 (the natural OKMC hop prefactor) lands
within a factor ~2. D0_BULK is the knob that line calibrates.

The equilibrium pressure for 5.6 H/nm^3 with this S is ~1e15 Pa, not the
0.1-1 GPa the paper quotes (Sect. 3.3); see PARAMETERS for why neither of
Esteban's constants gives their number. P_eq is printed per run.

OUTPUTS
-------
"concentration":  diaz-fig7-<outer>.png
"flux":           diaz-fig8-<outer>.png
"""

from dataclasses import dataclass

from mpi4py import MPI

import dolfinx
import festim as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ufl

# tiling and mesh
# ----------------------------------------------------------------------------
NX, NY = 1, 1  # columns per side; 1 x 1 with OUTER_WALLS = "full" is their box
CPG = 4  # mesh cells across one column, per axis
DZ = 2.5e-9  # axial cell size, m, the same in every box (header, FLUX SPLIT)
NZ_MIN = 10

# the outer lateral faces of the box: "half" (periodic-tiling equivalent,
# density 2/L for any NX, NY) or "full" (the paper's one-sided box, 4/L, use
# with NX = NY = 1). See the header. The interior walls exist only when
# NX > 1 or NY > 1.
OUTER_WALLS = "full"
OUTER_SOURCE_FACTOR = {"half": 2.0, "full": 1.0}
OUTER_CURRENT_WEIGHT = {"half": 0.5, "full": 1.0}

# interior planes (fractions of d) on which the flux split is evaluated
F_GB_PLANES = (0.25, 0.5, 0.75)

LX, LY = NX, NY  # in units of L
# the mesh FESTIM sees is in units of L
L_HAT = 1.0
LX_HAT, LY_HAT = NX * L_HAT, NY * L_HAT

# material and coupling parameters (SI)
# ----------------------------------------------------------------------------
E_M_BULK = 0.20  # eV, DFT, their Fig. 3
E_M_GB = 0.12  # eV, DFT, their Fig. 3
E_BIND = 1.05  # eV, DFT, H binding to an empty GB
E_FORWARD = 0.145  # eV, bulk->GB approach barrier; Zhou 2010 gives 0.13-0.16

D0_BULK = 1.9e-7  # m^2/s, Fernandez 2015; see PERMEABILITY in the header
D0_GB = 1.5 * D0_BULK  # m^2/s, assumed: 2D vs 3D random-walk prefactor

A_LAT = 3.165e-10  # m, lattice constant (Sect. 2.3)
N_B_SITES = 12.0 / A_LAT**3  # 3.8e29 m^-3, tetrahedral sites: 12 per bcc cell

LAMBDA_JUMP = 1.12e-10  # m, tetrahedral-site hop in bcc W, for K
NU_ATTEMPT = 1e13  # s^-1
DA_H_MAX = 100.0  # cap on K h / D_b (see header); sweep it in "flux" mode

# Esteban et al. 2001 lattice Sieverts' constant, protium (their Sect. 5):
# 2.90e-2 mol m^-3 Pa^-1/2 exp(-26.9 kJ/mol / RT), in H m^-3 Pa^-1/2 and eV
S0_ESTEBAN = 2.90e-2 * 6.02214e23  # = 1.75e22
E_S_ESTEBAN = 26.9e3 / 96485.3  # = 0.279 eV

# their experimental permeability fit, Eq. (6), H2 m^-1 s^-1 Pa^-1/2
PHI0_EXP = 1.74e15
EA_EXP = 0.569  # eV

# inlet
INLET_MODE = "concentration"  # "concentration" | "flux"
C_CONC = 5.6e27  # m^-3, their 5.6 H/nm^3 (Fig. 7 caption)
PHI_IMP = 2.0 * 1e23  # H m^-2 s^-1, their 1e23 H2 m^-2 s^-1 (Fig. 8 caption)

N_FLUX_SIDES = 2  # interior walls: two grain faces, BC applied once per facet

# the two studies (header, WHICH FIGURE, WHICH INLET). Boxes are (d, L) in m.
FIG7_TEMPERATURES = [770.0, 705.0, 655.0, 615.0, 571.0]  # 1000/T = 1.3-1.75
FIG7_BOXES = [(50e-9, 150e-9), (100e-9, 150e-9), (200e-9, 150e-9)]
FIG7_GAMMA_MAX = 6.25e18  # m^-2

FIG8_TEMPERATURE = 705.0
FIG8_BOXES = [(d, L) for d in (50e-9, 100e-9, 200e-9) for L in (50e-9, 100e-9, 150e-9)]
FIG8_GAMMA_MAX = [0.25e18, 1.0e18, 4.0e18]  # m^-2, their legend

# time, in units of L^2/D_b. Steady state along the column needs t >> D_HAT^2;
# the GB reverse channel needs t >> 1/(2 Ph_r), which the mass-balance check
# reports on.
T_END_PER_DHAT2 = 10.0
T_END_MIN = 500.0
DT0_HAT = 1e-3  # first step must not swallow the whole inlet jump
DT_STEPS_MIN = 100  # DT_MAX = T_END / this

DISORDER = 0
E_M_GB_RANGE = (0.191, 0.547)  # eV, Wei et al. 2026, eight tungsten GBs
SEED = 0


@dataclass(frozen=True)
class Box:
    """One simulation box: column side L, depth d, GB capacity Gamma_max (SI)."""

    L: float
    d: float
    gamma_max: float

    @property
    def d_hat(self):
        return self.d / self.L

    @property
    def nz(self):
        return max(NZ_MIN, int(round(self.d / DZ)))

    @property
    def h_wall(self):
        """Wall-normal cell size, m; sets the K cap."""
        return self.L / CPG

    @property
    def area(self):
        return NX * self.L * NY * self.L

    @property
    def label(self):
        return f"{self.d * 1e9:.0f}x{self.L * 1e9:.0f}x{self.L * 1e9:.0f}"


def segregation_length(T, box):
    """ell_s [m]: K/k_r. Equals Gamma/c_b at equilibrium in the dilute limit;
    the McLean prefactor N_s/N_b times exp(E_bind/kT). Unchanged by the
    reverse blocking factor, which enters the rate law and not this ratio."""
    return (box.gamma_max / N_B_SITES) * np.exp(E_BIND / (F.k_B * T))


def k_forward(T, D_b, box):
    """Absorption coefficient K [m/s]. Kinetic, from E_forward; capped, see header."""
    k_phys = LAMBDA_JUMP * NU_ATTEMPT * np.exp(-E_FORWARD / (F.k_B * T))
    return min(k_phys, DA_H_MAX * D_b / box.h_wall)


def solubility(T):
    """Esteban Sieverts constant, H m^-3 Pa^-1/2."""
    return S0_ESTEBAN * np.exp(-E_S_ESTEBAN / (F.k_B * T))


def inlet_scale(T, D_b, box):
    """Concentration scale C [m^-3] for the chosen INLET_MODE (header)."""
    if INLET_MODE == "concentration":
        return C_CONC
    if INLET_MODE == "flux":
        return PHI_IMP * box.d / D_b  # pure-Fick surface value; u(0) <= 1
    raise ValueError(INLET_MODE)


def make_mesh(box, comm=None):
    """A structured box, in units of L, whose node planes fall on the walls."""
    comm = MPI.COMM_WORLD if comm is None else comm
    mesh = dolfinx.mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([LX_HAT, LY_HAT, box.d_hat])],
        [NX * CPG, NY * CPG, box.nz],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    if comm.rank == 0:
        ntet = NX * CPG * NY * CPG * box.nz * 6
        print(
            f"box {box.label} nm, {NX * NY} column(s), d/L = {box.d_hat:.2f}, "
            f"NZ = {box.nz}, ~{ntet:.2e} tetrahedra"
        )
    return mesh


class ColumnWalls(F.VolumeSubdomain):
    """The whole GB network as one codim-1 subdomain: interior facets, dim=2.

    The x-normal and y-normal wall sets are located separately and unioned.
    Locating them with a single predicate would also catch facets that have
    some vertices on an x-wall and the rest on a y-wall, which near a column
    corner lie on no wall at all.
    """

    def __init__(self, id, material):
        super().__init__(id=id, material=material, dim=2)
        self.entity_axis = None  # 0 or 1 per located facet, for the D_gb field

    def locate_subdomain_entities(self, mesh):
        tol = 0.05 * L_HAT / CPG

        def on_walls(axis):
            extent = LX_HAT if axis == 0 else LY_HAT

            def pred(x):
                r = np.remainder(x[axis] + tol, L_HAT)
                interior = (x[axis] > tol) & (x[axis] < extent - tol)
                return (r < 2 * tol) & interior

            return pred

        fx = dolfinx.mesh.locate_entities(mesh, 2, on_walls(0))
        fy = dolfinx.mesh.locate_entities(mesh, 2, on_walls(1))
        self.entity_axis = np.concatenate(
            [np.zeros(fx.size, dtype=np.int8), np.ones(fy.size, dtype=np.int8)]
        )
        return np.concatenate([fx, fy]).astype(np.int32)


class OuterWalls(F.VolumeSubdomain):
    """The four lateral faces of the box as one codim-1 subdomain on the
    domain boundary. Each face is located on its own and the sets unioned,
    for the same reason as in ColumnWalls: a single predicate on
    (x in {0, LX}) | (y in {0, LY}) would also catch the diagonal facets of
    the corner cells, whose three vertices sit on two different faces, and
    FESTIM would then (rightly) refuse the manifold as mixing interior and
    exterior facets.
    """

    def __init__(self, id, material):
        super().__init__(id=id, material=material, dim=2)

    def locate_subdomain_entities(self, mesh):
        tol = 0.05 * L_HAT / CPG
        planes = [(0, 0.0), (0, LX_HAT), (1, 0.0), (1, LY_HAT)]
        found = []
        for axis, value in planes:
            found.append(
                dolfinx.mesh.locate_entities_boundary(
                    mesh, 2, lambda x, a=axis, v=value: np.isclose(x[a], v, atol=tol)
                )
            )
        return np.unique(np.concatenate(found)).astype(np.int32)


def gb_diffusivity_field(network, T, D_b):
    """Per-boundary D_gb/D_b as a DG0 field on the network submesh.

    The coefficient has to live on the one submesh, because splitting the
    network into two subdomains would disconnect it and reintroduce junction
    conditions. Here the per-boundary label is a random draw rather than a
    measured disorientation -- swap in theta when you point this at a real
    map. The value is dimensionless (divided by D_b) because the solve is.
    """
    submesh = network.submesh

    V = dolfinx.fem.functionspace(submesh, ("DG", 0))
    d = dolfinx.fem.Function(V, name="D_gb")
    n_local = submesh.topology.index_map(submesh.topology.dim).size_local

    # one barrier per wall, not per element: label walls by (axis, index) so a
    # whole boundary plane shares a value, which is what a boundary character
    # means. Facet midpoints give the wall index.
    mid = dolfinx.mesh.compute_midpoints(
        submesh, submesh.topology.dim, np.arange(n_local, dtype=np.int32)
    )
    ix = np.rint(mid[:, 0] / L_HAT).astype(int)
    iy = np.rint(mid[:, 1] / L_HAT).astype(int)
    on_x = np.isclose(mid[:, 0] / L_HAT, ix, atol=0.1)
    wall_id = np.where(on_x, ix, 10_000 + iy)

    rng = np.random.default_rng(SEED)
    lo, hi = E_M_GB_RANGE
    barrier = rng.uniform(lo, hi, size=20_001)[wall_id]
    d.x.array[:n_local] = D0_GB * np.exp(-barrier / (F.k_B * T)) / D_b
    d.x.scatter_forward()
    return d


# ----------------------------------------------------------------------------
# flux accounting
# ----------------------------------------------------------------------------
def _plane_measure(c, z, interior):
    """Measure over the entities of c's own mesh lying in the plane z: the
    exterior-facet measure ds when z is the inlet or outlet plane, the
    interior-facet measure dS otherwise (then integrands must be restricted)."""
    m = c.function_space.mesh
    fdim = m.topology.dim - 1
    marker = lambda x: np.isclose(x[2], z, atol=1e-10)  # noqa: E731
    if interior:
        facets = dolfinx.mesh.locate_entities(m, fdim, marker)
    else:
        facets = dolfinx.mesh.locate_entities_boundary(m, fdim, marker)
    tags = dolfinx.mesh.meshtags(
        m, fdim, np.sort(facets), np.ones(facets.size, dtype=np.int32)
    )
    return ufl.Measure("dS" if interior else "ds", domain=m, subdomain_data=tags)(1)


def axial_current(c, D_hat, z, interior=False):
    """Dimensionless current in +z through the plane z, assembled.

    Works for both the 3D bulk field (an area integral over the plane) and the
    2D GB submesh field (a line integral over the edge of the submesh in that
    plane), because in each case it is the facet measure of that field's own
    mesh restricted to the plane. Positive means flow towards the outlet at
    every plane, so inlet, interior and outlet values can be compared. For an
    interior plane the gradient is single-valued across the facet (same
    function on both sides) and the "+" restriction is arbitrary.
    """
    dm = _plane_measure(c, z, interior)
    # the submesh of a plane set has no meaningful outward normal in 3D, so
    # take the axial component of the gradient directly rather than dot(.., n)
    g = ufl.grad(c)[2]
    form = -D_hat * (g("+") if interior else g) * dm
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))
    return c.function_space.mesh.comm.allreduce(local, op=MPI.SUM)


def plane_mean(c, z, area_hat):
    """Mean of a 3D field over the boundary plane z (dimensionless area area_hat)."""
    ds = _plane_measure(c, z, interior=False)
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(c * ds))
    return c.function_space.mesh.comm.allreduce(local, op=MPI.SUM) / area_hat


def run(T, box, mesh=None):
    """One steady permeation solve for temperature T and geometry/capacity box."""
    if mesh is None:
        mesh = make_mesh(box)
    comm = mesh.comm
    kT = F.k_B * T
    L, d_hat, gamma_max = box.L, box.d_hat, box.gamma_max
    D_b = D0_BULK * np.exp(-E_M_BULK / kT)
    D_g = D0_GB * np.exp(-E_M_GB / kT)
    ell = segregation_length(T, box)
    K = k_forward(T, D_b, box)
    kr = K / ell
    tau = L**2 / D_b
    C = inlet_scale(T, D_b, box)
    S = solubility(T)

    # dimensionless groups (header, THE SOLVE IS DIMENSIONLESS)
    Da_f = K * L / D_b
    Da_r = kr * gamma_max * L / (D_b * C)
    Ph_f = K * C * L**2 / (D_b * gamma_max)
    Ph_r = kr * L**2 / D_b
    D_g_hat = D_g / D_b
    eps = C / N_B_SITES  # lattice site fraction at u = 1
    beta = eps * np.exp(E_BIND / kT)  # inlet theta/(1-theta), dilute form
    Lambda = C * L / gamma_max
    if eps >= 1.0:
        raise ValueError(
            f"C = {C:.2e} m^-3 exceeds the lattice site density {N_B_SITES:.2e}; "
            "the site-fraction isotherm is meaningless here"
        )
    # the wall's equilibrium at u = 1 with the (1 - eps u) reverse blocking:
    # Da_f (1-theta) = Da_r theta (1-eps)  ->  theta/(1-theta) = beta/(1-eps).
    # This is what the mouths must be held at, or the inlet line is not an
    # equilibrium point of the coupling law it sits on.
    beta_eff = beta / (1.0 - eps)
    theta_surf = beta_eff / (1.0 + beta_eff)

    t_end_hat = max(T_END_MIN, T_END_PER_DHAT2 * d_hat**2)

    if comm.rank == 0:
        print(
            f"T={T:.0f} K  box {box.label}  Gmax={gamma_max * 1e-18:.3g} /nm^2  "
            f"D_b={D_b:.2e}  D_gb/D_b={D_g_hat:.2f}  "
            f"K={K:.2e} m/s (Da_f={Da_f:.1f})  k_r={kr:.2e} 1/s  tau={tau:.2e} s  "
            f"C={C:.2e} m^-3 ({C * 1e-27:.2e} H/nm^3)"
        )
        print(
            f"  Da_r={Da_r:.2e}  Ph_f={Ph_f:.2e}  Ph_r={Ph_r:.2e}  "
            f"beta={beta:.2e}  eps={eps:.2e}  Lambda={Lambda:.2e}  S={S:.2e}  "
            f"t_end={t_end_hat:.0f} (1/(2 Ph_r)={0.5 / Ph_r:.0f})"
        )

    grains = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1.0, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    inlet = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[2], 0.0))
    outlet = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[2], d_hat))
    # the lines where the walls meet the charged face and the sink face. One
    # object serves both manifolds: FESTIM resolves which manifold a codim-2
    # surface bounds through the species of the condition using it.
    mouths = F.SurfaceSubdomain(id=5, dim=1, locator=lambda x: np.isclose(x[2], 0.0))
    drains = F.SurfaceSubdomain(id=6, dim=1, locator=lambda x: np.isclose(x[2], d_hat))

    u = F.Species("u", subdomains=[grains])  # c_b / C

    # the GB network as (manifold, species, bulk-side factor, source factor,
    # current weight) per manifold present. Each GB species lives on exactly
    # one manifold, which is what lets mouths/drains be reused.
    walls = []
    if NX > 1 or NY > 1:
        network = ColumnWalls(id=2, material=F.Material(D_0=D_g_hat, E_D=0.0))
        th_int = F.Species("theta_int", subdomains=[network])  # Gamma / gamma_max
        walls.append((network, th_int, float(N_FLUX_SIDES), 2.0, 1.0))
    outer = OuterWalls(id=7, material=F.Material(D_0=D_g_hat, E_D=0.0))
    th_ext = F.Species("theta_ext", subdomains=[outer])
    walls.append(
        (
            outer,
            th_ext,
            1.0,  # one grain face in the domain: ds, applied once
            OUTER_SOURCE_FACTOR[OUTER_WALLS],
            OUTER_CURRENT_WEIGHT[OUTER_WALLS],
        )
    )

    # GB length per unit area credited to the columns, in units of 1/L
    gb_density_hat = OUTER_CURRENT_WEIGHT[OUTER_WALLS] * 2 * (NX + NY) / (NX * NY)
    if NX > 1 or NY > 1:
        gb_density_hat += ((NX - 1) * NY + (NY - 1) * NX) / (NX * NY)

    wall_bcs, gb_sources, drain_bcs, mouth_bcs = [], [], [], []
    for manifold, th, n_bulk, n_src, _w in walls:
        wall_bcs.append(
            F.ParticleFluxBC(
                subdomain=manifold,
                species=u,
                # the bulk loses J per grain face: forward capture blocked by
                # GB occupancy, reverse release blocked by lattice occupancy.
                # Keep this expression and the GB source below textually
                # identical up to sign and the side factors -- that is what
                # makes the exchange conservative.
                value=lambda cb, cg, n=n_bulk: (
                    n * (Da_r * cg * (1.0 - eps * cb) - Da_f * cb * (1.0 - cg))
                ),
                species_dependent_value={"cb": u, "cg": th},
            )
        )
        gb_sources.append(
            F.ParticleSource(
                # the same J, as a rate of change of the site fraction
                value=lambda cb, cg, n=n_src: (
                    n * (Ph_f * cb * (1.0 - cg) - Ph_r * cg * (1.0 - eps * cb))
                ),
                species=th,
                volume=manifold,
                species_dependent_value={"cb": u, "cg": th},
            )
        )
        drain_bcs.append(
            F.FixedConcentrationBC(subdomain=drains, value=0.0, species=th)
        )
        mouth_bcs.append(
            F.FixedConcentrationBC(subdomain=mouths, value=theta_surf, species=th)
        )

    sink_bcs = [
        F.FixedConcentrationBC(subdomain=outlet, value=0.0, species=u),
        *drain_bcs,
    ]
    if INLET_MODE == "flux":
        # reflecting front face carrying the whole implanted flux; the GB
        # mouths get the natural no-flux condition
        inlet_bcs = [F.ParticleFluxBC(subdomain=inlet, species=u, value=1.0 / d_hat)]
    else:
        # Dirichlet on the bulk face and the isotherm value at the mouths, so
        # the mouth condition is the equilibrium one rather than an arbitrary
        # number
        inlet_bcs = [
            F.FixedConcentrationBC(subdomain=inlet, value=1.0, species=u),
            *mouth_bcs,
        ]

    petsc_options = {
        "pc_factor_mat_solver_type": "superlu_dist",
        "snes_linesearch_type": "bt",
    }
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[u, *[w[1] for w in walls]],
        subdomains=[grains, *[w[0] for w in walls], inlet, outlet, mouths, drains],
        sources=gb_sources,
        boundary_conditions=[*wall_bcs, *inlet_bcs, *sink_bcs],
        temperature=T,
        settings=F.Settings(
            atol=1e-8,
            rtol=1e-8,
            transient=True,
            final_time=t_end_hat,
            max_iterations=30,
            stepsize=F.Stepsize(
                initial_value=DT0_HAT,
                growth_factor=1.2,
                cutback_factor=0.5,
                target_nb_iterations=5,
                max_stepsize=t_end_hat / DT_STEPS_MIN,
            ),
        ),
        exports=[],
        petsc_options=petsc_options,
    )
    model.initialise()
    if DISORDER:
        # the same seeded draw indexed by wall id, so a wall gets the same
        # barrier whichever manifold it belongs to
        for manifold, *_ in walls:
            manifold.material = F.Material(
                D_0=gb_diffusivity_field(manifold, T, D_b), E_D=0.0
            )
        model.initialise()

    model.run()

    cb = u.subdomain_to_post_processing_solution[grains]
    cgs = [(th.subdomain_to_post_processing_solution[m], w) for m, th, _, _, w in walls]

    # dimensionless currents, then back to atoms/s; an outer wall's current is
    # credited with its weight (header, THE COUPLING LAW)
    def gb_current(z, interior=False):
        return sum(
            w * D_b * gamma_max * axial_current(cg, D_g_hat, z, interior)
            for cg, w in cgs
        )

    jb_out = D_b * C * L * axial_current(cb, 1.0, d_hat)
    jg_out = gb_current(d_hat)
    jb_in = D_b * C * L * axial_current(cb, 1.0, 0.0)
    jg_in = gb_current(0.0)
    j_out = jb_out + jg_out
    j_in = jb_in + jg_in
    balance = (j_in - j_out) / j_in if j_in else float("nan")

    # the flux split on interior planes, snapped to mesh node planes (header,
    # FLUX SPLIT); keyed by the plane's fraction of d
    f_gb_planes = {}
    for frac in F_GB_PLANES:
        z = round(frac * box.nz) * d_hat / box.nz
        jb = D_b * C * L * axial_current(cb, 1.0, z, interior=True)
        jg = gb_current(z, interior=True)
        f_gb_planes[frac] = jg / (jb + jg) if (jb + jg) else float("nan")

    # inlet concentration: imposed in the Dirichlet modes, read off the
    # solution in flux mode
    u_in = plane_mean(cb, 0.0, LX_HAT * LY_HAT)
    C_in = C * u_in

    # how much of the network is saturated, and whether Newton overshot
    theta = np.concatenate([cg.x.array for cg, _ in cgs])
    n_loc = theta.size
    n_sat = int(np.sum(theta > 0.99))
    n_tot = mesh.comm.allreduce(n_loc, op=MPI.SUM)
    n_sat = mesh.comm.allreduce(n_sat, op=MPI.SUM)
    theta_max = mesh.comm.allreduce(float(theta.max()) if n_loc else 0.0, op=MPI.MAX)

    # their Eq. (1): D_eff = (J/A) d / C_in; phi = J_H2 d / sqrt(P) = S D_eff/2
    D_eff = (j_out / box.area) * box.d / C_in
    P_eq = (C_in / S) ** 2
    phi = (j_out / box.area / 2.0) * box.d / np.sqrt(P_eq)
    phi_mono = S * D_b / 2.0
    phi_exp = PHI0_EXP * np.exp(-EA_EXP / kT)

    return dict(
        T=T,
        inv_T_1000=1000.0 / T,
        box=box.label,
        L=L,
        d=box.d,
        L_over_d=1.0 / d_hat,
        gamma_max=gamma_max,
        inlet_mode=INLET_MODE,
        outer_walls=OUTER_WALLS,
        gb_density_hat=gb_density_hat,
        D_b=D_b,
        D_gb=D_g,
        K=K,
        k_r=kr,
        C_scale=C,
        C_in=C_in,
        beta=beta,
        Lambda=Lambda,
        Da_f=Da_f,
        Da_r=Da_r,
        Ph_f=Ph_f,
        Ph_r=Ph_r,
        S=S,
        P_eq=P_eq,
        j_bulk=jb_out,
        j_gb=jg_out,
        j_tot=j_out,
        j_in=j_in,
        mass_balance=balance,
        f_gb=jg_out / j_out if j_out else float("nan"),
        **{f"f_gb_{int(100 * k)}": v for k, v in f_gb_planes.items()},
        D_eff=D_eff,
        phi=phi,
        phi_mono=phi_mono,
        phi_exp=phi_exp,
        saturated_fraction=n_sat / n_tot if n_tot else float("nan"),
        theta_max=theta_max,
    )


def report(r, comm):
    """One block of stdout per run, plus the two sanity warnings."""
    if comm.rank != 0:
        return
    print(
        f"  -> f_GB(exit)={r['f_gb']:.4f}  D_eff/D_b={r['D_eff'] / r['D_b']:.3f}  "
        f"C_in={r['C_in'] * 1e-27:.2e} H/nm^3  P_eq={r['P_eq']:.2e} Pa\n"
        f"     phi={r['phi']:.3e}  phi_mono={r['phi_mono']:.3e}  "
        f"phi_exp={r['phi_exp']:.3e}  H2 m^-1 s^-1 Pa^-1/2\n"
        f"     sat={r['saturated_fraction']:.2f}  theta_max={r['theta_max']:.6f}  "
        f"(in-out)/in={r['mass_balance']:+.3e}\n"
        "     f_GB on interior planes: "
        + "  ".join(
            f"{int(100 * k)}%: {r[f'f_gb_{int(100 * k)}']:.4f}" for k in F_GB_PLANES
        )
    )
    if abs(r["mass_balance"]) > 1e-2:
        print("  WARNING: mass balance fails; not at steady state? raise T_END_*")
    if r["theta_max"] > 1.0 + 1e-6:
        print("  WARNING: theta exceeded 1; Newton overshot the GB capacity.")


# ----------------------------------------------------------------------------
# the studies
# ----------------------------------------------------------------------------
def study_fig7():
    """Their Fig. 7: permeability against 1000/T for the 150 nm boxes at
    constant concentration and Gamma_max = 6.25 H/nm^2."""
    comm = MPI.COMM_WORLD
    rows = []
    for d, L in FIG7_BOXES:
        box = Box(L=L, d=d, gamma_max=FIG7_GAMMA_MAX)
        mesh = make_mesh(box)
        for T in FIG7_TEMPERATURES:
            r = run(T, box, mesh)
            report(r, comm)
            rows.append(r)
    if comm.rank != 0:
        return rows

    tag = f"fig7-{OUTER_WALLS}"
    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = {"50x150x150": "o", "100x150x150": "^", "200x150x150": "v"}
    for label, mk in markers.items():
        sub = sorted(
            (r for r in rows if r["box"] == label), key=lambda r: r["inv_T_1000"]
        )
        if sub:
            ax.semilogy(
                [r["inv_T_1000"] for r in sub],
                [r["phi"] for r in sub],
                mk,
                mfc="none",
                label=f"{label} nm",
            )
    ref = sorted(rows, key=lambda r: r["inv_T_1000"])
    x = np.array([r["inv_T_1000"] for r in ref])
    ax.semilogy(x, [r["phi_mono"] for r in ref], "k--", label=r"$S D_b/2$ (no GB)")
    ax.semilogy(
        x,
        [r["phi_exp"] for r in ref],
        "-",
        color="purple",
        alpha=0.6,
        label="their Eq. (6) fit",
    )
    ax.set_xlabel("1000/T (1/K)")
    ax.set_ylabel(r"permeability (H$_2$ m$^{-1}$ s$^{-1}$ Pa$^{-1/2}$)")
    ax.set_title(
        f"const. conc. {C_CONC * 1e-27:.1f} H/nm$^3$, "
        f"$\\Gamma_{{max}}$ = {FIG7_GAMMA_MAX * 1e-18:.3g} H/nm$^2$, outer={OUTER_WALLS}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"diaz-{tag}.png", dpi=150)
    return rows


def study_fig8():
    """Their Fig. 8: J_GB/J against L/d at 705 K and constant flux, for all
    nine boxes and three GB occupation limits."""
    comm = MPI.COMM_WORLD
    rows = []
    for d, L in FIG8_BOXES:
        mesh = None
        for gmax in FIG8_GAMMA_MAX:
            box = Box(L=L, d=d, gamma_max=gmax)
            if mesh is None:  # the mesh does not depend on Gamma_max
                mesh = make_mesh(box)
            r = run(FIG8_TEMPERATURE, box, mesh)
            report(r, comm)
            rows.append(r)
    if comm.rank != 0:
        return rows

    tag = f"fig8-{OUTER_WALLS}"
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = dict(
        zip(FIG8_GAMMA_MAX, ["^", "o", "s"], strict=True)
    )  # their legend order 0.25, 1, 4
    for gmax, mk in markers.items():
        sub = sorted(
            (r for r in rows if r["gamma_max"] == gmax), key=lambda r: r["L_over_d"]
        )
        x = [r["L_over_d"] for r in sub]
        ax.plot(
            x, [r["f_gb"] for r in sub], mk, ms=7, label=f"Max. Occ. = {gmax * 1e-18:g}"
        )
    # the box labels, once per L/d, as in their figure
    seen = set()
    for r in rows:
        if r["L_over_d"] in seen:
            continue
        seen.add(r["L_over_d"])
        boxes = sorted({q["box"] for q in rows if q["L_over_d"] == r["L_over_d"]})
        ax.annotate(
            " & ".join(boxes),
            (r["L_over_d"], 0.02),
            rotation=90,
            fontsize=7,
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("ratio of lateral size to depth (L/d)")
    ax.set_ylabel(r"fraction of flux along grain boundaries ($J_{GB}/J$)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{FIG8_TEMPERATURE:.0f} K, flux {PHI_IMP / 2:.0e} H$_2$ m$^{{-2}}$ s$^{{-1}}$, "
        f"outer={OUTER_WALLS}, DA_H_MAX={DA_H_MAX:g}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"diaz-{tag}.png", dpi=150)
    return rows


STUDIES = {
    "concentration": study_fig7,
    "flux": study_fig8,
}


def main():
    """Run the study that INLET_MODE selects (header, WHICH FIGURE, WHICH INLET)."""
    mpl.use("Agg")
    if INLET_MODE not in STUDIES:
        raise ValueError(INLET_MODE)
    if OUTER_WALLS not in OUTER_SOURCE_FACTOR:
        raise ValueError(OUTER_WALLS)
    if MPI.COMM_WORLD.rank == 0:
        print(f"inlet mode: {INLET_MODE}  ->  {STUDIES[INLET_MODE].__name__}")
        print(
            f"outer walls: {OUTER_WALLS}, NX x NY = {NX} x {NY}, DA_H_MAX = {DA_H_MAX:g}"
        )
    STUDIES[INLET_MODE]()


if __name__ == "__main__":
    main()
