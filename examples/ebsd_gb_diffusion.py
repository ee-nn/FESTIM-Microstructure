"""Short-circuit diffusion through the GB network of a polycrystal measured by
EBSD, in 2D.

The mesh is Neper's direct meshing of the raster, so the grain boundaries are
the measured ones and each boundary's disorientation comes from the two grains'
measured orientations. See :mod:`festim_microstructure.meshing.ebsd`.

Every measured grain carries a lattice species of its own and exchanges with the
one network species at the rate ``k``; the network is the list of edge ids above
the disorientation threshold.

Prerequisite: the ``.tesr`` written by ``examples/ebsd_ctf_to_tesr.py`` (or by
``convert(...)``). Neper and Gmsh are found through ``FM_NEPER_BIN`` /
``FM_GMSH_BIN`` or ``PATH``.

Run::

    python examples/ebsd_gb_diffusion.py
"""

from pathlib import Path

import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem

HERE = Path(__file__).resolve().parent

# Simulation parameters
ebsd = fm.EbsdOptions(
    tesr=str(HERE / "results" / "ebsd_ctf_to_tesr" / "d7.tesr"),
    unit=1e-6,
    theta_min=10.0,
    mesh=fm.TesrMeshOptions(rcl=0.25, mesh_qual_min=0.7),
)
workdir = OUTPUT_DIR
force = True

D_B = 1e-14  # lattice diffusivity     [m^2/s]
D_GB = 1e-8  # GB diffusivity          [m^2/s]
delta = 5e-9  # GB width               [m]
k_exchange = 1e-4  # bulk <-> GB exchange [m/s]
c0 = 1.0
t_end = 36000.0
dt = 600.0
theta_dependent_D = False  # see gb_diffusivity_field, CHECK before enabling

# Build the microstructure and simulation
unit, uname = ebsd.unit, fm.meshing.ebsd.unit_name(ebsd.unit)

base = fm.meshing.ebsd.run_ebsd_pipeline(ebsd, workdir=workdir, force=force)
LX, LY = fm.meshing.ebsd.read_extent(base, unit)
mesh, cell_tags, facet_tags = fm.formats.msh4.read_mesh(base, gdim=2, unit=unit)
micro = fm.EbsdMicrostructure.from_mesh(
    base, mesh, cell_tags, facet_tags, (LX, LY), theta_min=ebsd.theta_min
)
micro.check_orientations()
fm.meshing.ebsd.write_network_png(
    base, mesh, micro, base.parent / "poly-raw.tesr", unit, uname
)

# The measured map as the model sees it: one tagged subdomain per grain, the
# network as the kept edge ids. The measured orientations drive the lattice
# tensor only when crystal_anisotropy is set away from 1, which it is not
# here, so the grains are left untextured.
poly = fm.TaggedPolycrystal(
    mesh=mesh,
    cell_tags=cell_tags,
    facet_tags=facet_tags,
    gb_tag=micro.network_ids,
    name=f"EBSD map {Path(ebsd.tesr).name}",
)
physics = fm.Physics(
    T=500.0,  # with E_D = 0 on both phases, nothing depends on it
    D_0_bulk=D_B,
    E_D_bulk=0.0,
    D_0_gb=D_GB,
    E_D_gb=0.0,
    delta=delta,
    k_exchange=k_exchange,
    crystal_anisotropy=1.0,
)
# the charged surface is the top edge of the map, wherever that now is
bcs = [("charged", lambda x: np.isclose(x[1], LY), c0)]
solve = fm.SolveOptions(
    transient=True, final_time=t_end, stepsize=dt, atol=1e-14, rtol=1e-12
)

model = fm.build(poly, physics, bcs, solve=solve)
if theta_dependent_D:
    # the submesh has to exist before a field can live on it, so initialise
    # once, swap the material in, and initialise again
    model.initialise()
    model.network.material = F.Material(
        D_0=fm.materials.gb_diffusivity_field(model.network, micro.theta, D_B, D_GB),
        E_D=0.0,
    )
    model.initialise(force=True)
model.run()
fm.exports.averages.write_vtx(model, str(base.parent / "ebsd"), time=t_end)
network, cgb = model.network, model.network_solution

# Inspect the mesh and boundary network
print(micro.report())
len_mesh, len_tess = (
    fm.exports.measures.submesh_measure(network),
    micro.network_measure,
)
n_tri = mesh.topology.index_map(2).size_global
print(f"  mesh                            : {n_tri} triangles")
print(f"  grains (tagged pieces)          : {poly.n_grains}")
print(f"  network captured by the submesh : {100 * len_mesh / len_tess:.2f} %")
print(
    f"  connected components            : "
    f"{fm.exports.measures.component_count(network)}"
)
print(
    f"  interior facets                 : {model.model.manifold_is_interior(network)}"
)

# Analyse grain-boundary transport
depth = micro.junction_only_below(LY)
gb_y = cgb.function_space.tabulate_dof_coordinates()[:, 1]
deep = gb_y < depth
c_deep = cgb.x.array[deep].max() if deep.any() else 0.0
print(
    f"\n  inventory                      : {fm.exports.averages.inventory(model):.4e}"
)
beta = fm.materials.beta_parameter(delta, D_GB, D_B, t_end)
print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")
print(
    f"  interface / lattice resistance : "
    f"{physics.interface_resistance_ratio(np.sqrt(LX * LY / poly.n_grains)):.1e}"
    f"  (Fisher is the 0 limit)"
)

c_grain_deep = fm.exports.averages.lattice_mean_below(model, axis=1, depth=depth)
print("\njunction transport: no boundary touching the charged edge reaches below")
print(f"y = {depth:.4g}, so everything the network holds there has")
print("crossed at least one triple junction.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
# both are solver noise around zero whenever the boundary tail is shorter
# than the junction-only depth, and noise has a sign; the ratio is then 0/0
if c_grain_deep > 1e-12 * c0:
    print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
else:
    print("  ratio                          : n/a - nothing has reached this depth")
