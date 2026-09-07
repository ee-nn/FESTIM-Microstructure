"""Short-circuit diffusion through the GB network of a polycrystal measured by
EBSD, in 2D.

The mesh is Neper's direct meshing of the raster, so the grain boundaries are
the measured ones and each boundary's disorientation comes from the two grains'
measured orientations. See :mod:`festim_microstructure.meshing.ebsd.pipeline`.

Prerequisite: the ``.tesr`` written by ``examples/ebsd_ctf_to_tesr.py`` (or by
``fm-ebsd``). Neper and Gmsh are found through ``FM_NEPER_BIN`` /
``FM_GMSH_BIN`` or ``PATH``.

Run::

    python examples/ebsd_gb_diffusion.py
"""

from dataclasses import dataclass, field
from pathlib import Path

import festim as F
import numpy as np

from festim_microstructure.meshing.ebsd.pipeline import (
    EbsdMicrostructure,
    EbsdOptions,
    read_extent,
    run_ebsd_pipeline,
    unit_name,
    write_network_png,
)
from festim_microstructure.meshing.neper import TesrMeshOptions, read_mesh
from festim_microstructure.models.fisher import (
    ShortCircuitParams,
    ShortCircuitProblem,
    beta_parameter,
)
from festim_microstructure.models.properties import gb_diffusivity_field
from festim_microstructure.postprocessing.measures import (
    component_count,
    inventory,
    submesh_measure,
)
from festim_microstructure.subdomains import TaggedGrainBoundaryNetwork

HERE = Path(__file__).resolve().parent


@dataclass
class Setup:
    ebsd: EbsdOptions = field(
        default_factory=lambda: EbsdOptions(
            tesr=str(HERE / "data" / "d7.tesr"),
            unit=1e-6,
            theta_min=10.0,
            mesh=TesrMeshOptions(rcl=0.25, mesh_qual_min=0.7),
        )
    )
    workdir: Path = HERE / "results"
    force: bool = True

    D_B: float = 1e-14  # lattice diffusivity     [m^2/s]
    D_GB: float = 1e-8  # GB diffusivity          [m^2/s]
    delta: float = 5e-9  # GB width               [m]
    k_exchange: float = 1e-4  # bulk <-> GB exchange [m/s]
    c0: float = 1.0
    t_end: float = 36000.0
    dt: float = 600.0
    theta_dependent_D: bool = False  # see gb_diffusivity_field, CHECK before enabling


def main(s=Setup()):
    D_B, D_GB = s.D_B, s.D_GB
    unit, uname = s.ebsd.unit, unit_name(s.ebsd.unit)

    base = run_ebsd_pipeline(s.ebsd, workdir=s.workdir, force=s.force)
    LX, LY = read_extent(base, unit)
    mesh, cell_tags, facet_tags = read_mesh(base, gdim=2, unit=unit)
    micro = EbsdMicrostructure(
        base, mesh, cell_tags, facet_tags, (LX, LY), theta_min=s.ebsd.theta_min
    )
    micro.check_orientations()
    write_network_png(base, mesh, micro, base.parent / "poly-raw.tesr", unit, uname)

    network = TaggedGrainBoundaryNetwork(
        id=ShortCircuitProblem.NETWORK_ID,
        material=F.Material(D_0=D_GB, E_D=0.0),
        facet_tags=facet_tags,
        entity_ids=micro.network_edge_ids,
        dim=1,
    )
    params = ShortCircuitParams(
        D_b=D_B,
        D_gb=D_GB,
        delta=s.delta,
        k_exchange=s.k_exchange,
        c0=s.c0,
        t_end=s.t_end,
        dt=s.dt,
        atol=1e-14,
        rtol=1e-12,
    )
    # the charged surface is the top edge of the map, wherever that now is
    problem = ShortCircuitProblem(
        mesh, network, charged_surface=lambda x: np.isclose(x[1], LY), params=params
    )
    exports = problem.vtx_exports(str(base.parent / "ebsd"))
    if s.theta_dependent_D:
        problem.build(D_GB, exports).initialise()
        network.material = F.Material(
            D_0=gb_diffusivity_field(network, micro.theta, D_B, D_GB), E_D=0.0
        )
    model, cb, cgb = problem.solve(D_GB, exports)

    # what we built
    print(micro.report())
    len_mesh, len_tess = submesh_measure(network), micro.network_length
    n_tri = mesh.topology.index_map(2).size_global
    print(f"  mesh                            : {n_tri} triangles")
    print(f"  network captured by the submesh : {100 * len_mesh / len_tess:.2f} %")
    print(f"  connected components            : {component_count(network)}")
    print(f"  interior facets                 : {model.manifold_is_interior(network)}")

    # effect of the network
    depth = micro.junction_only_below(LY)
    gb_y = cgb.function_space.tabulate_dof_coordinates()[:, 1]
    deep = gb_y < depth
    c_deep = cgb.x.array[deep].max() if deep.any() else 0.0
    print(f"\n  inventory                      : {inventory(cb, cgb, s.delta):.4e}")
    beta = beta_parameter(s.delta, D_GB, D_B, s.t_end)
    print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")

    bulk_y = cb.function_space.tabulate_dof_coordinates()[:, 1]
    c_grain_deep = cb.x.array[bulk_y < depth].mean()
    print("\njunction transport: no boundary touching the charged edge reaches below")
    print(f"y = {depth:.4g}, so everything the network holds there has")
    print("crossed at least one triple junction.")
    print(f"  max c on the network there     : {c_deep:.4e}")
    print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
    # both are solver noise around zero whenever the Fisher tail is shorter than
    # the junction-only depth, and noise has a sign; the ratio is then 0/0
    if c_grain_deep > 1e-12 * s.c0:
        print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
    else:
        print("  ratio                          : n/a - nothing has reached this depth")


if __name__ == "__main__":
    main()
