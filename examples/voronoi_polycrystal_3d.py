"""Short-circuit diffusion through the grain-boundary network of a 3D Voronoi
polycrystal.

Same idea as the 2D example: the entire network is declared as **one** codim-1
subdomain with **one** species. In 3D "codim-1" means the network is made of
*facets* (triangles) rather than edges, so ``dim=2`` in a ``tdim == 3`` mesh, and the
junctions are triple *lines* (edges shared by three faces) plus quadruple *points*.
The submesh is connected through those shared edges exactly as the 2D one was
connected through shared vertices, so nothing about the FESTIM setup changes.

What does change relative to the 2D script:

* A Voronoi ridge in 3D is a convex *polygon*, not a segment, so Liang-Barsky is
  replaced by Sutherland-Hodgman clipping against the six half-spaces of the box.
* The network is tagged in gmsh with a physical group and picked up from the facet
  tags (:class:`TaggedGrainBoundaryNetwork`) instead of being located geometrically.
  The geometric route still works (``locate_geometrically=True`` uses
  :func:`near_faces`) but a point-to-polygon distance is much more delicate than a
  point-to-segment distance, and the fragment operation already knows exactly which
  facets are grain boundaries.
* ``mouths`` is ``dim=1`` (curves where the network meets the charged face), which
  :class:`ShortCircuitProblem` derives from the mesh dimension.
* The grain-boundary volume fraction is delta * area / L**3 instead of
  delta * length / L**2.

Run::

    python examples/voronoi_polycrystal_3d.py
"""

from dataclasses import dataclass

import festim as F
import numpy as np

from festim_microstructure.meshing.voronoi import (
    GB_TAG_3D,
    build_mesh_3d,
    connected_components_3d,
    near_faces,
    polygon_area,
    triple_lines,
    voronoi_faces,
)
from festim_microstructure.models.fisher import (
    ShortCircuitParams,
    ShortCircuitProblem,
    beta_parameter,
    hart_bound,
)
from festim_microstructure.postprocessing.measures import (
    component_count,
    inventory,
    junction_only_below,
    submesh_measure,
)
from festim_microstructure.subdomains import (
    GrainBoundaryNetwork,
    TaggedGrainBoundaryNetwork,
)


@dataclass
class Setup:
    L: float = 1.0  # specimen size
    n_seeds: int = 12  # number of grains
    seed: int = 3

    D_B: float = 1e-3  # lattice diffusivity
    D_GB: float = 30.0  # grain-boundary diffusivity
    delta: float = 1e-3  # grain-boundary width
    k_exchange: float = 1.0  # bulk <-> grain-boundary exchange
    c0: float = 1.0
    t_end: float = 3.0
    dt: float = 0.05

    h_gb: float = 0.05  # mesh size at the boundaries
    h_bulk: float = 0.15  # mesh size in the grain interiors
    locate_geometrically: bool = False  # use near_faces instead of the facet tags


def main(s=Setup()):
    L, D_B, D_GB = s.L, s.D_B, s.D_GB

    faces = voronoi_faces(s.n_seeds, L, np.random.default_rng(s.seed))
    mesh, facet_tags = build_mesh_3d(faces, L, s.h_gb, s.h_bulk)
    material = F.Material(D_0=D_GB, E_D=0.0)
    if s.locate_geometrically:
        network = GrainBoundaryNetwork(
            id=ShortCircuitProblem.NETWORK_ID,
            material=material,
            locator=lambda x: near_faces(x, faces),
            dim=2,
        )
    else:
        network = TaggedGrainBoundaryNetwork(
            id=ShortCircuitProblem.NETWORK_ID,
            material=material,
            facet_tags=facet_tags,
            entity_ids=[GB_TAG_3D],
            dim=2,
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
    problem = ShortCircuitProblem(
        mesh, network, charged_surface=lambda x: np.isclose(x[2], L), params=params
    )
    model, cb_fast, cgb_fast = problem.solve(
        D_GB, exports=problem.vtx_exports("voronoi3d")
    )

    # what we built
    lines, triple_length, quadruple = triple_lines(faces)
    face_area = sum(polygon_area(poly) for poly in faces)
    sub_area = submesh_measure(network)
    print(f"microstructure: {s.n_seeds} seeds, {len(faces)} boundary polygons")
    print(
        f"  triple lines                    : {len(lines)} (length {triple_length:.3f})"
    )
    print(f"  quadruple points                : {len(quadruple)}")
    print(f"  connected components (faces)    : {connected_components_3d(faces)}")
    print(f"  connected components (submesh)  : {component_count(network)}")
    n_cells = mesh.topology.index_map(3).size_global
    print(f"  mesh                            : {n_cells} cells")
    print(
        f"  network captured by the submesh : {sub_area:.4f} of {face_area:.4f}"
        f" ({100 * sub_area / face_area:.2f} %)"
    )
    print(f"  interior facets                 : {model.manifold_is_interior(network)}")

    # effect of the network
    fast = inventory(cb_fast, cgb_fast, s.delta)
    depth = junction_only_below(faces, axis=2, top=L)
    gb_z = cgb_fast.function_space.tabulate_dof_coordinates()[:, 2]
    deep = gb_z < depth
    c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

    _, cb_ref, cgb_ref = problem.solve(D_B)
    ref = inventory(cb_ref, cgb_ref, s.delta)

    print(
        f"\nafter t = {s.t_end} (lattice diffusion alone reaches "
        f"~{2 * np.sqrt(D_B * s.t_end):.3g})"
    )
    print(f"  inventory with fast boundaries : {fast:.4e}")
    print(f"  inventory with D_gb = D_b      : {ref:.4e}")
    print(f"  enhancement                    : x {fast / ref:.1f}")
    f_gb = s.delta * face_area / L**3
    print(f"  boundary volume fraction f     : {f_gb:.3e}")
    print(
        f"  Hart bound f D_gb + (1-f) D_b  : {hart_bound(f_gb, D_GB, D_B):.3e}"
        f"  (vs D_b = {D_B:.3e})"
    )
    beta = beta_parameter(s.delta, D_GB, D_B, s.t_end)
    print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")

    bulk_z = cb_fast.function_space.tabulate_dof_coordinates()[:, 2]
    c_grain_deep = cb_fast.x.array[bulk_z < depth].mean()
    print("\njunction transport: no boundary touching the charged face reaches below")
    print(f"z = {depth:.3f}, so everything the network holds there has")
    print("crossed at least one triple line.")
    print(f"  max c on the network there     : {c_deep:.4e}")
    print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
    print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")


if __name__ == "__main__":
    main()
