"""Build periodic 2D/3D Voronoi microstructures and conforming Gmsh meshes.

GBs are codimension-one facets; their physical width remains a model coefficient.

This was one module; it is now split by concern, and the names it exported are
re-exported here, so ``from festim_microstructure.voronoi import X``
keeps working:

``_geometry``
    dimension-independent clipping, polygon and union-find helpers
``geometry2d`` / ``geometry3d``
    the tessellation proper, and the measures taken off it
``gmsh_builder``
    the only module that touches Gmsh or DOLFINx
``polycrystal``
    the microstructure classes the models consume
"""

from ._geometry import (
    clip_polygon_to_box,
    clip_to_box,
    order_ring,
    polygon_area,
)
from .geometry2d import (
    connected_components,
    near_segments,
    network_tensor,
    snap_segments,
    triple_junctions,
    voronoi_segments,
)
from .geometry3d import (
    connected_components_3d,
    face_edges,
    near_faces,
    network_area,
    triple_lines,
    voronoi_faces,
)
from .gmsh_builder import (
    GB_TAG_3D,
    MeshSizing,
    build_mesh,
    build_mesh_3d,
    grain_tags_from_seeds,
)
from .polycrystal import VoronoiMicrostructure, VoronoiMicrostructure3D

__all__ = [
    "GB_TAG_3D",
    "MeshSizing",
    "VoronoiMicrostructure",
    "VoronoiMicrostructure3D",
    "build_mesh",
    "build_mesh_3d",
    "clip_polygon_to_box",
    "clip_to_box",
    "connected_components",
    "connected_components_3d",
    "face_edges",
    "grain_tags_from_seeds",
    "near_faces",
    "near_segments",
    "network_area",
    "network_tensor",
    "order_ring",
    "polygon_area",
    "snap_segments",
    "triple_junctions",
    "triple_lines",
    "voronoi_faces",
    "voronoi_segments",
]
