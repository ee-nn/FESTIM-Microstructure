"""The tessellation geometry in meshing.voronoi: needs neither gmsh nor dolfinx."""

import numpy as np
import pytest

from festim_microstructure.meshing import voronoi as V


@pytest.fixture
def segments():
    return V.voronoi_segments(24, 1.0, np.random.default_rng(3), aspect=1.0)


def test_segments_lie_inside_the_box(segments):
    pts = np.array([p for seg in segments for p in seg])
    assert pts.min() >= -1e-12 and pts.max() <= 1.0 + 1e-12


def _component_lengths(segments, size):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    scale = 1e-9 * size
    key = lambda p: tuple(np.round(p / scale).astype(np.int64))  # noqa: E731
    for p, q in segments:
        parent[find(key(p))] = find(key(q))
    lengths = {}
    for p, q in segments:
        lengths[find(key(p))] = lengths.get(find(key(p)), 0.0) + np.linalg.norm(q - p)
    return sorted(lengths.values(), reverse=True)


def test_main_component_carries_the_network(segments):
    """The periodic tiling occasionally leaves a corner sliver of one or two
    ridges that is not joined to the rest; the network proper must still be one
    component holding essentially all the ridge length."""
    lengths = _component_lengths(segments, 1.0)
    assert V.connected_components(segments, 1.0) == len(lengths)
    assert lengths[0] / sum(lengths) > 0.95
    assert len(V.triple_junctions(segments, 1.0)) > 0


def test_network_tensor_trace_is_ridge_length(segments):
    tensor = V.network_tensor(segments)
    length = sum(np.linalg.norm(q - p) for p, q in segments)
    assert np.isclose(np.trace(tensor), length)
    assert np.allclose(tensor, tensor.T)


def test_elongated_grains_give_an_anisotropic_tensor():
    rng = np.random.default_rng(3)
    segs = V.voronoi_segments(24, 1.0, rng, aspect=4.0)
    evals = np.linalg.eigvalsh(V.network_tensor(segs))
    # ridges run mostly along x, so the xx moment dominates
    assert evals[1] > 2 * evals[0]


def test_snap_puts_edge_endpoints_back_on_the_box(segments):
    tol = 0.0137  # deliberately not a divisor of the box side
    snapped = V.snap_segments(segments, tol, 1.0)
    pts = np.array([p for seg in snapped for p in seg])
    near_edge = np.abs(pts) < tol
    assert np.all(pts[near_edge] == 0.0)
    near_far = np.abs(pts - 1.0) < tol
    assert np.all(pts[near_far] == 1.0)


def test_near_segments_marks_midpoints_but_not_far_points(segments):
    mids = np.array([(p + q) / 2 for p, q in segments]).T
    mids = np.vstack([mids, np.zeros(mids.shape[1])])
    assert V.near_segments(mids, segments, 1e-9).all()
    far = np.array([[-1.0], [-1.0], [0.0]])
    assert not V.near_segments(far, segments, 1e-3).any()


def test_clip_to_box_keeps_interior_segment_and_drops_outside():
    box = np.array([1.0, 1.0])
    p, q = np.array([0.2, 0.2]), np.array([0.8, 0.8])
    a, b = V.clip_to_box(p, q, box)
    assert np.allclose(a, p) and np.allclose(b, q)
    assert V.clip_to_box(np.array([2.0, 2.0]), np.array([3.0, 3.0]), box) is None


def test_3d_faces_are_planar_rings_inside_the_cube():
    faces = V.voronoi_faces(8, 1.0, np.random.default_rng(1))
    assert len(faces) > 0
    for poly in faces:
        assert poly.shape[1] == 3 and len(poly) >= 3
        assert poly.min() >= -1e-9 and poly.max() <= 1.0 + 1e-9
        n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        assert np.allclose((poly - poly[0]) @ n, 0.0, atol=1e-9 * np.linalg.norm(n))
        assert V.polygon_area(poly) > 0
    assert V.connected_components_3d(faces) == 1
    lines, length, _ = V.triple_lines(faces)
    assert len(lines) > 0 and length > 0
