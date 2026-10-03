"""Check that the evaluation detects lost holes and retains separate loops."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "evaluate_upxo", Path(__file__).resolve().parents[1] / "tools/evaluate_upxo.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def pixel_triangles(labels):
    """A conforming triangular mesh of a labelled pixel map."""
    nx = labels.shape[1]
    triangles, owners = [], []
    for (row, col), grain in np.ndenumerate(labels):
        if grain == 0:
            continue
        a = row * (nx + 1) + col
        b, c, d = a + 1, a + nx + 1, a + nx + 2
        triangles.extend([(a, b, d), (a, d, c)])
        owners.extend([grain, grain])
    return np.asarray(triangles), np.asarray(owners)


@pytest.mark.parametrize(
    ("case", "expected_pairs"),
    [
        ("island", [(1, 0), (1, 2)]),
        ("nested_islands", [(1, 0), (1, 2), (2, 3)]),
        ("disconnected_grain", [(1, 0), (1, 2), (1, 2)]),
        ("interior_void", [(1, 0), (1, 0)]),
    ],
)
def test_boundary_arcs_keep_disconnected_closed_loops(case, expected_pairs):
    triangles, owners = pixel_triangles(evaluation.synthetic_cases()[case])
    arcs, nonmanifold = evaluation.boundary_arcs(triangles, owners)
    assert nonmanifold == 0
    assert sorted(pair for pair, _edges in arcs) == sorted(expected_pairs)
    for _pair, edges in arcs:
        nodes, counts = np.unique(np.asarray(edges), return_counts=True)
        assert len(nodes) > 0
        np.testing.assert_array_equal(counts, 2)


def test_boundary_arcs_split_at_triple_junction():
    triangles, owners = pixel_triangles(evaluation.synthetic_cases()["triple_junction"])
    arcs, nonmanifold = evaluation.boundary_arcs(triangles, owners)
    interior = [(pair, edges) for pair, edges in arcs if 0 not in pair]
    assert nonmanifold == 0
    assert sorted(pair for pair, _edges in interior) == [(1, 2), (1, 3), (2, 3)]
    endpoints = []
    for _pair, edges in interior:
        nodes, counts = np.unique(np.asarray(edges), return_counts=True)
        endpoints.append(set(nodes[counts == 1]))
    assert len(set.intersection(*endpoints)) == 1


def test_geometry_metrics_reject_filled_parent_hole():
    shapely = pytest.importorskip("shapely")
    pytest.importorskip("rasterio")
    original = evaluation.raster_polygons(evaluation.synthetic_cases()["island"])
    assert evaluation.geometry_metrics(original, original)["accepted"]
    altered = dict(original)
    altered[1] = shapely.Polygon(original[1].exterior)
    report = evaluation.geometry_metrics(original, altered)
    assert not report["accepted"]
    assert report["overlap_fraction"] > 0
    assert report["reference_topology"]["1"]["holes"] == 1
    assert report["actual_topology"]["1"]["holes"] == 0


def test_pixel_seeds_sample_their_own_pixel_including_first_row_and_column():
    from scipy.ndimage import map_coordinates

    labels = np.arange(35, dtype=np.int32).reshape(5, 7)
    seeds = evaluation.pixel_seeds(labels.shape)
    sampled = map_coordinates(labels, seeds[:, ::-1].T, order=0, mode="nearest")
    np.testing.assert_array_equal(sampled, labels.ravel())
