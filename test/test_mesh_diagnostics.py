"""Physical ownership survives export independently of raster displacement."""

from types import SimpleNamespace

import meshio
import numpy as np
import pytest

from festim_microstructure.formats.ebsd import EbsdData, write_ebsd
from festim_microstructure.meshing import upxo
from festim_microstructure.meshing.diagnostics import (
    AreaReportOptions,
    area_change,
    centroid_displacement,
    format_report,
    measure,
)


def strip_mesh(x, grain_ids):
    """Two triangles per strip, with shared vertices at strip boundaries."""
    points = np.array([(xi, y, 0) for xi in x for y in (0, 10)], dtype=float)
    triangles = np.array(
        [
            t
            for i in range(len(grain_ids))
            for t in ((2 * i, 2 * i + 2, 2 * i + 3), (2 * i, 2 * i + 3, 2 * i + 1))
        ]
    )
    return points, triangles, np.repeat(grain_ids, 2)


@pytest.fixture
def shifted_thin_grain(tmp_path):
    """A valid 0.8-pixel shift puts all thin-grain centroids over grain 3."""
    points, triangles, owners = strip_mesh([0, 50.8, 51.8, 101], [2, 1, 3])
    mesher = SimpleNamespace(
        nodes=points, elConn={"triangle": triangles}, _elem_owners={"triangle": owners}
    )
    labels = np.full((10, 101), 3, dtype=np.int32)
    labels[:, :50], labels[:, 50] = 2, 1
    vox, unit = (1.6, 0.8), 1e-6
    archive, base = tmp_path / "map.npz", tmp_path / "mesh"
    write_ebsd(archive, EbsdData(labels, np.array([[1.0, 0, 0, 0]] * 3), vox))
    report = upxo.export_mesh(mesher, base, vox, labels.shape, unit)
    assert report["accepted"], report
    return archive, base.with_suffix(".msh4"), unit


def test_valid_shifted_thin_grain_is_comparable(shifted_thin_grain):
    from shapely.geometry import box

    reference = {1: box(50, 0, 51, 10), 2: box(0, 0, 50, 10), 3: box(51, 0, 101, 10)}
    moved = {
        1: box(50.8, 0, 51.8, 10),
        2: box(0, 0, 50.8, 10),
        3: box(51.8, 0, 101, 10),
    }
    assert upxo.geometry_metrics(reference, moved)["accepted"]
    archive, msh4, unit = shifted_thin_grain
    report = measure(archive, msh4, AreaReportOptions(mesh_unit=unit), log=None)
    assert report.comparable
    np.testing.assert_allclose(report.delta, [0, 1.6, -1.6], atol=1e-10)
    assert report.displaced == pytest.approx(1 / 101)
    assert "centroid-sampled displaced area estimate" in "\n".join(
        format_report(report)
    )


@pytest.mark.parametrize(
    "replacement,missing,unexpected", [(2, [1], []), (4, [1], [4])]
)
def test_area_comparison_rejects_missing_or_extra_grain_ids(
    shifted_thin_grain, replacement, missing, unexpected
):
    archive, msh4, unit = shifted_thin_grain
    mesh = meshio.read(msh4, file_format="gmsh")
    for block, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        if block.type == "triangle":
            tags[tags == 1] = replacement
    meshio.write(msh4, mesh, file_format="gmsh22", binary=False)
    with pytest.raises(ValueError, match="physical grain IDs differ") as error:
        area_change(archive, msh4, mesh_unit=unit)
    assert f"missing {missing}, unexpected {unexpected}" in str(error.value)


def test_centroid_displacement_counts_outside_raster_without_clipping():
    cells = np.ones((1, 1), dtype=int)
    displaced = centroid_displacement(
        cells,
        (2, 1),
        np.ones(3),
        np.array([1, 2, 3]),
        np.array([[1, 0.5], [-0.01, 0.5], [2.01, 0.5]]),
    )
    assert displaced == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    "change,failed_check",
    [
        ("reorder", None),
        ("swap_grains", "triangle_physical_ids_preserved"),
        ("swap_boundaries", "boundary_physical_ids_preserved"),
        ("missing_physical", "triangle_physical_ids_preserved"),
        ("connectivity", "triangle_connectivity_preserved"),
        ("coordinates", "coordinates_preserved"),
    ],
)
def test_export_checks_exact_physical_ownership(tmp_path, change, failed_check):
    points, triangles, owners = strip_mesh([0, 1, 2], [1, 2])
    points *= 1e-6
    arcs, _ = upxo.boundary_arcs(triangles, owners)
    lines = np.array([edge for _, edges in arcs for edge in edges])
    edge_ids = np.array([i for i, (_, edges) in enumerate(arcs, 1) for _ in edges])
    mesh = meshio.Mesh(
        points.copy(),
        [("line", lines.copy()), ("triangle", triangles.copy())],
        cell_data={
            "gmsh:physical": [edge_ids.copy(), owners.copy()],
            "gmsh:geometrical": [edge_ids.copy() + 20, owners.copy() + 10],
        },
    )
    if change == "reorder":
        # Reorder vertices and elements, and reverse each element's node order.
        permutation = np.arange(len(points))[::-1]
        mesh.points = mesh.points[permutation]
        node_map = np.argsort(permutation)
        for i, block in enumerate(mesh.cells):
            block.data = node_map[block.data[::-1, ::-1]]
            for data in mesh.cell_data.values():
                data[i] = data[i][::-1]
    elif change == "swap_grains":
        mesh.cell_data["gmsh:physical"][1] = 3 - owners
    elif change == "swap_boundaries":
        mesh.cell_data["gmsh:physical"][0] = np.roll(edge_ids, 1)
    elif change == "missing_physical":
        del mesh.cell_data["gmsh:physical"]
    elif change == "connectivity":
        mesh.cells[1].data[0] = triangles[1]
    elif change == "coordinates":
        mesh.points[0, 0] += 1e-8
    path = tmp_path / "mesh.msh"
    meshio.write(path, mesh, file_format="gmsh22", binary=False)
    report = upxo.export_metrics(path, points, triangles, owners, lines, edge_ids)
    assert report["accepted"] is (failed_check is None), report
    if failed_check:
        assert not report[failed_check]
