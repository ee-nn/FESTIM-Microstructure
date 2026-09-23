"""Exercise the relocated readers through conversion and mesh diagnostics."""

import numpy as np
import pytest

from festim_microstructure.ebsd.convert import (
    MeasureOptions,
    convert,
    measure_tesr_against_ctf,
)
from festim_microstructure.ebsd.settings import settings_from_provenance
from festim_microstructure.formats.provenance import read_provenance
from festim_microstructure.formats.tesr import (
    TesrData,
    read_tesr,
    read_tesr_full,
    write_tesr,
)
from festim_microstructure.meshing.diagnostics import area_change, overlay


@pytest.mark.parametrize("voxel_ori", [False, True])
def test_tesr_roundtrip(tmp_path, voxel_ori):
    """Verify tesr roundtrip."""
    cells = np.array([[1, 1, 2], [1, 2, 2]])
    orientations = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    voxel = orientations[cells - 1] if voxel_ori else None
    mask = np.array([[True, False, True], [True, True, True]])
    path = tmp_path / "map.tesr"
    write_tesr(path, TesrData(cells, orientations, (0.5, 0.75), "cubic", voxel, mask))
    read_cells, vox = read_tesr(path)
    full = read_tesr_full(path)
    np.testing.assert_array_equal(read_cells, cells)
    np.testing.assert_array_equal(full["cells"], cells)
    np.testing.assert_allclose(full["cell_ori"], orientations)
    assert vox == full["vox"] == (0.5, 0.75)
    if voxel_ori:
        np.testing.assert_allclose(full["vox_ori"], voxel)
        np.testing.assert_array_equal(full["oridef"], mask)
    else:
        assert "vox_ori" not in full


def test_ctf_conversion_provenance_and_readback(tmp_path):
    """Verify ctf conversion provenance and readback."""
    path = tmp_path / "map.ctf"
    header = (
        "Channel Text File\nXCells\t4\nYCells\t3\nXStep\t1\nYStep\t1\n"
        "Phases\t1\n1;1;1\t90;90;90\tTungsten\t11\t229\n"
        "Phase\tX\tY\tEuler1\tEuler2\tEuler3\tMAD\tError\tBands\n"
    )
    # Reversed rows exercise coordinate-based grid placement.
    rows = [f"1\t{x}\t{y}\t0\t0\t0\t0.1\t0\t8\n" for y in range(3) for x in range(4)]
    path.write_text(header + "".join(reversed(rows)))
    result = convert(
        path, tmp_path / "map.tesr", min_pixels=1, flip_y=True, neper=None, log=None
    )
    assert result.ncells == 1
    assert result.rms_deg < 1e-5
    np.testing.assert_array_equal(read_tesr(result.tesr)[0], np.ones((3, 4)))
    saved = read_provenance(result.provenance)
    assert saved["flip_y"] is True
    assert saved["ncells"] == 1
    settings = settings_from_provenance(result.provenance)
    assert settings.flip_y is True
    assert settings.min_pixels == 1
    measured = measure_tesr_against_ctf(
        path,
        result.tesr,
        MeasureOptions(provenance=str(result.provenance)),
        log=None,
    )
    assert measured.indexed.rms < 1e-5


def test_mesh_area_and_overlay_use_shared_readers(tmp_path):
    """Verify mesh area and overlay use shared readers."""
    tesr = tmp_path / "map.tesr"
    write_tesr(
        tesr,
        TesrData(np.ones((2, 2), dtype=int), np.zeros((1, 3)), (0.5, 0.5), "cubic"),
    )
    msh = tmp_path / "map.msh4"
    msh.write_text(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n"
        "0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n"
        "$Elements\n2 6 1 6\n1 1 1 4\n"
        "1 1 2\n2 2 3\n3 3 4\n4 4 1\n"
        "2 1 2 2\n5 1 2 3\n6 1 3 4\n$EndElements\n"
    )
    report = area_change(tesr, msh)
    assert report.identity
    assert report.mesh_total == pytest.approx(1.0)
    np.testing.assert_allclose(report.delta, 0.0)
    out = tmp_path / "overlay.png"
    overlay(tesr, msh, output=out, log=None)
    assert out.stat().st_size > 0


def test_neper_render_has_explicit_camera_vectors(tmp_path):
    """Exercise the installed Neper/POV-Ray pair, including its camera defaults."""
    from PIL import Image

    from festim_microstructure._binaries import resolve_all
    from festim_microstructure.ebsd.diagnostics import _render_neper_png

    binaries = resolve_all()
    if not binaries["neper"] or not binaries["povray"]:
        pytest.skip("needs Neper and POV-Ray")
    tesr = tmp_path / "map.tesr"
    write_tesr(
        tesr,
        TesrData(np.array([[1, 2], [2, 1]]), np.zeros((2, 3)), (1, 1), "cubic"),
    )
    assert _render_neper_png(
        [
            binaries["neper"],
            "-V",
            tesr.name,
            "-povray",
            binaries["povray"],
            "-imagesize",
            "200:200",
            "-print",
            "map",
        ],
        tmp_path,
        print,
    )
    rgb = np.asarray(Image.open(tmp_path / "map.png").convert("RGB"))
    # Test the unannotated map: a scale bar cannot make a blank render pass.
    colorful = rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int) > 40
    assert colorful.mean() > 0.1
    assert not (tmp_path / "map.pov").exists()
