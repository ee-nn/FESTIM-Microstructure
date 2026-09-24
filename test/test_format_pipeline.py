"""Exercise the relocated readers through conversion and mesh diagnostics."""

import json

import numpy as np
import pytest

from festim_microstructure.ebsd.convert import (
    CtfConversion,
    MeasureOptions,
    convert,
    measure_tesr_against_ctf,
)
from festim_microstructure.ebsd.diagnostics import verify_readback
from festim_microstructure.ebsd.orientation import (
    ROT_X_180,
    cubic_disorientation_angle,
    euler_bunge_to_quat,
    qconj,
    qmul,
    rodrigues_to_quat,
)
from festim_microstructure.ebsd.settings import Settings, settings_from_provenance
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


GENERIC_EULER = (37.0, 52.0, 131.0)


def _write_ctf(path, euler=GENERIC_EULER, laue=11, nx=4, ny=3):
    """A single-grain map whose orientation is not a cubic symmetry operator."""
    header = (
        f"Channel Text File\nXCells\t{nx}\nYCells\t{ny}\nXStep\t1\nYStep\t1\n"
        f"Phases\t1\n1;1;1\t90;90;90\tPhase\t{laue}\t229\n"
        "Phase\tX\tY\tEuler1\tEuler2\tEuler3\tMAD\tError\tBands\n"
    )
    e1, e2, e3 = euler
    rows = [
        f"1\t{x}\t{y}\t{e1}\t{e2}\t{e3}\t0.1\t0\t8\n"
        for y in range(ny)
        for x in range(nx)
    ]
    path.write_text(header + "".join(rows))
    return path


def _file_vs(result, q):
    """Disorientation (deg) between the written cell orientation and ``q``."""
    back = rodrigues_to_quat(read_tesr_full(result.tesr)["cell_ori"])[0]
    return float(cubic_disorientation_angle(qmul(qconj(q), back)))


def test_flip_y_rotates_orientations_and_reads_back_clean(tmp_path):
    """flip_y is a proper frame change: rows mirrored, orientations rotated."""
    ctf = _write_ctf(tmp_path / "map.ctf")
    conv = CtfConversion(
        Settings(ctf=str(ctf), min_pixels=1, flip_y=True, neper=None), log=None
    )
    conv.read().segment().clean().orient().measure().write(tmp_path / "map.tesr")
    result = conv.result()
    q = euler_bunge_to_quat(*GENERIC_EULER)
    assert _file_vs(result, qmul(ROT_X_180, q)) < 1e-5
    assert _file_vs(result, q) > 1.0  # the unrotated orientation is wrong
    report = verify_readback(result.tesr, conv.qgrid, conv.ok, conv.cellids, True)
    assert not any("DIFFER" in line or "NOT" in line for line in report), report
    measured = measure_tesr_against_ctf(
        ctf,
        result.tesr,
        MeasureOptions(provenance=str(result.provenance), against="both"),
        log=None,
    )
    assert measured.indexed.rms < 1e-5
    assert measured.voxel.rms < 1e-5


def test_euler_correction_left_multiplies_and_round_trips(tmp_path):
    """euler_correction maps the Euler frame onto the map frame, as MTEX does."""
    ctf = _write_ctf(tmp_path / "map.ctf")
    result = convert(
        ctf,
        tmp_path / "map.tesr",
        min_pixels=1,
        euler_correction=(180.0, 0.0, 0.0),
        neper=None,
        log=None,
    )
    q = euler_bunge_to_quat(*GENERIC_EULER)
    assert _file_vs(result, qmul(euler_bunge_to_quat(180.0, 0.0, 0.0), q)) < 1e-5
    assert _file_vs(result, q) > 1.0
    assert settings_from_provenance(result.provenance).euler_correction == [
        180.0,
        0.0,
        0.0,
    ]
    measured = measure_tesr_against_ctf(
        ctf, result.tesr, MeasureOptions(provenance=str(result.provenance)), log=None
    )
    assert measured.indexed.rms < 1e-5


def test_laue_m3_is_rejected(tmp_path):
    """Laue 10 (m-3) has 12 proper rotations; the 24-operator math must refuse it."""
    ctf = _write_ctf(tmp_path / "map.ctf", laue=10)
    with pytest.raises(ValueError, match="Laue group 10"):
        convert(ctf, tmp_path / "map.tesr", min_pixels=1, neper=None, log=None)


def test_active_option_is_gone(tmp_path):
    """The removed option is refused, including from an old provenance file."""
    with pytest.raises(TypeError):
        Settings(ctf="x.ctf", active=True)
    old = tmp_path / "old-provenance.json"
    old.write_text(json.dumps({"ctf": "x.ctf", "active": True}))
    with pytest.raises(ValueError, match="active=True"):
        settings_from_provenance(old)
