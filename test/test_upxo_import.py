"""Scientific contracts for the native UPXO/DefDAP import chain."""

import json

import numpy as np
import pytest

from festim_microstructure.ebsd.convert import (
    MeasureOptions,
    convert,
    measure_against_ctf,
)
from festim_microstructure.ebsd.orientation import (
    cubic_disorientation_angle,
    euler_bunge_to_quat,
    qconj,
    qmul,
)
from festim_microstructure.formats.ebsd import EbsdData, read_ebsd, write_ebsd
from festim_microstructure.meshing.upxo import resolve_python

requires_upxo = pytest.mark.skipif(
    resolve_python(required=False) is None, reason="UPXO/DefDAP importer unavailable"
)


@requires_upxo
def test_rejected_bridge_cannot_connect_grains_or_bias_means(tmp_path, ctf_writer):
    euler = np.zeros((6, 9, 3))
    mad = np.full((6, 9), 0.1)
    mad[:, 4] = 4
    euler[:, 4] = [0, 40, 0]
    ctf = ctf_writer(tmp_path / "map.ctf", euler, mad=mad)
    result = convert(ctf, tmp_path / "map.npz", min_pixels=1, max_mad=1, log=None)
    assert result.ncells == 2
    assert not result.data.indexed[:, 4].any()
    assert np.all(result.data.source_labels[:, 4] == 0)
    assert np.all(result.data.labels[:, 4] > 0)
    assert result.rms_deg < 1e-5
    assert result.segmentation.indexed.n == 48
    assert result.segmentation.backfilled.n == 6
    assert result.segmentation.backfilled.rms == pytest.approx(40, abs=1e-4)
    measured = measure_against_ctf(
        ctf,
        result.archive,
        MeasureOptions(provenance=str(result.provenance), against="both"),
        log=None,
    )
    assert measured.indexed.n == result.segmentation.indexed.n
    assert measured.backfilled.n == result.segmentation.backfilled.n
    assert measured.backfilled.rms == pytest.approx(result.segmentation.backfilled.rms)
    assert measured.voxel.max < 1e-4


@requires_upxo
def test_crop_applies_before_minimum_grain_size(tmp_path, ctf_writer):
    ctf = ctf_writer(tmp_path / "map.ctf", np.zeros((6, 6, 3)))
    full = convert(ctf, tmp_path / "full.npz", min_pixels=5, log=None)
    assert full.ncells == 1
    with pytest.raises(RuntimeError, match="no grains survived"):
        convert(ctf, tmp_path / "crop.npz", min_pixels=5, crop="0,2,0,2", log=None)
    assert not (tmp_path / "crop.npz").exists()


@requires_upxo
def test_cubic_equivalent_pixels_share_symmetry_aware_mean(tmp_path, ctf_writer):
    euler = np.zeros((5, 6, 3))
    euler[:, 3:, 0] = 90
    ctf = ctf_writer(tmp_path / "map.ctf", euler)
    result = convert(ctf, tmp_path / "map.npz", min_pixels=1, log=None)
    assert result.ncells == 1
    assert result.rms_deg < 1e-5
    assert cubic_disorientation_angle(result.data.grain_quats)[0] < 1e-5


@requires_upxo
def test_pruned_pixels_stay_excluded_in_persisted_diagnostics(tmp_path, ctf_writer):
    euler = np.zeros((6, 6, 3))
    euler[2, 2] = [0, 40, 0]
    ctf = ctf_writer(tmp_path / "map.ctf", euler)
    result = convert(ctf, tmp_path / "map.npz", min_pixels=2, log=None)
    assert result.ncells == 1
    assert result.data.source_labels[2, 2] == -2
    assert result.segmentation.indexed.n == 35
    assert result.segmentation.backfilled.n == 1
    measured = measure_against_ctf(ctf, result.archive, log=None)
    assert measured.indexed.n == 35
    assert measured.indexed.rms < 1e-5
    assert json.loads(result.provenance.read_text())["pruned_pixels"] == 1


@requires_upxo
def test_phase_selection_reordered_rows_and_rectangular_pixels(tmp_path, ctf_writer):
    euler = np.zeros((4, 6, 3))
    euler[:, 3:] = [37, 52, 131]
    phases = np.ones((4, 6), dtype=int)
    phases[:, 3:] = 2
    ctf = ctf_writer(
        tmp_path / "map.ctf", euler, phases=phases, spacing=(0.5, 0.75), reverse=True
    )
    result = convert(
        ctf, tmp_path / "map.npz", phase=2, fill=False, min_pixels=1, log=None
    )
    assert result.ncells == 1
    assert result.extent == (3.0, 3.0)
    assert np.all(result.cellids[:, :3] == 0)
    expected = euler_bunge_to_quat(37, 52, 131)
    assert (
        cubic_disorientation_angle(qmul(qconj(expected), result.data.grain_quats))[0]
        < 1e-4
    )


@requires_upxo
def test_import_cache_invalidates_when_source_changes(
    tmp_path, ctf_writer, monkeypatch
):
    from festim_microstructure.meshing import upxo

    ctf = ctf_writer(tmp_path / "map.ctf", np.zeros((4, 4, 3)))
    output = tmp_path / "map.npz"
    first = convert(ctf, output, min_pixels=1, force=False, log=None)
    with monkeypatch.context() as context:
        context.setattr(
            upxo.subprocess,
            "run",
            lambda *args, **kwargs: pytest.fail(
                "valid cache should reuse imported grains"
            ),
        )
        second = convert(ctf, output, min_pixels=1, force=False, log=None)
    np.testing.assert_array_equal(first.data.labels, second.data.labels)
    ctf_writer(ctf, np.full((4, 4, 3), [0, 30, 0]))
    third = convert(ctf, output, min_pixels=1, force=False, log=None)
    assert cubic_disorientation_angle(third.data.grain_quats)[0] == pytest.approx(
        30, abs=1e-4
    )


def test_native_archive_retains_voids_components_and_membership(tmp_path):
    labels = np.ones((12, 12), dtype=np.int32)
    labels[3:5, 3:5] = labels[8:10, 8:10] = 2
    labels[5:7, 5:7] = 0
    quats = euler_bunge_to_quat(np.array([0, 0]), np.array([30, 40]), np.array([0, 0]))
    archive = tmp_path / "labels.npz"
    write_ebsd(archive, EbsdData(labels, quats, (0.5, 0.75)))
    read = read_ebsd(archive)
    np.testing.assert_array_equal(read.labels, labels)
    np.testing.assert_array_equal(read.source_labels, labels)
    assert not list(tmp_path.glob("*.tesr"))
    with pytest.raises(ValueError, match="original grain membership"):
        EbsdData(labels, quats, (0.5, 0.75), indexed=np.zeros_like(labels, dtype=bool))
