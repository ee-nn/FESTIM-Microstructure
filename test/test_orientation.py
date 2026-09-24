"""Orientation conventions, checked against values Neper publishes."""

import numpy as np
import pytest

from festim_microstructure.ebsd import orientation as ori
from festim_microstructure.ebsd.convert import CtfConversion
from festim_microstructure.ebsd.segmentation import grain_mean_orientations
from festim_microstructure.ebsd.settings import Settings


def test_bunge_0_30_0_matches_neper_table():
    """Verify bunge 0 30 0 matches neper table."""
    q = ori.euler_bunge_to_quat(np.array([0.0]), np.array([30.0]), np.array([0.0]))
    assert np.allclose(q[0], [0.965925826, 0.258819045, 0, 0], atol=1e-8)
    r = ori.quat_to_rodrigues(ori.to_fundamental_zone(q))
    assert np.allclose(r[0], [0.267949192, 0, 0], atol=1e-8)


def test_bunge_rotation_has_expected_ipf_z_direction():
    """Verify bunge rotation has expected ipf z direction."""
    q = ori.euler_bunge_to_quat(np.array([0.0]), np.array([30.0]), np.array([0.0]))
    w, x, y, z = q[0]
    third_row = np.array(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    )
    assert np.allclose(third_row, [0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])


def test_cubic_symmetry_quaternions_are_unit():
    """Verify cubic symmetry quaternions are unit."""
    sym = ori.cubic_symmetry_quaternions()
    assert np.allclose(np.linalg.norm(sym, axis=1), 1.0)


def test_crystal_symmetry_acts_on_the_right():
    """Verify crystal symmetry acts on the right."""
    q = ori.euler_bunge_to_quat(np.array([37.0]), np.array([52.0]), np.array([131.0]))
    equiv = ori.crystal_equivalents(q)[0]
    d = ori.cubic_disorientation_angle(ori.qmul(ori.qconj(np.repeat(q, 24, 0)), equiv))
    assert d.max() < 1e-4
    assert (
        ori.cubic_disorientation_angle(
            ori.qmul(ori.qconj(q), ori.to_fundamental_zone(q))
        )[0]
        < 1e-4
    )


def test_left_multiplication_is_not_crystal_equivalence():
    """Verify left multiplication is not crystal equivalence."""
    q = ori.euler_bunge_to_quat(np.array([37.0]), np.array([52.0]), np.array([131.0]))
    wrong = ori.qmul(ori.CUBIC_SYMMETRY[13][None], q)
    assert ori.cubic_disorientation_angle(ori.qmul(ori.qconj(q), wrong))[0] > 10.0


def test_cubic_symmetry_operation_has_zero_disorientation():
    """Verify cubic symmetry operation has zero disorientation."""
    q90 = np.array([[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]])
    assert ori.cubic_disorientation_angle(q90)[0] < 1e-4


def test_sigma3_twin_is_60_degrees():
    """Verify sigma3 twin is 60 degrees."""
    v = np.sin(np.pi / 6) / np.sqrt(3)
    q60 = np.array([[np.cos(np.pi / 6), v, v, v]])
    assert abs(ori.cubic_disorientation_angle(q60)[0] - 60.0) < 1e-6


def test_disorientation_never_exceeds_cubic_bound():
    """Verify disorientation never exceeds cubic bound."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(500, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    theta = ori.cubic_disorientation_angle(q)
    assert theta.min() >= 0.0
    assert theta.max() <= 62.8 + 1e-6


def test_filled_pixel_does_not_bias_representative_orientation():
    """Verify filled pixel does not bias representative orientation."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.radians(40.0) / 2
    rejected = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0])
    qgrid = np.array([[identity, identity, rejected]])
    cellids = np.ones((1, 3), dtype=int)
    sample_mask = np.array([[True, True, False]])

    filtered = grain_mean_orientations(qgrid, cellids, 1, sample_mask=sample_mask)
    unfiltered = grain_mean_orientations(qgrid, cellids, 1)

    assert ori.cubic_disorientation_angle(filtered)[0] < 1e-6
    assert ori.cubic_disorientation_angle(unfiltered)[0] > 10.0


def test_conversion_uses_only_originally_assigned_pixels_for_grain_mean():
    """Verify conversion uses only originally assigned pixels for grain mean."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.radians(40.0) / 2
    rejected = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0])
    conversion = CtfConversion(Settings(ctf="unused"), log=None)
    absorbed_angle = np.radians(30.0) / 2
    absorbed = np.array([np.cos(absorbed_angle), 0.0, np.sin(absorbed_angle), 0.0])
    conversion.qgrid = np.array([[identity, identity, rejected, absorbed]])
    conversion.segmented_cellids = np.array([[1, 1, 0, 2]])
    conversion.cellids = np.ones((1, 4), dtype=int)
    conversion.unassigned = conversion.segmented_cellids == 0
    conversion.ncells = 1
    conversion.shape = (1, 4)
    conversion.ctf = type("Ctf", (), {"header": {"XStep": 1.0, "YStep": 1.0}})()

    conversion.orient()

    assert ori.cubic_disorientation_angle(conversion.qcell)[0] < 1e-6


def test_orientation_mean_rejects_grain_without_a_sample():
    """Verify orientation mean rejects grain without a sample."""
    with pytest.raises(ValueError, match="grain 1 has no pixels"):
        grain_mean_orientations(
            np.array([[[1.0, 0.0, 0.0, 0.0]]]),
            np.ones((1, 1), dtype=int),
            1,
            sample_mask=np.zeros((1, 1), dtype=bool),
        )
