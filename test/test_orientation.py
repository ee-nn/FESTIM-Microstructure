"""Orientation conventions, checked against values Neper publishes."""

import numpy as np

from festim_microstructure.ebsd import orientation as ori


def test_bunge_0_30_0_matches_neper_table():
    q = ori.euler_bunge_to_quat(np.array([0.0]), np.array([30.0]), np.array([0.0]))
    assert np.allclose(q[0], [0.965925826, 0.258819045, 0, 0], atol=1e-8)
    r = ori.quat_to_rodrigues(
        ori.to_fundamental_zone(q, ori.cubic_symmetry_quaternions())
    )
    assert np.allclose(r[0], [0.267949192, 0, 0], atol=1e-8)


def test_bunge_rotation_has_expected_ipf_z_direction():
    q = ori.euler_bunge_to_quat(np.array([0.0]), np.array([30.0]), np.array([0.0]))
    w, x, y, z = q[0]
    third_row = np.array(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    )
    assert np.allclose(third_row, [0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])


def test_cubic_symmetry_quaternions_are_unit():
    sym = ori.cubic_symmetry_quaternions()
    assert np.allclose(np.linalg.norm(sym, axis=1), 1.0)


def test_crystal_symmetry_acts_on_the_right():
    sym = ori.cubic_symmetry_quaternions()
    q = ori.euler_bunge_to_quat(np.array([37.0]), np.array([52.0]), np.array([131.0]))
    equiv = ori.crystal_equivalents(q, sym)[0]
    d = ori.cubic_disorientation_angle(ori.qmul(ori.qconj(np.repeat(q, 24, 0)), equiv))
    assert d.max() < 1e-4
    assert (
        ori.cubic_disorientation_angle(
            ori.qmul(ori.qconj(q), ori.to_fundamental_zone(q, sym))
        )[0]
        < 1e-4
    )


def test_left_multiplication_is_not_crystal_equivalence():
    sym = ori.cubic_symmetry_quaternions()
    q = ori.euler_bunge_to_quat(np.array([37.0]), np.array([52.0]), np.array([131.0]))
    wrong = ori.qmul(sym[13][None], q)
    assert ori.cubic_disorientation_angle(ori.qmul(ori.qconj(q), wrong))[0] > 10.0


def test_cubic_symmetry_operation_has_zero_disorientation():
    q90 = np.array([[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]])
    assert ori.cubic_disorientation_angle(q90)[0] < 1e-4


def test_sigma3_twin_is_60_degrees():
    v = np.sin(np.pi / 6) / np.sqrt(3)
    q60 = np.array([[np.cos(np.pi / 6), v, v, v]])
    assert abs(ori.cubic_disorientation_angle(q60)[0] - 60.0) < 1e-6


def test_disorientation_never_exceeds_cubic_bound():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(500, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    theta = ori.cubic_disorientation_angle(q)
    assert theta.min() >= 0.0
    assert theta.max() <= 62.8 + 1e-6
