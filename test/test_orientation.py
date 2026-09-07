"""Orientation conventions, checked against values Neper publishes."""

import numpy as np

from festim_microstructure.meshing.ebsd import orientation as ori


def test_self_test_runs():
    ori.self_test()


def test_bunge_0_30_0_matches_neper_table():
    q = ori.euler_bunge_to_quat(np.array([0.0]), np.array([30.0]), np.array([0.0]))
    assert np.allclose(q[0], [0.965925826, 0.258819045, 0, 0], atol=1e-8)
    r = ori.quat_to_rodrigues(
        ori.to_fundamental_zone(q, ori.cubic_symmetry_quaternions())
    )
    assert np.allclose(r[0], [0.267949192, 0, 0], atol=1e-8)


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
