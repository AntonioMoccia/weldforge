"""Round-trip and invariant tests for the SE(3) layer.

Every conversion in ``weldforge.geom.transform`` is exercised over random
rotations plus the degenerate cases that break naive implementations
(identity, 180 deg, gimbal lock, antiparallel vectors).
"""

from __future__ import annotations

import numpy as np
import pytest

from weldforge.geom import transform as tf

RNG = np.random.default_rng(20251122)


def random_rotation(rng: np.random.Generator = RNG) -> np.ndarray:
    """Uniformly distributed rotation via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def random_pose(rng: np.random.Generator = RNG) -> np.ndarray:
    return tf.pose(random_rotation(rng), rng.uniform(-1000, 1000, size=3))


# --- basics -----------------------------------------------------------------


def test_identity_is_a_pose():
    assert tf.is_pose(tf.identity())


def test_random_poses_are_poses():
    for _ in range(50):
        assert tf.is_pose(random_pose())


def test_non_pose_rejected():
    bad = np.eye(4)
    bad[:3, :3] *= 2.0  # scaling is not rigid
    assert not tf.is_pose(bad)
    assert not tf.is_pose(np.eye(3))


def test_inverse_matches_linalg_inv():
    for _ in range(30):
        T = random_pose()
        assert np.allclose(tf.inverse(T), np.linalg.inv(T), atol=1e-9)
        assert np.allclose(tf.compose(T, tf.inverse(T)), np.eye(4), atol=1e-9)


def test_compose_is_left_to_right():
    A, B, C = random_pose(), random_pose(), random_pose()
    assert np.allclose(tf.compose(A, B, C), A @ B @ C)


def test_apply_point_vs_vector():
    T = tf.pose(tf.rot_z(0.3), [10.0, 20.0, 30.0])
    p = np.array([1.0, 0.0, 0.0])
    assert np.allclose(tf.apply_point(T, p), tf.rot_z(0.3) @ p + [10, 20, 30])
    assert np.allclose(tf.apply_vector(T, p), tf.rot_z(0.3) @ p)


def test_apply_points_batch():
    T = random_pose()
    pts = RNG.uniform(-100, 100, size=(25, 3))
    expected = np.array([tf.apply_point(T, p) for p in pts])
    assert np.allclose(tf.apply_points(T, pts), expected)


# --- quaternions ------------------------------------------------------------


def test_quat_matrix_roundtrip():
    for _ in range(200):
        R = random_rotation()
        assert np.allclose(tf.quat_to_matrix(tf.matrix_to_quat(R)), R, atol=1e-9)


def test_quat_roundtrip_at_180_degrees():
    """The trace formula degenerates here; Shepperd's method must not."""
    for axis in np.eye(3):
        R = tf.rot_axis_angle(axis, np.pi)
        assert np.allclose(tf.quat_to_matrix(tf.matrix_to_quat(R)), R, atol=1e-9)


def test_quat_is_canonical_hemisphere():
    for _ in range(100):
        assert tf.matrix_to_quat(random_rotation())[3] >= 0.0


def test_quat_multiply_matches_matrix_product():
    for _ in range(50):
        Ra, Rb = random_rotation(), random_rotation()
        qa, qb = tf.matrix_to_quat(Ra), tf.matrix_to_quat(Rb)
        assert np.allclose(tf.quat_to_matrix(tf.quat_multiply(qa, qb)), Ra @ Rb, atol=1e-9)


# --- euler ------------------------------------------------------------------


def test_rpy_roundtrip():
    for _ in range(200):
        R = random_rotation()
        roll, pitch, yaw = tf.matrix_to_rpy(R)
        assert np.allclose(tf.rpy_to_matrix(roll, pitch, yaw), R, atol=1e-9)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_rpy_at_gimbal_lock(sign):
    R = tf.rpy_to_matrix(0.4, sign * np.pi / 2, 0.7)
    roll, pitch, yaw = tf.matrix_to_rpy(R)
    assert np.allclose(tf.rpy_to_matrix(roll, pitch, yaw), R, atol=1e-7)
    assert yaw == 0.0


# --- axis/angle -------------------------------------------------------------


def test_axis_angle_roundtrip():
    for _ in range(200):
        R = random_rotation()
        axis, angle = tf.axis_angle_of(R)
        assert np.allclose(tf.rot_axis_angle(axis, angle), R, atol=1e-8)


def test_axis_angle_of_identity():
    axis, angle = tf.axis_angle_of(np.eye(3))
    assert angle == 0.0
    assert np.isclose(np.linalg.norm(axis), 1.0)


def test_axis_angle_at_180_recovers_axis():
    for axis_in in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), tf.unit([1.0, 1.0, 0.3])):
        R = tf.rot_axis_angle(axis_in, np.pi)
        axis_out, angle = tf.axis_angle_of(R)
        assert np.isclose(angle, np.pi)
        # Axis sign is arbitrary at 180 deg; the rotation must still match.
        assert np.allclose(tf.rot_axis_angle(axis_out, angle), R, atol=1e-8)


# --- interpolation ----------------------------------------------------------


def test_slerp_endpoints():
    """Endpoints must be recovered *as rotations*.

    ``slerp`` may negate the far quaternion to take the short arc, and ``-q``
    is the same rotation as ``q``, so comparing raw components would be wrong.
    """
    R0, R1 = random_rotation(), random_rotation()
    q0, q1 = tf.matrix_to_quat(R0), tf.matrix_to_quat(R1)
    assert np.allclose(tf.quat_to_matrix(tf.slerp(q0, q1, 0.0)), R0, atol=1e-9)
    assert np.allclose(tf.quat_to_matrix(tf.slerp(q0, q1, 1.0)), R1, atol=1e-9)


def test_slerp_takes_short_arc():
    R0 = np.eye(3)
    R1 = tf.rot_z(np.deg2rad(350.0))
    q_mid = tf.slerp(tf.matrix_to_quat(R0), tf.matrix_to_quat(R1), 0.5)
    _, angle = tf.axis_angle_of(tf.quat_to_matrix(q_mid))
    # Short way round is -10 deg, so the midpoint must be 5 deg, not 175 deg.
    assert np.isclose(np.rad2deg(angle), 5.0, atol=1e-6)


def test_interpolate_pose_midpoint_position():
    T0 = tf.pose(np.eye(3), [0.0, 0.0, 0.0])
    T1 = tf.pose(tf.rot_x(0.5), [100.0, 0.0, 0.0])
    Tm = tf.interpolate_pose(T0, T1, 0.5)
    assert np.allclose(tf.translation_of(Tm), [50.0, 0.0, 0.0])
    assert tf.is_pose(Tm)


# --- frame construction -----------------------------------------------------


def test_frame_from_z_x_axes():
    z = tf.unit([0.0, 0.0, 1.0])
    T = tf.frame_from_z_x(z, [1.0, 0.0, 0.0], [5.0, 6.0, 7.0])
    assert np.allclose(T[:3, 2], z)
    assert np.allclose(T[:3, 0], [1.0, 0.0, 0.0])
    assert np.allclose(tf.translation_of(T), [5.0, 6.0, 7.0])
    assert tf.is_pose(T)


def test_frame_from_z_x_projects_a_skew_hint():
    """A hint that is not perpendicular to Z must still give an orthonormal frame."""
    T = tf.frame_from_z_x([0.0, 0.0, 1.0], [1.0, 0.0, 5.0], [0.0, 0.0, 0.0])
    assert tf.is_pose(T)
    assert np.allclose(T[:3, 0], [1.0, 0.0, 0.0])


def test_frame_from_z_x_survives_parallel_hint():
    T = tf.frame_from_z_x([0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0])
    assert tf.is_pose(T)
    assert np.allclose(T[:3, 2], [0.0, 0.0, 1.0])


def test_frame_from_z_x_rejects_degenerate_z():
    with pytest.raises(ValueError):
        tf.frame_from_z_x([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_frame_from_x_z_keeps_x_exact():
    x = tf.unit([1.0, 1.0, 0.0])
    T = tf.frame_from_x_z(x, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    assert np.allclose(T[:3, 0], x)
    assert tf.is_pose(T)


def test_look_at_points_z_at_target():
    T = tf.look_at([0.0, 0.0, 100.0], [0.0, 0.0, 0.0])
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0])


# --- misc -------------------------------------------------------------------


def test_rotation_between_generic():
    for _ in range(100):
        a = RNG.normal(size=3)
        b = RNG.normal(size=3)
        R = tf.rotation_between(a, b)
        assert np.allclose(R @ tf.unit(a), tf.unit(b), atol=1e-9)


def test_rotation_between_antiparallel():
    a = np.array([0.0, 0.0, 1.0])
    R = tf.rotation_between(a, -a)
    assert np.allclose(R @ a, -a, atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_between_parallel_is_identity():
    a = np.array([0.0, 1.0, 0.0])
    assert np.allclose(tf.rotation_between(a, 2.0 * a), np.eye(3))


def test_orthonormalize_fixes_drift():
    R = random_rotation()
    drifted = R + RNG.normal(scale=1e-3, size=(3, 3))
    fixed = tf.orthonormalize(drifted)
    assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(fixed), 1.0)


def test_unit_rejects_zero():
    assert tf.unit([0.0, 0.0, 0.0]) is None


def test_angle_between_known_values():
    assert np.isclose(tf.angle_between([1, 0, 0], [0, 1, 0]), np.pi / 2)
    assert np.isclose(tf.angle_between([1, 0, 0], [-1, 0, 0]), np.pi)
    assert np.isclose(tf.angle_between([1, 0, 0], [2, 0, 0]), 0.0)


def test_pose_distance():
    T0 = tf.pose(np.eye(3), [0, 0, 0])
    T1 = tf.pose(tf.rot_z(np.pi / 4), [30.0, 40.0, 0.0])
    lin, ang = tf.pose_distance(T0, T1)
    assert np.isclose(lin, 50.0)
    assert np.isclose(ang, np.pi / 4)
