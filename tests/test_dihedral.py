"""Exhaustive tests of the dihedral maths.

The groove-angle formula is derived in the module docstring from a wedge of
material filling the angular sector ``[0, theta]`` about the seam axis. That
same construction is reproduced here analytically, which lets every wedge
angle from 5 deg to 355 deg be checked against the closed-form answer -- no
CAD, no tolerances beyond floating point.
"""

from __future__ import annotations

import numpy as np
import pytest

from weldforge.geom import transform as tf
from weldforge.seams import dihedral as dh


def wedge(theta_deg: float):
    """Analytic flanks of a wedge of material filling ``[0, theta]``.

    Seam along +X; directions live in the Y-Z plane measured from +Y.
    Returns ``(n_a, d_a, n_b, d_b)``.
    """
    th = np.deg2rad(theta_deg)

    def yz(angle: float) -> np.ndarray:
        return np.array([0.0, np.cos(angle), np.sin(angle)])

    d_a = yz(0.0)  # into face A
    d_b = yz(th)  # into face B
    n_a = yz(np.deg2rad(270.0))  # outward normal of face A
    n_b = yz(th + np.pi / 2.0)  # outward normal of face B
    return n_a, d_a, n_b, d_b


ALL_ANGLES = [float(a) for a in range(5, 356, 5)]


# ---------------------------------------------------------------------------
# groove angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", ALL_ANGLES)
def test_material_angle_recovers_the_wedge(theta):
    """The whole point: material angle must come back as theta, for every theta."""
    if abs(theta - 180.0) < 1e-9:
        pytest.skip("tangent case is tested separately")
    n_a, d_a, n_b, d_b = wedge(theta)
    assert np.isclose(np.rad2deg(dh.material_angle(n_a, d_a, n_b, d_b)), theta, atol=1e-6)


@pytest.mark.parametrize("theta", ALL_ANGLES)
def test_groove_is_the_complement_of_material(theta):
    n_a, d_a, n_b, d_b = wedge(theta)
    groove, _ = dh.groove_angle(n_a, d_a, n_b, d_b)
    material = dh.material_angle(n_a, d_a, n_b, d_b)
    assert np.isclose(groove + material, 2.0 * np.pi, atol=1e-9)


def test_inside_corner_of_an_l_bracket_is_a_90_degree_groove():
    """theta = 270 -> a fillet seam."""
    groove, concavity = dh.groove_angle(*wedge(270.0))
    assert np.isclose(np.rad2deg(groove), 90.0, atol=1e-6)
    assert concavity == 1


def test_outside_edge_of_a_cube_is_a_270_degree_groove():
    """theta = 90 -> an outside corner, NOT a fillet, even though phi is also 90."""
    groove, concavity = dh.groove_angle(*wedge(90.0))
    assert np.isclose(np.rad2deg(groove), 270.0, atol=1e-6)
    assert concavity == -1


def test_the_two_90_degree_cases_are_distinguished_only_by_concavity():
    """Guards the exact shortcut this module exists to avoid."""
    n_a, d_a, n_b, d_b = wedge(270.0)
    m_a, e_a, m_b, e_b = wedge(90.0)
    phi_concave = tf.angle_between(d_a, d_b)
    phi_convex = tf.angle_between(e_a, e_b)
    assert np.isclose(phi_concave, phi_convex, atol=1e-9)  # same phi...
    assert dh.groove_angle(n_a, d_a, n_b, d_b)[0] != pytest.approx(
        dh.groove_angle(m_a, e_a, m_b, e_b)[0]
    )  # ...different joint


@pytest.mark.parametrize("theta", [5.0, 45.0, 90.0, 135.0, 179.0])
def test_convex_branch_detected(theta):
    assert dh.groove_angle(*wedge(theta))[1] == -1


@pytest.mark.parametrize("theta", [181.0, 225.0, 270.0, 315.0, 355.0])
def test_concave_branch_detected(theta):
    assert dh.groove_angle(*wedge(theta))[1] == 1


def test_tangent_faces_give_180_and_zero_concavity():
    groove, concavity = dh.groove_angle(*wedge(180.0))
    assert np.isclose(np.rad2deg(groove), 180.0)
    assert concavity == 0


def test_shallow_angles_either_side_of_flat_do_not_collapse():
    """179 deg and 181 deg of material are different joints, 2 deg apart."""
    g_179, c_179 = dh.groove_angle(*wedge(179.0))
    g_181, c_181 = dh.groove_angle(*wedge(181.0))
    assert np.isclose(np.rad2deg(g_179), 181.0, atol=1e-6)
    assert np.isclose(np.rad2deg(g_181), 179.0, atol=1e-6)
    assert c_179 == -1 and c_181 == 1


def test_degenerate_inputs_fall_back_to_flat():
    zero = np.zeros(3)
    groove, concavity = dh.groove_angle(zero, zero, zero, zero)
    assert np.isclose(groove, np.pi)
    assert concavity == 0


def test_tangent_tolerance_is_configurable():
    n_a, d_a, n_b, d_b = wedge(178.0)
    assert dh.groove_angle(n_a, d_a, n_b, d_b)[1] == -1
    # A deliberately loose tolerance swallows the 2 deg bevel.
    assert dh.groove_angle(n_a, d_a, n_b, d_b, tangent_tolerance=0.5)[1] == 0


# ---------------------------------------------------------------------------
# bisector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", ALL_ANGLES)
def test_bisector_is_unit_and_square_to_the_seam(theta):
    n_a, d_a, n_b, d_b = wedge(theta)
    b = dh.joint_bisector(n_a, n_b, d_a, d_b)
    assert np.isclose(np.linalg.norm(b), 1.0)
    assert np.isclose(b[0], 0.0, atol=1e-9)  # seam runs along X


@pytest.mark.parametrize("theta", ALL_ANGLES)
def test_bisector_points_into_the_material(theta):
    """The wire must point from the air towards the material, for every wedge.

    Checked by stepping along the bisector and asking whether the resulting
    direction's angle falls inside the material sector ``[0, theta]``.
    """
    if abs(theta - 180.0) < 1e-9:
        pytest.skip("tangent case has no unique material side")
    n_a, d_a, n_b, d_b = wedge(theta)
    b = dh.joint_bisector(n_a, n_b, d_a, d_b)
    angle = np.rad2deg(np.arctan2(b[2], b[1])) % 360.0
    assert 0.0 <= angle <= theta, f"bisector at {angle:.1f} deg is outside [0, {theta}]"


def test_bisector_bisects_the_weld_side_wedge():
    """For a 90 deg fillet the wire points at 45 deg into the corner."""
    n_a, d_a, n_b, d_b = wedge(270.0)
    b = dh.joint_bisector(n_a, n_b, d_a, d_b)
    assert np.isclose(np.rad2deg(np.arctan2(b[2], b[1])) % 360.0, 135.0, atol=1e-6)
    # Equal angles to both flanks.
    assert np.isclose(tf.angle_between(b, -n_a), tf.angle_between(b, -n_b), atol=1e-9)


def test_bisector_for_a_butt_points_straight_down_into_the_gap():
    n_a = n_b = np.array([0.0, 0.0, 1.0])
    b = dh.joint_bisector(n_a, n_b)
    assert np.allclose(b, [0.0, 0.0, -1.0])


def test_bisector_falls_back_when_normals_are_antiparallel():
    """Parallel faces (an edge/slit weld) carry no normal-sum information."""
    n_a = np.array([0.0, 1.0, 0.0])
    n_b = np.array([0.0, -1.0, 0.0])
    d_a = np.array([0.0, 0.0, 1.0])
    d_b = np.array([0.0, 0.0, 1.0])
    b = dh.joint_bisector(n_a, n_b, d_a, d_b)
    assert np.isclose(np.linalg.norm(b), 1.0)
    assert np.allclose(b, [0.0, 0.0, -1.0])


def test_bisector_with_no_fallback_available():
    n_a = np.array([0.0, 1.0, 0.0])
    b = dh.joint_bisector(n_a, -n_a)
    assert np.isclose(np.linalg.norm(b), 1.0)


def test_bisector_of_degenerate_normals():
    b = dh.joint_bisector(np.zeros(3), np.zeros(3))
    assert np.allclose(b, [0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------
# in-face direction
# ---------------------------------------------------------------------------


def test_in_face_direction_sign_flips():
    n = np.array([0.0, 0.0, 1.0])
    t = np.array([1.0, 0.0, 0.0])
    plus = dh.in_face_direction(n, t, +1.0)
    minus = dh.in_face_direction(n, t, -1.0)
    assert np.allclose(plus, -minus)
    assert np.isclose(np.linalg.norm(plus), 1.0)
    assert np.isclose(np.dot(plus, n), 0.0)
    assert np.isclose(np.dot(plus, t), 0.0)


def test_in_face_direction_of_degenerate_inputs():
    d = dh.in_face_direction(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 1.0)
    assert np.isclose(np.linalg.norm(d), 1.0)


# ---------------------------------------------------------------------------
# SeamGeometry aggregate
# ---------------------------------------------------------------------------


def test_seam_geometry_aggregates():
    n = 11
    geom = dh.SeamGeometry(
        points=np.zeros((n, 3)),
        tangents=np.tile([1.0, 0, 0], (n, 1)),
        normal_a=np.zeros((n, 3)),
        normal_b=np.zeros((n, 3)),
        inward_a=np.zeros((n, 3)),
        inward_b=np.zeros((n, 3)),
        groove=np.full(n, np.pi / 2),
        concavity=np.ones(n, dtype=int),
        bisector=np.zeros((n, 3)),
    )
    assert geom.n == n
    assert np.isclose(np.rad2deg(geom.mean_groove), 90.0)
    assert np.isclose(geom.groove_spread, 0.0)
    assert geom.dominant_concavity == 1


def test_seam_geometry_spread_flags_a_varying_joint():
    groove = np.deg2rad(np.linspace(80.0, 140.0, 7))
    geom = dh.SeamGeometry(
        points=np.zeros((7, 3)),
        tangents=np.zeros((7, 3)),
        normal_a=np.zeros((7, 3)),
        normal_b=np.zeros((7, 3)),
        inward_a=np.zeros((7, 3)),
        inward_b=np.zeros((7, 3)),
        groove=groove,
        concavity=np.array([1, 1, 1, -1, -1, 0, 0]),
        bisector=np.zeros((7, 3)),
    )
    assert np.isclose(np.rad2deg(geom.groove_spread), 60.0)
    assert geom.dominant_concavity == 1


def test_empty_seam_geometry():
    geom = dh.SeamGeometry(
        points=np.zeros((0, 3)),
        tangents=np.zeros((0, 3)),
        normal_a=np.zeros((0, 3)),
        normal_b=np.zeros((0, 3)),
        inward_a=np.zeros((0, 3)),
        inward_b=np.zeros((0, 3)),
        groove=np.zeros(0),
        concavity=np.zeros(0, dtype=int),
        bisector=np.zeros((0, 3)),
    )
    assert geom.n == 0
    assert np.isclose(geom.mean_groove, np.pi)
    assert geom.groove_spread == 0.0
    assert geom.dominant_concavity == 0


# ---------------------------------------------------------------------------
# inward direction from the opposite normal (the contact-joint rule)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [185.0, 225.0, 270.0, 315.0, 350.0])
def test_opposite_normal_rule_matches_the_wedge_for_concave_joints(theta):
    """The rule contact detection relies on, checked against the exact wedge."""
    n_a, d_a, n_b, d_b = wedge(theta)
    t = np.array([1.0, 0.0, 0.0])
    got_a = dh.inward_from_opposite_normal(n_a, n_b, t)
    got_b = dh.inward_from_opposite_normal(n_b, n_a, t)
    assert got_a is not None and got_b is not None
    assert np.allclose(got_a, d_a, atol=1e-9)
    assert np.allclose(got_b, d_b, atol=1e-9)


@pytest.mark.parametrize("theta", [10.0, 90.0, 170.0])
def test_opposite_normal_rule_is_reversed_for_convex_joints(theta):
    """Documented limitation: the rule is only valid where the joint is concave."""
    n_a, d_a, n_b, d_b = wedge(theta)
    got_a = dh.inward_from_opposite_normal(n_a, n_b, np.array([1.0, 0.0, 0.0]))
    assert got_a is not None
    assert np.allclose(got_a, -d_a, atol=1e-9)


def test_opposite_normal_rule_gives_up_on_parallel_flanks():
    up = np.array([0.0, 0.0, 1.0])
    t = np.array([1.0, 0.0, 0.0])
    assert dh.inward_from_opposite_normal(up, up, t) is None
    assert dh.inward_from_opposite_normal(up, -up, t) is None


def test_opposite_normal_rule_recovers_a_tee_joint():
    """Base plate top face and web side face: away from the web, up the web."""
    n_base = np.array([0.0, 0.0, 1.0])
    n_web = np.array([0.0, -1.0, 0.0])
    t = np.array([1.0, 0.0, 0.0])
    d_base = dh.inward_from_opposite_normal(n_base, n_web, t)
    d_web = dh.inward_from_opposite_normal(n_web, n_base, t)
    assert np.allclose(d_base, [0.0, -1.0, 0.0])
    assert np.allclose(d_web, [0.0, 0.0, 1.0])
    groove, concavity = dh.groove_angle(n_base, d_base, n_web, d_web)
    assert np.isclose(np.rad2deg(groove), 90.0)
    assert concavity == 1
    # Wire points down into the corner, at 45 deg.
    b = dh.joint_bisector(n_base, n_web)
    assert np.allclose(b, [0.0, np.sqrt(0.5), -np.sqrt(0.5)], atol=1e-9)
