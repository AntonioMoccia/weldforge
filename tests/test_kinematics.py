"""Kinematics tests.

The decisive test is the **round trip**: take random joint values, run forward
kinematics, solve inverse kinematics on the result, and require the pose back.
It cannot be satisfied by a solver that is merely self-consistent -- which is
what the two bugs found while writing this module both were. The first had the
elbow relation guessed instead of derived; the second read ``cos(theta5)`` from
the wrong element of the wrist rotation. Both produced plausible joint angles
and put the torch in the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from weldforge.geom import transform as tf
from weldforge.kin import (
    JointSpec,
    RobotModel,
    available_robots,
    extract_arm_geometry,
    fk,
    fk_all,
    jacobian,
    load_robot,
    manipulability,
    nearest_solution,
    solve,
    solve_analytic,
    solve_numeric,
    torch_frame,
)

ROBOT_NAMES = available_robots()
RNG = np.random.default_rng(20251122)


def random_joints(robot: RobotModel, margin: float = 0.85) -> np.ndarray:
    """Random joints inside a margin of the limits, away from the stops."""
    return RNG.uniform(robot.limits_lo * margin, robot.limits_hi * margin)


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------


def test_library_has_the_expected_robots():
    assert {"kuka_kr16_2", "abb_irb1520id", "fanuc_arcmate_100id", "generic_6axis"} <= set(
        ROBOT_NAMES
    )


def test_unknown_robot_names_the_alternatives():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_robot("acme_weldotron")


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_every_robot_is_a_six_axis_with_sane_limits(name):
    robot = load_robot(name)
    assert robot.n_joints == 6
    assert np.all(robot.limits_hi > robot.limits_lo)
    assert np.all(robot.max_velocities > 0.0)
    assert robot.reach_mm > 0.0
    # Nominal geometry must be flagged as needing calibration against the cell.
    assert any("Confirm" in note or "Round-number" in note for note in robot.notes)


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_published_reach_matches_the_geometry(name):
    """Fully extended, the arm must reach what the data sheet claims.

    Measured horizontally from axis 1 to the wrist centre, which is how robot
    makers quote reach -- not as the distance from the origin, which would
    include the base height.
    """
    robot = load_robot(name)
    geometry = extract_arm_geometry(robot)
    assert geometry is not None
    extended = geometry.a1 + geometry.a2 + geometry.forearm
    assert np.isclose(extended, robot.reach_mm, rtol=0.12), (
        f"{name}: geometry reaches {extended:.0f} mm, data sheet says {robot.reach_mm:.0f}"
    )


# ---------------------------------------------------------------------------
# forward kinematics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_fk_returns_valid_poses(name):
    robot = load_robot(name)
    for _ in range(50):
        pose = fk(robot, random_joints(robot))
        assert tf.is_pose(pose)


def test_fk_all_frame_count_and_chaining():
    robot = load_robot("generic_6axis")
    frames = fk_all(robot, np.zeros(6))
    # base + one per joint + tool
    assert len(frames) == robot.n_joints + 2
    assert np.allclose(frames[0], robot.base)
    assert np.allclose(frames[-1], frames[-2] @ robot.tool)


def test_fk_rejects_wrong_joint_count():
    robot = load_robot("generic_6axis")
    with pytest.raises(ValueError, match="6 joints"):
        fk(robot, np.zeros(4))


def test_base_transform_moves_the_whole_robot():
    offset = tf.pose(tf.rot_z(0.4), [1000.0, -500.0, 250.0])
    plain = load_robot("generic_6axis")
    mounted = load_robot("generic_6axis", base=offset)
    q = random_joints(plain)
    assert np.allclose(fk(mounted, q), offset @ fk(plain, q))


def test_tool_transform_moves_only_the_tcp():
    tool = torch_frame()
    plain = load_robot("generic_6axis")
    with_torch = load_robot("generic_6axis", tool=tool)
    q = random_joints(plain)
    assert np.allclose(fk(with_torch, q), fk(plain, q) @ tool)
    # The flange is unaffected.
    assert np.allclose(fk(with_torch, q, with_tool=False), fk(plain, q, with_tool=False))


def test_torch_frame_geometry():
    """Z along the wire, origin at the tip, bent by the gooseneck angle."""
    frame = torch_frame(length_mm=300.0, angle_deg=22.0, stickout_mm=15.0)
    assert tf.is_pose(frame)
    # The tip is length + stickout away from the flange.
    assert np.isclose(np.linalg.norm(tf.translation_of(frame)), 315.0, atol=1e-6)
    # Z is tilted off the flange axis by the gooseneck angle.
    assert np.isclose(np.rad2deg(tf.angle_between(frame[:3, 2], [0.0, 0.0, 1.0])), 22.0, atol=1e-6)


def test_straight_torch_points_along_the_flange():
    frame = torch_frame(angle_deg=0.0, length_mm=400.0, stickout_mm=20.0)
    assert np.allclose(tf.translation_of(frame), [0.0, 0.0, 420.0])
    assert np.allclose(frame[:3, 2], [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_jacobian_matches_finite_differences(name):
    """The analytic Jacobian must agree with a numerical derivative of the FK."""
    robot = load_robot(name)
    q = random_joints(robot, margin=0.6)
    J = jacobian(robot, q)
    eps = 1e-7
    base_pose = fk(robot, q)

    for i in range(robot.n_joints):
        dq = np.zeros(robot.n_joints)
        dq[i] = eps
        moved = fk(robot, q + dq)
        linear = (tf.translation_of(moved) - tf.translation_of(base_pose)) / eps
        R_delta = tf.rotation_of(moved) @ tf.rotation_of(base_pose).T
        axis, angle = tf.axis_angle_of(R_delta)
        angular = axis * angle / eps
        assert np.allclose(J[:3, i], linear, atol=1e-3)
        assert np.allclose(J[3:, i], angular, atol=1e-3)


def test_manipulability_collapses_at_a_stretched_singularity():
    """Arm fully extended: the wrist cannot move along the arm axis any more."""
    robot = load_robot("generic_6axis")
    good = manipulability(robot, np.deg2rad([0.0, -45.0, 45.0, 0.0, 60.0, 0.0]))
    # Elbow straight: upper arm and forearm colinear.
    straight = manipulability(robot, np.deg2rad([0.0, 0.0, -90.0, 0.0, 60.0, 0.0]))
    assert 0.0 <= straight < good


def test_manipulability_is_zero_at_a_wrist_singularity():
    robot = load_robot("generic_6axis")
    value = manipulability(
        robot, np.deg2rad([0.0, -45.0, 45.0, 0.0, 0.0, 0.0]), translation_only=False
    )
    assert value < 0.05


# ---------------------------------------------------------------------------
# analytic IK: the round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_analytic_ik_round_trip(name):
    """FK then IK must return the pose, to machine precision, every time."""
    robot = load_robot(name)
    solved = 0
    attempts = 150
    for _ in range(attempts):
        q = random_joints(robot)
        target = fk(robot, q)
        valid = [s for s in solve_analytic(robot, target) if s.ok]
        if not valid:
            continue
        solved += 1
        best = min(valid, key=lambda s: s.position_error_mm)
        assert best.position_error_mm < 1e-6
        assert best.orientation_error_deg < 1e-4
        assert robot.within_limits(best.joints)
    assert solved == attempts, f"{name}: only {solved}/{attempts} poses solved"


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_analytic_ik_returns_several_configurations(name):
    """Multiple branches are what let a path stay on one configuration."""
    robot = load_robot(name)
    counts = []
    for _ in range(40):
        target = fk(robot, random_joints(robot, margin=0.6))
        counts.append(len([s for s in solve_analytic(robot, target) if s.ok]))
    assert max(counts) >= 2
    assert np.mean(counts) > 1.5


def test_analytic_ik_branch_names_cover_the_eight_combinations():
    robot = load_robot("generic_6axis")
    target = fk(robot, np.deg2rad([10.0, -50.0, 40.0, 20.0, 50.0, -30.0]))
    branches = {s.branch for s in solve_analytic(robot, target)}
    assert len(branches) == 8
    assert {"front/up/noflip", "back/down/flip"} <= branches


def test_analytic_ik_works_with_a_torch_attached():
    """The TCP is at the wire tip, 300-odd mm off the flange on a bent neck."""
    robot = load_robot("kuka_kr16_2", tool=torch_frame())
    for _ in range(60):
        target = fk(robot, random_joints(robot, margin=0.8))
        valid = [s for s in solve_analytic(robot, target) if s.ok]
        assert valid
        assert min(s.position_error_mm for s in valid) < 1e-6


def test_analytic_ik_works_with_the_robot_mounted_off_origin():
    base = tf.pose(tf.rot_z(1.1), [-800.0, 400.0, 300.0])
    robot = load_robot("abb_irb1520id", base=base, tool=torch_frame())
    for _ in range(40):
        target = fk(robot, random_joints(robot, margin=0.8))
        assert any(s.ok for s in solve_analytic(robot, target))


def test_unreachable_target_yields_no_valid_solution():
    robot = load_robot("generic_6axis")
    far = tf.pose(np.eye(3), [8000.0, 0.0, 500.0])
    assert not [s for s in solve_analytic(robot, far) if s.ok]


def test_analytic_ik_tries_full_turn_aliases():
    """An industrial wrist travels past +-180 deg; the principal arctangent
    value may be outside limits while an equivalent angle a turn away is not.
    Discarding those shows up as seams the real robot welds without complaint."""
    robot = load_robot("fanuc_arcmate_100id")  # J6 reaches +-450 deg
    solved = 0
    for _ in range(80):
        target = fk(robot, random_joints(robot, margin=0.95))
        if any(s.ok for s in solve_analytic(robot, target)):
            solved += 1
    assert solved >= 78


# ---------------------------------------------------------------------------
# structure detection
# ---------------------------------------------------------------------------


def offset_wrist_robot() -> RobotModel:
    """A chain whose wrist axes do not intersect: no closed form exists."""
    return RobotModel(
        name="offset-wrist",
        wrist="offset",
        joints=[
            JointSpec("J1", 0.0, 0.0, 400.0, limit_lo=-np.pi, limit_hi=np.pi),
            JointSpec("J2", -np.pi / 2, 150.0, 0.0, limit_lo=-2.0, limit_hi=2.0),
            JointSpec("J3", 0.0, 600.0, 0.0, limit_lo=-2.5, limit_hi=2.5),
            JointSpec("J4", -np.pi / 2, 80.0, 500.0, limit_lo=-3.0, limit_hi=3.0),
            JointSpec("J5", np.pi / 2, 60.0, 0.0, limit_lo=-2.2, limit_hi=2.2),
            JointSpec("J6", -np.pi / 2, 0.0, 120.0, limit_lo=-3.0, limit_hi=3.0),
        ],
    )


def test_offset_wrist_is_detected_and_refused():
    """Better to refuse than to apply a closed form the geometry breaks."""
    robot = offset_wrist_robot()
    assert extract_arm_geometry(robot) is None
    assert solve_analytic(robot, fk(robot, np.zeros(6))) == []


def test_non_six_axis_is_refused():
    robot = RobotModel(
        name="scara",
        joints=[JointSpec(f"J{i}", 0.0, 100.0, 0.0) for i in range(4)],
    )
    assert extract_arm_geometry(robot) is None


# ---------------------------------------------------------------------------
# numeric IK
# ---------------------------------------------------------------------------


def test_numeric_ik_solves_the_offset_wrist():
    robot = offset_wrist_robot()
    solved = 0
    for _ in range(25):
        q = RNG.uniform(robot.limits_lo * 0.7, robot.limits_hi * 0.7)
        solution = solve_numeric(robot, fk(robot, q))
        if solution is not None and solution.ok:
            solved += 1
    assert solved >= 20


def test_numeric_ik_respects_joint_limits():
    robot = load_robot("generic_6axis")
    target = fk(robot, random_joints(robot, margin=0.5))
    solution = solve_numeric(robot, target)
    assert solution is not None
    assert robot.within_limits(solution.joints)


def test_numeric_ik_can_be_seeded_for_speed():
    robot = load_robot("generic_6axis")
    q = random_joints(robot, margin=0.5)
    target = fk(robot, q)
    seeded = solve_numeric(robot, target, seed=q + 0.01)
    assert seeded is not None and seeded.ok
    assert seeded.iterations < 30


def test_numeric_ik_reports_how_far_out_of_reach():
    robot = load_robot("generic_6axis")
    solution = solve_numeric(robot, tf.pose(np.eye(3), [9000.0, 0.0, 0.0]))
    assert solution is not None
    assert not solution.ok
    # The residual must be informative, not just a failure flag.
    assert solution.position_error_mm > 1000.0


# ---------------------------------------------------------------------------
# solve(): dispatch and continuity
# ---------------------------------------------------------------------------


def test_solve_prefers_the_closed_form():
    robot = load_robot("generic_6axis")
    solution = solve(robot, fk(robot, random_joints(robot, margin=0.6)))
    assert solution is not None and solution.method == "analytic"


def test_solve_falls_back_to_numeric():
    robot = offset_wrist_robot()
    q = RNG.uniform(robot.limits_lo * 0.6, robot.limits_hi * 0.6)
    solution = solve(robot, fk(robot, q))
    assert solution is not None and solution.method == "numeric"


def test_solve_keeps_the_configuration_across_a_path():
    """The property that makes welding paths usable.

    Stepping along a short line, each station seeded with the previous answer,
    the arm must not reconfigure. A mid-bead configuration change means the
    robot stops, swings round and restarts -- ruining the weld.
    """
    robot = load_robot("kuka_kr16_2", tool=torch_frame())
    start = fk(robot, np.deg2rad([15.0, -60.0, 70.0, 10.0, 50.0, -20.0]))
    previous = None
    branches = set()
    joint_jumps = []

    for i in range(40):
        target = tf.pose(tf.rotation_of(start), tf.translation_of(start) + [0.0, i * 2.0, 0.0])
        solution = solve(robot, target, seed=previous)
        assert solution is not None and solution.ok
        branches.add(solution.branch)
        if previous is not None:
            joint_jumps.append(float(np.max(np.abs(solution.joints - previous))))
        previous = solution.joints

    assert len(branches) == 1, f"configuration changed mid-path: {branches}"
    # 2 mm of travel must never need more than a few degrees at any joint.
    assert max(joint_jumps) < np.deg2rad(5.0)


def test_solve_honours_a_requested_branch():
    robot = load_robot("generic_6axis")
    target = fk(robot, np.deg2rad([10.0, -50.0, 40.0, 20.0, 50.0, -30.0]))
    valid = {s.branch for s in solve_analytic(robot, target) if s.ok}
    choice = sorted(valid)[0]
    solution = solve(robot, target, prefer_branch=choice)
    assert solution is not None and solution.branch == choice


def test_nearest_solution_weights_the_major_axes():
    """A few degrees at axis 1 is a much bigger move than at the wrist."""
    from weldforge.kin.ik import IKSolution

    seed = np.zeros(6)
    base_swing = IKSolution(joints=np.array([0.5, 0, 0, 0, 0, 0]))
    wrist_twist = IKSolution(joints=np.array([0, 0, 0, 0, 0, 0.9]))
    assert nearest_solution([base_swing, wrist_twist], seed) is wrist_twist


def test_solve_returns_the_closest_attempt_when_out_of_reach():
    robot = load_robot("generic_6axis")
    solution = solve(robot, tf.pose(np.eye(3), [6000.0, 0.0, 0.0]))
    assert solution is not None
    assert not solution.ok
    assert solution.position_error_mm > 100.0


# ---------------------------------------------------------------------------
# singular postures
# ---------------------------------------------------------------------------


def test_wrist_singularity_still_returns_a_usable_solution():
    """theta5 = 0 leaves only theta4 + theta6 determined; the solver must not fail."""
    robot = load_robot("generic_6axis")
    q = np.deg2rad([20.0, -50.0, 45.0, 30.0, 0.0, -30.0])
    target = fk(robot, q)
    valid = [s for s in solve_analytic(robot, target) if s.ok]
    assert valid
    assert min(s.position_error_mm for s in valid) < 1e-6


def test_target_on_the_base_axis_is_handled():
    """The wrist centre directly above axis 1 leaves theta1 undetermined."""
    robot = load_robot("generic_6axis")
    target = tf.pose(tf.rot_y(np.pi), [0.0, 0.0, 1400.0])
    solutions = solve_analytic(robot, target)
    assert solutions  # must not crash or return nothing at all


# ---------------------------------------------------------------------------
# limits and reporting
# ---------------------------------------------------------------------------


def test_limit_margin_is_lowest_at_a_stop():
    robot = load_robot("generic_6axis")
    assert np.isclose(robot.limit_margin(robot.limits_lo), 0.0, atol=1e-9)
    mid = (robot.limits_lo + robot.limits_hi) / 2.0
    assert robot.limit_margin(mid) > 0.45


def test_limit_violations_are_named():
    robot = load_robot("generic_6axis")
    q = robot.limits_hi + 0.5
    violations = robot.limit_violations(q)
    assert len(violations) == robot.n_joints
    assert "J1" in violations[0]


def test_clamp_brings_joints_inside():
    robot = load_robot("generic_6axis")
    assert robot.within_limits(robot.clamp(robot.limits_hi * 3.0))


def test_robot_summary_is_serialisable():
    import json

    payload = json.loads(json.dumps(load_robot("kuka_kr16_2").summary()))
    assert payload["n_joints"] == 6
    assert payload["vendor"] == "KUKA"


# ---------------------------------------------------------------------------
# actionable failure reporting
# ---------------------------------------------------------------------------


def test_unreachable_target_says_out_of_reach():
    robot = load_robot("generic_6axis")
    solution = solve(robot, tf.pose(np.eye(3), [9000.0, 0.0, 500.0]))
    assert solution is not None and not solution.ok
    assert "out of reach" in solution.reason
    assert "mm" in solution.reason


def test_a_joint_limit_failure_is_distinguished_from_out_of_reach():
    """Three failures need three different fixes, so they must read differently.

    "0 mm short" told an operator nothing: a target can be geometrically within
    reach and still unreachable because an axis runs out of travel.
    """
    from weldforge.kin.ik import IKSolution

    out_of_reach = IKSolution(joints=np.zeros(6), position_error_mm=240.0)
    at_a_stop = IKSolution(joints=np.zeros(6), within_limits=False)
    wrist = IKSolution(joints=np.zeros(6), orientation_error_deg=37.0)
    fine = IKSolution(joints=np.zeros(6))

    assert "out of reach" in out_of_reach.reason
    assert "travel" in at_a_stop.reason
    assert "orientation" in wrist.reason
    assert fine.reason == ""
    assert len({out_of_reach.reason, at_a_stop.reason, wrist.reason}) == 3
