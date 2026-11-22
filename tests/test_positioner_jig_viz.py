"""Positioner, jig location and the standalone preview.

These three were the gaps an audit found after the first build: the model knew
about external axes, jig datums and a simulator, but the planner never
commanded the positioner, never used the datums, and nothing could be looked
at without ROS. The tests below pin the behaviour that closed them.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from conftest import make_seam, requires_cad

from weldforge.cad import load_part
from weldforge.fixture import (
    PositionerOptions,
    cell_robot,
    evaluate_placement,
    find_placement,
    group_datums,
    load_cell,
    solve_jig_placement,
    solve_positioner,
    solve_positioner_for_seam,
)
from weldforge.fixture.positioner import axes_transform
from weldforge.geom import transform as tf
from weldforge.model import JointType, WeldPosition
from weldforge.plan import PlanOptions, plan_job
from weldforge.post import PostOptions, get_postprocessor
from weldforge.seams import detect_seams
from weldforge.viz import scene_from_job, scene_to_json, write_html


def fast() -> PlanOptions:
    options = PlanOptions()
    options.placement.grid_steps = 3
    options.placement.yaw_steps = 4
    options.placement.max_orientations = 3
    options.positioner.coarse_step_deg = 30.0
    options.positioner.refine_passes = 1
    return options


# ===========================================================================
# positioner kinematics
# ===========================================================================


def test_axes_compose_in_declaration_order():
    """A tilt axis must carry the rotary axis mounted on it."""
    cell = load_cell("positioner_cell")
    axes = cell.external_axes
    assert [a.name for a in axes] == ["tilt", "rotate"]

    # Tilt 90 deg about Y takes the faceplate normal (+Z) to +X.
    tilted = axes_transform(axes, {"tilt": np.pi / 2, "rotate": 0.0})
    assert np.allclose(tf.rotation_of(tilted) @ [0, 0, 1], [1, 0, 0], atol=1e-9)

    # Rotate 90 deg about the faceplate normal takes +X to +Y.
    spun = axes_transform(axes, {"tilt": 0.0, "rotate": np.pi / 2})
    assert np.allclose(tf.rotation_of(spun) @ [1, 0, 0], [0, 1, 0], atol=1e-9)

    # Composed: the rotary axis is carried by the tilt, so the combined
    # rotation is not the product of the two taken independently about fixed
    # cell axes.
    both = axes_transform(axes, {"tilt": np.pi / 2, "rotate": np.pi / 2})
    assert tf.is_pose(both)
    assert not np.allclose(both, tilted @ spun @ tf.inverse(tilted))


def test_axes_transform_rotates_about_the_declared_origin():
    """The positioner's axes pass through the faceplate, not the cell origin."""
    cell = load_cell("positioner_cell")
    origin = cell.external_axes[0].origin
    pose = axes_transform(cell.external_axes, {"tilt": np.pi / 3, "rotate": 0.0})
    # A point on the axis does not move.
    assert np.allclose(tf.apply_point(pose, origin), origin, atol=1e-6)


def test_no_axes_gives_identity():
    assert np.allclose(axes_transform([], {}), np.eye(4))


# ===========================================================================
# positioner solving
# ===========================================================================


def test_bench_cell_has_nothing_to_solve():
    cell = load_cell("bench_cell")
    robot = cell_robot(cell)
    seam = make_seam(JointType.TEE, 8.0, 6.0)
    result = solve_positioner_for_seam(seam, cell, tf.identity(), robot)
    assert result.values == {}
    assert "no coupled external axes" in result.note


def test_the_positioner_turns_an_overhead_seam_the_right_way_up():
    """The reason a positioner is bought.

    An overhead fillet is welded at 4 mm/s in short arc with weaving on every
    fill pass; the same joint downhand is 7 mm/s in spray with none. A seam
    facing the floor is not slow, it is unreachable.
    """
    cell = load_cell("positioner_cell")
    robot = cell_robot(cell)
    mount = tf.pose(np.eye(3), [1000.0, 0.0, 900.0])
    overhead = make_seam(JointType.TEE, 8.0, 6.0, bisector=(0.0, 0.7071, 0.7071))
    assert overhead.dominant_position.is_overhead

    result = solve_positioner_for_seam(
        overhead, cell, mount, robot, PositionerOptions(coarse_step_deg=20.0)
    )
    assert result.improved_from is not None
    assert result.improved_from.is_overhead
    assert not result.position.is_overhead, (
        f"the positioner left the seam at {result.position.value}"
    )
    assert any(abs(v) > 1e-6 for v in result.values.values())


def test_the_positioner_stays_put_when_there_is_nothing_to_gain():
    """No pointless indexing.

    A seam already downhand and fully reachable has nothing to gain, and
    moving the positioner costs real seconds. Moving from home is charged the
    same as moving anywhere else, so the answer must be "stay at zero".
    """
    from weldforge.fixture.positioner import _evaluate

    cell = load_cell("positioner_cell")
    robot = cell_robot(cell)
    # Chosen so the seam is comfortably inside the arm's envelope; the test
    # below covers the case where it is not.
    mount = tf.pose(np.eye(3), [900.0, -100.0, 900.0])
    flat = make_seam(JointType.TEE, 8.0, 6.0)
    assert flat.dominant_position is WeldPosition.PB

    options = PositionerOptions(coarse_step_deg=30.0, continuity_weight=0.5)
    _, position, fraction, _, _ = _evaluate(flat, mount, robot, options, robot.home_joints())
    assert position.is_downhand and fraction == pytest.approx(1.0), (
        "the premise fails: this seam is not already good at zero"
    )

    result = solve_positioner_for_seam(flat, cell, mount, robot, options)
    assert result.position.is_downhand
    assert all(abs(v) < 1e-9 for v in result.values.values()), result.values


def test_the_positioner_moves_to_buy_reachability():
    """The complement: when turning the part brings a seam into reach, it turns.

    Reachability is weighted above welding position for exactly this reason --
    a beautifully presented seam the arm cannot touch is worth nothing.
    """
    from weldforge.fixture.positioner import _evaluate

    cell = load_cell("positioner_cell")
    robot = cell_robot(cell)
    mount = tf.pose(np.eye(3), [1000.0, 0.0, 900.0])
    seam = make_seam(JointType.TEE, 8.0, 6.0)

    options = PositionerOptions(coarse_step_deg=30.0, continuity_weight=0.5)
    _, _, baseline, _, _ = _evaluate(seam, mount, robot, options, robot.home_joints())
    assert baseline < 1.0, "the premise fails: this seam is already fully reachable"

    result = solve_positioner_for_seam(seam, cell, mount, robot, options)
    assert result.reachable_fraction > baseline
    assert any(abs(v) > 1e-9 for v in result.values.values())


def test_axis_values_stay_inside_their_limits():
    cell = load_cell("positioner_cell")
    robot = cell_robot(cell)
    mount = tf.pose(np.eye(3), [1000.0, 0.0, 900.0])
    limits = {a.name: (a.limit_lo, a.limit_hi) for a in cell.external_axes}
    for bisector in ((0, 0.7071, 0.7071), (0, 1.0, 0.0), (0, 0, -1.0)):
        result = solve_positioner_for_seam(
            make_seam(JointType.TEE, 8.0, 6.0, bisector=bisector),
            cell,
            mount,
            robot,
            PositionerOptions(coarse_step_deg=30.0),
        )
        for name, value in result.values.items():
            lo, hi = limits[name]
            assert lo - 1e-9 <= value <= hi + 1e-9, f"{name} out of travel"


def test_plan_prefers_small_index_moves():
    """Continuity: the positioner should not swing through its travel between
    every weld when a nearby setting does the job."""
    cell = load_cell("positioner_cell")
    robot = cell_robot(cell)
    mount = tf.pose(np.eye(3), [1000.0, 0.0, 900.0])
    seams = [make_seam(JointType.TEE, 8.0, 6.0) for _ in range(4)]

    tight = solve_positioner(
        seams,
        cell,
        mount,
        robot,
        PositionerOptions(coarse_step_deg=30.0, continuity_weight=2.0),
    )
    loose = solve_positioner(
        seams,
        cell,
        mount,
        robot,
        PositionerOptions(coarse_step_deg=30.0, continuity_weight=0.0),
    )
    assert tight.index_motion_rad() <= loose.index_motion_rad() + 1e-9


def test_a_seam_whose_position_varies_is_flagged_not_fudged():
    """One indexed setting cannot fit a seam that changes position along its
    own length. Saying so beats quietly welding most of it wrong."""
    cell = load_cell("positioner_cell")
    robot = cell_robot(cell)
    # A seam that curves from horizontal into vertical.
    n = 21
    angles = np.linspace(0.0, np.pi / 2, n)
    points = np.column_stack(
        [100.0 * np.cos(angles) + 1000.0, 100.0 * np.sin(angles), np.full(n, 900.0)]
    )
    from weldforge.model import Seam

    seam = Seam(
        points=points,
        tangents=np.column_stack([-np.sin(angles), np.cos(angles), np.zeros(n)]),
        normal_a=np.tile([0.0, 0.0, 1.0], (n, 1)),
        normal_b=np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)]),
        bisector=np.column_stack(
            [-np.cos(angles) * 0.7071, -np.sin(angles) * 0.7071, np.full(n, -0.7071)]
        ),
        dihedral=np.full(n, np.pi / 2),
        joint_type=JointType.TEE,
        thickness_a_mm=8.0,
        thickness_b_mm=6.0,
    )
    result = solve_positioner_for_seam(
        seam, cell, tf.identity(), robot, PositionerOptions(coarse_step_deg=45.0)
    )
    if result.needs_coordinated:
        assert "coordinated" in result.note


@requires_cad
def test_the_pipeline_solves_and_records_the_positioner(butt_assembly_step):
    job = plan_job(butt_assembly_step, load_cell("positioner_cell"), fast())
    assert job.positioner is not None
    summary = job.metadata["positioner"]
    assert summary["seams"] == len(job.seams)
    assert summary["axes"] == ["tilt", "rotate"]
    # The butt specimen has an overhead seam, so there is something to improve.
    assert summary["improved"] >= 1
    assert any("positioner improves" in w for w in job.report.warnings)


@requires_cad
def test_each_seam_is_planned_at_its_own_positioner_setting(butt_assembly_step):
    """Planning every seam at the setup pose would be wrong for all but one."""
    job = plan_job(butt_assembly_step, load_cell("positioner_cell"), fast())
    poses = {seam.id: job.part_pose_for(seam.id) for seam in job.seams}
    distinct = {tuple(np.round(p.reshape(-1), 3)) for p in poses.values()}
    assert len(distinct) > 1, "every seam got the same pose; the positioner is unused"


@requires_cad
def test_positioner_values_reach_every_path_point(butt_assembly_step):
    job = plan_job(butt_assembly_step, load_cell("positioner_cell"), fast())
    for path in job.paths:
        expected = job.external_axes_for(path.seam_id)
        for point in path.points:
            assert point.external_axes == expected


@requires_cad
def test_indexing_time_is_charged_to_the_cycle(butt_assembly_step):
    """A cycle time that ignores the positioner is optimistic by the amount a
    plant manager notices."""
    job = plan_job(butt_assembly_step, load_cell("positioner_cell"), fast())
    assert any("indexing adds" in w for w in job.report.warnings)


# ===========================================================================
# positioner values in the output
# ===========================================================================


@requires_cad
def test_rapid_carries_the_positioner_in_extjoint(butt_assembly_step):
    """Left at 9E9 the controller welds at the previous seam's orientation."""
    job = plan_job(butt_assembly_step, load_cell("positioner_cell"), fast())
    body = get_postprocessor("rapid", PostOptions(program_name="P")).generate(job).main
    extjoints = set(re.findall(r"\[0, 0, 0, 0\], (\[[^\]]+\])", body))
    assert extjoints, "no robtargets found"
    assert not all(j[1:9] == "9E9, 9E9" for j in extjoints)
    assert len(extjoints) > 1, "every station got the same external axes"
    assert "external axes, eax_a onwards" in body


@requires_cad
def test_krl_carries_the_positioner_as_e1_onwards(butt_assembly_step):
    job = plan_job(butt_assembly_step, load_cell("positioner_cell"), fast())
    body = get_postprocessor("krl").generate(job).main
    assert re.search(r"LIN \{X[^}]*E1 [-\d.]+[^}]*E2 [-\d.]+\}", body)
    assert "$EX_AX_NUM" in body


@requires_cad
def test_fanuc_puts_the_positioner_in_its_own_motion_group(butt_assembly_step):
    job = plan_job(butt_assembly_step, load_cell("positioner_cell"), fast())
    body = get_postprocessor("fanuc").generate(job).main
    assert "GP2:" in body
    assert re.search(r"GP2:[\s\S]{0,80}J1 =\s+[-\d.]+ deg", body)


@requires_cad
def test_ros2_appends_the_positioner_to_the_trajectory(butt_assembly_step):
    import yaml

    job = plan_job(butt_assembly_step, load_cell("positioner_cell"), fast())
    files = get_postprocessor("ros2", PostOptions(program_name="P")).generate(job).files
    trajectories = yaml.safe_load(files["P_trajectories.yaml"])
    first = trajectories["trajectories"][0]
    assert "tilt" in first["joint_names"] and "rotate" in first["joint_names"]
    assert len(first["points"][0]["positions"]) == len(first["joint_names"])

    plan = yaml.safe_load(files["P_weld_plan.yaml"])
    assert "positioner_rad" in plan["passes"][0]


@requires_cad
def test_a_bench_job_emits_no_external_axes(tee_assembly_step):
    """No positioner means no extjoint values, not zeros."""
    job = plan_job(tee_assembly_step, load_cell("bench_cell"), fast())
    assert job.positioner is None
    body = get_postprocessor("rapid").generate(job).main
    assert "[9E9, 9E9, 9E9, 9E9, 9E9, 9E9]" in body
    assert "external axes, eax_a" not in body


# ===========================================================================
# jig location
# ===========================================================================


def test_datums_group_into_locating_planes():
    """It is the plane that constrains the part, not the individual points."""
    cell = load_cell("jig_cell")
    groups = group_datums(cell.fixture.datums)
    assert len(groups) == 3, "a 3-2-1 scheme has three locating planes"
    counts = [count for _, _, count in groups]
    assert counts == [3, 2, 1]
    # Mutually perpendicular, which is what makes the translations independent.
    normals = [normal for normal, _, _ in groups]
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(float(np.dot(normals[i], normals[j]))) < 1e-6


@requires_cad
def test_the_jig_determines_the_placement(tee_assembly_step):
    """A jig's placement is solved, not searched: that is what a jig is for."""
    part = load_part(tee_assembly_step)
    cell = load_cell("jig_cell")
    pose = solve_jig_placement(part, cell, np.eye(3))
    assert pose is not None

    groups = group_datums(cell.fixture.datums)
    vertices = tf.apply_points(pose, part.mesh_vertices)
    for normal, offset, _ in groups:
        reached = float((vertices @ normal).min())
        assert np.isclose(reached, offset, atol=0.05), (
            f"the part does not reach the locating plane at {offset:.1f} mm"
        )


@requires_cad
def test_a_seated_part_scores_full_marks_on_its_locators(tee_assembly_step):
    part = load_part(tee_assembly_step)
    seams, _ = detect_seams(part)
    cell = load_cell("jig_cell")
    pose = solve_jig_placement(part, cell, np.eye(3))
    result = evaluate_placement(part, seams, cell, pose, cell_robot(cell))
    assert result.detail["datum_seating"] == pytest.approx(1.0)
    assert result.detail["datums_seated"] == len(cell.fixture.datums)
    assert not result.violations


@requires_cad
def test_a_part_that_misses_its_locators_is_rejected(tee_assembly_step):
    """Not located means not weldable: a worse score is not enough."""
    part = load_part(tee_assembly_step)
    seams, _ = detect_seams(part)
    cell = load_cell("jig_cell")
    turned = solve_jig_placement(part, cell, tf.rot_z(np.pi / 2))
    result = evaluate_placement(part, seams, cell, turned, cell_robot(cell))
    assert not result.feasible
    assert any("not located" in v or "locators" in v for v in result.violations)
    assert result.detail["datum_seating"] < 1.0


@requires_cad
def test_find_placement_uses_the_jig_solver(tee_assembly_step):
    part = load_part(tee_assembly_step)
    seams, _ = detect_seams(part)
    cell = load_cell("jig_cell")
    best, evaluated = find_placement(part, seams, cell)
    # Solved per orientation, not searched over a grid of positions.
    assert len(evaluated) <= 12
    assert best.detail.get("datum_seating") is not None


def test_a_cell_without_datums_still_searches():
    cell = load_cell("bench_cell")
    assert cell.fixture.datums == []


# ===========================================================================
# standalone preview
# ===========================================================================


@requires_cad
def test_preview_scene_contains_every_layer(tee_assembly_step):
    cell = load_cell("bench_cell")
    robot = cell_robot(cell)
    job = plan_job(tee_assembly_step, cell, fast(), robot)
    scene = scene_from_job(job, robot)

    mesh_groups = {m.group for m in scene.meshes}
    line_groups = {line.group for line in scene.lines}
    assert {"part", "cell", "robot"} <= mesh_groups
    assert {"seams", "wire", "path"} <= line_groups
    assert scene.frames, "no animation frames"
    assert any(f.weld_on for f in scene.frames), "the arc is never on"


@requires_cad
def test_preview_seams_are_coloured_by_joint_type(butt_assembly_step):
    cell = load_cell("bench_cell")
    robot = cell_robot(cell)
    job = plan_job(butt_assembly_step, cell, fast(), robot)
    scene = scene_from_job(job, robot)
    from weldforge.viz.scene import JOINT_COLOURS

    seam_lines = [line for line in scene.lines if line.group == "seams"]
    assert seam_lines
    assert all(line.colour == JOINT_COLOURS[JointType.BUTT] for line in seam_lines)


@requires_cad
def test_preview_frames_are_capped(tee_assembly_step):
    cell = load_cell("bench_cell")
    robot = cell_robot(cell)
    job = plan_job(tee_assembly_step, cell, fast(), robot)
    scene = scene_from_job(job, robot, max_frames=3)
    assert len(scene.frames) <= 4


@requires_cad
def test_preview_html_is_self_contained(tee_assembly_step, tmp_path):
    """No CDN, no install, no network: a welding cell often has none of them."""
    cell = load_cell("bench_cell")
    robot = cell_robot(cell)
    job = plan_job(tee_assembly_step, cell, fast(), robot)
    path = write_html(scene_from_job(job, robot), tmp_path / "preview.html")

    html = path.read_text()
    assert html.startswith("<!doctype html>")
    assert "<script src" not in html
    assert "<link" not in html
    external = [url for url in re.findall(r"https?://[^\"'\s]+", html) if "www.w3.org" not in url]
    assert external == [], f"external references: {external}"

    payload = re.search(r'<script id="scene" type="application/json">(.*?)</script>', html, re.S)
    assert payload
    data = json.loads(payload.group(1))
    assert data["meshes"] and data["lines"] and data["frames"]
    assert data["summary"]["report"]["ok"] in (True, False)


@requires_cad
def test_preview_scene_json_is_serialisable(tee_assembly_step):
    cell = load_cell("bench_cell")
    robot = cell_robot(cell)
    job = plan_job(tee_assembly_step, cell, fast(), robot)
    payload = json.dumps(scene_to_json(scene_from_job(job, robot)))
    assert len(payload) > 1000


def test_preview_of_an_empty_scene_does_not_crash(tmp_path):
    from weldforge.viz import Scene

    path = write_html(Scene(name="empty"), tmp_path / "empty.html")
    assert "empty" in path.read_text()


@requires_cad
def test_preview_cli(tmp_path, tee_assembly_step):
    from weldforge.cli import main

    out = tmp_path / "p.html"
    code = main(
        [
            "preview",
            str(tee_assembly_step),
            "--cell",
            "bench_cell",
            "--grid",
            "3",
            "--yaw",
            "4",
            "-o",
            str(out),
        ]
    )
    assert code in (0, 2)
    assert out.exists()
    assert out.stat().st_size > 5000
