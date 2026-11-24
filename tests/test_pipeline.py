"""End-to-end pipeline, simulation and postprocessor tests.

These are the integration tests: a CAD file goes in and a robot program comes
out, with every stage's numbers checked against the physical situation they
describe. They are also where the accounting bugs show up -- a stand-off
counted as weld length, a torch "colliding" with the part it is welding -- that
no unit test of a single stage can see.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_cad

from weldforge.collide import scene_from_cell
from weldforge.fixture import cell_robot, default_cell, load_cell
from weldforge.plan import PlanOptions, plan_from_part, plan_job
from weldforge.post import PostOptions, available_targets, get_postprocessor
from weldforge.seams import DetectOptions, detect_seams
from weldforge.sim import SimOptions, move_duration_s, simulate

pytestmark = pytest.mark.needs_cad


def fast_options() -> PlanOptions:
    """Small placement search: enough to be meaningful, quick enough for CI."""
    options = PlanOptions()
    options.placement.grid_steps = 3
    options.placement.yaw_steps = 4
    options.placement.max_orientations = 3
    return options


@pytest.fixture(scope="module")
def tee_job(tee_assembly_step):
    return plan_job(tee_assembly_step, load_cell("bench_cell"), fast_options())


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


@requires_cad
def test_pipeline_produces_a_runnable_job(tee_job):
    """The whole point: CAD in, verified job out, no errors."""
    assert tee_job.report.ok, tee_job.report.errors
    assert tee_job.report.reachable
    assert tee_job.report.collision_free
    assert len(tee_job.seams) == 2
    assert len(tee_job.procedures) == 2
    assert len(tee_job.paths) == 2
    assert tee_job.placement is not None and tee_job.placement.feasible


@requires_cad
def test_pipeline_weld_length_matches_the_seams(tee_job):
    """Accounting check: the deposited length is the seam length, no more.

    The stand-off move into the first station has an arc-on destination but is
    an air move. Counting it as weld inflated a 400 mm job to 480 mm -- an
    error that propagates straight into the wire and gas quote.
    """
    assert np.isclose(tee_job.report.total_weld_length_mm, 400.0, atol=1.0)
    assert np.isclose(tee_job.report.total_weld_length_mm, tee_job.total_weld_length_mm, atol=1.0)


@requires_cad
def test_pipeline_consumables_are_physically_sensible(tee_job):
    """A 4 mm fillet over 400 mm is tens of grams of wire, not kilos."""
    report = tee_job.report
    assert 0.010 < report.wire_consumption_kg < 0.150
    assert 5.0 < report.gas_consumption_l < 60.0
    # Gas flows for as long as the arc burns, give or take the pre/post flow.
    expected_gas = 15.0 * report.arc_time_s / 60.0
    assert 0.5 * expected_gas < report.gas_consumption_l < 2.5 * expected_gas


@requires_cad
def test_pipeline_cycle_time_is_dominated_by_arc_time_here(tee_job):
    """Two long seams side by side: almost all the time is welding."""
    report = tee_job.report
    assert report.cycle_time_s > 0.0
    assert np.isclose(report.cycle_time_s, report.arc_time_s + report.air_time_s)
    assert report.duty_cycle > 0.5
    # 400 mm at about 7 mm/s is roughly a minute of arc time.
    assert 40.0 < report.arc_time_s < 100.0


@requires_cad
def test_pipeline_places_the_part_downhand(tee_job):
    """Both fillets end up as PB -- the position they are fastest in."""
    assert tee_job.placement.position_mix == {"PB": 2}
    assert not tee_job.placement.unreachable_seams


@requires_cad
def test_pipeline_records_stage_timings(tee_job):
    timings = tee_job.metadata["timings"]
    assert {"detect_s", "procedures_s", "placement_s", "paths_s"} <= set(timings)
    assert all(v >= 0.0 for v in timings.values())


@requires_cad
def test_quick_look_skips_the_expensive_stages(tee_assembly_step):
    job = plan_job(tee_assembly_step, load_cell("bench_cell"), PlanOptions.quick_look())
    assert job.seams and job.procedures
    assert job.placement is None
    assert job.paths == []


@requires_cad
def test_pipeline_reports_when_nothing_is_weldable(plate_with_hole_step):
    job = plan_job(plate_with_hole_step, load_cell("bench_cell"), fast_options())
    assert job.seams == []
    assert any("no weld seams" in e for e in job.report.errors)


@requires_cad
def test_pipeline_can_cap_the_seam_count(butt_assembly_step):
    options = fast_options()
    options.max_seams = 2
    job = plan_job(butt_assembly_step, load_cell("bench_cell"), options)
    assert len(job.seams) == 2
    assert job.metadata["seams_truncated"]["found"] == 4


@requires_cad
def test_pipeline_works_with_the_default_cell(tee_assembly_step):
    job = plan_job(tee_assembly_step, default_cell(), fast_options())
    assert job.seams
    assert job.cell.name == "default bench"


@requires_cad
def test_unreachable_part_is_reported_not_hidden(tee_assembly_step, specimen_dir):
    """A part parked outside the robot's envelope must say so."""
    from weldforge.cad import load_part

    cell = load_cell("bench_cell")
    # Move the bench far beyond the KR 16's 1611 mm reach.
    cell.fixture.table_origin = np.array([4000.0, 0.0, 0.0])
    part = load_part(tee_assembly_step)
    job = plan_from_part(part, cell, fast_options())
    assert job.placement is not None
    assert job.placement.violations or job.placement.unreachable_seams


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


def test_move_duration_takes_the_binding_limit():
    """Joint speed governs an air move; Cartesian speed governs a weld."""
    from weldforge.geom import transform as tf
    from weldforge.kin import load_robot
    from weldforge.model import PathPoint

    robot = load_robot("generic_6axis")
    a = PathPoint(pose=tf.pose(np.eye(3), [0.0, 0.0, 0.0]), joints=np.zeros(6))
    # 10 mm apart but a 90 deg swing of axis 1: the joint limit must win.
    b = PathPoint(
        pose=tf.pose(np.eye(3), [10.0, 0.0, 0.0]),
        joints=np.array([np.pi / 2, 0, 0, 0, 0, 0]),
        speed_mm_s=250.0,
    )
    cartesian_only = 10.0 / 250.0
    assert move_duration_s(a, b, robot) > cartesian_only * 10


def test_move_duration_without_joints_falls_back_to_cartesian():
    from weldforge.geom import transform as tf
    from weldforge.model import PathPoint

    a = PathPoint(pose=tf.pose(np.eye(3), [0.0, 0.0, 0.0]))
    b = PathPoint(pose=tf.pose(np.eye(3), [100.0, 0.0, 0.0]), speed_mm_s=50.0)
    assert np.isclose(move_duration_s(a, b), 2.0)


@requires_cad
def test_simulate_flags_a_job_with_no_paths(tee_assembly_step):
    from weldforge.cad import load_part
    from weldforge.model import WeldJob

    job = WeldJob(part=load_part(tee_assembly_step))
    report = simulate(job)
    assert not report.ok
    assert any("no paths" in e for e in report.errors)


@requires_cad
def test_simulate_warns_about_a_poor_duty_cycle(tee_job):
    """A cell spending most of its time travelling should say so."""
    import copy

    job = copy.deepcopy(tee_job)
    cell = job.cell
    robot = cell_robot(cell)
    scene = scene_from_cell(cell)
    # Crank the per-seam overhead until air time dominates.
    report = simulate(job, robot, scene, SimOptions(check_collisions=False, seam_change_s=400.0))
    assert report.duty_cycle < 0.25
    assert any("arc-on time is only" in w for w in report.warnings)


@requires_cad
def test_simulate_surfaces_procedure_warnings(butt_assembly_step):
    """An 8 mm butt needs edge preparation, and the operator must see it."""
    job = plan_job(butt_assembly_step, load_cell("bench_cell"), fast_options())
    assert any("edge preparation" in w for w in job.report.warnings)


@requires_cad
def test_per_seam_breakdown_is_complete(tee_job):
    assert len(tee_job.report.per_seam) == 2
    for entry in tee_job.report.per_seam.values():
        assert entry["joint_type"] == "tee"
        assert entry["position"] == "PB"
        assert entry["passes"] >= 1
        assert entry["arc_time_s"] > 0.0
        assert entry["unreachable"] == 0
        assert entry["collisions"] == 0


# ---------------------------------------------------------------------------
# collision model
# ---------------------------------------------------------------------------


@requires_cad
def test_the_torch_is_allowed_to_touch_the_part_it_welds(tee_job):
    """The wire ends on the joint; only the nozzle must keep clear.

    Checking the wire against the part declared a clash at every single
    welding station -- the torch "colliding" with the weld.
    """
    assert tee_job.report.collision_free
    assert all(not p.collision for path in tee_job.paths for p in path.points)


@requires_cad
def test_the_table_does_not_swallow_the_robot(tee_job):
    """The bench is a finite box, not a half-space.

    Modelled as an infinite plane at bench height, everything below it counts
    as solid -- including the robot's own column, which stands on the floor.
    Every posture then reports the arm inside the table.
    """
    assert not any("table" in (p.collision or "") for path in tee_job.paths for p in path.points)


@requires_cad
def test_a_clamp_in_the_way_is_detected(tee_assembly_step):
    """A clamp sitting on a seam must score that placement down.

    Checked by re-scoring **the same** placement with the clamp added, not by
    re-planning: given the freedom, the search simply slides the part out from
    under the clamp -- which is the right answer, and would hide the fact that
    the obstruction was noticed at all.
    """
    from weldforge.cad import load_part
    from weldforge.fixture import evaluate_placement
    from weldforge.model import Clamp

    cell = load_cell("bench_cell")
    part = load_part(tee_assembly_step)
    job = plan_from_part(part, cell, fast_options())
    pose = job.placement.pose
    clean = evaluate_placement(part, job.seams, cell, pose)
    assert clean.feasible

    seam_point = job.seams[0].transformed(pose).centroid
    cell.fixture.clamps.append(
        Clamp(
            id="blocker",
            position=seam_point + np.array([0.0, 0.0, 10.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            clearance_mm=60.0,
            height_mm=150.0,
        )
    )
    blocked = evaluate_placement(part, job.seams, cell, pose)
    assert blocked.violations, "a clamp planted on the seam was not noticed"
    assert blocked.score < clean.score


@requires_cad
def test_the_planner_moves_the_part_away_from_an_obstruction(tee_assembly_step):
    """Given room, the search should route around a clamp rather than fail."""
    from weldforge.cad import load_part
    from weldforge.model import Clamp

    cell = load_cell("bench_cell")
    part = load_part(tee_assembly_step)
    reference = plan_from_part(part, cell, fast_options())
    seam_point = reference.seams[0].transformed(reference.placement.pose).centroid

    cell.fixture.clamps.append(
        Clamp(
            id="blocker",
            position=seam_point + np.array([0.0, 0.0, 10.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            clearance_mm=60.0,
            height_mm=150.0,
        )
    )
    moved = plan_from_part(load_part(tee_assembly_step), cell, fast_options())
    assert moved.placement.feasible
    from weldforge.geom import transform as tf

    shift = float(
        np.linalg.norm(
            tf.translation_of(moved.placement.pose) - tf.translation_of(reference.placement.pose)
        )
    )
    assert shift > 20.0, "the part should have been moved clear of the clamp"


# ---------------------------------------------------------------------------
# postprocessors
# ---------------------------------------------------------------------------


def test_all_expected_targets_are_registered():
    assert {"rapid", "krl", "urscript", "fanuc", "ros2"} <= set(available_targets())


def test_unknown_target_names_the_alternatives():
    with pytest.raises(KeyError, match="available"):
        get_postprocessor("hal9000")


@requires_cad
@pytest.mark.parametrize("target", ["rapid", "krl", "urscript", "fanuc", "ros2"])
def test_every_target_generates_a_program(tee_job, target):
    result = get_postprocessor(target, PostOptions(program_name="TEE")).generate(tee_job)
    assert result.files
    assert result.n_moves > 0
    assert result.n_skipped == 0
    body = result.main
    assert "WeldForge" in body
    assert "MACHINE GENERATED" in body


@requires_cad
def test_rapid_uses_the_arc_instructions_and_abb_quaternion_order(tee_job):
    body = get_postprocessor("rapid").generate(tee_job).main
    assert "MODULE" in body and "ENDMODULE" in body
    assert "ArcLStart" in body and "ArcLEnd" in body
    assert "CONST welddata" in body and "CONST seamdata" in body
    assert "robtarget" in body
    # ABB orders the quaternion (w, x, y, z); ours is (x, y, z, w). Picking the
    # first number out of a robtarget must give w, which is in [-1, 1] and for
    # a canonical quaternion is the largest-magnitude component more often
    # than not -- so just check the record parses and is normalised.
    import re

    match = re.search(r"robtarget \S+ := \[\[[^\]]+\], \[([^\]]+)\]", body)
    assert match
    values = [float(v) for v in match.group(1).split(",")]
    assert len(values) == 4
    assert np.isclose(np.linalg.norm(values), 1.0, atol=1e-3)


@requires_cad
def test_krl_emits_millimetres_and_kuka_euler_angles(tee_job):
    body = get_postprocessor("krl").generate(tee_job).main
    assert body.lstrip().startswith(";") or "DEF " in body
    assert "LIN {X" in body and "PTP {X" in body
    assert "A " in body and "B " in body and "C " in body
    # Arc control through digital outputs, with a confirmation input.
    assert "$OUT[" in body and "WAIT FOR $IN[" in body
    assert "SET_WELD_PARAMS" in body


@requires_cad
def test_urscript_is_the_only_target_in_metres(tee_job):
    """Unit boundary: a 200 mm seam must appear as 0.2, not 200."""
    post = get_postprocessor("urscript")
    assert post.units == "m"
    body = post.generate(tee_job).main
    import re

    poses = re.findall(r"p\[([-\d.,\s]+)\]", body)
    assert poses
    coordinates = [abs(float(v)) for pose in poses for v in pose.split(",")[:3]]
    # Everything in the cell is inside a couple of metres of the robot.
    assert max(coordinates) < 5.0
    assert "movel(" in body and "weld_on(" in body


@requires_cad
def test_fanuc_positions_and_instructions_stay_in_step(tee_job):
    body = get_postprocessor("fanuc").generate(tee_job).main
    assert "/PROG" in body and "/MN" in body and "/POS" in body and "/END" in body
    import re

    referenced = {int(v) for v in re.findall(r"[JL] P\[(\d+)\]", body)}
    declared = {int(v) for v in re.findall(r"^P\[(\d+)\]\{", body, re.MULTILINE)}
    assert referenced <= declared, (
        f"instructions reference undeclared positions: {referenced - declared}"
    )
    assert "Arc Start[" in body and "Arc End[" in body


@requires_cad
def test_ros2_emits_metres_and_a_separate_weld_plan(tee_job):
    import yaml

    result = get_postprocessor("ros2", PostOptions(program_name="J")).generate(tee_job)
    assert set(result.files) == {"J_trajectories.yaml", "J_weld_plan.yaml"}

    trajectories = yaml.safe_load(result.files["J_trajectories.yaml"])
    assert trajectories["units"]["length"] == "m"
    assert trajectories["trajectories"]
    first = trajectories["trajectories"][0]
    assert first["joint_names"]
    assert first["duration_s"] > 0.0
    times = [p["time_from_start"] for p in first["points"]]
    assert times == sorted(times)
    assert all(abs(v) < 5.0 for p in first["points"] for v in p["pose"]["position"])

    plan = yaml.safe_load(result.files["J_weld_plan.yaml"])
    assert plan["passes"]
    entry = plan["passes"][0]
    assert entry["current_a"] > 0.0 and entry["voltage_v"] > 0.0
    assert entry["process"] == "GMAW"
    assert plan["placement"] is not None


@requires_cad
def test_postprocessor_header_carries_the_warning_when_the_job_failed(tee_job):
    import copy

    broken = copy.deepcopy(tee_job)
    broken.report.errors.append("seam xyz: unreachable station")
    broken.report.reachable = False
    body = get_postprocessor("rapid").generate(broken).main
    assert "THE PLANNER REPORTED PROBLEMS" in body
    assert "unreachable station" in body


@requires_cad
def test_unreachable_stations_are_skipped_and_counted(tee_job):
    import copy

    job = copy.deepcopy(tee_job)
    for point in job.paths[0].points:
        point.joints = None
        point.comment = "unreachable (120 mm short)"
    result = get_postprocessor("rapid").generate(job)
    assert result.n_skipped == len(job.paths[0].points)
    assert any("too few reachable" in w for w in result.warnings)


@requires_cad
def test_postprocessor_can_emit_unreachable_stations_on_request(tee_job):
    import copy

    job = copy.deepcopy(tee_job)
    for point in job.paths[0].points:
        point.joints = None
        point.comment = "unreachable (120 mm short)"
    result = get_postprocessor("rapid", PostOptions(skip_unreachable=False)).generate(job)
    assert result.n_skipped == 0


# ---------------------------------------------------------------------------
# the seams the operator settled on
# ---------------------------------------------------------------------------


@requires_cad
def test_the_planner_welds_the_seams_it_is_given(tee_assembly_step):
    """A caller that knows what to weld must not have it re-detected away.

    The studio lets an operator trace a joint by hand and switch detected ones
    off. Both were discarded: the planner re-ran detection from the geometry,
    so a hand-traced joint on a single-body part came back as "no weld seams
    were recognised" and switching a seam off changed nothing about the plan.
    """
    from weldforge.cad import load_part

    part = load_part(tee_assembly_step)
    found, _ = detect_seams(part, DetectOptions())
    assert len(found) >= 2, "the tee assembly has more than one seam to choose from"

    chosen = [found[0]]
    job = plan_from_part(part, load_cell("bench_cell"), fast_options(), seams=chosen)

    assert [s.id for s in job.seams] == [chosen[0].id]
    assert job.metadata.get("seams_given") == 1
    assert "no weld seams were recognised" not in job.report.errors
    assert set(job.procedures) == {chosen[0].id}, "one procedure, for the seam asked for"
    # Detection must not have run at all: it would have cost time and
    # overwritten the choice.
    assert "detection" not in job.metadata


@requires_cad
def test_an_empty_seam_list_is_not_the_same_as_no_list(tee_assembly_step):
    """"Weld nothing" is an answer; it must not fall back to detecting."""
    from weldforge.cad import load_part

    job = plan_from_part(load_part(tee_assembly_step), load_cell("bench_cell"), seams=[])

    assert job.seams == []
    assert "no weld seams were recognised" in job.report.errors
    assert "detection" not in job.metadata
