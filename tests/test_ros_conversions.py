"""The ROS unit boundary, tested without ROS.

The conversion functions in ``weldforge_cell.conversions`` take the message
*modules* as arguments rather than importing them, which was done precisely so
this test can exist: stub message classes stand in for the generated ones, and
the millimetre-to-metre boundary gets verified on a development machine with no
ROS installed. Without this the conversions would only ever be exercised by
running a cell, which is the worst place to discover a factor of a thousand.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import make_seam

from weldforge.geom import transform as tf
from weldforge.model import JointType, MachineSetpoints, WeaveShape

# The node package lives in the ROS workspace, not in the installed core.
ROS_PACKAGE = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "weldforge_cell"
if str(ROS_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ROS_PACKAGE))

conversions = pytest.importorskip("weldforge_cell.conversions")


# ---------------------------------------------------------------------------
# stub messages
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class _Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class _Pose:
    position: _Point = field(default_factory=_Point)
    orientation: _Quaternion = field(default_factory=_Quaternion)


@dataclass
class _Header:
    frame_id: str = ""
    stamp: object = None


def _msg(**fields):
    """A dataclass-like message whose fields default to sensible empties."""

    class Message(SimpleNamespace):
        def __init__(self) -> None:
            super().__init__(**{k: (v() if callable(v) else v) for k, v in fields.items()})

    return Message


GEOMETRY = SimpleNamespace(Pose=_Pose, Point=_Point, Quaternion=_Quaternion)

MSGS = SimpleNamespace(
    WeldSeam=_msg(
        header=_Header,
        id="",
        label="",
        joint_type="",
        stations=list,
        length=0.0,
        groove_angle=0.0,
        groove_angle_spread=0.0,
        welding_position="",
        thickness_a=0.0,
        thickness_b=0.0,
        root_gap=0.0,
        confidence=0.0,
        closed=False,
        selected=False,
        spans_two_solids=False,
        solid_a_id="",
        solid_b_id="",
        notes=list,
    ),
    WeldPass=_msg(
        index=0,
        name="",
        leg=0.0,
        throat=0.0,
        offset_across=0.0,
        offset_normal=0.0,
        process="",
        iso_4063=0,
        transfer_mode="",
        current=0.0,
        voltage=0.0,
        wire_speed=0.0,
        wire_diameter=0.0,
        gas="",
        gas_flow=0.0,
        polarity="",
        travel_speed=0.0,
        work_angle=0.0,
        travel_angle=0.0,
        stickout=0.0,
        weave_shape="",
        weave_amplitude=0.0,
        weave_frequency=0.0,
        weave_dwell=0.0,
        arc_start_delay=0.0,
        crater_fill=0.0,
        burnback=0.0,
        heat_input=0.0,
    ),
    WeldProcedure=_msg(
        wps_id="",
        seam_id="",
        base_material="",
        filler="",
        standard="",
        welding_position="",
        joint_type="",
        thickness=0.0,
        target_leg=0.0,
        preheat=0.0,
        interpass_max=0.0,
        heat_input_max=0.0,
        passes=list,
        warnings=list,
    ),
    MachineSetpoints=_msg(
        pass_index=0,
        has_job_number=False,
        job_number=0,
        current=0.0,
        voltage=0.0,
        wire_speed=0.0,
        arc_length_correction=0.0,
        dynamic_correction=0.0,
        gas_flow=0.0,
        pulse_frequency=0.0,
        machine_profile="",
        analog_names=list,
        analog_values=list,
        digital_names=list,
        digital_values=list,
        notes=list,
    ),
    JobReport=_msg(
        ok=False,
        reachable=False,
        collision_free=False,
        cycle_time=0.0,
        arc_time=0.0,
        air_time=0.0,
        duty_cycle=0.0,
        total_weld_length=0.0,
        wire_consumption=0.0,
        gas_consumption=0.0,
        errors=list,
        warnings=list,
    ),
    WeldPlan=_msg(
        header=_Header,
        job_id="",
        job_name="",
        part_name="",
        cell_name="",
        robot_name="",
        seams=list,
        procedures=list,
        part_pose=_Pose,
        placement_feasible=False,
        placement_score=0.0,
        placement_violations=list,
        trajectories=list,
        trajectory_seam_ids=list,
        trajectory_pass_indices=list,
        arc_on=list,
        arc_on_offsets=list,
        report=None,
    ),
)


@dataclass
class _Duration:
    sec: int = 0
    nanosec: int = 0


@dataclass
class _TrajectoryPoint:
    positions: list = field(default_factory=list)
    time_from_start: _Duration = field(default_factory=_Duration)


@dataclass
class _JointTrajectory:
    joint_names: list = field(default_factory=list)
    points: list = field(default_factory=list)


TRAJECTORY = SimpleNamespace(
    JointTrajectory=_JointTrajectory, JointTrajectoryPoint=_TrajectoryPoint
)
BUILTIN = SimpleNamespace(Duration=_Duration)


# ---------------------------------------------------------------------------
# poses
# ---------------------------------------------------------------------------


def test_pose_converts_millimetres_to_metres():
    pose = tf.pose(tf.rot_z(0.7), [1234.5, -600.0, 750.0])
    msg = conversions.pose_to_msg(pose, GEOMETRY)
    assert np.isclose(msg.position.x, 1.2345)
    assert np.isclose(msg.position.y, -0.6)
    assert np.isclose(msg.position.z, 0.75)
    quaternion = np.array(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    )
    assert np.isclose(np.linalg.norm(quaternion), 1.0)


def test_pose_round_trip_loses_nothing_that_matters():
    """Through metres and a quaternion and back, to well under what a robot can do.

    The tolerances are set against physical reality, not against float64: a
    micrometre and a microdegree are both orders of magnitude below any
    industrial robot's repeatability, so anything inside them cannot affect a
    weld.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        rotation = tf.rot_axis_angle(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        pose = tf.pose(rotation, rng.uniform(-2000.0, 2000.0, size=3))
        back = conversions.msg_to_pose(conversions.pose_to_msg(pose, GEOMETRY))
        lin, ang = tf.pose_distance(pose, back)
        assert lin < 1e-6, f"{lin:.3e} mm"
        assert np.rad2deg(ang) < 1e-4, f"{np.rad2deg(ang):.3e} deg"


def test_metres_in_a_message_become_millimetres_in_the_core():
    msg = _Pose(position=_Point(x=1.5, y=0.0, z=0.8), orientation=_Quaternion())
    assert np.allclose(tf.translation_of(conversions.msg_to_pose(msg)), [1500.0, 0.0, 800.0])


# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------


def test_seam_lengths_and_thicknesses_are_in_metres():
    seam = make_seam(JointType.TEE, 8.0, 6.0, length=200.0, gap=2.0)
    msg = conversions.seam_to_msg(seam, MSGS, GEOMETRY)
    assert np.isclose(msg.length, 0.2)
    assert np.isclose(msg.thickness_a, 0.008)
    assert np.isclose(msg.thickness_b, 0.006)
    assert np.isclose(msg.root_gap, 0.002)
    # Angles stay in radians, as REP-103 requires.
    assert np.isclose(msg.groove_angle, np.pi / 2)
    assert msg.welding_position == "PB"
    assert msg.joint_type == "tee"


def test_seam_stations_carry_torch_poses():
    """Each station's Z must point along the wire, so a controller can use it."""
    seam = make_seam(JointType.TEE, 8.0, 6.0)
    msg = conversions.seam_to_msg(seam, MSGS, GEOMETRY, include_geometry=True)
    assert len(msg.stations) == seam.n_points
    pose = conversions.msg_to_pose(msg.stations[5])
    assert np.allclose(tf.rotation_of(pose)[:, 2], seam.bisector[5], atol=1e-6)


def test_seam_geometry_can_be_omitted():
    seam = make_seam(JointType.TEE, 8.0, 6.0)
    msg = conversions.seam_to_msg(seam, MSGS, GEOMETRY, include_geometry=False)
    assert msg.stations == []


# ---------------------------------------------------------------------------
# procedures
# ---------------------------------------------------------------------------


def test_wire_feed_converts_from_shop_units_to_si():
    """Procedure sheets say m/min; the message says m/s."""
    from weldforge.wps import generate_procedure

    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0))
    weld_pass = procedure.passes[0]
    msg = conversions.pass_to_msg(weld_pass, MSGS)
    assert np.isclose(msg.wire_speed, weld_pass.wire_speed_m_min / 60.0)
    assert np.isclose(msg.travel_speed, weld_pass.travel_speed_mm_s / 1000.0)
    assert np.isclose(msg.wire_diameter, weld_pass.wire_diameter_mm / 1000.0)
    assert np.isclose(msg.stickout, weld_pass.stickout_mm / 1000.0)


def test_angles_convert_from_degrees_to_radians():
    from weldforge.wps import generate_procedure

    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0))
    weld_pass = procedure.passes[0]
    msg = conversions.pass_to_msg(weld_pass, MSGS)
    assert np.isclose(msg.work_angle, np.deg2rad(weld_pass.work_angle_deg))
    assert np.isclose(msg.travel_angle, np.deg2rad(weld_pass.travel_angle_deg))


def test_heat_input_converts_from_kj_per_mm_to_j_per_m():
    from weldforge.wps import generate_procedure

    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0))
    weld_pass = procedure.passes[0]
    msg = conversions.pass_to_msg(weld_pass, MSGS)
    # 1 kJ/mm is 1e6 J/m.
    assert np.isclose(msg.heat_input, weld_pass.heat_input_kj_mm() * 1e6)


def test_procedure_carries_the_iso_number_and_warnings():
    from weldforge.wps import generate_procedure

    procedure = generate_procedure(
        make_seam(JointType.BUTT, 12.0, 12.0, gap=2.0, bisector=(0.0, 0.0, -1.0))
    )
    msg = conversions.procedure_to_msg(procedure, MSGS)
    assert msg.passes[0].iso_4063 == 135
    assert any("edge preparation" in w for w in msg.warnings)
    assert np.isclose(msg.thickness, 0.012)


def test_weave_dwell_converts_from_milliseconds_to_seconds():
    from weldforge.model import Weave, WeldPass

    weld_pass = WeldPass(
        weave=Weave(WeaveShape.TRIANGULAR, amplitude_mm=4.0, frequency_hz=1.2, dwell_ms=150.0)
    )
    msg = conversions.pass_to_msg(weld_pass, MSGS)
    assert np.isclose(msg.weave_amplitude, 0.004)
    assert np.isclose(msg.weave_dwell, 0.15)
    assert msg.weave_frequency == 1.2


# ---------------------------------------------------------------------------
# setpoints
# ---------------------------------------------------------------------------


def test_setpoints_round_trip():
    setpoints = MachineSetpoints(
        pass_index=2,
        job_number=7,
        current_a=240.0,
        voltage_v=27.0,
        wire_speed_m_min=8.4,
        gas_flow_l_min=15.0,
        analog_words={"AO1": 1234, "AO2": 5678},
        digital_bits={"DO1": True, "DO2": False},
        machine_profile="fronius_tps",
        notes=["synergic"],
    )
    msg = conversions.setpoints_to_msg(setpoints, MSGS)
    assert msg.has_job_number and msg.job_number == 7
    assert np.isclose(msg.wire_speed, 8.4 / 60.0)

    back = conversions.msg_to_setpoints(msg)
    assert back.job_number == 7
    assert np.isclose(back.wire_speed_m_min, 8.4)
    assert back.analog_words == {"AO1": 1234, "AO2": 5678}
    assert back.digital_bits == {"DO1": True, "DO2": False}


def test_setpoints_without_a_job_number():
    msg = conversions.setpoints_to_msg(MachineSetpoints(job_number=None), MSGS)
    assert not msg.has_job_number
    assert conversions.msg_to_setpoints(msg).job_number is None


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------


def test_trajectory_timestamps_increase_and_match_the_cycle_time():
    from weldforge.model import MotionType, PathPoint

    points = []
    for i in range(10):
        points.append(
            PathPoint(
                pose=tf.pose(np.eye(3), [i * 10.0, 0.0, 0.0]),
                motion=MotionType.LINEAR,
                speed_mm_s=10.0,
                weld_on=i not in (0, 9),
                joints=np.full(6, i * 0.01),
            )
        )
    trajectory, arc_on = conversions.trajectory_from_path(
        points, [f"joint_{i + 1}" for i in range(6)], TRAJECTORY, BUILTIN
    )
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in trajectory.points]
    assert times == sorted(times)
    assert times[0] == 0.0
    # Nine 10 mm steps at 10 mm/s is nine seconds.
    assert np.isclose(times[-1], 9.0, atol=0.01)
    assert arc_on == [p.weld_on for p in points]


def test_trajectory_drops_points_with_no_joint_solution():
    from weldforge.model import PathPoint

    points = [
        PathPoint(pose=tf.identity(), joints=np.zeros(6)),
        PathPoint(pose=tf.identity(), joints=None),
        PathPoint(pose=tf.pose(np.eye(3), [10.0, 0, 0]), joints=np.ones(6) * 0.1),
    ]
    trajectory, arc_on = conversions.trajectory_from_path(
        points, [f"joint_{i + 1}" for i in range(6)], TRAJECTORY, BUILTIN
    )
    assert len(trajectory.points) == 2
    assert len(arc_on) == 2


def test_speed_scale_stretches_the_timestamps():
    from weldforge.model import PathPoint

    points = [
        PathPoint(pose=tf.identity(), joints=np.zeros(6), speed_mm_s=10.0),
        PathPoint(pose=tf.pose(np.eye(3), [100.0, 0, 0]), joints=np.zeros(6), speed_mm_s=10.0),
    ]
    names = [f"joint_{i + 1}" for i in range(6)]
    full, _ = conversions.trajectory_from_path(points, names, TRAJECTORY, BUILTIN, 1.0)
    half, _ = conversions.trajectory_from_path(points, names, TRAJECTORY, BUILTIN, 0.5)

    def end(trajectory):
        last = trajectory.points[-1].time_from_start
        return last.sec + last.nanosec * 1e-9

    assert np.isclose(end(half), 2.0 * end(full), rtol=1e-6)
