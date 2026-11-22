"""The unit boundary between the planning core and ROS.

The core works in **millimetres**; ROS works in **metres** (REP-103). Every
conversion between them happens in this module and nowhere else, so there is
exactly one place to look when a number comes out a thousand times too big.

The same applies to the two other conventions that differ:

* wire feed speed is quoted in **m/min** on every procedure sheet and welding
  machine, so the core carries it that way; the ROS message carries **m/s**;
* the core stores poses as 4x4 matrices, ROS as position plus quaternion.

Nothing here imports rclpy, so the conversions can be unit-tested without a
ROS environment -- which is the only way they get tested at all on a
development machine without ROS installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from weldforge.geom import transform as tf
from weldforge.model import (
    MachineSetpoints,
    PathPoint,
    Seam,
    WeldJob,
    WeldPass,
    WeldProcedure,
)
from weldforge.units import mm_to_m

__all__ = [
    "MM_PER_M",
    "pose_to_msg",
    "msg_to_pose",
    "seam_to_msg",
    "pass_to_msg",
    "procedure_to_msg",
    "setpoints_to_msg",
    "msg_to_setpoints",
    "report_to_msg",
    "job_to_plan_msg",
    "trajectory_from_path",
]

MM_PER_M = 1000.0


def _vec3(values: np.ndarray, msg_type: Any) -> Any:
    out = msg_type()
    out.x, out.y, out.z = (float(v) for v in values)
    return out


# ---------------------------------------------------------------------------
# poses
# ---------------------------------------------------------------------------


def pose_to_msg(pose: np.ndarray, msg_module: Any) -> Any:
    """4x4 millimetre pose -> ``geometry_msgs/Pose`` in metres."""
    out = msg_module.Pose()
    position = tf.translation_of(pose)
    out.position.x = mm_to_m(float(position[0]))
    out.position.y = mm_to_m(float(position[1]))
    out.position.z = mm_to_m(float(position[2]))
    x, y, z, w = tf.matrix_to_quat(pose)
    out.orientation.x = float(x)
    out.orientation.y = float(y)
    out.orientation.z = float(z)
    out.orientation.w = float(w)
    return out


def msg_to_pose(msg: Any) -> np.ndarray:
    """``geometry_msgs/Pose`` in metres -> 4x4 millimetre pose."""
    quaternion = np.array(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
        dtype=float,
    )
    translation = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=float) * MM_PER_M
    return tf.pose(tf.quat_to_matrix(quaternion), translation)


# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------


def seam_to_msg(
    seam: Seam,
    msgs: Any,
    geometry_msgs: Any,
    include_geometry: bool = True,
    frame_id: str = "cell",
    stamp: Any = None,
) -> Any:
    """:class:`weldforge.model.Seam` -> ``weldforge_msgs/WeldSeam``."""
    out = msgs.WeldSeam()
    out.header.frame_id = frame_id
    if stamp is not None:
        out.header.stamp = stamp

    out.id = seam.id
    out.label = seam.label
    out.joint_type = seam.joint_type.value
    out.length = mm_to_m(seam.length_mm)
    out.groove_angle = float(seam.mean_dihedral)
    out.groove_angle_spread = float(np.ptp(seam.dihedral)) if len(seam.dihedral) else 0.0
    out.welding_position = seam.dominant_position.value
    out.thickness_a = mm_to_m(seam.thickness_a_mm)
    out.thickness_b = mm_to_m(seam.thickness_b_mm)
    out.root_gap = mm_to_m(seam.gap_mm)
    out.confidence = float(seam.confidence)
    out.closed = bool(seam.closed)
    out.selected = bool(seam.selected)
    out.spans_two_solids = bool(seam.spans_two_solids)
    out.solid_a_id = seam.solid_a_id
    out.solid_b_id = seam.solid_b_id
    out.notes = list(seam.notes)

    if include_geometry and seam.n_points:
        from weldforge.path.torch import torch_pose

        stations = []
        for i in range(seam.n_points):
            wire = seam.bisector[i] if len(seam.bisector) > i else np.array([0.0, 0.0, -1.0])
            tangent = seam.tangents[i] if len(seam.tangents) > i else np.array([1.0, 0.0, 0.0])
            stations.append(pose_to_msg(torch_pose(seam.points[i], wire, tangent), geometry_msgs))
        out.stations = stations
    return out


# ---------------------------------------------------------------------------
# procedures
# ---------------------------------------------------------------------------


def pass_to_msg(weld_pass: WeldPass, msgs: Any) -> Any:
    out = msgs.WeldPass()
    out.index = int(weld_pass.index)
    out.name = weld_pass.name
    out.leg = mm_to_m(weld_pass.leg_mm)
    out.throat = mm_to_m(weld_pass.throat_mm)
    out.offset_across = mm_to_m(weld_pass.offset_across_mm)
    out.offset_normal = mm_to_m(weld_pass.offset_normal_mm)
    out.process = weld_pass.process.value
    out.iso_4063 = int(weld_pass.process.iso_4063)
    out.transfer_mode = weld_pass.transfer_mode.value
    out.current = float(weld_pass.current_a)
    out.voltage = float(weld_pass.voltage_v)
    # Shop sheets quote wire feed in m/min; the message carries m/s.
    out.wire_speed = float(weld_pass.wire_speed_m_min) / 60.0
    out.wire_diameter = mm_to_m(weld_pass.wire_diameter_mm)
    out.gas = weld_pass.gas
    out.gas_flow = float(weld_pass.gas_flow_l_min)
    out.polarity = weld_pass.polarity
    out.travel_speed = mm_to_m(weld_pass.travel_speed_mm_s)
    out.work_angle = float(np.deg2rad(weld_pass.work_angle_deg))
    out.travel_angle = float(np.deg2rad(weld_pass.travel_angle_deg))
    out.stickout = mm_to_m(weld_pass.stickout_mm)
    out.weave_shape = weld_pass.weave.shape.value
    out.weave_amplitude = mm_to_m(weld_pass.weave.amplitude_mm)
    out.weave_frequency = float(weld_pass.weave.frequency_hz)
    out.weave_dwell = float(weld_pass.weave.dwell_ms) / 1000.0
    out.arc_start_delay = float(weld_pass.arc_start_delay_s)
    out.crater_fill = float(weld_pass.crater_fill_s)
    out.burnback = float(weld_pass.burnback_s)
    # kJ/mm -> J/m is a factor of 1e6.
    out.heat_input = float(weld_pass.heat_input_kj_mm()) * 1.0e6
    return out


def procedure_to_msg(procedure: WeldProcedure, msgs: Any) -> Any:
    out = msgs.WeldProcedure()
    out.wps_id = procedure.wps_id
    out.seam_id = procedure.seam_id
    out.base_material = procedure.base_material
    out.filler = procedure.filler
    out.standard = procedure.standard
    out.welding_position = procedure.position.value
    out.joint_type = procedure.joint_type.value
    out.thickness = mm_to_m(procedure.thickness_mm)
    out.target_leg = mm_to_m(procedure.target_leg_mm)
    out.preheat = float(procedure.preheat_c)
    out.interpass_max = float(procedure.interpass_max_c)
    out.heat_input_max = float(procedure.heat_input_max_kj_mm) * 1.0e6
    out.passes = [pass_to_msg(p, msgs) for p in procedure.passes]
    out.warnings = list(procedure.warnings)
    return out


# ---------------------------------------------------------------------------
# machine setpoints
# ---------------------------------------------------------------------------


def setpoints_to_msg(setpoints: MachineSetpoints, msgs: Any) -> Any:
    out = msgs.MachineSetpoints()
    out.pass_index = int(setpoints.pass_index)
    out.has_job_number = setpoints.job_number is not None
    out.job_number = int(setpoints.job_number or 0)
    out.current = float(setpoints.current_a)
    out.voltage = float(setpoints.voltage_v)
    out.wire_speed = float(setpoints.wire_speed_m_min) / 60.0
    out.arc_length_correction = float(setpoints.arc_length_correction)
    out.dynamic_correction = float(setpoints.dynamic_correction)
    out.gas_flow = float(setpoints.gas_flow_l_min)
    out.pulse_frequency = float(setpoints.pulse_frequency_hz)
    out.machine_profile = setpoints.machine_profile
    out.analog_names = list(setpoints.analog_words)
    out.analog_values = [int(v) for v in setpoints.analog_words.values()]
    out.digital_names = list(setpoints.digital_bits)
    out.digital_values = [bool(v) for v in setpoints.digital_bits.values()]
    out.notes = list(setpoints.notes)
    return out


def msg_to_setpoints(msg: Any) -> MachineSetpoints:
    return MachineSetpoints(
        pass_index=int(msg.pass_index),
        job_number=int(msg.job_number) if msg.has_job_number else None,
        current_a=float(msg.current),
        voltage_v=float(msg.voltage),
        wire_speed_m_min=float(msg.wire_speed) * 60.0,
        arc_length_correction=float(msg.arc_length_correction),
        dynamic_correction=float(msg.dynamic_correction),
        gas_flow_l_min=float(msg.gas_flow),
        pulse_frequency_hz=float(msg.pulse_frequency),
        analog_words=dict(zip(msg.analog_names, [int(v) for v in msg.analog_values], strict=False)),
        digital_bits=dict(
            zip(msg.digital_names, [bool(v) for v in msg.digital_values], strict=False)
        ),
        machine_profile=msg.machine_profile,
        notes=list(msg.notes),
    )


# ---------------------------------------------------------------------------
# report and plan
# ---------------------------------------------------------------------------


def report_to_msg(report: Any, msgs: Any) -> Any:
    out = msgs.JobReport()
    out.ok = bool(report.ok)
    out.reachable = bool(report.reachable)
    out.collision_free = bool(report.collision_free)
    out.cycle_time = float(report.cycle_time_s)
    out.arc_time = float(report.arc_time_s)
    out.air_time = float(report.air_time_s)
    out.duty_cycle = float(report.duty_cycle)
    out.total_weld_length = mm_to_m(report.total_weld_length_mm)
    out.wire_consumption = float(report.wire_consumption_kg)
    out.gas_consumption = float(report.gas_consumption_l)
    out.errors = list(report.errors)
    out.warnings = list(report.warnings)
    return out


def trajectory_from_path(
    points: list[PathPoint],
    joint_names: list[str],
    trajectory_msgs: Any,
    builtin_interfaces: Any,
    speed_scale: float = 1.0,
) -> tuple[Any, list[bool]]:
    """``(JointTrajectory, arc_on flags)`` from planned path points.

    Timestamps come from the same duration model the simulator uses, so the
    trajectory a controller receives takes as long as the cycle time that was
    quoted. Points with no joint solution are dropped and reported by the
    caller -- a trajectory must never contain a target the robot cannot reach.
    """
    from weldforge.sim.simulator import move_duration_s

    trajectory = trajectory_msgs.JointTrajectory()
    trajectory.joint_names = list(joint_names)

    usable = [p for p in points if p.joints is not None]
    arc_on: list[bool] = []
    elapsed = 0.0
    scale = max(1e-3, float(speed_scale))

    # Positioner axes ride in the same trajectory, appended after the arm's
    # joints. A controller that receives only the six arm axes holds the
    # positioner wherever the previous seam left it, which welds the part in
    # the wrong orientation.
    external_names = sorted({name for p in usable for name in (p.external_axes or {})})
    if external_names:
        joint_names = [*joint_names, *external_names]

    for index, point in enumerate(usable):
        if index > 0:
            elapsed += move_duration_s(usable[index - 1], point) / scale
        item = trajectory_msgs.JointTrajectoryPoint()
        item.positions = [float(v) for v in point.joints] + [
            float((point.external_axes or {}).get(name, 0.0)) for name in external_names
        ]
        seconds = int(elapsed)
        item.time_from_start = builtin_interfaces.Duration(
            sec=seconds, nanosec=int(round((elapsed - seconds) * 1e9))
        )
        trajectory.points.append(item)
        arc_on.append(bool(point.weld_on))

    return trajectory, arc_on


def job_to_plan_msg(
    job: WeldJob,
    msgs: Any,
    geometry_msgs: Any,
    trajectory_msgs: Any,
    builtin_interfaces: Any,
    joint_names: list[str],
    frame_id: str = "cell",
    stamp: Any = None,
    include_geometry: bool = True,
) -> Any:
    """A whole :class:`WeldJob` -> ``weldforge_msgs/WeldPlan``."""
    plan = msgs.WeldPlan()
    plan.header.frame_id = frame_id
    if stamp is not None:
        plan.header.stamp = stamp

    plan.job_id = job.id
    plan.job_name = job.name
    plan.part_name = job.part.name if job.part else ""
    plan.cell_name = job.cell.name if job.cell else ""
    plan.robot_name = job.cell.robot_name if job.cell else ""

    plan.seams = [
        seam_to_msg(s, msgs, geometry_msgs, include_geometry, frame_id, stamp) for s in job.seams
    ]
    plan.procedures = [procedure_to_msg(p, msgs) for p in job.procedures.values()]

    if job.placement is not None:
        plan.part_pose = pose_to_msg(job.placement.pose, geometry_msgs)
        plan.placement_feasible = bool(job.placement.feasible)
        plan.placement_score = float(job.placement.score)
        plan.placement_violations = list(job.placement.violations)
    else:
        plan.part_pose = pose_to_msg(np.eye(4), geometry_msgs)
        plan.placement_feasible = False
        plan.placement_score = 0.0

    trajectories = []
    seam_ids: list[str] = []
    pass_indices: list[int] = []
    flags: list[bool] = []
    offsets: list[int] = []

    for path in job.paths:
        trajectory, arc_on = trajectory_from_path(
            path.points, joint_names, trajectory_msgs, builtin_interfaces
        )
        if not trajectory.points:
            continue
        offsets.append(len(flags))
        trajectories.append(trajectory)
        seam_ids.append(path.seam_id)
        pass_indices.append(int(path.pass_index))
        flags.extend(arc_on)

    plan.trajectories = trajectories
    plan.trajectory_seam_ids = seam_ids
    plan.trajectory_pass_indices = pass_indices
    plan.arc_on = flags
    plan.arc_on_offsets = offsets
    plan.report = report_to_msg(job.report, msgs)
    return plan
