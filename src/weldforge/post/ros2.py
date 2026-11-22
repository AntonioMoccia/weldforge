"""ROS 2 output: joint trajectories and a seam/procedure description.

Two files, because the ROS side needs both halves:

* a **trajectory** file holding ``trajectory_msgs/JointTrajectory`` content as
  YAML, one trajectory per pass, with timestamps from the simulated motion;
* a **weld plan** file holding the seams, procedures and machine setpoints, so
  the welder-I/O node knows what to command and when.

This is the one target that works in **metres and radians**, because that is
what ROS and MoveIt use. The conversion happens here and nowhere else.

The files are plain YAML on purpose: they load with ``yaml.safe_load`` in any
node, need no custom message definitions to inspect, and can be diffed and
version-controlled alongside the cell configuration.
"""

from __future__ import annotations

import yaml

from ..geom import transform as tf
from ..model import WeldJob
from ..units import mm_to_m
from .base import Postprocessor, PostResult, header_lines
from .registry import register

__all__ = ["Ros2Post"]


@register
class Ros2Post(Postprocessor):
    key = "ros2"
    target = "ROS 2 (JointTrajectory YAML + weld plan)"
    units = "m"
    extension = "yaml"
    comment = "#"

    def generate(self, job: WeldJob) -> PostResult:
        result = PostResult(target=self.target)
        opts = self.options
        robot = job.cell.robot_name if job.cell else ""

        joint_names = [f"joint_{i + 1}" for i in range(6)]
        if job.paths:
            first = next(
                (p for path in job.paths for p in path.points if p.joints is not None), None
            )
            if first is not None:
                joint_names = [f"joint_{i + 1}" for i in range(len(first.joints))]

        trajectories = []
        plan_entries = []

        from ..sim.simulator import move_duration_s

        for path in job.paths:
            points, skipped = self.usable_points(path)
            result.n_skipped += skipped
            usable = [p for p in points if p.joints is not None]
            if len(usable) < 2:
                result.warnings.append(
                    f"seam {path.seam_id} pass {path.pass_index + 1}: "
                    "no joint solutions, trajectory omitted"
                )
                continue

            # Positioner axes ride in the same trajectory, appended after the
            # arm's joints. A controller given only the six arm axes holds the
            # positioner wherever the previous seam left it, so the part is
            # welded in the wrong orientation.
            external_names = sorted({name for p in usable for name in (p.external_axes or {})})
            names = [*joint_names, *external_names]

            elapsed = 0.0
            trajectory_points = []
            for index, point in enumerate(usable):
                if index > 0:
                    elapsed += move_duration_s(usable[index - 1], point)
                trajectory_points.append(
                    {
                        "positions": [round(float(v), 6) for v in point.joints]
                        + [
                            round(float((point.external_axes or {}).get(name, 0.0)), 6)
                            for name in external_names
                        ],
                        "time_from_start": round(elapsed, 4),
                        "weld_on": bool(point.weld_on),
                        # Cartesian pose in metres, for logging and for a
                        # Cartesian controller that wants it.
                        "pose": {
                            "position": [
                                round(mm_to_m(float(v)), 6) for v in tf.translation_of(point.pose)
                            ],
                            "orientation_xyzw": [
                                round(float(v), 6) for v in tf.matrix_to_quat(point.pose)
                            ],
                        },
                    }
                )
                result.n_moves += 1

            trajectories.append(
                {
                    "seam_id": path.seam_id,
                    "pass_index": path.pass_index,
                    "joint_names": names,
                    "duration_s": round(elapsed, 3),
                    "points": trajectory_points,
                }
            )

            weld_pass = self.pass_for(job, path)
            seam = job.seam(path.seam_id)
            if weld_pass is not None:
                plan_entries.append(
                    {
                        "seam_id": path.seam_id,
                        "pass_index": path.pass_index,
                        "pass_name": weld_pass.name,
                        "joint_type": seam.joint_type.value if seam else "",
                        "position": seam.dominant_position.value if seam else "",
                        "length_mm": round(seam.length_mm, 2) if seam else 0.0,
                        "process": weld_pass.process.value,
                        "transfer_mode": weld_pass.transfer_mode.value,
                        "current_a": weld_pass.current_a,
                        "voltage_v": weld_pass.voltage_v,
                        "wire_speed_m_min": weld_pass.wire_speed_m_min,
                        "wire_diameter_mm": weld_pass.wire_diameter_mm,
                        "gas": weld_pass.gas,
                        "gas_flow_l_min": weld_pass.gas_flow_l_min,
                        "travel_speed_mm_s": weld_pass.travel_speed_mm_s,
                        "stickout_mm": weld_pass.stickout_mm,
                        "arc_start_delay_s": weld_pass.arc_start_delay_s,
                        "crater_fill_s": weld_pass.crater_fill_s,
                        "weave": {
                            "shape": weld_pass.weave.shape.value,
                            "amplitude_mm": weld_pass.weave.amplitude_mm,
                            "frequency_hz": weld_pass.weave.frequency_hz,
                            "dwell_ms": weld_pass.weave.dwell_ms,
                        },
                        "heat_input_kj_mm": round(weld_pass.heat_input_kj_mm(), 3),
                        "positioner_rad": {
                            name: round(float(value), 6)
                            for name, value in (usable[0].external_axes or {}).items()
                        },
                    }
                )

        header = (
            "\n".join(header_lines(job, opts, self.comment)) + "\n" if opts.include_header else ""
        )

        trajectory_doc = {
            "weldforge_version": 1,
            "job_id": job.id,
            "job_name": job.name,
            "robot": robot,
            "units": {"length": "m", "angle": "rad", "time": "s"},
            "frame_id": "cell",
            "trajectories": trajectories,
        }
        plan_doc = {
            "weldforge_version": 1,
            "job_id": job.id,
            "job_name": job.name,
            "part": job.part.name if job.part else "",
            "cell": job.cell.name if job.cell else "",
            "placement": (
                {
                    "position_m": [
                        round(mm_to_m(float(v)), 6) for v in tf.translation_of(job.placement.pose)
                    ],
                    "orientation_xyzw": [
                        round(float(v), 6) for v in tf.matrix_to_quat(job.placement.pose)
                    ],
                }
                if job.placement is not None
                else None
            ),
            "report": job.report.summary(),
            "passes": plan_entries,
        }

        name = opts.program_name
        result.files[f"{name}_trajectories.yaml"] = header + yaml.safe_dump(
            trajectory_doc, sort_keys=False, default_flow_style=False
        )
        result.files[f"{name}_weld_plan.yaml"] = header + yaml.safe_dump(
            plan_doc, sort_keys=False, default_flow_style=False
        )
        return result
