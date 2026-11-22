"""Offline simulation of a welding job.

Produces the numbers an operator has to sign off before the cell runs: is
every station reachable, does anything collide, how long is the cycle, how
much wire and gas does it burn. Nothing here moves a real robot -- that is the
point. A plan that fails here fails for free.

Cycle time is computed from whichever limit binds, not from the programmed
speed alone:

* a **welding** move is governed by the travel speed the procedure asked for --
  that is the whole reason the number exists;
* an **air** move is governed by the robot's own joint speeds, because a
  250 mm/s Cartesian command on a move that swings axis 1 through 90 degrees is
  not what the robot will do.

Taking the greater of the two keeps the estimate honest. Published cycle times
that assume Cartesian speed throughout are routinely 30% optimistic, and a
cell sold on an optimistic cycle time is a cell that misses its quoted output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..collide import CollisionScene, robot_capsules, torch_capsules
from ..kin import RobotModel, fk_all
from ..model import JobReport, PathPoint, WeldJob

logger = logging.getLogger(__name__)

__all__ = ["SimOptions", "simulate", "move_duration_s"]


@dataclass
class SimOptions:
    """What to check and how much slack to allow."""

    check_collisions: bool = True
    check_arm: bool = True
    torch_clearance_mm: float = 1.0
    arm_clearance_mm: float = 20.0
    # Fraction of each joint's rated speed used for air moves. Controllers
    # never run at 100% of the data-sheet figure once acceleration limits and
    # path blending are taken into account.
    joint_speed_factor: float = 0.7
    # Fixed overheads, seconds.
    arc_start_s: float = 0.4
    arc_stop_s: float = 0.5
    seam_change_s: float = 1.0
    # Positioner indexing between seams. A two-axis positioner takes real
    # seconds to swing, and a cycle time that ignores it is optimistic by
    # exactly the amount a plant manager notices.
    positioner_speed_deg_s: float = 30.0
    positioner_settle_s: float = 0.5
    gas_preflow_s: float = 0.3
    gas_postflow_s: float = 1.0
    max_reported_errors: int = 50


def move_duration_s(
    previous: PathPoint,
    current: PathPoint,
    robot: RobotModel | None = None,
    joint_speed_factor: float = 0.7,
) -> float:
    """Time for one move, from whichever of Cartesian or joint speed binds."""
    distance = float(np.linalg.norm(current.position - previous.position))
    cartesian = distance / max(current.speed_mm_s, 1e-6)

    joint_time = 0.0
    if (
        robot is not None
        and previous.joints is not None
        and current.joints is not None
        and len(previous.joints) == len(current.joints)
    ):
        delta = np.abs(current.joints - previous.joints)
        speeds = robot.max_velocities * max(joint_speed_factor, 1e-3)
        joint_time = float(np.max(delta / np.maximum(speeds, 1e-9)))

    return max(cartesian, joint_time)


def simulate(
    job: WeldJob,
    robot: RobotModel | None = None,
    scene: CollisionScene | None = None,
    options: SimOptions | None = None,
) -> JobReport:
    """Simulate the job's paths and return the report.

    Warnings and errors already on the job's report are **carried over**.
    Earlier stages -- seam detection, the welding procedure, the placement
    search, the positioner -- have things to say that the operator needs, and
    simply returning a fresh report threw all of them away the moment
    simulation ran.
    """
    opts = options or SimOptions()
    report = JobReport(
        errors=list(job.report.errors),
        warnings=list(job.report.warnings),
    )

    if not job.paths:
        report.errors.append("job has no paths to simulate")
        return report

    reachable = True
    collision_free = True

    for path in job.paths:
        seam = job.seam(path.seam_id)
        procedure = job.procedures.get(path.seam_id)
        weld_pass = (
            procedure.passes[path.pass_index]
            if procedure is not None and path.pass_index < len(procedure.passes)
            else None
        )

        arc_time = 0.0
        air_time = 0.0
        weld_length = 0.0

        previous: PathPoint | None = None
        for point in path.points:
            if point.joints is None and robot is not None:
                reachable = False
                if len(report.errors) < opts.max_reported_errors:
                    report.errors.append(
                        f"seam {path.seam_id} pass {path.pass_index + 1}: "
                        f"unreachable station {point.comment or ''}".strip()
                    )

            if point.collision:
                collision_free = False
                if len(report.errors) < opts.max_reported_errors:
                    report.errors.append(
                        f"seam {path.seam_id} pass {path.pass_index + 1}: {point.collision}"
                    )
            elif opts.check_collisions and scene is not None:
                capsules = torch_capsules(point.pose)
                wire = [c for c in capsules if c.name.endswith(":wire")]
                nozzle = [c for c in capsules if not c.name.endswith(":wire")]
                contacts = scene.check(wire, opts.torch_clearance_mm, ignore={"part"})
                contacts.extend(scene.check(nozzle, opts.torch_clearance_mm))
                if opts.check_arm and robot is not None and point.joints is not None:
                    contacts.extend(
                        scene.check(
                            robot_capsules(fk_all(robot, point.joints)),
                            opts.arm_clearance_mm,
                        )
                    )
                if contacts:
                    collision_free = False
                    point.collision = str(contacts[0])
                    if len(report.errors) < opts.max_reported_errors:
                        report.errors.append(
                            f"seam {path.seam_id} pass {path.pass_index + 1}: {contacts[0]}"
                        )

            if previous is not None:
                duration = move_duration_s(previous, point, robot, opts.joint_speed_factor)
                # Arc-on applies to a move only when **both** ends are welding.
                # The move from the approach point to the first station has an
                # arc-on destination but is an air move, and counting it as
                # weld inflated the deposited length by the whole stand-off --
                # 480 mm reported for a 400 mm weld.
                if point.weld_on and previous.weld_on:
                    arc_time += duration
                    weld_length += float(np.linalg.norm(point.position - previous.position))
                else:
                    air_time += duration
            previous = point

        if any(p.weld_on for p in path.points):
            arc_time += opts.arc_start_s + opts.arc_stop_s
            air_time += opts.gas_preflow_s + opts.gas_postflow_s

        report.arc_time_s += arc_time
        report.air_time_s += air_time + opts.seam_change_s
        report.total_weld_length_mm += weld_length

        if weld_pass is not None and weld_length > 0.0:
            report.wire_consumption_kg += weld_pass.deposited_mass_kg(weld_length)
            gas_time = arc_time + opts.gas_preflow_s + opts.gas_postflow_s
            report.gas_consumption_l += weld_pass.gas_flow_l_min * gas_time / 60.0

        key = path.seam_id
        entry = report.per_seam.setdefault(
            key,
            {
                "length_mm": round(seam.length_mm, 1) if seam is not None else 0.0,
                "joint_type": seam.joint_type.value if seam is not None else "",
                "position": seam.dominant_position.value if seam is not None else "",
                "passes": 0,
                "arc_time_s": 0.0,
                "air_time_s": 0.0,
                "unreachable": 0,
                "collisions": 0,
            },
        )
        entry["passes"] += 1
        entry["arc_time_s"] = round(entry["arc_time_s"] + arc_time, 2)
        entry["air_time_s"] = round(entry["air_time_s"] + air_time, 2)
        entry["unreachable"] += sum(1 for p in path.points if p.joints is None and robot)
        entry["collisions"] += sum(1 for p in path.points if p.collision)

    # --- positioner indexing ------------------------------------------------
    index_time = 0.0
    previous_axes: dict[str, float] | None = None
    for path in job.paths:
        axes: dict[str, float] = {}
        for point in path.points:
            if point.external_axes:
                axes = point.external_axes
                break
        if not axes:
            continue
        if previous_axes is not None and axes != previous_axes:
            travel = max(abs(axes.get(name, 0.0) - previous_axes.get(name, 0.0)) for name in axes)
            index_time += (
                float(np.rad2deg(travel)) / max(opts.positioner_speed_deg_s, 1e-6)
                + opts.positioner_settle_s
            )
        previous_axes = axes
    if index_time > 0.0:
        report.air_time_s += index_time
        total = report.arc_time_s + report.air_time_s
        report.warnings.append(
            f"positioner indexing adds {index_time:.0f} s to the cycle "
            f"({index_time / max(total, 1e-9):.0%} of it)"
        )

    report.cycle_time_s = report.arc_time_s + report.air_time_s
    report.reachable = reachable
    report.collision_free = collision_free

    # --- warnings the operator should see before running ------------------
    for seam_id, procedure in job.procedures.items():
        for warning in procedure.warnings:
            report.warnings.append(f"seam {seam_id}: {warning}")
    # Carried-over warnings can repeat what a stage already said; keep the
    # first of each so the list stays readable.
    report.warnings = list(dict.fromkeys(report.warnings))
    report.errors = list(dict.fromkeys(report.errors))

    if report.duty_cycle < 0.25 and report.cycle_time_s > 0.0:
        report.warnings.append(
            f"arc-on time is only {report.duty_cycle:.0%} of the cycle: the part "
            "is spending most of its time being travelled over, so the seam "
            "order or the fixture layout is worth revisiting"
        )

    logger.info(
        "simulate: %s, cycle %.1fs (arc %.1fs, air %.1fs), %.0f mm weld, %.0f g wire, %.0f l gas",
        "OK" if report.ok else "FAILED",
        report.cycle_time_s,
        report.arc_time_s,
        report.air_time_s,
        report.total_weld_length_mm,
        report.wire_consumption_kg * 1000.0,
        report.gas_consumption_l,
    )
    return report
