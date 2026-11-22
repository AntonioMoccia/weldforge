"""Assemble the robot path: torch poses, air moves and joint solutions.

This is where the geometric plan becomes something a robot can run. For every
pass of every seam, in welding order:

1. generate the welding stations (:mod:`weldforge.path.torch`);
2. bracket them with an approach and a retract backed off along the torch axis;
3. solve inverse kinematics at each station, **seeded from the previous one**,
   so the arm stays in one configuration for the whole bead;
4. check the torch and the arm against the cell.

Seeding is the part that makes the difference between a path a robot will run
and one it will refuse. Solved independently, consecutive stations can land in
different arm configurations and the robot stops mid-weld to swing round.
Seeded, each station takes the branch nearest its predecessor.

Where a station genuinely cannot be reached, it is recorded with the reason
rather than dropped: an operator needs to know *which* 20 mm of a seam is out
of reach, so they can move the part or the fixture, not just that "the seam
failed".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..collide import CollisionScene, robot_capsules, torch_capsules
from ..kin import IKOptions, RobotModel, fk_all, solve
from ..model import (
    MotionType,
    PathPoint,
    Seam,
    TorchPath,
    WeldJob,
    WeldPass,
    WeldProcedure,
)
from .sequence import SequencedSeam, SequenceOptions, sequence_seams
from .torch import approach_point, retract_point, torch_poses_for_pass

logger = logging.getLogger(__name__)

__all__ = ["PathOptions", "PathBuildReport", "build_pass_path", "build_job_paths"]


@dataclass
class PathOptions:
    """Stand-offs, speeds and what to check."""

    approach_standoff_mm: float = 40.0
    retract_standoff_mm: float = 60.0
    air_speed_mm_s: float = 250.0
    # Clearance the torch nozzle must keep from anything static, mm. Tight,
    # because the nozzle has to get into the groove; the arm's own clearance is
    # separate and generous.
    torch_clearance_mm: float = 2.0
    arm_clearance_mm: float = 20.0
    torch_nozzle_length_mm: float = 120.0
    torch_nozzle_radius_mm: float = 22.0  # gas cup outer radius
    torch_tip_radius_mm: float = 4.0  # wire and contact tip
    check_collisions: bool = True
    check_arm: bool = True
    solve_ik: bool = True
    station_spacing_mm: float = 0.0  # 0 = keep the seam's own stations
    # Interpolate the weave into the path instead of letting the controller
    # weave. Only for a controller with no weave function: see
    # weldforge.path.torch.torch_poses_for_pass for why it is off.
    bake_weave: bool = False
    max_stations_per_pass: int = 2000
    ik: IKOptions = field(default_factory=IKOptions)
    sequence: SequenceOptions = field(default_factory=SequenceOptions)
    # Return to a safe posture between seams when the air move is long.
    safe_move_threshold_mm: float = 600.0


@dataclass
class PathBuildReport:
    """What happened while building the paths."""

    n_paths: int = 0
    n_points: int = 0
    n_unreachable: int = 0
    n_collisions: int = 0
    unreachable_seams: list[str] = field(default_factory=list)
    colliding_seams: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "paths": self.n_paths,
            "points": self.n_points,
            "unreachable_points": self.n_unreachable,
            "collision_points": self.n_collisions,
            "unreachable_seams": list(dict.fromkeys(self.unreachable_seams)),
            "colliding_seams": list(dict.fromkeys(self.colliding_seams)),
            "messages": self.messages[:40],
            "seconds": round(self.seconds, 3),
        }


# ---------------------------------------------------------------------------
# one pass
# ---------------------------------------------------------------------------


def build_pass_path(
    seam: Seam,
    weld_pass: WeldPass,
    robot: RobotModel | None = None,
    scene: CollisionScene | None = None,
    options: PathOptions | None = None,
    seed: np.ndarray | None = None,
    report: PathBuildReport | None = None,
) -> TorchPath:
    """Build one pass: approach, welding stations, retract, with IK and checks."""
    opts = options or PathOptions()
    rep = report or PathBuildReport()

    stations = torch_poses_for_pass(
        seam,
        weld_pass,
        opts.station_spacing_mm,
        opts.bake_weave,
        opts.max_stations_per_pass,
    )
    if not stations:
        rep.messages.append(f"seam {seam.id}: no stations generated")
        return TorchPath(seam_id=seam.id, pass_index=weld_pass.index)

    points: list[PathPoint] = [
        approach_point(stations[0], opts.approach_standoff_mm, opts.air_speed_mm_s),
        *stations,
        retract_point(stations[-1], opts.retract_standoff_mm, opts.air_speed_mm_s),
    ]
    # The approach move itself is a joint move: getting there is unconstrained.
    points[0].motion = MotionType.JOINT

    current_seed = seed
    for point in points:
        if robot is not None and opts.solve_ik:
            solution = solve(robot, point.pose, current_seed, opts.ik)
            if solution is None or not solution.ok:
                rep.n_unreachable += 1
                rep.unreachable_seams.append(seam.id)
                detail = solution.reason if solution is not None else "no solution"
                if solution is not None and not solution.within_limits and robot is not None:
                    violations = robot.limit_violations(solution.joints)
                    if violations:
                        detail = f"{detail}: {violations[0]}"
                point.comment = (point.comment + " " if point.comment else "") + (
                    f"unreachable ({detail})"
                )
            else:
                point.joints = solution.joints
                point.manipulability = solution.manipulability
                current_seed = solution.joints

        if scene is not None and opts.check_collisions:
            capsules = torch_capsules(
                point.pose,
                opts.torch_nozzle_length_mm,
                opts.torch_tip_radius_mm,
                opts.torch_nozzle_radius_mm,
            )
            # The wire ends *on the joint*: touching the part there is the
            # whole point, so the part is excluded for the wire alone. The
            # nozzle is checked against everything, the part included -- that
            # is the clash that actually breaks a torch.
            wire = [c for c in capsules if c.name.endswith(":wire")]
            nozzle = [c for c in capsules if not c.name.endswith(":wire")]
            clearance = opts.torch_clearance_mm
            contacts = scene.check(wire, clearance, ignore={"part"})
            contacts.extend(scene.check(nozzle, clearance))

            if opts.check_arm and robot is not None and point.joints is not None:
                arm = robot_capsules(fk_all(robot, point.joints))
                contacts.extend(scene.check(arm, opts.arm_clearance_mm))

            if contacts:
                rep.n_collisions += 1
                rep.colliding_seams.append(seam.id)
                point.collision = str(contacts[0])

    rep.n_points += len(points)
    return TorchPath(seam_id=seam.id, pass_index=weld_pass.index, points=points)


# ---------------------------------------------------------------------------
# a whole job
# ---------------------------------------------------------------------------


def build_job_paths(
    job: WeldJob,
    robot: RobotModel | None = None,
    scene: CollisionScene | None = None,
    options: PathOptions | None = None,
) -> tuple[list[TorchPath], PathBuildReport]:
    """Build every path of a job, in welding order.

    The seams are taken in cell coordinates: if the job has a placement, each
    seam is transformed by it first, so the paths come out where the part
    actually sits rather than where it was modelled.
    """
    import time

    opts = options or PathOptions()
    report = PathBuildReport()
    t0 = time.perf_counter()

    # Each seam is placed by its own pose. On a bench that is the single
    # placement, the same for every seam; with a positioner each seam has its
    # own indexed setting, and planning them all at the setup pose would be
    # wrong for all but one of them.
    seams = [seam.transformed(job.part_pose_for(seam.id)) for seam in job.selected_seams]
    if not seams:
        report.messages.append("no seams selected")
        return [], report

    ordered: list[SequencedSeam] = sequence_seams(seams, opts.sequence)

    paths: list[TorchPath] = []
    seed = robot.home_joints() if robot is not None else None

    for item in ordered:
        oriented = item.oriented()
        procedure: WeldProcedure | None = job.procedures.get(item.seam.id)
        if procedure is None or not procedure.passes:
            report.messages.append(f"seam {item.seam.id}: no welding procedure, skipped")
            continue

        if robot is not None and item.air_travel_mm > opts.safe_move_threshold_mm and paths:
            report.messages.append(
                f"long air move of {item.air_travel_mm:.0f} mm before seam "
                f"{item.seam.id}: consider a via point"
            )

        axis_values = job.external_axes_for(item.seam.id)
        for weld_pass in procedure.passes:
            path = build_pass_path(oriented, weld_pass, robot, scene, opts, seed, report)
            # Stamp the positioner setting on every station: the
            # postprocessors need it to emit external-axis targets, and a
            # program with the positioner values missing welds the part in
            # whatever orientation it happens to be left in.
            for point in path.points:
                point.external_axes = dict(axis_values)
            if path.points:
                last = path.points[-1]
                if last.joints is not None:
                    seed = last.joints
                paths.append(path)

    report.n_paths = len(paths)
    report.seconds = time.perf_counter() - t0
    logger.info(
        "build_job_paths: %d paths, %d points, %d unreachable, %d colliding, %.2fs",
        report.n_paths,
        report.n_points,
        report.n_unreachable,
        report.n_collisions,
        report.seconds,
    )
    return paths, report
