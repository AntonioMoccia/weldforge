"""Commanding the positioner.

What a positioner is for
------------------------
A two-axis positioner tilts and rotates the work so that each seam, in turn,
is presented to the torch in the best position it can be. That is not a
convenience: it is the single biggest lever on weld quality and cycle time in
the whole cell.

Turn a seam from overhead (PD) to horizontal-vertical (PB) and the same joint
goes from

* 4 mm/s with a stiff short-arc pool and weaving on every fill pass, in seven
  passes, to
* 7 mm/s in spray transfer with no weaving at all, in four.

And a seam facing the floor is not slow, it is **unreachable**. On the
customer weldment in this repository, a static bench placement leaves 5 seams
of 12 out of the robot's envelope; the positioner is what recovers them.

Indexed, not coordinated
------------------------
This module solves **indexed** positioning: the positioner moves to a setting,
stops, the robot welds the seam, then it moves on. That is how most two-axis
cells actually run discrete seams, and it is what the kinematics here can
honestly promise.

Fully **coordinated** motion -- positioner and robot moving together so a
circumferential seam stays in the gutter throughout -- is a different problem.
It needs the controller's own coordinated-motion option (RAPID's
``MultiMove``, KUKA's ``#EX_AX``) and a velocity-level solution across nine
axes. It is deliberately **not** implemented here rather than approximated,
because an approximation of coordinated motion drags the torch off the joint.
Seams that would need it are reported: see
:attr:`PositionerPlan.needs_coordinated`.

Axis chaining
-------------
Axes compose left to right in the order the cell declares them:

.. code-block:: text

    T_cell_part  =  A1(v1) @ A2(v2) @ ... @ T_mount_part

Each ``Ai`` is defined by its **nominal** axis and origin in the cell frame,
at the zero position. Composition then carries the later axes correctly: at
``v1 = 90 deg`` the second axis's effective direction in the cell is
``A1 . axis2``, which is what physically happens when a tilt table carries a
rotary faceplate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ..geom import transform as tf
from ..kin import RobotModel, solve_analytic
from ..kin.ik import IKOptions, nearest_solution
from ..model import ExternalAxis, Seam, WeldCell, WeldPosition
from ..path.torch import torch_pose, wire_direction

logger = logging.getLogger(__name__)

__all__ = [
    "PositionerOptions",
    "SeamPositioning",
    "PositionerPlan",
    "axes_transform",
    "solve_positioner_for_seam",
    "solve_positioner",
]


@dataclass
class PositionerOptions:
    """Search effort and what counts as a good presentation."""

    # Coarse grid step per axis, degrees for rotary and millimetres for linear.
    coarse_step_deg: float = 20.0
    coarse_step_mm: float = 150.0
    # Refinement passes around the best coarse setting, halving the step each time.
    refine_passes: int = 3
    samples_per_seam: int = 3
    # Relative weight of welding position against reachability. Reachability
    # has to dominate: a beautifully presented seam the robot cannot touch is
    # worth nothing.
    position_weight: float = 1.0
    reach_weight: float = 3.0
    # Prefer settings close to the previous seam's, so the positioner makes
    # small indexed moves instead of swinging through its whole travel between
    # every weld. Expressed as a cost per radian of axis motion.
    continuity_weight: float = 0.15
    ik: IKOptions = field(default_factory=IKOptions)
    # A seam whose welding position changes along its own length by more than
    # this cannot be fixed by one indexed setting, and is flagged as needing
    # coordinated motion.
    coordinated_threshold: float = 0.5
    max_candidates: int = 4000


@dataclass
class SeamPositioning:
    """The positioner setting chosen for one seam."""

    seam_id: str
    values: dict[str, float] = field(default_factory=dict)
    part_pose: np.ndarray = field(default_factory=tf.identity)
    position: WeldPosition = WeldPosition.PA
    reachable_fraction: float = 0.0
    score: float = 0.0
    improved_from: WeldPosition | None = None
    needs_coordinated: bool = False
    note: str = ""

    @property
    def reachable(self) -> bool:
        return self.reachable_fraction >= 0.999

    def summary(self) -> dict:
        return {
            "seam_id": self.seam_id,
            "values_deg": {k: round(float(np.rad2deg(v)), 2) for k, v in self.values.items()},
            "position": self.position.value,
            "was": self.improved_from.value if self.improved_from else None,
            "reachable_fraction": round(self.reachable_fraction, 3),
            "score": round(self.score, 4),
            "needs_coordinated": self.needs_coordinated,
            "note": self.note,
        }


@dataclass
class PositionerPlan:
    """Positioner settings for every seam of a job."""

    per_seam: dict[str, SeamPositioning] = field(default_factory=dict)
    axis_names: list[str] = field(default_factory=list)
    mount_pose: np.ndarray = field(default_factory=tf.identity)
    seconds: float = 0.0

    def values_for(self, seam_id: str) -> dict[str, float]:
        entry = self.per_seam.get(seam_id)
        return dict(entry.values) if entry is not None else {}

    def pose_for(self, seam_id: str) -> np.ndarray:
        """Part pose in the cell while this seam is being welded."""
        entry = self.per_seam.get(seam_id)
        return entry.part_pose.copy() if entry is not None else self.mount_pose.copy()

    @property
    def needs_coordinated(self) -> list[str]:
        return [s.seam_id for s in self.per_seam.values() if s.needs_coordinated]

    @property
    def improved(self) -> list[str]:
        """Seams the positioner actually improved the position of."""
        return [
            s.seam_id
            for s in self.per_seam.values()
            if s.improved_from is not None and s.improved_from is not s.position
        ]

    def index_motion_rad(self) -> float:
        """Total positioner travel across the sequence, for the cycle time."""
        total = 0.0
        previous: dict[str, float] | None = None
        for entry in self.per_seam.values():
            if previous is not None:
                total += sum(
                    abs(entry.values.get(name, 0.0) - previous.get(name, 0.0))
                    for name in self.axis_names
                )
            previous = entry.values
        return total

    def summary(self) -> dict:
        downhand = sum(1 for s in self.per_seam.values() if s.position.is_downhand)
        return {
            "axes": list(self.axis_names),
            "seams": len(self.per_seam),
            "downhand": downhand,
            "reachable": sum(1 for s in self.per_seam.values() if s.reachable),
            "improved": len(self.improved),
            "needs_coordinated": self.needs_coordinated,
            "index_motion_deg": round(float(np.rad2deg(self.index_motion_rad())), 1),
            "seconds": round(self.seconds, 3),
            "per_seam": [s.summary() for s in self.per_seam.values()],
        }


# ---------------------------------------------------------------------------
# kinematics of the positioner
# ---------------------------------------------------------------------------


def axes_transform(axes: list[ExternalAxis], values: dict[str, float]) -> np.ndarray:
    """Pose contributed by the external axes at the given values.

    Composed left to right in declaration order, so a tilt axis correctly
    carries the rotary axis mounted on it.
    """
    out = tf.identity()
    for axis in axes:
        out = out @ axis.transform(float(values.get(axis.name, 0.0)))
    return out


def _axis_samples(axis: ExternalAxis, step_deg: float, step_mm: float) -> np.ndarray:
    """Candidate values for one axis on a coarse grid, always including zero."""
    step = float(np.deg2rad(step_deg)) if axis.kind == "rotary" else float(step_mm)
    span = axis.limit_hi - axis.limit_lo
    count = max(2, int(np.ceil(span / max(step, 1e-6))) + 1)
    values = np.linspace(axis.limit_lo, axis.limit_hi, count)
    # The home position must always be a candidate: often it is the answer,
    # and a grid that misses it makes the positioner move for nothing.
    if not np.any(np.isclose(values, 0.0, atol=step / 4)) and (
        axis.limit_lo <= 0.0 <= axis.limit_hi
    ):
        values = np.sort(np.append(values, 0.0))
    return values


# ---------------------------------------------------------------------------
# scoring one setting
# ---------------------------------------------------------------------------


def _position_quality(position: WeldPosition) -> float:
    """How good a welding position is, 1.0 best.

    Flat and horizontal-vertical are what a positioner exists to achieve;
    vertical-up is workable; overhead is what it exists to avoid.
    """
    return {
        WeldPosition.PA: 1.00,
        WeldPosition.PB: 0.95,
        WeldPosition.PC: 0.60,
        WeldPosition.PG: 0.55,
        WeldPosition.PF: 0.45,
        WeldPosition.PD: 0.15,
        WeldPosition.PE: 0.10,
    }[position]


def _evaluate(
    seam: Seam,
    pose: np.ndarray,
    robot: RobotModel,
    options: PositionerOptions,
    seed: np.ndarray,
) -> tuple[float, WeldPosition, float, bool, np.ndarray]:
    """``(score, position, reachable_fraction, varies, seed)`` for one setting."""
    moved = seam.transformed(pose)
    position = moved.dominant_position

    indices = (
        list(range(moved.n_points))
        if moved.n_points <= options.samples_per_seam
        else [int(round(v)) for v in np.linspace(0, moved.n_points - 1, options.samples_per_seam)]
    )

    solved = 0
    positions_seen: set[WeldPosition] = set()
    for i in indices:
        positions_seen.add(moved.position_at(i))
        groove = float(moved.dihedral[i]) if len(moved.dihedral) > i else np.pi / 2
        wire = wire_direction(
            moved.bisector[i] if len(moved.bisector) > i else np.array([0.0, 0.0, -1.0]),
            moved.tangents[i] if len(moved.tangents) > i else np.array([1.0, 0.0, 0.0]),
            moved.normal_a[i] if len(moved.normal_a) > i else np.array([0.0, 0.0, 1.0]),
            moved.normal_b[i] if len(moved.normal_b) > i else np.array([0.0, 1.0, 0.0]),
            groove,
            work_angle_deg=float(np.rad2deg(groove / 2.0)),
            travel_angle_deg=0.0,
        )
        target = torch_pose(
            moved.points[i],
            wire,
            moved.tangents[i] if len(moved.tangents) > i else np.array([1.0, 0.0, 0.0]),
        )
        solutions = [s for s in solve_analytic(robot, target, options=options.ik) if s.ok]
        if solutions:
            solved += 1
            seed = nearest_solution(solutions, seed).joints

    fraction = solved / max(1, len(indices))
    # A seam whose own position varies along its length cannot be fixed by one
    # indexed setting.
    varies = len(positions_seen) > 1

    total_weight = options.position_weight + options.reach_weight
    score = (
        options.position_weight * _position_quality(position) + options.reach_weight * fraction
    ) / max(total_weight, 1e-9)
    return score, position, fraction, varies, seed


# ---------------------------------------------------------------------------
# solving
# ---------------------------------------------------------------------------


def solve_positioner_for_seam(
    seam: Seam,
    cell: WeldCell,
    mount_pose: np.ndarray,
    robot: RobotModel,
    options: PositionerOptions | None = None,
    previous: dict[str, float] | None = None,
    seed: np.ndarray | None = None,
) -> SeamPositioning:
    """Best indexed positioner setting for one seam.

    *mount_pose* places the part on the faceplate with the axes at zero;
    *previous* is the setting used for the seam before, so the search can
    prefer a small index move.
    """
    opts = options or PositionerOptions()
    axes = [a for a in cell.external_axes if a.coupled]
    joint_seed = robot.home_joints() if seed is None else seed

    result = SeamPositioning(seam_id=seam.id, part_pose=mount_pose.copy())

    if not axes:
        score, position, fraction, varies, _ = _evaluate(seam, mount_pose, robot, opts, joint_seed)
        result.values = {}
        result.position = position
        result.reachable_fraction = fraction
        result.score = score
        result.note = "cell has no coupled external axes"
        return result

    # Baseline: axes at zero, which is what a bench-only cell gives.
    zero = {a.name: 0.0 for a in axes}
    base_pose = axes_transform(axes, zero) @ mount_pose
    base_score, base_position, base_fraction, base_varies, joint_seed = _evaluate(
        seam, base_pose, robot, opts, joint_seed
    )

    best = (base_score, zero, base_pose, base_position, base_fraction, base_varies)

    grids = [_axis_samples(a, opts.coarse_step_deg, opts.coarse_step_mm) for a in axes]
    combinations = int(np.prod([len(g) for g in grids]))
    if combinations > opts.max_candidates:
        # Thin the grids evenly rather than truncating, so the whole travel is
        # still covered.
        factor = (combinations / opts.max_candidates) ** (1.0 / len(grids))
        grids = [g[:: max(1, int(np.ceil(factor)))] for g in grids]
        logger.debug(
            "positioner grid thinned from %d to %d candidates",
            combinations,
            int(np.prod([len(g) for g in grids])),
        )

    def consider(values: dict[str, float]) -> None:
        nonlocal best, joint_seed
        pose = axes_transform(axes, values) @ mount_pose
        score, position, fraction, varies, joint_seed = _evaluate(
            seam, pose, robot, opts, joint_seed
        )
        if opts.continuity_weight > 0.0:
            # No previous seam means the positioner is at home, and moving
            # from home costs the same indexing time as moving anywhere else.
            # Treating the first seam as free makes the planner swing the
            # positioner for no gain on every job's first weld.
            reference = previous if previous is not None else zero
            motion = sum(abs(values.get(a.name, 0.0) - reference.get(a.name, 0.0)) for a in axes)
            score -= opts.continuity_weight * motion / (2.0 * np.pi)
        if score > best[0]:
            best = (score, dict(values), pose, position, fraction, varies)

    for combination in product(*grids):
        consider({a.name: float(v) for a, v in zip(axes, combination, strict=True)})

    # Refine around the winner, halving the step each pass.
    step_deg = opts.coarse_step_deg
    step_mm = opts.coarse_step_mm
    for _ in range(max(0, opts.refine_passes)):
        step_deg /= 2.0
        step_mm /= 2.0
        centre = dict(best[1])
        for axis in axes:
            delta = float(np.deg2rad(step_deg)) if axis.kind == "rotary" else step_mm
            for sign in (-1.0, 1.0):
                candidate = dict(centre)
                value = candidate.get(axis.name, 0.0) + sign * delta
                if not (axis.limit_lo <= value <= axis.limit_hi):
                    continue
                candidate[axis.name] = value
                consider(candidate)

    score, values, pose, position, fraction, varies = best
    result.values = values
    result.part_pose = pose
    result.position = position
    result.reachable_fraction = fraction
    result.score = score
    result.improved_from = base_position
    result.needs_coordinated = bool(varies)
    if varies:
        result.note = (
            "the welding position changes along this seam, so no single indexed "
            "setting fits it; coordinated motion would be needed"
        )
    elif position is not base_position:
        result.note = f"positioner turns this seam from {base_position.value} to {position.value}"
    elif fraction > base_fraction:
        result.note = "positioner brings this seam into reach"
    return result


def solve_positioner(
    seams: list[Seam],
    cell: WeldCell,
    mount_pose: np.ndarray,
    robot: RobotModel,
    options: PositionerOptions | None = None,
) -> PositionerPlan:
    """Positioner settings for a whole job, one indexed setting per seam.

    Seams are solved in the order given, each preferring a setting near the
    previous one, so the positioner makes small index moves through the
    sequence rather than swinging through its travel between every weld.
    """
    import time

    opts = options or PositionerOptions()
    axes = [a for a in cell.external_axes if a.coupled]
    plan = PositionerPlan(axis_names=[a.name for a in axes], mount_pose=mount_pose.copy())
    t0 = time.perf_counter()

    previous: dict[str, float] | None = None
    seed = robot.home_joints()
    for seam in seams:
        entry = solve_positioner_for_seam(seam, cell, mount_pose, robot, opts, previous, seed)
        plan.per_seam[seam.id] = entry
        previous = entry.values

    plan.seconds = time.perf_counter() - t0
    if axes:
        logger.info(
            "solve_positioner: %d seams, %d downhand, %d improved, "
            "%d need coordinated motion, %.0f deg of indexing, %.1fs",
            len(seams),
            sum(1 for s in plan.per_seam.values() if s.position.is_downhand),
            len(plan.improved),
            len(plan.needs_coordinated),
            np.rad2deg(plan.index_motion_rad()),
            plan.seconds,
        )
    return plan
