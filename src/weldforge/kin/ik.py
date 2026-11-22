"""Inverse kinematics: closed-form for a spherical-wrist arm, numeric otherwise.

Why both
--------
A welding path has a station every 2 mm. A 300 mm seam is 150 targets, each of
which must be solved, checked against joint limits, and compared with its
neighbour so the arm does not flip wrist configuration mid-bead. Numeric IK
alone is too slow for that and -- worse -- it returns *one* solution, chosen by
whatever the seed happened to be, so consecutive stations can land in different
configurations and the controller reconfigures in the middle of a weld.

The closed form returns **all eight** solutions of a standard industrial arm
at once, which is what makes configuration-consistent path planning possible:
pick the branch nearest the previous station and stay on it.

The eight branches are the product of three binary choices:

* **shoulder** -- base turned to face the target, or turned away and reaching
  backwards over itself;
* **elbow** -- up or down;
* **wrist** -- flipped or not (``theta5`` positive or negative).

Geometry assumed by the closed form
-----------------------------------
The classic 6R industrial arm: three axes positioning the wrist centre, then
three intersecting axes orienting the tool. Expressed in the modified DH table
as::

    j1  alpha=0     a=0    d=d1   (base rotation)
    j2  alpha=-90   a=a1   d=0    (shoulder)
    j3  alpha=0     a=a2   d=0    (elbow)
    j4  alpha=-90   a=a3   d=d4   (forearm roll)
    j5  alpha=+90   a=0    d=0    (wrist bend)
    j6  alpha=-90   a=0    d=d6   (tool roll)

KUKA, Fanuc, ABB, Motoman and Comau arms all fit it. A robot that does not --
a UR-style arm with offset wrist axes, or a collaborative arm with a
non-intersecting wrist -- is detected and falls through to the numeric solver,
which has no structural requirements at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..geom import transform as tf
from .robot import RobotModel, fk, jacobian

logger = logging.getLogger(__name__)

__all__ = [
    "IKSolution",
    "IKOptions",
    "ArmGeometry",
    "extract_arm_geometry",
    "solve_analytic",
    "solve_numeric",
    "solve",
    "nearest_solution",
]


@dataclass
class IKSolution:
    """One joint solution with the information needed to choose between them."""

    joints: np.ndarray
    branch: str = ""  # e.g. 'front/up/noflip'
    within_limits: bool = True
    position_error_mm: float = 0.0
    orientation_error_deg: float = 0.0
    manipulability: float = 0.0
    limit_margin: float = 0.0
    iterations: int = 0
    method: str = "analytic"

    @property
    def ok(self) -> bool:
        return (
            self.within_limits and self.position_error_mm < 0.5 and self.orientation_error_deg < 0.5
        )

    @property
    def reason(self) -> str:
        """Why this solution is not usable, in terms an operator can act on.

        "0 mm short" is not actionable. A target can be geometrically within
        reach and still unreachable because a joint runs out of travel, or
        because the wrist cannot hold the orientation -- and the three cases
        call for three different fixes: move the part, turn it over, or change
        the torch angle.
        """
        if self.ok:
            return ""
        if not self.within_limits:
            return "a joint runs out of travel at this posture"
        if self.position_error_mm >= 0.5:
            return f"out of reach by {self.position_error_mm:.0f} mm"
        if self.orientation_error_deg >= 0.5:
            return (
                f"the wrist cannot hold the torch orientation "
                f"({self.orientation_error_deg:.1f} deg off)"
            )
        return "no solution"


@dataclass
class IKOptions:
    """Tolerances and effort for the numeric solver."""

    position_tol_mm: float = 0.05
    orientation_tol_deg: float = 0.05
    max_iterations: int = 150
    damping: float = 0.05  # Levenberg-Marquardt lambda
    step_clip_rad: float = 0.3  # per-iteration joint step cap
    random_restarts: int = 12
    seed: int = 20251122
    respect_limits: bool = True
    # Weighting of orientation against position in the numeric residual. A
    # degree of torch tilt matters about as much as a millimetre of position
    # for a fillet weld, which is what this ratio encodes.
    orientation_weight_mm_per_rad: float = 50.0

    @property
    def orientation_tol_rad(self) -> float:
        return float(np.deg2rad(self.orientation_tol_deg))


# ---------------------------------------------------------------------------
# structure detection
# ---------------------------------------------------------------------------


@dataclass
class ArmGeometry:
    """The six lengths the closed form needs."""

    d1: float
    a1: float
    a2: float
    a3: float
    d4: float
    d6: float
    # Sign conventions lifted from the DH table so the solver matches the model.
    sign: np.ndarray = field(default_factory=lambda: np.ones(6))
    theta_offset: np.ndarray = field(default_factory=lambda: np.zeros(6))

    @property
    def forearm(self) -> float:
        """Effective forearm length from the elbow to the wrist centre."""
        return float(np.hypot(self.a3, self.d4))


_HALF_PI = np.pi / 2.0


def extract_arm_geometry(robot: RobotModel) -> ArmGeometry | None:
    """Read the closed-form parameters off the DH table, or ``None`` if it does not fit.

    Checked rather than assumed: a silently mismatched structure would produce
    joint angles that satisfy the solver's own equations and put the torch
    somewhere else entirely.
    """
    if robot.n_joints != 6 or robot.wrist != "spherical":
        return None
    joints = robot.joints
    if any(j.kind != "revolute" for j in joints):
        return None

    expected_alpha = (0.0, -_HALF_PI, 0.0, -_HALF_PI, _HALF_PI, -_HALF_PI)
    for joint, alpha in zip(joints, expected_alpha, strict=True):
        if not np.isclose(joint.alpha, alpha, atol=1e-6):
            logger.debug(
                "%s: alpha %s does not match the spherical-wrist pattern",
                robot.name,
                [round(j.alpha, 4) for j in joints],
            )
            return None
    # A spherical wrist has axes 4, 5 and 6 intersecting: no link length or
    # offset between 5 and 6.
    if not (
        np.isclose(joints[4].a, 0.0, atol=1e-6)
        and np.isclose(joints[4].d, 0.0, atol=1e-6)
        and np.isclose(joints[5].a, 0.0, atol=1e-6)
    ):
        return None

    return ArmGeometry(
        d1=joints[0].d,
        a1=joints[1].a,
        a2=joints[2].a,
        a3=joints[3].a,
        d4=joints[3].d,
        d6=joints[5].d,
        sign=np.array([j.sign for j in joints], dtype=float),
        theta_offset=np.array([j.theta_offset for j in joints], dtype=float),
    )


# ---------------------------------------------------------------------------
# closed form
# ---------------------------------------------------------------------------


def solve_analytic(
    robot: RobotModel,
    target: np.ndarray,
    geometry: ArmGeometry | None = None,
    options: IKOptions | None = None,
) -> list[IKSolution]:
    """All closed-form solutions for a TCP pose, best branch first.

    *target* is the desired **TCP** pose in cell coordinates; the robot's base
    and tool frames are removed internally, so callers never have to.
    """
    # The closed form has no tolerances to tune -- it is exact or it does not
    # apply -- so *options* is accepted only to keep the signature uniform
    # with solve_numeric and solve.
    del options
    geom = geometry or extract_arm_geometry(robot)
    if geom is None:
        return []

    # Strip the mounting and the tool: solve for the flange.
    flange = tf.compose(tf.inverse(robot.base), target, tf.inverse(robot.tool))
    R = tf.rotation_of(flange)
    p = tf.translation_of(flange)

    # Wrist centre: back off along the flange's own Z by the wrist-to-flange
    # offset. This is what decouples position from orientation.
    wrist = p - geom.d6 * R[:, 2]

    solutions: list[IKSolution] = []

    for shoulder in ("front", "back"):
        theta1 = _solve_theta1(wrist, shoulder)
        if theta1 is None:
            continue

        for elbow in ("up", "down"):
            pair = _solve_theta23(wrist, theta1, geom, elbow)
            if pair is None:
                continue
            theta2, theta3 = pair

            # Orientation of frame 3, to peel off the wrist.
            R03 = _rotation_0_to_3(theta1, theta2, theta3, geom)
            R36 = R03.T @ R

            for flip in (False, True):
                wrist_angles = _solve_wrist(R36, flip)
                if wrist_angles is None:
                    continue
                theta4, theta5, theta6 = wrist_angles

                raw = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
                q = _to_joint_values(raw, geom)
                q = _wrap_into_limits(robot, q)

                solution = IKSolution(
                    joints=q,
                    branch=f"{shoulder}/{elbow}/{'flip' if flip else 'noflip'}",
                    within_limits=robot.within_limits(q),
                    method="analytic",
                )
                _score(robot, solution, target)
                solutions.append(solution)

    solutions.sort(key=lambda s: (not s.ok, s.position_error_mm, -s.limit_margin))
    return solutions


def _solve_theta1(wrist: np.ndarray, shoulder: str) -> float | None:
    """Base rotation. Two branches: facing the target, or reaching backwards."""
    x, y = wrist[0], wrist[1]
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        # The wrist centre is on the base axis: theta1 is undetermined. Pick 0
        # rather than returning a random value, and let the caller's branch
        # selection keep continuity.
        return 0.0 if shoulder == "front" else np.pi
    base = float(np.arctan2(y, x))
    return base if shoulder == "front" else _wrap(base + np.pi)


def _solve_theta23(
    wrist: np.ndarray, theta1: float, geom: ArmGeometry, elbow: str
) -> tuple[float, float] | None:
    """Shoulder and elbow angles from the planar two-link problem.

    Working in the arm plane with ``r`` radial from axis 1 and ``s`` upward,
    both measured from the shoulder, this chain's geometry is:

    .. code-block:: text

        upper arm direction  =  -theta2
        forearm direction    = -(theta2 + theta3) - 90 deg
        bend at the elbow    =  forearm - upper arm  =  -theta3 - 90 deg

    (Derived from the DH table rather than assumed: axis 2 turns the upper arm
    the opposite way to the DH angle, and the forearm leaves the elbow along
    frame 3's **y**, which is where the extra 90 deg comes from. Guessing it
    gives a solver that is self-consistent and still puts the torch in the
    wrong place.)

    With the bend ``psi`` the law of cosines reads
    ``D^2 = L1^2 + L2^2 + 2*L1*L2*cos(psi)`` -- a plus sign, because ``psi`` is
    measured as the forearm's deviation from *continuing straight on*, not as
    the interior angle of the triangle.

    A non-zero forearm offset ``a3`` tilts the forearm off frame 3's y axis by
    ``atan2(a3, d4)``, which is folded into ``theta3``.
    """
    # Project the wrist centre into the arm plane. Reaching backwards puts it
    # at negative radius, which is what distinguishes the two shoulder branches.
    direction = np.array([np.cos(theta1), np.sin(theta1)])
    radial = float(direction @ wrist[:2])
    r = radial - geom.a1
    s = float(wrist[2] - geom.d1)

    upper = geom.a2
    forearm = geom.forearm
    if upper <= 0.0 or forearm <= 0.0:
        return None

    distance = float(np.hypot(r, s))
    cos_psi = (distance * distance - upper * upper - forearm * forearm) / (2.0 * upper * forearm)
    if not -1.0 - 1e-9 <= cos_psi <= 1.0 + 1e-9:
        return None  # out of reach, or inside the shoulder's dead zone
    psi = float(np.arccos(float(np.clip(cos_psi, -1.0, 1.0))))
    if elbow == "down":
        psi = -psi

    # Offset of the forearm from frame 3's y axis. Measured, not assumed: the
    # forearm direction is exactly ``-(theta2 + theta3) - 90 deg + phi`` with
    # ``phi = atan2(a3, d4)``, constant over the whole workspace, so it enters
    # theta3 with a plus sign.
    phi = float(np.arctan2(geom.a3, geom.d4))

    theta3 = _wrap(-_HALF_PI + phi - psi)

    beta = float(np.arctan2(s, r))
    gamma = float(np.arctan2(forearm * np.sin(psi), upper + forearm * np.cos(psi)))
    theta2 = _wrap(gamma - beta)
    return theta2, theta3


def _rotation_0_to_3(theta1: float, theta2: float, theta3: float, geom: ArmGeometry) -> np.ndarray:
    """Orientation of frame 3, built from the same DH convention as the FK."""

    def link(alpha: float, theta: float) -> np.ndarray:
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array(
            [[ct, -st, 0.0], [st * ca, ct * ca, -sa], [st * sa, ct * sa, ca]], dtype=float
        )

    return link(0.0, theta1) @ link(-_HALF_PI, theta2) @ link(0.0, theta3)


def _solve_wrist(R36: np.ndarray, flip: bool) -> tuple[float, float, float] | None:
    """Wrist angles from the residual rotation ``R36``.

    Derived from this chain rather than taken from a generic Euler
    decomposition, because which element carries ``cos(theta5)`` depends on the
    twist sequence. With ``A(alpha, theta) = Rx(alpha) Rz(theta)`` and the
    wrist's twists ``-90, +90, -90``:

    .. code-block:: text

        A4 A5 = [[ c4 c5, -c4 s5,  s4],
                 [    s5,     c5,   0],
                 [-s4 c5,  s4 s5,  c4]]

        R36 = A4 A5 A6   =>   R36[1,2] =  cos(theta5)
                              R36[0,2] = -cos(theta4) sin(theta5)
                              R36[2,2] =  sin(theta4) sin(theta5)
                              R36[1,0] =  sin(theta5) cos(theta6)
                              R36[1,1] = -sin(theta5) sin(theta6)

    So ``cos(theta5)`` sits at ``[1, 2]``, **not** at ``[2, 2]`` where a
    textbook Z-Y-Z decomposition would put it. Using the wrong element gives a
    wrist that satisfies its own algebra and leaves the torch pointing
    somewhere else -- it was found here by round-tripping against the forward
    kinematics, which is why that test exists.

    *flip* selects the negative ``sin(theta5)`` branch: the same tool
    orientation reached with the wrist turned over.
    """
    cos5 = float(np.clip(R36[1, 2], -1.0, 1.0))
    sin5 = float(np.hypot(R36[0, 2], R36[2, 2]))
    if flip:
        sin5 = -sin5

    if abs(sin5) < 1e-7:
        # Wrist singularity: axes 4 and 6 are collinear, so only their sum (or
        # difference) is determined. theta4 is pinned at 0 and the whole
        # rotation put on theta6; the path planner then keeps theta4 continuous
        # from the previous station instead of letting it jump.
        theta5 = 0.0 if cos5 > 0.0 else np.pi
        theta4 = 0.0
        if cos5 > 0.0:
            theta6 = float(np.arctan2(-R36[0, 1], R36[0, 0]))
        else:
            theta6 = float(np.arctan2(R36[0, 1], -R36[0, 0]))
        return theta4, theta5, theta6

    theta5 = float(np.arctan2(sin5, cos5))
    theta4 = float(np.arctan2(R36[2, 2] / sin5, -R36[0, 2] / sin5))
    theta6 = float(np.arctan2(-R36[1, 1] / sin5, R36[1, 0] / sin5))
    return theta4, theta5, theta6


def _to_joint_values(raw: np.ndarray, geom: ArmGeometry) -> np.ndarray:
    """Convert solver angles into commanded joint values (offsets and signs)."""
    out = np.empty(6)
    for i in range(6):
        sign = geom.sign[i] if geom.sign[i] != 0.0 else 1.0
        out[i] = _wrap((raw[i] - geom.theta_offset[i]) / sign)
    return out


def _wrap(angle: float) -> float:
    """Into ``(-pi, pi]``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _wrap_into_limits(robot: RobotModel, q: np.ndarray) -> np.ndarray:
    """Try the +-2*pi aliases of each joint and keep one inside its limits.

    An industrial wrist often travels well past +-180 deg, so the principal
    value of an arctangent can be outside the limits while an equivalent angle
    a full turn away is comfortably inside. Not trying the aliases throws away
    perfectly good solutions -- which shows up as "unreachable" seams that the
    real robot welds without complaint.
    """
    out = q.copy()
    for i, joint in enumerate(robot.joints):
        if joint.within(out[i]):
            continue
        for turns in (-2.0, 2.0, -4.0, 4.0):
            candidate = out[i] + turns * np.pi
            if joint.within(candidate):
                out[i] = candidate
                break
    return out


def _score(robot: RobotModel, solution: IKSolution, target: np.ndarray) -> None:
    """Fill in the achieved accuracy and posture quality of a solution."""
    from .robot import manipulability as manip

    achieved = fk(robot, solution.joints)
    lin, ang = tf.pose_distance(target, achieved)
    solution.position_error_mm = lin
    solution.orientation_error_deg = float(np.rad2deg(ang))
    solution.manipulability = manip(robot, solution.joints)
    solution.limit_margin = robot.limit_margin(solution.joints)


# ---------------------------------------------------------------------------
# numeric fallback
# ---------------------------------------------------------------------------


def solve_numeric(
    robot: RobotModel,
    target: np.ndarray,
    seed: np.ndarray | None = None,
    options: IKOptions | None = None,
) -> IKSolution | None:
    """Damped least-squares IK with joint limits and random restarts.

    Works for any chain, including external axes and non-spherical wrists. The
    damping keeps it stable through singularities, where an undamped
    pseudo-inverse asks for unbounded joint rates.
    """
    opts = options or IKOptions()
    rng = np.random.default_rng(opts.seed)
    n = robot.n_joints
    lo, hi = robot.limits_lo, robot.limits_hi

    starts: list[np.ndarray] = []
    if seed is not None:
        starts.append(np.asarray(seed, dtype=float).reshape(n).copy())
    starts.append(robot.home_joints())
    for _ in range(max(0, opts.random_restarts)):
        starts.append(rng.uniform(lo, hi))

    best: IKSolution | None = None

    for start in starts:
        q = robot.clamp(start.copy()) if opts.respect_limits else start.copy()
        iterations = 0
        for iterations in range(1, opts.max_iterations + 1):  # noqa: B007 - reported below
            achieved = fk(robot, q)
            error = _pose_error(target, achieved)
            lin = float(np.linalg.norm(error[:3]))
            ang = float(np.linalg.norm(error[3:]))
            if lin <= opts.position_tol_mm and ang <= opts.orientation_tol_rad:
                break

            J = jacobian(robot, q)
            # Scale the rotational rows so position and orientation trade off
            # on comparable terms, independent of the robot's size.
            weight = np.ones(6)
            weight[3:] = opts.orientation_weight_mm_per_rad
            Jw = J * weight[:, None]
            ew = error * weight

            # Levenberg-Marquardt: (J'J + lambda^2 I)^-1 J' e
            lhs = Jw.T @ Jw + (opts.damping**2) * np.eye(n)
            step = np.linalg.solve(lhs, Jw.T @ ew)
            step = np.clip(step, -opts.step_clip_rad, opts.step_clip_rad)
            q = q + step
            if opts.respect_limits:
                q = robot.clamp(q)

        solution = IKSolution(
            joints=q,
            branch="numeric",
            within_limits=robot.within_limits(q),
            iterations=iterations,
            method="numeric",
        )
        _score(robot, solution, target)
        if best is None or (
            solution.position_error_mm + solution.orientation_error_deg
            < best.position_error_mm + best.orientation_error_deg
        ):
            best = solution
        if solution.ok:
            break

    return best


def _pose_error(target: np.ndarray, achieved: np.ndarray) -> np.ndarray:
    """``[dx dy dz rx ry rz]`` taking *achieved* to *target*."""
    error = np.zeros(6)
    error[:3] = tf.translation_of(target) - tf.translation_of(achieved)
    R_err = tf.rotation_of(target) @ tf.rotation_of(achieved).T
    axis, angle = tf.axis_angle_of(R_err)
    error[3:] = axis * angle
    return error


# ---------------------------------------------------------------------------
# the public entry point
# ---------------------------------------------------------------------------


def solve(
    robot: RobotModel,
    target: np.ndarray,
    seed: np.ndarray | None = None,
    options: IKOptions | None = None,
    prefer_branch: str | None = None,
) -> IKSolution | None:
    """Best solution for a TCP pose: closed form if possible, numeric if not.

    When *seed* is given, the solution closest to it in joint space wins among
    the valid ones. That is what keeps a welding path on one configuration: the
    seed is the previous station's answer, so the arm never reconfigures mid-bead.
    """
    opts = options or IKOptions()
    candidates = solve_analytic(robot, target, options=opts)
    valid = [s for s in candidates if s.ok]

    if valid:
        if prefer_branch is not None:
            on_branch = [s for s in valid if s.branch == prefer_branch]
            if on_branch:
                valid = on_branch
        if seed is not None:
            return nearest_solution(valid, seed)
        return max(valid, key=lambda s: (s.manipulability, s.limit_margin))

    numeric = solve_numeric(robot, target, seed, opts)
    if numeric is not None and numeric.ok:
        return numeric

    # Nothing reached the tolerance. Return the closest attempt so the caller
    # can report *how far* out of reach the target was, which is far more
    # useful to an operator than a bare failure.
    attempts = [s for s in [*candidates, numeric] if s is not None]
    if not attempts:
        return None
    return min(attempts, key=lambda s: s.position_error_mm + s.orientation_error_deg)


def nearest_solution(solutions: list[IKSolution], seed: np.ndarray) -> IKSolution:
    """Solution closest to *seed* in joint space, weighted towards the big axes.

    Axes 1-3 carry the arm's mass, so a few degrees there is a much larger
    motion than the same change at the wrist. Weighting keeps the chosen
    solution the one the robot actually gets to fastest.
    """
    seed = np.asarray(seed, dtype=float).reshape(-1)
    weights = np.array([3.0, 3.0, 2.0, 1.0, 1.0, 0.5])[: len(seed)]
    if len(weights) < len(seed):
        weights = np.ones(len(seed))

    def distance(solution: IKSolution) -> float:
        delta = solution.joints[: len(seed)] - seed
        return float(np.sum(weights * np.abs(delta)))

    return min(solutions, key=distance)
