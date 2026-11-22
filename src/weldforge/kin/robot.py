"""Robot model, forward kinematics and Jacobian.

Kinematics are expressed in **modified Denavit-Hartenberg** (Craig) form,
which is how every industrial robot maker publishes its geometry, so a cell
can be set up from the data sheet rather than by reverse-engineering a URDF.
A URDF importer is provided for the ROS side (:mod:`weldforge.kin.urdf`), and
both end up in the same :class:`RobotModel`.

Craig's convention, link *i* from frame *i-1*:

.. code-block:: text

    T = Rx(alpha[i-1]) * Tx(a[i-1]) * Rz(theta[i]) * Tz(d[i])

Note the order: the twist and link length belong to the **previous** link.
Mixing this up with standard DH (where they belong to the current link) gives a
chain that looks plausible, matches at the home position, and is wrong
everywhere else -- so the convention is stated here and tested against the
published reach of each library robot.

Lengths in millimetres, angles in radians, per :mod:`weldforge.units`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..geom import transform as tf

logger = logging.getLogger(__name__)

__all__ = ["JointSpec", "RobotModel", "fk", "fk_all", "jacobian", "manipulability"]


@dataclass
class JointSpec:
    """One revolute or prismatic joint: geometry, limits and dynamics."""

    name: str
    # Modified DH parameters of the link *preceding* this joint.
    alpha: float = 0.0  # twist about X_{i-1}, rad
    a: float = 0.0  # length along X_{i-1}, mm
    d: float = 0.0  # offset along Z_i, mm
    theta_offset: float = 0.0  # added to the commanded angle, rad
    kind: str = "revolute"  # 'revolute' | 'prismatic'
    limit_lo: float = -np.pi
    limit_hi: float = np.pi
    max_velocity: float = 3.0  # rad/s or mm/s
    max_acceleration: float = 10.0
    # Direction convention of the real controller, if it differs from the model.
    sign: float = 1.0

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.limit_lo, self.limit_hi))

    def within(self, value: float, tol: float = 1e-9) -> bool:
        return bool(self.limit_lo - tol <= value <= self.limit_hi + tol)

    @property
    def span(self) -> float:
        return self.limit_hi - self.limit_lo

    def transform(self, value: float) -> np.ndarray:
        """Link transform for this joint at *value*."""
        if self.kind == "prismatic":
            theta = self.theta_offset
            d = self.d + self.sign * value
        else:
            theta = self.theta_offset + self.sign * value
            d = self.d
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        ct, st = np.cos(theta), np.sin(theta)
        # Craig's modified DH, written out rather than composed from four
        # matrices: this is called once per joint per IK iteration.
        return np.array(
            [
                [ct, -st, 0.0, self.a],
                [st * ca, ct * ca, -sa, -sa * d],
                [st * sa, ct * sa, ca, ca * d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


@dataclass
class RobotModel:
    """A serial manipulator plus its mounting and tool frames."""

    name: str
    joints: list[JointSpec] = field(default_factory=list)
    # Base frame: where the robot sits in the cell.
    base: np.ndarray = field(default_factory=tf.identity)
    # Tool frame: flange to torch tip (the TCP), in flange coordinates.
    tool: np.ndarray = field(default_factory=tf.identity)
    home: np.ndarray | None = None
    payload_kg: float = 0.0
    reach_mm: float = 0.0
    vendor: str = ""
    # Structure hint for the analytic solver; see weldforge.kin.ik.
    wrist: str = "spherical"
    notes: list[str] = field(default_factory=list)

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    @property
    def limits_lo(self) -> np.ndarray:
        return np.array([j.limit_lo for j in self.joints])

    @property
    def limits_hi(self) -> np.ndarray:
        return np.array([j.limit_hi for j in self.joints])

    @property
    def max_velocities(self) -> np.ndarray:
        return np.array([j.max_velocity for j in self.joints])

    def home_joints(self) -> np.ndarray:
        if self.home is not None:
            return np.asarray(self.home, dtype=float).copy()
        return np.zeros(self.n_joints)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.array([j.clamp(v) for j, v in zip(self.joints, q, strict=True)])

    def within_limits(self, q: np.ndarray, tol: float = 1e-9) -> bool:
        return all(j.within(v, tol) for j, v in zip(self.joints, q, strict=True))

    def limit_violations(self, q: np.ndarray) -> list[str]:
        out = []
        for joint, value in zip(self.joints, q, strict=True):
            if not joint.within(value):
                out.append(
                    f"{joint.name}: {np.rad2deg(value):.1f} deg outside "
                    f"[{np.rad2deg(joint.limit_lo):.0f}, {np.rad2deg(joint.limit_hi):.0f}]"
                )
        return out

    def limit_margin(self, q: np.ndarray) -> float:
        """Smallest normalised distance to any joint limit, in ``[0, 0.5]``.

        0 means a joint is at its stop; 0.5 means every joint is mid-range.
        Used to prefer postures that keep room to move -- a trajectory that
        grazes a limit will fault out on the real controller even though the
        geometry said it fitted.
        """
        if self.n_joints == 0:
            return 0.0
        margins = []
        for joint, value in zip(self.joints, q, strict=True):
            if joint.span <= 0.0:
                margins.append(0.0)
                continue
            lower = (value - joint.limit_lo) / joint.span
            upper = (joint.limit_hi - value) / joint.span
            margins.append(min(lower, upper))
        return float(max(0.0, min(margins)))

    # -- kinematics ----------------------------------------------------------

    def fk(self, q: np.ndarray, with_tool: bool = True) -> np.ndarray:
        return fk(self, q, with_tool)

    def fk_all(self, q: np.ndarray) -> list[np.ndarray]:
        return fk_all(self, q)

    def jacobian(self, q: np.ndarray, with_tool: bool = True) -> np.ndarray:
        return jacobian(self, q, with_tool)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "n_joints": self.n_joints,
            "reach_mm": self.reach_mm,
            "payload_kg": self.payload_kg,
            "wrist": self.wrist,
            "limits_deg": [
                [round(float(np.rad2deg(lo)), 1), round(float(np.rad2deg(hi)), 1)]
                for lo, hi in zip(self.limits_lo, self.limits_hi, strict=True)
            ],
            "tcp_mm": tf.translation_of(self.tool).round(2).tolist(),
        }


# ---------------------------------------------------------------------------
# forward kinematics
# ---------------------------------------------------------------------------


def fk_all(robot: RobotModel, q: np.ndarray) -> list[np.ndarray]:
    """Cumulative frames: base, then one per joint, then the tool.

    Returned in cell coordinates, so the list can be fed straight to collision
    checking and to RViz without further composition.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    if len(q) != robot.n_joints:
        raise ValueError(f"{robot.name} has {robot.n_joints} joints, got {len(q)} values")
    frames = [robot.base.copy()]
    current = robot.base.copy()
    for joint, value in zip(robot.joints, q, strict=True):
        current = current @ joint.transform(value)
        frames.append(current.copy())
    frames.append(current @ robot.tool)
    return frames


def fk(robot: RobotModel, q: np.ndarray, with_tool: bool = True) -> np.ndarray:
    """Pose of the TCP (or the flange) in cell coordinates."""
    frames = fk_all(robot, q)
    return frames[-1] if with_tool else frames[-2]


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


def jacobian(robot: RobotModel, q: np.ndarray, with_tool: bool = True) -> np.ndarray:
    """Geometric Jacobian, ``(6, n)``: rows are ``[vx vy vz wx wy wz]``.

    Built from the joint axes and the lever arms to the tool point, which is
    both exact and cheaper than finite differences -- and finite differences
    near a singularity are exactly where they go wrong.
    """
    frames = fk_all(robot, q)
    tip = tf.translation_of(frames[-1] if with_tool else frames[-2])
    n = robot.n_joints
    J = np.zeros((6, n))

    for i, joint in enumerate(robot.joints):
        frame = frames[i + 1]  # the joint's own frame after its rotation
        axis = frame[:3, 2]  # Z of the joint frame
        origin = frame[:3, 3]
        if joint.kind == "prismatic":
            J[:3, i] = axis * joint.sign
        else:
            J[:3, i] = np.cross(axis, tip - origin) * joint.sign
            J[3:, i] = axis * joint.sign
    return J


def manipulability(robot: RobotModel, q: np.ndarray, translation_only: bool = True) -> float:
    """Yoshikawa manipulability of the posture; 0 at a singularity.

    Only the translational rows are used by default. A welding torch must
    track a line at a controlled speed; what matters is whether the arm can
    move the tip smoothly, and mixing rotational rows into the measure makes
    the number depend on the choice of length unit.
    """
    J = jacobian(robot, q)
    block = J[:3, :] if translation_only else J
    singular = np.linalg.svd(block, compute_uv=False)
    if len(singular) == 0:
        return 0.0
    # Normalised: the ratio of the smallest to the largest singular value is
    # scale-free and is what degrades as a singularity is approached.
    return float(singular[-1] / singular[0]) if singular[0] > 1e-12 else 0.0
