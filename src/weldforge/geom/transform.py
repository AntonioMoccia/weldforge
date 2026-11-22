"""SE(3) / SO(3) utilities on plain NumPy arrays.

Conventions
-----------
* A **pose** is a ``(4, 4)`` homogeneous matrix, column-vector convention:
  ``p_world = T @ p_local``.
* A **rotation** is a ``(3, 3)`` orthonormal matrix with ``det == +1``.
* A **quaternion** is ``(x, y, z, w)`` -- ROS order, *not* ``(w, x, y, z)``.
* Translations are in millimetres (see :mod:`weldforge.units`).
* Euler angles are intrinsic ``Z-Y-X`` (yaw-pitch-roll), the industrial
  robotics default, and are always passed in radians.

No dependency on ``spatialmath``/``roboticstoolbox``: this module is imported
by the ROS 2 nodes, the CAD plugins and the API, and each extra wheel is one
more thing that can fail to install on a shop-floor PC.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EPS",
    "identity",
    "pose",
    "rotation_of",
    "translation_of",
    "compose",
    "inverse",
    "apply_point",
    "apply_points",
    "apply_vector",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_axis_angle",
    "axis_angle_of",
    "rpy_to_matrix",
    "matrix_to_rpy",
    "quat_to_matrix",
    "matrix_to_quat",
    "quat_multiply",
    "quat_normalize",
    "slerp",
    "interpolate_pose",
    "orthonormalize",
    "rotation_between",
    "frame_from_z_x",
    "frame_from_x_z",
    "look_at",
    "is_pose",
    "pose_distance",
    "angle_between",
    "unit",
]

EPS = 1e-9


# ---------------------------------------------------------------------------
# construction / accessors
# ---------------------------------------------------------------------------


def identity() -> np.ndarray:
    return np.eye(4)


def pose(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    """Build a 4x4 pose from a 3x3 rotation and a 3-vector translation (mm)."""
    T = np.eye(4)
    if rotation is not None:
        R = np.asarray(rotation, dtype=float)
        if R.shape == (4, 4):
            R = R[:3, :3]
        if R.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3, got {R.shape}")
        T[:3, :3] = R
    if translation is not None:
        t = np.asarray(translation, dtype=float).reshape(3)
        T[:3, 3] = t
    return T


def rotation_of(T: np.ndarray) -> np.ndarray:
    return np.asarray(T, dtype=float)[:3, :3].copy()


def translation_of(T: np.ndarray) -> np.ndarray:
    return np.asarray(T, dtype=float)[:3, 3].copy()


def is_pose(T: np.ndarray, tol: float = 1e-6) -> bool:
    """True if *T* is a valid 4x4 rigid transform."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=tol):
        return False
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=tol):
        return False
    return abs(np.linalg.det(R) - 1.0) <= tol * 10


# ---------------------------------------------------------------------------
# algebra
# ---------------------------------------------------------------------------


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Left-to-right chain: ``compose(A, B, C) == A @ B @ C``."""
    out = np.eye(4)
    for T in transforms:
        out = out @ np.asarray(T, dtype=float)
    return out


def inverse(T: np.ndarray) -> np.ndarray:
    """Rigid-transform inverse (transpose + back-rotated translation).

    Much better conditioned than ``np.linalg.inv`` for a pose, and ~10x faster.
    """
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Transform a single point (translation applies)."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float).reshape(3)
    return T[:3, :3] @ p + T[:3, 3]


def apply_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform an ``(N, 3)`` array of points."""
    T = np.asarray(T, dtype=float)
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def apply_vector(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Transform a direction (translation does **not** apply)."""
    T = np.asarray(T, dtype=float)
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return T[:3, :3] @ v
    return v @ T[:3, :3].T


# ---------------------------------------------------------------------------
# elementary rotations
# ---------------------------------------------------------------------------


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula. *axis* need not be normalised."""
    a = unit(axis)
    if a is None:
        return np.eye(3)
    x, y, z = a
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def axis_angle_of(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Inverse of :func:`rot_axis_angle`; returns ``(axis, angle)``.

    Handles the two degenerate cases (identity and the 180 deg rotations where
    the usual ``(R - R.T)`` extraction collapses) explicitly -- a silent wrong
    axis at 180 deg is a classic source of flipped torch orientations.
    """
    R = np.asarray(R, dtype=float)[:3, :3]
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < EPS:
        return np.array([0.0, 0.0, 1.0]), 0.0

    if abs(theta - np.pi) < 1e-6:
        # Near 180 deg: axis from the dominant column of (R + I) / 2.
        A = (R + np.eye(3)) / 2.0
        col = int(np.argmax(np.diag(A)))
        axis = A[:, col]
        n = np.linalg.norm(axis)
        if n < EPS:
            return np.array([0.0, 0.0, 1.0]), float(np.pi)
        return axis / n, float(np.pi)

    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float) / (
        2.0 * np.sin(theta)
    )
    return axis, theta


# ---------------------------------------------------------------------------
# Euler / quaternion
# ---------------------------------------------------------------------------


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z-Y-X: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rpy_to_matrix`; returns ``(roll, pitch, yaw)``.

    At the pitch = +/-90 deg gimbal lock the yaw is pinned to 0 and the whole
    rotation is folded into roll, which keeps the result continuous.
    """
    R = np.asarray(R, dtype=float)[:3, :3]
    sin_pitch = -R[2, 0]
    sin_pitch = float(np.clip(sin_pitch, -1.0, 1.0))
    pitch = float(np.arcsin(sin_pitch))

    if abs(abs(sin_pitch) - 1.0) < 1e-9:  # gimbal lock
        yaw = 0.0
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
    else:
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch, yaw


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion ``(x, y, z, w)`` -> 3x3 rotation."""
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 (or 4x4) rotation -> quaternion ``(x, y, z, w)``, ``w >= 0``.

    Uses Shepperd's branch-on-largest-component method: numerically stable for
    every rotation, unlike the naive trace formula which loses precision near
    180 deg.
    """
    R = np.asarray(R, dtype=float)[:3, :3]
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=float)
    if q[3] < 0.0:  # canonical hemisphere
        q = -q
    return quat_normalize(q)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product in ``(x, y, z, w)`` order."""
    ax, ay, az, aw = np.asarray(a, dtype=float).reshape(4)
    bx, by, bz, bw = np.asarray(b, dtype=float).reshape(4)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc spherical interpolation between two quaternions."""
    a = quat_normalize(q0)
    b = quat_normalize(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # take the short way round
        b = -b
        dot = -dot
    if dot > 1.0 - 1e-9:  # almost parallel -> linear is exact enough
        return quat_normalize(a + t * (b - a))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return quat_normalize(s0 * a + s1 * b)


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, t: float) -> np.ndarray:
    """Linear in position, SLERP in orientation."""
    t = float(np.clip(t, 0.0, 1.0))
    p = (1.0 - t) * translation_of(T0) + t * translation_of(T1)
    q = slerp(matrix_to_quat(T0), matrix_to_quat(T1), t)
    return pose(quat_to_matrix(q), p)


# ---------------------------------------------------------------------------
# frame building -- the part torch orientation depends on
# ---------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray | None:
    """Normalise, or ``None`` if the vector is degenerate."""
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return None
    return v / n


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle in ``[0, pi]`` between two directions."""
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0)))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest orthonormal matrix with ``det = +1`` (polar decomposition).

    Call this after any numeric fiddling with a rotation; drift of 1e-6 is
    enough to make a robot controller reject a target.
    """
    R = np.asarray(R, dtype=float)[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    if np.linalg.det(out) < 0.0:
        U[:, -1] *= -1.0
        out = U @ Vt
    return out


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation taking direction *a* onto direction *b*."""
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return np.eye(3)
    d = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    if d > 1.0 - 1e-12:
        return np.eye(3)
    if d < -1.0 + 1e-12:
        # Antiparallel: any perpendicular axis works; pick a stable one.
        axis = unit(np.cross(ua, np.array([1.0, 0.0, 0.0])))
        if axis is None:
            axis = unit(np.cross(ua, np.array([0.0, 1.0, 0.0])))
        return rot_axis_angle(axis, np.pi)  # type: ignore[arg-type]
    return rot_axis_angle(np.cross(ua, ub), float(np.arccos(d)))


def frame_from_z_x(z_dir: np.ndarray, x_hint: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Frame whose **Z** is *z_dir* and whose X lies as close as possible to *x_hint*.

    This is the workhorse for torch poses: Z is the tool approach axis (wire
    direction), X the travel direction. *x_hint* is projected onto the plane
    normal to Z, so a hint that is not perpendicular is still usable.

    Raises
    ------
    ValueError
        If *z_dir* is degenerate, or if *x_hint* is parallel to it and no
        fallback axis can be chosen.
    """
    z = unit(z_dir)
    if z is None:
        raise ValueError("frame_from_z_x: z_dir is degenerate")

    h = np.asarray(x_hint, dtype=float).reshape(3)
    x = h - np.dot(h, z) * z
    ux = unit(x)
    if ux is None:
        # Hint parallel to Z -> fall back to the world axis least aligned with Z.
        fallback = np.eye(3)[int(np.argmin(np.abs(z)))]
        ux = unit(fallback - np.dot(fallback, z) * z)
        if ux is None:  # unreachable for a unit z, but keep the invariant explicit
            raise ValueError("frame_from_z_x: cannot build an X axis")

    y = np.cross(z, ux)
    R = np.column_stack([ux, y, z])
    return pose(orthonormalize(R), origin)


def frame_from_x_z(x_dir: np.ndarray, z_hint: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Frame whose **X** is exactly *x_dir*, with Z as close as possible to *z_hint*.

    Used when the travel direction must be respected exactly and the approach
    axis may tilt (e.g. keeping a constant travel angle along a curved seam).
    """
    x = unit(x_dir)
    if x is None:
        raise ValueError("frame_from_x_z: x_dir is degenerate")

    h = np.asarray(z_hint, dtype=float).reshape(3)
    z = h - np.dot(h, x) * x
    uz = unit(z)
    if uz is None:
        fallback = np.eye(3)[int(np.argmin(np.abs(x)))]
        uz = unit(fallback - np.dot(fallback, x) * x)
        if uz is None:
            raise ValueError("frame_from_x_z: cannot build a Z axis")

    y = np.cross(uz, x)
    R = np.column_stack([x, y, uz])
    return pose(orthonormalize(R), origin)


def look_at(origin: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Frame at *origin* with Z pointing at *target*."""
    z = np.asarray(target, dtype=float).reshape(3) - np.asarray(origin, dtype=float).reshape(3)
    hint = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=float)
    return frame_from_z_x(z, hint, origin)


def pose_distance(T0: np.ndarray, T1: np.ndarray) -> tuple[float, float]:
    """``(linear_mm, angular_rad)`` separation between two poses."""
    lin = float(np.linalg.norm(translation_of(T1) - translation_of(T0)))
    R = rotation_of(T0).T @ rotation_of(T1)
    _, ang = axis_angle_of(R)
    return lin, ang
