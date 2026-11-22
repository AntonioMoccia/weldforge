"""Robot definitions, loaded from ``data/robots/*.yaml``.

A torch is attached separately: the flange-to-tip transform is a property of
the cell's torch and its stick-out, not of the robot, so it is built by
:func:`torch_frame` and composed on load.

The library geometry is **nominal**. Link lengths come from the data sheets and
are reliable; the zero-position mapping (per-axis sign and offset) differs
between vendors and sometimes between controller software versions, so every
robot here carries a note saying it must be confirmed against the machine in
the cell. The postprocessors emit joint angles, and a sign error there is the
kind of mistake that is only discovered by a torch hitting a fixture.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from ..geom import transform as tf
from .robot import JointSpec, RobotModel
from .wps_paths import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["load_robot", "available_robots", "torch_frame", "clear_cache"]

_cache: dict[str, RobotModel] = {}


def torch_frame(
    length_mm: float = 350.0,
    angle_deg: float = 22.0,
    stickout_mm: float = 15.0,
    offset_mm: float = 0.0,
) -> np.ndarray:
    """Flange-to-wire-tip transform for a bent welding torch.

    A GMAW torch is a gooseneck: the wire leaves at an angle to the flange
    axis, which is what lets the arm reach into a corner without the sixth axis
    folding back on itself. The returned frame has

    * **Z** along the wire, pointing at the work (so the TCP's Z is the
      approach direction the path generator uses);
    * origin at the wire tip, that is *stickout* beyond the contact tip.

    Parameters are in the torch's own terms so a cell can be set up from the
    torch drawing: *length_mm* flange to contact tip along the neck,
    *angle_deg* the gooseneck bend, *offset_mm* any lateral offset.
    """
    # Rotate about the flange Y by the gooseneck angle, then run out along the
    # (now tilted) Z by the neck length plus stick-out.
    bend = tf.pose(tf.rot_y(np.deg2rad(angle_deg)))
    reach = tf.pose(np.eye(3), [offset_mm, 0.0, length_mm + stickout_mm])
    return tf.compose(bend, reach)


def _joint_from_dict(data: dict, index: int) -> JointSpec:
    limits = data.get("limit_deg", [-180.0, 180.0])
    return JointSpec(
        name=str(data.get("name", f"J{index + 1}")),
        alpha=float(np.deg2rad(data.get("alpha", 0.0))),
        a=float(data.get("a", 0.0)),
        d=float(data.get("d", 0.0)),
        theta_offset=float(np.deg2rad(data.get("theta_offset_deg", 0.0))),
        kind=str(data.get("kind", "revolute")),
        limit_lo=float(np.deg2rad(limits[0])),
        limit_hi=float(np.deg2rad(limits[1])),
        max_velocity=float(np.deg2rad(data.get("max_speed_deg_s", 180.0))),
        max_acceleration=float(data.get("max_acceleration", 10.0)),
        sign=float(data.get("sign", 1.0)),
    )


def load_robot(
    name: str,
    base: np.ndarray | None = None,
    tool: np.ndarray | None = None,
    data_dir: Path | None = None,
) -> RobotModel:
    """Load ``data/robots/<name>.yaml``.

    *base* places the robot in the cell; *tool* is the flange-to-tip transform
    (see :func:`torch_frame`). Both default to identity, which puts the TCP at
    the flange -- useful for checking kinematics, useless for welding, so the
    cell builder always sets them.
    """
    key = name
    if key in _cache and base is None and tool is None:
        return _cache[key]

    folder = (Path(data_dir) if data_dir else DATA_DIR) / "robots"
    path = folder / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"robot {name!r} not found at {path}. "
            f"Available: {sorted(available_robots(folder.parent))}"
        )
    raw = yaml.safe_load(path.read_text()) or {}

    joints = [_joint_from_dict(item, i) for i, item in enumerate(raw.get("joints", []))]
    home = raw.get("home_deg")

    robot = RobotModel(
        name=str(raw.get("name", name)),
        joints=joints,
        base=np.eye(4) if base is None else np.asarray(base, dtype=float),
        tool=np.eye(4) if tool is None else np.asarray(tool, dtype=float),
        home=np.deg2rad(np.asarray(home, dtype=float)) if home else None,
        payload_kg=float(raw.get("payload_kg", 0.0)),
        reach_mm=float(raw.get("reach_mm", 0.0)),
        vendor=str(raw.get("vendor", "")),
        wrist=str(raw.get("wrist", "spherical")),
        notes=[str(n) for n in raw.get("notes", [])],
    )
    if base is None and tool is None:
        _cache[key] = robot
    logger.info(
        "loaded robot %s (%d joints, reach %.0f mm)", robot.name, robot.n_joints, robot.reach_mm
    )
    return robot


def available_robots(data_dir: Path | None = None) -> list[str]:
    folder = (Path(data_dir) if data_dir else DATA_DIR) / "robots"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))


def clear_cache() -> None:
    _cache.clear()
