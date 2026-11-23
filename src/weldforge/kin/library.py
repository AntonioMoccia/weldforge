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
from .robot import JointSpec, LinkVisual, RobotModel
from .wps_paths import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["load_robot", "available_robots", "torch_frame", "clear_cache"]

_cache: dict[tuple[str, str], RobotModel] = {}


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


def _link_visuals(raw: dict | None, folder: Path, n_joints: int) -> list[LinkVisual]:
    """Read the optional ``visual:`` block of a robot file.

    .. code-block:: yaml

        visual:
          meshes_dir: kr16_2          # under data/robots/meshes/, or absolute
          links:
            - { mesh: base_link.stl }                       # the base casting
            - { mesh: link_1.stl, rpy_deg: [0, 0, 0] }      # moves with A1
            - { mesh: link_2.stl, xyz: [0, 0, 0] }          # moves with A2
            # ... one entry per joint

    A link with no ``mesh`` is a hole in the arm, not an error: a robot whose
    vendor publishes five of six links is still better drawn than as a stick.
    Anything that cannot be read is logged and skipped for the same reason --
    the kinematics do not depend on any of it.
    """
    if not raw:
        return []

    base_dir = Path(str(raw.get("meshes_dir", ""))) if raw.get("meshes_dir") else None
    if base_dir is not None and not base_dir.is_absolute():
        base_dir = folder / "meshes" / base_dir

    entries = list(raw.get("links", []) or [])
    if len(entries) > n_joints + 1:
        logger.warning(
            "robot visual lists %d links for %d joints; the extra ones are ignored",
            len(entries),
            n_joints,
        )
        entries = entries[: n_joints + 1]

    visuals: list[LinkVisual] = []
    for index, entry in enumerate(entries):
        item = entry if isinstance(entry, dict) else {"mesh": entry}
        mesh_name = str(item.get("mesh", "") or "")
        if not mesh_name:
            visuals.append(LinkVisual())
            continue
        path = Path(mesh_name)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        xyz = np.asarray(item.get("xyz", [0.0, 0.0, 0.0]), dtype=float)
        rpy = np.deg2rad(np.asarray(item.get("rpy_deg", [0.0, 0.0, 0.0]), dtype=float))
        offset = tf.pose(tf.rpy_to_matrix(*rpy), xyz)
        try:
            vertices, faces = _read_link_mesh(path)
        except Exception as exc:
            logger.error("link %d: %s could not be read (%s)", index, path, exc)
            visuals.append(LinkVisual(source=str(path), offset=offset))
            continue
        visuals.append(
            LinkVisual(source=str(path), vertices=vertices, faces=faces, offset=offset)
        )
    return visuals


def _read_link_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of one link. Any format the CAD importer reads will do."""
    from ..cad import load_part

    part = load_part(path)
    if part.mesh_vertices is None or part.mesh_faces is None or len(part.mesh_faces) == 0:
        raise ValueError(f"{path.name} has no surface to draw")
    return (
        np.asarray(part.mesh_vertices, dtype=float),
        np.asarray(part.mesh_faces, dtype=np.int64),
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
    folder = (Path(data_dir) if data_dir else DATA_DIR) / "robots"
    # The folder belongs in the key. Keyed by name alone, a second data
    # directory holding a different robot of the same name silently returned
    # the first one -- which is how a test fixture, or a site with its own
    # robot files, gets geometry it did not ask for.
    key = (name, str(folder))
    if key in _cache and base is None and tool is None:
        return _cache[key]

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
        link_visuals=_link_visuals(raw.get("visual"), folder, len(joints)),
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
