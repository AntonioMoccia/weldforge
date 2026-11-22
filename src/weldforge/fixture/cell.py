"""Build a welding cell from a YAML description.

A cell is the robot, where it is bolted down, the torch on its flange, the
bench or jig the work sits on, the clamps, the positioner and whatever else is
in the way. All of it is geometry the planner has to respect, and all of it
changes when a cell is re-tooled -- so it lives in data, not in code.

Coordinates are the **cell frame**: X and Y along the bench, Z up, origin at
the robot's mounting face unless the file says otherwise. Millimetres.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from ..geom import transform as tf
from ..kin import RobotModel, load_robot, torch_frame
from ..model import Clamp, ExternalAxis, Fixture, Obstacle, WeldCell
from ..wps.tables import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["load_cell", "available_cells", "default_cell", "cell_robot"]


def _pose_from(data: dict | None) -> np.ndarray:
    """Pose from ``{xyz: [...], rpy_deg: [...]}``."""
    if not data:
        return tf.identity()
    xyz = np.asarray(data.get("xyz", [0.0, 0.0, 0.0]), dtype=float)
    rpy = np.deg2rad(np.asarray(data.get("rpy_deg", [0.0, 0.0, 0.0]), dtype=float))
    return tf.pose(tf.rpy_to_matrix(*rpy), xyz)


def _box_mesh(size: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box as a triangle mesh, for a simple obstacle."""
    import trimesh

    mesh = trimesh.creation.box(extents=np.asarray(size, dtype=float))
    mesh.apply_translation(np.asarray(centre, dtype=float))
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


def load_cell(name: str, data_dir: Path | None = None) -> WeldCell:
    """Load ``data/cells/<name>.yaml``."""
    base = Path(data_dir) if data_dir else DATA_DIR
    path = base / "cells" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"cell {name!r} not found at {path}. Available: {sorted(available_cells(base))}"
        )
    raw = yaml.safe_load(path.read_text()) or {}

    torch = raw.get("torch", {}) or {}
    fixture_raw = raw.get("fixture", {}) or {}

    clamps = [
        Clamp(
            id=str(item.get("id", f"clamp_{i}")),
            position=np.asarray(item["position"], dtype=float),
            approach=np.asarray(item.get("approach", [0.0, 0.0, 1.0]), dtype=float),
            clearance_mm=float(item.get("clearance_mm", 40.0)),
            height_mm=float(item.get("height_mm", 60.0)),
            name=str(item.get("name", "")),
            retractable=bool(item.get("retractable", False)),
        )
        for i, item in enumerate(fixture_raw.get("clamps", []) or [])
    ]

    obstacles: list[Obstacle] = []
    for i, item in enumerate(raw.get("obstacles", []) or []):
        size = item.get("size")
        centre = item.get("centre", [0.0, 0.0, 0.0])
        vertices = faces = None
        if size is not None:
            vertices, faces = _box_mesh(size, centre)
        obstacles.append(
            Obstacle(
                id=str(item.get("id", f"obstacle_{i}")),
                name=str(item.get("name", "")),
                kind=str(item.get("kind", "generic")),
                vertices=vertices,
                faces=faces,
            )
        )

    external = [
        ExternalAxis(
            name=str(item.get("name", f"E{i + 1}")),
            kind=str(item.get("kind", "rotary")),
            axis=np.asarray(item.get("axis", [0.0, 0.0, 1.0]), dtype=float),
            origin=np.asarray(item.get("origin", [0.0, 0.0, 0.0]), dtype=float),
            limit_lo=float(np.deg2rad(item.get("limit_deg", [-180.0, 180.0])[0]))
            if item.get("kind", "rotary") == "rotary"
            else float(item.get("limit_mm", [-1000.0, 1000.0])[0]),
            limit_hi=float(np.deg2rad(item.get("limit_deg", [-180.0, 180.0])[1]))
            if item.get("kind", "rotary") == "rotary"
            else float(item.get("limit_mm", [-1000.0, 1000.0])[1]),
            max_speed=float(item.get("max_speed", 1.0)),
            coupled=bool(item.get("coupled", True)),
        )
        for i, item in enumerate(raw.get("external_axes", []) or [])
    ]

    # Datums are the locating features of a jig: the points the part is
    # pressed against so it sits in a repeatable position. Each is a point and
    # the direction its locating surface faces.
    datums: list[tuple[np.ndarray, np.ndarray]] = []
    for item in fixture_raw.get("datums", []) or []:
        datums.append(
            (
                np.asarray(item["position"], dtype=float),
                np.asarray(item.get("normal", [0.0, 0.0, 1.0]), dtype=float),
            )
        )

    fixture = Fixture(
        name=str(fixture_raw.get("name", "bench")),
        kind=str(fixture_raw.get("kind", "bench")),
        table_z_mm=float(fixture_raw.get("table_z_mm", 0.0)),
        table_size_mm=tuple(fixture_raw.get("table_size_mm", [1500.0, 1000.0])),
        table_origin=np.asarray(fixture_raw.get("table_origin", [0.0, 0.0, 0.0]), dtype=float),
        clamps=clamps,
        datums=datums,
        load_capacity_kg=float(fixture_raw.get("load_capacity_kg", 500.0)),
    )

    cell = WeldCell(
        name=str(raw.get("name", name)),
        robot_name=str(raw.get("robot", "generic_6axis")),
        robot_base=_pose_from(raw.get("robot_base")),
        torch_tcp=torch_frame(
            length_mm=float(torch.get("length_mm", 350.0)),
            angle_deg=float(torch.get("angle_deg", 22.0)),
            stickout_mm=float(torch.get("stickout_mm", 15.0)),
            offset_mm=float(torch.get("offset_mm", 0.0)),
        ),
        fixture=fixture,
        external_axes=external,
        obstacles=obstacles,
        gravity=np.asarray(raw.get("gravity", [0.0, 0.0, -1.0]), dtype=float),
    )
    logger.info(
        "loaded cell %s: robot %s, %d clamps, %d obstacles, %d external axes",
        cell.name,
        cell.robot_name,
        len(clamps),
        len(obstacles),
        len(external),
    )
    return cell


def available_cells(data_dir: Path | None = None) -> list[str]:
    base = Path(data_dir) if data_dir else DATA_DIR
    folder = base / "cells"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))


def cell_robot(cell: WeldCell) -> RobotModel:
    """The cell's robot, mounted and with its torch attached."""
    return load_robot(cell.robot_name, base=cell.robot_base, tool=cell.torch_tcp)


def default_cell() -> WeldCell:
    """A plain bench cell, for tests and a first look at a new part."""
    from ..model import Fixture as FixtureModel

    return WeldCell(
        name="default bench",
        robot_name="generic_6axis",
        robot_base=tf.pose(np.eye(3), [0.0, 0.0, 0.0]),
        torch_tcp=torch_frame(),
        fixture=FixtureModel(
            name="bench",
            kind="bench",
            table_z_mm=750.0,
            table_size_mm=(1500.0, 1000.0),
            table_origin=np.array([900.0, 0.0, 0.0]),
        ),
    )
