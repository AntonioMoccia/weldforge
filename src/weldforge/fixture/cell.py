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

__all__ = [
    "load_cell",
    "available_cells",
    "default_cell",
    "cell_robot",
    "cell_from_config",
    "read_cell_config",
    "default_cell_config",
    "save_cell_config",
    "user_cells_dir",
    "bench_model_from_file",
]


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


def user_cells_dir() -> Path:
    """Where a cell the operator built in the application is kept.

    Separate from ``data/cells`` on purpose: the shipped cells are examples
    that an update may replace, and a bench someone measured on their own shop
    floor must not be one of the things an update replaces.
    """
    import os

    override = os.environ.get("WELDFORGE_CELLS_DIR")
    base = Path(override) if override else Path.home() / ".config" / "weldforge" / "cells"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cell_path(name: str, data_dir: Path | None = None) -> Path | None:
    """Where cell *name* lives: the operator's own cells win over the shipped ones."""
    candidates = []
    if data_dir is None:
        candidates.append(user_cells_dir() / f"{name}.yaml")
    base = Path(data_dir) if data_dir else DATA_DIR
    candidates.append(base / "cells" / f"{name}.yaml")
    return next((p for p in candidates if p.exists()), None)


def read_cell_config(name: str, data_dir: Path | None = None) -> dict:
    """The cell's description, as the plain dictionary the YAML holds.

    The application edits this dictionary and builds the cell from it, so
    there is one representation of a cell rather than two: what is on disk is
    exactly what is in the project file and exactly what the editor changes.
    """
    path = _cell_path(name, data_dir)
    if path is None:
        base = Path(data_dir) if data_dir else DATA_DIR
        raise FileNotFoundError(
            f"cell {name!r} not found in {user_cells_dir()} or {base / 'cells'}. "
            f"Available: {available_cells(data_dir)}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.setdefault("name", name)
    # Remember where it came from, so a relative bench model resolves against
    # the folder holding the cell rather than the working directory.
    raw["_dir"] = str(path.parent)
    return raw


def load_cell(name: str, data_dir: Path | None = None) -> WeldCell:
    """Load cell *name* from the operator's cells, or the shipped ones."""
    return cell_from_config(read_cell_config(name, data_dir), name)


def cell_from_config(raw: dict, name: str = "cell") -> WeldCell:
    """Build a :class:`WeldCell` from the YAML-shaped dictionary *raw*."""
    raw = dict(raw or {})
    base_dir = Path(raw.get("_dir", ".")) if raw.get("_dir") else None

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
        model = str(item.get("model", "") or "")
        if model:
            # An obstacle imported as a 3-D model -- a welding jig, a
            # positioner housing, a fence -- positioned by its footprint
            # centre like the bench is, so dragging it moves the model.
            model_path = Path(model)
            if not model_path.is_absolute() and base_dir is not None:
                model_path = base_dir / model_path
            try:
                vertices, faces = bench_model_from_file(model_path)
            except Exception as exc:
                logger.error("obstacle model %s could not be read: %s", model_path, exc)
                vertices = faces = None
            else:
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                target = np.asarray(centre, dtype=float)
                shift = np.array(
                    [
                        target[0] - (lo[0] + hi[0]) / 2.0,
                        target[1] - (lo[1] + hi[1]) / 2.0,
                        # `centre` z is where the model's own floor sits, so a
                        # jig imported at z=0 stands on the floor.
                        target[2] - lo[2],
                    ]
                )
                vertices = vertices + shift
        elif size is not None:
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

    # A bench imported as a 3-D model. It becomes an obstacle belonging to the
    # fixture, which is all that collision checking and the 3-D view need: both
    # already walk ``all_obstacles()``. The box dimensions stay authoritative
    # for the support plane, because placement, the collision table and the
    # URDF export are written against them.
    fixture_obstacles: list[Obstacle] = []
    model_source = str(fixture_raw.get("model", "") or "")
    if model_source:
        model_path = Path(model_source)
        if not model_path.is_absolute() and base_dir is not None:
            model_path = base_dir / model_path
        try:
            vertices, faces = bench_model_from_file(model_path)
        except Exception as exc:
            # A missing bench model must not cost the operator the rest of
            # their cell: the numbers still describe a usable bench.
            logger.error("bench model %s could not be read: %s", model_path, exc)
            model_source = ""
        else:
            # The three bench numbers have to move the model, not just the box
            # that stands in for it: an operator who nudges the bench 100 mm
            # further from the robot expects the bench to move. So the model is
            # placed by them -- its footprint centred on ``table_origin``, its
            # top on ``table_z_mm``. Right after an import the numbers are read
            # off the model itself, so nothing jumps.
            lo = vertices.min(axis=0)
            hi = vertices.max(axis=0)
            origin = np.asarray(fixture_raw.get("table_origin", [0.0, 0.0, 0.0]), dtype=float)
            offset = np.array(
                [
                    float(origin[0]) - (lo[0] + hi[0]) / 2.0,
                    float(origin[1]) - (lo[1] + hi[1]) / 2.0,
                    float(fixture_raw.get("table_z_mm", hi[2])) - hi[2],
                ]
            )
            fixture_obstacles.append(
                Obstacle(
                    id="bench_model",
                    name=str(fixture_raw.get("name", "bench")),
                    kind="bench",
                    vertices=vertices,
                    faces=faces,
                    pose=tf.pose(np.eye(3), offset),
                )
            )

    fixture = Fixture(
        name=str(fixture_raw.get("name", "bench")),
        kind=str(fixture_raw.get("kind", "bench")),
        table_z_mm=float(fixture_raw.get("table_z_mm", 0.0)),
        table_size_mm=tuple(fixture_raw.get("table_size_mm", [1500.0, 1000.0])),
        table_origin=np.asarray(fixture_raw.get("table_origin", [0.0, 0.0, 0.0]), dtype=float),
        clamps=clamps,
        obstacles=fixture_obstacles,
        datums=datums,
        load_capacity_kg=float(fixture_raw.get("load_capacity_kg", 500.0)),
        model_source=model_source,
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


#: Bench meshes already read, keyed by file identity. The cell is rebuilt on
#: every nudge of a bench dimension in the editor, and re-reading a STEP of a
#: real bench each time would make the spin boxes unusable. Keyed by size and
#: modification time as well as path, so a model edited in CAD and re-saved is
#: picked up rather than remembered wrongly.
_MESH_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
_MESH_CACHE_LIMIT = 8


def bench_model_from_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of a bench model, in millimetres, in cell coordinates.

    Any format the CAD importer reads will do -- a STEP of the real bench, or
    an STL someone scanned. Only the triangles are kept: a bench needs no
    exact geometry, because nothing is welded to it.
    """
    from ..cad import load_part

    source = Path(path)
    key: tuple[str, int, int] | None = None
    try:
        stat = source.stat()
        key = (str(source.resolve()), stat.st_size, int(stat.st_mtime_ns))
    except OSError:
        key = None
    if key is not None and key in _MESH_CACHE:
        return _MESH_CACHE[key]

    part = load_part(path)
    if part.mesh_vertices is None or part.mesh_faces is None or len(part.mesh_faces) == 0:
        raise ValueError(
            f"{Path(path).name} produced no surface to show. A bench must be a solid "
            "or a mesh, not a wireframe or a drawing."
        )
    mesh = (
        np.asarray(part.mesh_vertices, dtype=float),
        np.asarray(part.mesh_faces, dtype=np.int64),
    )
    if key is not None:
        if len(_MESH_CACHE) >= _MESH_CACHE_LIMIT:
            _MESH_CACHE.pop(next(iter(_MESH_CACHE)))
        _MESH_CACHE[key] = mesh
    return mesh


def default_cell_config() -> dict:
    """A plain bench, as the dictionary the editor and the YAML both use."""
    return {
        "name": "My cell",
        "robot": "kuka_kr16_2",
        "robot_base": {"xyz": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]},
        "torch": {"length_mm": 350.0, "angle_deg": 22.0, "stickout_mm": 15.0},
        "fixture": {
            "name": "Welding bench",
            "kind": "bench",
            "table_z_mm": 750.0,
            "table_size_mm": [1000.0, 800.0],
            "table_origin": [1000.0, 0.0, 0.0],
            "load_capacity_kg": 300.0,
            "clamps": [],
        },
        "obstacles": [],
    }


def save_cell_config(raw: dict, name: str, directory: Path | None = None) -> Path:
    """Write cell *name* to ``<cells dir>/<name>.yaml`` and return the path.

    A bench model named by a path outside the cells folder is copied in beside
    the YAML, so the saved cell is one self-contained thing that can be copied
    to the machine PC -- the same reason a project embeds its CAD.
    """
    folder = Path(directory) if directory else user_cells_dir()
    folder.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in (raw or {}).items() if not k.startswith("_")}
    payload["name"] = payload.get("name") or name

    fixture_raw = dict(payload.get("fixture", {}) or {})
    model = str(fixture_raw.get("model", "") or "")
    if model:
        source = Path(model)
        if not source.is_absolute() and raw.get("_dir"):
            source = Path(raw["_dir"]) / source
        if source.exists():
            target = folder / f"{name}{source.suffix.lower()}"
            if source.resolve() != target.resolve():
                target.write_bytes(source.read_bytes())
            fixture_raw["model"] = target.name
        else:
            logger.error("bench model %s is gone; saving the cell without it", source)
            fixture_raw.pop("model", None)
    payload["fixture"] = fixture_raw

    # Obstacle models travel with the cell for the same reason the bench does.
    obstacles = []
    for entry in payload.get("obstacles", []) or []:
        entry = dict(entry)
        model = str(entry.get("model", "") or "")
        if model:
            source = Path(model)
            if not source.is_absolute() and raw.get("_dir"):
                source = Path(raw["_dir"]) / source
            if source.exists():
                target = folder / f"{name}_{entry.get('id', 'obstacle')}{source.suffix.lower()}"
                if source.resolve() != target.resolve():
                    target.write_bytes(source.read_bytes())
                entry["model"] = target.name
            else:
                logger.error("obstacle model %s is gone; saving without it", source)
                entry.pop("model", None)
        obstacles.append(entry)
    payload["obstacles"] = obstacles

    path = folder / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("saved cell %s to %s", name, path)
    return path


def available_cells(data_dir: Path | None = None) -> list[str]:
    """Every cell that can be loaded by name: the operator's own and the shipped ones."""
    folders = []
    if data_dir is None:
        folders.append(user_cells_dir())
    base = Path(data_dir) if data_dir else DATA_DIR
    folders.append(base / "cells")
    names = {p.stem for folder in folders if folder.is_dir() for p in folder.glob("*.yaml")}
    return sorted(names)


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
