"""Building a viewable scene from a planned job.

Why this exists. The ROS 2 workspace has Gazebo and RViz, which are the right
tools in a cell -- and useless on an engineering workstation that has neither.
Yet looking at the plan is how most mistakes get caught: a torch pointing the
wrong way, a seam detected where there is no joint, an arm reaching through
the bench. Those are obvious in a picture and invisible in a table.

So the scene is assembled here into plain geometry -- triangle meshes and
polylines -- and :mod:`weldforge.viz.html` writes it into one self-contained
HTML file. No ROS, no Qt, no CDN, no install: a file that opens in a browser.

Everything is in millimetres in cell coordinates, as the rest of the core is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..geom import transform as tf
from ..kin import RobotModel, fk_all
from ..model import JointType, WeldCell, WeldJob

logger = logging.getLogger(__name__)

__all__ = ["SceneMesh", "SceneLines", "SceneFrame", "Scene", "scene_from_job"]


# Joint-type colours, the same as the FreeCAD overlay and the RViz markers, so
# a seam looks the same wherever it is shown.
JOINT_COLOURS = {
    JointType.FILLET: "#33b4ff",
    JointType.TEE: "#33b4ff",
    JointType.LAP: "#f2bf1a",
    JointType.BUTT: "#66f266",
    JointType.CORNER: "#f273d9",
    JointType.EDGE: "#b3b3b3",
    JointType.PLUG: "#ff8c33",
    JointType.UNKNOWN: "#8c8c8c",
}


@dataclass
class SceneMesh:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    colour: str = "#9aa4b2"
    opacity: float = 1.0
    group: str = "part"


@dataclass
class SceneLines:
    """A polyline strip, or a set of disconnected segments when *segments*."""

    name: str
    points: np.ndarray
    colour: str = "#ffffff"
    width: float = 2.0
    group: str = "seams"
    segments: bool = False
    label: str = ""


@dataclass
class SceneFrame:
    """One animation step: where the arm and the torch are, and is the arc on."""

    arm: np.ndarray  # (N, 3) polyline through the joint origins
    torch: np.ndarray  # (2, 3) the wire, from the nozzle to the tip
    weld_on: bool = False
    seam_id: str = ""
    pass_index: int = 0
    time_s: float = 0.0


@dataclass
class Scene:
    name: str = ""
    meshes: list[SceneMesh] = field(default_factory=list)
    lines: list[SceneLines] = field(default_factory=list)
    frames: list[SceneFrame] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        chunks = [m.vertices for m in self.meshes if len(m.vertices)]
        chunks += [line.points for line in self.lines if len(line.points)]
        chunks += [f.arm for f in self.frames if len(f.arm)]
        if not chunks:
            return np.zeros(3), np.ones(3) * 1000.0
        stacked = np.vstack(chunks)
        return stacked.min(axis=0), stacked.max(axis=0)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _box(size: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size, dtype=float) / 2.0
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    vertices = signs * half + np.asarray(centre, dtype=float)
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _cylinder(
    base: np.ndarray, axis: np.ndarray, radius: float, height: float, sides: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    direction = tf.unit(axis)
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    rotation = tf.rotation_between(np.array([0.0, 0.0, 1.0]), direction)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circle = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(sides)])
    bottom = circle @ rotation.T + base
    top = (circle + np.array([0.0, 0.0, height])) @ rotation.T + base
    vertices = np.vstack([bottom, top])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j])
        faces.append([i, sides + j, sides + i])
    # Caps, fanned from the first vertex of each ring.
    for i in range(1, sides - 1):
        faces.append([0, i + 1, i])
        faces.append([sides, sides + i, sides + i + 1])
    return vertices, np.array(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# assembling
# ---------------------------------------------------------------------------


def _add_cell(scene: Scene, cell: WeldCell) -> None:
    fixture = cell.fixture
    thickness = 60.0
    centre = fixture.table_origin.astype(float).copy()
    centre[2] = fixture.table_z_mm - thickness / 2.0
    vertices, faces = _box(
        np.array([fixture.table_size_mm[0], fixture.table_size_mm[1], thickness]), centre
    )
    scene.meshes.append(SceneMesh("table", vertices, faces, colour="#3b4252", group="cell"))

    for clamp in fixture.clamps:
        vertices, faces = _cylinder(
            clamp.position, clamp.approach, clamp.clearance_mm, clamp.height_mm, 14
        )
        scene.meshes.append(
            SceneMesh(
                f"clamp:{clamp.id}",
                vertices,
                faces,
                colour="#bf616a",
                opacity=0.45,
                group="cell",
            )
        )

    # Jig locators, drawn as short spikes along the direction they face.
    if fixture.datums:
        segments = []
        for position, normal in fixture.datums:
            direction = tf.unit(normal)
            if direction is None:
                continue
            segments.extend([position - direction * 30.0, position])
        if segments:
            scene.lines.append(
                SceneLines(
                    "jig locators",
                    np.array(segments),
                    colour="#ebcb8b",
                    width=3.0,
                    group="cell",
                    segments=True,
                )
            )

    for obstacle in cell.all_obstacles():
        if obstacle.has_mesh:
            scene.meshes.append(
                SceneMesh(
                    f"{obstacle.kind}:{obstacle.id}",
                    tf.apply_points(obstacle.pose, obstacle.vertices),
                    np.asarray(obstacle.faces),
                    colour="#4c566a",
                    opacity=0.5,
                    group="cell",
                )
            )


def _add_robot_base(scene: Scene, robot: RobotModel) -> None:
    """A stub column so the arm does not appear to float."""
    origin = tf.translation_of(robot.base)
    vertices, faces = _cylinder(
        origin - np.array([0.0, 0.0, origin[2]]),
        np.array([0.0, 0.0, 1.0]),
        160.0,
        max(origin[2], 100.0),
        18,
    )
    scene.meshes.append(SceneMesh("robot base", vertices, faces, colour="#5e81ac", group="robot"))


def scene_from_job(
    job: WeldJob,
    robot: RobotModel | None = None,
    max_frames: int = 400,
    wire_marker_mm: float = 30.0,
) -> Scene:
    """Build a viewable scene from a planned job.

    *max_frames* caps the animation. A 200-seam job has thousands of stations
    and a browser does not need them all to show the motion; frames are
    sampled evenly so the whole cycle is still covered.
    """
    scene = Scene(name=job.name or "weld job", summary=job.summary())

    if job.cell is not None:
        _add_cell(scene, job.cell)
    if robot is not None:
        _add_robot_base(scene, robot)

    # --- the part, at each seam's own pose ---------------------------------
    if job.part is not None and job.part.mesh_vertices is not None:
        # Drawn once, at the first seam's pose. With a positioner the part
        # moves between seams, and the animation shows the arm rather than
        # re-posing the mesh -- redrawing a 50 000-triangle weldment per frame
        # is what makes a browser viewer unusable.
        first = job.selected_seams[0].id if job.selected_seams else ""
        pose = job.part_pose_for(first) if first else np.eye(4)
        scene.meshes.append(
            SceneMesh(
                "part",
                tf.apply_points(pose, job.part.mesh_vertices),
                np.asarray(job.part.mesh_faces),
                colour="#8a94a6",
                group="part",
            )
        )
        scene.summary["part_pose_note"] = (
            "the part is drawn at the first seam's positioner setting"
            if job.positioner is not None
            else ""
        )

    # --- seams ------------------------------------------------------------
    for seam in job.selected_seams:
        placed = seam.transformed(job.part_pose_for(seam.id))
        colour = JOINT_COLOURS.get(seam.joint_type, "#8c8c8c")
        scene.lines.append(
            SceneLines(
                seam.id,
                placed.points,
                colour=colour,
                width=4.0,
                group="seams",
                label=seam.label or f"{seam.joint_type.value} {seam.length_mm:.0f}mm",
            )
        )
        # Wire direction: the display that makes a torch pointing the wrong
        # way obvious.
        if len(placed.bisector):
            step = max(1, placed.n_points // 12)
            segments = []
            for i in range(0, placed.n_points, step):
                start = placed.points[i]
                segments.extend([start - placed.bisector[i] * wire_marker_mm, start])
            scene.lines.append(
                SceneLines(
                    f"{seam.id}:wire",
                    np.array(segments),
                    colour="#ffd966",
                    width=1.5,
                    group="wire",
                    segments=True,
                )
            )

    # --- torch path -------------------------------------------------------
    weld_segments: list[np.ndarray] = []
    air_segments: list[np.ndarray] = []
    for path in job.paths:
        for a, b in zip(path.points, path.points[1:], strict=False):
            target = weld_segments if b.weld_on and a.weld_on else air_segments
            target.extend([a.position, b.position])
    if weld_segments:
        scene.lines.append(
            SceneLines(
                "arc on",
                np.array(weld_segments),
                colour="#ff6b35",
                width=3.0,
                group="path",
                segments=True,
            )
        )
    if air_segments:
        scene.lines.append(
            SceneLines(
                "air moves",
                np.array(air_segments),
                colour="#5c6370",
                width=1.0,
                group="path",
                segments=True,
            )
        )

    # --- animation --------------------------------------------------------
    if robot is not None:
        from ..sim.simulator import move_duration_s

        stations = [(path, point) for path in job.paths for point in path.points]
        solved = [(p, pt) for p, pt in stations if pt.joints is not None]
        if solved:
            step = max(1, len(solved) // max(1, max_frames))
            elapsed = 0.0
            previous = None
            for index, (path, point) in enumerate(solved):
                if previous is not None:
                    elapsed += move_duration_s(previous, point, robot)
                previous = point
                if index % step:
                    continue
                frames = fk_all(robot, point.joints)
                arm = np.array([tf.translation_of(frame) for frame in frames])
                tip = tf.translation_of(point.pose)
                back = tip - tf.rotation_of(point.pose)[:, 2] * 120.0
                scene.frames.append(
                    SceneFrame(
                        arm=arm,
                        torch=np.array([back, tip]),
                        weld_on=bool(point.weld_on),
                        seam_id=path.seam_id,
                        pass_index=int(path.pass_index),
                        time_s=elapsed,
                    )
                )

    logger.info(
        "scene_from_job: %d meshes, %d line sets, %d animation frames",
        len(scene.meshes),
        len(scene.lines),
        len(scene.frames),
    )
    return scene
