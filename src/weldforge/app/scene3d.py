"""The 3-D viewport: building it, and picking in it.

Kept free of Qt on purpose. A ``vtkRenderer`` works offscreen with real
rendering and real picking, so everything in this module -- the scene
construction, the actor-to-domain-object mapping, the pick resolution -- is
testable without a display. Only the thin widget that embeds the renderer in a
window needs one, and that lives in :mod:`weldforge.app.main_window`.

Picking is the capability the application is built around. "Trace the weld
lines" means an operator clicks edges on the model, so a click has to come back
as *which CAD edge*, not as a point in space. That mapping is maintained here:

* **edges** -- all of the part's edges in one polydata, one VTK cell per edge,
  with a cell-index to edge-id list. One actor rather than one per edge,
  because a 900-face weldment has two thousand edges and two thousand actors
  makes a renderer crawl.
* **faces** -- the part tessellated face by face, with a per-triangle face-id
  array. The combined mesh loses which CAD face each triangle came from, which
  is exactly the information a click needs.
* **seams** -- one actor each, so a click resolves to a seam id directly.

Coordinates. Seam and edge geometry stays in **part** coordinates throughout,
as it does everywhere else in the core; the placement is applied as the
actors' transform. Moving the part is then a matrix update rather than a
re-tessellation, which is what makes dragging it feel immediate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geom import transform as tf
from ..model import JointType, Part, WeldCell
from .project import Project

logger = logging.getLogger(__name__)

__all__ = [
    "LAYERS",
    "JOINT_COLOURS",
    "PickResult",
    "Scene3D",
]

#: Layer name, label, whether it is pickable, whether it starts visible.
LAYERS: list[tuple[str, str, bool, bool]] = [
    ("part", "Part", True, True),
    ("edges", "Part edges", True, False),
    ("cell", "Bench and jig", True, True),
    ("clamps", "Clamps", False, True),
    ("datums", "Jig locators", False, True),
    ("seams", "Weld seams", True, True),
    ("wire", "Wire direction", False, True),
    ("path", "Torch path", False, True),
    ("robot", "Robot", False, True),
]

JOINT_COLOURS: dict[JointType, tuple[float, float, float]] = {
    JointType.FILLET: (0.20, 0.70, 1.00),
    JointType.TEE: (0.20, 0.70, 1.00),
    JointType.LAP: (0.95, 0.75, 0.10),
    JointType.BUTT: (0.40, 0.95, 0.40),
    JointType.CORNER: (0.95, 0.45, 0.85),
    JointType.EDGE: (0.70, 0.70, 0.70),
    JointType.PLUG: (1.00, 0.55, 0.20),
    JointType.UNKNOWN: (0.55, 0.55, 0.55),
}

_BACKGROUND = (0.105, 0.125, 0.153)
_HIGHLIGHT = (1.0, 0.45, 0.1)
_SELECTED = (1.0, 0.9, 0.3)


@dataclass
class PickResult:
    """What was under the cursor."""

    kind: str = "none"  # edge | face | seam | part | cell | none
    id: str = ""  # edge id, face id or seam id
    point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    layer: str = ""

    def __bool__(self) -> bool:
        return self.kind != "none"

    def __str__(self) -> str:
        if not self:
            return "nothing"
        where = " ".join(f"{v:.1f}" for v in self.point)
        return f"{self.kind} {self.id} at ({where})"


def _vtk():
    import vtk

    return vtk


def _polydata_from_mesh(vertices: np.ndarray, faces: np.ndarray):
    """Triangle polydata from NumPy arrays."""
    vtk = _vtk()
    from vtkmodules.util import numpy_support

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(vertices, dtype=float)))

    # The offset/connectivity pair, which is how VTK stores cells internally:
    # offsets[i] is where cell i starts in the flat connectivity array. Going
    # through this rather than the legacy [count, i, j, k] format keeps a
    # 900-face part a couple of array copies rather than a Python loop.
    triangles = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1, 3)
    offsets = np.arange(0, 3 * len(triangles) + 1, 3, dtype=np.int64)
    array = vtk.vtkCellArray()
    array.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_support.numpy_to_vtkIdTypeArray(triangles.reshape(-1), deep=True),
    )

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(array)
    return polydata


def _polydata_from_polylines(strips: list[np.ndarray]):
    """Polyline polydata, one VTK cell per strip."""
    vtk = _vtk()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    offset = 0
    for strip in strips:
        strip = np.asarray(strip, dtype=float).reshape(-1, 3)
        if len(strip) < 2:
            continue
        for point in strip:
            points.InsertNextPoint(*point)
        lines.InsertNextCell(len(strip))
        for i in range(len(strip)):
            lines.InsertCellPoint(offset + i)
        offset += len(strip)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


def _actor(polydata, colour, opacity=1.0, line_width=1.0, representation="surface"):
    vtk = _vtk()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*colour)
    prop.SetOpacity(opacity)
    prop.SetLineWidth(line_width)
    if representation == "wireframe":
        prop.SetRepresentationToWireframe()
    # Enough ambient that nothing is pitch black, enough specular that a
    # curved surface reads as curved.
    prop.SetAmbient(0.30)
    prop.SetDiffuse(0.80)
    prop.SetSpecular(0.25)
    prop.SetSpecularPower(25)
    return actor


def _cylinder_polydata(base, axis, radius, height, sides=18):
    vtk = _vtk()
    source = vtk.vtkCylinderSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(sides)
    source.CappingOn()
    source.Update()

    # vtkCylinderSource makes a Y-aligned cylinder centred on the origin.
    direction = tf.unit(axis)
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    rotation = tf.rotation_between(np.array([0.0, 1.0, 0.0]), direction)
    pose = tf.pose(rotation, np.asarray(base, dtype=float) + direction * (height / 2.0))

    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, float(pose[i, j]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetTransform(transform)
    filt.SetInputConnection(source.GetOutputPort())
    filt.Update()
    return filt.GetOutput()


def _box_polydata(size, centre):
    vtk = _vtk()
    source = vtk.vtkCubeSource()
    source.SetXLength(float(size[0]))
    source.SetYLength(float(size[1]))
    source.SetZLength(float(size[2]))
    source.SetCenter(*[float(v) for v in centre])
    source.Update()
    return source.GetOutput()


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------


class Scene3D:
    """Owns the renderer and everything in it."""

    def __init__(self, renderer: Any | None = None) -> None:
        vtk = _vtk()
        self.renderer = renderer if renderer is not None else vtk.vtkRenderer()
        self.renderer.SetBackground(*_BACKGROUND)
        self.renderer.SetUseDepthPeeling(0)
        # A two-light rig rather than VTK's single headlight: a headlight
        # alone flattens everything it illuminates, which is the opposite of
        # what a geometry review needs.
        self.renderer.SetAutomaticLightCreation(0)
        self.renderer.RemoveAllLights()
        for position, intensity in (
            ((-1.0, -2.0, 3.0), 0.95),
            ((2.0, 1.5, 1.0), 0.35),
        ):
            light = _vtk().vtkLight()
            light.SetLightTypeToCameraLight()
            light.SetPosition(*position)
            light.SetFocalPoint(0.0, 0.0, 0.0)
            light.SetIntensity(intensity)
            self.renderer.AddLight(light)

        self._actors: dict[str, list[Any]] = {name: [] for name, _, _, _ in LAYERS}
        self._visible: dict[str, bool] = {name: default for name, _, _, default in LAYERS}
        self._pickable: dict[str, bool] = {name: pickable for name, _, pickable, _ in LAYERS}
        #: actor -> ("kind", id) so a pick resolves to a domain object.
        self._actor_meta: dict[int, tuple[str, str, str]] = {}
        #: VTK cell index -> edge id, for the single edges actor.
        self._edge_by_cell: list[str] = []
        #: VTK cell index -> face id, for the single faces actor.
        self._face_by_cell: list[str] = []
        self._part_transform = tf.identity()
        self._part_actors: list[Any] = []
        self._robot_actor: Any | None = None
        self._torch_actor: Any | None = None
        self._frames: list[dict[str, Any]] = []
        self._highlighted: list[tuple[Any, tuple[float, float, float]]] = []
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.004)
        self._bounds: tuple[np.ndarray, np.ndarray] = (np.zeros(3), np.ones(3) * 1000.0)

    # -- layer bookkeeping --------------------------------------------------

    def _add(self, layer: str, actor: Any, kind: str = "", identifier: str = "") -> Any:
        self.renderer.AddActor(actor)
        self._actors.setdefault(layer, []).append(actor)
        actor.SetVisibility(1 if self._visible.get(layer, True) else 0)
        actor.SetPickable(1 if self._pickable.get(layer, False) else 0)
        if kind:
            self._actor_meta[id(actor)] = (kind, identifier, layer)
        return actor

    def clear(self) -> None:
        for actors in self._actors.values():
            for actor in actors:
                self.renderer.RemoveActor(actor)
            actors.clear()
        self._actor_meta.clear()
        self._edge_by_cell.clear()
        self._face_by_cell.clear()
        self._part_actors.clear()
        self._robot_actor = None
        self._torch_actor = None
        self._frames.clear()
        self._highlighted.clear()

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        self._visible[layer] = visible
        for actor in self._actors.get(layer, []):
            actor.SetVisibility(1 if visible else 0)

    def layer_visible(self, layer: str) -> bool:
        return self._visible.get(layer, False)

    def toggle_layer(self, layer: str) -> bool:
        self.set_layer_visible(layer, not self.layer_visible(layer))
        return self.layer_visible(layer)

    # -- building -----------------------------------------------------------

    def set_project(self, project: Project, robot: Any | None = None) -> None:
        """Rebuild the whole scene from a project."""
        self.clear()
        if project.cell is not None:
            self._build_cell(project.cell)
        if project.part is not None:
            self._build_part(project.part, project.placement)
        self._build_seams(project)
        if robot is not None:
            self._build_robot(robot)
        self._build_path(project)
        self._update_bounds()
        logger.debug(
            "scene: %s",
            {k: len(v) for k, v in self._actors.items() if v},
        )

    def _build_cell(self, cell: WeldCell) -> None:
        fixture = cell.fixture
        thickness = 60.0
        centre = np.asarray(fixture.table_origin, dtype=float).copy()
        centre[2] = fixture.table_z_mm - thickness / 2.0
        table = _actor(
            _box_polydata([fixture.table_size_mm[0], fixture.table_size_mm[1], thickness], centre),
            (0.23, 0.26, 0.31),
        )
        self._add("cell", table, "cell", "table")

        for clamp in fixture.clamps:
            actor = _actor(
                _cylinder_polydata(
                    clamp.position, clamp.approach, clamp.clearance_mm, clamp.height_mm
                ),
                (0.75, 0.38, 0.42),
                opacity=0.45,
            )
            self._add("clamps", actor, "cell", f"clamp:{clamp.id}")

        if fixture.datums:
            strips = []
            for position, normal in fixture.datums:
                direction = tf.unit(normal)
                if direction is None:
                    continue
                strips.append(np.array([position - direction * 30.0, position]))
            if strips:
                actor = _actor(_polydata_from_polylines(strips), (0.92, 0.80, 0.55), line_width=4.0)
                self._add("datums", actor)

        for obstacle in cell.all_obstacles():
            if not obstacle.has_mesh:
                continue
            # Obstacles are context, not subject: a solid jig frame hides the
            # bench and the part, which is what the operator is looking at.
            actor = _actor(
                _polydata_from_mesh(
                    tf.apply_points(obstacle.pose, obstacle.vertices), obstacle.faces
                ),
                (0.26, 0.30, 0.36),
                opacity=0.18,
            )
            self._add("cell", actor, "cell", f"{obstacle.kind}:{obstacle.id}")

    def _build_part(self, part: Part, placement: np.ndarray) -> None:
        """Part faces and edges, each individually pickable."""
        vtk = _vtk()
        self._part_transform = np.asarray(placement, dtype=float).copy()

        matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                matrix.SetElement(i, j, float(self._part_transform[i, j]))

        # --- faces, with a per-triangle face id -----------------------------
        per_face = self._tessellate_by_face(part)
        if per_face:
            vertices: list[np.ndarray] = []
            triangles: list[np.ndarray] = []
            offset = 0
            for face_id, face_vertices, face_triangles in per_face:
                vertices.append(face_vertices)
                triangles.append(face_triangles + offset)
                offset += len(face_vertices)
                self._face_by_cell.extend([face_id] * len(face_triangles))
            actor = _actor(
                _polydata_from_mesh(np.vstack(vertices), np.vstack(triangles)),
                (0.56, 0.60, 0.66),
            )
            actor.SetUserMatrix(matrix)
            self._part_actors.append(actor)
            self._add("part", actor, "face", "")
        elif part.mesh_vertices is not None and part.mesh_faces is not None:
            # Mesh import, or no B-rep: still show it, just not face-pickable.
            actor = _actor(
                _polydata_from_mesh(part.mesh_vertices, part.mesh_faces),
                (0.56, 0.60, 0.66),
            )
            actor.SetUserMatrix(matrix)
            self._part_actors.append(actor)
            self._add("part", actor, "part", part.id)

        # --- edges, one VTK cell each ---------------------------------------
        if part.edges:
            strips = []
            for edge_id, edge in part.edges.items():
                if len(edge.points) < 2:
                    continue
                strips.append(edge.points)
                self._edge_by_cell.append(edge_id)
            if strips:
                actor = _actor(_polydata_from_polylines(strips), (0.45, 0.50, 0.58), line_width=1.5)
                actor.SetUserMatrix(matrix)
                self._part_actors.append(actor)
                self._add("edges", actor, "edge", "")

    def _tessellate_by_face(self, part: Part) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """``(face_id, vertices, triangles)`` per CAD face, for face picking."""
        if not part.brep.available:
            return []
        try:
            backend = part.brep.backend
            topology = part.brep.topology
            out = []
            for key, vertices, triangles in backend.tessellate_by_face(part.brep.shape, 0.5):
                face_id = topology.face_ids.get(key, "")
                if face_id:
                    out.append((face_id, vertices, triangles))
            return out
        except Exception as exc:
            logger.warning("per-face tessellation unavailable: %s", exc)
            return []

    def _build_seams(self, project: Project) -> None:
        vtk = _vtk()
        for seam in project.seams:
            if seam.n_points < 2:
                continue
            placed = tf.apply_points(project.placement, seam.points)
            colour = JOINT_COLOURS.get(seam.joint_type, (0.6, 0.6, 0.6))
            if not seam.selected:
                # A deselected seam stays visible but recedes: the operator
                # needs to see it is there and see that it is off.
                colour = tuple(0.35 + 0.2 * c for c in colour)

            tube = vtk.vtkTubeFilter()
            tube.SetInputData(_polydata_from_polylines([placed]))
            tube.SetRadius(2.0)
            tube.SetNumberOfSides(8)
            tube.CappingOn()
            tube.Update()
            actor = _actor(tube.GetOutput(), colour)
            if project.is_manual(seam.id):
                actor.GetProperty().SetSpecular(0.6)
            self._add("seams", actor, "seam", seam.id)

            if len(seam.bisector):
                step = max(1, seam.n_points // 12)
                strips = []
                for i in range(0, seam.n_points, step):
                    start = placed[i]
                    direction = tf.apply_vector(project.placement, seam.bisector[i])
                    strips.append(np.array([start - direction * 28.0, start]))
                actor = _actor(_polydata_from_polylines(strips), (1.0, 0.85, 0.25), line_width=2.0)
                self._add("wire", actor)

    def _build_path(self, project: Project) -> None:
        job = project.job
        if job is None or not job.paths:
            return
        weld: list[np.ndarray] = []
        air: list[np.ndarray] = []
        for path in job.paths:
            for a, b in zip(path.points, path.points[1:], strict=False):
                (weld if a.weld_on and b.weld_on else air).append(
                    np.array([a.position, b.position])
                )
        if weld:
            self._add(
                "path",
                _actor(_polydata_from_polylines(weld), (1.0, 0.42, 0.21), line_width=3.0),
            )
        if air:
            self._add(
                "path",
                _actor(_polydata_from_polylines(air), (0.36, 0.40, 0.44), line_width=1.0),
            )

        # Animation frames: the arm polyline and the wire, per station.
        from ..kin import fk_all

        robot = getattr(project, "_robot", None)
        if robot is None:
            return
        for path in job.paths:
            for point in path.points:
                if point.joints is None:
                    continue
                frames = fk_all(robot, point.joints)
                self._frames.append(
                    {
                        "arm": np.array([tf.translation_of(f) for f in frames]),
                        "torch": np.array(
                            [
                                tf.translation_of(point.pose)
                                - tf.rotation_of(point.pose)[:, 2] * 120.0,
                                tf.translation_of(point.pose),
                            ]
                        ),
                        "weld": bool(point.weld_on),
                        "seam": path.seam_id,
                        "pass": path.pass_index,
                    }
                )

    def _build_robot(self, robot: Any) -> None:
        from ..kin import fk_all

        # The column it stands on. Without it the arm appears to float, and an
        # operator judging whether the part is too close to the robot has
        # nothing to judge it against.
        origin = tf.translation_of(robot.base)
        column_height = max(float(origin[2]), 120.0)
        column = _actor(
            _cylinder_polydata(
                origin - np.array([0.0, 0.0, column_height]),
                np.array([0.0, 0.0, 1.0]),
                150.0,
                column_height,
                20,
            ),
            (0.30, 0.38, 0.47),
        )
        self._add("robot", column, "cell", "robot_base")

        frames = fk_all(robot, robot.home_joints())
        arm = np.array([tf.translation_of(frame) for frame in frames])
        vtk = _vtk()
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(_polydata_from_polylines([arm]))
        tube.SetRadius(28.0)
        tube.SetNumberOfSides(10)
        tube.CappingOn()
        tube.Update()
        self._robot_actor = _actor(tube.GetOutput(), (0.42, 0.58, 0.72))
        self._add("robot", self._robot_actor)

        self._torch_actor = _actor(
            _polydata_from_polylines([arm[-2:]]), (0.85, 0.87, 0.91), line_width=5.0
        )
        self._add("robot", self._torch_actor)

    # -- cheap updates ------------------------------------------------------

    def update_placement(self, pose: np.ndarray) -> None:
        """Move the part without rebuilding it.

        Dragging the part has to feel immediate, and re-tessellating a
        50 000-triangle weldment per mouse move does not. The actors carry the
        placement as their own matrix instead.
        """
        vtk = _vtk()
        self._part_transform = np.asarray(pose, dtype=float).copy()
        matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                matrix.SetElement(i, j, float(self._part_transform[i, j]))
        for actor in self._part_actors:
            actor.SetUserMatrix(matrix)

    def refresh_seams(self, project: Project) -> None:
        for actor in self._actors.get("seams", []):
            self.renderer.RemoveActor(actor)
            self._actor_meta.pop(id(actor), None)
        for actor in self._actors.get("wire", []):
            self.renderer.RemoveActor(actor)
        self._actors["seams"] = []
        self._actors["wire"] = []
        self._build_seams(project)

    def set_frame(self, index: int) -> dict[str, Any] | None:
        """Move the arm to animation frame *index*."""
        if not self._frames or self._robot_actor is None:
            return None
        frame = self._frames[index % len(self._frames)]
        vtk = _vtk()
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(_polydata_from_polylines([frame["arm"]]))
        tube.SetRadius(28.0)
        tube.SetNumberOfSides(10)
        tube.CappingOn()
        tube.Update()
        self._robot_actor.GetMapper().SetInputData(tube.GetOutput())
        if self._torch_actor is not None:
            self._torch_actor.GetMapper().SetInputData(_polydata_from_polylines([frame["torch"]]))
            # The torch glows while the arc is burning: the clearest possible
            # indication of what the program is doing at this instant.
            self._torch_actor.GetProperty().SetColor(
                *((1.0, 0.42, 0.13) if frame["weld"] else (0.85, 0.87, 0.91))
            )
            self._torch_actor.GetProperty().SetLineWidth(7.0 if frame["weld"] else 4.0)
        return frame

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    # -- picking ------------------------------------------------------------

    def pick(self, x: float, y: float) -> PickResult:
        """What is under viewport coordinates ``(x, y)``."""
        if not self._picker.Pick(float(x), float(y), 0.0, self.renderer):
            return PickResult()
        actor = self._picker.GetActor()
        if actor is None:
            return PickResult()

        meta = self._actor_meta.get(id(actor))
        if meta is None:
            return PickResult()
        kind, identifier, layer = meta
        world = np.array(self._picker.GetPickPosition(), dtype=float)
        cell = int(self._picker.GetCellId())

        if kind == "edge":
            if 0 <= cell < len(self._edge_by_cell):
                identifier = self._edge_by_cell[cell]
            else:
                return PickResult()
        elif kind == "face":
            if 0 <= cell < len(self._face_by_cell):
                identifier = self._face_by_cell[cell]
            else:
                return PickResult()

        local = tf.apply_point(tf.inverse(self._part_transform), world)
        return PickResult(kind=kind, id=identifier, point=world, point_local=local, layer=layer)

    def set_pickable(self, layer: str, pickable: bool) -> None:
        """Restrict picking to the layer a tool cares about.

        An edge-tracing tool that also picks faces makes the operator fight
        the viewport, so the active tool narrows what can be hit.
        """
        self._pickable[layer] = pickable
        for actor in self._actors.get(layer, []):
            actor.SetPickable(1 if pickable else 0)

    def pick_only(self, *layers: str) -> None:
        for name, _, _, _ in LAYERS:
            self.set_pickable(name, name in layers)

    def restore_pickable(self) -> None:
        for name, _, pickable, _ in LAYERS:
            self.set_pickable(name, pickable)

    # -- highlighting -------------------------------------------------------

    def clear_highlight(self) -> None:
        for actor, colour in self._highlighted:
            actor.GetProperty().SetColor(*colour)
        self._highlighted.clear()

    def highlight(self, kind: str, identifier: str, colour=None) -> bool:
        """Highlight one domain object; returns whether it was found."""
        self.clear_highlight()
        target = colour or _HIGHLIGHT
        for actor, (actor_kind, actor_id, _) in list(
            zip(
                [a for layer in self._actors.values() for a in layer],
                [
                    self._actor_meta.get(id(a), ("", "", ""))
                    for layer in self._actors.values()
                    for a in layer
                ],
                strict=True,
            )
        ):
            if actor_kind == kind and actor_id == identifier:
                prop = actor.GetProperty()
                self._highlighted.append((actor, prop.GetColor()))
                prop.SetColor(*target)
                return True
        return False

    def highlight_points(self, points: np.ndarray, name: str = "trace") -> None:
        """Show a provisional polyline, for a seam being traced.

        Drawn in the selection colour and rebuilt on every pick, so the
        operator sees the weld line take shape as they click along it.
        """
        for actor in list(self._actors.get("seams", [])):
            if self._actor_meta.get(id(actor), ("", "", ""))[1] == f"__{name}__":
                self.renderer.RemoveActor(actor)
                self._actors["seams"].remove(actor)
                self._actor_meta.pop(id(actor), None)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) < 2:
            return
        vtk = _vtk()
        placed = tf.apply_points(self._part_transform, points)
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(_polydata_from_polylines([placed]))
        tube.SetRadius(2.5)
        tube.SetNumberOfSides(8)
        tube.Update()
        actor = _actor(tube.GetOutput(), _SELECTED)
        self._add("seams", actor, "trace", f"__{name}__")

    # -- camera -------------------------------------------------------------

    def _update_bounds(self) -> None:
        bounds = self.renderer.ComputeVisiblePropBounds()
        if bounds[1] < bounds[0]:
            return
        self._bounds = (
            np.array([bounds[0], bounds[2], bounds[4]]),
            np.array([bounds[1], bounds[3], bounds[5]]),
        )

    def reset_camera(self, view: str = "iso") -> None:
        """Standard views. Z is up, as it is everywhere in the toolchain."""
        self._update_bounds()
        lo, hi = self._bounds
        centre = (lo + hi) / 2.0
        span = float(np.max(hi - lo)) or 1000.0
        directions = {
            "iso": np.array([-1.0, -1.0, 0.8]),
            "top": np.array([0.0, 0.0, 1.0]),
            "front": np.array([0.0, -1.0, 0.0]),
            "back": np.array([0.0, 1.0, 0.0]),
            "left": np.array([-1.0, 0.0, 0.0]),
            "right": np.array([1.0, 0.0, 0.0]),
        }
        direction = tf.unit(directions.get(view, directions["iso"]))
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*centre)
        camera.SetPosition(*(centre + direction * span * 2.2))
        camera.SetViewUp(0.0, 0.0, 1.0) if view != "top" else camera.SetViewUp(0.0, 1.0, 0.0)
        self.renderer.ResetCameraClippingRange()

    def look_at(self, point: np.ndarray, span: float = 300.0) -> None:
        """Zoom to a seam the operator selected in the list."""
        camera = self.renderer.GetActiveCamera()
        position = np.array(camera.GetPosition(), dtype=float)
        focal = np.array(camera.GetFocalPoint(), dtype=float)
        offset = position - focal
        direction = tf.unit(offset)
        if direction is None:
            direction = np.array([-1.0, -1.0, 0.8]) / np.sqrt(2.64)
        camera.SetFocalPoint(*point)
        camera.SetPosition(*(np.asarray(point, dtype=float) + direction * span * 2.0))
        self.renderer.ResetCameraClippingRange()

    # -- offscreen output ---------------------------------------------------

    def render_to_png(self, path: str, width: int = 1280, height: int = 800) -> str:
        """Render the scene to a PNG.

        Used by the tests, and by the application to drop a picture of the
        setup next to the generated program -- which is what makes a job
        folder reviewable a year later.
        """
        vtk = _vtk()
        window = self.renderer.GetRenderWindow()
        owned = window is None
        if owned:
            window = vtk.vtkRenderWindow()
            window.SetOffScreenRendering(1)
            window.AddRenderer(self.renderer)
        window.SetSize(width, height)
        window.Render()

        grabber = vtk.vtkWindowToImageFilter()
        grabber.SetInput(window)
        grabber.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(grabber.GetOutputPort())
        writer.Write()
        if owned:
            window.RemoveRenderer(self.renderer)
        return str(path)
