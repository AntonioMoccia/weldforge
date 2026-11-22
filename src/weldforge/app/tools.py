"""The interactive tools, written as state machines.

Deliberately free of Qt and of VTK. A tool receives picks and commands and
mutates the project; what it does not do is know how the pick arrived or how
the result is drawn. That split is what makes the behaviour testable -- and
the behaviour is where the mistakes are. The widget layer is a few hundred
lines of wiring; the rules below are the application.

Each tool exposes the same small surface:

``prompt()``      what the operator should do next, in words
``on_pick()``     a click in the viewport landed on something
``on_command()``  a keystroke or button: accept, cancel, undo the last pick
``preview()``     provisional geometry to draw while the tool is mid-action
``commit()``      apply the result to the project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geom import polyline as pl
from ..geom import transform as tf
from ..model import JointType, Seam
from ..seams.classify import ContactKind, JointContext, classify
from ..seams.dihedral import (
    FaceSide,
    InwardMode,
    compute_seam_geometry,
    resolve_inward_sign,
)
from .project import Project
from .scene3d import PickResult

logger = logging.getLogger(__name__)

__all__ = [
    "ToolResult",
    "Tool",
    "SelectTool",
    "TraceSeamTool",
    "PlacePartTool",
    "ToolError",
]


class ToolError(RuntimeError):
    """The tool cannot do what was asked, with a reason for the operator."""


@dataclass
class ToolResult:
    """What a tool did, for the status bar and the undo stack."""

    ok: bool = True
    message: str = ""
    changed: bool = False
    seam_id: str = ""
    finished: bool = False

    @classmethod
    def info(cls, message: str, **kwargs: Any) -> ToolResult:
        return cls(ok=True, message=message, **kwargs)

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(ok=False, message=message)


class Tool:
    """Base class. A tool lives as long as it is the active one."""

    name = "tool"
    title = "Tool"

    #: Scene layers this tool wants pickable; empty means leave them alone.
    pick_layers: tuple[str, ...] = ()

    def __init__(self, project: Project) -> None:
        self.project = project

    def prompt(self) -> str:
        return ""

    def on_pick(self, pick: PickResult) -> ToolResult:
        return ToolResult.info("")

    def on_command(self, command: str) -> ToolResult:
        return ToolResult.info("")

    def preview(self) -> np.ndarray | None:
        return None

    def cancel(self) -> ToolResult:
        return ToolResult.info("cancelled", finished=True)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


class SelectTool(Tool):
    """Click a seam to select it; click again to toggle whether it is welded."""

    name = "select"
    title = "Select"
    pick_layers = ("seams", "part", "cell")

    def __init__(self, project: Project) -> None:
        super().__init__(project)
        self.selected_id: str = ""

    def prompt(self) -> str:
        return "Click a weld seam to select it."

    def on_pick(self, pick: PickResult) -> ToolResult:
        if pick.kind != "seam":
            self.selected_id = ""
            return ToolResult.info("")
        self.selected_id = pick.id
        seam = self.project.seam(pick.id)
        if seam is None:
            return ToolResult.info("")
        return ToolResult.info(
            f"{seam.label or seam.id}: {seam.joint_type.value}, "
            f"{seam.length_mm:.0f} mm, {seam.dominant_position.value}, "
            f"{'welded' if seam.selected else 'not welded'}",
            seam_id=pick.id,
        )

    def on_command(self, command: str) -> ToolResult:
        if command not in ("toggle", "delete"):
            return ToolResult.info("")
        seam = self.project.seam(self.selected_id)
        if seam is None:
            return ToolResult.error("no seam selected")
        if command == "toggle":
            seam.selected = not seam.selected
            self.project.touch()
            return ToolResult.info(
                f"{seam.label or seam.id}: {'will be welded' if seam.selected else 'excluded'}",
                changed=True,
                seam_id=seam.id,
            )
        # Only a hand-traced seam is the operator's to delete; a detected one
        # would come straight back on the next detection run, which looks like
        # the application ignoring them.
        if not self.project.is_manual(seam.id):
            return ToolResult.error(
                "that seam was detected from the geometry. Switch it off "
                "instead of deleting it, or it returns on the next detection."
            )
        self.project.remove_seam(seam.id)
        self.selected_id = ""
        return ToolResult.info("seam deleted", changed=True)


# ---------------------------------------------------------------------------
# tracing a weld line by hand
# ---------------------------------------------------------------------------


@dataclass
class _TraceState:
    edge_ids: list[str] = field(default_factory=list)
    flank_face_id: str = ""
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def clear(self) -> None:
        self.edge_ids.clear()
        self.flank_face_id = ""
        self.points = np.zeros((0, 3))


class TraceSeamTool(Tool):
    """Draw a weld line by clicking the model.

    Two ways to say where a weld goes, because weldments come two ways.

    **Along edges.** Click one edge after another and they chain into a seam.
    The flanks are the two faces that meet at the edge, so this works wherever
    the joint is visible in the geometry -- a bent body, a fused weldment.

    **Edge plus face.** Click an edge, then click the face on the *other* part
    that it beds against. This is the assembly case, where the two bodies
    touch but share no topology, and it is the one an operator reaches for
    most: it says "weld this edge down onto that plate", which is exactly how
    a welder describes it.

    The resulting seam carries the same geometry as a detected one -- flank
    normals, groove angle, wire direction -- so everything downstream treats
    it identically. A hand-traced seam is not a second-class annotation; it is
    a seam.
    """

    name = "trace"
    title = "Trace weld line"
    pick_layers = ("edges", "part")

    def __init__(self, project: Project, joint_type: JointType | None = None) -> None:
        super().__init__(project)
        self.state = _TraceState()
        self.joint_type = joint_type
        self.chain_tolerance_mm = 1.0

    # -- prompting ----------------------------------------------------------

    def prompt(self) -> str:
        if not self.state.edge_ids:
            return "Click an edge of the joint. Esc cancels."
        if not self.state.flank_face_id:
            return (
                f"{len(self.state.edge_ids)} edge(s). Click the next edge, or the "
                "face the joint beds against, then Enter to accept."
            )
        return (
            f"{len(self.state.edge_ids)} edge(s) against a face. "
            "Enter accepts, Backspace undoes the last pick."
        )

    def preview(self) -> np.ndarray | None:
        return self.state.points if len(self.state.points) >= 2 else None

    # -- picking ------------------------------------------------------------

    def on_pick(self, pick: PickResult) -> ToolResult:
        if pick.kind == "edge":
            return self._add_edge(pick.id)
        if pick.kind == "face":
            return self._set_flank(pick.id)
        return ToolResult.info("")

    def _add_edge(self, edge_id: str) -> ToolResult:
        part = self.project.part
        if part is None or edge_id not in part.edges:
            return ToolResult.error("that edge is not part of the loaded model")
        if edge_id in self.state.edge_ids:
            return ToolResult.error("that edge is already in the line")

        candidate = [*self.state.edge_ids, edge_id]
        points = self._chain(candidate)
        if points is None:
            return ToolResult.error(
                "that edge does not meet the end of the line. Pick an edge that "
                "continues it, or Esc and start again."
            )
        self.state.edge_ids = candidate
        self.state.points = points
        return ToolResult.info(
            f"{len(candidate)} edge(s), {pl.total_length(points):.0f} mm", changed=False
        )

    def _set_flank(self, face_id: str) -> ToolResult:
        part = self.project.part
        if part is None or face_id not in part.faces:
            return ToolResult.error("that face is not part of the loaded model")
        if not self.state.edge_ids:
            return ToolResult.error("pick the edge of the joint first")

        face = part.faces[face_id]
        edge = part.edges[self.state.edge_ids[0]]
        if face_id in edge.face_ids:
            return ToolResult.error(
                "that face already meets the edge. Pick the face on the other "
                "part, the one the edge beds against."
            )
        self.state.flank_face_id = face_id
        return ToolResult.info(f"bedding against {face_id} ({face.face_type}). Enter to accept.")

    def _chain(self, edge_ids: list[str]) -> np.ndarray | None:
        """Points of the chained edges, in order, or ``None`` if they do not meet."""
        part = self.project.part
        if part is None:
            return None
        points = np.asarray(part.edges[edge_ids[0]].points, dtype=float)
        for edge_id in edge_ids[1:]:
            nxt = np.asarray(part.edges[edge_id].points, dtype=float)
            # Try the four ways two polylines can meet end to end.
            options = [
                (np.linalg.norm(points[-1] - nxt[0]), points, nxt),
                (np.linalg.norm(points[-1] - nxt[-1]), points, nxt[::-1]),
                (np.linalg.norm(points[0] - nxt[-1]), points[::-1], nxt[::-1]),
                (np.linalg.norm(points[0] - nxt[0]), points[::-1], nxt),
            ]
            distance, head, tail = min(options, key=lambda item: item[0])
            if distance > self.chain_tolerance_mm:
                return None
            points = np.vstack([head, tail[1:]])
        return pl.dedupe(points)

    # -- commands -----------------------------------------------------------

    def on_command(self, command: str) -> ToolResult:
        if command == "undo":
            if self.state.flank_face_id:
                self.state.flank_face_id = ""
                return ToolResult.info("bedding face cleared")
            if self.state.edge_ids:
                self.state.edge_ids.pop()
                self.state.points = (
                    self._chain(self.state.edge_ids) if self.state.edge_ids else np.zeros((0, 3))
                )
                return ToolResult.info(f"{len(self.state.edge_ids)} edge(s)")
            return ToolResult.info("")
        if command == "cancel":
            self.state.clear()
            return ToolResult.info("tracing cancelled", finished=True)
        if command == "accept":
            return self.commit()
        return ToolResult.info("")

    # -- committing ---------------------------------------------------------

    def commit(self) -> ToolResult:
        """Turn the picked geometry into a real seam."""
        part = self.project.part
        if part is None or not self.state.edge_ids:
            return ToolResult.error("nothing traced yet")
        if not part.brep.available:
            return ToolResult.error(
                "tracing needs the exact geometry, which this part does not "
                "carry (it was imported as a mesh)."
            )

        try:
            seam = self._build_seam()
        except ToolError as exc:
            return ToolResult.error(str(exc))

        self.project.add_manual_seam(seam)
        length = seam.length_mm
        groove = float(np.rad2deg(seam.mean_dihedral))
        self.state.clear()

        # The operator's instruction is accepted -- they pointed at this line
        # deliberately -- but the geometry gets its say. A groove outside the
        # weldable band means the line is along an outside corner or a pair of
        # nearly parallel faces, and welding it needs preparation or is not a
        # joint at all. Saying so beats letting it through unremarked.
        warning = ""
        if groove > 215.0:
            warning = (
                f" WARNING: this measures a {groove:.0f} deg groove, which is an "
                "outside corner. There is no joint here unless the edges are "
                "prepared -- check you clicked the inside of the joint."
            )
        elif groove < 30.0:
            warning = (
                f" WARNING: this measures a {groove:.0f} deg groove: the faces are "
                "nearly parallel, so there is no room for a torch."
            )
        if warning:
            seam.notes.append(warning.strip())

        return ToolResult.info(
            f"weld line added: {seam.joint_type.value}, {length:.0f} mm, "
            f"groove {groove:.0f} deg{warning}",
            changed=True,
            seam_id=seam.id,
            finished=True,
        )

    def _build_seam(self) -> Seam:
        part = self.project.part
        backend = part.brep.backend
        topology = part.brep.topology
        evaluators = part.brep.evaluators

        points = self.state.points
        if len(points) < 2:
            raise ToolError("the traced line is too short")
        tangents = pl.tangents(points, closed=pl.is_closed(points))

        first_edge = part.edges[self.state.edge_ids[0]]
        face_ids = list(first_edge.face_ids)

        # Which two faces are the flanks, and how to find which way they run.
        if self.state.flank_face_id:
            # Edge plus bedding face: the flank rising from the edge, against
            # the face it beds on. The faces continue past the seam, so the
            # in-face directions come from normal projection.
            flank_id = face_ids[0] if face_ids else ""
            if not flank_id:
                raise ToolError("that edge has no face to weld from")
            a_id, b_id = self.state.flank_face_id, flank_id
            mode = InwardMode.CONCAVE
            contact = ContactKind.FACE_CONTACT
        elif len(face_ids) >= 2:
            a_id, b_id = face_ids[0], face_ids[1]
            mode = InwardMode.TRIM
            contact = ContactKind.SHARED_TOPOLOGY
        else:
            raise ToolError(
                "that edge has only one face, so there is no joint at it. "
                "Click the face the edge beds against as well."
            )

        side_a = self._side(a_id, backend, topology, evaluators)
        side_b = self._side(b_id, backend, topology, evaluators)

        if mode is InwardMode.TRIM:
            middle = len(points) // 2
            side_a.inward_sign = resolve_inward_sign(
                backend, side_a.face, side_a.evaluator, points[middle], tangents[middle]
            )
            side_b.inward_sign = resolve_inward_sign(
                backend, side_b.face, side_b.evaluator, points[middle], tangents[middle]
            )

        geometry = compute_seam_geometry(points, tangents, side_a, side_b, mode)

        context = JointContext(
            groove_rad=geometry.mean_groove,
            concavity=geometry.dominant_concavity,
            groove_spread_rad=geometry.groove_spread,
            same_solid=side_a.solid_id == side_b.solid_id,
            contact_kind=contact,
            face_a_type=side_a.evaluator.face_type,
            face_b_type=side_b.evaluator.face_type,
            thickness_a_mm=side_a.thickness_mm,
            thickness_b_mm=side_b.thickness_mm,
            seam_length_mm=pl.total_length(points),
            closed=pl.is_closed(points),
            exact_geometry=True,
        )
        result = classify(context)
        joint_type = self.joint_type or result.joint_type
        if joint_type is JointType.UNKNOWN:
            # The operator pointed at this line deliberately, so refusing it
            # outright would be wrong -- but calling it nothing is no help
            # either. Fall back to a fillet and say so.
            joint_type = JointType.FILLET
            result.notes.append(
                "the geometry did not classify; assumed a fillet because the "
                "line was traced by hand"
            )

        seam = Seam(
            joint_type=joint_type,
            points=geometry.points,
            tangents=geometry.tangents,
            normal_a=geometry.normal_a,
            normal_b=geometry.normal_b,
            bisector=geometry.bisector,
            dihedral=geometry.groove,
            face_a_id=side_a.face_id,
            face_b_id=side_b.face_id,
            solid_a_id=side_a.solid_id,
            solid_b_id=side_b.solid_id,
            edge_ids=tuple(self.state.edge_ids),
            thickness_a_mm=side_a.thickness_mm,
            thickness_b_mm=side_b.thickness_mm,
            # A hand-traced line is a decision, not a guess: the operator said
            # the joint is here. The confidence reflects that, and the note
            # says where it came from.
            confidence=1.0,
            closed=pl.is_closed(points),
            notes=[*result.notes],
        )
        seam.label = (
            f"manual - {joint_type.value} {seam.length_mm:.0f} mm ({seam.dominant_position.value})"
        )
        return seam

    def _side(self, face_id: str, backend, topology, evaluators) -> FaceSide:
        part = self.project.part
        face_ref = part.faces.get(face_id)
        if face_ref is None:
            raise ToolError(f"unknown face {face_id}")
        key = next((k for k, v in topology.face_ids.items() if v == face_id), None)
        if key is None:
            raise ToolError(f"face {face_id} is not in the loaded topology")
        solid = part.solid(face_ref.solid_id)
        return FaceSide(
            face_id=face_id,
            solid_id=face_ref.solid_id,
            face=topology.faces[key],
            evaluator=evaluators[key],
            thickness_mm=solid.thickness_mm if solid else 0.0,
        )


# ---------------------------------------------------------------------------
# placing the part
# ---------------------------------------------------------------------------


class PlacePartTool(Tool):
    """Move the part on the bench, or into the jig.

    Four ways, because an operator wants different ones at different moments:

    * **nudge** -- translate and rotate by a step, for fine adjustment;
    * **drop** -- rest it on the bench wherever it is in plan;
    * **snap** -- seat it on the jig's locators, which is a solve, not a search;
    * **auto** -- run the placement search and take its answer.
    """

    name = "place"
    title = "Place part"
    pick_layers = ("part", "cell")

    def __init__(self, project: Project) -> None:
        super().__init__(project)
        self.step_mm = 10.0
        self.step_deg = 15.0
        self._undo: list[np.ndarray] = []

    def prompt(self) -> str:
        if self.project.placement_locked:
            return "The placement is locked. Unlock it to move the part."
        return (
            "Arrow keys move, PageUp/PageDown raise and lower, "
            "Q/E rotate. 'Drop' rests it on the bench, 'Snap' seats it in the jig."
        )

    def _apply(self, pose: np.ndarray, message: str) -> ToolResult:
        if self.project.placement_locked:
            return ToolResult.error("the placement is locked: unlock it before moving the part")
        self._undo.append(self.project.placement.copy())
        self.project.placement = pose
        self.project.touch()
        return ToolResult.info(message, changed=True)

    def translate(self, delta: np.ndarray) -> ToolResult:
        pose = self.project.placement.copy()
        pose[:3, 3] = pose[:3, 3] + np.asarray(delta, dtype=float)
        position = tf.translation_of(pose)
        return self._apply(
            pose, f"part at {position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f} mm"
        )

    def rotate(self, axis: np.ndarray, degrees: float) -> ToolResult:
        """Rotate about the part's own centre, not the cell origin.

        Rotating about the origin sends the part flying off the bench, which
        is never what the operator meant.
        """
        part = self.project.part
        if part is None:
            return ToolResult.error("no part loaded")
        lo, hi = part.bbox
        centre_world = tf.apply_point(self.project.placement, (lo + hi) / 2.0)
        rotation = tf.rot_axis_angle(np.asarray(axis, dtype=float), np.deg2rad(degrees))
        about = tf.compose(
            tf.pose(np.eye(3), centre_world),
            tf.pose(rotation),
            tf.pose(np.eye(3), -centre_world),
        )
        return self._apply(about @ self.project.placement, f"rotated {degrees:+.0f} deg")

    def drop_to_table(self) -> ToolResult:
        """Rest the part on the bench, keeping its plan position and heading."""
        part, cell = self.project.part, self.project.cell
        if part is None or cell is None:
            return ToolResult.error("load a part and a cell first")
        if part.mesh_vertices is None or not len(part.mesh_vertices):
            return ToolResult.error("the part has no mesh to rest on the bench")
        placed = tf.apply_points(self.project.placement, part.mesh_vertices)
        lift = cell.fixture.table_z_mm - float(placed[:, 2].min())
        pose = self.project.placement.copy()
        pose[2, 3] += lift
        return self._apply(pose, f"dropped {lift:+.1f} mm onto the bench")

    def snap_to_jig(self) -> ToolResult:
        """Seat the part on the jig's locators, keeping its current orientation."""
        from ..fixture import solve_jig_placement

        part, cell = self.project.part, self.project.cell
        if part is None or cell is None:
            return ToolResult.error("load a part and a cell first")
        if not cell.fixture.datums:
            return ToolResult.error(
                f"{cell.fixture.name} has no locators. Use 'Drop' on a plain bench."
            )
        pose = solve_jig_placement(part, cell, tf.rotation_of(self.project.placement))
        if pose is None:
            return ToolResult.error("could not seat the part on the locators")
        return self._apply(pose, "seated on the jig locators")

    def auto_place(self, robot: Any | None = None) -> ToolResult:
        """Run the placement search and take its answer."""
        from ..fixture import cell_robot, find_placement

        part, cell = self.project.part, self.project.cell
        if part is None or cell is None:
            return ToolResult.error("load a part and a cell first")
        seams = self.project.selected_seams or self.project.seams
        if not seams:
            return ToolResult.error(
                "detect or trace the weld seams first: the search places the "
                "part so the robot can reach them"
            )
        arm = robot if robot is not None else cell_robot(cell)
        best, evaluated = find_placement(part, seams, cell, robot=arm)
        message = (
            f"{len(evaluated)} placements tried, best scores {best.score:.2f} "
            f"with {len(best.reachable_seams)}/{len(seams)} seams reachable"
        )
        if best.violations:
            message += f"; {best.violations[0]}"
        return self._apply(best.pose, message)

    def set_pose(self, pose: np.ndarray) -> ToolResult:
        return self._apply(np.asarray(pose, dtype=float), "placement set")

    def undo(self) -> ToolResult:
        if not self._undo:
            return ToolResult.error("nothing to undo")
        self.project.placement = self._undo.pop()
        self.project.touch()
        return ToolResult.info("placement restored", changed=True)

    def on_command(self, command: str) -> ToolResult:
        moves = {
            "left": np.array([-self.step_mm, 0.0, 0.0]),
            "right": np.array([self.step_mm, 0.0, 0.0]),
            "forward": np.array([0.0, self.step_mm, 0.0]),
            "back": np.array([0.0, -self.step_mm, 0.0]),
            "up": np.array([0.0, 0.0, self.step_mm]),
            "down": np.array([0.0, 0.0, -self.step_mm]),
        }
        if command in moves:
            return self.translate(moves[command])
        rotations = {
            "yaw_ccw": (np.array([0.0, 0.0, 1.0]), self.step_deg),
            "yaw_cw": (np.array([0.0, 0.0, 1.0]), -self.step_deg),
            "pitch_up": (np.array([0.0, 1.0, 0.0]), self.step_deg),
            "pitch_down": (np.array([0.0, 1.0, 0.0]), -self.step_deg),
            "roll_ccw": (np.array([1.0, 0.0, 0.0]), self.step_deg),
            "roll_cw": (np.array([1.0, 0.0, 0.0]), -self.step_deg),
        }
        if command in rotations:
            axis, degrees = rotations[command]
            return self.rotate(axis, degrees)
        if command == "drop":
            return self.drop_to_table()
        if command == "snap":
            return self.snap_to_jig()
        if command == "auto":
            return self.auto_place()
        if command == "undo":
            return self.undo()
        return ToolResult.info("")
