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
from ..i18n import t
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
    "TwoFaceTool",
    "MoveCellTool",
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

    # -- dragging -----------------------------------------------------------
    #
    # A tool that moves things needs the mouse between press and release, not
    # just the click. The window asks ``dragging()`` after every pick and,
    # while it is true, feeds ``on_drag`` a point in the world -- already
    # projected onto the plane the drag runs in, so no tool knows about
    # cameras. Still no Qt and no VTK here.

    def dragging(self) -> bool:
        """True while this tool is holding something and wants mouse moves."""
        return False

    def drag_plane_z(self) -> float | None:
        """Height of the plane the drag runs in, or ``None`` when not dragging."""
        return None

    def on_drag(self, point: np.ndarray) -> ToolResult:
        return ToolResult.info("")

    def on_drop(self) -> ToolResult:
        return ToolResult.info("")

    def drag_preview(self) -> tuple[str, np.ndarray] | None:
        """``(scene key, offset)`` to shift while dragging, before committing.

        Moving the actor is what makes a drag feel like a drag; rebuilding the
        whole cell on every mouse move would not. The model is written on
        release, in one step the operator can undo as one step.
        """
        return None

    def drag_through(self) -> np.ndarray | None:
        """The grab point, for a drag plane that faces the camera.

        The horizontal plane is correct but, seen at a shallow angle, turns a
        nudge into metres. When this is given the window projects the cursor
        onto the view-facing plane through it instead.
        """
        return None

    def busy(self) -> bool:
        """True while mid-action, when Enter and Esc mean something.

        Lets the window route those keys here from wherever the focus is: an
        operator who typed a bench height and then traced a line presses
        Enter with the focus still in the spin box, and the line must still
        be accepted.
        """
        return False


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
        return t("Click a weld seam to select it.")

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
        # Two real edges on a filleted part are separated by the blend
        # between them, a few millimetres that are not pickable (see
        # MIN_PICK_EDGE_MM). At 1 mm they never "met" and the line could not
        # be continued; 5 mm bridges a blend and is still far below any gap
        # that could be mistaken for a different seam.
        self.chain_tolerance_mm = 5.0

    # -- prompting ----------------------------------------------------------

    def prompt(self) -> str:
        if not self.state.edge_ids:
            return t("Click an edge of the joint. Esc cancels.")
        if not self.state.flank_face_id:
            return t(
                "{count} edge(s). Click the next edge, or the face the joint beds "
                "against, then Enter to accept.",
                count=len(self.state.edge_ids),
            )
        return t(
            "{count} edge(s) against a face. Enter accepts, Backspace undoes the "
            "last pick.",
            count=len(self.state.edge_ids),
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

    def busy(self) -> bool:
        return bool(self.state.edge_ids)

    def _add_edge(self, edge_id: str) -> ToolResult:
        part = self.project.part
        if part is None or edge_id not in part.edges:
            return ToolResult.error("that edge is not part of the loaded model")
        if self.state.edge_ids and edge_id == self.state.edge_ids[-1]:
            # Clicking the edge you just picked again is how people say "done"
            # -- the log showed it thirteen times in one session, each one
            # answered with a refusal. Treat it as accepting the line; a seam
            # accepted by mistake is one click to delete.
            return self.commit()
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
# tracing a weld from the two faces that form the joint
# ---------------------------------------------------------------------------


class TwoFaceTool(Tool):
    """Click the two surfaces to be joined; the seam and its angle follow.

    This is how a welder describes a joint -- "this plate onto that one" --
    and how the better offline-programming tools let you point at it. The
    operator says *what* is welded; where the seam runs, how wide the groove
    is and which way the torch points are derived, not clicked:

    * two faces of the **same body** meet along an edge: that edge is the
      seam, and the groove is the angle between the faces there;
    * two faces of **different bodies** touch (an assembly): the contact
      detector already knows where, so the seam comes from that contact,
      with the measured gap carried into it for the procedure;
    * faces that do not meet are refused with a reason, not guessed at.

    Edge-by-edge tracing stays available for the odd case. It is no longer
    the front door, because on a real part the front door was a hundred
    fillet blends an operator had to pick around.
    """

    name = "twofaces"
    title = "Trace weld from two faces"
    pick_layers = ("part",)

    def __init__(self, project: Project, joint_type: JointType | None = None) -> None:
        super().__init__(project)
        self.face_a: str = ""
        self.face_b: str = ""
        self.joint_type = joint_type
        self._seam: Seam | None = None

    # -- prompting ----------------------------------------------------------

    def prompt(self) -> str:
        if not self.face_a:
            return t("Click the first surface of the joint.")
        if self._seam is None:
            return t("Click the surface it is welded to. Esc starts over.")
        return t("Accept with Enter, a click on the seam, or pick two new surfaces.")

    def busy(self) -> bool:
        return bool(self.face_a)

    def preview(self) -> np.ndarray | None:
        return self._seam.points if self._seam is not None else None

    # -- picking ------------------------------------------------------------

    def on_pick(self, pick: PickResult) -> ToolResult:
        if pick.kind == "seam" and self._seam is not None:
            return self.commit()
        if pick.kind != "face":
            return ToolResult.info("")
        part = self.project.part
        if part is None or pick.id not in part.faces:
            return ToolResult.error(t("that surface is not part of the loaded model"))

        if self._seam is not None or not self.face_a:
            # Starting a new joint: either the first click, or a new pair after
            # a seam was computed but not accepted.
            self.face_a, self.face_b, self._seam = pick.id, "", None
            return ToolResult.info(t("first surface: {face}", face=pick.id), changed=False)
        if pick.id == self.face_a:
            return ToolResult.error(t("that is the same surface; pick the one it is welded to"))

        self.face_b = pick.id
        try:
            self._seam = self._derive()
        except ToolError as exc:
            self.face_b = ""
            return ToolResult.error(str(exc))
        groove = float(np.rad2deg(self._seam.mean_dihedral))
        gap = f", gap {self._seam.gap_mm:.1f} mm" if self._seam.gap_mm > 0.05 else ""
        return ToolResult.info(
            t(
                "{joint}: {length:.0f} mm, groove {groove:.0f} deg{gap}. Enter to accept.",
                joint=self._seam.joint_type.value,
                length=self._seam.length_mm,
                groove=groove,
                gap=gap,
            ),
            changed=False,
        )

    def on_command(self, command: str) -> ToolResult:
        if command == "accept":
            return self.commit()
        if command in ("cancel", "undo"):
            had = bool(self.face_a)
            self.face_a, self.face_b, self._seam = "", "", None
            return ToolResult.info(t("tracing cancelled") if had else "", finished=had)
        return ToolResult.info("")

    def commit(self) -> ToolResult:
        if self._seam is None:
            return ToolResult.error(t("pick the two surfaces of the joint first"))
        seam = self._seam
        self.project.add_manual_seam(seam)
        self.face_a, self.face_b, self._seam = "", "", None
        return ToolResult.info(
            t(
                "weld added: {joint}, {length:.0f} mm, groove {groove:.0f} deg",
                joint=seam.joint_type.value,
                length=seam.length_mm,
                groove=float(np.rad2deg(seam.mean_dihedral)),
            ),
            changed=True,
            seam_id=seam.id,
            finished=True,
        )

    # -- deriving the seam ---------------------------------------------------

    def _derive(self) -> Seam:
        part = self.project.part
        if part is None or not part.brep.available:
            raise ToolError(
                t("this needs the exact geometry, which this part does not carry (mesh import)")
            )
        a, b = part.faces[self.face_a], part.faces[self.face_b]
        if a.solid_id == b.solid_id:
            return self._from_shared_edge()
        return self._from_contact()

    def _from_shared_edge(self) -> Seam:
        """Same body: the seam is the edge the two faces share."""
        part = self.project.part
        topo = part.brep.topology
        keys = {v: k for k, v in topo.face_ids.items()}
        ka, kb = keys.get(self.face_a), keys.get(self.face_b)
        shared = [
            topo.edge_ids[ek]
            for ek, faces in topo.edge_faces.items()
            if ka in faces
            and kb in faces
            and ek in topo.edge_ids
            and topo.edge_ids[ek] in part.edges
        ]
        if not shared:
            raise ToolError(
                t("these two surfaces do not meet along an edge on this body")
            )
        # Reuse the edge tracer's geometry: it already knows how to chain
        # the shared edges and measure the groove at them.
        tracer = TraceSeamTool(self.project, self.joint_type)
        tracer.chain_tolerance_mm = 5.0
        for edge_id in shared:
            result = tracer._add_edge(edge_id)
            if not result.ok and edge_id not in tracer.state.edge_ids:
                # An edge that does not continue the chain belongs to a second
                # run of the same joint (a face wrapping round a hole, say);
                # the first run is the seam the operator pointed at.
                break
        return tracer._build_seam()

    def _from_contact(self) -> Seam:
        """Different bodies: the seam is where these two faces touch."""
        from ..seams import detect_contact_seams

        part = self.project.part
        picked = {self.face_a, self.face_b}
        # The welder clicks the two surfaces that form the angle -- the
        # receiving plate and the flank rising from it. Those are exactly the
        # faces a contact seam records as its flanks, so the match is made on
        # the seams the detector would produce, not on the hidden landing face
        # underneath the body.
        candidates = detect_contact_seams(part)
        exact = [s for s in candidates if {s.face_a_id, s.face_b_id} == picked]
        if not exact:
            # Relaxed: one flank is the picked face, the other picked face is on
            # the other body -- the operator clicked the landing face itself.
            solids = {part.faces[f].solid_id for f in picked}
            exact = [
                s
                for s in candidates
                if {s.face_a_id, s.face_b_id} & picked
                and {s.solid_a_id, s.solid_b_id} == solids
            ]
        if not exact:
            raise ToolError(
                t(
                    "these two surfaces do not touch (or the gap is wider than the "
                    "contact tolerance). Bring the bodies together in the CAD, or "
                    "trace along edges."
                )
            )
        seam = max(exact, key=lambda s: s.length_mm)
        if self.joint_type is not None:
            seam.joint_type = self.joint_type
        seam.confidence = 1.0
        seam.label = (
            f"manual - {seam.joint_type.value} {seam.length_mm:.0f} mm "
            f"({seam.dominant_position.value})"
        )
        return seam


# ---------------------------------------------------------------------------
# moving the cell around
# ---------------------------------------------------------------------------


class MoveCellTool(Tool):
    """Drag the bench, a clamp or an obstacle to where it really is.

    Why dragging as well as the numbers. Both are how a cell gets described,
    and they are good at different things. Typing 1250 mm is how an operator
    records a tape measurement; dragging is how they find out that the bench
    is too close to the robot column, which is a judgement made by looking.
    The panel and the viewport edit the same cell description, so a drag moves
    the numbers and a typed number moves the model.

    Everything moves in the horizontal plane it was grabbed at. A bench that
    can be dragged into the floor, or a clamp that comes off its table, is a
    cell that no longer describes anything -- and height is exactly the
    quantity an operator has measured and wants left alone. Z stays in the
    panel, where it is a number.
    """

    name = "move"
    title = "Move cell elements"
    pick_layers = ("cell", "clamps")

    #: Grab targets, as the scene labels them.
    BENCH_KEYS = ("table", "bench:bench_model")

    def __init__(self, project: Project) -> None:
        super().__init__(project)
        self.target: str = ""
        self.label: str = ""
        self._grab = np.zeros(3)
        self._offset = np.zeros(3)
        self._plane_z: float | None = None

    # -- prompting ----------------------------------------------------------

    def prompt(self) -> str:
        if not self.target:
            return t("Drag the bench, a clamp or an obstacle. Esc leaves it alone.")
        return t("moving {what}: release to drop it, Esc to put it back", what=self.label)

    # -- grabbing -----------------------------------------------------------

    def on_pick(self, pick: PickResult) -> ToolResult:
        if pick.kind != "cell" or not pick.id:
            self.target = ""
            return ToolResult.info("")

        label = self._describe(pick.id)
        if label is None:
            return ToolResult.error(
                t(
                    "{key} is not something that can be moved. Drag the bench, "
                    "a clamp or an obstacle.",
                    key=pick.id,
                )
            )
        self.target = pick.id
        self.label = label
        self._grab = np.asarray(pick.point, dtype=float).copy()
        self._offset = np.zeros(3)
        self._plane_z = float(self._grab[2])
        return ToolResult.info(t("holding {what}", what=label), changed=False)

    def _describe(self, key: str) -> str | None:
        """A name for what was grabbed, or ``None`` when it cannot be moved."""
        if key in self.BENCH_KEYS:
            return t("the bench")
        kind, _, identifier = key.partition(":")
        if kind == "clamp":
            clamp = self._clamp(identifier)
            return t("clamp {name}", name=clamp.get("name") or identifier) if clamp else None
        if self._obstacle(identifier) is not None:
            obstacle = self._obstacle(identifier) or {}
            return f"{obstacle.get('name') or identifier}"
        return None

    # -- dragging -----------------------------------------------------------

    def dragging(self) -> bool:
        return bool(self.target)

    def busy(self) -> bool:
        return bool(self.target)

    def drag_through(self) -> np.ndarray | None:
        return self._grab.copy() if self.target else None

    def drag_plane_z(self) -> float | None:
        return self._plane_z

    def on_drag(self, point: np.ndarray) -> ToolResult:
        if not self.target:
            return ToolResult.info("")
        offset = np.asarray(point, dtype=float) - self._grab
        offset[2] = 0.0  # the drag plane is horizontal; height stays a number
        self._offset = offset
        return ToolResult.info(
            t("{what}: {dx:+.0f}, {dy:+.0f} mm", what=self.label, dx=offset[0], dy=offset[1]),
            changed=False,
        )

    def drag_preview(self) -> tuple[str, np.ndarray] | None:
        if not self.target:
            return None
        return self.target, self._offset.copy()

    def on_drop(self) -> ToolResult:
        """Write the move into the cell description."""
        if not self.target:
            return ToolResult.info("")
        offset = self._offset.copy()
        label = self.label
        moved = self._apply(self.target, offset)
        self.target = ""
        self._offset = np.zeros(3)
        self._plane_z = None
        if not moved:
            return ToolResult.info("")
        if float(np.linalg.norm(offset[:2])) < 0.5:
            # A click that did not really move anything is a click, not an
            # edit: reporting a change would dirty the project for nothing.
            return ToolResult.info(t("{what} left where it was", what=label))
        return ToolResult.info(
            t("{what} moved {dx:+.0f}, {dy:+.0f} mm", what=label, dx=offset[0], dy=offset[1]),
            changed=True,
        )

    def on_command(self, command: str) -> ToolResult:
        if command == "cancel":
            held = bool(self.target)
            self.target = ""
            self._offset = np.zeros(3)
            self._plane_z = None
            return ToolResult.info(t("put back") if held else "", changed=held)
        return ToolResult.info("")

    # -- writing it down ----------------------------------------------------

    def _apply(self, key: str, offset: np.ndarray) -> bool:
        config = self.project.cell_config
        if not config:
            return False
        if key in self.BENCH_KEYS:
            fixture = dict(config.get("fixture", {}) or {})
            origin = list(fixture.get("table_origin", [0.0, 0.0, 0.0]))
            while len(origin) < 3:
                origin.append(0.0)
            fixture["table_origin"] = [
                float(origin[0] + offset[0]),
                float(origin[1] + offset[1]),
                float(origin[2]),
            ]
            config["fixture"] = fixture
            self.project.touch()
            return True

        kind, _, identifier = key.partition(":")
        entry = self._clamp(identifier) if kind == "clamp" else self._obstacle(identifier)
        if entry is None:
            return False
        field_name = "position" if kind == "clamp" else "centre"
        current = list(entry.get(field_name, [0.0, 0.0, 0.0]))
        while len(current) < 3:
            current.append(0.0)
        entry[field_name] = [
            float(current[0] + offset[0]),
            float(current[1] + offset[1]),
            float(current[2]),
        ]
        self.project.touch()
        return True

    def _clamp(self, identifier: str) -> dict[str, Any] | None:
        fixture = (self.project.cell_config or {}).get("fixture", {}) or {}
        for index, clamp in enumerate(fixture.get("clamps", []) or []):
            if str(clamp.get("id", f"clamp_{index}")) == identifier:
                return clamp
        return None

    def _obstacle(self, identifier: str) -> dict[str, Any] | None:
        for index, obstacle in enumerate((self.project.cell_config or {}).get("obstacles") or []):
            if str(obstacle.get("id", f"obstacle_{index}")) == identifier:
                return obstacle
        return None


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

    def seat_on_bench(self) -> ToolResult:
        """Put the part in the middle of the bench, resting on it.

        A part arrives in its CAD coordinates, which puts it at the cell
        origin -- inside the robot column. ``drop_to_table`` deliberately keeps
        the plan position, so on a fresh import it left the part off the
        bench, and the first thing every operator did was spend a minute
        dragging it into view. This is what a freshly imported part should
        do on its own: centre over the bench, rest on its surface, heading
        unchanged.
        """
        part, cell = self.project.part, self.project.cell
        if part is None or cell is None:
            return ToolResult.error("load a part and a cell first")
        if part.mesh_vertices is None or not len(part.mesh_vertices):
            return ToolResult.error("the part has no mesh to rest on the bench")
        pose = self.project.placement.copy()
        placed = tf.apply_points(pose, part.mesh_vertices)
        lo, hi = placed.min(axis=0), placed.max(axis=0)
        origin = np.asarray(cell.fixture.table_origin, dtype=float)
        pose[0, 3] += float(origin[0] - (lo[0] + hi[0]) / 2.0)
        pose[1, 3] += float(origin[1] - (lo[1] + hi[1]) / 2.0)
        pose[2, 3] += float(cell.fixture.table_z_mm - lo[2])
        return self._apply(pose, "seated on the bench")

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
