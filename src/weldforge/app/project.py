"""The project document, and the file it saves to.

"Every program must be savable" is the requirement this module exists for, and
it drives three decisions.

**One file, not a folder.** A ``.weldforge`` file is a ZIP. It gets emailed,
dropped in a job folder on a server, and opened again two years later by
someone who was not there. A folder of loose files loses a piece every time it
is moved.

**The CAD goes inside it.** The original STEP is embedded, not referenced.
A project that points at ``\\\\server\\drawings\\rev3.step`` stops working the
day that path changes or the drawing is revised -- and silently, because the
file still opens, just with different geometry. Embedding costs a megabyte or
two and makes the project an actual record of what was welded.

**The content is JSON.** Readable, diffable, and forward-tolerant: an older
build opening a newer project reports exactly which fields it did not
understand instead of failing or, worse, dropping them on the next save.

What is saved, and what is not
------------------------------
Saved: the part, the cell, where the part sits, every seam -- including the
ones traced by hand -- which are selected, the welding procedure for each with
any hand edits, the positioner settings, and the generated programs with a
record of where they were sent.

The cell goes inside it too, for the same reason the CAD does. A project that
names its cell and looks it up in a folder opens on the machine PC with
whatever bench happens to be installed there -- silently, because it still
opens. So the bench someone measured, and the 3-D model of it they imported,
travel with the job.

Not saved: the planned trajectories. They are large, they are derived, and
they are only valid for the robot calibration of the day they were made.
Re-planning takes seconds and is the safer default; a stale trajectory that
loads without complaint is a trap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..geom import transform as tf
from ..model import (
    JointType,
    Part,
    Seam,
    Weave,
    WeaveShape,
    WeldCell,
    WeldJob,
    WeldPass,
    WeldPosition,
    WeldProcedure,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FORMAT_VERSION",
    "PROJECT_SUFFIX",
    "ProjectError",
    "SentRecord",
    "Project",
    "save_project",
    "load_project",
    "RecentFiles",
    "attach_bench",
    "clear_bench",
    "cell_for",
    "materialise_bench",
    "attach_obstacle",
    "remove_obstacle",
]

#: Bumped when the on-disk shape changes incompatibly. Readers refuse a
#: higher major version rather than guessing.
# 2 added setup.cell_config and the embedded bench model, so a cell the
# operator measured and a bench they imported travel with the project instead
# of being looked up by name in a folder that may not be on the next machine.
FORMAT_VERSION = 2
PROJECT_SUFFIX = ".weldforge"

_PART_NAME = "part"
_BENCH_NAME = "bench"
_OBSTACLE_DIR = "obstacles"
_PROJECT_JSON = "project.json"
_PROGRAM_DIR = "programs"


class ProjectError(RuntimeError):
    """The project file is unreadable, or from an incompatible version."""


# ---------------------------------------------------------------------------
# serialisation helpers
# ---------------------------------------------------------------------------


def _pose_to_json(pose: np.ndarray) -> dict[str, Any]:
    """A pose as position and quaternion, not a 4x4 matrix.

    The matrix would round-trip too, but 16 numbers hide a non-rigid transform
    while position + quaternion cannot represent one at all.
    """
    return {
        "position_mm": [round(float(v), 4) for v in tf.translation_of(pose)],
        "quaternion_xyzw": [round(float(v), 8) for v in tf.matrix_to_quat(pose)],
    }


def _pose_from_json(data: dict[str, Any] | None) -> np.ndarray:
    if not data:
        return tf.identity()
    quaternion = np.asarray(data.get("quaternion_xyzw", [0, 0, 0, 1]), dtype=float)
    position = np.asarray(data.get("position_mm", [0, 0, 0]), dtype=float)
    return tf.pose(tf.quat_to_matrix(quaternion), position)


def _array_to_json(array: np.ndarray, digits: int = 4) -> list:
    if array is None or not len(array):
        return []
    return np.asarray(array, dtype=float).round(digits).tolist()


def _seam_to_json(seam: Seam) -> dict[str, Any]:
    return {
        "id": seam.id,
        "label": seam.label,
        "joint_type": seam.joint_type.value,
        "points_mm": _array_to_json(seam.points, 4),
        "tangents": _array_to_json(seam.tangents, 6),
        "normal_a": _array_to_json(seam.normal_a, 6),
        "normal_b": _array_to_json(seam.normal_b, 6),
        "bisector": _array_to_json(seam.bisector, 6),
        "groove_rad": _array_to_json(seam.dihedral, 6),
        "face_a_id": seam.face_a_id,
        "face_b_id": seam.face_b_id,
        "solid_a_id": seam.solid_a_id,
        "solid_b_id": seam.solid_b_id,
        "edge_ids": list(seam.edge_ids),
        "thickness_a_mm": seam.thickness_a_mm,
        "thickness_b_mm": seam.thickness_b_mm,
        "gap_mm": seam.gap_mm,
        "confidence": seam.confidence,
        "closed": seam.closed,
        "selected": seam.selected,
        "notes": list(seam.notes),
    }


def _seam_from_json(data: dict[str, Any]) -> Seam:
    def array(key: str, width: int = 3) -> np.ndarray:
        values = data.get(key) or []
        if not values:
            return np.zeros((0, width)) if width > 1 else np.zeros(0)
        return np.asarray(values, dtype=float)

    seam = Seam(
        id=str(data.get("id", "")),
        label=str(data.get("label", "")),
        joint_type=JointType(data.get("joint_type", "unknown")),
        points=array("points_mm"),
        tangents=array("tangents"),
        normal_a=array("normal_a"),
        normal_b=array("normal_b"),
        bisector=array("bisector"),
        dihedral=array("groove_rad", width=1),
        face_a_id=str(data.get("face_a_id", "")),
        face_b_id=str(data.get("face_b_id", "")),
        solid_a_id=str(data.get("solid_a_id", "")),
        solid_b_id=str(data.get("solid_b_id", "")),
        edge_ids=tuple(data.get("edge_ids", [])),
        thickness_a_mm=float(data.get("thickness_a_mm", 0.0)),
        thickness_b_mm=float(data.get("thickness_b_mm", 0.0)),
        gap_mm=float(data.get("gap_mm", 0.0)),
        confidence=float(data.get("confidence", 0.0)),
        closed=bool(data.get("closed", False)),
        selected=bool(data.get("selected", True)),
        notes=list(data.get("notes", [])),
    )
    return seam


def _pass_to_json(weld_pass: WeldPass) -> dict[str, Any]:
    return {
        "index": weld_pass.index,
        "name": weld_pass.name,
        "leg_mm": weld_pass.leg_mm,
        "throat_mm": weld_pass.throat_mm,
        "offset_across_mm": weld_pass.offset_across_mm,
        "offset_normal_mm": weld_pass.offset_normal_mm,
        "process": weld_pass.process.value,
        "transfer_mode": weld_pass.transfer_mode.value,
        "current_a": weld_pass.current_a,
        "voltage_v": weld_pass.voltage_v,
        "wire_speed_m_min": weld_pass.wire_speed_m_min,
        "wire_diameter_mm": weld_pass.wire_diameter_mm,
        "polarity": weld_pass.polarity,
        "gas": weld_pass.gas,
        "gas_flow_l_min": weld_pass.gas_flow_l_min,
        "travel_speed_mm_s": weld_pass.travel_speed_mm_s,
        "work_angle_deg": weld_pass.work_angle_deg,
        "travel_angle_deg": weld_pass.travel_angle_deg,
        "stickout_mm": weld_pass.stickout_mm,
        "weave": {
            "shape": weld_pass.weave.shape.value,
            "amplitude_mm": weld_pass.weave.amplitude_mm,
            "frequency_hz": weld_pass.weave.frequency_hz,
            "dwell_ms": weld_pass.weave.dwell_ms,
        },
        "arc_start_delay_s": weld_pass.arc_start_delay_s,
        "crater_fill_s": weld_pass.crater_fill_s,
        "burnback_s": weld_pass.burnback_s,
    }


def _pass_from_json(data: dict[str, Any]) -> WeldPass:
    from ..model import Process, TransferMode

    weave = data.get("weave") or {}
    return WeldPass(
        index=int(data.get("index", 0)),
        name=str(data.get("name", "")),
        leg_mm=float(data.get("leg_mm", 0.0)),
        throat_mm=float(data.get("throat_mm", 0.0)),
        offset_across_mm=float(data.get("offset_across_mm", 0.0)),
        offset_normal_mm=float(data.get("offset_normal_mm", 0.0)),
        process=Process(data.get("process", "GMAW")),
        transfer_mode=TransferMode(data.get("transfer_mode", "spray")),
        current_a=float(data.get("current_a", 0.0)),
        voltage_v=float(data.get("voltage_v", 0.0)),
        wire_speed_m_min=float(data.get("wire_speed_m_min", 0.0)),
        wire_diameter_mm=float(data.get("wire_diameter_mm", 1.2)),
        polarity=str(data.get("polarity", "DCEP")),
        gas=str(data.get("gas", "")),
        gas_flow_l_min=float(data.get("gas_flow_l_min", 15.0)),
        travel_speed_mm_s=float(data.get("travel_speed_mm_s", 0.0)),
        work_angle_deg=float(data.get("work_angle_deg", 45.0)),
        travel_angle_deg=float(data.get("travel_angle_deg", -10.0)),
        stickout_mm=float(data.get("stickout_mm", 15.0)),
        weave=Weave(
            shape=WeaveShape(weave.get("shape", "none")),
            amplitude_mm=float(weave.get("amplitude_mm", 0.0)),
            frequency_hz=float(weave.get("frequency_hz", 0.0)),
            dwell_ms=float(weave.get("dwell_ms", 0.0)),
        ),
        arc_start_delay_s=float(data.get("arc_start_delay_s", 0.2)),
        crater_fill_s=float(data.get("crater_fill_s", 0.3)),
        burnback_s=float(data.get("burnback_s", 0.08)),
    )


def _procedure_to_json(procedure: WeldProcedure) -> dict[str, Any]:
    return {
        "wps_id": procedure.wps_id,
        "seam_id": procedure.seam_id,
        "process": procedure.process.value,
        "base_material": procedure.base_material,
        "filler": procedure.filler,
        "position": procedure.position.value,
        "joint_type": procedure.joint_type.value,
        "thickness_mm": procedure.thickness_mm,
        "target_leg_mm": procedure.target_leg_mm,
        "preheat_c": procedure.preheat_c,
        "interpass_max_c": procedure.interpass_max_c,
        "heat_input_max_kj_mm": procedure.heat_input_max_kj_mm,
        "standard": procedure.standard,
        "passes": [_pass_to_json(p) for p in procedure.passes],
        "warnings": list(procedure.warnings),
    }


def _procedure_from_json(data: dict[str, Any]) -> WeldProcedure:
    from ..model import Process

    return WeldProcedure(
        wps_id=str(data.get("wps_id", "")),
        seam_id=str(data.get("seam_id", "")),
        process=Process(data.get("process", "GMAW")),
        base_material=str(data.get("base_material", "S355")),
        filler=str(data.get("filler", "")),
        position=WeldPosition(data.get("position", "PB")),
        joint_type=JointType(data.get("joint_type", "unknown")),
        thickness_mm=float(data.get("thickness_mm", 0.0)),
        target_leg_mm=float(data.get("target_leg_mm", 0.0)),
        preheat_c=float(data.get("preheat_c", 20.0)),
        interpass_max_c=float(data.get("interpass_max_c", 250.0)),
        heat_input_max_kj_mm=float(data.get("heat_input_max_kj_mm", 0.0)),
        standard=str(data.get("standard", "EN ISO 15614-1")),
        passes=[_pass_from_json(p) for p in data.get("passes", [])],
        warnings=list(data.get("warnings", [])),
    )


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


@dataclass
class SentRecord:
    """One dispatch of a program, kept so there is a record of what went out."""

    when: str = ""
    target: str = ""  # postprocessor key
    destination: str = ""  # folder, ftp host, or cell URL
    method: str = "file"  # file | ftp | cell
    files: list[str] = field(default_factory=list)
    ok: bool = True
    message: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "when": self.when,
            "target": self.target,
            "destination": self.destination,
            "method": self.method,
            "files": list(self.files),
            "ok": self.ok,
            "message": self.message,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SentRecord:
        return cls(
            when=str(data.get("when", "")),
            target=str(data.get("target", "")),
            destination=str(data.get("destination", "")),
            method=str(data.get("method", "file")),
            files=list(data.get("files", [])),
            ok=bool(data.get("ok", True)),
            message=str(data.get("message", "")),
        )


@dataclass
class Project:
    """Everything the operator set up, and everything that came out of it."""

    name: str = "Untitled"
    path: Path | None = None
    created: str = ""
    modified: str = ""
    notes: str = ""

    # --- inputs -----------------------------------------------------------
    part: Part | None = None
    #: The CAD bytes, embedded so the project does not depend on a path.
    part_bytes: bytes | None = None
    part_filename: str = ""
    cell_name: str = "bench_cell"
    cell: WeldCell | None = None
    #: The cell's own description, in the dictionary form the YAML holds. This
    #: is what the cell editor changes and what the project file carries, so a
    #: bench measured on this shop floor survives being opened anywhere else.
    #: ``None`` means "whatever ``cell_name`` resolves to", which is how every
    #: project written before this existed behaves.
    cell_config: dict[str, Any] | None = None
    #: The imported bench model, embedded for the same reason the CAD is: a
    #: project that opens without its bench is a project that looks like it
    #: worked.
    bench_bytes: bytes | None = None
    bench_filename: str = ""
    #: Obstacles imported as 3-D models -- a welding jig, a positioner
    #: housing -- keyed by obstacle id, embedded for the same reason the bench
    #: is. The matching entry in ``cell_config["obstacles"]`` names the file.
    obstacle_models: dict[str, bytes] = field(default_factory=dict)
    obstacle_filenames: dict[str, str] = field(default_factory=dict)
    base_material: str = "S355"
    machine_profile: str = "generic_analog"
    wire_diameter_mm: float | None = None
    leg_thickness_ratio: float = 0.7

    # --- the setup the operator made --------------------------------------
    #: Part pose in the cell. Set by the placement search, by snapping to the
    #: jig, or by dragging it.
    placement: np.ndarray = field(default_factory=tf.identity)
    placement_locked: bool = False
    seams: list[Seam] = field(default_factory=list)
    #: Seam ids traced by hand, kept apart so re-running detection does not
    #: silently delete the operator's own work.
    manual_seam_ids: set[str] = field(default_factory=set)
    procedures: dict[str, WeldProcedure] = field(default_factory=dict)
    #: Seam ids whose procedure was edited by hand; regeneration leaves them.
    edited_procedure_ids: set[str] = field(default_factory=set)
    positioner_values: dict[str, dict[str, float]] = field(default_factory=dict)

    # --- outputs ----------------------------------------------------------
    programs: dict[str, dict[str, str]] = field(default_factory=dict)
    sent: list[SentRecord] = field(default_factory=list)
    last_report: dict[str, Any] = field(default_factory=dict)

    # --- transient --------------------------------------------------------
    #: Not saved: large, derived, and only valid for the day's calibration.
    job: WeldJob | None = field(default=None, repr=False)
    dirty: bool = False
    unknown_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- convenience --------------------------------------------------------

    @property
    def display_name(self) -> str:
        return self.path.stem if self.path is not None else self.name

    @property
    def selected_seams(self) -> list[Seam]:
        return [s for s in self.seams if s.selected]

    @property
    def total_weld_length_mm(self) -> float:
        return sum(s.length_mm for s in self.selected_seams)

    def seam(self, seam_id: str) -> Seam | None:
        return next((s for s in self.seams if s.id == seam_id), None)

    def is_manual(self, seam_id: str) -> bool:
        return seam_id in self.manual_seam_ids

    def touch(self) -> None:
        self.modified = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.dirty = True

    def replace_detected_seams(self, seams: list[Seam]) -> int:
        """Swap in freshly detected seams, keeping hand-traced ones.

        Re-running detection after moving the part must not throw away a weld
        line the operator drew: that is their work, and losing it silently is
        the kind of thing that stops an application being used.
        """
        manual = [s for s in self.seams if s.id in self.manual_seam_ids]
        self.seams = [*seams, *manual]
        self.touch()
        return len(manual)

    def add_manual_seam(self, seam: Seam) -> Seam:
        seam.notes = [*seam.notes, "traced by hand"]
        self.seams.append(seam)
        self.manual_seam_ids.add(seam.id)
        self.touch()
        return seam

    def remove_seam(self, seam_id: str) -> bool:
        before = len(self.seams)
        self.seams = [s for s in self.seams if s.id != seam_id]
        self.manual_seam_ids.discard(seam_id)
        self.procedures.pop(seam_id, None)
        self.edited_procedure_ids.discard(seam_id)
        self.positioner_values.pop(seam_id, None)
        if len(self.seams) != before:
            self.touch()
            return True
        return False

    def mark_procedure_edited(self, seam_id: str) -> None:
        self.edited_procedure_ids.add(seam_id)
        self.touch()

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "path": str(self.path) if self.path else None,
            "modified": self.modified,
            "part": self.part.name if self.part else None,
            "cell": self.cell_name,
            "material": self.base_material,
            "seams": len(self.seams),
            "selected": len(self.selected_seams),
            "manual": len(self.manual_seam_ids),
            "weld_length_mm": round(self.total_weld_length_mm, 1),
            "procedures": len(self.procedures),
            "edited_procedures": len(self.edited_procedure_ids),
            "programs": sorted(self.programs),
            "dispatches": len(self.sent),
            "dirty": self.dirty,
        }

    # -- JSON ---------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "generator": "WeldForge Studio",
            "name": self.name,
            "created": self.created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "modified": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": self.notes,
            "setup": {
                "cell": self.cell_name,
                "cell_config": _clean_cell_config(self.cell_config),
                "bench_filename": self.bench_filename,
                "obstacle_filenames": dict(self.obstacle_filenames),
                "base_material": self.base_material,
                "machine_profile": self.machine_profile,
                "wire_diameter_mm": self.wire_diameter_mm,
                "leg_thickness_ratio": self.leg_thickness_ratio,
                "placement": _pose_to_json(self.placement),
                "placement_locked": self.placement_locked,
            },
            "part": {
                "filename": self.part_filename,
                "name": self.part.name if self.part else "",
                "embedded": self.part_bytes is not None,
            },
            "seams": [_seam_to_json(s) for s in self.seams],
            "manual_seam_ids": sorted(self.manual_seam_ids),
            "procedures": [_procedure_to_json(p) for p in self.procedures.values()],
            "edited_procedure_ids": sorted(self.edited_procedure_ids),
            "positioner_values": {
                seam_id: {name: float(v) for name, v in values.items()}
                for seam_id, values in self.positioner_values.items()
            },
            "programs": {target: dict(files) for target, files in self.programs.items()},
            "sent": [record.to_json() for record in self.sent],
            "last_report": self.last_report,
            **self.unknown_fields,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Project:
        version = int(data.get("format", 0))
        if version > FORMAT_VERSION:
            raise ProjectError(
                f"this project was written by a newer version of WeldForge "
                f"(format {version}, this build reads {FORMAT_VERSION}). "
                "Update WeldForge rather than opening it read-only: saving it "
                "here would drop whatever the newer version added."
            )

        setup = data.get("setup") or {}
        known = {
            "format",
            "generator",
            "name",
            "created",
            "modified",
            "notes",
            "setup",
            "part",
            "seams",
            "manual_seam_ids",
            "procedures",
            "edited_procedure_ids",
            "positioner_values",
            "programs",
            "sent",
            "last_report",
        }

        project = cls(
            name=str(data.get("name", "Untitled")),
            created=str(data.get("created", "")),
            modified=str(data.get("modified", "")),
            notes=str(data.get("notes", "")),
            part_filename=str((data.get("part") or {}).get("filename", "")),
            cell_name=str(setup.get("cell", "bench_cell")),
            cell_config=_clean_cell_config(setup.get("cell_config")),
            bench_filename=str(setup.get("bench_filename", "")),
            obstacle_filenames={
                str(k): str(v) for k, v in (setup.get("obstacle_filenames") or {}).items()
            },
            base_material=str(setup.get("base_material", "S355")),
            machine_profile=str(setup.get("machine_profile", "generic_analog")),
            wire_diameter_mm=(
                float(setup["wire_diameter_mm"])
                if setup.get("wire_diameter_mm") is not None
                else None
            ),
            leg_thickness_ratio=float(setup.get("leg_thickness_ratio", 0.7)),
            placement=_pose_from_json(setup.get("placement")),
            placement_locked=bool(setup.get("placement_locked", False)),
            seams=[_seam_from_json(s) for s in data.get("seams", [])],
            manual_seam_ids=set(data.get("manual_seam_ids", [])),
            edited_procedure_ids=set(data.get("edited_procedure_ids", [])),
            positioner_values={
                seam_id: {name: float(v) for name, v in values.items()}
                for seam_id, values in (data.get("positioner_values") or {}).items()
            },
            programs={
                target: dict(files) for target, files in (data.get("programs") or {}).items()
            },
            sent=[SentRecord.from_json(r) for r in data.get("sent", [])],
            last_report=dict(data.get("last_report") or {}),
        )
        project.procedures = {
            p["seam_id"]: _procedure_from_json(p) for p in data.get("procedures", [])
        }
        # Anything this build does not understand is carried through to the
        # next save, so opening a project in an older version and saving it
        # does not quietly strip what a newer one wrote.
        project.unknown_fields = {k: v for k, v in data.items() if k not in known}
        if project.unknown_fields:
            logger.info(
                "project contains %d field(s) this build does not know: %s (preserved on save)",
                len(project.unknown_fields),
                sorted(project.unknown_fields),
            )
        return project


# ---------------------------------------------------------------------------
# file I/O
# ---------------------------------------------------------------------------


def _clean_cell_config(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """A cell description without the bookkeeping keys.

    ``read_cell_config`` records the folder it came from under ``_dir`` so a
    relative bench model resolves. That is true of the machine it was read on
    and nowhere else, so it must not be written into a project.
    """
    if not raw:
        return None
    return {key: value for key, value in raw.items() if not str(key).startswith("_")}


def cell_for(project: Project) -> WeldCell:
    """Build the project's cell: its own description, or the named one.

    The embedded description wins. A project that carries a bench someone
    measured must not quietly fall back to a shipped example because the
    folder of cells looks different on this machine.
    """
    from ..fixture import cell_from_config, load_cell

    if not project.cell_config:
        return load_cell(project.cell_name)

    raw = dict(project.cell_config)
    bench = materialise_bench(project)
    if bench is not None:
        raw.setdefault("fixture", {})
        raw["fixture"] = {**raw["fixture"], "model": str(bench)}
    if project.obstacle_models:
        # Same treatment for imported obstacles: the bytes live in the
        # project, the cell builder wants a path.
        obstacles = []
        for entry in raw.get("obstacles", []) or []:
            entry = dict(entry)
            path = _materialise_obstacle(project, str(entry.get("id", "")))
            if path is not None:
                entry["model"] = str(path)
            obstacles.append(entry)
        raw["obstacles"] = obstacles
    return cell_from_config(raw, project.cell_name)


def materialise_bench(project: Project) -> Path | None:
    """Put the embedded bench model somewhere ``cell_from_config`` can read it.

    Cached under the user's config folder rather than a temporary file,
    because the cell is rebuilt every time the operator nudges a number and
    re-reading a 40 MB STEP for each keystroke is not acceptable.
    """
    if project.bench_bytes is None:
        return None
    suffix = Path(project.bench_filename or "bench.stl").suffix or ".stl"
    folder = Path.home() / ".config" / "weldforge" / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(project.bench_bytes).hexdigest()[:16]
    target = folder / f"bench_{digest}{suffix}"
    if not target.exists():
        target.write_bytes(project.bench_bytes)
    return target


def attach_bench(project: Project, model_path: str | Path) -> dict[str, Any]:
    """Import a 3-D model as the project's bench, and fit the box to it.

    The box dimensions are what placement, collision checking and the URDF
    export measure against, so they are derived from the model rather than
    left for the operator to re-type: the top of the model is the support
    plane, and its footprint is the area a part may sit on.
    """
    from ..fixture import bench_model_from_file, default_cell_config

    source = Path(model_path)
    vertices, _ = bench_model_from_file(source)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)

    raw = dict(project.cell_config or default_cell_config())
    fixture = dict(raw.get("fixture", {}) or {})
    fixture["model"] = source.name
    fixture["table_z_mm"] = float(hi[2])
    fixture["table_size_mm"] = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
    fixture["table_origin"] = [float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0), 0.0]
    if not fixture.get("name"):
        fixture["name"] = source.stem
    raw["fixture"] = fixture

    project.cell_config = raw
    project.bench_bytes = source.read_bytes()
    project.bench_filename = source.name
    project.touch()
    logger.info(
        "bench %s: %.0f x %.0f mm, top at %.0f mm",
        source.name,
        fixture["table_size_mm"][0],
        fixture["table_size_mm"][1],
        fixture["table_z_mm"],
    )
    return raw


def clear_bench(project: Project) -> dict[str, Any] | None:
    """Drop the imported bench model, keeping the numbers that describe it."""
    if project.cell_config:
        fixture = dict(project.cell_config.get("fixture", {}) or {})
        fixture.pop("model", None)
        project.cell_config = {**project.cell_config, "fixture": fixture}
    project.bench_bytes = None
    project.bench_filename = ""
    project.touch()
    return project.cell_config


def _materialise_obstacle(project: Project, obstacle_id: str) -> Path | None:
    """Put one embedded obstacle model where ``cell_from_config`` can read it."""
    payload = project.obstacle_models.get(obstacle_id)
    if payload is None:
        return None
    suffix = Path(project.obstacle_filenames.get(obstacle_id, "x.stl")).suffix or ".stl"
    folder = Path.home() / ".config" / "weldforge" / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"obstacle_{hashlib.sha256(payload).hexdigest()[:16]}{suffix}"
    if not target.exists():
        target.write_bytes(payload)
    return target


def attach_obstacle(
    project: Project, model_path: str | Path, name: str = "", kind: str = "jig"
) -> dict[str, Any]:
    """Import a 3-D model as an obstacle in the cell -- a welding jig, say.

    It lands with its footprint centred on the bench and its feet on the
    floor, where it can be seen and dragged to where it really stands. It is
    an obstacle in the full sense: the collision checker and the 3-D view
    both read ``all_obstacles()``, so the torch is kept out of it from the
    moment it is imported.
    """
    from ..fixture import default_cell_config

    source = Path(model_path)
    payload = source.read_bytes()
    raw = dict(project.cell_config or default_cell_config())
    obstacles = [dict(o) for o in (raw.get("obstacles") or [])]

    stem = "".join(c if c.isalnum() else "_" for c in source.stem.lower()).strip("_") or "obstacle"
    used = {str(o.get("id", "")) for o in obstacles}
    obstacle_id, index = stem, 2
    while obstacle_id in used:
        obstacle_id, index = f"{stem}_{index}", index + 1

    fixture = raw.get("fixture", {}) or {}
    origin = list(fixture.get("table_origin", [1000.0, 0.0, 0.0]))
    obstacles.append(
        {
            "id": obstacle_id,
            "name": name or source.stem,
            "kind": kind,
            "model": source.name,
            "centre": [float(origin[0]), float(origin[1]) if len(origin) > 1 else 0.0, 0.0],
        }
    )
    raw["obstacles"] = obstacles
    project.cell_config = raw
    project.obstacle_models[obstacle_id] = payload
    project.obstacle_filenames[obstacle_id] = source.name
    project.touch()
    logger.info("obstacle %s imported from %s", obstacle_id, source.name)
    return obstacles[-1]


def remove_obstacle(project: Project, obstacle_id: str) -> bool:
    """Drop an obstacle, imported or not, from the cell description."""
    if not project.cell_config:
        return False
    before = list(project.cell_config.get("obstacles") or [])
    after = [o for o in before if str(o.get("id", "")) != obstacle_id]
    if len(after) == len(before):
        return False
    project.cell_config = {**project.cell_config, "obstacles": after}
    project.obstacle_models.pop(obstacle_id, None)
    project.obstacle_filenames.pop(obstacle_id, None)
    project.touch()
    return True


def save_project(project: Project, path: str | Path) -> Path:
    """Write the project to a ``.weldforge`` file.

    Written to a temporary file and moved into place, so an interrupted save
    cannot leave a half-written project where the good one was.
    """
    out = Path(path)
    if out.suffix != PROJECT_SUFFIX:
        out = out.with_suffix(PROJECT_SUFFIX)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = project.to_json()
    temporary = out.with_name(out.name + ".part")

    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_PROJECT_JSON, json.dumps(payload, indent=2))
        if project.part_bytes is not None:
            suffix = Path(project.part_filename or "part.step").suffix or ".step"
            archive.writestr(f"{_PART_NAME}{suffix}", project.part_bytes)
        if project.bench_bytes is not None:
            suffix = Path(project.bench_filename or "bench.stl").suffix or ".stl"
            archive.writestr(f"{_BENCH_NAME}{suffix}", project.bench_bytes)
        for obstacle_id, payload_bytes in project.obstacle_models.items():
            suffix = Path(project.obstacle_filenames.get(obstacle_id, "x.stl")).suffix or ".stl"
            archive.writestr(f"{_OBSTACLE_DIR}/{obstacle_id}{suffix}", payload_bytes)
        for target, files in project.programs.items():
            for filename, text in files.items():
                archive.writestr(f"{_PROGRAM_DIR}/{target}/{filename}", text)

    temporary.replace(out)
    project.path = out
    project.name = out.stem
    project.modified = payload["modified"]
    project.created = payload["created"]
    project.dirty = False
    logger.info("saved %s (%.1f kB)", out, out.stat().st_size / 1024.0)
    return out


def load_project(path: str | Path, load_cad: bool = True) -> Project:
    """Read a ``.weldforge`` file.

    When *load_cad* the embedded CAD is extracted to a temporary file and
    re-imported, which is what gives the project its exact geometry back --
    seam detection and the groove maths need the B-rep, not the saved
    polylines.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no project at {source}")

    try:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if _PROJECT_JSON not in names:
                raise ProjectError(f"{source.name} is not a WeldForge project: no {_PROJECT_JSON}")
            data = json.loads(archive.read(_PROJECT_JSON))
            project = Project.from_json(data)
            project.path = source
            project.name = source.stem

            cad_names = [n for n in names if n.startswith(_PART_NAME + ".")]
            if cad_names:
                project.part_bytes = archive.read(cad_names[0])

            bench_names = [n for n in names if n.startswith(_BENCH_NAME + ".")]
            if bench_names:
                project.bench_bytes = archive.read(bench_names[0])
                if not project.bench_filename:
                    project.bench_filename = bench_names[0]

            for name in names:
                if not name.startswith(_OBSTACLE_DIR + "/"):
                    continue
                obstacle_id = Path(name).stem
                project.obstacle_models[obstacle_id] = archive.read(name)
                project.obstacle_filenames.setdefault(obstacle_id, Path(name).name)

            for name in names:
                if not name.startswith(_PROGRAM_DIR + "/"):
                    continue
                _, target, filename = name.split("/", 2)
                project.programs.setdefault(target, {})[filename] = archive.read(name).decode(
                    "utf-8", "replace"
                )
    except zipfile.BadZipFile as exc:
        raise ProjectError(f"{source.name} is not readable as a project file") from exc

    if load_cad and project.part_bytes is not None:
        project.part = _reimport_part(project)

    project.dirty = False
    logger.info(
        "loaded %s: %d seams (%d by hand), %d procedures",
        source.name,
        len(project.seams),
        len(project.manual_seam_ids),
        len(project.procedures),
    )
    return project


def _reimport_part(project: Project) -> Part | None:
    """Re-import the embedded CAD so the exact geometry is available again."""
    from ..cad import load_part

    suffix = Path(project.part_filename or "part.step").suffix or ".step"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(project.part_bytes or b"")
        temporary = Path(handle.name)
    try:
        return load_part(temporary, name=project.part_filename or None)
    except Exception as exc:
        logger.error("could not re-import the embedded CAD: %s", exc)
        return None
    finally:
        temporary.unlink(missing_ok=True)


def attach_part(project: Project, cad_path: str | Path, part: Part | None = None) -> Part:
    """Attach a CAD file to the project, embedding its bytes."""
    from ..cad import load_part

    source = Path(cad_path)
    project.part_bytes = source.read_bytes()
    project.part_filename = source.name
    project.part = part if part is not None else load_part(source)
    if project.name in ("", "Untitled"):
        project.name = source.stem
    project.touch()
    return project.part


# ---------------------------------------------------------------------------
# recent files
# ---------------------------------------------------------------------------


class RecentFiles:
    """The recent-projects list, in a small JSON file next to the settings."""

    def __init__(self, path: Path | None = None, limit: int = 12) -> None:
        self.path = path or (Path.home() / ".config" / "weldforge" / "recent.json")
        self.limit = limit

    def load(self) -> list[Path]:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        # Drop entries that have gone: a recent list full of dead paths is
        # worse than a short one.
        return [Path(p) for p in raw.get("files", []) if Path(p).exists()]

    def add(self, project_path: str | Path) -> list[Path]:
        entry = Path(project_path).resolve()
        files = [p for p in self.load() if p.resolve() != entry]
        files.insert(0, entry)
        files = files[: self.limit]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"files": [str(p) for p in files]}, indent=2))
        return files

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
