"""The WeldForge domain model.

Every layer -- CAD import, seam recognition, WPS generation, fixturing,
kinematics, postprocessing, the REST API and the ROS 2 nodes -- speaks in the
types defined here. Keeping them in one module is deliberate: these objects
are the contract between the layers, and scattering them invites the kind of
drift (metres vs millimetres, degrees vs radians) that the previous
generations of this software suffered from.

Units follow :mod:`weldforge.units` without exception: **mm, rad, s, A, V**.
Fields that carry a different unit say so in their name
(``wire_speed_m_min``, ``work_angle_deg`` in serialised form, ...).

The classes are plain dataclasses holding NumPy arrays, which keeps the maths
fast and ergonomic. Serialisation to and from JSON lives in
:mod:`weldforge.schemas`, so nothing here depends on Pydantic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .geom import polyline as pl
from .geom import transform as tf

# ===========================================================================
# Enumerations -- the vocabulary
# ===========================================================================


class JointType(str, Enum):
    """Weld joint geometry, in the sense of ISO 9692-1.

    The classifier in :mod:`weldforge.seams` assigns one of these to every
    seam candidate; the WPS engine keys its tables off it.
    """

    FILLET = "fillet"  # two faces meeting at roughly 90 deg, T or L
    BUTT = "butt"  # coplanar faces, square/V/U prepared
    LAP = "lap"  # overlapping plates, weld along the top edge
    CORNER = "corner"  # outside corner, two faces forming an L outward
    EDGE = "edge"  # faces nearly parallel, welded along the edge
    TEE = "tee"  # stem landing on a face of another solid
    PLUG = "plug"  # hole/slot filled, treated as a closed path
    UNKNOWN = "unknown"

    @property
    def is_weldable(self) -> bool:
        return self is not JointType.UNKNOWN


class WeldPosition(str, Enum):
    """Welding position per EN ISO 6947.

    Position drives nearly every process parameter: the same fillet needs a
    different current, a different travel speed and sometimes a different
    transfer mode flat (PA) than vertical-up (PF).
    """

    PA = "PA"  # flat / downhand, fillet in the gutter
    PB = "PB"  # horizontal-vertical fillet (plate flat, web vertical)
    PC = "PC"  # horizontal butt on a vertical plate
    PD = "PD"  # overhead horizontal-vertical fillet
    PE = "PE"  # overhead
    PF = "PF"  # vertical up
    PG = "PG"  # vertical down

    @property
    def is_overhead(self) -> bool:
        return self in (WeldPosition.PD, WeldPosition.PE)

    @property
    def is_vertical(self) -> bool:
        return self in (WeldPosition.PF, WeldPosition.PG)

    @property
    def is_downhand(self) -> bool:
        """Positions a positioner should try to reach: easiest, fastest, best."""
        return self in (WeldPosition.PA, WeldPosition.PB)


class Process(str, Enum):
    """Welding process, with the ISO 4063 reference number."""

    GMAW = "GMAW"  # 131/135 MIG/MAG solid wire
    FCAW = "FCAW"  # 136/138 flux/metal cored
    GTAW = "GTAW"  # 141 TIG
    SMAW = "SMAW"  # 111 stick
    SAW = "SAW"  # 121 submerged arc
    LASER = "LASER"  # 52x

    @property
    def iso_4063(self) -> int:
        return {
            Process.GMAW: 135,
            Process.FCAW: 136,
            Process.GTAW: 141,
            Process.SMAW: 111,
            Process.SAW: 121,
            Process.LASER: 521,
        }[self]

    @property
    def default_efficiency(self) -> float:
        """Thermal efficiency for heat-input calculation (EN 1011-1)."""
        return {
            Process.GMAW: 0.8,
            Process.FCAW: 0.8,
            Process.GTAW: 0.6,
            Process.SMAW: 0.8,
            Process.SAW: 1.0,
            Process.LASER: 0.7,
        }[self]


class TransferMode(str, Enum):
    """Arc / metal transfer mode -- what the synergic job on the machine selects."""

    SHORT_ARC = "short_arc"
    GLOBULAR = "globular"
    SPRAY = "spray"
    PULSE = "pulse"
    DOUBLE_PULSE = "double_pulse"
    CMT = "cmt"  # cold metal transfer, thin sheet and root passes
    DC_CONSTANT = "dc_constant"  # GTAW/SMAW


class WeaveShape(str, Enum):
    NONE = "none"
    ZIGZAG = "zigzag"
    TRIANGULAR = "triangular"
    CIRCULAR = "circular"
    TROCHOIDAL = "trochoidal"


class MotionType(str, Enum):
    """Robot motion primitive for a path point."""

    JOINT = "joint"  # PTP in joint space -- air moves
    LINEAR = "linear"  # straight TCP line -- welding
    CIRCULAR = "circular"


# ===========================================================================
# CAD side
# ===========================================================================


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class SolidInfo:
    """One solid body of the imported assembly.

    A weldment is an assembly of solids; knowing which solid each side of a
    seam belongs to is what distinguishes a real joint (two parts to be fused)
    from an internal edge of a single machined body.
    """

    id: str
    name: str = ""
    volume_mm3: float = 0.0
    area_mm2: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    material: str = ""
    # Dominant sheet thickness, when the solid is plate-like; 0 if not.
    thickness_mm: float = 0.0

    @property
    def bbox_size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    @property
    def mass_kg(self) -> float:
        """Mass assuming mild steel unless the material says otherwise."""
        density_kg_mm3 = MATERIAL_DENSITY_KG_MM3.get(self.material, 7.85e-6)
        return self.volume_mm3 * density_kg_mm3


MATERIAL_DENSITY_KG_MM3: dict[str, float] = {
    "S235": 7.85e-6,
    "S275": 7.85e-6,
    "S355": 7.85e-6,
    "304": 7.90e-6,
    "316L": 8.00e-6,
    "AL5083": 2.66e-6,
    "AL6082": 2.70e-6,
}


@dataclass
class FaceRef:
    """Lightweight reference to a CAD face, with the properties seams need."""

    id: str
    solid_id: str
    face_type: str  # 'planar' | 'cylindrical' | 'conical' | 'spherical' | 'other'
    area_mm2: float
    normal: np.ndarray  # outward unit normal, evaluated at the face centre
    centroid: np.ndarray  # mm
    # For a planar face: signed distance of the plane from the origin.
    plane_offset_mm: float = 0.0
    radius_mm: float = 0.0  # cylindrical/conical/spherical only

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=float).reshape(3)
        self.centroid = np.asarray(self.centroid, dtype=float).reshape(3)

    @property
    def is_planar(self) -> bool:
        return self.face_type == "planar"


@dataclass
class EdgeRef:
    """A CAD edge sampled into a polyline, with its adjacent faces."""

    id: str
    points: np.ndarray  # (N, 3) mm, ordered along the edge
    curve_type: str = "line"  # 'line' | 'circle' | 'bspline' | ...
    face_ids: tuple[str, ...] = ()  # 1 = boundary, 2 = manifold, >2 = non-manifold
    solid_ids: tuple[str, ...] = ()
    radius_mm: float = 0.0

    def __post_init__(self) -> None:
        self.points = pl.as_points(self.points)

    @property
    def length_mm(self) -> float:
        """Arc length along the sampled curve."""
        return pl.total_length(self.points)

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    @property
    def is_manifold(self) -> bool:
        return len(self.face_ids) == 2

    @property
    def is_closed(self) -> bool:
        return pl.is_closed(self.points)


@dataclass
class Part:
    """The imported weldment: solids, faces, edges and a display mesh."""

    id: str = field(default_factory=lambda: _new_id("part"))
    name: str = ""
    source_path: str = ""
    solids: list[SolidInfo] = field(default_factory=list)
    faces: dict[str, FaceRef] = field(default_factory=dict)
    edges: dict[str, EdgeRef] = field(default_factory=dict)
    # Tessellated display/collision mesh in part coordinates.
    mesh_vertices: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None
    units_of_source: str = "mm"
    metadata: dict[str, Any] = field(default_factory=dict)
    # Live OpenCASCADE handles, when the part came from a B-rep file. Held in
    # a holder that is shared on copy and dropped on pickling -- see
    # :class:`BrepHandles` for why.
    brep: BrepHandles = field(default_factory=lambda: BrepHandles())

    # -- derived -------------------------------------------------------------

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """``(min, max)`` over all solids, or over the mesh if there are none."""
        if self.solids:
            lo = np.min([s.bbox_min for s in self.solids], axis=0)
            hi = np.max([s.bbox_max for s in self.solids], axis=0)
            return lo, hi
        if self.mesh_vertices is not None and len(self.mesh_vertices):
            return self.mesh_vertices.min(axis=0), self.mesh_vertices.max(axis=0)
        return np.zeros(3), np.zeros(3)

    @property
    def bbox_size(self) -> np.ndarray:
        lo, hi = self.bbox
        return hi - lo

    @property
    def mass_kg(self) -> float:
        return sum(s.mass_kg for s in self.solids)

    def solid(self, solid_id: str) -> SolidInfo | None:
        return next((s for s in self.solids if s.id == solid_id), None)

    def face(self, face_id: str) -> FaceRef | None:
        return self.faces.get(face_id)

    def summary(self) -> dict[str, Any]:
        lo, hi = self.bbox
        return {
            "id": self.id,
            "name": self.name,
            "n_solids": len(self.solids),
            "n_faces": len(self.faces),
            "n_edges": len(self.edges),
            "bbox_min_mm": lo.tolist(),
            "bbox_max_mm": hi.tolist(),
            "size_mm": (hi - lo).tolist(),
            "mass_kg": round(self.mass_kg, 3),
        }


# ===========================================================================
# Seams
# ===========================================================================


@dataclass
class Seam:
    """A continuous weld seam: where to weld, and the geometry the torch needs.

    The per-point arrays all share the same length *N* and index the same
    stations along the seam, so the path generator can work on them with plain
    NumPy. ``bisector`` is the inward bisector of the two adjacent faces -- the
    direction the wire should point for a symmetric fillet -- and is the single
    most important derived quantity in the whole pipeline.
    """

    id: str = field(default_factory=lambda: _new_id("seam"))
    joint_type: JointType = JointType.UNKNOWN

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) mm
    tangents: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) travel dir
    normal_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) face A normal
    normal_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) face B normal
    bisector: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) into the joint
    dihedral: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (N,) rad

    face_a_id: str = ""
    face_b_id: str = ""
    solid_a_id: str = ""
    solid_b_id: str = ""
    edge_ids: tuple[str, ...] = ()

    thickness_a_mm: float = 0.0
    thickness_b_mm: float = 0.0
    gap_mm: float = 0.0  # root opening measured from the CAD
    confidence: float = 0.0  # 0..1 from the classifier
    closed: bool = False
    selected: bool = True  # operator can switch a seam off
    label: str = ""  # free text, e.g. "cordone 3 - lato destro"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points = pl.as_points(self.points) if len(self.points) else np.zeros((0, 3))

    # -- derived -------------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def length_mm(self) -> float:
        return pl.total_length(self.points)

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    @property
    def centroid(self) -> np.ndarray:
        return pl.centroid(self.points)

    @property
    def mean_dihedral(self) -> float:
        return float(np.mean(self.dihedral)) if len(self.dihedral) else 0.0

    @property
    def is_straight(self) -> bool:
        """Straight to within 0.2 mm -- can be welded as a single linear move."""
        return pl.chord_deviation(self.points) <= 0.2

    @property
    def min_thickness_mm(self) -> float:
        """Governing thickness: the thinner member sets the heat input limit."""
        pos = [t for t in (self.thickness_a_mm, self.thickness_b_mm) if t > 0.0]
        return min(pos) if pos else 0.0

    @property
    def spans_two_solids(self) -> bool:
        return bool(self.solid_a_id and self.solid_b_id and self.solid_a_id != self.solid_b_id)

    def position_at(self, index: int, gravity: np.ndarray | None = None) -> WeldPosition:
        """Welding position at station *index* for a given gravity direction.

        *gravity* defaults to ``-Z`` of whatever frame ``points`` are expressed
        in. The classification follows EN ISO 6947 by looking at two angles:
        how far the seam axis tilts out of horizontal, and where the joint
        bisector points relative to gravity.
        """
        g = np.array([0.0, 0.0, -1.0]) if gravity is None else tf.unit(gravity)
        if g is None or index >= self.n_points:
            return WeldPosition.PA

        t = self.tangents[index] if len(self.tangents) > index else np.array([1.0, 0.0, 0.0])
        b = self.bisector[index] if len(self.bisector) > index else np.array([0.0, 0.0, 1.0])

        # Seam axis inclination: 0 = horizontal run, 90 deg = running up or
        # down the plate.
        axis_tilt = abs(np.pi / 2.0 - tf.angle_between(t, g))
        if axis_tilt > np.deg2rad(60.0):
            return WeldPosition.PF

        # ``bisector`` is the direction the **wire points**. Pointing with
        # gravity means the torch is above the joint welding down into it
        # (downhand); pointing against gravity means the torch is underneath
        # welding upwards (overhead).
        wire_vs_gravity = tf.angle_between(b, g)

        if wire_vs_gravity < np.deg2rad(60.0):
            # Downhand. A butt lies in the gutter (PA); a fillet with one
            # member vertical is the classic horizontal-vertical fillet (PB).
            return WeldPosition.PA if self.joint_type is JointType.BUTT else WeldPosition.PB
        if wire_vs_gravity > np.deg2rad(120.0):
            return WeldPosition.PE if self.joint_type is JointType.BUTT else WeldPosition.PD
        return WeldPosition.PC

    @property
    def dominant_position(self) -> WeldPosition:
        """Most common position along the seam -- what the WPS is written for."""
        if self.n_points == 0:
            return WeldPosition.PA
        counts: dict[WeldPosition, int] = {}
        for i in range(self.n_points):
            p = self.position_at(i)
            counts[p] = counts.get(p, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def reversed(self) -> Seam:
        """Same seam welded the other way round.

        Tangents flip; face normals and the bisector do not. Used by the
        sequencer when welding back towards the next seam saves an air move.
        """
        import copy

        out = copy.deepcopy(self)
        out.points = self.points[::-1].copy()
        out.tangents = -self.tangents[::-1].copy() if len(self.tangents) else self.tangents
        out.normal_a = self.normal_a[::-1].copy() if len(self.normal_a) else self.normal_a
        out.normal_b = self.normal_b[::-1].copy() if len(self.normal_b) else self.normal_b
        out.bisector = self.bisector[::-1].copy() if len(self.bisector) else self.bisector
        out.dihedral = self.dihedral[::-1].copy() if len(self.dihedral) else self.dihedral
        out.notes = [*self.notes, "reversed"]
        return out

    def transformed(self, T: np.ndarray) -> Seam:
        """Same seam expressed in another frame (part -> cell, typically)."""
        import copy

        out = copy.deepcopy(self)
        out.points = tf.apply_points(T, self.points)
        for attr in ("tangents", "normal_a", "normal_b", "bisector"):
            v = getattr(self, attr)
            if len(v):
                setattr(out, attr, tf.apply_vector(T, v))
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "joint_type": self.joint_type.value,
            "length_mm": round(self.length_mm, 2),
            "n_points": self.n_points,
            "mean_dihedral_deg": round(np.rad2deg(self.mean_dihedral), 1),
            "thickness_mm": [self.thickness_a_mm, self.thickness_b_mm],
            "position": self.dominant_position.value,
            "confidence": round(self.confidence, 3),
            "closed": self.closed,
            "straight": self.is_straight,
            "spans_two_solids": self.spans_two_solids,
        }


# ===========================================================================
# Welding procedure (WPS)
# ===========================================================================


@dataclass
class Weave:
    """Weaving motion superimposed on the travel direction."""

    shape: WeaveShape = WeaveShape.NONE
    amplitude_mm: float = 0.0  # peak-to-peak, across the seam
    frequency_hz: float = 0.0
    dwell_ms: float = 0.0  # pause at each side, for sidewall fusion

    @property
    def active(self) -> bool:
        return self.shape is not WeaveShape.NONE and self.amplitude_mm > 0.0


@dataclass
class WeldPass:
    """One pass of one seam: the numbers that go on the machine and the robot.

    A fillet above about 6 mm leg, or any thick butt, needs several passes;
    each gets its own offset from the joint root, its own electrical settings
    and its own travel speed.
    """

    index: int = 0
    name: str = ""  # 'root', 'fill 1', 'cap', ...
    # --- geometry of the deposit ---
    leg_mm: float = 0.0  # fillet leg (z), 0 for a butt
    throat_mm: float = 0.0  # effective throat (a)
    offset_across_mm: float = 0.0  # lateral offset from the joint root
    offset_normal_mm: float = 0.0  # offset along the bisector (build-up)
    # --- electrical / process ---
    process: Process = Process.GMAW
    transfer_mode: TransferMode = TransferMode.SPRAY
    current_a: float = 0.0
    voltage_v: float = 0.0
    wire_speed_m_min: float = 0.0
    wire_diameter_mm: float = 1.2
    polarity: str = "DCEP"
    gas: str = "M21-ArC-18"  # EN ISO 14175 designation
    gas_flow_l_min: float = 15.0
    # --- motion ---
    travel_speed_mm_s: float = 0.0
    work_angle_deg: float = 45.0  # rotation about the travel axis
    travel_angle_deg: float = -10.0  # push (<0) / drag (>0) from vertical
    stickout_mm: float = 15.0  # contact tip to work distance
    weave: Weave = field(default_factory=Weave)
    # --- timing ---
    arc_start_delay_s: float = 0.2
    crater_fill_s: float = 0.3
    burnback_s: float = 0.08

    def heat_input_kj_mm(self, efficiency: float | None = None) -> float:
        from .units import heat_input_kj_mm

        eta = self.process.default_efficiency if efficiency is None else efficiency
        if self.travel_speed_mm_s <= 0.0:
            return 0.0
        return heat_input_kj_mm(self.current_a, self.voltage_v, self.travel_speed_mm_s, eta)

    def arc_time_s(self, seam_length_mm: float) -> float:
        if self.travel_speed_mm_s <= 0.0:
            return 0.0
        base = seam_length_mm / self.travel_speed_mm_s
        if self.weave.active and self.weave.frequency_hz > 0.0:
            # Weaving stretches the path the wire actually travels; the
            # along-seam speed is what the operator sets, so the extra length
            # does not change arc time, but dwell does.
            cycles = base * self.weave.frequency_hz
            base += cycles * 2.0 * self.weave.dwell_ms / 1000.0
        return base + self.arc_start_delay_s + self.crater_fill_s

    def deposited_mass_kg(self, seam_length_mm: float) -> float:
        from .units import deposition_rate_kg_h

        rate = deposition_rate_kg_h(self.wire_diameter_mm, self.wire_speed_m_min)
        return rate * self.arc_time_s(seam_length_mm) / 3600.0


@dataclass
class WeldProcedure:
    """A WPS instance bound to one seam: material, joint, position, passes."""

    seam_id: str = ""
    process: Process = Process.GMAW
    base_material: str = "S235"
    filler: str = "G3Si1"  # EN ISO 14341
    position: WeldPosition = WeldPosition.PB
    joint_type: JointType = JointType.FILLET
    thickness_mm: float = 0.0
    target_leg_mm: float = 0.0
    passes: list[WeldPass] = field(default_factory=list)
    preheat_c: float = 20.0
    interpass_max_c: float = 250.0
    heat_input_max_kj_mm: float = 0.0  # 0 = unconstrained
    standard: str = "EN ISO 15614-1"
    wps_id: str = field(default_factory=lambda: _new_id("wps"))
    warnings: list[str] = field(default_factory=list)

    @property
    def n_passes(self) -> int:
        return len(self.passes)

    def total_arc_time_s(self, seam_length_mm: float) -> float:
        return sum(p.arc_time_s(seam_length_mm) for p in self.passes)

    def total_wire_kg(self, seam_length_mm: float) -> float:
        return sum(p.deposited_mass_kg(seam_length_mm) for p in self.passes)

    def max_heat_input_kj_mm(self) -> float:
        return max((p.heat_input_kj_mm() for p in self.passes), default=0.0)

    def summary(self) -> dict[str, Any]:
        return {
            "wps_id": self.wps_id,
            "seam_id": self.seam_id,
            "process": self.process.value,
            "iso_4063": self.process.iso_4063,
            "base_material": self.base_material,
            "filler": self.filler,
            "position": self.position.value,
            "joint_type": self.joint_type.value,
            "thickness_mm": self.thickness_mm,
            "target_leg_mm": self.target_leg_mm,
            "n_passes": self.n_passes,
            "max_heat_input_kj_mm": round(self.max_heat_input_kj_mm(), 3),
            "preheat_c": self.preheat_c,
            "warnings": list(self.warnings),
        }


@dataclass
class MachineSetpoints:
    """What actually gets written to the power source for one pass.

    Translating a :class:`WeldPass` into this is the job of a machine profile
    (:mod:`weldforge.wps.machines`): the same 220 A / 24 V is job 7 with a
    correction of +3 on one brand and a pair of analog words on another.
    """

    pass_index: int = 0
    job_number: int | None = None  # synergic program / JOB on the machine
    current_a: float = 0.0
    voltage_v: float = 0.0
    wire_speed_m_min: float = 0.0
    arc_length_correction: float = 0.0  # -10..+10 on most brands
    dynamic_correction: float = 0.0  # inductance / arc force
    gas_flow_l_min: float = 15.0
    pulse_frequency_hz: float = 0.0
    # Raw fieldbus image, filled by the machine profile.
    analog_words: dict[str, int] = field(default_factory=dict)
    digital_bits: dict[str, bool] = field(default_factory=dict)
    machine_profile: str = ""
    notes: list[str] = field(default_factory=list)


# ===========================================================================
# Cell, fixturing and placement
# ===========================================================================


@dataclass
class Obstacle:
    """A static volume the robot must avoid, as a mesh in cell coordinates."""

    id: str
    name: str = ""
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    kind: str = "generic"  # 'table' | 'jig' | 'clamp' | 'fence' | 'generic'
    pose: np.ndarray = field(default_factory=tf.identity)

    @property
    def has_mesh(self) -> bool:
        return self.vertices is not None and self.faces is not None and len(self.faces) > 0


@dataclass
class Clamp:
    """A clamp or locator on the jig.

    ``approach`` is the direction the clamp arm comes from; the planner keeps
    the torch out of a cylinder of radius ``clearance_mm`` around ``position``
    and treats the approach direction as blocked.
    """

    id: str
    position: np.ndarray  # mm, cell frame
    approach: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    clearance_mm: float = 40.0
    height_mm: float = 60.0
    name: str = ""
    retractable: bool = False  # can be opened mid-cycle

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.approach = np.asarray(self.approach, dtype=float).reshape(3)


@dataclass
class Fixture:
    """The jig ("maschera") or bare bench the part sits on."""

    id: str = field(default_factory=lambda: _new_id("fix"))
    name: str = "bench"
    kind: str = "bench"  # 'bench' | 'jig' | 'positioner'
    # Support plane in cell coordinates: parts rest on z = table_z_mm.
    table_z_mm: float = 0.0
    table_size_mm: tuple[float, float] = (1500.0, 1000.0)
    table_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    clamps: list[Clamp] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    # Datum features the part must register against, as (point, normal) pairs.
    datums: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    load_capacity_kg: float = 500.0
    #: Filename of a 3-D model imported as the bench, empty when the bench is
    #: just the box above. The mesh itself is an obstacle in ``obstacles`` so
    #: that collision checking and the 3-D view pick it up with no special
    #: case; this field is what lets the view draw the model instead of the
    #: box, and what names the file to re-embed when a project is saved.
    model_source: str = ""

    def __post_init__(self) -> None:
        self.table_origin = np.asarray(self.table_origin, dtype=float).reshape(3)


@dataclass
class ExternalAxis:
    """A positioner or track axis the planner may command."""

    name: str
    kind: str = "rotary"  # 'rotary' | 'linear'
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    limit_lo: float = -np.pi
    limit_hi: float = np.pi
    max_speed: float = 1.0  # rad/s or mm/s
    coupled: bool = True  # moves with the robot (coordinated)

    def __post_init__(self) -> None:
        self.axis = np.asarray(self.axis, dtype=float).reshape(3)
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)

    def transform(self, value: float) -> np.ndarray:
        """Pose contributed by this axis at joint value *value*."""
        if self.kind == "rotary":
            R = tf.rot_axis_angle(self.axis, value)
            to_origin = tf.pose(np.eye(3), -self.origin)
            back = tf.pose(np.eye(3), self.origin)
            return tf.compose(back, tf.pose(R), to_origin)
        return tf.pose(np.eye(3), self.axis * value)


@dataclass
class WeldCell:
    """The whole cell: robot base, torch, fixture, external axes, obstacles."""

    id: str = field(default_factory=lambda: _new_id("cell"))
    name: str = "cell"
    robot_name: str = ""
    robot_base: np.ndarray = field(default_factory=tf.identity)
    # Tool frame of the torch tip relative to the robot flange.
    torch_tcp: np.ndarray = field(default_factory=tf.identity)
    torch_mesh_vertices: np.ndarray | None = None
    torch_mesh_faces: np.ndarray | None = None
    fixture: Fixture = field(default_factory=Fixture)
    external_axes: list[ExternalAxis] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    home_joints: np.ndarray | None = None
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))

    def all_obstacles(self) -> list[Obstacle]:
        return [*self.obstacles, *self.fixture.obstacles]


@dataclass
class PlacementCandidate:
    """One candidate pose of the part on the fixture, with its score."""

    pose: np.ndarray = field(default_factory=tf.identity)  # part -> cell
    score: float = 0.0
    reachable_seams: list[str] = field(default_factory=list)
    unreachable_seams: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    position_mix: dict[str, int] = field(default_factory=dict)  # position -> seam count
    detail: dict[str, float] = field(default_factory=dict)  # per-criterion scores
    external_axis_values: dict[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.violations and bool(self.reachable_seams)

    def summary(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "feasible": self.feasible,
            "translation_mm": tf.translation_of(self.pose).round(2).tolist(),
            "rpy_deg": [round(np.rad2deg(a), 2) for a in tf.matrix_to_rpy(self.pose)],
            "n_reachable": len(self.reachable_seams),
            "n_unreachable": len(self.unreachable_seams),
            "position_mix": dict(self.position_mix),
            "violations": list(self.violations),
            "detail": {k: round(v, 4) for k, v in self.detail.items()},
            "external_axis_values": {k: round(v, 5) for k, v in self.external_axis_values.items()},
        }


# ===========================================================================
# Motion
# ===========================================================================


@dataclass
class PathPoint:
    """One station of the torch path, in cell coordinates."""

    pose: np.ndarray = field(default_factory=tf.identity)
    motion: MotionType = MotionType.LINEAR
    speed_mm_s: float = 50.0
    weld_on: bool = False
    seam_id: str = ""
    pass_index: int = 0
    # Filled by the kinematics stage.
    joints: np.ndarray | None = None
    external_axes: dict[str, float] = field(default_factory=dict)
    # Diagnostics filled by the simulator.
    manipulability: float = 0.0
    collision: str = ""
    comment: str = ""

    @property
    def position(self) -> np.ndarray:
        return tf.translation_of(self.pose)


@dataclass
class TorchPath:
    """The torch path for one pass of one seam, plus its air moves."""

    seam_id: str = ""
    pass_index: int = 0
    points: list[PathPoint] = field(default_factory=list)
    setpoints: MachineSetpoints | None = None

    @property
    def weld_length_mm(self) -> float:
        total = 0.0
        for a, b in zip(self.points, self.points[1:], strict=False):
            if b.weld_on:
                total += float(np.linalg.norm(b.position - a.position))
        return total

    @property
    def cycle_time_s(self) -> float:
        total = 0.0
        for a, b in zip(self.points, self.points[1:], strict=False):
            d = float(np.linalg.norm(b.position - a.position))
            v = max(b.speed_mm_s, 1e-3)
            total += d / v
        return total


# ===========================================================================
# The job -- what the whole pipeline produces
# ===========================================================================


@dataclass
class JobReport:
    """Verdict and numbers for an operator to sign off on."""

    reachable: bool = False
    collision_free: bool = False
    cycle_time_s: float = 0.0
    arc_time_s: float = 0.0
    air_time_s: float = 0.0
    total_weld_length_mm: float = 0.0
    wire_consumption_kg: float = 0.0
    gas_consumption_l: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_seam: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.reachable and self.collision_free and not self.errors

    @property
    def duty_cycle(self) -> float:
        """Arc-on time over total cycle time -- the number a plant manager asks for."""
        return self.arc_time_s / self.cycle_time_s if self.cycle_time_s > 0.0 else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reachable": self.reachable,
            "collision_free": self.collision_free,
            "cycle_time_s": round(self.cycle_time_s, 2),
            "arc_time_s": round(self.arc_time_s, 2),
            "air_time_s": round(self.air_time_s, 2),
            "duty_cycle": round(self.duty_cycle, 3),
            "total_weld_length_mm": round(self.total_weld_length_mm, 1),
            "wire_consumption_kg": round(self.wire_consumption_kg, 3),
            "gas_consumption_l": round(self.gas_consumption_l, 1),
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass
class WeldJob:
    """Everything needed to weld one part, from CAD to robot program."""

    id: str = field(default_factory=lambda: _new_id("job"))
    name: str = ""
    part: Part | None = None
    cell: WeldCell | None = None
    seams: list[Seam] = field(default_factory=list)
    procedures: dict[str, WeldProcedure] = field(default_factory=dict)  # seam_id -> WPS
    placement: PlacementCandidate | None = None
    # Indexed positioner setting per seam, when the cell has external axes.
    # Typed loosely to keep the model free of a dependency on the fixture
    # package; it is a weldforge.fixture.positioner.PositionerPlan.
    positioner: Any = None
    paths: list[TorchPath] = field(default_factory=list)
    report: JobReport = field(default_factory=JobReport)
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_seams(self) -> list[Seam]:
        return [s for s in self.seams if s.selected]

    def seam(self, seam_id: str) -> Seam | None:
        return next((s for s in self.seams if s.id == seam_id), None)

    def procedure(self, seam_id: str) -> WeldProcedure | None:
        return self.procedures.get(seam_id)

    @property
    def total_weld_length_mm(self) -> float:
        return sum(s.length_mm for s in self.selected_seams)

    def part_pose_for(self, seam_id: str) -> np.ndarray:
        """Where the part sits while *seam_id* is welded.

        On a bench that is the placement, the same for every seam. With a
        positioner each seam gets its own indexed setting, so the pose differs
        -- and using one pose for the whole job would plan every seam as if it
        were still facing the way it was at setup.
        """
        if self.positioner is not None:
            entry = getattr(self.positioner, "per_seam", {}).get(seam_id)
            if entry is not None:
                return entry.part_pose.copy()
        if self.placement is not None:
            return self.placement.pose.copy()
        return np.eye(4)

    def external_axes_for(self, seam_id: str) -> dict[str, float]:
        """Positioner axis values held while *seam_id* is welded."""
        if self.positioner is None:
            return {}
        return self.positioner.values_for(seam_id)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "part": self.part.summary() if self.part else None,
            "cell": self.cell.name if self.cell else None,
            "n_seams": len(self.seams),
            "n_selected": len(self.selected_seams),
            "total_weld_length_mm": round(self.total_weld_length_mm, 1),
            "n_procedures": len(self.procedures),
            "placement": self.placement.summary() if self.placement else None,
            "positioner": self.positioner.summary() if self.positioner is not None else None,
            "n_paths": len(self.paths),
            "report": self.report.summary(),
        }


# ===========================================================================
# Live CAD handles
# ===========================================================================


@dataclass
class BrepHandles:
    """Live OpenCASCADE objects belonging to a loaded :class:`Part`.

    Seam recognition needs the exact B-rep -- surface normals at arbitrary
    points, trim classification, ray casts -- so the OCCT handles have to
    survive past the loader. But they are C++ objects wrapped by a binding that
    holds module references, which makes them neither copyable nor picklable:

    * ``copy.deepcopy(part)`` raised ``cannot pickle 'module' object``;
    * any attempt to cache a part, queue one to a worker, or return one through
      the API failed the same way.

    So they live in this holder, which declares itself **shared, never
    copied**: a deep copy of a part keeps pointing at the same handles (they
    are immutable as far as we use them), and pickling drops them, leaving a
    part that still carries all its measured geometry and simply cannot run
    B-rep-exact seam detection again. That is the right trade: the measurements
    are what downstream stages need, and re-detecting means re-loading the file
    anyway.
    """

    backend: Any = None
    shape: Any = None
    topology: Any = None
    evaluators: dict[int, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.topology is not None and self.shape is not None

    def __deepcopy__(self, memo: dict) -> BrepHandles:
        # Shared, not copied: the handles are read-only in our use.
        memo[id(self)] = self
        return self

    def __copy__(self) -> BrepHandles:
        return self

    def __getstate__(self) -> dict:
        # Dropped on pickling; see the class docstring.
        return {}

    def __setstate__(self, state: dict) -> None:
        self.backend = None
        self.shape = None
        self.topology = None
        self.evaluators = {}
