"""JSON shapes for the REST API.

The core works in NumPy dataclasses, which are fast and ergonomic but not
serialisable. These Pydantic models are the wire format: they are the only
place that knows how a seam or a pass looks as JSON, so the CAD plugins have
one stable contract to code against while the core is free to change.

Lengths stay in **millimetres** and angles are converted to **degrees** at
this boundary, because that is what a CAD user and a welding engineer read.
The one exception is any field whose name says otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..geom import transform as tf
from ..model import (
    MachineSetpoints,
    Part,
    PlacementCandidate,
    Seam,
    TorchPath,
    WeldJob,
    WeldPass,
    WeldProcedure,
)

__all__ = [
    "PoseModel",
    "SolidModel",
    "PartModel",
    "SeamModel",
    "WeavModel",
    "PassModel",
    "ProcedureModel",
    "SetpointsModel",
    "PlacementModel",
    "PathPointModel",
    "TorchPathModel",
    "ReportModel",
    "JobModel",
    "pose_to_model",
    "part_to_model",
    "seam_to_model",
    "procedure_to_model",
    "placement_to_model",
    "path_to_model",
    "job_to_model",
    "setpoints_to_model",
]


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, np.ndarray):
        return [round(float(v), digits) for v in value.reshape(-1)]
    return round(float(value), digits)


class PoseModel(BaseModel):
    """A rigid transform, given three ways so a client can use whichever suits."""

    position_mm: list[float] = Field(..., description="Translation in millimetres")
    quaternion_xyzw: list[float] = Field(..., description="Rotation, ROS order")
    rpy_deg: list[float] = Field(..., description="Intrinsic Z-Y-X roll, pitch, yaw")


def pose_to_model(pose: np.ndarray) -> PoseModel:
    roll, pitch, yaw = tf.matrix_to_rpy(pose)
    return PoseModel(
        position_mm=_round(tf.translation_of(pose), 3),
        quaternion_xyzw=_round(tf.matrix_to_quat(pose), 6),
        rpy_deg=[round(float(np.rad2deg(a)), 3) for a in (roll, pitch, yaw)],
    )


# ---------------------------------------------------------------------------
# part
# ---------------------------------------------------------------------------


class SolidModel(BaseModel):
    id: str
    name: str = ""
    volume_mm3: float
    area_mm2: float
    thickness_mm: float
    mass_kg: float
    bbox_min_mm: list[float]
    bbox_max_mm: list[float]


class PartModel(BaseModel):
    id: str
    name: str
    source_path: str = ""
    geometry_kind: str = ""
    n_solids: int
    n_faces: int
    n_edges: int
    bbox_min_mm: list[float]
    bbox_max_mm: list[float]
    size_mm: list[float]
    mass_kg: float
    thicknesses_mm: list[float]
    solids: list[SolidModel] = Field(default_factory=list)
    load_seconds: float = 0.0


def part_to_model(part: Part, with_solids: bool = False) -> PartModel:
    lo, hi = part.bbox
    return PartModel(
        id=part.id,
        name=part.name,
        source_path=part.source_path,
        geometry_kind=str(part.metadata.get("geometry_kind", "")),
        n_solids=len(part.solids),
        n_faces=len(part.faces),
        n_edges=len(part.edges),
        bbox_min_mm=_round(lo, 3),
        bbox_max_mm=_round(hi, 3),
        size_mm=_round(hi - lo, 3),
        mass_kg=round(part.mass_kg, 3),
        thicknesses_mm=sorted(
            {round(s.thickness_mm, 2) for s in part.solids if s.thickness_mm > 0}
        ),
        solids=[
            SolidModel(
                id=s.id,
                name=s.name,
                volume_mm3=round(s.volume_mm3, 2),
                area_mm2=round(s.area_mm2, 2),
                thickness_mm=round(s.thickness_mm, 3),
                mass_kg=round(s.mass_kg, 4),
                bbox_min_mm=_round(s.bbox_min, 3),
                bbox_max_mm=_round(s.bbox_max, 3),
            )
            for s in part.solids
        ]
        if with_solids
        else [],
        load_seconds=float(part.metadata.get("load_seconds", 0.0)),
    )


# ---------------------------------------------------------------------------
# seam
# ---------------------------------------------------------------------------


class SeamModel(BaseModel):
    id: str
    label: str = ""
    joint_type: str
    length_mm: float
    n_points: int
    groove_deg: float = Field(..., description="Included angle on the weld side")
    groove_spread_deg: float = 0.0
    position: str = Field(..., description="EN ISO 6947 welding position")
    thickness_a_mm: float
    thickness_b_mm: float
    gap_mm: float
    confidence: float
    closed: bool
    straight: bool
    selected: bool
    spans_two_solids: bool
    solid_a_id: str = ""
    solid_b_id: str = ""
    start_mm: list[float]
    end_mm: list[float]
    centroid_mm: list[float]
    notes: list[str] = Field(default_factory=list)
    # Full geometry, only when the client asks: a 200-station seam is 5 arrays
    # of 200 triples, which is not what a seam list wants to carry.
    points_mm: list[list[float]] | None = None
    tangents: list[list[float]] | None = None
    bisectors: list[list[float]] | None = None


def seam_to_model(seam: Seam, with_geometry: bool = False) -> SeamModel:
    spread = float(np.rad2deg(np.ptp(seam.dihedral))) if len(seam.dihedral) else 0.0
    model = SeamModel(
        id=seam.id,
        label=seam.label,
        joint_type=seam.joint_type.value,
        length_mm=round(seam.length_mm, 2),
        n_points=seam.n_points,
        groove_deg=round(float(np.rad2deg(seam.mean_dihedral)), 2),
        groove_spread_deg=round(spread, 2),
        position=seam.dominant_position.value,
        thickness_a_mm=round(seam.thickness_a_mm, 3),
        thickness_b_mm=round(seam.thickness_b_mm, 3),
        gap_mm=round(seam.gap_mm, 3),
        confidence=round(seam.confidence, 3),
        closed=seam.closed,
        straight=seam.is_straight,
        selected=seam.selected,
        spans_two_solids=seam.spans_two_solids,
        solid_a_id=seam.solid_a_id,
        solid_b_id=seam.solid_b_id,
        start_mm=_round(seam.start, 3),
        end_mm=_round(seam.end, 3),
        centroid_mm=_round(seam.centroid, 3),
        notes=list(seam.notes),
    )
    if with_geometry and seam.n_points:
        model.points_mm = [[round(float(v), 3) for v in p] for p in seam.points]
        model.tangents = [[round(float(v), 5) for v in p] for p in seam.tangents]
        model.bisectors = [[round(float(v), 5) for v in p] for p in seam.bisector]
    return model


# ---------------------------------------------------------------------------
# procedure
# ---------------------------------------------------------------------------


class WeavModel(BaseModel):
    shape: str
    amplitude_mm: float
    frequency_hz: float
    dwell_ms: float


class PassModel(BaseModel):
    index: int
    name: str
    leg_mm: float
    throat_mm: float
    offset_across_mm: float
    offset_normal_mm: float
    process: str
    transfer_mode: str
    current_a: float
    voltage_v: float
    wire_speed_m_min: float
    wire_diameter_mm: float
    gas: str
    gas_flow_l_min: float
    travel_speed_mm_s: float
    work_angle_deg: float
    travel_angle_deg: float
    stickout_mm: float
    heat_input_kj_mm: float
    weave: WeavModel


class ProcedureModel(BaseModel):
    wps_id: str
    seam_id: str
    process: str
    iso_4063: int
    base_material: str
    filler: str
    position: str
    joint_type: str
    thickness_mm: float
    target_leg_mm: float
    preheat_c: float
    interpass_max_c: float
    heat_input_max_kj_mm: float
    standard: str
    passes: list[PassModel]
    warnings: list[str] = Field(default_factory=list)


def _pass_to_model(weld_pass: WeldPass) -> PassModel:
    return PassModel(
        index=weld_pass.index,
        name=weld_pass.name,
        leg_mm=weld_pass.leg_mm,
        throat_mm=weld_pass.throat_mm,
        offset_across_mm=weld_pass.offset_across_mm,
        offset_normal_mm=weld_pass.offset_normal_mm,
        process=weld_pass.process.value,
        transfer_mode=weld_pass.transfer_mode.value,
        current_a=weld_pass.current_a,
        voltage_v=weld_pass.voltage_v,
        wire_speed_m_min=weld_pass.wire_speed_m_min,
        wire_diameter_mm=weld_pass.wire_diameter_mm,
        gas=weld_pass.gas,
        gas_flow_l_min=weld_pass.gas_flow_l_min,
        travel_speed_mm_s=weld_pass.travel_speed_mm_s,
        work_angle_deg=weld_pass.work_angle_deg,
        travel_angle_deg=weld_pass.travel_angle_deg,
        stickout_mm=weld_pass.stickout_mm,
        heat_input_kj_mm=round(weld_pass.heat_input_kj_mm(), 4),
        weave=WeavModel(
            shape=weld_pass.weave.shape.value,
            amplitude_mm=weld_pass.weave.amplitude_mm,
            frequency_hz=weld_pass.weave.frequency_hz,
            dwell_ms=weld_pass.weave.dwell_ms,
        ),
    )


def procedure_to_model(procedure: WeldProcedure) -> ProcedureModel:
    return ProcedureModel(
        wps_id=procedure.wps_id,
        seam_id=procedure.seam_id,
        process=procedure.process.value,
        iso_4063=procedure.process.iso_4063,
        base_material=procedure.base_material,
        filler=procedure.filler,
        position=procedure.position.value,
        joint_type=procedure.joint_type.value,
        thickness_mm=procedure.thickness_mm,
        target_leg_mm=procedure.target_leg_mm,
        preheat_c=procedure.preheat_c,
        interpass_max_c=procedure.interpass_max_c,
        heat_input_max_kj_mm=procedure.heat_input_max_kj_mm,
        standard=procedure.standard,
        passes=[_pass_to_model(p) for p in procedure.passes],
        warnings=list(procedure.warnings),
    )


class SetpointsModel(BaseModel):
    pass_index: int
    job_number: int | None = None
    current_a: float
    voltage_v: float
    wire_speed_m_min: float
    gas_flow_l_min: float
    machine_profile: str
    analog_words: dict[str, int] = Field(default_factory=dict)
    digital_bits: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def setpoints_to_model(setpoints: MachineSetpoints) -> SetpointsModel:
    return SetpointsModel(
        pass_index=setpoints.pass_index,
        job_number=setpoints.job_number,
        current_a=setpoints.current_a,
        voltage_v=setpoints.voltage_v,
        wire_speed_m_min=setpoints.wire_speed_m_min,
        gas_flow_l_min=setpoints.gas_flow_l_min,
        machine_profile=setpoints.machine_profile,
        analog_words=dict(setpoints.analog_words),
        digital_bits=dict(setpoints.digital_bits),
        notes=list(setpoints.notes),
    )


# ---------------------------------------------------------------------------
# placement, paths, report, job
# ---------------------------------------------------------------------------


class PlacementModel(BaseModel):
    pose: PoseModel
    score: float
    feasible: bool
    n_reachable: int
    n_unreachable: int
    position_mix: dict[str, int]
    detail: dict[str, float]
    violations: list[str]
    unreachable_seams: list[str]


def placement_to_model(placement: PlacementCandidate) -> PlacementModel:
    return PlacementModel(
        pose=pose_to_model(placement.pose),
        score=round(placement.score, 4),
        feasible=placement.feasible,
        n_reachable=len(placement.reachable_seams),
        n_unreachable=len(placement.unreachable_seams),
        position_mix=dict(placement.position_mix),
        detail={k: round(v, 4) for k, v in placement.detail.items()},
        violations=list(placement.violations),
        unreachable_seams=list(placement.unreachable_seams),
    )


class PathPointModel(BaseModel):
    pose: PoseModel
    motion: str
    speed_mm_s: float
    weld_on: bool
    joints_deg: list[float] | None = None
    manipulability: float = 0.0
    collision: str = ""
    comment: str = ""


class TorchPathModel(BaseModel):
    seam_id: str
    pass_index: int
    n_points: int
    weld_length_mm: float
    cycle_time_s: float
    points: list[PathPointModel] | None = None


def path_to_model(path: TorchPath, with_points: bool = False) -> TorchPathModel:
    model = TorchPathModel(
        seam_id=path.seam_id,
        pass_index=path.pass_index,
        n_points=len(path.points),
        weld_length_mm=round(path.weld_length_mm, 2),
        cycle_time_s=round(path.cycle_time_s, 3),
    )
    if with_points:
        model.points = [
            PathPointModel(
                pose=pose_to_model(p.pose),
                motion=p.motion.value,
                speed_mm_s=p.speed_mm_s,
                weld_on=p.weld_on,
                joints_deg=(
                    [round(float(np.rad2deg(v)), 4) for v in p.joints]
                    if p.joints is not None
                    else None
                ),
                manipulability=round(p.manipulability, 4),
                collision=p.collision,
                comment=p.comment,
            )
            for p in path.points
        ]
    return model


class ReportModel(BaseModel):
    ok: bool
    reachable: bool
    collision_free: bool
    cycle_time_s: float
    arc_time_s: float
    air_time_s: float
    duty_cycle: float
    total_weld_length_mm: float
    wire_consumption_kg: float
    gas_consumption_l: float
    errors: list[str]
    warnings: list[str]
    per_seam: dict[str, dict[str, Any]] = Field(default_factory=dict)


class JobModel(BaseModel):
    id: str
    name: str
    created_at: str = ""
    part: PartModel | None = None
    cell: str = ""
    robot: str = ""
    total_weld_length_mm: float
    seams: list[SeamModel] = Field(default_factory=list)
    procedures: list[ProcedureModel] = Field(default_factory=list)
    placement: PlacementModel | None = None
    paths: list[TorchPathModel] = Field(default_factory=list)
    report: ReportModel
    metadata: dict[str, Any] = Field(default_factory=dict)


def job_to_model(
    job: WeldJob,
    with_geometry: bool = False,
    with_points: bool = False,
) -> JobModel:
    report = job.report
    return JobModel(
        id=job.id,
        name=job.name,
        created_at=job.created_at,
        part=part_to_model(job.part) if job.part else None,
        cell=job.cell.name if job.cell else "",
        robot=job.cell.robot_name if job.cell else "",
        total_weld_length_mm=round(job.total_weld_length_mm, 2),
        seams=[seam_to_model(s, with_geometry) for s in job.seams],
        procedures=[procedure_to_model(p) for p in job.procedures.values()],
        placement=placement_to_model(job.placement) if job.placement else None,
        paths=[path_to_model(p, with_points) for p in job.paths],
        report=ReportModel(
            ok=report.ok,
            reachable=report.reachable,
            collision_free=report.collision_free,
            cycle_time_s=round(report.cycle_time_s, 2),
            arc_time_s=round(report.arc_time_s, 2),
            air_time_s=round(report.air_time_s, 2),
            duty_cycle=round(report.duty_cycle, 4),
            total_weld_length_mm=round(report.total_weld_length_mm, 2),
            wire_consumption_kg=round(report.wire_consumption_kg, 4),
            gas_consumption_l=round(report.gas_consumption_l, 2),
            errors=list(report.errors),
            warnings=list(report.warnings),
            per_seam=dict(report.per_seam),
        ),
        metadata={k: v for k, v in job.metadata.items() if not k.startswith("_")},
    )
