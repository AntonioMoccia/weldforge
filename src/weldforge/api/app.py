"""The REST API.

This is the integration surface: the FreeCAD workbench, the SolidWorks add-in,
a shop-floor web page and the ROS 2 cell all drive the same endpoints, so the
planning logic exists once and every client gets the same answers.

Shape of a session
------------------
.. code-block:: text

    POST /parts                      upload a STEP/IGES/STL, get a part id
    GET  /parts/{id}/seams           recognise seams (fast, cacheable)
    POST /jobs                       plan: seams + procedures + placement +
                                     paths + simulation, in the background
    GET  /jobs/{id}                  poll status, then read the result
    GET  /jobs/{id}/report           just the verdict and the numbers
    POST /jobs/{id}/export           robot program for a given controller

Planning runs in a **background task** and the client polls, because a
200-seam weldment takes tens of seconds to plan and no CAD plugin should block
its UI thread on an HTTP request that long.

Reference data -- cells, robots, machines, procedures, materials, export
targets -- is exposed read-only so a plugin can populate its dialogs from the
server instead of hard-coding lists that drift.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

from ..cad import BREP_SUFFIXES, MESH_SUFFIXES, LoadOptions, backend_name, load_part
from ..fixture import available_cells, cell_robot, load_cell
from ..kin import available_robots, load_robot
from ..model import Part
from ..plan import PlanOptions, plan_from_part
from ..post import PostOptions, available_targets, get_postprocessor
from ..seams import DetectOptions, detect_seams
from ..wps import available_machines, available_procedures, load_machine, load_materials
from ..wps.machines import setpoints_for_procedure
from .schemas import (
    PartModel,
    ReportModel,
    job_to_model,
    part_to_model,
    seam_to_model,
    setpoints_to_model,
)
from .store import Store, get_store

logger = logging.getLogger(__name__)

__all__ = ["create_app", "PlanRequest", "ExportRequest"]

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional extra
    raise ImportError("the REST API needs the 'api' extra: pip install 'weldforge[api]'") from exc


MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_SUFFIXES = BREP_SUFFIXES | MESH_SUFFIXES


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


class PlanRequest(BaseModel):
    """Everything a client may choose about a planning run."""

    part_id: str = Field(..., description="Part returned by POST /parts")
    cell: str = Field("bench_cell", description="Cell from GET /cells")
    name: str = ""
    base_material: str = Field("S355", description="Grade from GET /materials")
    wire_diameter_mm: float | None = None
    leg_thickness_ratio: float = Field(
        0.7, description="Fillet leg as a fraction of the thinner member"
    )
    target_leg_mm: float | None = None
    # Stage switches, so a client can get seams fast and plan motion later.
    do_placement: bool = True
    do_paths: bool = True
    do_simulate: bool = True
    max_seams: int | None = None
    min_seam_length_mm: float = 0.0
    # Placement search effort. The defaults are a compromise; raise for a
    # final plan, lower for a first look.
    placement_grid_steps: int = 5
    placement_yaw_steps: int = 8
    placement_orientations: int = 12
    distortion_weight: float = 150.0
    selected_seam_ids: list[str] | None = Field(
        None, description="Weld only these seams; omit for all of them"
    )


class ExportRequest(BaseModel):
    target: str = Field(..., description="Controller from GET /targets")
    program_name: str = "WELDJOB"
    machine_profile: str | None = Field(
        None, description="Power source from GET /machines, for the setpoint table"
    )
    include_comments: bool = True
    skip_unreachable: bool = True


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------


def create_app(store: Store | None = None) -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="WeldForge",
        version="0.1.0",
        description=(
            "From 3D model to robotic weld: seam recognition, welding procedure "
            "generation, part placement, kinematics and robot program output."
        ),
    )
    state = store if store is not None else get_store()

    # -- reference data ------------------------------------------------------

    @app.get("/health", tags=["service"])
    def health() -> dict[str, Any]:
        """Liveness plus what this server can actually do.

        ``cad_backend`` is the one a client must check: without an
        OpenCASCADE binding the server can still handle meshes but cannot read
        STEP, and that is better discovered here than on upload.
        """
        return {
            "status": "ok",
            "version": app.version,
            "cad_backend": backend_name(),
            "brep_supported": backend_name() is not None,
            "formats": sorted(ALLOWED_SUFFIXES),
            "jobs_held": len(state.list_records()),
        }

    @app.get("/cells", tags=["reference"])
    def list_cells() -> dict[str, Any]:
        out = []
        for name in available_cells():
            cell = load_cell(name)
            out.append(
                {
                    "key": name,
                    "name": cell.name,
                    "robot": cell.robot_name,
                    "table_z_mm": cell.fixture.table_z_mm,
                    "table_size_mm": list(cell.fixture.table_size_mm),
                    "clamps": len(cell.fixture.clamps),
                    "external_axes": [a.name for a in cell.external_axes],
                }
            )
        return {"cells": out}

    @app.get("/robots", tags=["reference"])
    def list_robots() -> dict[str, Any]:
        return {
            "robots": [{"key": name, **load_robot(name).summary()} for name in available_robots()]
        }

    @app.get("/machines", tags=["reference"])
    def list_machines() -> dict[str, Any]:
        out = []
        for name in available_machines():
            profile = load_machine(name)
            out.append(
                {
                    "key": name,
                    "name": profile.name,
                    "vendor": profile.vendor,
                    "interface": profile.interface,
                    "synergic": profile.is_synergic,
                    "description": profile.description,
                }
            )
        return {"machines": out}

    @app.get("/procedures", tags=["reference"])
    def list_procedures() -> dict[str, Any]:
        return {"procedures": available_procedures()}

    @app.get("/materials", tags=["reference"])
    def list_materials() -> dict[str, Any]:
        return {
            "materials": [
                {
                    "key": key,
                    "group": data.group,
                    "procedure": data.procedure,
                    "yield_mpa": data.yield_mpa,
                    "cev": data.cev,
                    "heat_input_max_kj_mm": data.heat_input_max_kj_mm,
                    "interpass_max_c": data.interpass_max_c,
                }
                for key, data in sorted(load_materials().items())
            ]
        }

    @app.get("/targets", tags=["reference"])
    def list_targets() -> dict[str, Any]:
        return {
            "targets": [
                {
                    "key": key,
                    "name": get_postprocessor(key).target,
                    "units": get_postprocessor(key).units,
                    "extension": get_postprocessor(key).extension,
                }
                for key in available_targets()
            ]
        }

    # -- parts ---------------------------------------------------------------

    @app.post("/parts", response_model=PartModel, tags=["parts"])
    async def upload_part(
        file: UploadFile = File(..., description="STEP, IGES, BREP, STL, OBJ or PLY"),
        name: str = Form(""),
        edge_spacing_mm: float = Form(2.0),
        mesh_deflection_mm: float = Form(0.5),
    ) -> PartModel:
        """Upload a CAD file and get back its measured geometry."""
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                415,
                f"unsupported format {suffix!r}; expected one of {sorted(ALLOWED_SUFFIXES)}",
            )
        data = await file.read()
        if not data:
            raise HTTPException(400, "the uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"file is {len(data) / 1e6:.0f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB",
            )
        if suffix in BREP_SUFFIXES and backend_name() is None:
            raise HTTPException(
                503,
                "this server has no OpenCASCADE backend, so it cannot read B-rep "
                "files. Install cadquery-ocp, or upload a mesh (STL/OBJ/PLY).",
            )

        import uuid

        upload_id = f"upload_{uuid.uuid4().hex[:10]}"
        path = state.save_upload(upload_id, file.filename or "part.step", data)
        try:
            part = load_part(
                path,
                LoadOptions(
                    edge_spacing_mm=edge_spacing_mm,
                    mesh_deflection_mm=mesh_deflection_mm,
                ),
                name or None,
            )
        except Exception as exc:
            logger.exception("upload failed")
            raise HTTPException(422, f"could not read the CAD file: {exc}") from exc

        state.put_part(part)
        return part_to_model(part, with_solids=True)

    @app.get("/parts", tags=["parts"])
    def list_parts() -> dict[str, Any]:
        return {"parts": [part_to_model(p).model_dump() for p in state.list_parts()]}

    @app.get("/parts/{part_id}", response_model=PartModel, tags=["parts"])
    def get_part(part_id: str) -> PartModel:
        part = _require_part(state, part_id)
        return part_to_model(part, with_solids=True)

    @app.get("/parts/{part_id}/seams", tags=["parts"])
    def get_seams(
        part_id: str,
        with_geometry: bool = Query(
            False, description="Include every station's point, tangent and wire direction"
        ),
        min_length_mm: float = Query(5.0),
        min_confidence: float = Query(0.25),
    ) -> dict[str, Any]:
        """Recognise the seams. Cached on the part, so repeat calls are free."""
        part = _require_part(state, part_id)
        cached = part.metadata.get("_seam_cache")
        if cached is None:
            options = DetectOptions()
            options.min_length_mm = min_length_mm
            options.min_confidence = min_confidence
            seams, report = detect_seams(part, options)
            cached = (seams, report)
            part.metadata["_seam_cache"] = cached
        seams, report = cached
        return {
            "part_id": part.id,
            "n_seams": len(seams),
            "total_length_mm": round(sum(s.length_mm for s in seams), 2),
            "detection": report.summary(),
            "seams": [seam_to_model(s, with_geometry).model_dump() for s in seams],
        }

    @app.delete("/parts/{part_id}", tags=["parts"])
    def delete_part(part_id: str) -> dict[str, Any]:
        _require_part(state, part_id)
        state._parts.pop(part_id, None)  # noqa: SLF001 - the store is ours
        return {"deleted": part_id}

    # -- jobs ----------------------------------------------------------------

    @app.post("/jobs", tags=["jobs"], status_code=202)
    def create_job(request: PlanRequest, background: BackgroundTasks) -> dict[str, Any]:
        """Start a planning run. Returns immediately; poll ``GET /jobs/{id}``."""
        part = _require_part(state, request.part_id)
        try:
            cell = load_cell(request.cell)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

        import uuid

        job_id = f"job_{uuid.uuid4().hex[:10]}"
        record = state.create_record(
            job_id,
            part=part,
            status="pending",
            options=request.model_dump(),
        )
        background.add_task(_run_plan, state, job_id, part, cell, request)
        return {"job_id": job_id, **record.summary()}

    @app.get("/jobs", tags=["jobs"])
    def list_jobs() -> dict[str, Any]:
        return {"jobs": [r.summary() for r in state.list_records()]}

    @app.get("/jobs/{job_id}", tags=["jobs"])
    def get_job(
        job_id: str,
        with_geometry: bool = Query(False),
        with_points: bool = Query(False),
    ) -> dict[str, Any]:
        record = _require_record(state, job_id)
        payload: dict[str, Any] = record.summary()
        if record.job is not None:
            payload["result"] = job_to_model(record.job, with_geometry, with_points).model_dump()
        return payload

    @app.get("/jobs/{job_id}/report", response_model=ReportModel, tags=["jobs"])
    def get_report(job_id: str) -> ReportModel:
        record = _require_record(state, job_id)
        if record.job is None:
            raise HTTPException(409, f"job {job_id} is {record.status}, not finished")
        return job_to_model(record.job).report

    @app.get("/jobs/{job_id}/setpoints", tags=["jobs"])
    def get_setpoints(job_id: str, machine: str = Query("generic_analog")) -> dict[str, Any]:
        """Machine setpoints for every pass, for a given power source."""
        record = _require_record(state, job_id)
        if record.job is None:
            raise HTTPException(409, f"job {job_id} is {record.status}, not finished")
        try:
            profile = load_machine(machine)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

        out = []
        for seam_id, procedure in record.job.procedures.items():
            group = profile.job_mode and "" or ""
            del group
            material_group = "carbon_steel"
            materials = load_materials()
            if procedure.base_material in materials:
                material_group = materials[procedure.base_material].group
            out.append(
                {
                    "seam_id": seam_id,
                    "machine": profile.name,
                    "synergic": profile.is_synergic,
                    "setpoints": [
                        setpoints_to_model(sp).model_dump()
                        for sp in setpoints_for_procedure(procedure, profile, material_group)
                    ],
                }
            )
        return {"job_id": job_id, "machine": machine, "seams": out}

    @app.post("/jobs/{job_id}/export", tags=["jobs"])
    def export_job(job_id: str, request: ExportRequest) -> dict[str, Any]:
        """Generate a robot program and keep it in the job's workspace."""
        record = _require_record(state, job_id)
        if record.job is None:
            raise HTTPException(409, f"job {job_id} is {record.status}, not finished")
        try:
            post = get_postprocessor(
                request.target,
                PostOptions(
                    program_name=request.program_name,
                    include_comments=request.include_comments,
                    skip_unreachable=request.skip_unreachable,
                ),
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

        result = post.generate(record.job)
        written = {}
        for filename, text in result.files.items():
            path = state.save_export(job_id, filename, text)
            written[filename] = str(path)
        return {
            "job_id": job_id,
            **result.summary(),
            "paths": written,
            "files": {name: text for name, text in result.files.items()},
        }

    @app.get("/jobs/{job_id}/export/{filename}", response_class=PlainTextResponse, tags=["jobs"])
    def download_export(job_id: str, filename: str) -> str:
        record = _require_record(state, job_id)
        path = record.exports.get(Path(filename).name)
        if path is None or not path.exists():
            raise HTTPException(404, f"{filename!r} has not been generated for job {job_id}")
        return path.read_text()

    @app.delete("/jobs/{job_id}", tags=["jobs"])
    def delete_job(job_id: str) -> dict[str, Any]:
        if not state.purge(job_id):
            raise HTTPException(404, f"no job {job_id}")
        return {"deleted": job_id}

    return app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _require_part(state: Store, part_id: str) -> Part:
    part = state.get_part(part_id)
    if part is None:
        raise HTTPException(404, f"no part {part_id}; upload one with POST /parts")
    return part


def _require_record(state: Store, job_id: str):
    record = state.get_record(job_id)
    if record is None:
        raise HTTPException(404, f"no job {job_id}")
    return record


def _plan_options(request: PlanRequest) -> PlanOptions:
    options = PlanOptions(name=request.name)
    options.procedure.base_material = request.base_material
    options.procedure.wire_diameter_mm = request.wire_diameter_mm
    options.procedure.leg_thickness_ratio = request.leg_thickness_ratio
    options.procedure.target_leg_mm = request.target_leg_mm
    options.do_placement = request.do_placement
    options.do_paths = request.do_paths
    options.do_simulate = request.do_simulate
    options.max_seams = request.max_seams
    options.min_seam_length_mm = request.min_seam_length_mm
    options.placement.grid_steps = request.placement_grid_steps
    options.placement.yaw_steps = request.placement_yaw_steps
    options.placement.max_orientations = request.placement_orientations
    options.path.sequence.distortion_weight = request.distortion_weight
    return options


def _run_plan(
    state: Store,
    job_id: str,
    part: Part,
    cell,
    request: PlanRequest,
) -> None:
    """Background planning task.

    Any failure is recorded on the job rather than raised: the client is
    polling, so an exception here would otherwise vanish into the server log
    and leave the job stuck at "running" forever.
    """
    state.update_record(job_id, status="running", stage="planning")
    try:
        options = _plan_options(request)
        job = plan_from_part(part, cell, options, cell_robot(cell))

        if request.selected_seam_ids is not None:
            wanted = set(request.selected_seam_ids)
            for seam in job.seams:
                seam.selected = seam.id in wanted

        job.id = job_id
        state.update_record(
            job_id,
            status="done",
            stage="finished",
            job=job,
            finished_at=__import__("time").time(),
        )
        logger.info("job %s finished: %s", job_id, "ok" if job.report.ok else "with problems")
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        state.update_record(
            job_id,
            status="failed",
            stage="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=__import__("time").time(),
        )
        logger.debug("traceback:\n%s", traceback.format_exc())
