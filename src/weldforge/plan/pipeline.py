"""The end-to-end pipeline: a CAD file in, a verified weld job out.

.. code-block:: text

    CAD file
       |  weldforge.cad          load the B-rep, mesh it, measure thicknesses
       v
    Part
       |  weldforge.seams        recognise seams, classify joints
       v
    Seams
       |  weldforge.wps          generate a procedure per seam
       v
    Procedures
       |  weldforge.fixture      choose where and which way up the part goes
       v
    Placement
       |  weldforge.path         torch poses, seam order, inverse kinematics
       v
    Paths
       |  weldforge.sim          reachability, collisions, cycle time, cost
       v
    WeldJob + report
       |  weldforge.post         robot program for the controller in the cell
       v
    Program

Each stage is independently usable and independently testable; this module
only wires them together and records what happened. A stage that fails adds to
the report and the pipeline carries on, because partial results are what let an
operator see *where* the problem is -- a job that stops at "seam detection
failed" tells them nothing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..cad import LoadOptions, load_part
from ..collide import scene_from_cell
from ..fixture import PlacementOptions, cell_robot, default_cell, find_placement
from ..fixture.positioner import PositionerOptions, solve_positioner
from ..kin import RobotModel
from ..model import Part, Seam, WeldCell, WeldJob
from ..path import PathOptions, build_job_paths
from ..seams import DetectOptions, detect_seams
from ..sim import SimOptions, simulate
from ..wps import ProcedureOptions, generate_procedures

logger = logging.getLogger(__name__)

__all__ = ["PlanOptions", "plan_job", "plan_from_part"]


@dataclass
class PlanOptions:
    """Options for every stage, in one place."""

    load: LoadOptions = field(default_factory=LoadOptions)
    detect: DetectOptions = field(default_factory=DetectOptions)
    procedure: ProcedureOptions = field(default_factory=ProcedureOptions)
    placement: PlacementOptions = field(default_factory=PlacementOptions)
    positioner: PositionerOptions = field(default_factory=PositionerOptions)
    path: PathOptions = field(default_factory=PathOptions)
    simulation: SimOptions = field(default_factory=SimOptions)
    # Stage switches, for running part of the pipeline from the API or the CLI.
    do_detect: bool = True
    do_procedures: bool = True
    do_placement: bool = True
    # Solve the positioner when the cell has coupled external axes. Switching
    # it off welds everything at the setup orientation, which is what a
    # bench-only cell does anyway.
    do_positioner: bool = True
    do_paths: bool = True
    do_simulate: bool = True
    # Cap the seams carried forward, longest first. A first look at a 200-seam
    # weldment does not need all of them, and path building is the expensive
    # stage.
    max_seams: int | None = None
    min_seam_length_mm: float = 0.0
    name: str = ""

    @classmethod
    def quick_look(cls) -> PlanOptions:
        """Detection and procedures only -- seconds, not minutes."""
        return cls(do_placement=False, do_paths=False, do_simulate=False)

    @classmethod
    def full(cls) -> PlanOptions:
        return cls()


def plan_job(
    cad_path: str | Path,
    cell: WeldCell | None = None,
    options: PlanOptions | None = None,
    robot: RobotModel | None = None,
) -> WeldJob:
    """Run the whole pipeline on a CAD file."""
    opts = options or PlanOptions()
    part = load_part(cad_path, opts.load, opts.name or None)
    return plan_from_part(part, cell, opts, robot)


def plan_from_part(
    part: Part,
    cell: WeldCell | None = None,
    options: PlanOptions | None = None,
    robot: RobotModel | None = None,
    seams: list[Seam] | None = None,
) -> WeldJob:
    """Run the pipeline on an already-loaded part.

    *seams* plans exactly those and skips detection. This is what an
    application passes: the seams the operator settled on, which include the
    ones they traced by hand and exclude the ones they switched off. Without
    it the planner re-detected from the geometry and quietly discarded both --
    a hand-traced joint on a single-body part came back as "no weld seams were
    recognised", and switching a seam off changed nothing about the plan.
    """
    opts = options or PlanOptions()
    work_cell = cell if cell is not None else default_cell()
    arm = robot if robot is not None else cell_robot(work_cell)

    job = WeldJob(
        name=opts.name or part.name,
        part=part,
        cell=work_cell,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    timings: dict[str, float] = {}

    # --- seams --------------------------------------------------------------
    if seams is not None:
        # The caller already knows what is to be welded. Detection would
        # overwrite it.
        job.seams = list(seams)
        job.metadata["seams_given"] = len(job.seams)
    elif opts.do_detect:
        t0 = time.perf_counter()
        seams, detect_report = detect_seams(part, opts.detect)
        timings["detect_s"] = round(time.perf_counter() - t0, 3)
        job.metadata["detection"] = detect_report.summary()

        if opts.min_seam_length_mm > 0.0:
            seams = [s for s in seams if s.length_mm >= opts.min_seam_length_mm]
        if opts.max_seams is not None and len(seams) > opts.max_seams:
            job.metadata["seams_truncated"] = {
                "found": len(seams),
                "kept": opts.max_seams,
                "note": "longest seams kept; raise max_seams for the rest",
            }
            seams = seams[: opts.max_seams]
        job.seams = seams

    if not job.seams:
        job.report.errors.append("no weld seams were recognised")
        job.metadata["timings"] = timings
        return job

    # --- procedures ---------------------------------------------------------
    if opts.do_procedures:
        t0 = time.perf_counter()
        job.procedures = generate_procedures(job.selected_seams, opts.procedure)
        timings["procedures_s"] = round(time.perf_counter() - t0, 3)

    # --- placement ----------------------------------------------------------
    if opts.do_placement:
        t0 = time.perf_counter()
        best, evaluated = find_placement(part, job.selected_seams, work_cell, opts.placement, arm)
        timings["placement_s"] = round(time.perf_counter() - t0, 3)
        job.placement = best
        job.metadata["placement_candidates"] = len(evaluated)
        if best.violations:
            job.report.warnings.extend(f"placement: {v}" for v in best.violations)
        if best.unreachable_seams:
            job.report.warnings.append(
                f"placement leaves {len(best.unreachable_seams)} of "
                f"{len(job.selected_seams)} seams out of reach"
            )

    # --- positioner ---------------------------------------------------------
    if opts.do_positioner and work_cell.external_axes and job.placement is not None:
        t0 = time.perf_counter()
        job.positioner = solve_positioner(
            job.selected_seams,
            work_cell,
            job.placement.pose,
            arm,
            opts.positioner,
        )
        timings["positioner_s"] = round(time.perf_counter() - t0, 3)
        positioner_summary = job.positioner.summary()
        job.metadata["positioner"] = positioner_summary
        improved = int(positioner_summary["improved"])
        if improved:
            job.report.warnings.append(
                f"the positioner improves the welding position of {improved} seam(s): "
                f"{positioner_summary['downhand']}/{positioner_summary['seams']} end up "
                "flat or horizontal-vertical"
            )
        for seam_id in job.positioner.needs_coordinated:
            job.report.warnings.append(
                f"seam {seam_id}: its welding position changes along its own length, "
                "so no single positioner setting fits it. That needs coordinated "
                "motion, which this planner does not generate: split the seam, or "
                "accept the best single orientation."
            )

    # --- paths --------------------------------------------------------------
    scene = scene_from_cell(
        work_cell,
        part.mesh_vertices,
        part.mesh_faces,
        job.placement.pose if job.placement is not None else None,
    )

    if opts.do_paths:
        t0 = time.perf_counter()
        paths, path_report = build_job_paths(job, arm, scene, opts.path)
        timings["paths_s"] = round(time.perf_counter() - t0, 3)
        job.paths = paths
        job.metadata["paths"] = path_report.summary()

    # --- simulation ---------------------------------------------------------
    if opts.do_simulate and job.paths:
        t0 = time.perf_counter()
        # The paths were already checked while being built; re-checking every
        # station here would double the cost for no new information.
        sim_opts = opts.simulation
        sim_opts.check_collisions = False
        job.report = simulate(job, arm, scene, sim_opts)
        timings["simulate_s"] = round(time.perf_counter() - t0, 3)

    job.metadata["timings"] = timings
    logger.info(
        "plan_from_part: %s -> %d seams, %d paths, %s",
        part.name,
        len(job.seams),
        len(job.paths),
        "OK" if job.report.ok else "with problems",
    )
    return job
