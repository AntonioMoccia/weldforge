"""Command-line interface.

Built on ``argparse`` rather than a CLI framework on purpose: the CLI is what
gets run on a shop-floor PC and in CI, and it should work from a bare Python
install with nothing but the core dependencies.

.. code-block:: text

    weldforge info                       what this installation can do
    weldforge doctor part.step           why an import failed, with the traceback
    weldforge seams part.step            recognise seams and print them
    weldforge wps part.step              seams plus a welding procedure sheet
    weldforge plan part.step -o out/     full plan, report and robot programs
    weldforge serve                      run the REST API

Output is plain aligned text so it pipes, greps and pastes into an email.
``--json`` switches any command to machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _table(rows: list[list[Any]], headers: list[str]) -> str:
    """Left-aligned text table, no dependencies."""
    if not rows:
        return "  (nothing to show)"
    text = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in text), default=0)) for i in range(len(headers))
    ]
    lines = ["  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for row in text:
        lines.append("  " + "  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def _emit(payload: dict[str, Any], as_json: bool, text: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    from . import __version__
    from .cad import backend_name
    from .fixture import available_cells
    from .kin import available_robots
    from .post import available_targets
    from .wps import available_machines, available_procedures, load_materials

    backend = backend_name()
    payload = {
        "version": __version__,
        "cad_backend": backend,
        "brep_supported": backend is not None,
        "cells": available_cells(),
        "robots": available_robots(),
        "machines": available_machines(),
        "procedures": available_procedures(),
        "materials": sorted(load_materials()),
        "export_targets": available_targets(),
    }
    lines = [
        f"WeldForge {payload['version']}",
        "",
        f"  CAD backend     : {backend or 'NONE -- B-rep files cannot be read'}",
        f"  B-rep support   : {'yes' if backend else 'no (install cadquery-ocp)'}",
        "",
        "  cells           : " + ", ".join(payload["cells"]),
        "  robots          : " + ", ".join(payload["robots"]),
        "  power sources   : " + ", ".join(payload["machines"]),
        "  procedure data  : " + ", ".join(payload["procedures"]),
        "  materials       : " + ", ".join(payload["materials"]),
        "  export targets  : " + ", ".join(payload["export_targets"]),
    ]
    _emit(payload, args.json, "\n".join(lines))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report the installation, and optionally try to import one CAD file.

    This is the command to reach for when an import fails in the studio: the
    studio is a GUI entry point with no console, so the traceback that
    explains the failure only ever reaches the log file. Here it is printed.
    """
    import traceback

    from .diagnostics import format_report, log_file, report, setup_logging

    setup_logging("doctor", console=False)
    data = report()
    payload: dict[str, Any] = dict(data)
    if not args.json:
        print(format_report(data))

    missing = [name for name, entry in data["packages"].items() if not entry["importable"]]
    exit_code = 1 if missing else 0

    if args.file:
        from .cad import LoadOptions, load_part

        path = Path(args.file)
        if not args.json:
            print(f"\nImporting {path}")
        try:
            part = load_part(path, LoadOptions(with_mesh=not args.no_mesh))
        except Exception as exc:
            detail = traceback.format_exc()
            exit_code = 1
            logging.getLogger("weldforge").error("doctor: import failed\n%s", detail)
            payload["import"] = {
                "file": str(path),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": detail,
            }
            if not args.json:
                print(f"\n  FAILED: {type(exc).__name__}: {exc}\n")
                print(detail)
                print(f"  Also written to {log_file('doctor')}")
        else:
            payload["import"] = {
                "file": str(path),
                "ok": True,
                "name": part.name,
                "geometry_kind": part.metadata.get("geometry_kind"),
                "solids": len(part.solids),
                "faces": len(part.faces),
                "edges": len(part.edges),
                "mass_kg": round(part.mass_kg, 3),
                "bbox_mm": [round(float(v), 2) for v in part.bbox_size],
                "load_seconds": part.metadata.get("load_seconds"),
            }
            if not args.json:
                got = payload["import"]
                size = got["bbox_mm"]
                print(
                    f"\n  OK: {got['solids']} solids, {got['faces']} faces, "
                    f"{got['edges']} edges, {got['mass_kg']} kg, "
                    f"{size[0]} x {size[1]} x {size[2]} mm "
                    f"in {got['load_seconds']}s"
                )

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    elif missing:
        print(f"\nMissing: {', '.join(missing)}  --  run  weldforge-setup install")
    return exit_code


def cmd_seams(args: argparse.Namespace) -> int:
    import numpy as np

    from .cad import LoadOptions, load_part
    from .seams import DetectOptions, detect_seams, explain_no_seams

    part = load_part(args.file, LoadOptions(edge_spacing_mm=args.spacing))
    options = DetectOptions()
    options.min_length_mm = args.min_length
    options.min_confidence = args.min_confidence
    seams, report = detect_seams(part, options)

    rows = [
        [
            i,
            s.joint_type.value,
            f"{s.length_mm:.1f}",
            f"{np.rad2deg(s.mean_dihedral):.0f}",
            s.dominant_position.value,
            f"{s.thickness_a_mm:.1f}/{s.thickness_b_mm:.1f}",
            f"{s.gap_mm:.1f}",
            f"{s.confidence:.2f}",
            "curved" if not s.is_straight else "",
        ]
        for i, s in enumerate(seams, start=1)
    ]
    headers = ["#", "joint", "len mm", "groove", "pos", "thk mm", "gap", "conf", ""]

    lines = [
        f"{part.name}: {len(part.solids)} solids, {len(part.faces)} faces, "
        f"{part.mass_kg:.1f} kg",
        "",
        _table(rows, headers),
        "",
        f"  {len(seams)} seams, {sum(s.length_mm for s in seams):.0f} mm of weld "
        f"in {report.seconds:.2f}s",
    ]
    if not seams:
        # Zero seams with no reason given reads as a broken importer.
        lines += ["", f"  why: {explain_no_seams(part, report, options)}"]
    text = "\n".join(lines)
    payload = {
        "part": part.summary(),
        "detection": report.summary(),
        "seams": [s.summary() for s in seams],
    }
    _emit(payload, args.json, text)
    return 0


def cmd_wps(args: argparse.Namespace) -> int:
    from .cad import LoadOptions, load_part
    from .seams import DetectOptions, detect_seams
    from .wps import ProcedureOptions, generate_procedures

    part = load_part(args.file, LoadOptions(edge_spacing_mm=args.spacing))
    options = DetectOptions()
    options.min_length_mm = args.min_length
    seams, _ = detect_seams(part, options)
    if args.max_seams:
        seams = seams[: args.max_seams]

    procedures = generate_procedures(
        seams,
        ProcedureOptions(
            base_material=args.material,
            leg_thickness_ratio=args.leg_ratio,
            wire_diameter_mm=args.wire,
        ),
    )

    blocks = []
    for seam in seams:
        procedure = procedures[seam.id]
        rows = [
            [
                p.name,
                f"{p.leg_mm:.1f}" if p.leg_mm else "-",
                f"{p.current_a:.0f}",
                f"{p.voltage_v:.1f}",
                f"{p.wire_speed_m_min:.2f}",
                f"{p.travel_speed_mm_s:.2f}",
                p.transfer_mode.value,
                f"{p.work_angle_deg:.0f}/{p.travel_angle_deg:.0f}",
                p.weave.shape.value if p.weave.active else "-",
                f"{p.heat_input_kj_mm():.2f}",
            ]
            for p in procedure.passes
        ]
        headers = ["pass", "leg", "A", "V", "m/min", "mm/s", "mode", "work/trav", "weave", "kJ/mm"]
        header = (
            f"{seam.label or seam.id}\n"
            f"  {procedure.joint_type.value} in {procedure.position.value}, "
            f"{procedure.thickness_mm:.1f} mm {procedure.base_material}, "
            f"leg {procedure.target_leg_mm:.1f} mm, {procedure.n_passes} pass(es), "
            f"preheat {procedure.preheat_c:.0f} C"
        )
        block = [header, "", _table(rows, headers)]
        if procedure.warnings:
            block.append("")
            block.extend(f"  ! {w}" for w in procedure.warnings)
        block.append("")
        block.append(
            f"  arc time {procedure.total_arc_time_s(seam.length_mm):.1f} s, "
            f"wire {procedure.total_wire_kg(seam.length_mm) * 1000:.0f} g"
        )
        blocks.append("\n".join(block))

    payload = {
        "part": part.summary(),
        "procedures": [procedures[s.id].summary() for s in seams],
    }
    _emit(payload, args.json, ("\n\n" + "=" * 78 + "\n\n").join(blocks))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from .fixture import cell_robot, default_cell, load_cell
    from .plan import PlanOptions, plan_job
    from .post import PostOptions, get_postprocessor

    cell = default_cell() if args.cell == "default" else load_cell(args.cell)
    options = PlanOptions(name=args.name or "")
    options.load.edge_spacing_mm = args.spacing
    options.procedure.base_material = args.material
    options.procedure.leg_thickness_ratio = args.leg_ratio
    options.procedure.wire_diameter_mm = args.wire
    options.min_seam_length_mm = args.min_length
    options.max_seams = args.max_seams
    options.placement.grid_steps = args.grid
    options.placement.yaw_steps = args.yaw
    options.path.sequence.distortion_weight = args.distortion

    job = plan_job(args.file, cell, options, cell_robot(cell))
    report = job.report

    lines = [
        f"{job.name}  ({job.id})",
        f"  cell            : {cell.name} / {cell.robot_name}",
        f"  seams           : {len(job.seams)}  ({job.total_weld_length_mm:.0f} mm of weld)",
        f"  passes          : {sum(p.n_passes for p in job.procedures.values())}",
    ]
    if job.placement is not None:
        placement = job.placement.summary()
        lines += [
            f"  placement       : {placement['translation_mm']} mm, "
            f"rpy {placement['rpy_deg']} deg  (score {placement['score']:.3f})",
            f"  positions       : {placement['position_mix']}",
        ]
        for violation in placement["violations"]:
            lines.append(f"    ! {violation}")
    lines += [
        "",
        f"  VERDICT         : {'OK' if report.ok else 'PROBLEMS'}",
        f"  reachable       : {report.reachable}",
        f"  collision free  : {report.collision_free}",
        f"  cycle time      : {report.cycle_time_s:.1f} s "
        f"(arc {report.arc_time_s:.1f} s, air {report.air_time_s:.1f} s, "
        f"duty {report.duty_cycle:.0%})",
        f"  consumables     : {report.wire_consumption_kg * 1000:.0f} g wire, "
        f"{report.gas_consumption_l:.0f} l gas",
    ]
    for error in report.errors[:15]:
        lines.append(f"    ERROR   {error}")
    for warning in report.warnings[:15]:
        lines.append(f"    warning {warning}")

    written: dict[str, str] = {}
    if args.target:
        out_dir = Path(args.out) if args.out else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        for target in args.target:
            post = get_postprocessor(target, PostOptions(program_name=args.program_name))
            result = post.generate(job)
            for filename, text in result.files.items():
                path = out_dir / filename
                path.write_text(text)
                written[filename] = str(path)
            if result.warnings:
                for warning in result.warnings[:5]:
                    lines.append(f"    {target}: {warning}")
        lines.append("")
        lines.append("  written:")
        for path in written.values():
            lines.append(f"    {path}")

    payload = job.summary() | {"exports": written}
    _emit(payload, args.json, "\n".join(lines))
    return 0 if report.ok else 2


def cmd_preview(args: argparse.Namespace) -> int:
    from .fixture import cell_robot, default_cell, load_cell
    from .plan import PlanOptions, plan_job
    from .viz import scene_from_job, write_html

    cell = default_cell() if args.cell == "default" else load_cell(args.cell)
    options = PlanOptions(name=args.name or "")
    options.procedure.base_material = args.material
    options.min_seam_length_mm = args.min_length
    options.max_seams = args.max_seams
    options.placement.grid_steps = args.grid
    options.placement.yaw_steps = args.yaw
    options.do_paths = not args.seams_only
    options.do_simulate = not args.seams_only

    robot = cell_robot(cell)
    job = plan_job(args.file, cell, options, robot)
    scene = scene_from_job(job, robot, max_frames=args.frames)
    path = write_html(scene, args.out, title=job.name)

    report = job.report
    text = "\n".join(
        [
            f"{job.name}: {len(job.seams)} seams, {len(job.paths)} passes",
            f"  verdict        : {'OK' if report.ok else 'PROBLEMS'}",
            f"  animation      : {len(scene.frames)} frames",
            f"  written        : {path}",
            "",
            f"  open it with:  open {path}"
            if sys.platform == "darwin"
            else f"  open it with:  xdg-open {path}",
        ]
    )
    _emit(job.summary() | {"preview": str(path), "frames": len(scene.frames)}, args.json, text)
    return 0 if report.ok else 2


def cmd_urdf(args: argparse.Namespace) -> int:
    from .fixture import cell_robot, load_cell
    from .kin import available_robots, load_robot, torch_frame
    from .kin.urdf import cell_to_srdf, cell_to_urdf, robot_to_urdf, write_urdf

    out_dir = Path(args.out)
    written: dict[str, str] = {}

    if args.cell:
        cell = load_cell(args.cell)
        robot = cell_robot(cell)
        path = write_urdf(cell_to_urdf(cell, robot), out_dir / f"{args.cell}.urdf")
        written[args.cell] = str(path)
        # The SRDF names the URDF's links, so the two are generated together.
        srdf = write_urdf(cell_to_srdf(cell, robot, name=args.cell), out_dir / f"{args.cell}.srdf")
        written[f"{args.cell}.srdf"] = str(srdf)
    else:
        names = [args.robot] if args.robot else available_robots()
        for name in names:
            robot = load_robot(name, tool=torch_frame())
            path = write_urdf(robot_to_urdf(robot), out_dir / f"{name}.urdf")
            written[name] = str(path)

    text = "written:\n" + "\n".join(f"  {p}" for p in written.values())
    _emit({"written": written}, args.json, text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "the server needs the 'api' extra: pip install 'weldforge[api]'",
            file=sys.stderr,
        )
        return 1

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weldforge",
        description="From 3D model to robotic weld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="repeat for more logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="what this installation can do")
    info.set_defaults(func=cmd_info)

    doctor = sub.add_parser(
        "doctor",
        help="check the installation, and try importing a file to see the real error",
    )
    doctor.add_argument(
        "file", nargs="?", help="optional CAD file to attempt, printing the full traceback"
    )
    doctor.add_argument(
        "--no-mesh", action="store_true", help="skip tessellation to isolate a reader failure"
    )
    doctor.set_defaults(func=cmd_doctor)

    def add_load_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("file", help="STEP, IGES, BREP, STL, OBJ or PLY file")
        p.add_argument("--spacing", type=float, default=2.0, help="seam station spacing in mm")
        p.add_argument(
            "--min-length", type=float, default=5.0, help="ignore seams shorter than this"
        )

    seams = sub.add_parser("seams", help="recognise weld seams")
    add_load_args(seams)
    seams.add_argument("--min-confidence", type=float, default=0.25)
    seams.set_defaults(func=cmd_seams)

    wps = sub.add_parser("wps", help="generate welding procedures")
    add_load_args(wps)
    wps.add_argument("--material", default="S355")
    wps.add_argument("--leg-ratio", type=float, default=0.7)
    wps.add_argument("--wire", type=float, default=None, help="wire diameter in mm")
    wps.add_argument("--max-seams", type=int, default=None)
    wps.set_defaults(func=cmd_wps)

    plan = sub.add_parser("plan", help="plan a whole job and export programs")
    add_load_args(plan)
    plan.add_argument("--cell", default="bench_cell")
    plan.add_argument("--material", default="S355")
    plan.add_argument("--leg-ratio", type=float, default=0.7)
    plan.add_argument("--wire", type=float, default=None)
    plan.add_argument("--max-seams", type=int, default=None)
    plan.add_argument("--name", default=None)
    plan.add_argument("--grid", type=int, default=5, help="placement grid steps")
    plan.add_argument("--yaw", type=int, default=8, help="placement yaw steps")
    plan.add_argument(
        "--distortion",
        type=float,
        default=150.0,
        help="mm of air travel traded for heat balance; 0 = fastest cycle",
    )
    plan.add_argument(
        "-t",
        "--target",
        action="append",
        help="export target (repeatable): rapid, krl, urscript, fanuc, ros2",
    )
    plan.add_argument("-o", "--out", default=None, help="directory for the programs")
    plan.add_argument("--program-name", default="WELDJOB")
    plan.set_defaults(func=cmd_plan)

    preview = sub.add_parser(
        "preview",
        help="plan a part and write an interactive 3D preview as one HTML file",
        description=(
            "Writes a self-contained HTML file showing the part, the cell, the "
            "detected seams with their wire direction, the torch path and the "
            "robot animated along it. No ROS, no install, no network -- it opens "
            "in a browser."
        ),
    )
    add_load_args(preview)
    preview.add_argument("--cell", default="bench_cell")
    preview.add_argument("--material", default="S355")
    preview.add_argument("--name", default=None)
    preview.add_argument("--max-seams", type=int, default=None)
    preview.add_argument("--grid", type=int, default=5)
    preview.add_argument("--yaw", type=int, default=8)
    preview.add_argument("--frames", type=int, default=400, help="animation frames to keep")
    preview.add_argument(
        "--seams-only",
        action="store_true",
        help="skip motion planning: show the seams and stop",
    )
    preview.add_argument("-o", "--out", default="preview.html")
    preview.set_defaults(func=cmd_preview)

    urdf = sub.add_parser(
        "urdf",
        help="generate URDF from the robot and cell data",
        description=(
            "The URDF is generated from data/robots/*.yaml so the ROS model and "
            "the planner's kinematics cannot drift apart. Regenerate after "
            "changing either."
        ),
    )
    urdf.add_argument("--robot", default=None, help="one robot; omit for all of them")
    urdf.add_argument("--cell", default=None, help="a whole cell instead of a bare arm")
    urdf.add_argument("-o", "--out", default="urdf", help="output directory")
    urdf.set_defaults(func=cmd_urdf)

    serve = sub.add_parser("serve", help="run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
