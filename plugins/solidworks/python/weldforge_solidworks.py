"""Drive SOLIDWORKS from Python, for people who cannot deploy the C# add-in.

Why this exists. The C# add-in is the better experience -- a toolbar and a
task pane inside SOLIDWORKS -- but it has to be compiled, COM-registered with
administrator rights, and matched to the installed SOLIDWORKS interop version.
In a lot of shops that is a two-week IT request. This script needs only
``pip install pywin32`` and gets the same job done from a command prompt:

.. code-block:: text

    python weldforge_solidworks.py                  the open document
    python weldforge_solidworks.py --file X.SLDASM   a specific file
    python weldforge_solidworks.py --target rapid --out C:\\programs

It talks to SOLIDWORKS over COM exactly as the add-in does, including the two
export settings that matter: **AP214** and **no body flattening**. A weldment's
seams are the contacts between its parts, and a flattened STEP has none.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from weldforge_client import WeldForgeClient, WeldForgeError  # noqa: E402

# SOLIDWORKS enumerations, from swconst. Hard-coded so the script does not need
# the interop assemblies.
SW_DOC_PART = 1
SW_DOC_ASSEMBLY = 2
SW_SAVE_AS_CURRENT_VERSION = 0
SW_SAVE_AS_SILENT = 1
SW_PREF_STEP_AP = 85                  # swStepAP
SW_STEP_AP214 = 214                   # swSTEPAP214
SW_PREF_STEP_SINGLE_BODY = 251        # swStepExportPartAsSingleBody


class SolidWorksError(RuntimeError):
    """SOLIDWORKS is not running, or refused what we asked."""


def connect():
    """Attach to a running SOLIDWORKS, or start one."""
    try:
        import win32com.client
    except ImportError as exc:
        raise SolidWorksError(
            "pywin32 is required: pip install pywin32"
        ) from exc

    try:
        # GetActiveObject attaches to the instance the user already has open,
        # which is almost always what they want -- Dispatch would start a
        # second copy.
        return win32com.client.GetActiveObject("SldWorks.Application")
    except Exception:
        pass
    try:
        import win32com.client

        app = win32com.client.Dispatch("SldWorks.Application")
        app.Visible = True
        return app
    except Exception as exc:
        raise SolidWorksError(
            "could not connect to SOLIDWORKS. Is it installed and running?"
        ) from exc


def open_document(app, path: str):
    errors = 0
    warnings = 0
    extension = os.path.splitext(path)[1].lower()
    doc_type = SW_DOC_ASSEMBLY if extension == ".sldasm" else SW_DOC_PART
    document = app.OpenDoc6(path, doc_type, 0, "", errors, warnings)
    if document is None:
        raise SolidWorksError(f"could not open {path} (error {errors})")
    return document


def export_step(app, document=None) -> str:
    """Save the document as a STEP AP214 assembly with the bodies separate."""
    document = document or app.ActiveDoc
    if document is None:
        raise SolidWorksError(
            "no document is open in SOLIDWORKS. Open the weldment, or pass --file."
        )

    title = os.path.splitext(document.GetTitle())[0] or "weldment"
    path = os.path.join(
        tempfile.gettempdir(), title.replace(" ", "_") + "_weldforge.step"
    )

    previous_format = app.GetUserPreferenceIntegerValue(SW_PREF_STEP_AP)
    previous_flatten = app.GetUserPreferenceToggle(SW_PREF_STEP_SINGLE_BODY)
    try:
        # AP214 keeps the assembly structure.
        app.SetUserPreferenceIntegerValue(SW_PREF_STEP_AP, SW_STEP_AP214)
        # Never fuse the bodies: the contact between them is the weld.
        app.SetUserPreferenceToggle(SW_PREF_STEP_SINGLE_BODY, False)

        errors = 0
        warnings = 0
        ok = document.Extension.SaveAs3(
            path,
            SW_SAVE_AS_CURRENT_VERSION,
            SW_SAVE_AS_SILENT,
            None,
            None,
            errors,
            warnings,
        )
        if not ok or not os.path.exists(path):
            raise SolidWorksError(f"STEP export failed (error {errors})")
        return path
    finally:
        app.SetUserPreferenceIntegerValue(SW_PREF_STEP_AP, previous_format)
        app.SetUserPreferenceToggle(SW_PREF_STEP_SINGLE_BODY, previous_flatten)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the SOLIDWORKS model and plan its welding with WeldForge."
    )
    parser.add_argument("--file", help="document to open; default is the active one")
    parser.add_argument("--url", default=None, help="WeldForge server URL")
    parser.add_argument("--cell", default="bench_cell")
    parser.add_argument("--material", default="S355")
    parser.add_argument("--leg-ratio", type=float, default=0.7)
    parser.add_argument(
        "--target",
        action="append",
        help="robot controller to export for (repeatable)",
    )
    parser.add_argument("--out", default=".", help="where to write the programs")
    parser.add_argument(
        "--seams-only",
        action="store_true",
        help="recognise the seams and stop, without planning motion",
    )
    args = parser.parse_args(argv)

    client = WeldForgeClient(args.url) if args.url else WeldForgeClient()
    if not client.reachable():
        print(
            f"error: no WeldForge server at {client.base_url}. "
            "Start it with: weldforge serve",
            file=sys.stderr,
        )
        return 1

    try:
        app = connect()
        document = open_document(app, args.file) if args.file else None
        print("exporting STEP from SOLIDWORKS...")
        step_path = export_step(app, document)
        print(f"  {step_path}")

        part = client.upload_part(step_path)
        print(
            f"part {part['name']}: {part['n_solids']} solids, {part['n_faces']} faces, "
            f"{part['mass_kg']:.1f} kg, thicknesses {part['thicknesses_mm']} mm"
        )

        seams = client.seams(part["id"], with_geometry=False)
        print(f"\n{seams['n_seams']} seams, {seams['total_length_mm']:.0f} mm of weld")
        print(f"  {'joint':8s} {'mm':>7s} {'pos':>4s} {'conf':>5s}")
        for seam in seams["seams"]:
            print(
                f"  {seam['joint_type']:8s} {seam['length_mm']:7.1f} "
                f"{seam['position']:>4s} {seam['confidence']:5.2f}"
            )
        if args.seams_only:
            return 0

        print("\nplanning...")
        job_id = client.plan(
            part["id"],
            cell=args.cell,
            base_material=args.material,
            leg_thickness_ratio=args.leg_ratio,
        )
        status = client.wait_for_job(
            job_id,
            on_progress=lambda s: print(f"  {s['status']} ({s['elapsed_s']:.0f}s)"),
        )
        if status["status"] == "failed":
            print(f"error: planning failed: {status.get('error')}", file=sys.stderr)
            return 2

        report = status["result"]["report"]
        print(f"\nVERDICT: {'OK' if report['ok'] else 'PROBLEMS'}")
        print(
            f"  cycle {report['cycle_time_s']:.0f} s "
            f"(arc {report['arc_time_s']:.0f} s, duty {report['duty_cycle']:.0%})"
        )
        print(
            f"  {report['total_weld_length_mm']:.0f} mm of weld, "
            f"{report['wire_consumption_kg'] * 1000:.0f} g wire, "
            f"{report['gas_consumption_l']:.0f} l gas"
        )
        for error in report["errors"][:10]:
            print(f"  ERROR   {error}")
        for warning in report["warnings"][:10]:
            print(f"  warning {warning}")

        for target in args.target or []:
            result = client.export(job_id, target)
            os.makedirs(args.out, exist_ok=True)
            for filename, text in result["files"].items():
                path = os.path.join(args.out, filename)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text)
                print(f"  wrote {path}")

        try:
            os.unlink(step_path)
        except OSError:
            pass
        return 0 if report["ok"] else 2

    except (SolidWorksError, WeldForgeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
