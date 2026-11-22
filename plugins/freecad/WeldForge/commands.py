"""FreeCAD commands: the buttons on the WeldForge toolbar.

The workflow is split into separate commands rather than one "do everything"
button, because the stages have very different costs and a user wants to stop
between them:

* **Detect seams** is seconds, and is what a designer runs while still editing
  the model -- it shows what will and will not be recognised as a joint.
* **Plan job** is tens of seconds to minutes, and only makes sense once the
  geometry is settled.
* **Export program** is instant, and gets run repeatedly for different
  controllers.

Each command reports through FreeCAD's own console and status bar, and
long-running ones run the HTTP call with ``processEvents`` so the window does
not appear frozen.
"""

from __future__ import annotations

import os

from .exporter import ExportError, describe_selection, export_step, selected_objects
from .overlay import clear_overlay, draw_placement, draw_seams
from .weldforge_client import WeldForgeClient, WeldForgeError

__all__ = [
    "state",
    "register_commands",
    "WeldForgeCheckServer",
    "WeldForgeDetectSeams",
    "WeldForgePlanJob",
    "WeldForgeExportProgram",
    "WeldForgeClearOverlay",
]

ICON_DIR = os.path.join(os.path.dirname(__file__), "icons")


class _State:
    """What the commands share between invocations.

    Module-level state rather than a document property: a plan is not part of
    the user's CAD data and should not be saved into their file.
    """

    def __init__(self) -> None:
        self.client = WeldForgeClient()
        self.part_id: str | None = None
        self.job_id: str | None = None
        self.seams: list[dict] = []
        self.last_report: dict | None = None
        self.cell: str = "bench_cell"
        self.material: str = "S355"
        self.leg_ratio: float = 0.7

    def reset_part(self) -> None:
        self.part_id = None
        self.job_id = None
        self.seams = []
        self.last_report = None


state = _State()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _console():
    import FreeCAD

    return FreeCAD.Console


def _say(message: str) -> None:
    _console().PrintMessage(f"[WeldForge] {message}\n")


def _warn(message: str) -> None:
    _console().PrintWarning(f"[WeldForge] {message}\n")


def _error(message: str) -> None:
    _console().PrintError(f"[WeldForge] {message}\n")


def _pump() -> None:
    """Let the window repaint during a long call."""
    try:
        from PySide import QtGui

        QtGui.QApplication.processEvents()
    except Exception:
        try:
            from PySide2 import QtWidgets

            QtWidgets.QApplication.processEvents()
        except Exception:
            pass


def _dialog(title: str, message: str, kind: str = "info") -> None:
    try:
        from PySide import QtGui

        box = QtGui.QMessageBox()
        box.setWindowTitle(title)
        box.setText(message)
        box.setIcon(
            {
                "info": QtGui.QMessageBox.Information,
                "warning": QtGui.QMessageBox.Warning,
                "error": QtGui.QMessageBox.Critical,
            }.get(kind, QtGui.QMessageBox.Information)
        )
        box.exec_()
    except Exception:
        _say(f"{title}: {message}")


def _ensure_part() -> str:
    """Export the selection and upload it, reusing the last upload if unchanged."""
    objects = selected_objects()
    _say(f"exporting {describe_selection(objects)}")
    path = export_step(objects)
    _pump()
    try:
        part = state.client.upload_part(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    state.reset_part()
    state.part_id = part["id"]
    _say(
        f"uploaded {part['name']}: {part['n_solids']} solids, {part['n_faces']} faces, "
        f"{part['mass_kg']:.1f} kg, thicknesses {part['thicknesses_mm']} mm"
    )
    return state.part_id


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


class _Command:
    """Base for a FreeCAD command, with error handling in one place."""

    name = ""
    menu = ""
    tooltip = ""
    icon = ""

    def GetResources(self) -> dict:  # noqa: N802 - FreeCAD's API
        resources = {"MenuText": self.menu, "ToolTip": self.tooltip}
        icon = os.path.join(ICON_DIR, self.icon) if self.icon else ""
        if icon and os.path.exists(icon):
            resources["Pixmap"] = icon
        return resources

    def IsActive(self) -> bool:  # noqa: N802
        import FreeCAD

        return FreeCAD.ActiveDocument is not None

    def Activated(self) -> None:  # noqa: N802
        try:
            self.run()
        except ExportError as exc:
            _warn(str(exc))
            _dialog("WeldForge", str(exc), "warning")
        except WeldForgeError as exc:
            _error(str(exc))
            _dialog("WeldForge", str(exc), "error")
        except Exception as exc:  # pragma: no cover - last resort in a GUI
            import traceback

            _error(f"{type(exc).__name__}: {exc}")
            _console().PrintError(traceback.format_exc())
            _dialog("WeldForge", f"{type(exc).__name__}: {exc}", "error")

    def run(self) -> None:
        raise NotImplementedError


class WeldForgeCheckServer(_Command):
    name = "WeldForge_CheckServer"
    menu = "Check server"
    tooltip = "Check that the WeldForge server is reachable and what it supports"
    icon = "server.svg"

    def IsActive(self) -> bool:  # noqa: N802
        return True

    def run(self) -> None:
        health = state.client.health()
        cells = ", ".join(c["key"] for c in state.client.cells())
        message = (
            f"WeldForge {health.get('version', '?')} at {state.client.base_url}\n\n"
            f"CAD backend: {health.get('cad_backend') or 'NONE'}\n"
            f"B-rep (STEP) support: {'yes' if health.get('brep_supported') else 'no'}\n"
            f"Cells: {cells}"
        )
        _say(message.replace("\n", " | "))
        _dialog("WeldForge server", message)


class WeldForgeDetectSeams(_Command):
    name = "WeldForge_DetectSeams"
    menu = "Detect weld seams"
    tooltip = (
        "Export the selection, recognise its weld seams, and draw them on the model"
    )
    icon = "seams.svg"

    def run(self) -> None:
        part_id = _ensure_part()
        _pump()
        payload = state.client.seams(part_id, with_geometry=True)
        state.seams = payload["seams"]

        drawn = draw_seams(state.seams)
        total = payload["total_length_mm"]
        kinds: dict[str, int] = {}
        for seam in state.seams:
            kinds[seam["joint_type"]] = kinds.get(seam["joint_type"], 0) + 1

        summary = (
            f"{payload['n_seams']} seams, {total:.0f} mm of weld\n\n"
            + "\n".join(f"  {count} x {kind}" for kind, count in sorted(kinds.items()))
        )
        doubtful = [s for s in state.seams if s["confidence"] < 0.5]
        if doubtful:
            summary += (
                f"\n\n{len(doubtful)} seam(s) are doubtful and drawn dashed. "
                "Check those before planning."
            )
        _say(f"detected {payload['n_seams']} seams, {total:.0f} mm, drew {drawn}")
        _dialog("Weld seams", summary)


class WeldForgePlanJob(_Command):
    name = "WeldForge_PlanJob"
    menu = "Plan weld job"
    tooltip = (
        "Generate welding procedures, choose where the part goes, build the torch "
        "paths and simulate the cycle"
    )
    icon = "plan.svg"

    def run(self) -> None:
        part_id = state.part_id or _ensure_part()
        _say(
            f"planning on cell {state.cell} in {state.material}, "
            f"leg ratio {state.leg_ratio}"
        )
        job_id = state.client.plan(
            part_id,
            cell=state.cell,
            base_material=state.material,
            leg_thickness_ratio=state.leg_ratio,
        )
        state.job_id = job_id

        def progress(status: dict) -> None:
            _say(f"job {job_id}: {status['status']} ({status['elapsed_s']:.0f}s)")
            _pump()

        status = state.client.wait_for_job(job_id, on_progress=progress)
        if status["status"] == "failed":
            raise WeldForgeError(f"planning failed: {status.get('error')}")

        result = status["result"]
        report = result["report"]
        state.last_report = report

        if result.get("placement"):
            draw_placement(result["placement"])

        verdict = "OK" if report["ok"] else "PROBLEMS"
        lines = [
            f"Verdict: {verdict}",
            "",
            f"Seams            {len(result['seams'])}",
            f"Passes           {sum(len(p['passes']) for p in result['procedures'])}",
            f"Weld length      {report['total_weld_length_mm']:.0f} mm",
            f"Cycle time       {report['cycle_time_s']:.0f} s "
            f"(arc {report['arc_time_s']:.0f} s, duty {report['duty_cycle']:.0%})",
            f"Wire / gas       {report['wire_consumption_kg'] * 1000:.0f} g / "
            f"{report['gas_consumption_l']:.0f} l",
        ]
        placement = result.get("placement")
        if placement:
            lines += [
                "",
                f"Placement        {placement['position_mix']}",
                f"Reachable        {placement['n_reachable']}/"
                f"{placement['n_reachable'] + placement['n_unreachable']} seams",
            ]
        if report["errors"]:
            lines += ["", "Errors:"] + [f"  - {e}" for e in report["errors"][:6]]
        if report["warnings"]:
            lines += ["", "Warnings:"] + [f"  - {w}" for w in report["warnings"][:6]]

        _say(" | ".join(lines[:6]))
        _dialog("Weld job", "\n".join(lines), "info" if report["ok"] else "warning")


class WeldForgeExportProgram(_Command):
    name = "WeldForge_ExportProgram"
    menu = "Export robot program"
    tooltip = "Write the planned job as a program for a robot controller"
    icon = "export.svg"

    def IsActive(self) -> bool:  # noqa: N802
        return state.job_id is not None

    def run(self) -> None:
        if state.job_id is None:
            raise ExportError("plan a job first")

        targets = [t["key"] for t in state.client.targets()]
        target = _choose(
            "Export robot program", "Controller:", targets, default="rapid"
        )
        if target is None:
            return

        directory = _choose_directory()
        if not directory:
            return

        result = state.client.export(state.job_id, target)
        written = []
        for filename, text in result["files"].items():
            path = os.path.join(directory, filename)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            written.append(path)

        for warning in result.get("warnings", []):
            _warn(warning)
        _say(f"wrote {len(written)} file(s) for {target}")
        _dialog(
            "Robot program",
            f"{result['moves']} moves written for {target}:\n\n"
            + "\n".join(f"  {p}" for p in written)
            + "\n\nDry-run it at reduced speed with the arc disabled before welding.",
        )


class WeldForgeClearOverlay(_Command):
    name = "WeldForge_ClearOverlay"
    menu = "Clear seam overlay"
    tooltip = "Remove the drawn seams from the 3D view"
    icon = "clear.svg"

    def run(self) -> None:
        clear_overlay()
        _say("overlay cleared")


# ---------------------------------------------------------------------------
# small Qt helpers
# ---------------------------------------------------------------------------


def _choose(title: str, label: str, options: list[str], default: str | None = None):
    try:
        from PySide import QtGui

        index = options.index(default) if default in options else 0
        value, ok = QtGui.QInputDialog.getItem(None, title, label, options, index, False)
        return value if ok else None
    except Exception:
        return default or (options[0] if options else None)


def _choose_directory() -> str:
    try:
        from PySide import QtGui

        return QtGui.QFileDialog.getExistingDirectory(None, "Where to write the program")
    except Exception:
        return os.path.expanduser("~")


def register_commands() -> list[str]:
    """Register every command with FreeCAD; returns their names in toolbar order."""
    import FreeCADGui

    commands = [
        WeldForgeCheckServer(),
        WeldForgeDetectSeams(),
        WeldForgePlanJob(),
        WeldForgeExportProgram(),
        WeldForgeClearOverlay(),
    ]
    for command in commands:
        FreeCADGui.addCommand(command.name, command)
    return [c.name for c in commands]
