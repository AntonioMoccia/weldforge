"""WeldForge Studio: the application shell.

What lives here and what does not. The window owns the project, the 3-D view
and the panels, and routes between them. It does not contain planning logic,
seam maths or tool rules -- those are in modules that run without a display,
which is why they can be tested and this cannot.

Two design points worth stating.

**Long work runs off the UI thread.** Planning a weldment takes from seconds
to minutes. A window that freezes for a minute looks broken, and an operator
who thinks it is broken force-quits it. Every slow action runs in a worker and
reports progress.

**The project is the single source of truth.** Panels show it and ask for
changes; nothing keeps its own copy. That is what makes "every program can be
saved" true rather than approximately true: one object goes to disk, and it
holds everything the operator set up.
"""

from __future__ import annotations

import contextlib
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..geom import transform as tf
from .panels import (
    OutputPanel,
    PlacementPanel,
    ProcedurePanel,
    SeamPanel,
    SetupPanel,
    SimulationPanel,
)
from .project import (
    PROJECT_SUFFIX,
    Project,
    ProjectError,
    RecentFiles,
    attach_part,
    load_project,
    save_project,
)
from .scene3d import LAYERS, Scene3D
from .send import SendTarget, dispatch
from .tools import PlacePartTool, SelectTool, Tool, TraceSeamTool

logger = logging.getLogger(__name__)

__all__ = ["MainWindow", "main"]

CAD_FILTER = "CAD files (*.step *.stp *.iges *.igs *.brep *.stl *.obj *.ply);;All files (*)"
PROJECT_FILTER = f"WeldForge project (*{PROJECT_SUFFIX});;All files (*)"


# ---------------------------------------------------------------------------
# background work
# ---------------------------------------------------------------------------


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)


class _Worker(QtCore.QRunnable):
    """Runs one callable off the UI thread and reports back.

    Exceptions are caught and reported rather than killing the thread pool: a
    planning run that hits bad geometry must leave the application usable.
    """

    def __init__(self, work: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.signals = _WorkerSignals()
        self._work = work
        self._args = args
        self._kwargs = kwargs

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._work(*self._args, **self._kwargs)
        except Exception as exc:
            logger.exception("background task failed")
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            logger.debug(traceback.format_exc())
        else:
            self.signals.finished.emit(result)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------


class MainWindow(QtWidgets.QMainWindow):
    """From a CAD model to a program on the machine, in one window."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("WeldForge Studio")
        self.resize(1680, 1000)

        self.project = Project()
        self.robot: Any | None = None
        self.tool: Tool = SelectTool(self.project)
        self.recent = RecentFiles()
        self.pool = QtCore.QThreadPool.globalInstance()
        self._busy = 0

        self._build_viewport()
        self._build_panels()
        self._build_actions()
        self._build_status()

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(45)
        self._play_timer.timeout.connect(self._advance_frame)
        self._frame = 0

        self._load_cell(self.project.cell_name)
        self._refresh_all()

    # -- construction -------------------------------------------------------

    def _build_viewport(self) -> None:
        import vtk
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(frame)
        layout.addWidget(self.vtk_widget)
        self.setCentralWidget(frame)

        renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(renderer)
        self.scene = Scene3D(renderer)

        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        # Left click goes to the active tool; the camera keeps middle and
        # right, so orbiting never fights picking.
        interactor.AddObserver("LeftButtonPressEvent", self._on_left_click, 1.0)
        self.vtk_widget.Initialize()
        interactor.Start()

    def _build_panels(self) -> None:
        self.setDockOptions(
            QtWidgets.QMainWindow.AnimatedDocks | QtWidgets.QMainWindow.AllowTabbedDocks
        )

        self.setup_panel = SetupPanel()
        self.setup_panel.cellChanged.connect(self._on_cell_changed)
        self.setup_panel.settingsChanged.connect(self._on_settings_changed)

        self.placement_panel = PlacementPanel()
        self.placement_panel.actionRequested.connect(self._on_placement_action)
        self.placement_panel.poseEdited.connect(self._on_pose_edited)
        self.placement_panel.lockToggled.connect(self._on_lock_toggled)

        self.seam_panel = SeamPanel()
        self.seam_panel.actionRequested.connect(self._on_seam_action)
        self.seam_panel.seamSelected.connect(self._on_seam_selected)
        self.seam_panel.seamToggled.connect(self._on_seam_toggled)
        self.seam_panel.deleteRequested.connect(self._on_seam_deleted)
        self.seam_panel.zoomRequested.connect(self._on_zoom_to_seam)

        self.procedure_panel = ProcedurePanel()
        self.procedure_panel.regenerateRequested.connect(self._on_regenerate)
        self.procedure_panel.passEdited.connect(self._on_pass_edited)

        self.simulation_panel = SimulationPanel()
        self.simulation_panel.planRequested.connect(self._on_plan)
        self.simulation_panel.frameChanged.connect(self._on_frame_changed)
        self.simulation_panel.playToggled.connect(self._on_play_toggled)

        self.output_panel = OutputPanel()
        self.output_panel.generateRequested.connect(self._on_generate)
        self.output_panel.sendRequested.connect(self._on_send)
        self.output_panel.browseRequested.connect(self._on_browse_destination)

        left = QtWidgets.QDockWidget("Setup", self)
        left.setObjectName("dock_setup")
        container = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(container)
        box.addWidget(self.setup_panel)
        box.addWidget(self._titled("Placement", self.placement_panel), 1)
        left.setWidget(container)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, left)

        right = QtWidgets.QDockWidget("Weld seams", self)
        right.setObjectName("dock_seams")
        right.setWidget(self.seam_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, right)

        procedure = QtWidgets.QDockWidget("Welding procedure", self)
        procedure.setObjectName("dock_procedure")
        procedure.setWidget(self.procedure_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, procedure)

        simulation = QtWidgets.QDockWidget("Plan and simulate", self)
        simulation.setObjectName("dock_simulation")
        simulation.setWidget(self.simulation_panel)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, simulation)

        output = QtWidgets.QDockWidget("Program", self)
        output.setObjectName("dock_output")
        output.setWidget(self.output_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, output)
        self.tabifyDockWidget(procedure, output)
        procedure.raise_()

        self._docks = {
            "setup": left,
            "seams": right,
            "procedure": procedure,
            "simulation": simulation,
            "output": output,
        }

    @staticmethod
    def _titled(title: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(widget)
        return box

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._act(file_menu, "&New", self.new_project, "Ctrl+N")
        self._act(file_menu, "&Open project...", self.open_project, "Ctrl+O")
        self._act(file_menu, "Import &CAD...", self.import_cad, "Ctrl+I")
        file_menu.addSeparator()
        self._act(file_menu, "&Save", self.save, "Ctrl+S")
        self._act(file_menu, "Save &as...", self.save_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        self.recent_menu = file_menu.addMenu("Recent projects")
        self._rebuild_recent()
        file_menu.addSeparator()
        self._act(file_menu, "&Quit", self.close, "Ctrl+Q")

        tools_menu = self.menuBar().addMenu("&Tools")
        self.tool_group = QtGui.QActionGroup(self)
        for key, label, shortcut in (
            ("select", "Select", "1"),
            ("trace", "Trace weld line", "2"),
            ("place", "Place part", "3"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _=False, k=key: self.set_tool(k))
            self.tool_group.addAction(action)
            tools_menu.addAction(action)
            if key == "select":
                action.setChecked(True)

        plan_menu = self.menuBar().addMenu("&Plan")
        self._act(plan_menu, "&Detect seams", self.detect_seams, "F5")
        self._act(plan_menu, "Generate &procedures", lambda: self._on_regenerate(""), "F6")
        self._act(
            plan_menu, "Place &automatically", lambda: self._on_placement_action("auto"), "F7"
        )
        self._act(plan_menu, "Plan the &job", self._on_plan, "F8")

        view_menu = self.menuBar().addMenu("&View")
        for name, label, _, default in LAYERS:
            action = QtGui.QAction(label, self, checkable=True)
            action.setChecked(default)
            action.toggled.connect(
                lambda state, n=name: (self.scene.set_layer_visible(n, state), self._render())
            )
            view_menu.addAction(action)
        view_menu.addSeparator()
        for key, label, shortcut in (
            ("iso", "Isometric", "Ctrl+0"),
            ("top", "Top", "Ctrl+1"),
            ("front", "Front", "Ctrl+2"),
            ("right", "Right", "Ctrl+3"),
        ):
            self._act(view_menu, label, lambda _=False, k=key: self._set_view(k), shortcut)

        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("toolbar_main")
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        for label, slot in (
            ("Import CAD", self.import_cad),
            ("Detect seams", self.detect_seams),
            ("Plan", self._on_plan),
            ("Save", self.save),
        ):
            action = QtGui.QAction(label, self)
            action.triggered.connect(slot)
            toolbar.addAction(action)

    def _act(self, menu, label: str, slot, shortcut: str = "") -> QtGui.QAction:
        action = QtGui.QAction(label, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_status(self) -> None:
        self.status_prompt = QtWidgets.QLabel("")
        self.status_detail = QtWidgets.QLabel("")
        self.status_detail.setStyleSheet("color: #9aa6b5;")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.setRange(0, 0)
        self.progress.hide()
        bar = self.statusBar()
        bar.addWidget(self.status_prompt, 1)
        bar.addWidget(self.status_detail, 2)
        bar.addPermanentWidget(self.progress)

    # -- helpers ------------------------------------------------------------

    def _render(self) -> None:
        self.vtk_widget.GetRenderWindow().Render()

    def _say(self, message: str, error: bool = False) -> None:
        self.status_detail.setText(message)
        self.status_detail.setStyleSheet("color: #bf616a;" if error else "color: #9aa6b5;")
        if message:
            logger.info("%s", message)

    def _busy_start(self, message: str) -> None:
        self._busy += 1
        self.progress.show()
        self._say(message)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)

    def _busy_end(self) -> None:
        self._busy = max(0, self._busy - 1)
        if not self._busy:
            self.progress.hide()
            QtWidgets.QApplication.restoreOverrideCursor()

    def _run(self, work, on_done, message: str) -> None:
        """Run *work* off the UI thread; call *on_done* with its result."""
        self._busy_start(message)
        worker = _Worker(work)

        def done(result):
            self._busy_end()
            on_done(result)

        def failed(text):
            self._busy_end()
            self._say(text, error=True)
            QtWidgets.QMessageBox.critical(self, "WeldForge", text)

        worker.signals.finished.connect(done)
        worker.signals.failed.connect(failed)
        self.pool.start(worker)

    def _update_title(self) -> None:
        mark = " *" if self.project.dirty else ""
        self.setWindowTitle(f"WeldForge Studio - {self.project.display_name}{mark}")

    def _refresh_all(self) -> None:
        self.setup_panel.load(self.project)
        self.setup_panel.describe(self.project)
        self.placement_panel.load(self.project)
        self.seam_panel.load(self.project)
        self.simulation_panel.show_report(self.project.last_report)
        self.output_panel.show_history(self.project)
        self._update_title()
        self.status_prompt.setText(self.tool.prompt())

    def _rebuild_scene(self) -> None:
        self.project._robot = self.robot  # used by the scene for animation
        self.scene.set_project(self.project, self.robot)
        self.scene.reset_camera("iso")
        self.simulation_panel.set_frames(self.scene.frame_count)
        self._render()

    # -- cell ---------------------------------------------------------------

    def _load_cell(self, name: str) -> None:
        from ..fixture import cell_robot, load_cell

        try:
            self.project.cell = load_cell(name)
            self.project.cell_name = name
            self.robot = cell_robot(self.project.cell)
        except FileNotFoundError as exc:
            self._say(str(exc), error=True)

    def _on_cell_changed(self, name: str) -> None:
        self._load_cell(name)
        self.project.touch()
        self.setup_panel.describe(self.project)
        self._rebuild_scene()
        self._refresh_all()
        self._say(f"cell: {name}")

    def _on_settings_changed(self) -> None:
        self.setup_panel.apply(self.project)
        self._update_title()

    # -- file ---------------------------------------------------------------

    def _confirm_discard(self) -> bool:
        if not self.project.dirty:
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "WeldForge",
            f"{self.project.display_name} has unsaved changes. Save it?",
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
        )
        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Save:
            return self.save()
        return True

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.project = Project()
        self.tool = SelectTool(self.project)
        self._load_cell(self.project.cell_name)
        self.scene.clear()
        self._rebuild_scene()
        self._refresh_all()
        self._say("new project")

    def import_cad(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import CAD", "", CAD_FILTER)
        if not path:
            return

        def work():
            attach_part(self.project, path)
            return self.project.part

        def done(part):
            self._load_cell(self.project.cell_name)
            self._rebuild_scene()
            self._refresh_all()
            lo, hi = part.bbox
            self._say(
                f"{part.name}: {len(part.solids)} solids, {len(part.faces)} faces, "
                f"{part.mass_kg:.1f} kg, "
                f"{(hi - lo)[0]:.0f} x {(hi - lo)[1]:.0f} x {(hi - lo)[2]:.0f} mm"
            )

        self._run(work, done, f"importing {Path(path).name}...")

    def open_project(self, path: str = "") -> None:
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open project", "", PROJECT_FILTER
            )
        if not path:
            return

        def work():
            return load_project(path)

        def done(project):
            self.project = project
            self.tool = SelectTool(self.project)
            self._load_cell(project.cell_name)
            self._rebuild_scene()
            self._refresh_all()
            self.recent.add(path)
            self._rebuild_recent()
            self._say(f"opened {Path(path).name}: {len(project.seams)} seams")

        self._run(work, done, f"opening {Path(path).name}...")

    def save(self) -> bool:
        if self.project.path is None:
            return self.save_as()
        return self._save_to(self.project.path)

    def save_as(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save project",
            f"{self.project.display_name}{PROJECT_SUFFIX}",
            PROJECT_FILTER,
        )
        return self._save_to(Path(path)) if path else False

    def _save_to(self, path: Path) -> bool:
        try:
            self.setup_panel.apply(self.project)
            saved = save_project(self.project, path)
        except (OSError, ProjectError) as exc:
            self._say(f"could not save: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "WeldForge", f"Could not save:\n{exc}")
            return False
        self.recent.add(saved)
        self._rebuild_recent()
        self._update_title()
        self._say(f"saved {saved.name}")
        return True

    def _rebuild_recent(self) -> None:
        self.recent_menu.clear()
        files = self.recent.load()
        if not files:
            action = self.recent_menu.addAction("(none)")
            action.setEnabled(False)
            return
        for path in files:
            self.recent_menu.addAction(path.name, lambda _=False, p=str(path): self.open_project(p))

    # -- tools --------------------------------------------------------------

    def set_tool(self, key: str) -> None:
        self.scene.clear_highlight()
        if key == "trace":
            self.tool = TraceSeamTool(self.project)
        elif key == "place":
            self.tool = PlacePartTool(self.project)
        else:
            self.tool = SelectTool(self.project)
        if self.tool.pick_layers:
            self.scene.pick_only(*self.tool.pick_layers)
            # Tracing needs the edges visible to click them.
            if key == "trace":
                self.scene.set_layer_visible("edges", True)
        else:
            self.scene.restore_pickable()
        self.status_prompt.setText(self.tool.prompt())
        self._render()

    def _on_left_click(self, interactor, event) -> None:
        x, y = interactor.GetEventPosition()
        pick = self.scene.pick(x, y)
        result = self.tool.on_pick(pick)
        self._handle_tool_result(result)

    def _handle_tool_result(self, result) -> None:
        if result.message:
            self._say(result.message, error=not result.ok)
        preview = self.tool.preview()
        if preview is not None:
            self.scene.highlight_points(preview)
        if result.seam_id:
            self.scene.highlight("seam", result.seam_id)
            self.seam_panel.select(result.seam_id)
            self.procedure_panel.show_seam(self.project, result.seam_id)
        if result.changed:
            self.scene.refresh_seams(self.project)
            self.seam_panel.load(self.project)
            self._update_title()
        if result.finished:
            self.set_tool("select")
        self.status_prompt.setText(self.tool.prompt())
        self._render()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        commands = {
            QtCore.Qt.Key_Return: "accept",
            QtCore.Qt.Key_Enter: "accept",
            QtCore.Qt.Key_Escape: "cancel",
            QtCore.Qt.Key_Backspace: "undo",
            QtCore.Qt.Key_Delete: "delete",
            QtCore.Qt.Key_Space: "toggle",
        }
        if key in commands:
            self._handle_tool_result(self.tool.on_command(commands[key]))
            return
        super().keyPressEvent(event)

    # -- placement ----------------------------------------------------------

    def _on_placement_action(self, action: str) -> None:
        if self.project.part is None:
            self._say("import a CAD model first", error=True)
            return
        tool = self.tool if isinstance(self.tool, PlacePartTool) else PlacePartTool(self.project)

        if action == "auto":

            def work():
                return tool.auto_place(self.robot)

            def done(result):
                self._after_placement(result)

            self._run(work, done, "searching for the best placement...")
            return

        self._after_placement(tool.on_command(action))

    def _after_placement(self, result) -> None:
        if result.changed:
            self.scene.update_placement(self.project.placement)
            self.scene.refresh_seams(self.project)
            self.placement_panel.load(self.project)
            self._update_title()
        self.placement_panel.set_status(result.message)
        self._say(result.message, error=not result.ok)
        self._render()

    def _on_pose_edited(self, pose) -> None:
        if self.project.placement_locked:
            return
        self.project.placement = np.asarray(pose, dtype=float)
        self.project.touch()
        self.scene.update_placement(self.project.placement)
        self.scene.refresh_seams(self.project)
        self._update_title()
        self._render()

    def _on_lock_toggled(self, locked: bool) -> None:
        self.project.placement_locked = locked
        self.project.touch()
        self._say("placement locked" if locked else "placement unlocked")

    # -- seams --------------------------------------------------------------

    def detect_seams(self) -> None:
        if self.project.part is None:
            self._say("import a CAD model first", error=True)
            return

        from ..seams import DetectOptions, detect_seams

        part = self.project.part

        def work():
            return detect_seams(part, DetectOptions())

        def done(result):
            seams, report = result
            kept = self.project.replace_detected_seams(seams)
            self.scene.refresh_seams(self.project)
            self.seam_panel.load(self.project)
            self._update_title()
            self._render()
            message = (
                f"{len(seams)} seams, {sum(s.length_mm for s in seams):.0f} mm "
                f"in {report.seconds:.1f}s"
            )
            if kept:
                message += f"; {kept} hand-traced seam(s) kept"
            self._say(message)

        self._run(work, done, "detecting weld seams...")

    def _on_seam_action(self, action: str) -> None:
        if action == "detect":
            self.detect_seams()
        elif action == "trace":
            self.set_tool("trace")
        elif action in ("all_on", "all_off"):
            for seam in self.project.seams:
                seam.selected = action == "all_on"
            self.project.touch()
            self.scene.refresh_seams(self.project)
            self.seam_panel.load(self.project)
            self._update_title()
            self._render()

    def _on_seam_selected(self, seam_id: str) -> None:
        self.scene.highlight("seam", seam_id)
        self.procedure_panel.show_seam(self.project, seam_id)
        seam = self.project.seam(seam_id)
        if seam is not None:
            notes = "; ".join(seam.notes[:2])
            self._say(
                f"{seam.label or seam.id}: {seam.joint_type.value}, "
                f"{seam.length_mm:.0f} mm, {seam.dominant_position.value}"
                + (f" - {notes}" if notes else "")
            )
        self._render()

    def _on_seam_toggled(self, seam_id: str, selected: bool) -> None:
        seam = self.project.seam(seam_id)
        if seam is None:
            return
        seam.selected = selected
        self.project.touch()
        self.scene.refresh_seams(self.project)
        self.seam_panel.load(self.project)
        self._update_title()
        self._render()

    def _on_seam_deleted(self, seam_id: str) -> None:
        if not self.project.is_manual(seam_id):
            QtWidgets.QMessageBox.information(
                self,
                "WeldForge",
                "That seam was detected from the geometry. Switch it off instead "
                "of deleting it, or it comes back the next time seams are "
                "detected.",
            )
            return
        self.project.remove_seam(seam_id)
        self.scene.refresh_seams(self.project)
        self.seam_panel.load(self.project)
        self.procedure_panel.show_seam(self.project, "")
        self._update_title()
        self._render()
        self._say("seam deleted")

    def _on_zoom_to_seam(self, seam_id: str) -> None:
        seam = self.project.seam(seam_id)
        if seam is None:
            return
        centre = tf.apply_point(self.project.placement, seam.centroid)
        self.scene.look_at(centre, max(seam.length_mm, 120.0))
        self._render()

    # -- procedures ---------------------------------------------------------

    def _on_regenerate(self, seam_id: str) -> None:
        from ..wps import ProcedureOptions, generate_procedure, generate_procedures

        self.setup_panel.apply(self.project)
        options = ProcedureOptions(
            base_material=self.project.base_material,
            wire_diameter_mm=self.project.wire_diameter_mm,
            leg_thickness_ratio=self.project.leg_thickness_ratio,
        )

        if seam_id:
            seam = self.project.seam(seam_id)
            if seam is None:
                return
            self.project.procedures[seam_id] = generate_procedure(seam, options)
            self.project.edited_procedure_ids.discard(seam_id)
            self.project.touch()
            self.procedure_panel.show_seam(self.project, seam_id)
            self._say(f"procedure regenerated for {seam.label or seam_id}")
            self._update_title()
            return

        seams = [
            s
            for s in self.project.selected_seams
            # A procedure the operator edited is theirs; a bulk regenerate
            # must not quietly undo their work.
            if s.id not in self.project.edited_procedure_ids
        ]
        if not seams:
            self._say("no seams to generate procedures for", error=True)
            return

        def work():
            return generate_procedures(seams, options)

        def done(procedures):
            self.project.procedures.update(procedures)
            self.project.touch()
            self.procedure_panel.show_seam(self.project, self.procedure_panel.current_seam_id)
            self._update_title()
            kept = len(self.project.edited_procedure_ids)
            message = f"{len(procedures)} procedures generated"
            if kept:
                message += f"; {kept} hand-edited one(s) left alone"
            self._say(message)

        self._run(work, done, "generating welding procedures...")

    def _on_pass_edited(self, seam_id: str, index: int, field: str, value: float) -> None:
        procedure = self.project.procedures.get(seam_id)
        if procedure is None or index >= len(procedure.passes):
            return
        setattr(procedure.passes[index], field, value)
        self.project.mark_procedure_edited(seam_id)
        self.procedure_panel.show_seam(self.project, seam_id)
        self._update_title()
        self._say(
            f"pass {index + 1} {field.replace('_', ' ')} set to {value:g}; "
            "this procedure is now hand-edited and will not be regenerated in bulk"
        )

    # -- planning -----------------------------------------------------------

    def _on_plan(self) -> None:
        if self.project.part is None or self.project.cell is None:
            self._say("import a CAD model first", error=True)
            return
        if not self.project.selected_seams:
            self._say("no seams selected to weld", error=True)
            return

        from ..plan import PlanOptions, plan_from_part

        self.setup_panel.apply(self.project)
        project = self.project
        robot = self.robot

        def work():
            options = PlanOptions(name=project.display_name)
            options.procedure.base_material = project.base_material
            options.procedure.wire_diameter_mm = project.wire_diameter_mm
            options.procedure.leg_thickness_ratio = project.leg_thickness_ratio
            # The operator placed the part, so the search must not undo that.
            options.do_placement = not project.placement_locked
            job = plan_from_part(project.part, project.cell, options, robot)
            if project.placement_locked and job.placement is not None:
                job.placement.pose = project.placement
            return job

        def done(job):
            self.project.job = job
            self.project.last_report = job.report.summary()
            if job.placement is not None and not self.project.placement_locked:
                self.project.placement = job.placement.pose
            if job.positioner is not None:
                self.project.positioner_values = {
                    seam_id: dict(entry.values)
                    for seam_id, entry in job.positioner.per_seam.items()
                }
            for seam_id, procedure in job.procedures.items():
                if seam_id not in self.project.edited_procedure_ids:
                    self.project.procedures[seam_id] = procedure
            self.project.touch()

            self._rebuild_scene()
            self._refresh_all()
            self.simulation_panel.show_report(self.project.last_report)
            report = job.report
            self._say(
                f"{'plan OK' if report.ok else 'plan reported problems'}: "
                f"{report.cycle_time_s:.0f} s cycle, "
                f"{report.wire_consumption_kg * 1000:.0f} g wire, "
                f"{len(report.errors)} error(s)",
                error=not report.ok,
            )

        self._run(work, done, "planning the job...")

    # -- playback -----------------------------------------------------------

    def _on_play_toggled(self, playing: bool) -> None:
        if playing and self.scene.frame_count > 1:
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _advance_frame(self) -> None:
        if self.scene.frame_count < 1:
            self._play_timer.stop()
            return
        self._frame = (self._frame + 1) % self.scene.frame_count
        self._show_frame(self._frame)
        self.simulation_panel.set_frame(self._frame)

    def _on_frame_changed(self, index: int) -> None:
        self._frame = index
        self._show_frame(index)

    def _show_frame(self, index: int) -> None:
        frame = self.scene.set_frame(index)
        if frame is not None:
            self.status_prompt.setText(
                f"seam {frame['seam']} pass {frame['pass'] + 1} - "
                + ("ARC ON" if frame["weld"] else "arc off")
            )
        self._render()

    # -- output -------------------------------------------------------------

    def _on_generate(self, target: str) -> None:
        if self.project.job is None:
            self._say("plan the job first", error=True)
            return
        from ..post import PostOptions, get_postprocessor

        try:
            post = get_postprocessor(
                target,
                PostOptions(program_name=(self.project.display_name[:8] or "WELDJOB").upper()),
            )
            result = post.generate(self.project.job)
        except Exception as exc:
            self._say(f"could not generate: {exc}", error=True)
            return

        self.project.programs[target] = dict(result.files)
        self.project.touch()
        self.output_panel.show_program(result.files)
        self._update_title()
        message = f"{target}: {result.n_moves} moves, {len(result.files)} file(s)"
        if result.warnings:
            message += f"; {result.warnings[0]}"
        self._say(message)

    def _on_browse_destination(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Where to write the program")
        if folder:
            self.output_panel.set_destination(folder)

    def _on_send(self, target: str, send_target: SendTarget, force: bool) -> None:
        project = self.project

        def work():
            return dispatch(project, target, send_target, force)

        def done(result):
            self.output_panel.show_history(self.project)
            self._update_title()
            QtWidgets.QMessageBox.information(
                self,
                "WeldForge",
                result.message + "\n\nDry-run the program at reduced speed with the arc "
                "disabled before welding a part.",
            )
            self._say(result.message)

        def failed(text):
            self._busy_end()
            self._say(text, error=True)
            QtWidgets.QMessageBox.warning(self, "WeldForge", text)

        self._busy_start(f"sending to {send_target.describe()}...")
        worker = _Worker(work)
        worker.signals.finished.connect(lambda r: (self._busy_end(), done(r)))
        worker.signals.failed.connect(failed)
        self.pool.start(worker)

    # -- view ---------------------------------------------------------------

    def _set_view(self, view: str) -> None:
        self.scene.reset_camera(view)
        self._render()

    # -- shutdown -----------------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if not self._confirm_discard():
            event.ignore()
            return
        self._play_timer.stop()
        # Releasing the render window is best effort: on some drivers it is
        # already gone by the time the window closes, and failing to shut
        # down is not a reason to refuse to shut down.
        with contextlib.suppress(Exception):
            self.vtk_widget.GetRenderWindow().Finalize()
        event.accept()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("WeldForge Studio")
    app.setOrganizationName("WeldForge")
    app.setStyle("Fusion")
    _apply_dark_palette(app)

    window = MainWindow()
    window.show()

    # A file named on the command line opens straight away, so a project can
    # be opened by double-clicking it.
    arguments = (argv if argv is not None else sys.argv)[1:]
    for argument in arguments:
        path = Path(argument)
        if path.suffix == PROJECT_SUFFIX and path.exists():
            window.open_project(str(path))
            break
        if path.exists():
            attach_part(window.project, path)
            window._rebuild_scene()
            window._refresh_all()
            break

    return app.exec()


def _apply_dark_palette(app: QtWidgets.QApplication) -> None:
    """A dark palette, because a 3-D view sits badly in a light window."""
    palette = QtGui.QPalette()
    colours = {
        QtGui.QPalette.Window: "#232a33",
        QtGui.QPalette.WindowText: "#e6eaf0",
        QtGui.QPalette.Base: "#1b2027",
        QtGui.QPalette.AlternateBase: "#232a33",
        QtGui.QPalette.Text: "#e6eaf0",
        QtGui.QPalette.Button: "#2c3540",
        QtGui.QPalette.ButtonText: "#e6eaf0",
        QtGui.QPalette.Highlight: "#38bdf8",
        QtGui.QPalette.HighlightedText: "#10141a",
        QtGui.QPalette.ToolTipBase: "#2c3540",
        QtGui.QPalette.ToolTipText: "#e6eaf0",
    }
    for role, colour in colours.items():
        palette.setColor(role, QtGui.QColor(colour))
    app.setPalette(palette)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
