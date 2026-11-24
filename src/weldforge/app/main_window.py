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
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..diagnostics import format_report, log_dir, log_file, report, setup_logging
from ..geom import transform as tf
from ..i18n import (
    LANGUAGE_NAMES,
    available_languages,
    language,
    load_preferred_language,
    save_preferred_language,
    set_language,
    t,
)
from .panels import (
    CellPanel,
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
    attach_bench,
    attach_obstacle,
    attach_part,
    cell_for,
    clear_bench,
    load_project,
    materialise_bench,
    remove_obstacle,
    save_project,
)
from .scene3d import LAYERS, Scene3D
from .send import SendTarget, dispatch
from .tools import MoveCellTool, PlacePartTool, SelectTool, Tool, TraceSeamTool, TwoFaceTool
from .translate import apply_language

logger = logging.getLogger(__name__)

__all__ = ["MainWindow", "main"]


def _cad_filter() -> str:
    """Build the import filter from what the loader actually supports.

    Derived rather than written out, because a format added to the loader and
    forgotten here is a format the operator cannot select -- which looks
    exactly like a broken import.
    """
    from ..cad.loader import BREP_SUFFIXES, MESH_SUFFIXES

    brep = " ".join(f"*{s}" for s in sorted(BREP_SUFFIXES))
    mesh = " ".join(f"*{s}" for s in sorted(MESH_SUFFIXES))
    return (
        f"CAD files ({brep} {mesh})"
        f";;Exact geometry - STEP, IGES, BREP ({brep})"
        f";;Meshes - STL, OBJ, PLY ({mesh})"
        ";;All files (*)"
    )


def _default_language() -> str:
    """The language a machine with no saved choice starts in.

    Italian, because this is built for an Italian shop floor and an operator
    should not have to find a menu in a language they struggle with in order
    to change the language. English remains one menu entry away, and is what
    the command line and the API keep.
    """
    from ..i18n import DEFAULT_LANGUAGE, available_languages

    return "it" if "it" in available_languages() else DEFAULT_LANGUAGE


CAD_FILTER = _cad_filter()
PROJECT_FILTER = f"WeldForge project (*{PROJECT_SUFFIX});;All files (*)"

#: Keys the active tool gets first refusal on, wherever the focus is. Defined
#: once because they are dispatched from two places: the window's own key
#: handler, and the filter on the 3-D viewport, which would otherwise swallow
#: them (see ``_build_viewport``).
TOOL_KEYS: dict[int, str] = {
    QtCore.Qt.Key_Return: "accept",
    QtCore.Qt.Key_Enter: "accept",
    QtCore.Qt.Key_Escape: "cancel",
    QtCore.Qt.Key_Backspace: "undo",
    QtCore.Qt.Key_Delete: "delete",
    QtCore.Qt.Key_Space: "toggle",
}


# ---------------------------------------------------------------------------
# background work
# ---------------------------------------------------------------------------


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    # Summary plus the full traceback. A separate signal rather than a wider
    # ``failed``, so the handlers that only want the one-liner stay as they are.
    failed_detail = QtCore.Signal(str, str)
    progress = QtCore.Signal(str)


class _Worker(QtCore.QRunnable):
    """Runs one callable off the UI thread and reports back.

    Exceptions are caught and reported rather than killing the thread pool: a
    planning run that hits bad geometry must leave the application usable.
    """

    def __init__(
        self,
        work: Callable[..., Any],
        *args: Any,
        parent: QtCore.QObject | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # The thread pool deletes a runnable the moment run() returns -- and
        # with it, this signals object, if nothing else owns it. The result is
        # emitted as a *queued* call to the UI thread, so if the deletion wins
        # the race the connection is gone and the slot never runs: an import
        # that finished in the worker and never reached the window. Operators
        # saw exactly that, sometimes, and clicked Import again. The signals
        # object is parented to the window and the runnable is kept alive by
        # the caller until the result has been delivered.
        self.signals = _WorkerSignals(parent)
        self.setAutoDelete(False)
        self._work = work
        self._args = args
        self._kwargs = kwargs

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._work(*self._args, **self._kwargs)
        except Exception as exc:
            detail = traceback.format_exc()
            logger.error("background task failed\n%s", detail)
            # A message an operator can read, and the traceback behind it. The
            # message alone was not enough to diagnose a failed CAD import on a
            # machine we could not reach.
            summary = _summarise(exc)
            self.signals.failed_detail.emit(summary, detail)
            self.signals.failed.emit(summary)
        else:
            self.signals.finished.emit(result)


def _summarise(exc: BaseException) -> str:
    """One readable line (or paragraph) for an exception.

    Our own errors already carry text written for an operator, so they are
    shown as-is; anything else gets its class name, which is the only clue a
    bare ``KeyError: 3`` offers.
    """
    from ..cad.loader import UnsupportedCadFormat

    text = str(exc).strip()
    if isinstance(exc, UnsupportedCadFormat | FileNotFoundError) and text:
        return text
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _widget_needs_keys() -> bool:
    """True when the focused widget uses Enter / Esc / Backspace itself.

    Those keys mean "commit this field", "undo this edit" or "delete a
    character" inside a text or number editor, and a tool taking them from
    there would make the panels unusable while a line is being traced.
    """
    focus = QtWidgets.QApplication.focusWidget()
    if focus is None:
        return False
    editors = (
        QtWidgets.QLineEdit,
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QTextEdit,
        QtWidgets.QPlainTextEdit,
        QtWidgets.QComboBox,
    )
    return isinstance(focus, editors) or isinstance(focus.parent(), QtWidgets.QAbstractSpinBox)


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
        #: Workers in flight, kept alive until their result has been delivered.
        self._workers: list[_Worker] = []

        self._build_viewport()
        self._build_panels()
        self._build_actions()
        self._build_status()

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(45)
        self._play_timer.timeout.connect(self._advance_frame)
        self._frame = 0

        # The language before the first refresh, so nothing is ever shown in
        # a language the operator did not pick.
        set_language(load_preferred_language(_default_language()))
        apply_language(self)

        self._load_cell(self.project.cell_name)
        # The cell is drawn before the first paint. It used to appear only
        # once something forced a rebuild -- an import, a cell change -- so the
        # window opened on an empty viewport and the bench looked as if it
        # depended on there being a part. The cell is the room; the part is
        # what comes into it.
        self._rebuild_scene()
        self._refresh_all()

    # -- construction -------------------------------------------------------

    def _build_viewport(self) -> None:
        import vtk
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # One line above the viewport saying what to do next. The window has
        # four function keys and a dozen panels; an operator should not have to
        # hold the order of operations in their head to use it.
        self.guide = QtWidgets.QLabel("")
        self.guide.setWordWrap(True)
        self.guide.setStyleSheet(
            "background: #2b333f; color: #e8ecf2; padding: 10px 14px; font-size: 15px;"
        )
        layout.addWidget(self.guide)

        self.vtk_widget = QVTKRenderWindowInteractor(frame)
        layout.addWidget(self.vtk_widget, 1)
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
        # Dragging needs the mouse between press and release. The camera keeps
        # middle and right, so a drag never fights an orbit.
        interactor.AddObserver("MouseMoveEvent", self._on_mouse_move, 1.0)
        interactor.AddObserver("LeftButtonReleaseEvent", self._on_left_release, 1.0)

        # QVTKRenderWindowInteractor.keyPressEvent forwards the key to VTK and
        # returns without calling super() or ignoring the event, so Qt treats
        # it as handled and it never reaches this window. Since an operator
        # must click in the viewport to pick an edge -- which gives the widget
        # focus -- Enter, Esc, Backspace, Space and Delete all stopped working
        # exactly when a tool needed them: a traced weld line could be drawn
        # and never accepted. The filter runs before the widget sees the key.
        self.vtk_widget.installEventFilter(self)
        # And on the application, for the keys a tool is waiting for: Enter
        # pressed with the focus in a panel spin box was being eaten by the
        # spin box, so a traced line could never be accepted from there.
        QtWidgets.QApplication.instance().installEventFilter(self)

        self.vtk_widget.Initialize()
        interactor.Start()

    def _build_panels(self) -> None:
        self.setDockOptions(
            QtWidgets.QMainWindow.AnimatedDocks | QtWidgets.QMainWindow.AllowTabbedDocks
        )

        self.setup_panel = SetupPanel()
        self.setup_panel.cellChanged.connect(self._on_cell_changed)
        self.setup_panel.settingsChanged.connect(self._on_settings_changed)

        self.cell_panel = CellPanel()
        self.cell_panel.cellEdited.connect(self._on_cell_edited)
        self.cell_panel.benchImportRequested.connect(self.import_bench)
        self.cell_panel.benchClearRequested.connect(self.clear_bench_model)
        self.cell_panel.saveRequested.connect(self.save_cell)
        self.cell_panel.obstacleImportRequested.connect(self.import_obstacle)
        self.cell_panel.obstacleRemoveRequested.connect(self.remove_obstacle_model)

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
        box.addWidget(self._titled("The cell", self.cell_panel))
        box.addWidget(self._titled("Placement", self.placement_panel), 1)
        container_scroll = QtWidgets.QScrollArea()
        container_scroll.setWidgetResizable(True)
        container_scroll.setWidget(container)
        container_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left.setWidget(container_scroll)
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
        self._act(file_menu, "Import &bench model...", self.import_bench, "Ctrl+B")
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
            # Two surfaces -> seam, groove and torch direction derived. The
            # front door for tracing; edge-by-edge stays for the odd case.
            ("twofaces", "Trace weld from two faces", "2"),
            ("trace", "Trace along edges (advanced)", "Shift+2"),
            ("place", "Place part", "3"),
            ("move", "Move cell elements", "4"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _=False, k=key: self.set_tool(k))
            self.tool_group.addAction(action)
            tools_menu.addAction(action)
            if key == "select":
                action.setChecked(True)

        plan_menu = self.menuBar().addMenu("&Plan")
        self._act(plan_menu, "&Prepare the job", self.prepare_job, "F9")
        plan_menu.addSeparator()
        self._act(plan_menu, "&Detect seams", self.detect_seams, "F5")
        self._act(plan_menu, "Generate &procedures", lambda: self._on_regenerate(""), "F6")
        self._act(
            plan_menu, "Place &automatically", lambda: self._on_placement_action("auto"), "F7"
        )
        self._act(plan_menu, "Plan the &job", self._on_plan, "F8")

        view_menu = self.menuBar().addMenu("&View")
        self.layer_actions: dict[str, QtGui.QAction] = {}
        for name, label, _, default in LAYERS:
            action = QtGui.QAction(label, self, checkable=True)
            action.setChecked(default)
            action.toggled.connect(
                lambda state, n=name: (self.scene.set_layer_visible(n, state), self._render())
            )
            view_menu.addAction(action)
            self.layer_actions[name] = action
        view_menu.addSeparator()
        for key, label, shortcut in (
            ("iso", "Isometric", "Ctrl+0"),
            ("top", "Top", "Ctrl+1"),
            ("front", "Front", "Ctrl+2"),
            ("right", "Right", "Ctrl+3"),
        ):
            self._act(view_menu, label, lambda _=False, k=key: self._set_view(k), shortcut)

        # A failed import on a machine we cannot reach is diagnosed from these
        # two entries, so they are in the menu bar rather than buried.
        help_menu = self.menuBar().addMenu("&Help")
        self._act(help_menu, "&Diagnostics...", self.show_diagnostics, "F1")
        self._act(help_menu, "Open &log folder", self.open_log_folder)
        help_menu.addSeparator()
        language_menu = help_menu.addMenu("&Language")
        self.language_group = QtGui.QActionGroup(self)
        for code in available_languages():
            # Each language in its own name: somebody looking for theirs should
            # not have to read another one to find it.
            action = QtGui.QAction(LANGUAGE_NAMES.get(code, code), self, checkable=True)
            action.setChecked(code == language())
            action.triggered.connect(lambda _=False, c=code: self.set_language(c))
            self.language_group.addAction(action)
            language_menu.addAction(action)

        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("toolbar_main")
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        # Three buttons, in the order a job happens. Detect, procedures,
        # placement and plan are all inside "Prepare the job" now; they stay
        # in the Plan menu for when one of them has to be run on its own.
        for label, slot in (
            ("Import CAD", self.import_cad),
            ("Prepare the job", self.prepare_job),
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
        # Accept / cancel as buttons, shown only while a tool is mid-action.
        # The keys still work, but the log showed an operator reaching "Enter
        # to accept" twice and never getting a seam: Enter went to the spin
        # box that still had the focus. A button has no focus to lose.
        self.accept_button = QtWidgets.QPushButton("Accept (Enter)")
        self.accept_button.setStyleSheet("font-weight: bold;")
        self.accept_button.clicked.connect(lambda: self._tool_command(QtCore.Qt.Key_Return))
        self.cancel_button = QtWidgets.QPushButton("Cancel (Esc)")
        self.cancel_button.clicked.connect(lambda: self._tool_command(QtCore.Qt.Key_Escape))
        self.accept_button.hide()
        self.cancel_button.hide()

        bar = self.statusBar()
        bar.addWidget(self.status_prompt, 1)
        bar.addWidget(self.status_detail, 2)
        bar.addWidget(self.accept_button)
        bar.addWidget(self.cancel_button)
        bar.addPermanentWidget(self.progress)

    def _update_tool_buttons(self) -> None:
        busy = bool(self.tool.busy())
        self.accept_button.setVisible(busy)
        self.cancel_button.setVisible(busy)

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
        worker = _Worker(work, parent=self)
        # Held until the result lands; see _Worker.__init__ for why.
        self._workers.append(worker)

        def release():
            self._busy_end()
            with contextlib.suppress(ValueError):
                self._workers.remove(worker)

        def done(result):
            release()
            # ``on_done`` runs on the UI thread, where an exception would be
            # swallowed by Qt and leave the window looking as if nothing had
            # happened. Report it the same way as a failure in the worker.
            try:
                on_done(result)
            except Exception as exc:
                detail = traceback.format_exc()
                logger.error("completion handler failed\n%s", detail)
                self._report_failure(_summarise(exc), detail)

        def failed(summary, detail):
            release()
            self._report_failure(summary, detail)

        worker.signals.finished.connect(done)
        worker.signals.failed_detail.connect(failed)
        self.pool.start(worker)

    def _report_failure(self, summary: str, detail: str = "") -> None:
        """Show a failure, with the traceback and the log file behind it."""
        self._say(summary.splitlines()[0] if summary else t("failed"), error=True)
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle("WeldForge")
        box.setText(summary or t("The operation failed."))
        box.setInformativeText(f"Full details are in the log file:\n{log_file()}")
        if detail:
            box.setDetailedText(detail)
        box.exec()

    def _update_title(self) -> None:
        mark = " *" if self.project.dirty else ""
        self.setWindowTitle(f"WeldForge Studio - {self.project.display_name}{mark}")

    def _refresh_all(self) -> None:
        self.setup_panel.load(self.project)
        self.setup_panel.describe(self.project)
        self.cell_panel.load(self.project.cell_config)
        self.cell_panel.describe(self.project)
        self.placement_panel.load(self.project)
        self.seam_panel.load(self.project)
        self.simulation_panel.show_report(self.project.last_report)
        self.output_panel.show_history(self.project)
        self._update_title()
        self.status_prompt.setText(self.tool.prompt())
        self._update_guide()

    def _rebuild_scene(self) -> None:
        self.scene.set_project(self.project, self.robot)
        self.scene.reset_camera("iso")
        self.simulation_panel.set_frames(self.scene.frame_count)
        self._render()

    # -- cell ---------------------------------------------------------------

    def _load_cell(self, name: str) -> None:
        """Build the project's cell: its own description, or the named one."""
        from ..fixture import cell_robot, read_cell_config

        try:
            self.project.cell_name = name
            if not self.project.cell_config:
                # Every later edit -- a typed number, a dragged clamp -- works
                # on this description, so it has to exist from the first load
                # rather than appearing when something is changed.
                self.project.cell_config = read_cell_config(name)
            self.project.cell = cell_for(self.project)
            self.robot = cell_robot(self.project.cell)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            self._say(f"cell {name!r}: {exc}", error=True)

    def _on_cell_changed(self, name: str) -> None:
        """A different cell was picked from the library.

        Its description replaces the project's, because the operator asked for
        that cell -- keeping the old bench numbers under a new cell's name
        would be a cell that exists nowhere.
        """
        from ..fixture import read_cell_config

        try:
            self.project.cell_config = read_cell_config(name)
        except FileNotFoundError as exc:
            self._say(str(exc), error=True)
            return
        self.project.bench_bytes = None
        self.project.bench_filename = ""
        self._reload_cell_from_config(t("cell: {name}", name=name))

    def _on_cell_edited(self, raw: object) -> None:
        """The bench numbers changed in the cell panel."""
        self.project.cell_config = dict(raw)  # type: ignore[arg-type]
        # The panel is the source of these numbers, so it is not reloaded:
        # rebuilding the clamp table while somebody is typing in it takes the
        # cursor out of the cell they are editing.
        self._reload_cell_from_config("", reload_panel=False)

    def _reload_cell_from_config(self, message: str, reload_panel: bool = True) -> None:
        """Rebuild the cell from its description and redraw what stands on it.

        Deliberately not ``_rebuild_scene`` plus ``_refresh_all``. Those tear
        down every actor and reset the camera to isometric, and reload every
        panel -- so adjusting a bench dimension threw away the operator's view
        and rebuilt the clamp table under their hands, taking the focus out of
        the field they were typing in. A cell edit redraws the cell.

        *reload_panel* is false when the cell panel is the thing that asked for
        the change: it already holds those numbers, and reloading it mid-edit
        is what loses the row being edited.
        """
        self._load_cell(self.project.cell_name)
        self.project.touch()
        self.scene.refresh_cell(self.project, self.robot)
        if reload_panel:
            self.cell_panel.load(self.project.cell_config)
        self.cell_panel.describe(self.project)
        self.setup_panel.describe(self.project)
        self._update_title()
        self._render()
        if message:
            self._say(message)

    def import_bench(self) -> None:
        """Import a 3-D model of the real bench."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, t("Import bench model"), "", CAD_FILTER
        )
        if not path:
            return

        def work():
            return attach_bench(self.project, path)

        def done(_raw):
            self._reload_cell_from_config("")
            fixture = self.project.cell.fixture if self.project.cell else None
            if fixture is not None:
                self._say(
                    f"bench {Path(path).name}: "
                    f"{fixture.table_size_mm[0]:.0f} x {fixture.table_size_mm[1]:.0f} mm, "
                    f"surface at {fixture.table_z_mm:.0f} mm"
                )

        self._run(work, done, f"importing {Path(path).name} as the bench...")

    def clear_bench_model(self) -> None:
        if not self.project.bench_filename and not (
            self.project.cell_config or {}
        ).get("fixture", {}).get("model"):
            self._say(t("there is no bench model to remove"))
            return
        clear_bench(self.project)
        self._reload_cell_from_config(t("bench model removed; the bench is a plain slab again"))

    def import_obstacle(self) -> None:
        """Import a 3-D model -- a jig, a housing -- as an obstacle in the cell."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, t("Import obstacle model"), "", CAD_FILTER
        )
        if not path:
            return

        def work():
            return attach_obstacle(self.project, path)

        def done(entry):
            self._reload_cell_from_config(
                t("{name} imported: drag it (tool 4) to where it stands", name=entry["name"])
            )

        self._run(work, done, f"importing {Path(path).name} as an obstacle...")

    def remove_obstacle_model(self, obstacle_id: str) -> None:
        if not remove_obstacle(self.project, obstacle_id):
            self._say(t("there is no obstacle {id} to remove", id=obstacle_id), error=True)
            return
        self._reload_cell_from_config(t("obstacle {id} removed", id=obstacle_id))

    def save_cell(self, name: str) -> None:
        """Keep this cell in the library, so the next job starts from it."""
        from ..fixture import default_cell_config, save_cell_config

        raw = dict(self.project.cell_config or default_cell_config())
        # The bench model lives inside the project, so hand ``save_cell_config``
        # a real file to copy beside the YAML: a saved cell has to stand on its
        # own, the way a saved project does.
        bench = materialise_bench(self.project)
        if bench is not None:
            raw = {**raw, "fixture": {**raw.get("fixture", {}), "model": str(bench)}}
        if self.project.obstacle_models:
            # Imported obstacles likewise: give the saver a real file for each
            # one that lives only inside the project. Box obstacles have no
            # model and are left exactly as they are.
            from .project import _materialise_obstacle

            obstacles = []
            for entry in raw.get("obstacles", []) or []:
                entry = dict(entry)
                path = _materialise_obstacle(self.project, str(entry.get("id", "")))
                if path is not None:
                    entry["model"] = str(path)
                obstacles.append(entry)
            raw = {**raw, "obstacles": obstacles}
        try:
            path = save_cell_config(raw, name)
        except OSError as exc:
            self._report_failure(t("could not save the cell: {error}", error=exc))
            return
        self.project.cell_name = name
        self.project.cell_config = raw
        self.setup_panel.reload_cells(name)
        self._refresh_all()
        self._say(t("cell saved as {name} ({path})", name=name, path=path))

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
        self._say(t("new project"))

    def import_cad(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("Import CAD"), "", CAD_FILTER)
        if not path:
            return

        def work():
            attach_part(self.project, path)
            return self.project.part

        def done(part):
            self._load_cell(self.project.cell_name)
            # A part arrives in CAD coordinates, which is the cell origin --
            # inside the robot column. Seat it on the bench unless the
            # operator has locked a measured placement, in which case the
            # new geometry goes exactly where the old one was.
            if not self.project.placement_locked:
                PlacePartTool(self.project).seat_on_bench()
            self._rebuild_scene()
            self._refresh_all()
            lo, hi = part.bbox
            self._say(
                f"{part.name}: {len(part.solids)} solids, {len(part.faces)} faces, "
                f"{part.mass_kg:.1f} kg, "
                f"{(hi - lo)[0]:.0f} x {(hi - lo)[1]:.0f} x {(hi - lo)[2]:.0f} mm"
            )

        self._run(work, done, f"importing {Path(path).name}...")

    # -- diagnostics --------------------------------------------------------

    def show_diagnostics(self) -> None:
        """What this installation can read, in a dialog that copies out."""
        text = format_report(report())
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(t("WeldForge diagnostics"))
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setText(t("Send this to support if an import fails."))
        box.setDetailedText(text)
        copy = box.addButton(t("Copy"), QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Close)
        box.exec()
        if box.clickedButton() is copy:
            QtWidgets.QApplication.clipboard().setText(text)
            self._say(t("diagnostics copied to the clipboard"))

    def open_log_folder(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(log_dir())))

    def open_project(self, path: str = "") -> None:
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, t("Open project"), "", PROJECT_FILTER
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
            t("Save project"),
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
        elif key == "twofaces":
            self.tool = TwoFaceTool(self.project)
        elif key == "move":
            self.tool = MoveCellTool(self.project)
        else:
            self.tool = SelectTool(self.project)
        if self.tool.pick_layers:
            self.scene.pick_only(*self.tool.pick_layers)
            # Tracing needs the edges visible to click them: VTK will not pick
            # an actor it is not drawing, and the edges layer starts hidden.
            if key == "trace":
                self._show_layer("edges")
        else:
            self.scene.restore_pickable()
        self.status_prompt.setText(self.tool.prompt())
        self._update_tool_buttons()
        self._render()

    def _on_left_click(self, interactor, event) -> None:
        x, y = interactor.GetEventPosition()
        pick = self.scene.pick(x, y)
        self._handle_tool_result(self.tool.on_pick(pick))

    def _on_mouse_move(self, interactor, event) -> None:
        """Feed a dragging tool the cursor, projected onto its drag plane."""
        if not self.tool.dragging():
            return
        plane_z = self.tool.drag_plane_z()
        if plane_z is None:
            return
        x, y = interactor.GetEventPosition()
        through = self.tool.drag_through()
        if through is not None:
            # A plane facing the camera through the grab point: the object
            # stays under the cursor whatever the view angle. The horizontal
            # plane, seen nearly edge-on in an isometric view, turned a few
            # pixels into metres of bench.
            point = self.scene.point_on_view_plane(x, y, through)
        else:
            point = self.scene.point_on_plane(x, y, plane_z)
        if point is None:
            return
        result = self.tool.on_drag(point)
        nudge = self.tool.drag_preview()
        if nudge is not None:
            self.scene.nudge(nudge[0], nudge[1])
        if result.message:
            self._say(result.message, error=not result.ok)
        self._render()

    def _on_left_release(self, interactor, event) -> None:
        if not self.tool.dragging():
            return
        self._handle_tool_result(self.tool.on_drop())

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
        if result.changed and isinstance(self.tool, MoveCellTool):
            # Moving a cell element changes the cell, so the cell is what has
            # to be rebuilt -- and the panel's numbers follow the drag.
            self._reload_cell_from_config("")
        elif result.changed:
            self.scene.refresh_seams(self.project)
            self.seam_panel.load(self.project)
            self._update_title()
        if result.finished:
            self.set_tool("select")
        self.status_prompt.setText(self.tool.prompt())
        self._update_tool_buttons()
        self._render()

    def set_language(self, code: str) -> None:
        """Switch the window's language and remember the choice."""
        chosen = set_language(code)
        save_preferred_language(chosen)
        self._retranslate()
        self._say(f"{LANGUAGE_NAMES.get(chosen, chosen)}")

    def _retranslate(self) -> None:
        """Put every piece of furniture into the current language.

        The panels and the menus are walked rather than rebuilt: rebuilding
        them would throw away what the operator has selected, and a language
        change is not a reason to lose your place.
        """
        apply_language(self)
        self.status_prompt.setText(self.tool.prompt())
        self._refresh_all()

    def _show_layer(self, name: str) -> None:
        """Turn a layer on, keeping its View-menu tick honest."""
        self.scene.set_layer_visible(name, True)
        action = self.layer_actions.get(name)
        if action is not None and not action.isChecked():
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)

    def _tool_command(self, key: int) -> bool:
        """Route a key to the active tool. True when the tool took it."""
        command = TOOL_KEYS.get(key)
        if command is None:
            return False
        self._handle_tool_result(self.tool.on_command(command))
        return True

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if self._tool_command(event.key()):
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Give the tool first refusal on keys pressed inside the 3-D view."""
        if event.type() == QtCore.QEvent.KeyPress and event.key() in TOOL_KEYS:
            # Inside the viewport the tool always has first refusal. Anywhere
            # else it gets the key only while it is mid-action, and never from
            # a widget that needs Enter or Esc itself: a line edit being typed
            # in, a spin box, a combo that is open.
            in_viewport = watched is getattr(self, "vtk_widget", None)
            tool = getattr(self, "tool", None)
            wanted = in_viewport or (tool is not None and tool.busy() and not _widget_needs_keys())
            if wanted and self._tool_command(event.key()):
                return True
        return super().eventFilter(watched, event)

    # -- placement ----------------------------------------------------------

    def _on_placement_action(self, action: str) -> None:
        if self.project.part is None:
            self._say(t("import a CAD model first"), error=True)
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

        from ..seams import DetectOptions, detect_seams, explain_no_seams

        part = self.project.part
        options = DetectOptions()

        def work():
            return detect_seams(part, options)

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
            if not seams:
                # "0 seams" on its own reads as a broken import, and usually
                # is not: say which it is.
                reason = explain_no_seams(part, report, options)
                self._say(f"{message} -- {reason}", error=True)
                QtWidgets.QMessageBox.information(
                    self, "WeldForge", f"No weld seams were detected.\n\n{reason}."
                )
            else:
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

    # -- the one button -----------------------------------------------------

    def prepare_job(self) -> None:
        """Detect, generate, place, plan and simulate -- then say one thing.

        The steps were four function keys in a fixed order, and an operator who
        pressed them out of order got a refusal rather than a result. They are
        one action now. What it cannot decide for you it hands back: a part
        with no joints to find puts you straight into tracing, with the reason
        on screen, instead of reporting "0 seams" and stopping.

        It ends at the simulation -- the verdict that needs no cell connected.
        Moving the real robot stays a separate, deliberate action.
        """
        if self.project.part is None:
            self._say(t("import a CAD model first"), error=True)
            self._update_guide()
            return

        self.setup_panel.apply(self.project)
        part, options = self.project.part, None

        def work():
            from ..seams import DetectOptions, detect_seams

            return detect_seams(part, DetectOptions())

        def done(result):
            seams, report = result
            kept = self.project.replace_detected_seams(seams)
            self.scene.refresh_seams(self.project)
            self.seam_panel.load(self.project)

            if not self.project.selected_seams:
                # Nothing to weld that the geometry could find. Say why, and
                # leave the operator in the tool that solves it rather than
                # making them find it.
                from ..seams import DetectOptions, explain_no_seams

                reason = explain_no_seams(part, report, DetectOptions())
                self.set_tool("twofaces")
                self._say(reason, error=True)
                self._update_guide(
                    t(
                        "No joint was found automatically ({reason}). Click the two "
                        "surfaces of a joint to weld it.",
                        reason=reason,
                    )
                )
                return

            found = t(
                "{count} seams, {mm:.0f} mm",
                count=len(self.project.selected_seams),
                mm=sum(s.length_mm for s in self.project.selected_seams),
            )
            if kept:
                found += t(" (+{kept} traced by hand)", kept=kept)
            self._say(t("{found}: planning...", found=found))
            self._plan_job(after_prepare=True)

        _ = options
        self._run(work, done, t("looking for the weld seams..."))

    def _verdict(self) -> str:
        """The plan's answer in one line, in the operator's words."""
        report = self.project.last_report or {}
        if not report:
            return ""
        errors = list(report.get("errors") or [])
        warnings = list(report.get("warnings") or [])
        if errors:
            more = t(" (+{count} more)", count=len(errors) - 1) if len(errors) > 1 else ""
            return t("Cannot be welded as it is: {reason}{more}", reason=errors[0], more=more)
        ready = t(
            "Ready: {count} seams, cycle {cycle:.0f} s, {wire:.0f} g of wire",
            count=len(self.project.selected_seams),
            cycle=float(report.get("cycle_time_s", 0.0)),
            wire=float(report.get("wire_consumption_kg", 0.0)) * 1000.0,
        )
        if warnings:
            ready += t(". {count} warning(s) -- see the report", count=len(warnings))
        return ready

    def _update_guide(self, message: str = "") -> None:
        """Say what to do next, or show *message* when something is in hand."""
        if message:
            self.guide.setText(message)
            return
        if self.project.part is None:
            text = t("Step 1 - import the CAD model: File > Import CAD (Ctrl+I).")
        elif not self.project.seams:
            text = t(
                "Step 2 - press Prepare the job (F9): it looks for the joints and "
                "plans everything. If none are found you will be put into tracing."
            )
        elif not self.project.last_report:
            text = t("Step 3 - press Prepare the job (F9) to plan and simulate.")
        else:
            verdict = self._verdict()
            text = (
                t("{verdict} - generate the program in the Program panel.", verdict=verdict)
                if (self.project.last_report or {}).get("ok")
                else verdict
            )
        self.guide.setText(text)

    def _on_plan(self) -> None:
        self._plan_job(after_prepare=False)

    def _plan_job(self, after_prepare: bool = False) -> None:
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
            # The seams the operator settled on: the ones they traced by
            # hand included, the ones they switched off left out.
            job = plan_from_part(
                project.part, project.cell, options, robot, seams=project.selected_seams
            )
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
            # One line, in the operator's words, wherever the plan came from.
            # The numbers behind it are in the simulation panel.
            self._say(self._verdict(), error=not report.ok)
            self._update_guide()
            _ = after_prepare

        self._run(work, done, t("planning the job..."))

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
        worker = _Worker(work, parent=self)
        self._workers.append(worker)

        def release():
            with contextlib.suppress(ValueError):
                self._workers.remove(worker)

        worker.signals.finished.connect(lambda r: (release(), self._busy_end(), done(r)))
        worker.signals.failed.connect(lambda text: (release(), failed(text)))
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
    # Before anything else: this is a GUI entry point, so without a log file
    # a crash during start-up leaves no trace at all.
    setup_logging("studio")
    _install_excepthook()

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
            # A file that will not load is a reason to show the error, not to
            # refuse to start: the operator can still import another one.
            try:
                attach_part(window.project, path)
                window._rebuild_scene()
                window._refresh_all()
            except Exception as exc:
                window._report_failure(_summarise(exc), traceback.format_exc())
            break

    return app.exec()


def _install_excepthook() -> None:
    """Route uncaught exceptions to the log instead of a dead window.

    Qt catches exceptions raised inside slots and keeps going, which on a GUI
    build means they are lost entirely. Logging them is what makes a bug
    report possible.
    """
    previous = sys.excepthook

    def hook(kind, value, tb):
        logging.getLogger("weldforge").error(
            "uncaught exception\n%s", "".join(traceback.format_exception(kind, value, tb))
        )
        previous(kind, value, tb)

    sys.excepthook = hook


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
