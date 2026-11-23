"""The Qt layer, driven in a child process.

Why a child process. The panels are plain Qt and run fine under the offscreen
platform plug-in; the 3-D scene is plain VTK and also runs fine offscreen. The
combination -- ``QVTKRenderWindowInteractor``, the widget that puts a VTK
render window inside a Qt one -- segfaults under the offscreen plug-in on
macOS. Since ``test_app.py`` has already built a VTK scene in the pytest
process, anything Qt has to happen somewhere else.

That is not a workaround so much as the reason the application is layered the
way it is: the panels hold no planning logic, so a child process that builds
them and pokes their signals is a real test of everything they do.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="the desktop application needs PySide6")

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(body: str, tmp_path: Path) -> str:
    """Run *body* in a child interpreter with an offscreen Qt platform."""
    script = tmp_path / "child.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            sys.path.insert(0, %r)
            from PySide6 import QtWidgets
            app = QtWidgets.QApplication([])
            """
        )
        % SRC
        + textwrap.dedent(body)
    )
    environment = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": SRC,
    }
    finished = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )
    if finished.returncode != 0:
        raise AssertionError(
            f"the child process failed ({finished.returncode}):\n"
            f"{finished.stdout}\n{finished.stderr}"
        )
    return finished.stdout


def test_every_panel_builds_and_reports_a_project(tmp_path: Path):
    """The panels must survive a project with nothing in it and a full one.

    An empty project is the state the application starts in, so a panel that
    needs a part or a plan to render is a panel that crashes on launch.
    """
    output = _run(
        """
        import numpy as np
        from weldforge.app.panels import (
            OutputPanel, PlacementPanel, ProcedurePanel,
            SeamPanel, SetupPanel, SimulationPanel,
        )
        from weldforge.app.project import Project
        from weldforge.geom import transform as tf
        from weldforge.model import JointType, Seam
        from weldforge.wps import ProcedureOptions, generate_procedure

        panels = {
            "setup": SetupPanel(),
            "placement": PlacementPanel(),
            "seams": SeamPanel(),
            "procedure": ProcedurePanel(),
            "simulation": SimulationPanel(),
            "output": OutputPanel(),
        }

        # 1. The empty project the application opens with.
        empty = Project()
        panels["setup"].load(empty); panels["setup"].describe(empty)
        panels["placement"].load(empty)
        panels["seams"].load(empty)
        panels["procedure"].show_seam(empty, "")
        panels["simulation"].show_report({})
        panels["output"].show_history(empty)
        print("empty OK")

        # 2. A project with a seam, a procedure, a report and a program.
        full = Project(name="demo", base_material="S355", wire_diameter_mm=1.2)
        points = np.array([[0.0, 0.0, 8.0], [200.0, 0.0, 8.0]])
        seam = Seam(
            joint_type=JointType.FILLET,
            points=points,
            tangents=np.tile([1.0, 0.0, 0.0], (2, 1)),
            normal_a=np.tile([0.0, 0.0, 1.0], (2, 1)),
            normal_b=np.tile([0.0, -1.0, 0.0], (2, 1)),
            bisector=np.tile([0.0, 0.70710678, 0.70710678], (2, 1)),
            dihedral=np.full(2, np.pi / 2),
            thickness_a_mm=8.0,
            thickness_b_mm=6.0,
            confidence=0.9,
            label="cordone 1",
        )
        full.seams = [seam]
        full.procedures[seam.id] = generate_procedure(
            seam, ProcedureOptions(base_material="S355")
        )
        full.placement = tf.pose(tf.rot_z(0.5), [400.0, 100.0, 900.0])
        full.last_report = {
            "ok": False,
            "cycle_time_s": 142.5,
            "arc_time_s": 61.0,
            "weld_length_mm": 200.0,
            "wire_consumption_kg": 0.08,
            "errors": ["seam S1: A5 -136.2 deg outside [-130, 130]"],
            "warnings": ["seam S1: 2 of 40 stations close to a clamp"],
        }
        full.programs["rapid"] = {"WELDJOB.mod": "MODULE WELDJOB\\nENDMODULE\\n"}

        panels["setup"].load(full); panels["setup"].describe(full)
        panels["placement"].load(full)
        panels["seams"].load(full)
        panels["seams"].select(seam.id)
        panels["procedure"].show_seam(full, seam.id)
        panels["simulation"].show_report(full.last_report)
        panels["simulation"].set_frames(120)
        panels["simulation"].set_frame(60)
        panels["output"].show_program(full.programs["rapid"])
        panels["output"].show_history(full)
        print("full OK")

        # The seam table shows the seam, with its label and its length.
        table = panels["seams"].table
        assert table.rowCount() == 1, table.rowCount()
        row = " ".join(
            table.item(0, c).text() for c in range(table.columnCount())
            if table.item(0, c) is not None
        )
        assert "cordone 1" in row, row
        assert "200" in row, row
        print("seam row:", row)

        # A failed report has to say so where the operator looks.
        verdict = panels["simulation"].verdict.text().lower()
        assert "problem" in verdict or "not ok" in verdict or "fail" in verdict, verdict
        assert "136.2" in panels["simulation"].messages.toPlainText()
        print("verdict:", verdict)

        # Settings typed into the panel reach the project.
        panels["setup"].apply(full)
        assert full.base_material == "S355"
        print("apply OK")
        """,
        tmp_path,
    )
    assert "empty OK" in output
    assert "full OK" in output
    assert "apply OK" in output


def test_the_panel_signals_the_window_connects_all_exist(tmp_path: Path):
    """A mistyped signal name is a runtime error months later. Catch it now.

    ``MainWindow`` cannot be built under the offscreen plug-in, so instead of
    constructing it this checks the contract it relies on: every signal it
    connects, and every method it calls on a panel, is really there.
    """
    output = _run(
        """
        import inspect
        import re

        from weldforge.app import main_window, panels

        source = inspect.getsource(main_window)
        classes = {
            "setup_panel": panels.SetupPanel,
            "cell_panel": panels.CellPanel,
            "placement_panel": panels.PlacementPanel,
            "seam_panel": panels.SeamPanel,
            "procedure_panel": panels.ProcedurePanel,
            "simulation_panel": panels.SimulationPanel,
            "output_panel": panels.OutputPanel,
        }

        missing = []
        for attribute, cls in classes.items():
            used = set(re.findall(rf"self\\.{attribute}\\.(\\w+)", source))
            for member in used:
                if not hasattr(cls, member):
                    missing.append(f"{cls.__name__}.{member}")
        assert not missing, f"MainWindow uses members that do not exist: {missing}"
        print("checked", sum(1 for _ in classes), "panels")

        # And the window really does connect every signal each panel defines:
        # a panel signal nothing listens to is a control that does nothing.
        from PySide6 import QtCore
        unconnected = []
        for attribute, cls in classes.items():
            for name, value in vars(cls).items():
                if isinstance(value, QtCore.Signal) and f"self.{attribute}.{name}" not in source:
                    unconnected.append(f"{cls.__name__}.{name}")
        assert not unconnected, f"panel signals nothing is listening to: {unconnected}"
        print("every panel signal is connected")
        """,
        tmp_path,
    )
    assert "every panel signal is connected" in output


def test_the_window_only_calls_things_that_exist(tmp_path: Path):
    """The window cannot be built offscreen, so check its contract instead.

    ``MainWindow`` is the one module in the application no test can construct,
    which makes a typo in it exactly the kind of mistake that surfaces in front
    of an operator. Reading the source and checking every attribute it reaches
    for against the real classes is weaker than running it, but it catches the
    renames -- and it caught the window stashing the robot model on a private
    attribute of the project that nothing was meant to read.
    """
    output = _run(
        r"""
        import dataclasses
        import inspect
        import re

        from weldforge.app import main_window
        from weldforge.app.project import Project
        from weldforge.app.scene3d import Scene3D
        from weldforge.app.tools import (
            PlacePartTool, SelectTool, Tool, ToolResult, TraceSeamTool,
        )

        source = inspect.getsource(main_window)

        def used(attribute):
            return set(re.findall(rf"self\.{attribute}\.(\w+)", source))

        def members(cls):
            # A dataclass field with a default_factory is not a class
            # attribute, so hasattr alone would call every one of them missing.
            names = set(dir(cls))
            if dataclasses.is_dataclass(cls):
                names |= {f.name for f in dataclasses.fields(cls)}
            return names

        missing = []
        for member in used("scene") - members(Scene3D):
            missing.append(f"Scene3D.{member}")
        for member in used("project") - members(Project):
            missing.append(f"Project.{member}")
        # The active tool is any of the three, so a member has to be on the
        # base class or on a concrete tool the window can install.
        tools = (Tool, SelectTool, TraceSeamTool, PlacePartTool)
        for member in used("tool"):
            if not any(member in members(cls) for cls in tools):
                missing.append(f"Tool.{member}")
        assert not missing, f"MainWindow reaches for things that do not exist: {missing}"
        print("the window's contract holds")

        # And where it handles a tool result, the fields it branches on are
        # the ones ToolResult really has. Scoped to that one method, because
        # "result" elsewhere in the window is a postprocessor's or a
        # dispatch's result, with fields of its own.
        handler = inspect.getsource(main_window.MainWindow._handle_tool_result)
        fields = {f.name for f in dataclasses.fields(ToolResult)} | set(dir(ToolResult))
        unknown = set(re.findall(r"\bresult\.(\w+)", handler)) - fields
        assert not unknown, f"ToolResult has no {unknown}"
        print("tool results checked")
        """,
        tmp_path,
    )
    assert "the window's contract holds" in output
    assert "tool results checked" in output


def test_keys_pressed_in_the_viewport_reach_the_active_tool(tmp_path: Path):
    """Enter must accept a traced weld line even with the 3-D view focused.

    ``QVTKRenderWindowInteractor.keyPressEvent`` forwards the key to VTK and
    returns without calling ``super()`` or ignoring the event, so Qt counts it
    handled and it never reaches the window. Since picking an edge gives that
    widget the focus, Enter, Esc and Backspace all stopped working exactly
    when tracing needed them: the line could be drawn and never accepted. The
    window filters the viewport's keys to get in front of that.

    The window is built without ``__init__`` because a real one needs an
    OpenGL context, and this is about the dispatch, not the scene.
    """
    output = _run(
        """
        from PySide6 import QtCore, QtGui, QtWidgets

        from weldforge.app.main_window import TOOL_KEYS, MainWindow
        from weldforge.app.project import Project
        from weldforge.app.tools import TraceSeamTool

        # Every key the window dispatches has to be a command the tools know.
        understood = {"accept", "cancel", "undo", "delete", "toggle"}
        assert set(TOOL_KEYS.values()) <= understood, TOOL_KEYS
        for key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Escape, QtCore.Qt.Key_Backspace):
            assert key in TOOL_KEYS, key
        print("the key map is sound")

        seen = []

        class Spy(TraceSeamTool):
            def on_command(self, command):
                seen.append(command)
                return super().on_command(command)

        window = MainWindow.__new__(MainWindow)
        # The base class still has to be constructed -- Qt's C++ side refuses
        # to serve a half-built object -- but MainWindow's own __init__ is
        # skipped, because it builds a VTK render window.
        QtWidgets.QMainWindow.__init__(window)
        window.project = Project()
        window.tool = Spy(window.project)
        window.vtk_widget = QtWidgets.QWidget()
        window._handle_tool_result = lambda result: None

        def press(key, widget=None):
            event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier)
            return window.eventFilter(widget or window.vtk_widget, event)

        assert press(QtCore.Qt.Key_Return) is True, "Enter was not taken"
        assert press(QtCore.Qt.Key_Escape) is True, "Esc was not taken"
        assert press(QtCore.Qt.Key_Backspace) is True, "Backspace was not taken"
        assert seen == ["accept", "cancel", "undo"], seen

        # A key no tool wants must pass through, or VTK's own keys and the
        # camera would stop working.
        assert press(QtCore.Qt.Key_A) is False
        print("viewport keys reach the tool")

        # Only the viewport is filtered; every other widget keeps its keys.
        other = QtWidgets.QWidget()
        assert press(QtCore.Qt.Key_Return, other) is False, "only the viewport is filtered"
        print("other widgets are left alone")
        """,
        tmp_path,
    )
    assert "the key map is sound" in output
    assert "viewport keys reach the tool" in output
    assert "other widgets are left alone" in output


def test_the_cell_panel_edits_the_bench(tmp_path: Path):
    """The bench numbers have to be editable and come back out as a cell.

    An operator who cannot change bench height and distance cannot describe
    their own cell, and those two numbers decide whether a seam can be welded
    downhand at all.
    """
    output = _run(
        """
        from weldforge.app.panels import CellPanel
        from weldforge.fixture import cell_from_config, read_cell_config

        panel = CellPanel()
        panel.load(read_cell_config("bench_cell"))
        assert panel.length.value() > 0.0 and panel.height.value() > 0.0

        emitted = []
        panel.cellEdited.connect(emitted.append)
        panel.length.setValue(1800.0)
        panel.width.setValue(1000.0)
        panel.height.setValue(905.0)
        panel.origin_x.setValue(1300.0)
        panel.robot_z.setValue(250.0)
        assert len(emitted) == 5, emitted

        cell = cell_from_config(panel.config(), "edited")
        assert cell.fixture.table_size_mm == (1800.0, 1000.0)
        assert cell.fixture.table_z_mm == 905.0
        assert cell.fixture.table_origin[0] == 1300.0
        assert cell.robot_base[2, 3] == 250.0
        print("the bench is editable")

        # And the arm itself can be swapped from the panel.
        from weldforge.kin import available_robots

        others = [r for r in available_robots() if r != panel.robot_box.currentText()]
        assert others, "the library has more than one robot"
        panel.robot_box.setCurrentText(others[0])
        assert cell_from_config(panel.config(), "edited").robot_name == others[0]
        print("the robot can be swapped")

        # The name becomes a filename, so it has to survive being one.
        names = []
        panel.saveRequested.connect(names.append)
        panel.save_name.setText("Banco Officina 2")
        panel._on_save()
        assert names == ["banco_officina_2"], names
        print("the cell name is filename-safe")
        """,
        tmp_path,
    )
    assert "the bench is editable" in output
    assert "the robot can be swapped" in output
    assert "the cell name is filename-safe" in output


def test_the_cell_panel_edits_the_clamps(tmp_path: Path):
    """Clamps have to be addable, movable and removable from the panel.

    They are not decoration: the torch is kept out of a cylinder around each
    one, placement scores a pose on how clear the seams are of them, and the
    URDF export writes them out. So an edit here has to reach the cell.
    """
    output = _run(
        """
        from weldforge.app.panels import CellPanel
        from weldforge.fixture import cell_from_config, read_cell_config

        panel = CellPanel()
        panel.load(read_cell_config("bench_cell"))
        table = panel.clamp_table
        started_with = table.table.rowCount()
        assert started_with > 0, "the shipped bench cell has clamps"

        edits = []
        panel.cellEdited.connect(edits.append)

        # move one, turn its arm, widen it, and make it retractable
        table._spin(0, 1).setValue(742.0)
        table._spin(0, 3).setValue(760.0)
        table._spin(0, 5).setValue(60.0)
        table._combo(0, 4).setCurrentIndex(2)          # from +X
        table._check(0, 7).setChecked(True)
        table.table.item(0, 0).setText("morsetto anteriore")
        assert len(edits) == 6, edits

        clamp = cell_from_config(panel.config(), "edited").fixture.clamps[0]
        assert clamp.name == "morsetto anteriore"
        assert list(clamp.position) == [742.0, -320.0, 760.0]
        assert list(clamp.approach) == [1.0, 0.0, 0.0]
        assert clamp.clearance_mm == 60.0
        assert clamp.retractable
        print("a clamp can be moved")

        # adding lands it on the bench, not at the robot's feet
        panel.origin_x.setValue(1000.0)
        panel.origin_y.setValue(0.0)
        panel.height.setValue(800.0)
        panel._on_add_clamp()
        assert table.table.rowCount() == started_with + 1
        added = cell_from_config(panel.config(), "edited").fixture.clamps[-1]
        assert added.position[2] == 800.0, "a new clamp sits on the bench surface"
        assert abs(added.position[0] - 1000.0) < 400.0, added.position
        assert len({c.id for c in cell_from_config(panel.config(), "x").fixture.clamps}) == (
            started_with + 1
        ), "clamp ids stay unique"
        print("a clamp can be added")

        # and removing takes the selected row away
        table.table.selectRow(0)
        table.remove_selected()
        assert table.table.rowCount() == started_with
        names = [c.name for c in cell_from_config(panel.config(), "edited").fixture.clamps]
        assert "morsetto anteriore" not in names, names
        print("a clamp can be removed")

        # a clamp arm that is not one of the six axes keeps its own direction
        raw = read_cell_config("bench_cell")
        raw["fixture"]["clamps"][0]["approach"] = [0.0, 0.7071, 0.7071]
        raw["fixture"]["clamps"][1]["torque_nm"] = 42.0
        other = CellPanel()
        other.load(raw)
        clamps = other.config()["fixture"]["clamps"]
        assert clamps[0]["approach"] == [0.0, 0.7071, 0.7071], clamps[0]
        assert clamps[1]["torque_nm"] == 42.0, "a field the panel does not show is not dropped"
        print("odd clamps are not rounded off")
        """,
        tmp_path,
    )
    assert "a clamp can be moved" in output
    assert "a clamp can be added" in output
    assert "a clamp can be removed" in output
    assert "odd clamps are not rounded off" in output


def test_the_panels_translate_and_translate_back(tmp_path: Path):
    """The window's furniture goes into Italian and comes back to English.

    The walk reads chrome off the built widgets rather than wrapping every
    label at construction. Two things have to hold for that to be safe: data
    is never touched -- combo entries are cell and material names, spin boxes
    hold measurements -- and the English source survives being replaced, or
    English would be a language the operator could not get back to.
    """
    output = _run(
        """
        from PySide6 import QtWidgets

        from weldforge.app import panels
        from weldforge.app.translate import apply_language, source_strings
        from weldforge.fixture import read_cell_config
        from weldforge.i18n import set_language

        host = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(host)
        setup, cell = panels.SetupPanel(), panels.CellPanel()
        for widget in (setup, cell):
            layout.addWidget(widget)
        cell.load(read_cell_config("bench_cell"))

        english_button = cell.import_button.text()
        english_headers = [
            cell.clamp_table.table.horizontalHeaderItem(i).text()
            for i in range(cell.clamp_table.table.columnCount())
        ]
        cells_before = [setup.cell_box.itemText(i) for i in range(setup.cell_box.count())]
        height_before = cell.height.value()

        set_language("it")
        changed = apply_language(host)
        assert changed > 20, changed
        assert cell.import_button.text() != english_button
        assert "banco" in cell.import_button.text().lower(), cell.import_button.text()
        assert cell.clamp_table.table.horizontalHeaderItem(0).text() == "Nome"
        assert "Altezza" in cell.height.toolTip()
        print("the panels speak Italian")

        # Data is left alone.
        assert [setup.cell_box.itemText(i) for i in range(setup.cell_box.count())] == cells_before
        assert cell.height.value() == height_before
        print("data is untouched")

        # And back.
        set_language("en")
        apply_language(host)
        assert cell.import_button.text() == english_button
        assert [
            cell.clamp_table.table.horizontalHeaderItem(i).text()
            for i in range(cell.clamp_table.table.columnCount())
        ] == english_headers
        print("English comes back")

        # Every string the walk looks up should be worth translating: none of
        # them may be a bare number, which would mean a measurement got in.
        for text in source_strings(host):
            assert not text.replace(".", "").replace("-", "").strip().isdigit(), text
        print("no measurements in the catalogue")
        """,
        tmp_path,
    )
    assert "the panels speak Italian" in output
    assert "data is untouched" in output
    assert "English comes back" in output
    assert "no measurements in the catalogue" in output


def test_the_guide_says_what_to_do_next(tmp_path: Path):
    """The window has to tell an operator the next step without being asked.

    Four function keys in a fixed order, and pressing them out of order got a
    refusal rather than a result. The guide line and the single "Prepare the
    job" action replace that; this checks the line tracks the state and the
    verdict reads as a sentence rather than as a field dump.
    """
    output = _run(
        """
        from PySide6 import QtWidgets

        from weldforge.app.main_window import MainWindow
        from weldforge.app.project import Project
        from weldforge.app.tools import SelectTool

        window = MainWindow.__new__(MainWindow)
        QtWidgets.QMainWindow.__init__(window)
        window.project = Project()
        window.tool = SelectTool(window.project)
        window.guide = QtWidgets.QLabel("")

        # Nothing loaded: it asks for the CAD.
        window._update_guide()
        empty = window.guide.text()
        assert "CAD" in empty, empty

        # A part with no seams: it points at the one button.
        class _Part:
            name = "x"

        window.project.part = _Part()
        window._update_guide()
        no_seams = window.guide.text()
        assert no_seams != empty and "F9" in no_seams, no_seams
        print("the guide tracks the state")

        # A failed plan names the reason; a good one says it is ready.
        window.project.last_report = {
            "ok": False,
            "errors": ["seam S1: A5 -136.2 deg outside [-130, 130]"],
            "warnings": [],
            "cycle_time_s": 0.0,
            "wire_consumption_kg": 0.0,
        }
        bad = window._verdict()
        assert "A5" in bad, bad

        window.project.last_report = {
            "ok": True,
            "errors": [],
            "warnings": ["seam S1: close to a clamp"],
            "cycle_time_s": 142.5,
            "wire_consumption_kg": 0.08,
        }
        good = window._verdict()
        assert "142" in good and "80" in good, good
        assert "warning" in good.lower() or "avvis" in good.lower(), good
        print("the verdict reads as a sentence")
        """,
        tmp_path,
    )
    assert "the guide tracks the state" in output
    assert "the verdict reads as a sentence" in output


def test_every_background_result_reaches_the_window(tmp_path: Path):
    """A worker's result must arrive even when the pool is quick to clean up.

    The thread pool deletes a runnable the moment ``run()`` returns. With the
    signals object owned by that runnable, the queued ``finished`` emission
    raced the deletion: when deletion won, the slot never ran and an import
    that had succeeded in the worker never reached the window. Operators saw
    it sometimes, and clicked Import again. This drives many short workers
    through a pool and insists on every single result.
    """
    output = _run(
        """
        from PySide6 import QtCore

        from weldforge.app.main_window import _Worker

        pool = QtCore.QThreadPool()
        pool.setMaxThreadCount(4)
        owner = QtCore.QObject()
        got = []
        workers = []
        N = 60

        for i in range(N):
            worker = _Worker(lambda i=i: i * i, parent=owner)
            workers.append(worker)
            worker.signals.finished.connect(lambda r, w=worker: (got.append(r), workers.remove(w)))
            worker.signals.failed.connect(lambda text, w=worker: workers.remove(w))
            pool.start(worker)

        deadline = QtCore.QDeadlineTimer(10_000)
        while (len(got) < N or workers) and not deadline.hasExpired():
            QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        pool.waitForDone(5_000)
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)

        assert sorted(got) == [i * i for i in range(N)], (len(got), N)
        assert not workers, f"{len(workers)} workers never released"
        print("every result arrived")
        """,
        tmp_path,
    )
    assert "every result arrived" in output


def test_the_application_module_exposes_an_entry_point(tmp_path: Path):
    """``weldforge-studio`` has to resolve to something callable."""
    output = _run(
        """
        from importlib.metadata import entry_points

        import weldforge.app as app
        assert callable(app.main)

        found = [
            e for e in entry_points().select(group="gui_scripts")
            if e.name == "weldforge-studio"
        ]
        assert found, "weldforge-studio is not registered"
        assert found[0].value == "weldforge.app:main", found[0].value
        print("entry point:", found[0].value)
        """,
        tmp_path,
    )
    assert "weldforge.app:main" in output
