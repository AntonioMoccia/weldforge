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
