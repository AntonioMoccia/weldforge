"""The dock panels.

Qt, but no VTK and no planning logic: a panel shows the project and emits what
the operator asked for. The main window decides what to do about it. That
keeps the panels testable offscreen -- which they are, because Qt widgets run
headless even where the 3-D view cannot.

The panels are, left to right as the window lays them out:

``SetupPanel``       cell, material, consumable, torch
``PlacementPanel``   where the part sits, and how it got there
``SeamPanel``        every weld seam, selectable and editable
``ProcedurePanel``   the welding procedure for one seam, pass by pass
``SimulationPanel``  plan, play back, and the verdict
``OutputPanel``      generate a program, and send it to the machine
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..geom import transform as tf
from ..model import JointType
from .project import Project
from .send import SendTarget

logger = logging.getLogger(__name__)

__all__ = [
    "SetupPanel",
    "PlacementPanel",
    "SeamPanel",
    "ProcedurePanel",
    "SimulationPanel",
    "OutputPanel",
]

_JOINT_BRUSHES = {
    JointType.FILLET: QtGui.QColor(51, 180, 255),
    JointType.TEE: QtGui.QColor(51, 180, 255),
    JointType.LAP: QtGui.QColor(242, 191, 26),
    JointType.BUTT: QtGui.QColor(102, 242, 102),
    JointType.CORNER: QtGui.QColor(242, 115, 217),
    JointType.EDGE: QtGui.QColor(179, 179, 179),
    JointType.PLUG: QtGui.QColor(255, 140, 51),
    JointType.UNKNOWN: QtGui.QColor(140, 140, 140),
}


def _row(label: str, widget: QtWidgets.QWidget) -> tuple[QtWidgets.QLabel, QtWidgets.QWidget]:
    return QtWidgets.QLabel(label), widget


def _spin(
    minimum: float, maximum: float, value: float, step: float = 1.0, decimals: int = 1
) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


class SetupPanel(QtWidgets.QWidget):
    """Cell, material and consumable: the choices everything else follows from."""

    cellChanged = QtCore.Signal(str)
    settingsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from ..fixture import available_cells
        from ..wps import available_machines, load_materials

        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.cell_box = QtWidgets.QComboBox()
        self.cell_box.addItems(available_cells())
        self.cell_box.currentTextChanged.connect(self.cellChanged)
        form.addRow(*_row("Cell", self.cell_box))

        self.material_box = QtWidgets.QComboBox()
        self.material_box.addItems(sorted(load_materials()))
        self.material_box.setCurrentText("S355")
        self.material_box.currentTextChanged.connect(lambda _: self.settingsChanged.emit())
        form.addRow(*_row("Base material", self.material_box))

        self.machine_box = QtWidgets.QComboBox()
        self.machine_box.addItems(available_machines())
        self.machine_box.currentTextChanged.connect(lambda _: self.settingsChanged.emit())
        form.addRow(*_row("Power source", self.machine_box))

        self.wire_box = QtWidgets.QComboBox()
        self.wire_box.addItem("automatic", None)
        for diameter in (0.8, 1.0, 1.2, 1.6):
            self.wire_box.addItem(f"{diameter:.1f} mm", diameter)
        self.wire_box.currentIndexChanged.connect(lambda _: self.settingsChanged.emit())
        form.addRow(*_row("Wire", self.wire_box))

        self.leg_ratio = _spin(0.3, 1.5, 0.7, 0.05, 2)
        self.leg_ratio.setToolTip(
            "Fillet leg as a fraction of the thinner member.\n"
            "0.7 is the usual structural default; 1.0 gives a full-strength fillet."
        )
        self.leg_ratio.valueChanged.connect(lambda _: self.settingsChanged.emit())
        form.addRow(*_row("Fillet leg ratio", self.leg_ratio))

        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: #9aa6b5;")
        form.addRow(self.info)

    # -- binding ------------------------------------------------------------

    def load(self, project: Project) -> None:
        """Show the project's settings without emitting change signals."""
        for widget in (
            self.cell_box,
            self.material_box,
            self.machine_box,
            self.wire_box,
            self.leg_ratio,
        ):
            widget.blockSignals(True)
        self.cell_box.setCurrentText(project.cell_name)
        self.material_box.setCurrentText(project.base_material)
        self.machine_box.setCurrentText(project.machine_profile)
        index = self.wire_box.findData(project.wire_diameter_mm)
        self.wire_box.setCurrentIndex(index if index >= 0 else 0)
        self.leg_ratio.setValue(project.leg_thickness_ratio)
        for widget in (
            self.cell_box,
            self.material_box,
            self.machine_box,
            self.wire_box,
            self.leg_ratio,
        ):
            widget.blockSignals(False)
        self.describe(project)

    def apply(self, project: Project) -> None:
        project.cell_name = self.cell_box.currentText()
        project.base_material = self.material_box.currentText()
        project.machine_profile = self.machine_box.currentText()
        project.wire_diameter_mm = self.wire_box.currentData()
        project.leg_thickness_ratio = float(self.leg_ratio.value())
        project.touch()

    def describe(self, project: Project) -> None:
        if project.cell is None:
            self.info.setText("")
            return
        fixture = project.cell.fixture
        bits = [
            f"{project.cell.name}: {project.cell.robot_name}",
            f"bench {fixture.table_size_mm[0]:.0f} x {fixture.table_size_mm[1]:.0f} mm "
            f"at {fixture.table_z_mm:.0f} mm",
            f"{len(fixture.clamps)} clamp(s)",
        ]
        if fixture.datums:
            bits.append(f"{len(fixture.datums)} jig locator(s)")
        if project.cell.external_axes:
            bits.append("positioner: " + ", ".join(a.name for a in project.cell.external_axes))
        self.info.setText(" | ".join(bits))


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------


class PlacementPanel(QtWidgets.QWidget):
    """Where the part sits, and the four ways of putting it there."""

    actionRequested = QtCore.Signal(str)
    poseEdited = QtCore.Signal(object)
    lockToggled = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        buttons = QtWidgets.QGridLayout()
        for column, (key, label, tip) in enumerate(
            [
                ("drop", "Drop", "Rest the part on the bench where it is in plan"),
                ("snap", "Snap to jig", "Seat the part on the jig's locators"),
                ("auto", "Auto place", "Search for the placement that welds best"),
            ]
        ):
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, k=key: self.actionRequested.emit(k))
            buttons.addWidget(button, 0, column)
        layout.addLayout(buttons)

        nudge = QtWidgets.QGridLayout()
        moves = [
            ("forward", "Y+", 0, 1),
            ("left", "X-", 1, 0),
            ("right", "X+", 1, 2),
            ("back", "Y-", 2, 1),
            ("up", "Z+", 0, 3),
            ("down", "Z-", 2, 3),
            ("yaw_ccw", "rot +", 1, 4),
            ("yaw_cw", "rot -", 1, 5),
        ]
        for key, label, r, c in moves:
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setAutoRepeat(True)
            button.clicked.connect(lambda _=False, k=key: self.actionRequested.emit(k))
            nudge.addWidget(button, r, c)
        layout.addLayout(nudge)

        form = QtWidgets.QFormLayout()
        self.x = _spin(-5000, 5000, 0.0, 10.0, 2)
        self.y = _spin(-5000, 5000, 0.0, 10.0, 2)
        self.z = _spin(-5000, 5000, 0.0, 10.0, 2)
        self.rx = _spin(-180, 180, 0.0, 5.0, 2)
        self.ry = _spin(-180, 180, 0.0, 5.0, 2)
        self.rz = _spin(-180, 180, 0.0, 5.0, 2)
        for label, widget in (
            ("X mm", self.x),
            ("Y mm", self.y),
            ("Z mm", self.z),
            ("Roll deg", self.rx),
            ("Pitch deg", self.ry),
            ("Yaw deg", self.rz),
        ):
            widget.valueChanged.connect(self._emit_pose)
            form.addRow(*_row(label, widget))
        layout.addLayout(form)

        self.lock = QtWidgets.QCheckBox("Lock the placement")
        self.lock.setToolTip(
            "Stops the part being moved by accident once it is set.\n"
            "On a jig the placement is determined by the locators, so lock it."
        )
        self.lock.toggled.connect(self.lockToggled)
        layout.addWidget(self.lock)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #9aa6b5;")
        layout.addWidget(self.status)
        layout.addStretch(1)
        self._loading = False

    def _emit_pose(self) -> None:
        if self._loading:
            return
        rotation = tf.rpy_to_matrix(
            np.deg2rad(self.rx.value()),
            np.deg2rad(self.ry.value()),
            np.deg2rad(self.rz.value()),
        )
        self.poseEdited.emit(tf.pose(rotation, [self.x.value(), self.y.value(), self.z.value()]))

    def load(self, project: Project) -> None:
        self._loading = True
        position = tf.translation_of(project.placement)
        roll, pitch, yaw = tf.matrix_to_rpy(project.placement)
        self.x.setValue(float(position[0]))
        self.y.setValue(float(position[1]))
        self.z.setValue(float(position[2]))
        self.rx.setValue(float(np.rad2deg(roll)))
        self.ry.setValue(float(np.rad2deg(pitch)))
        self.rz.setValue(float(np.rad2deg(yaw)))
        self.lock.setChecked(project.placement_locked)
        self._loading = False

    def set_status(self, text: str) -> None:
        self.status.setText(text)


# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------


class SeamPanel(QtWidgets.QWidget):
    """Every seam, with what it is and whether it will be welded."""

    seamSelected = QtCore.Signal(str)
    seamToggled = QtCore.Signal(str, bool)
    actionRequested = QtCore.Signal(str)
    deleteRequested = QtCore.Signal(str)
    zoomRequested = QtCore.Signal(str)

    COLUMNS = ["", "Seam", "mm", "Pos", "Groove", "Thk", "Conf", ""]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        for key, label, tip in [
            ("detect", "Detect", "Find the seams from the geometry"),
            ("trace", "Trace", "Draw a weld line by clicking the model"),
            ("all_on", "All on", "Weld every seam"),
            ("all_off", "All off", "Weld none of them"),
        ]:
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, k=key: self.actionRequested.emit(k))
            bar.addWidget(button)
        layout.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(0, 26)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemDoubleClicked.connect(
            lambda item: self.zoomRequested.emit(self._id_at(item.row()))
        )
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.table, 1)

        self.summary = QtWidgets.QLabel("no seams")
        self.summary.setStyleSheet("color: #9aa6b5;")
        layout.addWidget(self.summary)
        self._ids: list[str] = []
        self._loading = False

    def _id_at(self, row: int) -> str:
        return self._ids[row] if 0 <= row < len(self._ids) else ""

    def _on_selection(self) -> None:
        if self._loading:
            return
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.seamSelected.emit(self._id_at(rows[0].row()))

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._loading or item.column() != 0:
            return
        self.seamToggled.emit(self._id_at(item.row()), item.checkState() == QtCore.Qt.Checked)

    def _context_menu(self, position: QtCore.QPoint) -> None:
        row = self.table.rowAt(position.y())
        if row < 0:
            return
        seam_id = self._id_at(row)
        menu = QtWidgets.QMenu(self)
        menu.addAction("Zoom to", lambda: self.zoomRequested.emit(seam_id))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self.deleteRequested.emit(seam_id))
        menu.exec(self.table.viewport().mapToGlobal(position))

    def load(self, project: Project) -> None:
        self._loading = True
        self.table.setRowCount(0)
        self._ids = []
        for seam in project.seams:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._ids.append(seam.id)

            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            check.setCheckState(QtCore.Qt.Checked if seam.selected else QtCore.Qt.Unchecked)
            self.table.setItem(row, 0, check)

            # The operator's own name for the seam wins over the joint type:
            # "cordone 3 - lato destro" is what they call it at the machine,
            # and the joint type is one hover away.
            joint = QtWidgets.QTableWidgetItem(seam.label or seam.joint_type.value)
            joint.setForeground(_JOINT_BRUSHES.get(seam.joint_type, QtGui.QColor("white")))
            tooltip = f"{seam.joint_type.value} - {seam.id}"
            if project.is_manual(seam.id):
                font = joint.font()
                font.setItalic(True)
                joint.setFont(font)
                tooltip += " - traced by hand"
            joint.setToolTip(tooltip)
            self.table.setItem(row, 1, joint)

            values = [
                f"{seam.length_mm:.0f}",
                seam.dominant_position.value,
                f"{np.rad2deg(seam.mean_dihedral):.0f}",
                f"{seam.min_thickness_mm:.1f}",
                f"{seam.confidence:.2f}",
            ]
            for offset, text in enumerate(values, start=2):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(row, offset, item)

            flag = QtWidgets.QTableWidgetItem("H" if project.is_manual(seam.id) else "")
            flag.setToolTip("traced by hand" if project.is_manual(seam.id) else "")
            self.table.setItem(row, 7, flag)

            # A doubtful detection should look doubtful.
            if seam.confidence < 0.5 and not project.is_manual(seam.id):
                for column in range(self.table.columnCount()):
                    cell = self.table.item(row, column)
                    if cell is not None:
                        cell.setBackground(QtGui.QColor(70, 56, 30))

        self._loading = False
        selected = len(project.selected_seams)
        self.summary.setText(
            f"{len(project.seams)} seams, {selected} to weld, "
            f"{project.total_weld_length_mm:.0f} mm"
            + (f", {len(project.manual_seam_ids)} by hand" if project.manual_seam_ids else "")
        )

    def select(self, seam_id: str) -> None:
        if seam_id in self._ids:
            self.table.selectRow(self._ids.index(seam_id))


# ---------------------------------------------------------------------------
# procedure
# ---------------------------------------------------------------------------


class ProcedurePanel(QtWidgets.QWidget):
    """The welding procedure for one seam, pass by pass and editable."""

    regenerateRequested = QtCore.Signal(str)
    passEdited = QtCore.Signal(str, int, str, float)

    COLUMNS = ["Pass", "Leg", "A", "V", "m/min", "mm/s", "Mode", "Work", "Travel", "Weave", "kJ/mm"]
    EDITABLE = {
        1: "leg_mm",
        2: "current_a",
        3: "voltage_v",
        4: "wire_speed_m_min",
        5: "travel_speed_mm_s",
        7: "work_angle_deg",
        8: "travel_angle_deg",
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.header = QtWidgets.QLabel("no seam selected")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        self.warnings = QtWidgets.QTextEdit()
        self.warnings.setReadOnly(True)
        self.warnings.setMaximumHeight(90)
        layout.addWidget(self.warnings)

        bar = QtWidgets.QHBoxLayout()
        self.regen_one = QtWidgets.QPushButton("Regenerate this seam")
        self.regen_one.clicked.connect(lambda: self.regenerateRequested.emit(self._seam_id))
        self.regen_all = QtWidgets.QPushButton("Regenerate all")
        self.regen_all.setToolTip(
            "Hand-edited procedures are left alone; everything else is rebuilt."
        )
        self.regen_all.clicked.connect(lambda: self.regenerateRequested.emit(""))
        bar.addWidget(self.regen_one)
        bar.addWidget(self.regen_all)
        layout.addLayout(bar)

        self._seam_id = ""
        self._loading = False

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._loading or not self._seam_id:
            return
        field = self.EDITABLE.get(item.column())
        if field is None:
            return
        try:
            value = float(item.text())
        except ValueError:
            item.setText("")
            return
        self.passEdited.emit(self._seam_id, item.row(), field, value)

    @property
    def current_seam_id(self) -> str:
        """Which seam the table is showing, or "" for none."""
        return self._seam_id

    def show_seam(self, project: Project, seam_id: str) -> None:
        self._loading = True
        self._seam_id = seam_id
        seam = project.seam(seam_id)
        procedure = project.procedures.get(seam_id)
        self.table.setRowCount(0)

        if seam is None:
            self.header.setText("no seam selected")
            self.warnings.setPlainText("")
            self._loading = False
            return
        if procedure is None:
            self.header.setText(
                f"{seam.label or seam.id}: no procedure yet. "
                "Generate the procedures to fill this in."
            )
            self.warnings.setPlainText("")
            self._loading = False
            return

        edited = " (hand edited)" if seam_id in project.edited_procedure_ids else ""
        self.header.setText(
            f"<b>{seam.label or seam.id}</b>{edited}<br>"
            f"{procedure.joint_type.value} in {procedure.position.value}, "
            f"{procedure.thickness_mm:.1f} mm {procedure.base_material}, "
            f"leg {procedure.target_leg_mm:.1f} mm, {procedure.n_passes} pass(es), "
            f"preheat {procedure.preheat_c:.0f} C"
        )

        for weld_pass in procedure.passes:
            row = self.table.rowCount()
            self.table.insertRow(row)
            cells = [
                weld_pass.name,
                f"{weld_pass.leg_mm:.1f}" if weld_pass.leg_mm else "-",
                f"{weld_pass.current_a:.0f}",
                f"{weld_pass.voltage_v:.1f}",
                f"{weld_pass.wire_speed_m_min:.2f}",
                f"{weld_pass.travel_speed_mm_s:.2f}",
                weld_pass.transfer_mode.value,
                f"{weld_pass.work_angle_deg:.0f}",
                f"{weld_pass.travel_angle_deg:.0f}",
                weld_pass.weave.shape.value if weld_pass.weave.active else "-",
                f"{weld_pass.heat_input_kj_mm():.2f}",
            ]
            for column, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if column not in self.EDITABLE:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    item.setForeground(QtGui.QColor(154, 166, 181))
                if column:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(row, column, item)

        messages = list(procedure.warnings)
        if seam_id in project.edited_procedure_ids:
            messages.insert(
                0,
                "This procedure was edited by hand. Regenerating all procedures "
                "will leave it alone; regenerating this one replaces your edits.",
            )
        self.warnings.setPlainText("\n".join(f"- {m}" for m in messages))
        self._loading = False


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


class SimulationPanel(QtWidgets.QWidget):
    """Plan it, watch it, and read the verdict."""

    planRequested = QtCore.Signal()
    frameChanged = QtCore.Signal(int)
    playToggled = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.plan_button = QtWidgets.QPushButton("Plan the job")
        self.plan_button.setToolTip(
            "Generate the procedures, place the part, build the torch paths and simulate the cycle."
        )
        self.plan_button.clicked.connect(self.planRequested)
        layout.addWidget(self.plan_button)

        self.verdict = QtWidgets.QLabel("not planned yet")
        self.verdict.setWordWrap(True)
        layout.addWidget(self.verdict)

        self.numbers = QtWidgets.QTableWidget(0, 2)
        self.numbers.horizontalHeader().setVisible(False)
        self.numbers.verticalHeader().setVisible(False)
        self.numbers.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.numbers.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.numbers.setMaximumHeight(190)
        layout.addWidget(self.numbers)

        controls = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton("Play")
        self.play.setCheckable(True)
        self.play.toggled.connect(self._on_play)
        controls.addWidget(self.play)
        self.scrub = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrub.setEnabled(False)
        self.scrub.valueChanged.connect(self.frameChanged)
        controls.addWidget(self.scrub, 1)
        self.clock = QtWidgets.QLabel("0 / 0")
        controls.addWidget(self.clock)
        layout.addLayout(controls)

        self.messages = QtWidgets.QTextEdit()
        self.messages.setReadOnly(True)
        layout.addWidget(self.messages, 1)

    def _on_play(self, playing: bool) -> None:
        self.play.setText("Pause" if playing else "Play")
        self.playToggled.emit(playing)

    def set_frames(self, count: int) -> None:
        self.scrub.setEnabled(count > 1)
        self.scrub.setMaximum(max(0, count - 1))
        self.clock.setText(f"0 / {max(0, count - 1)}")

    def set_frame(self, index: int, label: str = "") -> None:
        self.scrub.blockSignals(True)
        self.scrub.setValue(index)
        self.scrub.blockSignals(False)
        self.clock.setText(label or f"{index} / {self.scrub.maximum()}")

    def show_report(self, report: dict[str, Any]) -> None:
        if not report:
            self.verdict.setText("not planned yet")
            self.numbers.setRowCount(0)
            self.messages.setPlainText("")
            return

        ok = bool(report.get("ok"))
        self.verdict.setText(
            f"<b style='color:{'#a3be8c' if ok else '#bf616a'}'>"
            f"{'Plan OK' if ok else 'Plan reported problems'}</b>"
        )

        rows = [
            ("Weld length", f"{report.get('total_weld_length_mm', 0):.0f} mm"),
            ("Cycle time", f"{report.get('cycle_time_s', 0):.0f} s"),
            ("Arc time", f"{report.get('arc_time_s', 0):.0f} s"),
            ("Duty cycle", f"{report.get('duty_cycle', 0) * 100:.0f} %"),
            ("Wire", f"{report.get('wire_consumption_kg', 0) * 1000:.0f} g"),
            ("Gas", f"{report.get('gas_consumption_l', 0):.0f} l"),
            ("Reachable", "yes" if report.get("reachable") else "NO"),
            ("Collision free", "yes" if report.get("collision_free") else "NO"),
        ]
        self.numbers.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.numbers.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
            item = QtWidgets.QTableWidgetItem(value)
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            if value == "NO":
                item.setForeground(QtGui.QColor("#bf616a"))
            self.numbers.setItem(row, 1, item)

        lines = [f"ERROR  {e}" for e in report.get("errors", [])]
        lines += [f"warn   {w}" for w in report.get("warnings", [])]
        self.messages.setPlainText("\n".join(lines) or "no messages")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


class OutputPanel(QtWidgets.QWidget):
    """Generate the robot program, and get it to the machine."""

    generateRequested = QtCore.Signal(str)
    sendRequested = QtCore.Signal(str, object, bool)
    browseRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from ..post import available_targets

        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.target_box = QtWidgets.QComboBox()
        self.target_box.addItems(available_targets())
        self.target_box.setCurrentText("rapid")
        self.target_box.currentTextChanged.connect(lambda _: self.preview.clear())
        top.addWidget(QtWidgets.QLabel("Controller"))
        top.addWidget(self.target_box, 1)
        generate = QtWidgets.QPushButton("Generate")
        generate.clicked.connect(lambda: self.generateRequested.emit(self.target_box.currentText()))
        top.addWidget(generate)
        layout.addLayout(top)

        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QtGui.QFont("Menlo", 10))
        self.preview.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        layout.addWidget(self.preview, 1)

        box = QtWidgets.QGroupBox("Send to the machine")
        form = QtWidgets.QFormLayout(box)
        self.method_box = QtWidgets.QComboBox()
        self.method_box.addItem("Folder or share", "folder")
        self.method_box.addItem("Controller over FTP", "ftp")
        self.method_box.addItem("On-machine cell software", "cell")
        self.method_box.currentIndexChanged.connect(self._update_fields)
        form.addRow(*_row("Route", self.method_box))

        destination_row = QtWidgets.QHBoxLayout()
        self.destination = QtWidgets.QLineEdit()
        destination_row.addWidget(self.destination, 1)
        self.browse = QtWidgets.QToolButton()
        self.browse.setText("...")
        self.browse.clicked.connect(self.browseRequested)
        destination_row.addWidget(self.browse)
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(destination_row)
        form.addRow(*_row("Destination", wrapper))

        self.username = QtWidgets.QLineEdit()
        self.password = QtWidgets.QLineEdit()
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.remote_dir = QtWidgets.QLineEdit()
        form.addRow(*_row("User", self.username))
        form.addRow(*_row("Password", self.password))
        form.addRow(*_row("Remote folder", self.remote_dir))

        self.force = QtWidgets.QCheckBox("Send even though the plan has problems")
        self.force.setToolTip(
            "A job whose own report found a collision or an unreachable point "
            "is not sent by accident. Tick this only after a dry run."
        )
        form.addRow(self.force)

        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self._emit_send)
        form.addRow(self.send_button)
        layout.addWidget(box)

        self.history = QtWidgets.QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setMaximumHeight(90)
        layout.addWidget(self.history)

        self._update_fields()

    def _update_fields(self) -> None:
        method = self.method_box.currentData()
        ftp = method == "ftp"
        for widget in (self.username, self.password, self.remote_dir):
            widget.setEnabled(ftp)
        self.browse.setEnabled(method == "folder")
        self.destination.setPlaceholderText(
            {
                "folder": "where to write the program",
                "ftp": "controller address, e.g. 192.168.1.10",
                "cell": "cell address, e.g. http://192.168.1.50:8000",
            }[method]
        )

    def _emit_send(self) -> None:
        self.sendRequested.emit(
            self.target_box.currentText(), self.target(), self.force.isChecked()
        )

    def target(self) -> SendTarget:
        return SendTarget(
            method=self.method_box.currentData(),
            destination=self.destination.text().strip(),
            username=self.username.text().strip(),
            password=self.password.text(),
            remote_dir=self.remote_dir.text().strip(),
        )

    def set_destination(self, text: str) -> None:
        """Fill in the destination, after the operator browsed for a folder."""
        self.destination.setText(text)

    def show_program(self, files: dict[str, str]) -> None:
        if not files:
            self.preview.setPlainText("")
            return
        name, text = next(iter(files.items()))
        header = f"=== {name} ===\n"
        if len(files) > 1:
            header = f"=== {name} (and {len(files) - 1} more) ===\n"
        # Enough to check the header and the first moves; the whole file can
        # be megabytes and the preview is for a glance, not for reading.
        body = text if len(text) < 60000 else text[:60000] + "\n... truncated ..."
        self.preview.setPlainText(header + body)

    def show_history(self, project: Project) -> None:
        if not project.sent:
            self.history.setPlainText("nothing sent yet")
            return
        lines = [
            f"{r.when}  {r.target:9s} {r.method:7s} {'ok ' if r.ok else 'FAIL'} {r.destination}"
            for r in project.sent[-8:]
        ]
        self.history.setPlainText("\n".join(lines))
