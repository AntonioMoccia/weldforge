"""Putting the window into the operator's language, and back again.

Why a pass over the widgets rather than ``t()`` around every label. The
window's furniture -- a few hundred labels, buttons, tool tips, menu entries
and column headings -- is written once at construction and never computed.
Wrapping each one would touch every line that builds a widget for no gain over
reading them off the built window: the catalogue is keyed by the English text
either way.

What it does not touch is anything holding *data*. Combo box entries are cell,
material and machine names; table cells are seams and clamps the operator
named. Those are not ours to translate, so the walk reads chrome only --
``text``, ``toolTip``, ``placeholderText``, ``title``, ``windowTitle``, and a
table's column headings.

Switching back. Once a label reads "Importa CAD" the English is gone, so each
widget keeps its source string in a Qt dynamic property the first time it is
translated, and every later pass starts from that. Which is what makes English
a language the operator can return to rather than a one-way door.

Messages built while the application runs -- a status line with a seam count,
a refusal from a tool -- carry placeholders and cannot be read off a widget.
Those call :func:`weldforge.i18n.t` where they are written.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import t

logger = logging.getLogger(__name__)

__all__ = ["apply_language", "source_strings"]

#: Qt property prefix under which the English source of each string is kept.
_SOURCE = "wf_source_"

#: A private item-data role for a column heading's English. Qt leaves
#: everything from ``UserRole`` up to applications, and the clamp table
#: already uses ``UserRole`` itself.
_HEADER_ROLE = int(QtCore.Qt.UserRole) + 17

#: (role name, getter, setter) per translatable property. Chrome only.
_ROLES: tuple[tuple[str, str, str], ...] = (
    ("text", "text", "setText"),
    ("tooltip", "toolTip", "setToolTip"),
    ("placeholder", "placeholderText", "setPlaceholderText"),
    ("title", "title", "setTitle"),
    ("window", "windowTitle", "setWindowTitle"),
)

#: Widgets whose ``text`` is data the operator typed, chose or measured, not
#: chrome. A spin box's ``text`` is the number in it: translating that would
#: replace a measurement with whatever the catalogue happened to hold.
_TEXT_IS_DATA = (
    QtWidgets.QLineEdit,
    QtWidgets.QTextEdit,
    QtWidgets.QPlainTextEdit,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QProgressBar,
)


def apply_language(root: Any) -> int:
    """Translate the chrome of *root* and everything under it.

    Returns how many strings changed -- what a test can assert on, and the
    only part of this worth a log line.
    """
    changed = 0
    for widget in [root, *root.findChildren(QtWidgets.QWidget)]:
        changed += _translate(widget)
        if isinstance(widget, QtWidgets.QTableWidget):
            changed += _translate_headers(widget)
    for action in root.findChildren(QtGui.QAction):
        changed += _translate(action)
    for menu in root.findChildren(QtWidgets.QMenu):
        changed += _translate(menu)
    logger.info("language pass: %d strings changed", changed)
    return changed


def source_strings(root: Any) -> list[str]:
    """Every English string this walk looks up, for building a catalogue."""
    found: list[str] = []
    for widget in [root, *root.findChildren(QtWidgets.QWidget)]:
        found += _sources(widget)
        if isinstance(widget, QtWidgets.QTableWidget):
            for column in range(widget.columnCount()):
                item = widget.horizontalHeaderItem(column)
                if item is not None and item.text():
                    found.append(str(item.data(_HEADER_ROLE) or item.text()))
    for action in root.findChildren(QtGui.QAction):
        found += _sources(action)
    for menu in root.findChildren(QtWidgets.QMenu):
        found += _sources(menu)
    return sorted({s for s in found if s.strip()})


# ---------------------------------------------------------------------------
# one object
# ---------------------------------------------------------------------------


def _translate(obj: Any) -> int:
    changed = 0
    for role, getter, setter in _ROLES:
        if role == "text" and isinstance(obj, _TEXT_IS_DATA):
            continue
        read = getattr(obj, getter, None)
        write = getattr(obj, setter, None)
        if read is None or write is None:
            continue
        current = _read(read)
        if not current:
            continue

        key = f"{_SOURCE}{role}"
        source = obj.property(key)
        if not isinstance(source, str) or not source:
            source = current
            obj.setProperty(key, source)
        wanted = t(source)
        if wanted != current:
            write(wanted)
            changed += 1
    return changed


def _translate_headers(table: QtWidgets.QTableWidget) -> int:
    changed = 0
    for column in range(table.columnCount()):
        item = table.horizontalHeaderItem(column)
        if item is None or not item.text():
            continue
        source = item.data(_HEADER_ROLE)
        if not isinstance(source, str) or not source:
            source = item.text()
            item.setData(_HEADER_ROLE, source)
        wanted = t(source)
        if wanted != item.text():
            item.setText(wanted)
            changed += 1
    return changed


def _sources(obj: Any) -> list[str]:
    out: list[str] = []
    for role, getter, _setter in _ROLES:
        if role == "text" and isinstance(obj, _TEXT_IS_DATA):
            continue
        read = getattr(obj, getter, None)
        if read is None:
            continue
        current = _read(read)
        if not current:
            continue
        source = obj.property(f"{_SOURCE}{role}")
        out.append(source if isinstance(source, str) and source else current)
    return out


def _read(getter: Any) -> str:
    """A string property, or empty when the object no longer answers.

    Qt objects can outlive their C++ side during a rebuild, and a language
    pass is not worth a traceback.
    """
    try:
        value = getter()
    except RuntimeError:  # pragma: no cover - C++ object already deleted
        return ""
    return value if isinstance(value, str) else ""
