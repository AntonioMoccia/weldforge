"""Getting geometry out of FreeCAD and into WeldForge.

The important decision here is **what** to export, and the answer is: the
selected solids as a STEP assembly, with each body kept separate.

Why separate bodies matter. A weldment's seams are the places two parts touch,
and WeldForge finds them by contact. Export the assembly fused into one solid
-- which several FreeCAD export paths do by default -- and the contacts vanish
along with every seam between parts. Export each body separately and the
contacts are all there.

Placement is preserved as-is. FreeCAD works in millimetres and so does
WeldForge, so nothing is scaled: a 200 mm plate stays 200 mm.
"""

from __future__ import annotations

import os
import tempfile

__all__ = ["ExportError", "selected_objects", "export_step", "describe_selection"]


class ExportError(RuntimeError):
    """Nothing usable was selected, or the export itself failed."""


def _freecad():
    import FreeCAD  # noqa: N813 - FreeCAD's own capitalisation

    return FreeCAD


def _gui():
    import FreeCADGui  # noqa: N813

    return FreeCADGui


def selected_objects() -> list:
    """Shape-bearing objects from the selection, or the whole document.

    Falling back to the whole document is what a user expects: they open a
    weldment and press the button, without selecting anything first.
    """
    app = _freecad()
    try:
        selection = _gui().Selection.getSelection()
    except Exception:
        selection = []

    def has_shape(obj) -> bool:
        shape = getattr(obj, "Shape", None)
        return shape is not None and not shape.isNull()

    objects = [obj for obj in selection if has_shape(obj)]
    if objects:
        return objects

    document = app.ActiveDocument
    if document is None:
        raise ExportError("no document is open in FreeCAD")

    # Only top-level shapes: including both a Part container and its children
    # would export every body twice, and WeldForge would then find each part
    # in contact with its own duplicate.
    objects = [obj for obj in document.Objects if has_shape(obj)]
    consumed: set[str] = set()
    for obj in objects:
        for child in getattr(obj, "Group", []) or []:
            consumed.add(child.Name)
        base = getattr(obj, "Base", None)
        if base is not None:
            consumed.add(base.Name)
        for item in getattr(obj, "Shapes", []) or []:
            consumed.add(item.Name)

    top_level = [obj for obj in objects if obj.Name not in consumed]
    if not top_level:
        raise ExportError("the document contains no solid geometry to weld")
    return top_level


def describe_selection(objects: list) -> str:
    """One line for the panel: how many bodies and how big."""
    solids = 0
    for obj in objects:
        shape = getattr(obj, "Shape", None)
        if shape is not None:
            solids += len(shape.Solids) or 1

    bounds = None
    for obj in objects:
        shape = getattr(obj, "Shape", None)
        if shape is None or shape.isNull():
            continue
        box = shape.BoundBox
        if bounds is None:
            bounds = [box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax]
        else:
            bounds[0] = min(bounds[0], box.XMin)
            bounds[1] = min(bounds[1], box.YMin)
            bounds[2] = min(bounds[2], box.ZMin)
            bounds[3] = max(bounds[3], box.XMax)
            bounds[4] = max(bounds[4], box.YMax)
            bounds[5] = max(bounds[5], box.ZMax)

    if bounds is None:
        return f"{len(objects)} object(s), {solids} solid(s)"
    size = (bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
    return (
        f"{len(objects)} object(s), {solids} solid(s), "
        f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm"
    )


def export_step(objects: list | None = None, path: str | None = None) -> str:
    """Write the selection to a STEP file; returns its path.

    ``Import.export`` is used rather than ``Part.export`` because it writes a
    STEP **assembly** with the bodies kept apart, which is what contact-based
    seam recognition needs. ``Part.export`` on a compound can flatten it.
    """
    import Import

    items = objects if objects is not None else selected_objects()
    if not items:
        raise ExportError("nothing to export")

    if path is None:
        document = _freecad().ActiveDocument
        stem = (document.Name if document is not None else "weldment").replace(" ", "_")
        handle, path = tempfile.mkstemp(prefix=f"{stem}_", suffix=".step")
        os.close(handle)

    try:
        Import.export(items, path)
    except Exception as exc:
        raise ExportError(f"STEP export failed: {exc}") from exc

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise ExportError(f"STEP export produced no data at {path}")
    return path
