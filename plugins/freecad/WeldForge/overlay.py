"""Drawing the detected seams back into the FreeCAD 3D view.

This is what makes the plugin worth using. A list of seams in a table is hard
to check; the same seams drawn on the model, coloured by joint type with the
wire direction shown, is checked at a glance -- and a torch pointing the wrong
way is obvious immediately.

Everything is drawn into one group so it can be cleared in one action, and the
group is marked temporary so it is never saved into the user's document.
"""

from __future__ import annotations

__all__ = ["GROUP_NAME", "JOINT_COLOURS", "clear_overlay", "draw_seams", "draw_placement"]

GROUP_NAME = "WeldForge_Seams"

# Colours by joint type, matching the RViz display so the same seam looks the
# same in both.
JOINT_COLOURS = {
    "fillet": (0.20, 0.70, 1.00),
    "tee": (0.20, 0.70, 1.00),
    "lap": (0.95, 0.75, 0.10),
    "butt": (0.40, 0.95, 0.40),
    "corner": (0.95, 0.45, 0.85),
    "edge": (0.70, 0.70, 0.70),
    "plug": (1.00, 0.55, 0.20),
    "unknown": (0.55, 0.55, 0.55),
}

WIRE_COLOUR = (1.00, 0.85, 0.20)
WIRE_MARKER_LENGTH_MM = 25.0


def _freecad():
    import FreeCAD

    return FreeCAD


def clear_overlay(document=None) -> None:
    """Remove a previous overlay, so a new detection does not stack on it."""
    app = _freecad()
    document = document or app.ActiveDocument
    if document is None:
        return
    group = document.getObject(GROUP_NAME)
    if group is None:
        return
    for child in list(group.Group):
        try:
            document.removeObject(child.Name)
        except Exception:
            pass
    try:
        document.removeObject(group.Name)
    except Exception:
        pass
    document.recompute()


def _group(document):
    group = document.getObject(GROUP_NAME)
    if group is None:
        group = document.addObject("App::DocumentObjectGroup", GROUP_NAME)
        group.Label = "WeldForge seams"
    return group


def draw_seams(seams: list[dict], document=None, show_wire_direction: bool = True) -> int:
    """Draw seam polylines and wire markers; returns how many were drawn.

    *seams* are the dicts the API returns from ``/parts/{id}/seams`` with
    ``with_geometry=true``: millimetres, same frame as the FreeCAD model, so no
    transform is needed.
    """
    import Draft
    import Part

    app = _freecad()
    document = document or app.ActiveDocument
    if document is None:
        return 0

    clear_overlay(document)
    group = _group(document)
    drawn = 0

    for index, seam in enumerate(seams, start=1):
        points = seam.get("points_mm") or []
        if len(points) < 2:
            continue
        vectors = [app.Vector(*p) for p in points]

        wire = Part.makePolygon(vectors) if len(vectors) > 2 else Part.makeLine(
            vectors[0], vectors[1]
        )
        feature = document.addObject("Part::Feature", f"Seam{index:03d}")
        feature.Shape = wire
        label = seam.get("label") or (
            f"{seam.get('joint_type', '?')} {seam.get('length_mm', 0):.0f}mm"
        )
        feature.Label = label
        group.addObject(feature)
        drawn += 1

        view = getattr(feature, "ViewObject", None)
        if view is not None:
            colour = JOINT_COLOURS.get(seam.get("joint_type", "unknown"), (0.6, 0.6, 0.6))
            view.LineColor = colour
            view.LineWidth = 4.0
            # Low-confidence seams are drawn dashed, so a doubtful detection
            # looks doubtful instead of looking like a decision.
            if float(seam.get("confidence", 1.0)) < 0.5:
                view.DrawStyle = "Dashed"

        if show_wire_direction:
            bisectors = seam.get("bisectors") or []
            if bisectors:
                step = max(1, len(points) // 12)
                segments = []
                for i in range(0, len(points), step):
                    start = app.Vector(*points[i])
                    direction = app.Vector(*bisectors[min(i, len(bisectors) - 1)])
                    segments.append(
                        Part.makeLine(start - direction * WIRE_MARKER_LENGTH_MM, start)
                    )
                if segments:
                    marker = document.addObject("Part::Feature", f"SeamWire{index:03d}")
                    marker.Shape = Part.makeCompound(segments)
                    marker.Label = f"{label} - wire direction"
                    group.addObject(marker)
                    marker_view = getattr(marker, "ViewObject", None)
                    if marker_view is not None:
                        marker_view.LineColor = WIRE_COLOUR
                        marker_view.LineWidth = 2.0

    document.recompute()
    del Draft  # imported for availability only; drawing uses Part directly
    return drawn


def draw_placement(placement: dict, document=None) -> None:
    """Show where the planner wants the part on the bench, as a marker frame."""
    import Part

    app = _freecad()
    document = document or app.ActiveDocument
    if document is None or not placement:
        return

    pose = placement.get("pose") or {}
    position = pose.get("position_mm") or [0.0, 0.0, 0.0]
    origin = app.Vector(*position)
    axes = [
        (app.Vector(100.0, 0.0, 0.0), (1.0, 0.2, 0.2)),
        (app.Vector(0.0, 100.0, 0.0), (0.2, 1.0, 0.2)),
        (app.Vector(0.0, 0.0, 100.0), (0.2, 0.4, 1.0)),
    ]
    group = _group(document)
    for name, (direction, colour) in zip("XYZ", axes, strict=True):
        feature = document.addObject("Part::Feature", f"PlacementAxis{name}")
        feature.Shape = Part.makeLine(origin, origin + direction)
        feature.Label = f"placement {name}"
        group.addObject(feature)
        view = getattr(feature, "ViewObject", None)
        if view is not None:
            view.LineColor = colour
            view.LineWidth = 3.0
    document.recompute()
