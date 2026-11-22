"""Dihedral geometry of a weld joint.

This module answers, at every station along a candidate seam:

* which way does the material lie,
* what is the included angle **on the weld side**,
* which way must the wire point.

Why the sign work matters
-------------------------
Take the edge along +X, and measure directions in the Y-Z plane as angles from
+Y. Let the solid material fill the angular sector ``[0, theta]``. Then:

* ``d_a`` -- the in-face direction leading away from the edge into face A --
  sits at angle 0, and ``d_b`` at angle ``theta``;
* face A's **outward** normal ``n_a`` therefore points at angle 270 deg, and
  face B's at ``theta + 90`` deg;
* the angle the two in-face directions span is
  ``phi = theta`` for ``theta <= 180``, else ``360 - theta``;
* and ``dot(n_a, d_b) = -sin(theta)``.

So the sign of ``dot(n_a, d_b)`` tells concavity without any extra geometry:

======================  =================  ===========================
``dot(n_a, d_b)``       material angle     groove (weld-side) angle
======================  =================  ===========================
``< 0``  (convex)       ``phi``            ``360 - phi``
``> 0``  (concave)      ``360 - phi``      ``phi``
``~ 0``  (tangent)      180 deg            180 deg
======================  =================  ===========================

Worked check -- the two cases that look identical if you only measure ``phi``:

* **Inside corner of an L-bracket**: ``theta = 270``, ``phi = 90``,
  ``dot > 0`` -> groove 90 deg. A fillet weld. Correct.
* **Outside edge of a cube**: ``theta = 90``, ``phi = 90``, ``dot < 0`` ->
  groove 270 deg. An outside corner, welded quite differently. Correct.

Both give ``phi = 90``; only the concavity sign separates them. A detector
that classifies on ``phi`` alone -- which is the usual shortcut -- calls every
outside corner of the part a fillet seam and fills the program with welds that
do not exist.

Wire direction
--------------
``bisector = -normalise(n_a + n_b)``: the two outward normals sum to the
outward bisector of the weld-side wedge, so its negation points from the air
into the joint root. It is right for fillets, butts, laps and outside corners
alike. It degenerates only when ``n_a == -n_b`` (parallel faces, an edge/slit
weld), where the fallback ``-normalise(d_a + d_b)`` takes over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..cad.backend import CadBackend
from ..cad.topology import FaceEvaluator
from ..geom import transform as tf

logger = logging.getLogger(__name__)

__all__ = [
    "TANGENT_TOLERANCE",
    "InwardMode",
    "inward_from_opposite_normal",
    "groove_angle",
    "material_angle",
    "joint_bisector",
    "in_face_direction",
    "point_state_on_face",
    "point_on_face",
    "resolve_inward_sign",
    "FaceSide",
    "SeamGeometry",
    "compute_seam_geometry",
    "validate_in_face_directions",
]

# |dot(n_a, d_b)| below this counts as tangent (faces effectively coplanar).
# sin(0.5 deg) -- tight enough that a real 1 deg bevel is still classified,
# loose enough to absorb OCCT's surface-normal noise on a B-spline.
TANGENT_TOLERANCE = 0.0087


# ---------------------------------------------------------------------------
# pure maths -- no CAD dependency, exhaustively testable
# ---------------------------------------------------------------------------


def groove_angle(
    n_a: np.ndarray,
    d_a: np.ndarray,
    n_b: np.ndarray,
    d_b: np.ndarray,
    tangent_tolerance: float = TANGENT_TOLERANCE,
) -> tuple[float, int]:
    """Included angle on the **weld side**, in radians, plus the concavity sign.

    Parameters
    ----------
    n_a, n_b
        Outward unit normals of the two faces at the station.
    d_a, d_b
        Unit in-face directions leading away from the seam into each face.

    Returns
    -------
    (angle, concavity)
        *angle* in ``(0, 2*pi)``; *concavity* is ``+1`` concave (material more
        than half-space -- an inside corner, where a fillet goes), ``-1``
        convex (an outside corner), ``0`` tangent (coplanar -- a butt).
    """
    ua, ub = tf.unit(d_a), tf.unit(d_b)
    na, nb = tf.unit(n_a), tf.unit(n_b)
    if ua is None or ub is None or na is None or nb is None:
        return np.pi, 0

    phi = float(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0)))
    # Average the two equivalent concavity probes: dot(n_a, d_b) and
    # dot(n_b, d_a) agree analytically, and averaging halves the numeric noise
    # on tessellated or spline surfaces.
    probe = 0.5 * (float(np.dot(na, ub)) + float(np.dot(nb, ua)))

    if abs(probe) <= tangent_tolerance:
        return np.pi, 0
    if probe > 0.0:
        return phi, 1
    return 2.0 * np.pi - phi, -1


def material_angle(
    n_a: np.ndarray,
    d_a: np.ndarray,
    n_b: np.ndarray,
    d_b: np.ndarray,
    tangent_tolerance: float = TANGENT_TOLERANCE,
) -> float:
    """Included angle filled by **material**, in radians -- the groove's complement."""
    groove, _ = groove_angle(n_a, d_a, n_b, d_b, tangent_tolerance)
    return 2.0 * np.pi - groove


def joint_bisector(
    n_a: np.ndarray,
    n_b: np.ndarray,
    d_a: np.ndarray | None = None,
    d_b: np.ndarray | None = None,
) -> np.ndarray:
    """Unit direction the **wire must point**, from the air into the joint root."""
    na, nb = tf.unit(n_a), tf.unit(n_b)
    if na is None or nb is None:
        return np.array([0.0, 0.0, -1.0])

    bis = tf.unit(na + nb)
    if bis is not None:
        return -bis

    # n_a == -n_b: parallel faces, an edge or slit weld. The normals carry no
    # information about where the joint is, so use the in-face directions.
    if d_a is not None and d_b is not None:
        alt = tf.unit(np.asarray(d_a, dtype=float) + np.asarray(d_b, dtype=float))
        if alt is not None:
            return -alt
        # Exactly opposed in-face directions too: the seam is a flat slit, so
        # approach square to both faces.
        return na
    return na


class InwardMode:
    """How to work out which way each flank leads away from the seam.

    ``TRIM``
        Probe the face's 2-D trim with :func:`resolve_inward_sign`. Correct
        when the faces genuinely **end** at the seam, which is the case for a
        shared topological edge.

    ``CONCAVE``
        Project the opposite flank's outward normal into this face's tangent
        plane. Correct -- and far more robust -- when the faces **continue
        past** the seam, which is exactly what a contact joint looks like: the
        base plate's top face runs on underneath the web, so trim probing finds
        material on both sides and cannot decide. Valid only for concave
        joints, which every contact joint is by construction.

    ``LATERAL``
        Take the in-face directions from an explicitly supplied lateral vector.
        Used for butt joints, where the two flanks are coplanar and neither of
        the other two rules has any signal to work from.
    """

    TRIM = "trim"
    CONCAVE = "concave"
    LATERAL = "lateral"


def inward_from_opposite_normal(
    n_self: np.ndarray,
    n_other: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray | None:
    """In-face direction for a **concave** joint, from the opposite normal.

    The other flank's outward normal points away from the other body. Projected
    into this face's tangent plane (and squared to the seam) it becomes "along
    this face, away from the other body" -- which is the weld-side direction.

    Derivation check against the wedge model in this module's docstring: for
    material filling ``[0, theta]``, projecting ``n_b`` into A's tangent plane
    gives ``(0, -sign(sin theta), 0)``. For a concave joint
    (``180 < theta < 360``, so ``sin theta < 0``) that is ``+Y`` -- exactly
    ``d_a``. For a convex joint it comes out reversed, which is why this rule
    is restricted to concave joints.

    Returns ``None`` when the two normals are parallel or antiparallel, leaving
    nothing to project (a butt joint, or a fold).
    """
    ns = tf.unit(n_self)
    no = tf.unit(n_other)
    t = tf.unit(tangent)
    if ns is None or no is None:
        return None
    projected = no - float(np.dot(no, ns)) * ns
    if t is not None:
        # Also square it to the seam, so a slightly skew normal does not tilt
        # the in-face direction along the weld.
        projected = projected - float(np.dot(projected, t)) * t
    return tf.unit(projected)


def in_face_direction(normal: np.ndarray, tangent: np.ndarray, sign: float) -> np.ndarray:
    """In-face direction perpendicular to the seam: ``sign * cross(normal, tangent)``.

    Both candidates lie in the face's tangent plane and square to the seam;
    only one leads into the trimmed face, and *sign* is what
    :func:`resolve_inward_sign` determined once for the whole seam.
    """
    d = tf.unit(np.cross(np.asarray(normal, dtype=float), np.asarray(tangent, dtype=float)))
    if d is None:
        return np.array([1.0, 0.0, 0.0])
    return d * (1.0 if sign >= 0.0 else -1.0)


# ---------------------------------------------------------------------------
# CAD-side: which way is into the face?
# ---------------------------------------------------------------------------


def point_state_on_face(
    backend: CadBackend,
    face: Any,
    evaluator: FaceEvaluator,
    point: np.ndarray,
    tol: float = 1e-4,
) -> str:
    """``'in'`` | ``'on'`` | ``'out'`` | ``'unknown'`` for a point against a face's trim.

    The surface is unbounded; the *face* is the trimmed patch of it. Projecting
    to ``(u, v)`` and classifying against the face's 2-D boundary is the only
    way to tell "on the plane of the plate" from "on the plate".
    """
    uv = evaluator._project(point)  # noqa: SLF001 - evaluator is ours
    if uv is None:
        return "unknown"
    try:
        classifier_cls = backend.first_sym(
            (
                ("IntTools", "IntTools_FClass2d"),
                ("BRepTopAdaptor", "BRepTopAdaptor_FClass2d"),
            )
        )
        gp = backend.module("gp")
        state_enum = backend.sym("TopAbs", "TopAbs_State")
        classifier = classifier_cls(face, tol)
        state = classifier.Perform(gp.gp_Pnt2d(float(uv[0]), float(uv[1])))
        if state == state_enum.TopAbs_IN:
            return "in"
        if state == state_enum.TopAbs_ON:
            return "on"
        if state == state_enum.TopAbs_OUT:
            return "out"
        return "unknown"
    except Exception as exc:  # pragma: no cover - binding specific
        logger.debug("point_state_on_face failed: %s", exc)
        return "unknown"


def point_on_face(
    backend: CadBackend,
    face: Any,
    evaluator: FaceEvaluator,
    point: np.ndarray,
    distance_tol: float = 0.05,
    uv_tol: float = 1e-4,
) -> bool:
    """True if *point* lies on the trimmed face, within *distance_tol* mm.

    Two independent conditions, both required: close to the **surface**, and
    inside the face's 2-D **trim**. Checking only the first is what makes a
    naive contact finder weld along the plane of a plate far outside the plate
    itself.
    """
    if evaluator.distance_to_point(point) > distance_tol:
        return False
    return point_state_on_face(backend, face, evaluator, point, uv_tol) in ("in", "on")


def resolve_inward_sign(
    backend: CadBackend,
    face: Any,
    evaluator: FaceEvaluator,
    point: np.ndarray,
    tangent: np.ndarray,
    probes_mm: tuple[float, ...] = (0.5, 2.0, 0.1, 5.0),
) -> float:
    """Which of ``+-cross(n, t)`` leads **into** the trimmed face: ``+1`` or ``-1``.

    Determined once per (seam, face) pair at a representative station, then
    reused along the seam: the normal and tangent vary continuously, so the
    handedness cannot flip without the face folding back on itself.

    Several probe distances are tried in turn, because one fixed distance
    cannot suit both a 1 mm chamfer and a 500 mm plate: too large steps off a
    narrow face, too small lands inside the boundary tolerance. If no probe is
    decisive, the face's area centroid breaks the tie -- less exact, but it
    cannot return a nonsense direction.
    """
    normal = evaluator.normal_at_point(point)
    base = tf.unit(np.cross(normal, np.asarray(tangent, dtype=float)))
    if base is None:
        return 1.0

    for probe in probes_mm:
        plus = point + base * probe
        minus = point - base * probe
        in_plus = point_state_on_face(backend, face, evaluator, plus) in ("in",)
        in_minus = point_state_on_face(backend, face, evaluator, minus) in ("in",)
        if in_plus and not in_minus:
            return 1.0
        if in_minus and not in_plus:
            return -1.0

    # Tie-break on the face centroid: the in-face direction must at least point
    # towards the bulk of the face.
    _, centroid = backend.surface_properties(face)
    towards = centroid - np.asarray(point, dtype=float)
    projected = float(np.dot(towards, base))
    if abs(projected) > 1e-9:
        return 1.0 if projected > 0.0 else -1.0
    logger.debug("resolve_inward_sign: undecidable, defaulting to +1")
    return 1.0


# ---------------------------------------------------------------------------
# per-seam geometry
# ---------------------------------------------------------------------------


@dataclass
class FaceSide:
    """One flank of a seam: the face, how to evaluate it, and its inward sign."""

    face_id: str
    solid_id: str
    face: Any
    evaluator: FaceEvaluator
    inward_sign: float = 1.0
    thickness_mm: float = 0.0


@dataclass
class SeamGeometry:
    """Per-station geometry of a seam, all arrays of matching length *N*."""

    points: np.ndarray
    tangents: np.ndarray
    normal_a: np.ndarray
    normal_b: np.ndarray
    inward_a: np.ndarray
    inward_b: np.ndarray
    groove: np.ndarray  # (N,) radians, weld-side included angle
    concavity: np.ndarray  # (N,) +1 concave / -1 convex / 0 tangent
    bisector: np.ndarray  # (N, 3) wire direction, into the joint
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def mean_groove(self) -> float:
        return float(np.mean(self.groove)) if self.n else np.pi

    @property
    def groove_spread(self) -> float:
        """Peak-to-peak variation of the groove angle along the seam.

        A large spread means the two faces are not at a constant angle, so one
        set of welding parameters will not fit the whole seam -- the detector
        splits such a seam rather than averaging it away.
        """
        return float(np.ptp(self.groove)) if self.n else 0.0

    @property
    def dominant_concavity(self) -> int:
        if not self.n:
            return 0
        values, counts = np.unique(self.concavity, return_counts=True)
        return int(values[int(np.argmax(counts))])


def compute_seam_geometry(
    points: np.ndarray,
    tangents: np.ndarray,
    side_a: FaceSide,
    side_b: FaceSide,
    mode: str = InwardMode.TRIM,
    lateral: np.ndarray | None = None,
    tangent_tolerance: float = TANGENT_TOLERANCE,
) -> SeamGeometry:
    """Evaluate normals, groove angle and wire direction at every station.

    *mode* selects how the in-face directions are derived; see
    :class:`InwardMode`. For ``LATERAL``, *lateral* is the direction from
    flank A towards flank B, either one vector or one per station.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    tangents = np.asarray(tangents, dtype=float).reshape(-1, 3)
    n = len(points)

    if mode == InwardMode.LATERAL:
        if lateral is None:
            raise ValueError("InwardMode.LATERAL requires a lateral direction")
        lat = np.asarray(lateral, dtype=float).reshape(-1, 3)
        if len(lat) == 1:
            lat = np.repeat(lat, n, axis=0)
        elif len(lat) != n:
            raise ValueError(f"lateral must have 1 or {n} rows, got {len(lat)}")
    else:
        lat = None

    normal_a = np.empty((n, 3))
    normal_b = np.empty((n, 3))
    inward_a = np.empty((n, 3))
    inward_b = np.empty((n, 3))
    groove = np.empty(n)
    concavity = np.empty(n, dtype=int)
    bisector = np.empty((n, 3))
    fallbacks = 0

    for i in range(n):
        p, t = points[i], tangents[i]
        na = side_a.evaluator.normal_at_point(p)
        nb = side_b.evaluator.normal_at_point(p)

        if mode == InwardMode.CONCAVE:
            da = inward_from_opposite_normal(na, nb, t)
            db = inward_from_opposite_normal(nb, na, t)
            if da is None or db is None:
                # Parallel flanks: nothing to project. Fall back to the trim
                # signs, which at least respect the faces' own extent.
                da = in_face_direction(na, t, side_a.inward_sign)
                db = in_face_direction(nb, t, side_b.inward_sign)
                fallbacks += 1
        elif mode == InwardMode.LATERAL:
            assert lat is not None
            direction = lat[i]
            # A points away from B, B points away from A, both squared to the
            # seam and lying in their own tangent planes.
            da = _project_into_plane(-direction, na, t)
            db = _project_into_plane(direction, nb, t)
        else:
            da = in_face_direction(na, t, side_a.inward_sign)
            db = in_face_direction(nb, t, side_b.inward_sign)

        normal_a[i], normal_b[i] = na, nb
        inward_a[i], inward_b[i] = da, db
        g, c = groove_angle(na, da, nb, db, tangent_tolerance)
        groove[i], concavity[i] = g, c
        bisector[i] = joint_bisector(na, nb, da, db)

    geom = SeamGeometry(
        points=points,
        tangents=tangents,
        normal_a=normal_a,
        normal_b=normal_b,
        inward_a=inward_a,
        inward_b=inward_b,
        groove=groove,
        concavity=concavity,
        bisector=bisector,
    )
    if fallbacks:
        geom.notes.append(f"{fallbacks}/{n} stations fell back to trim probing (parallel flanks)")
    return geom


def _project_into_plane(
    direction: np.ndarray, normal: np.ndarray, tangent: np.ndarray
) -> np.ndarray:
    """Component of *direction* lying in the plane normal to *normal* and square to *tangent*."""
    d = np.asarray(direction, dtype=float)
    n = tf.unit(normal)
    t = tf.unit(tangent)
    if n is not None:
        d = d - float(np.dot(d, n)) * n
    if t is not None:
        d = d - float(np.dot(d, t)) * t
    u = tf.unit(d)
    return u if u is not None else np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# validating the concave assumption
# ---------------------------------------------------------------------------


def validate_in_face_directions(
    backend: CadBackend,
    side_a: FaceSide,
    side_b: FaceSide,
    geom: SeamGeometry,
    probe_mm: float = 1.0,
    distance_tol_mm: float = 0.1,
    stations: int = 3,
) -> tuple[bool, str]:
    """Check that each flank really extends the way the geometry says it does.

    :data:`InwardMode.CONCAVE` assumes both faces **continue past** the seam --
    true for a landing contact, where the base plate runs on under the web. It
    is false wherever two members finish flush with each other, and there the
    rule silently invents a joint:

    * the outer edge of an L made by standing a plate on the lip of another:
      the base does not continue outward, so there is no groove there;
    * the end of a web that is flush with the end of its base plate: again no
      groove, just two faces running out together.

    Both show up as confident 90 deg fillets if nobody checks, and both would
    send the torch along a line with nothing to weld. So the assumption is
    tested the only way that settles it: step *probe_mm* along each in-face
    direction and ask whether that point is still on the face.

    Returns ``(ok, reason)``; *reason* names the flank that failed.
    """
    if geom.n == 0:
        return False, "seam has no stations"

    indices = np.unique(np.linspace(0, geom.n - 1, max(1, min(stations, geom.n))).astype(int))

    for label, side, inward in (
        ("A", side_a, geom.inward_a),
        ("B", side_b, geom.inward_b),
    ):
        hits = 0
        for i in indices:
            probe = geom.points[i] + inward[i] * probe_mm
            if point_on_face(
                backend, side.face, side.evaluator, probe, distance_tol=distance_tol_mm
            ):
                hits += 1
        if hits == 0:
            return (
                False,
                f"flank {label} ({side.face_id}) does not extend away from the seam: "
                "members are flush here, so there is no groove",
            )
    return True, ""
