"""Torch poses along a seam.

Angle conventions, stated because they are the thing people get wrong
---------------------------------------------------------------------
Both angles are defined as a welder would describe them, not as a convenient
rotation of some arbitrary frame:

**Work angle** -- the angle between the wire and the surface of flank A,
measured in the plane square to the weld axis. For a 90 deg fillet, 45 deg is
the symmetric bead. A work angle equal to half the groove angle *is* the joint
bisector, whatever the groove happens to be, which is why the construction
starts from the bisector and rotates away from it by
``groove/2 - work_angle``.

**Travel angle** -- the tilt of the wire in the plane containing the weld
axis. Negative is **push** (forehand): the wire leans into the direction of
travel, which flattens the bead and improves gas coverage. Positive is drag
(backhand): deeper penetration, more convex bead.

Rather than trusting a hand-derived rotation sense, each rotation is applied
both ways and the one that **measures** the requested angle is kept. The
measurement is the definition, so the construction cannot silently come out
mirrored -- which on a fillet means the torch leaning against the wrong plate
and a bead up one leg only.

Offsets
-------
``offset_across`` moves a pass across the groove face, positive towards flank
B; ``offset_normal`` lifts it out of the root along the bisector. Together they
place the layers of a multi-pass weld.
"""

from __future__ import annotations

import logging

import numpy as np

from ..geom import polyline as pl
from ..geom import transform as tf
from ..model import MotionType, PathPoint, Seam, WeaveShape, WeldPass

logger = logging.getLogger(__name__)

__all__ = [
    "across_direction",
    "wire_direction",
    "torch_pose",
    "weave_offset",
    "torch_poses_for_pass",
    "approach_point",
    "retract_point",
    "measure_work_angle",
    "measure_travel_angle",
]


# ---------------------------------------------------------------------------
# frame construction
# ---------------------------------------------------------------------------


def across_direction(bisector: np.ndarray, tangent: np.ndarray, normal_b: np.ndarray) -> np.ndarray:
    """Unit direction across the groove, positive towards flank B.

    Square to both the wire and the weld axis, so stepping along it walks the
    bead from one leg of a fillet to the other -- which is what a multi-pass
    layout needs.
    """
    across = tf.unit(np.cross(bisector, tangent))
    if across is None:
        # Wire parallel to the weld axis: degenerate, fall back to anything
        # square to the tangent.
        fallback = np.eye(3)[int(np.argmin(np.abs(tangent)))]
        across = tf.unit(np.cross(fallback, tangent))
        if across is None:  # pragma: no cover
            return np.array([0.0, 1.0, 0.0])
    # Positive towards flank B: B's material lies opposite its outward normal.
    if float(np.dot(across, -np.asarray(normal_b, dtype=float))) < 0.0:
        across = -across
    return across


def measure_work_angle(wire: np.ndarray, normal_a: np.ndarray) -> float:
    """Angle in degrees between the wire and the plane of flank A."""
    return float(abs(90.0 - np.rad2deg(tf.angle_between(wire, normal_a))))


def measure_travel_angle(wire: np.ndarray, tangent: np.ndarray) -> float:
    """Signed tilt in degrees out of square, positive for drag.

    Zero means the wire is square to the weld axis. Negative means it leans
    into the direction of travel (push).
    """
    along = float(np.dot(tf.unit(wire), tf.unit(tangent)))
    return float(-np.rad2deg(np.arcsin(np.clip(along, -1.0, 1.0))))


def wire_direction(
    bisector: np.ndarray,
    tangent: np.ndarray,
    normal_a: np.ndarray,
    normal_b: np.ndarray,
    groove_rad: float,
    work_angle_deg: float,
    travel_angle_deg: float,
) -> np.ndarray:
    """Wire direction for the requested work and travel angles.

    Built in two stages, each verified against its own definition:

    1. rotate the bisector about the weld axis until the wire measures
       *work_angle_deg* to flank A;
    2. rotate that about the across axis until it measures
       *travel_angle_deg* out of square.

    Both rotations are tried in both senses and the one that measures right is
    kept, so the result cannot come out mirrored.
    """
    b = tf.unit(bisector)
    t = tf.unit(tangent)
    if b is None or t is None:
        return np.array([0.0, 0.0, -1.0])

    # --- work angle -------------------------------------------------------
    half_groove_deg = float(np.rad2deg(groove_rad)) / 2.0
    delta = np.deg2rad(half_groove_deg - work_angle_deg)
    if abs(delta) < 1e-9:
        wire = b
    else:
        candidates = [tf.rot_axis_angle(t, sign * delta) @ b for sign in (1.0, -1.0)]
        wire = min(candidates, key=lambda w: abs(measure_work_angle(w, normal_a) - work_angle_deg))

    # --- travel angle -----------------------------------------------------
    if abs(travel_angle_deg) > 1e-9:
        across = across_direction(wire, t, normal_b)
        target = float(travel_angle_deg)
        candidates = [
            tf.rot_axis_angle(across, sign * np.deg2rad(abs(target))) @ wire for sign in (1.0, -1.0)
        ]
        wire = min(candidates, key=lambda w: abs(measure_travel_angle(w, t) - target))

    unit = tf.unit(wire)
    return unit if unit is not None else b


def torch_pose(
    point: np.ndarray,
    wire: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """TCP pose at a station: **Z** along the wire, **X** along travel.

    Z-along-the-wire is what makes the rest of the system simple: the torch's
    approach direction is the tool frame's own Z, so a stand-off is a
    translation along Z and nothing has to know about welding to apply it.
    """
    return tf.frame_from_z_x(wire, tangent, point)


# ---------------------------------------------------------------------------
# weaving
# ---------------------------------------------------------------------------


def weave_offset(
    shape: WeaveShape,
    amplitude_mm: float,
    frequency_hz: float,
    arc_length_mm: float,
    travel_speed_mm_s: float,
) -> tuple[float, float]:
    """``(across_mm, along_mm)`` offset of the weave at an arc-length station.

    The weave is parameterised by **time**, derived from arc length and travel
    speed, because that is how the controller executes it: a frequency in hertz
    at a given travel speed gives a fixed wavelength along the bead.

    *amplitude_mm* is peak-to-peak, as the machine states it, so the excursion
    either side of centre is half of it.
    """
    if shape is WeaveShape.NONE or amplitude_mm <= 0.0 or frequency_hz <= 0.0:
        return 0.0, 0.0
    if travel_speed_mm_s <= 0.0:
        return 0.0, 0.0

    half = amplitude_mm / 2.0
    phase = 2.0 * np.pi * frequency_hz * (arc_length_mm / travel_speed_mm_s)

    if shape is WeaveShape.ZIGZAG:
        # Triangle wave: constant lateral speed, which is what a zigzag is.
        across = half * (2.0 / np.pi) * np.arcsin(np.sin(phase))
        return float(across), 0.0

    if shape is WeaveShape.TRIANGULAR:
        # Lateral sweep with a forward excursion at the centre, tracing a
        # triangle on the plate: the pattern used vertical-up so the pool is
        # carried on a shelf.
        across = half * (2.0 / np.pi) * np.arcsin(np.sin(phase))
        along = half * 0.5 * max(0.0, np.cos(phase))
        return float(across), float(along)

    if shape is WeaveShape.CIRCULAR:
        return float(half * np.sin(phase)), float(half * np.cos(phase))

    if shape is WeaveShape.TROCHOIDAL:
        # Loops that advance: a circle plus steady progress along the seam.
        return float(half * np.sin(phase)), float(half * (1.0 - np.cos(phase)) * 0.5)

    return 0.0, 0.0


# ---------------------------------------------------------------------------
# a pass
# ---------------------------------------------------------------------------


def torch_poses_for_pass(
    seam: Seam,
    weld_pass: WeldPass,
    station_spacing_mm: float = 0.0,
    bake_weave: bool = False,
    max_stations: int = 2000,
) -> list[PathPoint]:
    """Welding stations for one pass of one seam, in travel order.

    Only arc-on points: approach and retract are added by
    :func:`weldforge.path.build.build_pass_path`.

    Weaving is **left to the controller** by default, and that is a deliberate
    decision rather than a shortcut. Every arc-welding controller executes a
    weave natively -- RAPID has ``weavedata``, KUKA has it in ArcTech, FANUC
    and Motoman have weave schedules -- superimposing the oscillation on the
    programmed line in real time. Interpolating it into the path instead means:

    * a 200 mm seam weaving at 1.8 Hz and 4 mm/s needs a station every 0.2 mm
      to resolve the pattern, which is **over a thousand points** for one
      pass -- more than many controllers will hold for a whole part, and slow
      enough to plan that it dominated the pipeline's run time;
    * the weave becomes fixed at plan time, so a welder cannot trim its
      amplitude at the machine, which is the first thing they adjust;
    * the robot interpolates *between* the baked points, flattening the dwell
      at each side that the weave exists to provide.

    So the path carries the seam centreline and the weave travels with the
    procedure as parameters, which is what the postprocessors emit. Set
    *bake_weave* for a controller that genuinely has no weave function.
    """
    if seam.n_points < 2:
        return []

    points = seam.points
    tangents = seam.tangents
    normal_a = seam.normal_a
    normal_b = seam.normal_b
    bisector = seam.bisector
    groove = seam.dihedral

    spacing = station_spacing_mm
    if bake_weave and weld_pass.weave.active and weld_pass.travel_speed_mm_s > 0.0:
        wavelength = weld_pass.travel_speed_mm_s / max(weld_pass.weave.frequency_hz, 1e-6)
        spacing = min(spacing or wavelength, wavelength / 12.0)
        needed = seam.length_mm / max(spacing, 1e-6)
        if needed > max_stations:
            spacing = seam.length_mm / max_stations
            logger.warning(
                "seam %s: baking the weave would need %.0f stations; spacing "
                "widened to %.2f mm to stay under %d. The weave will be "
                "under-resolved -- prefer letting the controller weave.",
                seam.id,
                needed,
                spacing,
                max_stations,
            )

    if spacing > 0.0:
        resampled = pl.resample_by_spacing(points, spacing)
        cum_source = pl.cumulative_length(points)
        cum_target = pl.cumulative_length(resampled)

        def interp_vectors(values: np.ndarray) -> np.ndarray:
            if not len(values):
                return values
            out = np.empty((len(resampled), 3))
            for axis in range(3):
                out[:, axis] = np.interp(cum_target, cum_source, values[:, axis])
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            return out / norms

        normal_a = interp_vectors(normal_a)
        normal_b = interp_vectors(normal_b)
        bisector = interp_vectors(bisector)
        groove = (
            np.interp(cum_target, cum_source, groove)
            if len(groove)
            else np.full(len(resampled), np.pi / 2)
        )
        points = resampled
        tangents = pl.tangents(points, closed=seam.closed)

    if not len(groove):
        groove = np.full(len(points), np.pi / 2)
    if not len(bisector):
        logger.warning("seam %s has no bisector: torch orientation will be guessed", seam.id)
        bisector = np.tile([0.0, 0.0, -1.0], (len(points), 1))

    arc_length = pl.cumulative_length(points)
    out: list[PathPoint] = []

    for i, point in enumerate(points):
        tangent = tangents[i]
        n_a = normal_a[i] if len(normal_a) > i else np.array([0.0, 0.0, 1.0])
        n_b = normal_b[i] if len(normal_b) > i else np.array([0.0, 1.0, 0.0])
        wire = wire_direction(
            bisector[i],
            tangent,
            n_a,
            n_b,
            float(groove[i]),
            weld_pass.work_angle_deg,
            weld_pass.travel_angle_deg,
        )

        across = across_direction(bisector[i], tangent, n_b)
        offset = (
            across * weld_pass.offset_across_mm
            - np.asarray(bisector[i], dtype=float) * weld_pass.offset_normal_mm
        )

        if bake_weave and weld_pass.weave.active:
            across_weave, along_weave = weave_offset(
                weld_pass.weave.shape,
                weld_pass.weave.amplitude_mm,
                weld_pass.weave.frequency_hz,
                float(arc_length[i]),
                weld_pass.travel_speed_mm_s,
            )
            offset = offset + across * across_weave + tangent * along_weave

        out.append(
            PathPoint(
                pose=torch_pose(point + offset, wire, tangent),
                motion=MotionType.LINEAR,
                speed_mm_s=weld_pass.travel_speed_mm_s,
                weld_on=True,
                seam_id=seam.id,
                pass_index=weld_pass.index,
            )
        )

    return out


# ---------------------------------------------------------------------------
# air moves around a pass
# ---------------------------------------------------------------------------


def approach_point(
    first: PathPoint,
    standoff_mm: float = 40.0,
    speed_mm_s: float = 250.0,
) -> PathPoint:
    """Arc-off point backed off along the torch's own Z, above the start.

    Backing off along Z rather than straight up keeps the torch square to the
    joint as it comes in, so it enters the groove cleanly instead of dragging
    the nozzle along a plate.
    """
    pose = first.pose.copy()
    pose[:3, 3] = pose[:3, 3] - pose[:3, 2] * standoff_mm
    return PathPoint(
        pose=pose,
        motion=MotionType.LINEAR,
        speed_mm_s=speed_mm_s,
        weld_on=False,
        seam_id=first.seam_id,
        pass_index=first.pass_index,
        comment="approach",
    )


def retract_point(
    last: PathPoint,
    standoff_mm: float = 60.0,
    speed_mm_s: float = 250.0,
) -> PathPoint:
    """Retract point after crater fill, backed off along the torch's Z."""
    pose = last.pose.copy()
    pose[:3, 3] = pose[:3, 3] - pose[:3, 2] * standoff_mm
    return PathPoint(
        pose=pose,
        motion=MotionType.LINEAR,
        speed_mm_s=speed_mm_s,
        weld_on=False,
        seam_id=last.seam_id,
        pass_index=last.pass_index,
        comment="retract",
    )
