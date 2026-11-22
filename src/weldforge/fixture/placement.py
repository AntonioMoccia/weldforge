"""Where to put the part, and which way up.

This is the decision that most affects how well a weldment comes out, and the
one most often made by eye. Turn a part over and a seam that needed weaving
overhead at 4 mm/s becomes a flat fillet at 8 mm/s with a better profile. Put
it 200 mm further from the robot and half the seams stop being reachable.

What is being optimised
-----------------------
Six things, weighted and reported separately so an operator can see *why* a
placement scored what it did:

===================  =========================================================
Criterion            Why it matters
===================  =========================================================
reachability         A seam the robot cannot reach is a seam welded by hand.
welding position     Downhand (PA/PB) is faster, sounder and needs no weaving.
torch access         The nozzle has to get into the groove without fouling
                     the part itself -- a seam inside a box section is
                     unreachable however good the arm's reach is.
clamp clearance      Clamps hold the part and block the torch; both are real.
robot envelope       Too close and the arm folds; too far and it stretches.
                     Mid-envelope postures have the most freedom.
table fit            On the bench, inside its edges, within its load rating.
===================  =========================================================

Search
------
Orientation first, then position. Candidate orientations are the part's own
**stable resting poses** -- computed from its convex hull, so they are the ways
it will actually sit on a bench without a jig -- plus the axis-aligned
orientations for parts that will be clamped in a fixture anyway. For each
orientation the part is dropped onto the table and its position refined by a
coarse grid followed by a local pattern search.

Hard constraints (off the table, overweight, under the robot) reject a
candidate outright and are reported as violations rather than folded into the
score, because a violated placement is not merely worse -- it is not an option.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..collide import CollisionScene, scene_from_cell, torch_capsules
from ..geom import polyline as pl
from ..geom import transform as tf
from ..kin import RobotModel, solve_analytic
from ..kin.ik import IKOptions
from ..model import Part, PlacementCandidate, Seam, WeldCell, WeldPosition
from ..path.torch import torch_pose, wire_direction
from .cell import cell_robot

logger = logging.getLogger(__name__)

__all__ = [
    "PlacementOptions",
    "PlacementWeights",
    "candidate_orientations",
    "evaluate_placement",
    "find_placement",
    "stable_orientations",
    "group_datums",
    "solve_jig_placement",
]


@dataclass
class PlacementWeights:
    """Relative importance of the criteria. They need not sum to one."""

    reachability: float = 4.0
    position: float = 2.5
    access: float = 2.0
    clamp_clearance: float = 1.0
    envelope: float = 1.0
    # Only meaningful on a jig. A part that does not sit on its locators is
    # not located, and a weld placed from an unlocated part is placed wrong.
    datum_seating: float = 3.0

    @classmethod
    def quality_first(cls) -> PlacementWeights:
        """Favour downhand welding even at some cost in reach."""
        return cls(reachability=3.0, position=5.0, access=2.0)

    @classmethod
    def reach_first(cls) -> PlacementWeights:
        """Get everything reachable; accept awkward positions."""
        return cls(reachability=8.0, position=1.0, access=2.0)


@dataclass
class PlacementOptions:
    """Search effort and the rules a placement has to obey."""

    weights: PlacementWeights = field(default_factory=PlacementWeights)
    # Stations sampled per seam when scoring. Three (ends and middle) catches
    # a seam that runs out of reach partway, which is the common failure.
    samples_per_seam: int = 3
    max_orientations: int = 12
    include_axis_aligned: bool = True
    include_stable_poses: bool = True
    # Coarse grid over the table, then a pattern search around the best cell.
    # Forced odd internally so the table centre is always a candidate.
    grid_steps: int = 5
    refine_iterations: int = 3
    refine_step_mm: float = 60.0
    yaw_steps: int = 8
    # Hard constraints.
    require_on_table: bool = True
    min_distance_from_base_mm: float = 350.0
    reach_margin: float = 0.95
    check_access: bool = True
    access_standoff_mm: float = 60.0
    # How close a part surface must come to a locator to count as seated.
    # 0.5 mm is tighter than any fit-up the fixture will actually deliver, but
    # the check is against the *nominal* model, where it either touches or the
    # part is in the wrong place.
    datum_tolerance_mm: float = 0.5
    # Reject outright a placement that does not seat on every locator. A jig
    # exists to locate the part; a placement that misses a locator is not a
    # worse option, it is not an option.
    require_datum_contact: bool = True
    ik: IKOptions = field(default_factory=IKOptions)
    seed: int = 20251122
    # Score reachability with the closed form only. The numeric fallback costs
    # a hundred times more per call and is spent entirely on targets that are
    # out of reach -- which is most of them during a placement search, and the
    # answer "out of reach" is all the score needs. A chain with no closed
    # form must set this False and accept the cost.
    analytic_only: bool = True


# ---------------------------------------------------------------------------
# candidate orientations
# ---------------------------------------------------------------------------


def stable_orientations(part: Part, limit: int = 8) -> list[np.ndarray]:
    """Rotations under which the part rests stably on a flat table.

    Taken from the convex hull: a rigid body rests on a hull facet whose
    supporting polygon contains the projected centre of mass. These are the
    orientations a part will actually sit in on a bench without being held,
    which is what "place it on the table" means in a shop.

    Returns rotation matrices, most stable first. Falls back to an empty list
    when there is no mesh to work from.
    """
    if part.mesh_vertices is None or part.mesh_faces is None or not len(part.mesh_faces):
        return []
    try:
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray(part.mesh_vertices, dtype=float),
            faces=np.asarray(part.mesh_faces, dtype=np.int64),
            process=False,
        )
        # A tessellated weldment contains a few degenerate triangles, on which
        # trimesh's barycentric test divides by zero. The affected facets
        # simply do not become candidate resting poses, which is correct, so
        # the warning is silenced rather than left to alarm the operator.
        with np.errstate(divide="ignore", invalid="ignore"):
            transforms, probabilities = mesh.compute_stable_poses(n_samples=1, threshold=0.0)
    except Exception as exc:
        logger.debug("stable_orientations unavailable: %s", exc)
        return []

    order = np.argsort(-np.asarray(probabilities))
    out: list[np.ndarray] = []
    for index in order[:limit]:
        out.append(tf.rotation_of(np.asarray(transforms[index], dtype=float)))
    return out


_AXIS_ALIGNED = [
    tf.identity()[:3, :3],
    tf.rot_x(np.pi),
    tf.rot_x(np.pi / 2),
    tf.rot_x(-np.pi / 2),
    tf.rot_y(np.pi / 2),
    tf.rot_y(-np.pi / 2),
]


def candidate_orientations(part: Part, options: PlacementOptions) -> list[np.ndarray]:
    """Orientations to try, de-duplicated."""
    out: list[np.ndarray] = []
    if options.include_stable_poses:
        out.extend(stable_orientations(part, options.max_orientations))
    if options.include_axis_aligned:
        out.extend(_AXIS_ALIGNED)

    unique: list[np.ndarray] = []
    for R in out:
        if not any(np.allclose(R, existing, atol=1e-3) for existing in unique):
            unique.append(tf.orthonormalize(R))
    return unique[: max(1, options.max_orientations)]


# ---------------------------------------------------------------------------
# scoring one placement
# ---------------------------------------------------------------------------


def _drop_onto_table(part: Part, rotation: np.ndarray, table_z_mm: float) -> np.ndarray:
    """Pose that applies *rotation* and rests the part on the table.

    The part's lowest point after rotation is brought to the table surface, and
    its footprint centred on the origin so the position search starts from a
    sensible place.
    """
    lo, hi = part.bbox
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    rotated = corners @ rotation.T
    centre_xy = (rotated[:, :2].min(axis=0) + rotated[:, :2].max(axis=0)) / 2.0
    lift = table_z_mm - float(rotated[:, 2].min())
    return tf.pose(rotation, [-centre_xy[0], -centre_xy[1], lift])


def _datum_seating(
    part: Part,
    pose: np.ndarray,
    cell: WeldCell,
    tolerance_mm: float,
) -> tuple[int, int, int]:
    """``(seated, missed, clashing)`` against the jig's locators.

    Each locator is a point and the direction its face looks in. The part is
    **seated** when its surface touches that point, **missed** when the
    surface is away from it (so the part is not located), and **clashing**
    when the part has pushed through the locator -- which means it cannot be
    loaded into the jig in this orientation at all.

    Measured against the part's mesh in part coordinates, by moving the
    locator into that frame rather than transforming the whole mesh: a
    weldment mesh runs to tens of thousands of triangles and the search
    evaluates hundreds of placements.
    """
    if part.mesh_vertices is None or part.mesh_faces is None or not len(part.mesh_faces):
        return 0, len(cell.fixture.datums), 0

    try:
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray(part.mesh_vertices, dtype=float),
            faces=np.asarray(part.mesh_faces, dtype=np.int64),
            process=False,
        )
        query = trimesh.proximity.ProximityQuery(mesh)
    except Exception as exc:  # pragma: no cover - environment specific
        logger.debug("_datum_seating unavailable: %s", exc)
        return 0, len(cell.fixture.datums), 0

    inverse = tf.inverse(pose)
    points = np.array([tf.apply_point(inverse, position) for position, _ in cell.fixture.datums])
    _, distances, _ = query.on_surface(points)

    seated = missed = clashing = 0
    inside = np.zeros(len(points), dtype=bool)
    if mesh.is_watertight:
        with np.errstate(invalid="ignore"):
            inside = query.signed_distance(points) > tolerance_mm

    for index, distance in enumerate(distances):
        if inside[index]:
            clashing += 1
        elif distance <= tolerance_mm:
            seated += 1
        else:
            missed += 1
    return seated, missed, clashing


def _reach_check(
    arm: RobotModel,
    target: np.ndarray,
    seed: np.ndarray,
    options: PlacementOptions,
) -> np.ndarray | None:
    """Joint values if *target* is reachable, else ``None``.

    Reachability only -- no accuracy refinement, no numeric fallback by
    default. See ``PlacementOptions.analytic_only``.
    """
    solutions = [s for s in solve_analytic(arm, target, options=options.ik) if s.ok]
    if solutions:
        from ..kin import nearest_solution

        return nearest_solution(solutions, seed).joints
    if options.analytic_only:
        return None
    from ..kin import solve_numeric

    numeric = solve_numeric(arm, target, seed, options.ik)
    return numeric.joints if numeric is not None and numeric.ok else None


def _sample_indices(seam: Seam, count: int) -> list[int]:
    if seam.n_points <= count:
        return list(range(seam.n_points))
    return [int(round(v)) for v in np.linspace(0, seam.n_points - 1, count)]


def evaluate_placement(
    part: Part,
    seams: list[Seam],
    cell: WeldCell,
    pose: np.ndarray,
    robot: RobotModel | None = None,
    scene: CollisionScene | None = None,
    options: PlacementOptions | None = None,
) -> PlacementCandidate:
    """Score one placement of the part in the cell."""
    opts = options or PlacementOptions()
    arm = robot if robot is not None else cell_robot(cell)
    candidate = PlacementCandidate(pose=pose.copy())

    # --- hard constraints ---------------------------------------------------
    lo, hi = part.bbox
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    placed = tf.apply_points(pose, corners)

    if opts.require_on_table:
        half = np.asarray(cell.fixture.table_size_mm, dtype=float) / 2.0
        origin = cell.fixture.table_origin
        within = np.all(np.abs(placed[:, :2] - origin[:2]) <= half + 1e-6)
        if not within:
            candidate.violations.append("part overhangs the table")
        if float(placed[:, 2].min()) < cell.fixture.table_z_mm - 1.0:
            candidate.violations.append("part is below the table surface")

    if part.mass_kg > cell.fixture.load_capacity_kg > 0.0:
        candidate.violations.append(
            f"part mass {part.mass_kg:.0f} kg exceeds the fixture rating of "
            f"{cell.fixture.load_capacity_kg:.0f} kg"
        )

    base_xy = tf.translation_of(arm.base)[:2]
    nearest = float(np.min(np.linalg.norm(placed[:, :2] - base_xy, axis=1)))
    if nearest < opts.min_distance_from_base_mm:
        candidate.violations.append(f"part comes within {nearest:.0f} mm of the robot base")

    # --- per-seam evaluation ------------------------------------------------
    reachable: list[str] = []
    unreachable: list[str] = []
    position_counts: dict[str, int] = {}
    downhand = 0
    access_ok = 0
    access_total = 0
    seed = arm.home_joints()

    for seam in seams:
        moved = seam.transformed(pose)
        position = moved.dominant_position
        position_counts[position.value] = position_counts.get(position.value, 0) + 1
        if position.is_downhand:
            downhand += 1

        indices = _sample_indices(moved, opts.samples_per_seam)
        solved = 0
        for i in indices:
            wire = wire_direction(
                moved.bisector[i] if len(moved.bisector) > i else np.array([0.0, 0.0, -1.0]),
                moved.tangents[i] if len(moved.tangents) > i else np.array([1.0, 0.0, 0.0]),
                moved.normal_a[i] if len(moved.normal_a) > i else np.array([0.0, 0.0, 1.0]),
                moved.normal_b[i] if len(moved.normal_b) > i else np.array([0.0, 1.0, 0.0]),
                float(moved.dihedral[i]) if len(moved.dihedral) > i else np.pi / 2,
                work_angle_deg=float(np.rad2deg(moved.dihedral[i] / 2.0))
                if len(moved.dihedral) > i
                else 45.0,
                travel_angle_deg=0.0,
            )
            target = torch_pose(
                moved.points[i],
                wire,
                moved.tangents[i] if len(moved.tangents) > i else np.array([1.0, 0.0, 0.0]),
            )
            solution = _reach_check(arm, target, seed, opts)
            if solution is not None:
                solved += 1
                seed = solution

            if opts.check_access and scene is not None:
                access_total += 1
                contacts = scene.check(torch_capsules(target), clearance_mm=0.0, ignore={"part"})
                # The part itself is excluded: the torch is *meant* to touch
                # the joint. Everything else must be clear.
                if not contacts:
                    access_ok += 1

        if solved == len(indices):
            reachable.append(seam.id)
        else:
            unreachable.append(seam.id)

    candidate.reachable_seams = reachable
    candidate.unreachable_seams = unreachable
    candidate.position_mix = position_counts

    total = max(1, len(seams))
    reach_score = len(reachable) / total
    position_score = downhand / total
    access_score = (access_ok / access_total) if access_total else 1.0

    # --- clamp clearance ----------------------------------------------------
    clamp_score = 1.0
    if cell.fixture.clamps and seams:
        distances = []
        for clamp in cell.fixture.clamps:
            for seam in seams:
                moved_points = tf.apply_points(pose, seam.points)
                # Distance to the seam's **line**, not to its vertices. A
                # straight seam is two points, so a clamp halfway along a
                # 200 mm weld measures 100 mm from the nearer vertex while
                # sitting right on the bead.
                distances.append(
                    pl.distance_to_point(moved_points, clamp.position) - clamp.clearance_mm
                )
        worst = min(distances)
        # Full marks once every seam clears every clamp by 30 mm.
        clamp_score = float(np.clip(worst / 30.0, 0.0, 1.0))
        if worst < 0.0:
            candidate.violations.append(
                f"a seam passes within the clearance zone of a clamp ({worst:.0f} mm)"
            )

    # --- jig locators -------------------------------------------------------
    datum_score = 1.0
    if cell.fixture.datums:
        seated, missed, clashing = _datum_seating(part, pose, cell, opts.datum_tolerance_mm)
        total_datums = seated + missed + clashing
        datum_score = seated / max(1, total_datums)
        if clashing:
            candidate.violations.append(
                f"the part pushes through {clashing} of the jig's {total_datums} "
                "locators: it cannot be loaded in this orientation"
            )
        if missed and opts.require_datum_contact:
            candidate.violations.append(
                f"the part does not seat on {missed} of the jig's {total_datums} "
                "locators, so it is not located"
            )
        candidate.detail["datums_seated"] = float(seated)

    # --- robot envelope -----------------------------------------------------
    centre = tf.apply_point(pose, (lo + hi) / 2.0)
    distance = float(np.linalg.norm(centre[:2] - base_xy))
    best_distance = arm.reach_mm * 0.6 if arm.reach_mm > 0 else 900.0
    envelope_score = float(
        np.clip(1.0 - abs(distance - best_distance) / max(best_distance, 1.0), 0.0, 1.0)
    )
    if arm.reach_mm > 0.0 and distance > arm.reach_mm * opts.reach_margin:
        candidate.violations.append(
            f"part centre is {distance:.0f} mm from the base, beyond "
            f"{opts.reach_margin:.0%} of the {arm.reach_mm:.0f} mm reach"
        )

    weights = opts.weights
    has_datums = bool(cell.fixture.datums)
    candidate.detail.update(
        {
            "reachability": reach_score,
            "position": position_score,
            "access": access_score,
            "clamp_clearance": clamp_score,
            "envelope": envelope_score,
        }
    )
    if has_datums:
        candidate.detail["datum_seating"] = datum_score

    total_weight = (
        weights.reachability
        + weights.position
        + weights.access
        + weights.clamp_clearance
        + weights.envelope
        + (weights.datum_seating if has_datums else 0.0)
    )
    candidate.score = (
        weights.reachability * reach_score
        + weights.position * position_score
        + weights.access * access_score
        + weights.clamp_clearance * clamp_score
        + weights.envelope * envelope_score
        + (weights.datum_seating * datum_score if has_datums else 0.0)
    ) / max(total_weight, 1e-9)

    if candidate.violations:
        # Violated placements are kept for reporting but must never outrank a
        # feasible one.
        candidate.score *= 0.25

    return candidate


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def find_placement(
    part: Part,
    seams: list[Seam],
    cell: WeldCell,
    options: PlacementOptions | None = None,
    robot: RobotModel | None = None,
) -> tuple[PlacementCandidate, list[PlacementCandidate]]:
    """Search for the best placement; returns ``(best, all_evaluated)``.

    Orientation is chosen first, then position is refined for the best few --
    orientation changes which seams are downhand, and no amount of sliding the
    part around the table recovers a seam that is facing the floor.
    """
    import time

    opts = options or PlacementOptions()
    arm = robot if robot is not None else cell_robot(cell)
    scene = scene_from_cell(cell)
    t0 = time.perf_counter()

    evaluated: list[PlacementCandidate] = []
    orientations = candidate_orientations(part, opts)
    logger.info("find_placement: trying %d orientations", len(orientations))

    half = np.asarray(cell.fixture.table_size_mm, dtype=float) / 2.0
    lo, hi = part.bbox
    footprint = (hi - lo)[:2]
    margin = np.maximum(half - footprint / 2.0, 0.0)
    origin = cell.fixture.table_origin

    # --- a jig determines the placement; it is not searched ----------------
    if cell.fixture.datums:
        for rotation in orientations:
            pose = solve_jig_placement(part, cell, rotation)
            if pose is None:
                continue
            evaluated.append(evaluate_placement(part, seams, cell, pose, arm, scene, opts))
        if evaluated:
            evaluated.sort(key=lambda c: (-c.score, len(c.violations)))
            best = evaluated[0]
            logger.info(
                "find_placement: jig, %d orientations seated in %.1fs, best score "
                "%.3f (%d/%d seams reachable)",
                len(evaluated),
                time.perf_counter() - t0,
                best.score,
                len(best.reachable_seams),
                len(seams),
            )
            return best, evaluated

    # --- coarse pass over orientation and a grid of positions --------------
    for rotation in orientations:
        for yaw_index in range(max(1, opts.yaw_steps)):
            yaw = 2.0 * np.pi * yaw_index / max(1, opts.yaw_steps)
            spun = tf.rot_z(yaw) @ rotation
            rest = _drop_onto_table(part, spun, cell.fixture.table_z_mm)

            # Forced odd so the grid always includes the table centre. An even
            # grid samples only the extremes -- which on a bench whose clamps
            # sit near the corners means every candidate is planted against a
            # clamp, and the search never tries the one position most likely
            # to work.
            steps = max(1, opts.grid_steps | 1)
            xs = np.linspace(-margin[0], margin[0], steps) + origin[0]
            ys = np.linspace(-margin[1], margin[1], steps) + origin[1]
            for x in xs:
                for y in ys:
                    pose = rest.copy()
                    pose[0, 3] += x
                    pose[1, 3] += y
                    evaluated.append(evaluate_placement(part, seams, cell, pose, arm, scene, opts))

    if not evaluated:  # pragma: no cover - orientations is never empty
        return PlacementCandidate(), []

    evaluated.sort(key=lambda c: (-c.score, len(c.violations)))
    best = evaluated[0]

    # --- local pattern search around the winner ----------------------------
    step = opts.refine_step_mm
    for _ in range(max(0, opts.refine_iterations)):
        improved = False
        for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
            pose = best.pose.copy()
            pose[0, 3] += dx
            pose[1, 3] += dy
            trial = evaluate_placement(part, seams, cell, pose, arm, scene, opts)
            evaluated.append(trial)
            if trial.score > best.score:
                best, improved = trial, True
        if not improved:
            step /= 2.0

    elapsed = time.perf_counter() - t0
    logger.info(
        "find_placement: %d candidates in %.1fs, best score %.3f "
        "(%d/%d seams reachable, %d downhand)",
        len(evaluated),
        elapsed,
        best.score,
        len(best.reachable_seams),
        len(seams),
        sum(count for key, count in best.position_mix.items() if WeldPosition(key).is_downhand),
    )
    return best, evaluated


# ---------------------------------------------------------------------------
# jig location
# ---------------------------------------------------------------------------


def group_datums(
    datums: list[tuple[np.ndarray, np.ndarray]], tolerance: float = 0.99
) -> list[tuple[np.ndarray, float, int]]:
    """Group locators by the direction they face.

    Returns ``(normal, plane_offset, count)`` per locating plane, most
    locators first. Datums facing the same way lie in one locating plane, and
    it is the plane that constrains the part -- not the individual points.
    """
    groups: list[tuple[np.ndarray, list[float]]] = []
    for position, normal in datums:
        unit = tf.unit(normal)
        if unit is None:
            continue
        offset = float(np.dot(unit, position))
        for existing_normal, offsets in groups:
            if float(np.dot(existing_normal, unit)) >= tolerance:
                offsets.append(offset)
                break
        else:
            groups.append((unit, [offset]))

    out = [(normal, float(np.mean(offsets)), len(offsets)) for normal, offsets in groups]
    out.sort(key=lambda item: -item[2])
    return out


def solve_jig_placement(
    part: Part,
    cell: WeldCell,
    rotation: np.ndarray,
) -> np.ndarray | None:
    """Pose that seats the part on the jig's locators, for a given orientation.

    A jig's placement is **determined, not searched**. That is the whole point
    of a jig: the part is pressed against fixed locators so it sits in the
    same position every time, which is what makes an offline-planned program
    repeatable. Sliding the part around looking for a good score would
    describe a bench, not a jig.

    Solved by the 3-2-1 scheme the locators encode. Locators facing the same
    way form one locating plane; for each plane, in order of how many locators
    it has, the part is translated along that plane's normal until its
    extreme surface point touches. Three planes fix all six degrees of
    freedom, and the translations are independent because the planes are
    mutually perpendicular in any sane jig.

    Returns ``None`` when there is no mesh to seat, or no locators.
    """
    if not cell.fixture.datums:
        return None
    if part.mesh_vertices is None or not len(part.mesh_vertices):
        return None

    rotated = np.asarray(part.mesh_vertices, dtype=float) @ np.asarray(rotation).T
    translation = np.zeros(3)

    for normal, offset, _count in group_datums(cell.fixture.datums):
        # Where the part's surface currently reaches along this normal.
        projected = (rotated + translation) @ normal
        # Move it until the extreme point touches the locating plane. The
        # locator's normal points at the part, so the part's *minimum* along
        # it is the face that lands.
        translation = translation + normal * (offset - float(projected.min()))

    return tf.pose(rotation, translation)
