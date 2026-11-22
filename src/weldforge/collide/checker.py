"""Collision checking for the torch and the arm.

Approach and why
----------------
The robot's own CAD is almost never available to the planner, and a weldment's
mesh runs to tens of thousands of triangles. So the moving side is modelled as
**capsules** -- a cylinder with hemispherical ends swept along each link -- and
the static side as triangle meshes.

Capsules are conservative: a capsule that encloses a link cannot pass through
anything the link would hit. They over-report around a slim wrist, which is the
right direction to be wrong in: a false clash makes a planner find another way
round, while a missed clash breaks a torch.

Distance from a capsule to a mesh is evaluated by sampling the capsule's axis
and querying the mesh's nearest surface point for each sample -- a KD-tree
lookup, no FCL build required, which matters for installing on a shop-floor PC.
The sampling density is derived from the capsule's own radius so the
approximation error stays bounded: with samples no further apart than the
radius, the gap between consecutive spheres can never hide an obstacle larger
than the clearance being tested for.

Everything is in millimetres in **cell coordinates**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..geom import transform as tf
from ..model import Clamp, Obstacle, WeldCell

logger = logging.getLogger(__name__)

__all__ = [
    "Capsule",
    "Contact",
    "CollisionScene",
    "robot_capsules",
    "torch_capsule",
    "torch_capsules",
    "scene_from_cell",
]


@dataclass
class Capsule:
    """A swept sphere: the segment ``a``-``b`` inflated by *radius*."""

    a: np.ndarray
    b: np.ndarray
    radius: float
    name: str = ""

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float).reshape(3)
        self.b = np.asarray(self.b, dtype=float).reshape(3)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    def sample_axis(self, max_step_mm: float | None = None) -> np.ndarray:
        """Points along the axis, spaced no further apart than the radius.

        The spacing bound is what makes the sampled test conservative: between
        two spheres of radius *r* whose centres are *r* apart, the swept volume
        is covered to within a bulge far smaller than the clearances being
        checked.
        """
        step = max_step_mm if max_step_mm is not None else max(self.radius, 1.0)
        count = max(2, int(np.ceil(self.length / max(step, 1e-6))) + 1)
        return np.linspace(self.a, self.b, count)

    def transformed(self, T: np.ndarray) -> Capsule:
        return Capsule(tf.apply_point(T, self.a), tf.apply_point(T, self.b), self.radius, self.name)


@dataclass
class Contact:
    """A clearance violation, or the closest approach when there is none."""

    capsule: str
    obstacle: str
    distance_mm: float
    point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    penetrating: bool = False

    def __str__(self) -> str:
        verb = "penetrates" if self.penetrating else "clears by"
        amount = "" if self.penetrating else f" {self.distance_mm:.1f} mm"
        return f"{self.capsule} {verb} {self.obstacle}{amount}"


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------


class CollisionScene:
    """Static geometry, queried by capsule.

    Meshes are added once and their proximity structures built lazily, because
    a weldment's mesh is expensive to index and a plan may never query some
    obstacles at all.
    """

    def __init__(self, default_clearance_mm: float = 5.0) -> None:
        self.default_clearance_mm = default_clearance_mm
        self._meshes: dict[str, object] = {}
        self._queries: dict[str, object] = {}
        self._planes: dict[str, tuple[np.ndarray, float]] = {}
        self._cylinders: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}

    # -- building ------------------------------------------------------------

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        pose: np.ndarray | None = None,
    ) -> None:
        """Add a triangle mesh, optionally transformed into cell coordinates."""
        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover
            raise ImportError("trimesh is required for collision checking") from exc

        verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
        if pose is not None:
            verts = tf.apply_points(pose, verts)
        mesh = trimesh.Trimesh(
            vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False
        )
        if len(mesh.faces) == 0:
            logger.debug("collision scene: %s has no faces, skipping", name)
            return
        self._meshes[name] = mesh
        self._queries.pop(name, None)

    def add_plane(self, name: str, normal: np.ndarray, offset_mm: float) -> None:
        """A half-space, for the work table: everything below it is solid.

        A table is better modelled as a plane than as a mesh -- it has no
        edges to miss, and the test is one dot product.
        """
        unit = tf.unit(normal)
        if unit is None:
            raise ValueError("plane normal is degenerate")
        self._planes[name] = (unit, float(offset_mm))

    def add_cylinder(
        self,
        name: str,
        base: np.ndarray,
        axis: np.ndarray,
        radius_mm: float,
        height_mm: float | None = None,
    ) -> None:
        """A capped cylinder, for a clamp body and the space it needs around it.

        *height_mm* must be given for anything real. A semi-infinite cylinder
        (``None``) shadows the whole column of air above the clamp, so the
        planner refuses perfectly good torch positions a metre overhead.
        """
        unit = tf.unit(axis)
        if unit is None:
            raise ValueError("cylinder axis is degenerate")
        self._cylinders[name] = (
            np.asarray(base, dtype=float).reshape(3),
            unit,
            float(radius_mm),
            float(height_mm) if height_mm is not None else float("inf"),
        )

    def add_clamp(self, clamp: Clamp) -> None:
        """The clamp body, inflated by the clearance the torch must keep.

        Modelled as a capped cylinder of the clamp's own height along its
        approach direction: that is the volume a torch can actually foul.
        """
        self.add_cylinder(
            f"clamp:{clamp.id}",
            clamp.position,
            clamp.approach,
            clamp.clearance_mm,
            clamp.height_mm,
        )

    def add_obstacle(self, obstacle: Obstacle) -> None:
        if obstacle.has_mesh:
            self.add_mesh(
                f"{obstacle.kind}:{obstacle.id}",
                obstacle.vertices,
                obstacle.faces,
                obstacle.pose,
            )

    @property
    def names(self) -> list[str]:
        return [*self._meshes, *self._planes, *self._cylinders]

    @property
    def is_empty(self) -> bool:
        return not self.names

    # -- querying ------------------------------------------------------------

    def _query(self, name: str):
        if name not in self._queries:
            import trimesh

            self._queries[name] = trimesh.proximity.ProximityQuery(self._meshes[name])
        return self._queries[name]

    def distance_to_points(self, name: str, points: np.ndarray) -> np.ndarray:
        """Distance from each point to one obstacle; negative means inside."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)

        if name in self._planes:
            normal, offset = self._planes[name]
            return points @ normal - offset

        if name in self._cylinders:
            base, axis, radius, height = self._cylinders[name]
            relative = points - base
            along = relative @ axis
            radial = np.linalg.norm(relative - np.outer(along, axis), axis=1)
            # Distance to a capped cylinder: radial outside the wall, axial
            # beyond either cap, combined where both apply (the rim).
            radial_gap = radial - radius
            axial_gap = np.maximum(-along, along - height)
            inside_wall = radial_gap <= 0.0
            inside_caps = axial_gap <= 0.0
            return np.where(
                inside_wall & inside_caps,
                np.maximum(radial_gap, axial_gap),  # inside: nearest face
                np.where(
                    inside_wall,
                    axial_gap,  # off the end
                    np.where(
                        inside_caps,
                        radial_gap,  # beside the wall
                        np.hypot(radial_gap, axial_gap),  # past the rim
                    ),
                ),
            )

        if name in self._meshes:
            query = self._query(name)
            _, distance, _ = query.on_surface(points)
            inside = self._meshes[name].is_watertight and query.signed_distance(points) > 0.0
            return np.where(inside, -distance, distance)

        raise KeyError(f"no obstacle named {name!r} in the scene")

    def closest(self, capsule: Capsule) -> Contact | None:
        """Closest approach of *capsule* to anything in the scene."""
        if self.is_empty:
            return None
        samples = capsule.sample_axis()
        best: Contact | None = None
        for name in self.names:
            distances = self.distance_to_points(name, samples) - capsule.radius
            index = int(np.argmin(distances))
            value = float(distances[index])
            if best is None or value < best.distance_mm:
                best = Contact(
                    capsule=capsule.name or "capsule",
                    obstacle=name,
                    distance_mm=value,
                    point=samples[index],
                    penetrating=value < 0.0,
                )
        return best

    def check(
        self,
        capsules: list[Capsule],
        clearance_mm: float | None = None,
        ignore: set[str] | None = None,
    ) -> list[Contact]:
        """Contacts whose clearance is below the threshold, worst first."""
        threshold = self.default_clearance_mm if clearance_mm is None else clearance_mm
        skip = ignore or set()
        out: list[Contact] = []
        if self.is_empty:
            return out
        for capsule in capsules:
            samples = capsule.sample_axis()
            for name in self.names:
                if name in skip:
                    continue
                distances = self.distance_to_points(name, samples) - capsule.radius
                index = int(np.argmin(distances))
                value = float(distances[index])
                if value < threshold:
                    out.append(
                        Contact(
                            capsule=capsule.name or "capsule",
                            obstacle=name,
                            distance_mm=value,
                            point=samples[index],
                            penetrating=value < 0.0,
                        )
                    )
        out.sort(key=lambda c: c.distance_mm)
        return out

    def clear(self, capsules: list[Capsule], clearance_mm: float | None = None) -> bool:
        return not self.check(capsules, clearance_mm)


# ---------------------------------------------------------------------------
# capsules for the moving parts
# ---------------------------------------------------------------------------


# Nominal link radii for a mid-size welding arm, mm. Deliberately generous:
# a capsule must enclose the link's castings, motors and dress pack, and the
# cost of being generous is a planner that routes a little wider.
DEFAULT_LINK_RADII = (140.0, 120.0, 110.0, 80.0, 70.0, 60.0)


def robot_capsules(
    frames: list[np.ndarray],
    radii: tuple[float, ...] = DEFAULT_LINK_RADII,
    skip_base: int = 1,
    skip_tool: int = 1,
) -> list[Capsule]:
    """Capsules spanning consecutive joint frames.

    *frames* is what :func:`weldforge.kin.fk_all` returns.

    *skip_base* drops the first links: the base column never moves, so
    checking it every station is wasted work.

    *skip_tool* drops the last segment, flange to TCP -- which **is the
    torch**. Modelled as a single fat capsule it necessarily engulfs the weld,
    because its far end sits on the joint by construction, and every station
    reports the torch clashing with the part it is welding.
    :func:`torch_capsules` covers that volume properly instead, so the crude
    version is left out.
    """
    out: list[Capsule] = []
    for i in range(skip_base, len(frames) - 1 - max(0, skip_tool)):
        a = tf.translation_of(frames[i])
        b = tf.translation_of(frames[i + 1])
        if float(np.linalg.norm(b - a)) < 1e-6:
            continue  # coincident frames: a pure rotation, nothing swept
        radius = radii[min(i - 1, len(radii) - 1)] if radii else 100.0
        out.append(Capsule(a, b, radius, name=f"link{i}"))
    return out


def torch_capsule(
    tcp: np.ndarray,
    length_mm: float = 120.0,
    radius_mm: float = 22.0,
    name: str = "torch",
) -> Capsule:
    """Capsule for the gas nozzle and shroud, measured back from the wire tip.

    Only the business end is modelled here: the nozzle is what actually fouls a
    fixture, and it needs a much tighter radius than the arm. The torch body
    behind it is covered by the wrist link capsule.
    """
    tip = tf.translation_of(tcp)
    back = tip - tf.rotation_of(tcp)[:, 2] * length_mm
    return Capsule(back, tip, radius_mm, name=name)


def torch_capsules(
    tcp: np.ndarray,
    nozzle_length_mm: float = 120.0,
    tip_radius_mm: float = 5.0,
    cup_radius_mm: float = 22.0,
    stickout_mm: float = 15.0,
    wire_radius_mm: float = 1.5,
    segments: int = 4,
    name: str = "torch",
) -> list[Capsule]:
    """Faithful torch model: bare wire, then a nozzle widening to the gas cup.

    Why not one capsule. The gas cup's front is a flat ring about 22 mm across,
    but a capsule caps it with a **hemisphere** of that radius, hanging 22 mm
    below the contact tip. Worse, the TCP is at the *wire tip*, a stick-out
    ahead of the cup, so the phantom hemisphere ends up level with the work.
    Approaching a fillet on a plate lying on the bench, it dips straight
    through the table and every seam is declared inaccessible.

    What is actually there, from the work backwards:

    * the **wire** -- about 1.2 mm across, spanning the stick-out;
    * the **nozzle**, whose contact tip is a few millimetres across and which
      reaches the gas cup's full diameter only well back from the work.

    Modelled as a thin wire capsule plus *segments* capsules of increasing
    radius, each taking the frustum's radius at its own midpoint. Accurate to
    about a millimetre over a nozzle's length, and still conservative, while
    leaving the tight clearances a fillet actually needs.
    """
    tip = tf.translation_of(tcp)
    axis = tf.rotation_of(tcp)[:, 2]
    out: list[Capsule] = []

    contact_tip = tip - axis * stickout_mm
    if stickout_mm > 1e-6:
        out.append(Capsule(tip, contact_tip, wire_radius_mm, name=f"{name}:wire"))

    count = max(1, segments)
    step = nozzle_length_mm / count
    for i in range(count):
        near = contact_tip - axis * (step * i)
        far = contact_tip - axis * (step * (i + 1))
        # Radius at the segment's midpoint along the frustum.
        fraction = (i + 0.5) / count
        radius = tip_radius_mm + (cup_radius_mm - tip_radius_mm) * fraction
        out.append(Capsule(near, far, radius, name=f"{name}:nozzle{i}"))
    return out


def scene_from_cell(
    cell: WeldCell,
    part_vertices: np.ndarray | None = None,
    part_faces: np.ndarray | None = None,
    part_pose: np.ndarray | None = None,
    include_table: bool = True,
) -> CollisionScene:
    """Build a collision scene from a cell description and a placed part."""
    scene = CollisionScene()

    if include_table:
        # A finite box, not a half-space. An infinite plane at the table height
        # declares everything below it solid -- including the robot's own
        # column, which stands on the floor well below the bench. Every arm
        # posture then reports "link1 penetrates table".
        size = np.asarray(cell.fixture.table_size_mm, dtype=float)
        thickness = 60.0
        centre = cell.fixture.table_origin.copy()
        centre[2] = cell.fixture.table_z_mm - thickness / 2.0
        try:
            import trimesh

            box = trimesh.creation.box(extents=[size[0], size[1], thickness])
            box.apply_translation(centre)
            scene.add_mesh("table", np.asarray(box.vertices), np.asarray(box.faces, dtype=np.int64))
        except ImportError:  # pragma: no cover
            scene.add_plane("table", np.array([0.0, 0.0, 1.0]), cell.fixture.table_z_mm)

    for clamp in cell.fixture.clamps:
        scene.add_clamp(clamp)

    for obstacle in cell.all_obstacles():
        scene.add_obstacle(obstacle)

    if part_vertices is not None and part_faces is not None and len(part_faces):
        scene.add_mesh("part", part_vertices, part_faces, part_pose)

    logger.debug("collision scene: %s", scene.names)
    return scene
