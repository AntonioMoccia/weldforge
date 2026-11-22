"""Topology extraction and surface evaluation on top of the backend adapter.

Two jobs live here:

1. :func:`extract` walks the shape **once** and builds every relation the rest
   of the pipeline needs -- face ownership by solid, edge-to-face adjacency,
   face-to-edge lists. One traversal instead of OCCT's
   ``MapShapesAndAncestors`` (which cost 6 s on a 900-face weldment) because we
   want all three relations anyway.

2. :class:`FaceEvaluator` answers "what is the outward normal of this face at
   this 3-D point?". Weld seam recognition asks that question at every station
   of every candidate seam, so it is specialised: planes and cylinders are
   solved analytically, everything else falls back to an OCCT point projection.

The outward-normal sign is the part that silently ruins a weld program: the
surface stored on a ``TopAbs_REVERSED`` face has its normal pointing *into* the
solid, and 362 of this test part's 885 faces are reversed. Every normal
returned from this module is already corrected.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geom import transform as tf
from .backend import CadBackend

logger = logging.getLogger(__name__)

__all__ = [
    "RawTopology",
    "extract",
    "FaceEvaluator",
    "sample_edge",
    "estimate_thickness",
    "RayCaster",
    "face_min_extent",
]


# ---------------------------------------------------------------------------
# one-pass traversal
# ---------------------------------------------------------------------------


@dataclass
class RawTopology:
    """Shapes and relations from a single traversal of the loaded shape.

    Keys are the backend's stable shape keys (see
    :meth:`CadBackend.shape_key`), never Python ``id()``.
    """

    shape: Any
    solids: dict[int, Any] = field(default_factory=dict)
    faces: dict[int, Any] = field(default_factory=dict)
    edges: dict[int, Any] = field(default_factory=dict)
    face_solid: dict[int, int] = field(default_factory=dict)
    edge_faces: dict[int, list[int]] = field(default_factory=dict)
    face_edges: dict[int, list[int]] = field(default_factory=dict)
    # How many times each edge was met during the traversal, counting repeats.
    # Distinguishes the two 1-face cases that must be treated differently:
    #   occurrences == 1 -> a genuinely free edge (open shell / sheet body)
    #   occurrences == 2 -> the seam edge where a periodic face (a hole, a
    #                       full cylinder) closes on itself. Never a weld.
    edge_occurrences: dict[int, int] = field(default_factory=dict)
    # Stable, human-readable ids assigned in traversal order.
    solid_ids: dict[int, str] = field(default_factory=dict)
    face_ids: dict[int, str] = field(default_factory=dict)
    edge_ids: dict[int, str] = field(default_factory=dict)

    def edge_kind(self, edge_key: int) -> str:
        """``'manifold'`` | ``'periodic_seam'`` | ``'free'`` | ``'non_manifold'``."""
        n_faces = len(self.edge_faces.get(edge_key, ()))
        n_occ = self.edge_occurrences.get(edge_key, n_faces)
        if n_faces >= 3:
            return "non_manifold"
        if n_faces == 2:
            return "manifold"
        if n_faces == 1 and n_occ >= 2:
            return "periodic_seam"
        return "free"

    def edge_solids(self, edge_key: int) -> list[int]:
        """Solids touching an edge, via its adjacent faces."""
        out: list[int] = []
        for fk in self.edge_faces.get(edge_key, ()):
            sk = self.face_solid.get(fk)
            if sk is not None and sk not in out:
                out.append(sk)
        return out

    @property
    def counts(self) -> dict[str, int]:
        kinds: dict[str, int] = {}
        for ek in self.edges:
            k = self.edge_kind(ek)
            kinds[k] = kinds.get(k, 0) + 1
        return {
            "solids": len(self.solids),
            "faces": len(self.faces),
            "edges": len(self.edges),
            **{f"edges_{k}": v for k, v in sorted(kinds.items())},
        }


def extract(backend: CadBackend, shape: Any) -> RawTopology:
    """Walk *shape* and build the full topology relation set."""
    topo = RawTopology(shape=shape)

    solids = list(backend.explore_unique(shape, "solid"))
    if not solids:
        # A shell-only or face-only import (common for sheet-metal exports):
        # treat the whole shape as one pseudo-solid so ownership still works.
        logger.info("extract: no solids found, treating the shape as a single body")
        solids = [shape]

    for si, solid in enumerate(solids):
        sk = backend.shape_key(solid)
        topo.solids[sk] = solid
        topo.solid_ids[sk] = f"solid_{si}"

        for face in backend.explore_unique(solid, "face"):
            fk = backend.shape_key(face)
            if fk not in topo.faces:
                topo.faces[fk] = face
                topo.face_ids[fk] = f"face_{len(topo.face_ids)}"
                topo.face_edges[fk] = []
            # A face shared by two solids (a properly unioned weldment) keeps
            # its first owner; the shared relation is recovered from edge_faces.
            topo.face_solid.setdefault(fk, sk)

            # Deliberately NOT explore_unique here: a periodic face (a hole, a
            # full cylinder) meets its own closing seam edge twice, and that
            # repeat is the only thing distinguishing a periodic seam from a
            # genuinely free edge. Deduping it away mislabels every hole.
            for edge in backend.explore(face, "edge"):
                ek = backend.shape_key(edge)
                if ek not in topo.edges:
                    topo.edges[ek] = edge
                    topo.edge_ids[ek] = f"edge_{len(topo.edge_ids)}"
                    topo.edge_faces[ek] = []
                    topo.edge_occurrences[ek] = 0
                topo.edge_occurrences[ek] += 1
                if fk not in topo.edge_faces[ek]:
                    topo.edge_faces[ek].append(fk)
                if ek not in topo.face_edges[fk]:
                    topo.face_edges[fk].append(ek)

    logger.info("extract: %s", topo.counts)
    return topo


# ---------------------------------------------------------------------------
# edge sampling
# ---------------------------------------------------------------------------


def sample_edge(
    backend: CadBackend,
    edge: Any,
    spacing_mm: float = 2.0,
    deflection_mm: float = 0.05,
    max_points: int = 2000,
) -> tuple[np.ndarray, str, float]:
    """Sample an edge into a polyline; returns ``(points, curve_type, radius_mm)``.

    Sampling is by **arc length**, not curve parameter: a B-spline's parameter
    is not proportional to length, so uniform-parameter sampling bunches points
    where the knots are dense and leaves gaps elsewhere -- visible as speed
    ripple on the welded bead.

    A straight line gets exactly two points. Curves get
    ``ceil(length / spacing) + 1`` stations, with *deflection_mm* as an upper
    bound on how far the chord may stray from the true curve (so a tight radius
    is sampled finer than *spacing* alone would ask for).
    """
    adaptor = backend.curve_adaptor(edge)
    curve_type = backend.curve_type(adaptor)

    abscissa_cls = backend.sym("GCPnts", "GCPnts_AbscissaPoint")
    length_fn = getattr(abscissa_cls, "Length_s", None) or abscissa_cls.Length
    try:
        length = float(length_fn(adaptor))
    except Exception:  # pragma: no cover - degenerate edge
        length = 0.0

    radius = 0.0
    if curve_type == "circle":
        try:
            radius = float(adaptor.Circle().Radius())
        except Exception:  # pragma: no cover
            radius = 0.0

    def point_at(u: float) -> np.ndarray:
        p = adaptor.Value(u)
        return np.array([p.X(), p.Y(), p.Z()], dtype=float)

    u0, u1 = float(adaptor.FirstParameter()), float(adaptor.LastParameter())

    if length <= 1e-9:
        return np.array([point_at(u0)]), curve_type, radius

    if curve_type == "line":
        return np.array([point_at(u0), point_at(u1)]), curve_type, radius

    n_by_spacing = int(math.ceil(length / max(spacing_mm, 1e-6))) + 1
    n_points = max(3, min(n_by_spacing, max_points))

    # Refine for curvature: a circle of radius r sampled with chord angle
    # theta has deflection r * (1 - cos(theta / 2)).
    if radius > 1e-6 and deflection_mm > 0.0:
        ratio = max(0.0, min(1.0, 1.0 - deflection_mm / radius))
        theta = 2.0 * math.acos(ratio) if ratio < 1.0 else math.pi
        if theta > 1e-9:
            arc_angle = length / radius
            n_curv = int(math.ceil(arc_angle / theta)) + 1
            n_points = max(n_points, min(n_curv, max_points))

    sampler_cls = backend.sym("GCPnts", "GCPnts_UniformAbscissa")
    try:
        sampler = sampler_cls(adaptor, int(n_points))
        if sampler.IsDone() and sampler.NbPoints() >= 2:
            params = [sampler.Parameter(i + 1) for i in range(sampler.NbPoints())]
            return np.array([point_at(u) for u in params]), curve_type, radius
    except Exception as exc:  # pragma: no cover - binding/geometry specific
        logger.debug("sample_edge: UniformAbscissa failed (%s), falling back", exc)

    # Fallback: uniform in parameter. Less faithful, but never fails.
    params = np.linspace(u0, u1, n_points)
    return np.array([point_at(u) for u in params]), curve_type, radius


# ---------------------------------------------------------------------------
# surface evaluation
# ---------------------------------------------------------------------------


@dataclass
class FaceEvaluator:
    """Outward-normal queries on one CAD face.

    Construct once per face and reuse: the adaptor and the analytic surface
    parameters are cached, which matters because seam recognition queries
    thousands of points per part.
    """

    backend: CadBackend
    face: Any
    face_type: str = ""
    _adaptor: Any = field(default=None, repr=False)
    _flip: float = 1.0
    # plane
    _plane_normal: np.ndarray | None = field(default=None, repr=False)
    _plane_point: np.ndarray | None = field(default=None, repr=False)
    # cylinder
    _axis_dir: np.ndarray | None = field(default=None, repr=False)
    _axis_point: np.ndarray | None = field(default=None, repr=False)
    _radius: float = 0.0
    _projector: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        b = self.backend
        self._adaptor = b.surface_adaptor(self.face)
        self.face_type = self.face_type or b.surface_type(self._adaptor)
        # A REVERSED face stores a surface whose natural normal points inward.
        self._flip = -1.0 if b.is_reversed(self.face) else 1.0

        if self.face_type == "planar":
            try:
                pln = self._adaptor.Plane()
                d = pln.Axis().Direction()
                loc = pln.Axis().Location()
                self._plane_normal = np.array([d.X(), d.Y(), d.Z()], dtype=float)
                self._plane_point = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
            except Exception:  # pragma: no cover
                self.face_type = "other"
        elif self.face_type == "cylindrical":
            try:
                cyl = self._adaptor.Cylinder()
                ax = cyl.Axis()
                d, loc = ax.Direction(), ax.Location()
                self._axis_dir = np.array([d.X(), d.Y(), d.Z()], dtype=float)
                self._axis_point = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
                self._radius = float(cyl.Radius())
            except Exception:  # pragma: no cover
                self.face_type = "other"

    # -- properties ----------------------------------------------------------

    @property
    def is_planar(self) -> bool:
        return self.face_type == "planar"

    @property
    def radius_mm(self) -> float:
        return self._radius

    @property
    def plane_offset_mm(self) -> float:
        """Signed distance of the plane from the origin, along the outward normal."""
        if self._plane_normal is None or self._plane_point is None:
            return 0.0
        n = self._plane_normal * self._flip
        return float(np.dot(n, self._plane_point))

    # -- queries -------------------------------------------------------------

    def uv_bounds(self) -> tuple[float, float, float, float]:
        a = self._adaptor
        return (
            float(a.FirstUParameter()),
            float(a.LastUParameter()),
            float(a.FirstVParameter()),
            float(a.LastVParameter()),
        )

    def normal_at_uv(self, u: float, v: float) -> np.ndarray:
        """Outward unit normal at surface parameters ``(u, v)``."""
        gp = self.backend.module("gp")
        p, du, dv = gp.gp_Pnt(), gp.gp_Vec(), gp.gp_Vec()
        self._adaptor.D1(u, v, p, du, dv)
        n = np.cross(
            [du.X(), du.Y(), du.Z()],
            [dv.X(), dv.Y(), dv.Z()],
        )
        un = tf.unit(n)
        if un is None:
            # Degenerate parametrisation (pole of a sphere/cone): nudge and retry.
            u0, u1, v0, v1 = self.uv_bounds()
            du_eps = (u1 - u0) * 1e-4 or 1e-6
            dv_eps = (v1 - v0) * 1e-4 or 1e-6
            self._adaptor.D1(u + du_eps, v + dv_eps, p, du, dv)
            n = np.cross([du.X(), du.Y(), du.Z()], [dv.X(), dv.Y(), dv.Z()])
            un = tf.unit(n)
            if un is None:
                return np.array([0.0, 0.0, 1.0])
        return un * self._flip

    def point_at_uv(self, u: float, v: float) -> np.ndarray:
        p = self._adaptor.Value(u, v)
        return np.array([p.X(), p.Y(), p.Z()], dtype=float)

    def center(self) -> np.ndarray:
        """Point at the middle of the parametric domain."""
        u0, u1, v0, v1 = self.uv_bounds()
        return self.point_at_uv((u0 + u1) / 2.0, (v0 + v1) / 2.0)

    def normal_at_center(self) -> np.ndarray:
        u0, u1, v0, v1 = self.uv_bounds()
        return self.normal_at_uv((u0 + u1) / 2.0, (v0 + v1) / 2.0)

    def sample_on_face(self, max_points: int = 8) -> np.ndarray:
        """Points guaranteed to lie **inside the trimmed face**, as ``(N, 3)``.

        The parametric midpoint is not good enough: for a trimmed face -- a
        plate with a hole, an L-shaped flange, an annulus -- it can fall in the
        hole or entirely outside the boundary. Triangulation vertices and
        triangle barycentres are always on the real face, which is what any
        accessibility or containment test needs.

        Falls back to the parametric centre if the face carries no
        triangulation (tessellate the shape first to populate it).
        """
        b = self.backend
        try:
            brep_tool = b.sym("BRep", "BRep_Tool")
            triangulation_fn = (
                getattr(brep_tool, "Triangulation_s", None) or brep_tool.Triangulation
            )
            loc_cls = b.sym("TopLoc", "TopLoc_Location")
            loc = loc_cls()
            tri = triangulation_fn(self.face, loc)
            if tri is not None and tri.NbTriangles() > 0:
                trsf = loc.Transformation()
                n_tri = tri.NbTriangles()
                step = max(1, n_tri // max(1, max_points))
                out: list[np.ndarray] = []
                for i in range(1, n_tri + 1, step):
                    a, bb, c = tri.Triangle(i).Get()
                    pts = []
                    for idx in (a, bb, c):
                        p = tri.Node(idx).Transformed(trsf)
                        pts.append((p.X(), p.Y(), p.Z()))
                    out.append(np.mean(pts, axis=0))
                    if len(out) >= max_points:
                        break
                if out:
                    return np.array(out, dtype=float)
        except Exception as exc:  # pragma: no cover - geometry specific
            logger.debug("sample_on_face failed: %s", exc)
        return self.center().reshape(1, 3)

    def triangulation_nodes(self) -> np.ndarray:
        """All triangulation vertices of the face, as ``(N, 3)``.

        Distinct from :meth:`sample_on_face`, which returns triangle
        barycentres for containment probing: a planar rectangle meshes into two
        triangles, so barycentres give only two points -- not enough to measure
        the face's shape. Its four corner **nodes** describe it exactly.
        """
        b = self.backend
        try:
            brep_tool = b.sym("BRep", "BRep_Tool")
            triangulation_fn = (
                getattr(brep_tool, "Triangulation_s", None) or brep_tool.Triangulation
            )
            loc = b.sym("TopLoc", "TopLoc_Location")()
            tri = triangulation_fn(self.face, loc)
            if tri is None or tri.NbNodes() == 0:
                return np.zeros((0, 3))
            trsf = loc.Transformation()
            out = np.empty((tri.NbNodes(), 3))
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                out[i - 1] = (p.X(), p.Y(), p.Z())
            return out
        except Exception as exc:  # pragma: no cover - geometry specific
            logger.debug("triangulation_nodes failed: %s", exc)
            return np.zeros((0, 3))

    def boundary_points(self, spacing_mm: float = 2.0) -> np.ndarray:
        """Points sampled along the face's boundary edges, as ``(N, 3)``.

        Fallback for a face with no triangulation: the boundary still describes
        the face's extent exactly.
        """
        chunks: list[np.ndarray] = []
        for edge in self.backend.explore_unique(self.face, "edge"):
            points, _, _ = sample_edge(self.backend, edge, spacing_mm)
            if len(points):
                chunks.append(points)
        return np.vstack(chunks) if chunks else np.zeros((0, 3))

    def normal_at_point(self, point: np.ndarray) -> np.ndarray:
        """Outward unit normal at the 3-D *point*, assumed on (or near) the face.

        Planes and cylinders are closed-form. Anything else is projected onto
        the surface with OCCT and evaluated there; if projection fails the
        parametric centre normal is returned, which is still a usable
        approximation for a gently curved face.
        """
        p = np.asarray(point, dtype=float).reshape(3)

        if self._plane_normal is not None:
            n = tf.unit(self._plane_normal)
            return (n if n is not None else np.array([0.0, 0.0, 1.0])) * self._flip

        if self._axis_dir is not None and self._axis_point is not None:
            axis = tf.unit(self._axis_dir)
            if axis is not None:
                rel = p - self._axis_point
                radial = rel - np.dot(rel, axis) * axis
                ur = tf.unit(radial)
                if ur is not None:
                    return ur * self._flip

        uv = self._project(p)
        if uv is None:
            return self.normal_at_center()
        return self.normal_at_uv(*uv)

    def distance_to_point(self, point: np.ndarray) -> float:
        """Shortest distance from *point* to the face's surface, in mm."""
        p = np.asarray(point, dtype=float).reshape(3)
        if self._plane_normal is not None and self._plane_point is not None:
            n = tf.unit(self._plane_normal)
            if n is not None:
                return abs(float(np.dot(p - self._plane_point, n)))
        if self._axis_dir is not None and self._axis_point is not None:
            axis = tf.unit(self._axis_dir)
            if axis is not None:
                rel = p - self._axis_point
                radial = rel - np.dot(rel, axis) * axis
                return abs(float(np.linalg.norm(radial)) - self._radius)
        uv = self._project(p)
        if uv is None:
            return float(np.linalg.norm(p - self.center()))
        return float(np.linalg.norm(p - self.point_at_uv(*uv)))

    def _project(self, point: np.ndarray) -> tuple[float, float] | None:
        """Project a 3-D point onto the surface; returns ``(u, v)`` or ``None``."""
        try:
            if self._projector is None:
                brep_tool = self.backend.sym("BRep", "BRep_Tool")
                surface_fn = getattr(brep_tool, "Surface_s", None) or brep_tool.Surface
                surface = surface_fn(self.face)
                sa_cls = self.backend.sym("ShapeAnalysis", "ShapeAnalysis_Surface")
                self._projector = sa_cls(surface)
            gp = self.backend.module("gp")
            p = gp.gp_Pnt(*[float(x) for x in point])
            uv = self._projector.ValueOfUV(p, 1e-4)
            return float(uv.X()), float(uv.Y())
        except Exception as exc:  # pragma: no cover - geometry specific
            logger.debug("FaceEvaluator._project failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# plate thickness
# ---------------------------------------------------------------------------


def estimate_thickness(
    backend: CadBackend,
    solid: Any,
    evaluators: list[FaceEvaluator] | None = None,
    max_thickness_mm: float = 60.0,
) -> float:
    """Estimate the plate thickness of a solid, in mm; 0 if it is not plate-like.

    Thickness drives the whole WPS -- fillet leg, number of passes, heat input
    limit, preheat -- so it is worth getting from the geometry rather than
    asking the operator.

    Method: find pairs of planar faces whose outward normals are antiparallel
    and whose areas are comparable, then take the distance between their
    planes. The smallest such distance that is consistent with the solid's
    bounding box is the sheet thickness. Falls back to the smallest bounding-box
    dimension when no parallel pair exists (a turned or cast part).
    """
    if evaluators is None:
        evaluators = [FaceEvaluator(backend, f) for f in backend.explore_unique(solid, "face")]

    planar = [e for e in evaluators if e.is_planar]
    lo, hi = backend.bbox(solid)
    bbox_min_dim = float(np.min(hi - lo)) if np.all(np.isfinite(hi - lo)) else 0.0

    best: float | None = None
    if len(planar) >= 2:
        areas: list[float] = []
        for ev in planar:
            area, _ = backend.surface_properties(ev.face)
            areas.append(area)

        for i in range(len(planar)):
            for j in range(i + 1, len(planar)):
                a, b = planar[i], planar[j]
                na = a.normal_at_center()
                nb = b.normal_at_center()
                # Antiparallel within 2 deg -> the two skins of a plate.
                if float(np.dot(na, nb)) > -0.999:
                    continue
                # Comparable area: a plate's two skins match within 4x; a chamfer
                # or a small step would not.
                area_a, area_b = areas[i], areas[j]
                if min(area_a, area_b) <= 0.0:
                    continue
                if max(area_a, area_b) / min(area_a, area_b) > 4.0:
                    continue
                d = abs(a.distance_to_point(b.center()))
                if 0.2 <= d <= max_thickness_mm and (best is None or d < best):
                    best = d

    # Guard against picking up a narrow rib: the thickness must be the
    # smallest dimension of the body, not some interior offset.
    if best is not None and (bbox_min_dim <= 0.0 or best <= bbox_min_dim * 1.15):
        return round(best, 3)

    if 0.2 <= bbox_min_dim <= max_thickness_mm:
        return round(bbox_min_dim, 3)
    return 0.0


# ---------------------------------------------------------------------------
# local thickness by ray casting
# ---------------------------------------------------------------------------


class RayCaster:
    """Ray-shape intersection against one solid, with the shape loaded once.

    Used to measure **local** material thickness at a seam station. The
    per-solid thickness estimate from parallel face pairs is a property of the
    whole body, which is wrong in two common cases:

    * a weldment fused into one solid has no single thickness;
    * a body with a flange, a boss or a varying section has several.

    Thickness at the joint is what sets the fillet leg, the pass count and the
    heat-input limit, so it is measured where the weld actually is: fire a ray
    from the seam point into the material and take the distance to the first
    exit.
    """

    def __init__(self, backend: CadBackend, shape: Any, tolerance: float = 1e-6) -> None:
        self.backend = backend
        self._intersector = backend.sym("IntCurvesFace", "IntCurvesFace_ShapeIntersector")()
        self._intersector.Load(shape, tolerance)

    def hits(
        self, origin: np.ndarray, direction: np.ndarray, max_distance: float = 1e4
    ) -> list[float]:
        """Sorted distances from *origin* at which the ray crosses the shape."""
        d = tf.unit(direction)
        if d is None:
            return []
        gp = self.backend.module("gp")
        line = gp.gp_Lin(
            gp.gp_Pnt(*[float(x) for x in origin]),
            gp.gp_Dir(*[float(x) for x in d]),
        )
        try:
            self._intersector.Perform(line, 0.0, float(max_distance))
        except Exception as exc:  # pragma: no cover - geometry specific
            logger.debug("RayCaster.hits failed: %s", exc)
            return []
        if not self._intersector.IsDone():
            return []
        return sorted(
            float(self._intersector.WParameter(i + 1)) for i in range(self._intersector.NbPnt())
        )

    def thickness_along(
        self,
        point: np.ndarray,
        direction: np.ndarray,
        start_offset_mm: float = 0.01,
        max_thickness_mm: float = 300.0,
    ) -> float:
        """Material thickness from *point* going along *direction*, in mm.

        The ray starts a hair inside the material so the surface the point sits
        on is not itself counted as the exit; the first crossing after that is
        the far skin. Returns 0.0 when the ray never leaves (an open shell) or
        when the result exceeds *max_thickness_mm*.
        """
        d = tf.unit(direction)
        if d is None:
            return 0.0
        origin = np.asarray(point, dtype=float).reshape(3) + d * start_offset_mm
        for distance in self.hits(origin, d, max_thickness_mm):
            if distance > 1e-6:
                total = distance + start_offset_mm
                return round(total, 3) if total <= max_thickness_mm else 0.0
        return 0.0

    MIN_WELDABLE_GAUGE_MM = 0.5
    """Thinner than this is not a weldable member in a steel weldment.

    A gauge reading below it means the probe missed -- it started on an edge of
    the section and the ray left again almost at once -- not that the metal is
    really half a millimetre thick.
    """

    def gauge(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        face_min_extent_mm: float = 0.0,
        max_thickness_mm: float = 300.0,
        floor_mm: float | None = None,
    ) -> float:
        """Member gauge at a seam station, in mm -- the thickness a WPS needs.

        Two bounds are combined, because neither alone is right:

        **The ray along the inward normal.** Correct when the flank face is one
        of the member's *skins*: firing into a plate from its broad face exits
        through the opposite skin, which is the gauge. But fire from a plate's
        *edge* face and the ray runs the whole width of the plate instead -- on
        the lap specimen that is 100 mm for an 8 mm plate.

        **The flank face's own smallest extent.** Correct in exactly the case
        the ray gets wrong: a plate's edge face is ``length x thickness``, so
        its smaller dimension *is* the gauge. But it is wrong for a skin, where
        the smallest extent is the plate's width.

        Each bound is an over-estimate in the case the other handles, so the
        **minimum** of the two is right in both. Taking the minimum also fails
        safe: an under-estimated thickness yields a smaller fillet leg and less
        heat input, which is the conservative direction for a procedure a human
        will review.

        Known limitation: a small pad or boss face on a thick body reports the
        pad's size rather than the body's thickness. Conservative, and such a
        face is not a load-bearing weld flank.
        """
        floor = self.MIN_WELDABLE_GAUGE_MM if floor_mm is None else floor_mm
        by_ray = self.thickness_along(
            point, -np.asarray(normal, dtype=float), max_thickness_mm=max_thickness_mm
        )
        candidates = [v for v in (by_ray, face_min_extent_mm) if v >= floor]
        if not candidates:
            return 0.0
        return round(min(candidates), 3)


def face_min_extent(backend: CadBackend, face: Any, floor_mm: float = 0.5) -> float:
    """Narrow width of a face, in mm -- an upper bound on its member's gauge.

    Measured by principal-component spans of the face's own surface samples,
    not by its axis-aligned bounding box. The bounding box is only right when
    the face happens to line up with the world axes: tilt a plate 30 deg and
    all three box extents become non-zero, so the "flat" direction is no longer
    identifiable and the smallest extent can come out as a fraction of a
    millimetre. On the customer weldment that produced flank thicknesses of
    0.2 mm, which would drive a nonsense welding procedure.

    PCA is orientation-independent: the smallest span is the surface's own
    flatness (zero for a plane) and the **second** smallest is the width that
    bounds the gauge.

    Returns 0.0 when the face carries no triangulation to sample, in which case
    the caller falls back to ray casting alone.
    """
    evaluator = FaceEvaluator(backend, face)
    samples = evaluator.triangulation_nodes()
    if len(samples) < 3:
        samples = evaluator.boundary_points()
    if len(samples) < 3:
        lo, hi = backend.bbox(face)
        extents = sorted(v for v in (hi - lo) if v > floor_mm)
        return round(float(extents[0]), 3) if extents else 0.0

    centred = samples - samples.mean(axis=0)
    # Spans along the principal axes, largest first.
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    spans = sorted((float(np.ptp(centred @ axis)) for axis in axes), reverse=True)
    width = spans[1] if len(spans) >= 2 else spans[0]
    return round(width, 3) if width > floor_mm else 0.0
