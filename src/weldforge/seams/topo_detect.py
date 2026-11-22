"""Seam detection from shared topology.

Applies when the joint is visible in the B-rep itself: a single bent or
machined body with an inside corner, or a weldment that was booleaned into one
solid before export. Then the seam is a manifold edge shared by two faces, and
the groove angle follows from the faces' normals.

The flanks genuinely **end** at such an edge, so the in-face directions come
from probing the faces' 2-D trim (:data:`InwardMode.TRIM`).

What gets rejected, and why it matters
--------------------------------------
Most edges of a real part are not welds. Left unfiltered, a 900-face weldment
yields hundreds of phantom seams and the operator stops trusting the tool.
Rejected here:

* **convex edges of a single body** -- the outside of the part;
* **tangent edges** -- a face split by a construction edge;
* **blend faces** -- a chamfer or a fillet radius presents two plausible
  grooves either side of it, both machined features;
* **periodic seams** -- where a hole's cylindrical wall closes on itself;
* anything shorter than one weld pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cad.backend import CadBackend
from ..cad.topology import FaceEvaluator, RawTopology, RayCaster, face_min_extent
from ..geom import polyline as pl
from ..model import Part, Seam
from .classify import ContactKind, JointContext, classify
from .dihedral import FaceSide, InwardMode, compute_seam_geometry, resolve_inward_sign

logger = logging.getLogger(__name__)

__all__ = ["TopoDetectOptions", "detect_topological_seams", "is_blend_face"]


@dataclass
class TopoDetectOptions:
    min_seam_length_mm: float = 5.0
    # A cylindrical/toroidal flank no wider than this is a machined blend,
    # not a joint. 12 mm covers the usual chamfer and fillet radii on
    # structural steel without swallowing a real pipe-to-plate joint.
    blend_max_radius_mm: float = 12.0
    blend_max_width_mm: float = 12.0
    min_confidence: float = 0.25
    # Measure each flank's gauge at the seam by ray casting. Essential for a
    # fused weldment, where the body as a whole has no single thickness.
    local_thickness: bool = True
    # Convex edges of a single body are never welds, but convex edges between
    # two bodies of a fused weldment are outside corner joints.
    include_convex_cross_solid: bool = True


def is_blend_face(
    backend: CadBackend,
    face: Any,
    evaluator: FaceEvaluator,
    max_radius_mm: float,
    max_width_mm: float,
) -> bool:
    """True if the face looks like a chamfer or fillet-radius blend.

    A blend of radius *r* running *L* long has a bounding box of roughly
    ``r x r x L``, so the smallest box dimension gives its width directly --
    cheaper and more reliable than reasoning about area.
    """
    if evaluator.face_type not in ("cylindrical", "toroidal", "conical", "spherical"):
        return False
    if evaluator.radius_mm > max_radius_mm:
        return False
    lo, hi = backend.bbox(face)
    return float(np.min(hi - lo)) <= max_width_mm


def detect_topological_seams(
    part: Part,
    options: TopoDetectOptions | None = None,
) -> list[Seam]:
    """Find seams along manifold edges of *part*.

    Requires a part loaded through :func:`weldforge.cad.load_brep_part`, which
    stashes the live OCCT handles the groove maths needs.
    """
    opts = options or TopoDetectOptions()
    backend: CadBackend | None = part.brep.backend
    topo: RawTopology | None = part.brep.topology
    evaluators: dict[int, FaceEvaluator] | None = part.brep.evaluators or None

    if topo is None or evaluators is None:
        logger.info("detect_topological_seams: part has no B-rep topology, skipping")
        return []
    if backend is None:
        from ..cad.backend import get_backend

        backend = get_backend()

    thickness_by_solid = {s.id: s.thickness_mm for s in part.solids}
    casters: dict[str, RayCaster] = {}

    def caster_for(solid_id: str) -> RayCaster | None:
        if not opts.local_thickness:
            return None
        if solid_id not in casters:
            key = next((k for k, v in topo.solid_ids.items() if v == solid_id), None)
            if key is None:
                return None
            casters[solid_id] = RayCaster(backend, topo.solids[key])
        return casters.get(solid_id)

    seams: list[Seam] = []
    stats = {"edges": 0, "short": 0, "blend": 0, "rejected": 0, "kept": 0}

    for edge_key, face_keys in topo.edge_faces.items():
        if len(face_keys) != 2:
            continue  # free, periodic or non-manifold: not a simple joint
        edge_id = topo.edge_ids[edge_key]
        edge_ref = part.edges.get(edge_id)
        if edge_ref is None:
            continue
        stats["edges"] += 1

        points = edge_ref.points
        if len(points) < 2 or pl.total_length(points) < opts.min_seam_length_mm:
            stats["short"] += 1
            continue

        fa_key, fb_key = face_keys
        ev_a, ev_b = evaluators[fa_key], evaluators[fb_key]
        face_a, face_b = topo.faces[fa_key], topo.faces[fb_key]

        blend = is_blend_face(
            backend, face_a, ev_a, opts.blend_max_radius_mm, opts.blend_max_width_mm
        ) or is_blend_face(backend, face_b, ev_b, opts.blend_max_radius_mm, opts.blend_max_width_mm)
        if blend:
            stats["blend"] += 1
            continue

        solid_a = topo.solid_ids.get(topo.face_solid.get(fa_key, -1), "")
        solid_b = topo.solid_ids.get(topo.face_solid.get(fb_key, -1), "")
        same_solid = solid_a == solid_b

        tangents = pl.tangents(points, closed=edge_ref.is_closed)
        mid = len(points) // 2

        side_a = FaceSide(
            face_id=topo.face_ids[fa_key],
            solid_id=solid_a,
            face=face_a,
            evaluator=ev_a,
            inward_sign=resolve_inward_sign(backend, face_a, ev_a, points[mid], tangents[mid]),
            thickness_mm=thickness_by_solid.get(solid_a, 0.0),
        )
        side_b = FaceSide(
            face_id=topo.face_ids[fb_key],
            solid_id=solid_b,
            face=face_b,
            evaluator=ev_b,
            inward_sign=resolve_inward_sign(backend, face_b, ev_b, points[mid], tangents[mid]),
            thickness_mm=thickness_by_solid.get(solid_b, 0.0),
        )

        geom = compute_seam_geometry(points, tangents, side_a, side_b, InwardMode.TRIM)

        if same_solid and geom.dominant_concavity < 0:
            stats["rejected"] += 1
            continue
        if not same_solid and geom.dominant_concavity < 0 and not opts.include_convex_cross_solid:
            stats["rejected"] += 1
            continue

        probe = min(2.0, max(0.25, 0.25 * pl.total_length(points)))
        # Quarter, middle and three-quarter stations: three readings so one
        # landing badly cannot move the answer.
        stations = np.unique(
            np.clip(
                np.round(np.array([0.25, 0.5, 0.75]) * (len(points) - 1)).astype(int),
                0,
                len(points) - 1,
            )
        )
        for side, normal_array, inward_array, solid_id in (
            (side_a, geom.normal_a, geom.inward_a, side_a.solid_id),
            (side_b, geom.normal_b, geom.inward_b, side_b.solid_id),
        ):
            caster = caster_for(solid_id)
            if caster is None:
                continue
            readings = []
            for i in stations:
                value = caster.gauge(
                    points[i] + inward_array[i] * probe,
                    normal_array[i],
                    face_min_extent(backend, side.face),
                )
                if value > 0.0:
                    readings.append(value)
            if readings:
                # Median, not mean: one station landing badly should not move
                # the answer at all.
                side.thickness_mm = round(float(np.median(readings)), 3)

        area_a, _ = backend.surface_properties(face_a)
        area_b, _ = backend.surface_properties(face_b)
        ctx = JointContext(
            groove_rad=geom.mean_groove,
            concavity=geom.dominant_concavity,
            groove_spread_rad=geom.groove_spread,
            same_solid=same_solid,
            contact_kind=ContactKind.SHARED_TOPOLOGY,
            face_a_type=ev_a.face_type,
            face_b_type=ev_b.face_type,
            thickness_a_mm=side_a.thickness_mm,
            thickness_b_mm=side_b.thickness_mm,
            area_a_mm2=area_a,
            area_b_mm2=area_b,
            seam_length_mm=pl.total_length(points),
            closed=edge_ref.is_closed,
            exact_geometry=True,
        )
        result = classify(ctx)
        if not result.weldable or result.confidence < opts.min_confidence:
            stats["rejected"] += 1
            continue

        seams.append(
            Seam(
                joint_type=result.joint_type,
                points=geom.points,
                tangents=geom.tangents,
                normal_a=geom.normal_a,
                normal_b=geom.normal_b,
                bisector=geom.bisector,
                dihedral=geom.groove,
                face_a_id=side_a.face_id,
                face_b_id=side_b.face_id,
                solid_a_id=solid_a,
                solid_b_id=solid_b,
                edge_ids=(edge_id,),
                thickness_a_mm=side_a.thickness_mm,
                thickness_b_mm=side_b.thickness_mm,
                confidence=result.confidence,
                closed=edge_ref.is_closed,
                notes=[*result.notes, *geom.notes, "source: shared topology"],
            )
        )
        stats["kept"] += 1

    logger.info("detect_topological_seams: %s", stats)
    return seams
