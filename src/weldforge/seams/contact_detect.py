"""Seam detection from contact between separate bodies.

This is the detector that matters for real work. A weldment exported from
SolidWorks or FreeCAD arrives as an **assembly**: the parts are positioned
against each other but share no topology. On the customer weldment this tool
was built against -- 58 bodies, 885 faces -- every one of the 2266 edges is
internal to a single body, so a detector that only looks at shared edges finds
**zero** of the welds that actually have to be made.

How a contact becomes a seam
----------------------------
The weld does not run between the two faces that touch. Take a web standing on
a base plate:

* the **contact** is between the base's top face and the web's bottom face --
  two coincident planes with opposed normals;
* the **welds** run along the boundary of that contact footprint, each one
  between the base's top face and whichever *side* face of the web meets the
  footprint boundary there.

So the algorithm is: find the contacting face pair, walk the boundary edges of
the landing face, keep the stretches that really lie on the other face, and
form a seam between the receiving face and the third face adjacent to each
boundary edge. A T-joint footprint yields four seams -- two long sides and two
ends -- which is exactly what gets welded.

Why trim probing cannot be used here
------------------------------------
The base plate's top face continues *underneath* the web, so stepping either
way off the seam lands on material and the usual "which side is the face on"
probe has no answer. The in-face directions come instead from projecting each
flank's outward normal into the other's tangent plane
(:data:`InwardMode.CONCAVE`), which is exact for the concave joints that
contact always produces.

Butt joints
-----------
Two plates side by side never touch at all -- there is a root gap. They show up
as **coplanar** faces with the same normal direction, whose boundary edges run
parallel a gap apart, and the seam is the midline between those edges. Both the
top and the bottom pair are reported, since a double-sided butt is a legitimate
choice the operator makes, not one the software should make for them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from ..cad.backend import CadBackend
from ..cad.topology import FaceEvaluator, RawTopology, RayCaster, face_min_extent
from ..geom import polyline as pl
from ..model import Part, Seam, SolidInfo
from .classify import ContactKind, JointContext, classify
from .dihedral import (
    FaceSide,
    InwardMode,
    compute_seam_geometry,
    point_on_face,
    point_state_on_face,
    resolve_inward_sign,
    validate_in_face_directions,
)

logger = logging.getLogger(__name__)

__all__ = ["ContactOptions", "ContactPair", "detect_contact_seams", "find_contacts"]


@dataclass
class ContactOptions:
    """Tolerances for deciding two bodies are joined.

    Defaults suit a fabricated steel weldment modelled nominally:

    * ``gap_tol_mm = 0.5`` -- parts drawn in contact; allows for CAD round-off
      and for a deliberate 0.2-0.3 mm modelling clearance.
    * ``butt_gap_tol_mm = 4.0`` -- a root opening wider than 4 mm needs backing
      or a different joint preparation, so it should not be auto-detected.
    * ``min_overlap_fraction = 0.02`` -- a contact needs real area behind it;
      two parts merely grazing at a corner is not a joint.
    """

    gap_tol_mm: float = 0.5
    butt_gap_tol_mm: float = 4.0
    coplanar_tol_mm: float = 0.3
    parallel_dot: float = 0.999
    min_overlap_fraction: float = 0.02
    overlap_samples: int = 24
    min_seam_length_mm: float = 5.0
    min_confidence: float = 0.25
    # Skip flanks that are effectively coplanar with the receiving face: those
    # are the far side of the same landing, not a groove.
    min_groove_deg: float = 20.0
    max_groove_deg: float = 340.0
    include_butt: bool = True
    # Edge-to-edge contact: two members meeting along a line only, which is how
    # a true outside corner joint gets modelled. Their faces do end at the
    # seam, so these are measured by trim probing rather than normal
    # projection.
    include_edge_contact: bool = True
    edge_contact_tol_mm: float = 0.5
    edge_parallel_dot: float = 0.95
    # Validate that each flank extends away from the seam before believing a
    # contact joint. Switching this off restores the old, over-eager behaviour.
    validate_flanks: bool = True
    flank_probe_mm: float = 1.0
    # Measure thickness at the seam by ray casting instead of taking the
    # whole-body estimate. Matters for fused weldments and varying sections.
    local_thickness: bool = True
    # A closed boundary edge on a contact footprint is a hole rim. It is a plug
    # weld only if the receiving member is solid behind it; if the hole carries
    # straight on through, it is a fastener hole and must not be welded shut.
    reject_through_holes: bool = True
    max_solid_pairs: int = 50_000
    # Only planar faces take part in contact finding. Curved-to-curved contact
    # (a tube saddled onto another tube) needs a surface-surface intersection
    # and is out of scope for this pass; such joints must be picked manually.
    planar_only: bool = True


@dataclass
class ContactPair:
    """One characterised contact between two bodies."""

    solid_a: str
    solid_b: str
    face_a_key: int
    face_b_key: int
    kind: str
    gap_mm: float
    overlap_fraction: float = 0.0
    # Set only for edge-to-edge (line) contacts.
    edge_a_key: int | None = None
    edge_b_key: int | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "solid_a": self.solid_a,
            "solid_b": self.solid_b,
            "kind": self.kind,
            "gap_mm": round(self.gap_mm, 3),
            "overlap_fraction": round(self.overlap_fraction, 3),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bboxes_overlap(a: SolidInfo, b: SolidInfo, tol: float) -> bool:
    return bool(np.all(a.bbox_min - tol <= b.bbox_max) and np.all(b.bbox_min - tol <= a.bbox_max))


def _plate_axis(solid: SolidInfo) -> np.ndarray | None:
    """Unit vector along the solid's thickness, if it is plate-like.

    Found by matching the estimated thickness against the bounding-box extents:
    the axis whose extent equals the thickness is the through-thickness
    direction. Used to tell a plate lying **on** another plate (a lap) from one
    standing **on** it (a tee), which no amount of groove-angle measurement can
    distinguish.
    """
    if solid.thickness_mm <= 0.0:
        return None
    size = solid.bbox_size
    if not np.all(np.isfinite(size)):
        return None
    errors = np.abs(size - solid.thickness_mm)
    axis_index = int(np.argmin(errors))
    if errors[axis_index] > max(0.25, 0.1 * solid.thickness_mm):
        return None
    # Plate-like means the other two extents are clearly larger.
    others = [size[i] for i in range(3) if i != axis_index]
    if min(others) < 2.0 * solid.thickness_mm:
        return None
    return np.eye(3)[axis_index]


def _planar_faces(
    backend: CadBackend,
    topo: RawTopology,
    evaluators: dict[int, FaceEvaluator],
    solid_key: int,
    planar_only: bool,
) -> list[tuple[int, FaceEvaluator, np.ndarray, np.ndarray, float]]:
    """``(face_key, evaluator, normal, point_on_face, area)`` for one solid."""
    out = []
    for fk, owner in topo.face_solid.items():
        if owner != solid_key:
            continue
        ev = evaluators.get(fk)
        if ev is None:
            continue
        if planar_only and not ev.is_planar:
            continue
        area, centroid = backend.surface_properties(topo.faces[fk])
        normal = ev.normal_at_point(centroid)
        out.append((fk, ev, normal, centroid, area))
    return out


def _overlap_fraction(
    backend: CadBackend,
    receiving_face: Any,
    receiving_ev: FaceEvaluator,
    landing_ev: FaceEvaluator,
    n_samples: int,
) -> float:
    """Fraction of sampled points of the landing face that fall on the receiving face.

    The samples come from the landing face's triangulation, so they are
    genuinely inside its trim; each is then classified against the receiving
    face's 2-D boundary. Checking plane distance alone would happily report a
    contact between a plate and another plate 2 m away in the same plane.
    """
    samples = landing_ev.sample_on_face(n_samples)
    if not len(samples):
        return 0.0
    hits = sum(
        1
        for p in samples
        if point_state_on_face(backend, receiving_face, receiving_ev, p) in ("in", "on")
    )
    return hits / len(samples)


def _contiguous_runs(mask: np.ndarray, min_run: int = 2) -> list[np.ndarray]:
    """Index arrays for each run of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    runs = np.split(idx, splits)
    return [r for r in runs if len(r) >= min_run]


# ---------------------------------------------------------------------------
# contact finding
# ---------------------------------------------------------------------------


def find_contacts(
    part: Part,
    options: ContactOptions | None = None,
) -> list[ContactPair]:
    """Characterise every contact between the bodies of *part*."""
    opts = options or ContactOptions()
    backend: CadBackend | None = part.brep.backend
    topo: RawTopology | None = part.brep.topology
    evaluators: dict[int, FaceEvaluator] | None = part.brep.evaluators or None
    if topo is None or evaluators is None:
        return []
    if backend is None:
        from ..cad.backend import get_backend

        backend = get_backend()

    by_id = {s.id: s for s in part.solids}
    solid_keys = {topo.solid_ids[k]: k for k in topo.solids}
    plate_axes = {s.id: _plate_axis(s) for s in part.solids}
    face_cache: dict[int, list] = {}

    def faces_of(solid_id: str) -> list:
        key = solid_keys[solid_id]
        if key not in face_cache:
            face_cache[key] = _planar_faces(backend, topo, evaluators, key, opts.planar_only)
        return face_cache[key]

    contacts: list[ContactPair] = []
    examined = 0

    for sa, sb in combinations(sorted(by_id), 2):
        if examined >= opts.max_solid_pairs:
            logger.warning("find_contacts: hit max_solid_pairs=%d", opts.max_solid_pairs)
            break
        info_a, info_b = by_id[sa], by_id[sb]
        tol = max(opts.gap_tol_mm, opts.butt_gap_tol_mm if opts.include_butt else 0.0)
        if not _bboxes_overlap(info_a, info_b, tol):
            continue
        examined += 1

        for fa_key, ev_a, n_a, p_a, area_a in faces_of(sa):
            for fb_key, ev_b, n_b, p_b, area_b in faces_of(sb):
                alignment = float(np.dot(n_a, n_b))

                # --- opposed faces sitting against each other: a landing -----
                if alignment < -opts.parallel_dot:
                    gap = abs(float(np.dot(p_b - p_a, n_a)))
                    if gap > opts.gap_tol_mm:
                        continue
                    # Let the smaller face be the one that "lands".
                    if area_b <= area_a:
                        recv_key, recv_ev, land_key, land_ev = fa_key, ev_a, fb_key, ev_b
                    else:
                        recv_key, recv_ev, land_key, land_ev = fb_key, ev_b, fa_key, ev_a
                    fraction = _overlap_fraction(
                        backend,
                        topo.faces[recv_key],
                        recv_ev,
                        land_ev,
                        opts.overlap_samples,
                    )
                    if fraction < opts.min_overlap_fraction:
                        continue

                    kind = ContactKind.FACE_CONTACT
                    axis_landing = plate_axes[sb if land_key == fb_key else sa]
                    if axis_landing is not None and abs(float(np.dot(axis_landing, n_a))) > 0.9:
                        # The landing body's thickness runs along the contact
                        # normal: it lies flat on the other -> a lap.
                        kind = ContactKind.PARALLEL_OVERLAP
                    contacts.append(
                        ContactPair(
                            solid_a=sa,
                            solid_b=sb,
                            face_a_key=recv_key,
                            face_b_key=land_key,
                            kind=kind,
                            gap_mm=gap,
                            overlap_fraction=fraction,
                        )
                    )

                # --- coplanar, same direction: a butt ------------------------
                elif opts.include_butt and alignment > opts.parallel_dot:
                    offset = abs(float(np.dot(p_b - p_a, n_a)))
                    if offset > opts.coplanar_tol_mm:
                        continue
                    contacts.append(
                        ContactPair(
                            solid_a=sa,
                            solid_b=sb,
                            face_a_key=fa_key,
                            face_b_key=fb_key,
                            kind=ContactKind.COPLANAR_GAP,
                            gap_mm=offset,
                        )
                    )

        if opts.include_edge_contact:
            contacts.extend(_edge_contacts(part, topo, sa, sb, solid_keys, opts))

    logger.info(
        "find_contacts: %d solid pairs examined, %d contacts (%s)",
        examined,
        len(contacts),
        {k: sum(1 for c in contacts if c.kind == k) for k in {c.kind for c in contacts}},
    )
    return contacts


def _edge_contacts(
    part: Part,
    topo: RawTopology,
    solid_a: str,
    solid_b: str,
    solid_keys: dict[str, int],
    opts: ContactOptions,
) -> list[ContactPair]:
    """Edges of two bodies running close and parallel: a line contact.

    This is how an outside corner joint is modelled -- two plate ends meeting
    to form the corner, touching along a line and nowhere else, so no face pair
    ever overlaps. Without this pass such joints are invisible to the detector.
    """

    def edges_of(solid_id: str) -> list[tuple[int, np.ndarray]]:
        key = solid_keys[solid_id]
        out = []
        for ek, faces in topo.edge_faces.items():
            if topo.edge_kind(ek) != "manifold":
                continue
            if not any(topo.face_solid.get(fk) == key for fk in faces):
                continue
            ref = part.edges.get(topo.edge_ids[ek])
            if ref is None or len(ref.points) < 2:
                continue
            if pl.total_length(ref.points) < opts.min_seam_length_mm:
                continue
            out.append((ek, ref.points))
        return out

    list_a, list_b = edges_of(solid_a), edges_of(solid_b)
    out: list[ContactPair] = []
    for ek_a, pts_a in list_a:
        dir_a = pl.principal_axis(pts_a)
        for ek_b, pts_b in list_b:
            if abs(float(np.dot(dir_a, pl.principal_axis(pts_b)))) < opts.edge_parallel_dot:
                continue
            distances = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=2)
            gaps = distances.min(axis=1)
            close = gaps <= opts.edge_contact_tol_mm
            if not close.any():
                continue
            runs = _contiguous_runs(close, min_run=2)
            if not runs:
                continue
            longest = max(runs, key=lambda r: pl.total_length(pts_a[r]))
            if pl.total_length(pts_a[longest]) < opts.min_seam_length_mm:
                continue
            out.append(
                ContactPair(
                    solid_a=solid_a,
                    solid_b=solid_b,
                    face_a_key=-1,
                    face_b_key=-1,
                    kind=ContactKind.NEAR,
                    gap_mm=float(np.mean(gaps[longest])),
                    edge_a_key=ek_a,
                    edge_b_key=ek_b,
                )
            )
    return out


# ---------------------------------------------------------------------------
# seam generation
# ---------------------------------------------------------------------------


def detect_contact_seams(
    part: Part,
    options: ContactOptions | None = None,
    contacts: list[ContactPair] | None = None,
) -> list[Seam]:
    """Find seams from body-to-body contact in *part*."""
    opts = options or ContactOptions()
    backend: CadBackend | None = part.brep.backend
    topo: RawTopology | None = part.brep.topology
    evaluators: dict[int, FaceEvaluator] | None = part.brep.evaluators or None
    if topo is None or evaluators is None:
        logger.info("detect_contact_seams: part has no B-rep topology, skipping")
        return []
    if backend is None:
        from ..cad.backend import get_backend

        backend = get_backend()

    pairs = contacts if contacts is not None else find_contacts(part, opts)
    thickness = {s.id: s.thickness_mm for s in part.solids}
    seams: list[Seam] = []

    casters: dict[str, RayCaster] = {}

    def caster_for(solid_id: str) -> RayCaster | None:
        """Ray caster bound to one body, built on first use."""
        if not opts.local_thickness:
            return None
        if solid_id not in casters:
            key = next((k for k, v in topo.solid_ids.items() if v == solid_id), None)
            if key is None:
                return None
            casters[solid_id] = RayCaster(backend, topo.solids[key])
        return casters.get(solid_id)

    for pair in pairs:
        if pair.edge_a_key is not None and pair.edge_b_key is not None:
            seams.extend(
                _seams_from_edge_contact(
                    backend, part, topo, evaluators, pair, thickness, opts, caster_for
                )
            )
        elif pair.kind == ContactKind.COPLANAR_GAP:
            seams.extend(
                _seams_from_coplanar(
                    backend, part, topo, evaluators, pair, thickness, opts, caster_for
                )
            )
        else:
            # A landing is walked from both faces: the smaller face's boundary
            # gives the usual fillet, and the larger face's boundary catches the
            # case where the receiving plate ends inside the overlap.
            for swap in (False, True):
                seams.extend(
                    _seams_from_landing(
                        backend,
                        part,
                        topo,
                        evaluators,
                        pair,
                        thickness,
                        opts,
                        swap,
                        caster_for,
                    )
                )

    logger.info("detect_contact_seams: %d contacts -> %d seam fragments", len(pairs), len(seams))
    return seams


def _seams_from_landing(
    backend: CadBackend,
    part: Part,
    topo: RawTopology,
    evaluators: dict[int, FaceEvaluator],
    pair: ContactPair,
    thickness: dict[str, float],
    opts: ContactOptions,
    swap: bool,
    caster_for,
) -> list[Seam]:
    """Seams along the boundary of a contact footprint.

    Two passes, because the second needs a fact the first discovers.

    **Pass 1** walks every boundary edge of the landing face and asks, for
    each, whether the receiving face continues outward from it. An edge where
    it does not is a *flush* boundary: the two members run out together and
    there is no groove to weld. Those edges are dropped -- this is what stops
    the detector reporting a weld down the outside of an L, or across the end
    of a web that finishes level with its base plate.

    **Pass 2** builds the surviving seams. If pass 1 saw any flush boundary,
    the landing member reaches the edge of the receiving face, which makes the
    joint an L rather than a T -- and the classifier is told so.
    """
    recv_key = pair.face_b_key if swap else pair.face_a_key
    land_key = pair.face_a_key if swap else pair.face_b_key
    recv_face = topo.faces[recv_key]
    recv_ev = evaluators[recv_key]
    recv_solid = topo.solid_ids.get(topo.face_solid.get(recv_key, -1), "")
    land_solid = topo.solid_ids.get(topo.face_solid.get(land_key, -1), "")
    if recv_solid == land_solid:
        return []

    distance_tol = pair.gap_mm + 0.25
    candidates: list[tuple[str, int, np.ndarray, bool]] = []
    # Directions of boundary edges found to be flush. Only a flush boundary
    # running *parallel* to a seam tells us anything about that seam: it means
    # the receiving plate ends alongside the member, making the joint an L. A
    # flush boundary across the member is just its end finishing level with the
    # plate, which says nothing about the long seams.
    flush_directions: list[np.ndarray] = []

    # --- pass 1: which boundary edges have a groove at all? ----------------
    for edge_key in topo.face_edges.get(land_key, ()):
        if topo.edge_kind(edge_key) == "periodic_seam":
            continue
        edge_id = topo.edge_ids[edge_key]
        edge_ref = part.edges.get(edge_id)
        if edge_ref is None or len(edge_ref.points) < 2:
            continue

        # The third face: the flank of the landing body rising from this
        # boundary edge. That, not the landing face, is the weld flank.
        flanks = [k for k in topo.edge_faces.get(edge_key, ()) if k != land_key]
        if not flanks:
            continue

        points = edge_ref.points

        if (
            opts.reject_through_holes
            and edge_ref.is_closed
            and _is_through_hole(backend, recv_face, recv_ev, points)
        ):
            # A hole that continues into the receiving member: a bolt or pin
            # hole. Welding it shut would be a serious error, and on the
            # customer weldment this single check removes 60-odd phantom plug
            # welds that are actually the bolted connections.
            logger.debug("skipping through-hole rim %s", edge_id)
            continue

        on_receiver = np.array(
            [
                point_on_face(backend, recv_face, recv_ev, p, distance_tol=distance_tol)
                for p in points
            ]
        )
        runs = _contiguous_runs(on_receiver, min_run=2)
        if not runs:
            continue

        for flank_key in flanks:
            flank_ev = evaluators.get(flank_key)
            if flank_ev is None:
                continue
            for run in runs:
                sub = points[run]
                if pl.total_length(sub) < opts.min_seam_length_mm:
                    continue
                closed = bool(len(run) == len(points) and edge_ref.is_closed)
                ok = True
                if opts.validate_flanks:
                    ok, reason = _flank_check(
                        backend,
                        topo,
                        recv_key,
                        recv_ev,
                        flank_key,
                        flank_ev,
                        sub,
                        closed,
                        opts,
                    )
                    if not ok:
                        flush_directions.append(pl.principal_axis(sub))
                        logger.debug("rejected seam fragment: %s", reason)
                if ok:
                    candidates.append((edge_id, flank_key, sub, closed))

    # --- pass 2: build what survived ---------------------------------------
    out: list[Seam] = []
    for edge_id, flank_key, sub, closed in candidates:
        axis = pl.principal_axis(sub)
        flush_parallel = any(abs(float(np.dot(axis, other))) > 0.9 for other in flush_directions)
        seam = _build_seam(
            backend=backend,
            topo=topo,
            recv_key=recv_key,
            recv_ev=recv_ev,
            flank_key=flank_key,
            flank_ev=evaluators[flank_key],
            points=sub,
            closed=closed,
            pair=pair,
            thickness=thickness,
            opts=opts,
            mode=InwardMode.CONCAVE,
            lateral=None,
            edge_ids=(edge_id,),
            source="contact: footprint boundary",
            caster_for=caster_for,
            validate=False,  # already checked in pass 1
            landing_flush_boundary=flush_parallel,
        )
        if seam is not None:
            out.append(seam)
    return out


def _is_through_hole(
    backend: CadBackend,
    recv_face: Any,
    recv_ev: FaceEvaluator,
    rim_points: np.ndarray,
) -> bool:
    """Does a closed rim sit over a matching hole in the receiving face?

    The rim points themselves lie on the receiving face either way -- they fall
    on its boundary when the holes line up. The **centre** settles it: inside
    the receiving face means solid material behind the hole (a plug weld is
    possible), outside means the receiving member is open there too, so the two
    holes are a fastener passage.
    """
    centre = rim_points.mean(axis=0)
    return point_state_on_face(backend, recv_face, recv_ev, centre) != "in"


def _flank_check(
    backend: CadBackend,
    topo: RawTopology,
    recv_key: int,
    recv_ev: FaceEvaluator,
    flank_key: int,
    flank_ev: FaceEvaluator,
    points: np.ndarray,
    closed: bool,
    opts: ContactOptions,
) -> tuple[bool, str]:
    """Do both flanks extend away from this seam? See :func:`validate_in_face_directions`."""
    tangents = pl.tangents(points, closed=closed)
    side_a = FaceSide(
        face_id=topo.face_ids[recv_key],
        solid_id=topo.solid_ids.get(topo.face_solid.get(recv_key, -1), ""),
        face=topo.faces[recv_key],
        evaluator=recv_ev,
    )
    side_b = FaceSide(
        face_id=topo.face_ids[flank_key],
        solid_id=topo.solid_ids.get(topo.face_solid.get(flank_key, -1), ""),
        face=topo.faces[flank_key],
        evaluator=flank_ev,
    )
    geom = compute_seam_geometry(points, tangents, side_a, side_b, InwardMode.CONCAVE)
    return validate_in_face_directions(backend, side_a, side_b, geom, probe_mm=opts.flank_probe_mm)


def _seams_from_coplanar(
    backend: CadBackend,
    part: Part,
    topo: RawTopology,
    evaluators: dict[int, FaceEvaluator],
    pair: ContactPair,
    thickness: dict[str, float],
    opts: ContactOptions,
    caster_for,
) -> list[Seam]:
    """Butt seams along the gap between two coplanar faces."""
    fa_key, fb_key = pair.face_a_key, pair.face_b_key
    ev_a, ev_b = evaluators[fa_key], evaluators[fb_key]
    out: list[Seam] = []

    edges_b = [
        part.edges.get(topo.edge_ids[ek])
        for ek in topo.face_edges.get(fb_key, ())
        if topo.edge_kind(ek) != "periodic_seam"
    ]
    edges_b = [e for e in edges_b if e is not None and len(e.points) >= 2]
    if not edges_b:
        return []

    for ek_a in topo.face_edges.get(fa_key, ()):
        if topo.edge_kind(ek_a) == "periodic_seam":
            continue
        edge_a = part.edges.get(topo.edge_ids[ek_a])
        if edge_a is None or len(edge_a.points) < 2:
            continue
        points_a = edge_a.points

        for edge_b in edges_b:
            # Nearest point on edge B for every station of edge A.
            deltas = points_a[:, None, :] - edge_b.points[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            nearest = np.argmin(distances, axis=1)
            gaps = distances[np.arange(len(points_a)), nearest]
            close = gaps <= opts.butt_gap_tol_mm
            if not close.any():
                continue

            for run in _contiguous_runs(close, min_run=2):
                sub_a = points_a[run]
                sub_b = edge_b.points[nearest[run]]
                lateral = sub_b - sub_a
                lengths = np.linalg.norm(lateral, axis=1, keepdims=True)
                if float(np.max(lengths)) < 1e-6:
                    continue
                # Seam runs up the middle of the root gap.
                midline = (sub_a + sub_b) / 2.0
                if pl.total_length(midline) < opts.min_seam_length_mm:
                    continue
                safe = np.where(lengths < 1e-9, 1.0, lengths)
                seam = _build_seam(
                    backend=backend,
                    topo=topo,
                    recv_key=fa_key,
                    recv_ev=ev_a,
                    flank_key=fb_key,
                    flank_ev=ev_b,
                    points=midline,
                    closed=False,
                    pair=pair,
                    thickness=thickness,
                    opts=opts,
                    mode=InwardMode.LATERAL,
                    lateral=lateral / safe,
                    edge_ids=(edge_a.id, edge_b.id),
                    source="contact: coplanar root gap",
                    gap_override=float(np.mean(gaps[run])),
                    caster_for=caster_for,
                    validate=False,
                )
                if seam is not None:
                    out.append(seam)
    return out


def _seams_from_edge_contact(
    backend: CadBackend,
    part: Part,
    topo: RawTopology,
    evaluators: dict[int, FaceEvaluator],
    pair: ContactPair,
    thickness: dict[str, float],
    opts: ContactOptions,
    caster_for,
) -> list[Seam]:
    """Corner seams where two bodies meet along a line.

    Both members end at the seam, so the in-face directions come from trim
    probing (:data:`InwardMode.TRIM`) -- the normal-projection rule would be
    invalid here, since neither face continues past the contact.

    Of the two faces meeting at each edge, the pair forming the **widest**
    groove is the weld side: that is the one a torch can reach.
    """
    assert pair.edge_a_key is not None and pair.edge_b_key is not None
    edge_a = part.edges.get(topo.edge_ids[pair.edge_a_key])
    if edge_a is None:
        return []
    points = edge_a.points
    tangents = pl.tangents(points, closed=edge_a.is_closed)
    mid = len(points) // 2

    faces_a = [
        fk
        for fk in topo.edge_faces.get(pair.edge_a_key, ())
        if topo.solid_ids.get(topo.face_solid.get(fk, -1)) == pair.solid_a
    ]
    faces_b = [
        fk
        for fk in topo.edge_faces.get(pair.edge_b_key, ())
        if topo.solid_ids.get(topo.face_solid.get(fk, -1)) == pair.solid_b
    ]
    if not faces_a or not faces_b:
        return []

    best: tuple[float, Seam] | None = None
    for fa in faces_a:
        for fb in faces_b:
            ev_a, ev_b = evaluators.get(fa), evaluators.get(fb)
            if ev_a is None or ev_b is None:
                continue
            seam = _build_seam(
                backend=backend,
                topo=topo,
                recv_key=fa,
                recv_ev=ev_a,
                flank_key=fb,
                flank_ev=ev_b,
                points=points,
                closed=edge_a.is_closed,
                pair=pair,
                thickness=thickness,
                opts=opts,
                mode=InwardMode.TRIM,
                lateral=None,
                edge_ids=(edge_a.id,),
                source="contact: edge-to-edge line contact",
                caster_for=caster_for,
                validate=False,
            )
            if seam is None:
                continue
            groove = float(np.mean(seam.dihedral)) if len(seam.dihedral) else 0.0
            if best is None or groove > best[0]:
                best = (groove, seam)

    del mid, tangents  # the chosen seam recomputed both itself
    return [best[1]] if best is not None else []


def _build_seam(
    *,
    backend: CadBackend,
    topo: RawTopology,
    recv_key: int,
    recv_ev: FaceEvaluator,
    flank_key: int,
    flank_ev: FaceEvaluator,
    points: np.ndarray,
    closed: bool,
    pair: ContactPair,
    thickness: dict[str, float],
    opts: ContactOptions,
    mode: str,
    lateral: np.ndarray | None,
    edge_ids: tuple[str, ...],
    source: str,
    gap_override: float | None = None,
    caster_for=None,
    validate: bool = False,
    landing_flush_boundary: bool = False,
) -> Seam | None:
    """Evaluate, classify and assemble one candidate seam; ``None`` if rejected."""
    recv_solid = topo.solid_ids.get(topo.face_solid.get(recv_key, -1), "")
    flank_solid = topo.solid_ids.get(topo.face_solid.get(flank_key, -1), "")

    tangents = pl.tangents(points, closed=closed)
    side_a = FaceSide(
        face_id=topo.face_ids[recv_key],
        solid_id=recv_solid,
        face=topo.faces[recv_key],
        evaluator=recv_ev,
        thickness_mm=thickness.get(recv_solid, 0.0),
    )
    side_b = FaceSide(
        face_id=topo.face_ids[flank_key],
        solid_id=flank_solid,
        face=topo.faces[flank_key],
        evaluator=flank_ev,
        thickness_mm=thickness.get(flank_solid, 0.0),
    )
    if mode == InwardMode.TRIM:
        mid = len(points) // 2
        side_a.inward_sign = resolve_inward_sign(
            backend, side_a.face, side_a.evaluator, points[mid], tangents[mid]
        )
        side_b.inward_sign = resolve_inward_sign(
            backend, side_b.face, side_b.evaluator, points[mid], tangents[mid]
        )

    geom = compute_seam_geometry(points, tangents, side_a, side_b, mode, lateral)

    if validate:
        ok, reason = validate_in_face_directions(
            backend, side_a, side_b, geom, probe_mm=opts.flank_probe_mm
        )
        if not ok:
            logger.debug("rejected seam: %s", reason)
            return None

    # Thickness measured at the seam beats the whole-body estimate: cast into
    # each flank's material, which is the side its outward normal faces away
    # from.
    if caster_for is not None:
        mid = len(points) // 2
        # Probe a little way **into each flank** rather than at the seam
        # itself. At the seam the two members meet at a corner, and for a butt
        # the station sits in the root gap where a ray finds no material at
        # all -- which would silently fall back to the face's own size (100 mm
        # for a plate skin) instead of its 8 mm gauge.
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
            (side_a, geom.normal_a, geom.inward_a, recv_solid),
            (side_b, geom.normal_b, geom.inward_b, flank_solid),
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

    groove_deg = float(np.rad2deg(geom.mean_groove))
    if not (opts.min_groove_deg <= groove_deg <= opts.max_groove_deg):
        return None
    if pair.kind != ContactKind.COPLANAR_GAP and abs(groove_deg - 180.0) < opts.min_groove_deg:
        # The flank is effectively coplanar with the receiving face: this is the
        # far side of the same landing, not a groove.
        return None

    area_a, _ = backend.surface_properties(topo.faces[recv_key])
    area_b, _ = backend.surface_properties(topo.faces[flank_key])
    gap = pair.gap_mm if gap_override is None else gap_override

    ctx = JointContext(
        groove_rad=geom.mean_groove,
        concavity=geom.dominant_concavity,
        groove_spread_rad=geom.groove_spread,
        same_solid=False,
        contact_kind=pair.kind,
        gap_mm=gap,
        face_a_type=recv_ev.face_type,
        face_b_type=flank_ev.face_type,
        thickness_a_mm=side_a.thickness_mm,
        thickness_b_mm=side_b.thickness_mm,
        area_a_mm2=area_a,
        area_b_mm2=area_b,
        seam_length_mm=pl.total_length(points),
        closed=closed,
        exact_geometry=True,
        landing_flush_boundary=landing_flush_boundary,
    )
    result = classify(ctx)
    if not result.weldable or result.confidence < opts.min_confidence:
        return None

    return Seam(
        joint_type=result.joint_type,
        points=geom.points,
        tangents=geom.tangents,
        normal_a=geom.normal_a,
        normal_b=geom.normal_b,
        bisector=geom.bisector,
        dihedral=geom.groove,
        face_a_id=side_a.face_id,
        face_b_id=side_b.face_id,
        solid_a_id=recv_solid,
        solid_b_id=flank_solid,
        edge_ids=edge_ids,
        thickness_a_mm=side_a.thickness_mm,
        thickness_b_mm=side_b.thickness_mm,
        gap_mm=gap,
        confidence=result.confidence,
        closed=closed,
        notes=[*result.notes, *geom.notes, source],
    )
