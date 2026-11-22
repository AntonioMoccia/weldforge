"""Chain detected seam fragments into continuous welds.

A 200 mm seam rarely arrives as one entity. The CAD may split the face it runs
along, a B-spline may be stored as several segments, and the contact finder
emits one fragment per boundary edge. Welding them as separate passes means an
extra arc start, an extra crater and a visible notch at every joint, so
fragments that are really one weld must be merged before anything downstream
sees them.

Two fragments merge when **all** of these hold:

* they share the same pair of flank faces (or at least the same pair of
  bodies, with a compatible joint type);
* an endpoint of one is within ``position_tol`` of an endpoint of the other;
* the tangents are continuous there to within ``angle_tol`` -- a genuine corner
  must stay a corner, because the torch has to stop and re-orient;
* the groove angle matches to within ``groove_tol``.

Merging is order-independent: fragments are joined greedily through a
union-find over endpoint proximity, then each group is walked end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..geom import polyline as pl
from ..geom import transform as tf
from ..model import Seam

logger = logging.getLogger(__name__)

__all__ = ["ChainOptions", "chain_seams", "merge_pair"]


@dataclass
class ChainOptions:
    """Tolerances for deciding that two fragments are one weld."""

    position_tol_mm: float = 0.2
    angle_tol_deg: float = 12.0
    groove_tol_deg: float = 15.0
    require_same_faces: bool = False
    # Seams shorter than this are dropped after chaining: below one weld-pool
    # diameter there is nothing a robot can usefully deposit.
    min_length_mm: float = 5.0
    resample_spacing_mm: float = 0.0  # 0 = keep the stations as they are

    @property
    def angle_tol_rad(self) -> float:
        return float(np.deg2rad(self.angle_tol_deg))

    @property
    def groove_tol_rad(self) -> float:
        return float(np.deg2rad(self.groove_tol_deg))


# ---------------------------------------------------------------------------
# compatibility
# ---------------------------------------------------------------------------


def _same_flanks(a: Seam, b: Seam, strict: bool) -> bool:
    faces_a = {a.face_a_id, a.face_b_id}
    faces_b = {b.face_a_id, b.face_b_id}
    if faces_a == faces_b and "" not in faces_a:
        return True
    if strict:
        return False
    solids_a = {a.solid_a_id, a.solid_b_id}
    solids_b = {b.solid_a_id, b.solid_b_id}
    return solids_a == solids_b and a.joint_type is b.joint_type


def _endpoint_match(a: Seam, b: Seam, tol: float) -> tuple[bool, bool] | None:
    """Find which ends touch.

    Returns ``(reverse_a, reverse_b)`` such that ``a`` then ``b`` run
    head-to-tail, or ``None`` if no endpoints are within *tol*.
    """
    pairs = (
        # (a_end, b_start): already head to tail
        (np.linalg.norm(a.end - b.start), False, False),
        # (a_end, b_end): flip b
        (np.linalg.norm(a.end - b.end), False, True),
        # (a_start, b_start): flip a
        (np.linalg.norm(a.start - b.start), True, False),
        # (a_start, b_end): flip both
        (np.linalg.norm(a.start - b.end), True, True),
    )
    best = min(pairs, key=lambda p: p[0])
    if best[0] > tol:
        return None
    return bool(best[1]), bool(best[2])


def _tangent_continuous(a: Seam, b: Seam, reverse_a: bool, reverse_b: bool, tol: float) -> bool:
    """Is the travel direction continuous across the junction?"""
    if not len(a.tangents) or not len(b.tangents):
        return True
    t_a = a.tangents[0] if reverse_a else a.tangents[-1]
    if reverse_a:
        t_a = -t_a
    t_b = b.tangents[-1] if reverse_b else b.tangents[0]
    if reverse_b:
        t_b = -t_b
    return tf.angle_between(t_a, t_b) <= tol


def _groove_compatible(a: Seam, b: Seam, tol: float) -> bool:
    if not len(a.dihedral) or not len(b.dihedral):
        return True
    return abs(a.mean_dihedral - b.mean_dihedral) <= tol


def can_merge(a: Seam, b: Seam, options: ChainOptions) -> tuple[bool, bool] | None:
    """``(reverse_a, reverse_b)`` if *a* and *b* are one weld, else ``None``."""
    if not _same_flanks(a, b, options.require_same_faces):
        return None
    if not _groove_compatible(a, b, options.groove_tol_rad):
        return None
    ends = _endpoint_match(a, b, options.position_tol_mm)
    if ends is None:
        return None
    reverse_a, reverse_b = ends
    if not _tangent_continuous(a, b, reverse_a, reverse_b, options.angle_tol_rad):
        return None
    return reverse_a, reverse_b


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------


_ARRAY_FIELDS = ("points", "tangents", "normal_a", "normal_b", "bisector")


def merge_pair(a: Seam, b: Seam, reverse_a: bool, reverse_b: bool) -> Seam:
    """Concatenate two fragments into one seam, dropping the duplicate junction."""
    left = a.reversed() if reverse_a else a
    right = b.reversed() if reverse_b else b

    out = Seam(
        joint_type=left.joint_type if left.joint_type.is_weldable else right.joint_type,
        face_a_id=left.face_a_id,
        face_b_id=left.face_b_id,
        solid_a_id=left.solid_a_id,
        solid_b_id=left.solid_b_id,
        edge_ids=tuple(dict.fromkeys([*left.edge_ids, *right.edge_ids])),
        thickness_a_mm=max(left.thickness_a_mm, right.thickness_a_mm),
        thickness_b_mm=max(left.thickness_b_mm, right.thickness_b_mm),
        gap_mm=max(left.gap_mm, right.gap_mm),
        label=left.label or right.label,
    )

    # Length-weighted confidence: a long confident fragment should not be
    # dragged down by a 2 mm doubtful one.
    la, lb = max(left.length_mm, 1e-9), max(right.length_mm, 1e-9)
    out.confidence = (left.confidence * la + right.confidence * lb) / (la + lb)

    for name in _ARRAY_FIELDS:
        va, vb = getattr(left, name), getattr(right, name)
        if not len(va):
            setattr(out, name, vb.copy() if len(vb) else va)
            continue
        if not len(vb):
            setattr(out, name, va.copy())
            continue
        setattr(out, name, np.vstack([va, vb[1:]]))

    if len(left.dihedral) and len(right.dihedral):
        out.dihedral = np.concatenate([left.dihedral, right.dihedral[1:]])
    elif len(left.dihedral):
        out.dihedral = left.dihedral.copy()
    else:
        out.dihedral = right.dihedral.copy()

    out.closed = pl.is_closed(out.points)
    notes = [*left.notes, *right.notes]
    out.notes = list(dict.fromkeys(notes))
    return out


def chain_seams(seams: list[Seam], options: ChainOptions | None = None) -> list[Seam]:
    """Merge every chainable fragment; returns the consolidated seams.

    Greedy and repeated until nothing more merges, which handles a run of
    N fragments regardless of the order they were detected in.
    """
    opts = options or ChainOptions()
    work = list(seams)
    merged_any = True
    guard = 0

    while merged_any:
        merged_any = False
        guard += 1
        if guard > len(seams) + 10:  # pragma: no cover - safety net
            logger.warning("chain_seams: stopping after %d passes", guard)
            break

        out: list[Seam] = []
        consumed: set[int] = set()
        for i, seam in enumerate(work):
            if i in consumed:
                continue
            current = seam
            for j in range(i + 1, len(work)):
                if j in consumed:
                    continue
                ends = can_merge(current, work[j], opts)
                if ends is None:
                    continue
                current = merge_pair(current, work[j], *ends)
                consumed.add(j)
                merged_any = True
            out.append(current)
        work = out

    # Re-derive tangents once at the end: a merged seam's junction tangents
    # were computed from the fragments and can kink by a degree or two.
    for seam in work:
        if seam.n_points >= 2:
            seam.tangents = pl.tangents(seam.points, closed=seam.closed)

    if opts.resample_spacing_mm > 0.0:
        work = [_resample(seam, opts.resample_spacing_mm) for seam in work]

    kept = [s for s in work if s.length_mm >= opts.min_length_mm]
    dropped = len(work) - len(kept)
    logger.info(
        "chain_seams: %d fragments -> %d seams (%d dropped below %.1f mm)",
        len(seams),
        len(kept),
        dropped,
        opts.min_length_mm,
    )
    return kept


def _resample(seam: Seam, spacing_mm: float) -> Seam:
    """Re-station a seam at a uniform spacing, carrying the flank data along."""
    if seam.n_points < 2:
        return seam
    source_cum = pl.cumulative_length(seam.points)
    new_points = pl.resample_by_spacing(seam.points, spacing_mm)
    new_cum = pl.cumulative_length(new_points)

    seam.points = new_points
    for name in ("normal_a", "normal_b", "bisector"):
        values = getattr(seam, name)
        if not len(values):
            continue
        resampled = np.empty((len(new_points), 3))
        for axis in range(3):
            resampled[:, axis] = np.interp(new_cum, source_cum, values[:, axis])
        # Interpolating unit vectors shortens them; renormalise.
        norms = np.linalg.norm(resampled, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        setattr(seam, name, resampled / norms)
    if len(seam.dihedral):
        seam.dihedral = np.interp(new_cum, source_cum, seam.dihedral)
    seam.tangents = pl.tangents(seam.points, closed=seam.closed)
    return seam
