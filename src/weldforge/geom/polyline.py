"""Polyline maths for weld seams.

A seam is ultimately a 3-D polyline that the torch tip must follow, so every
operation here is written for that use: arc length (not chord length -- the
distinction is what makes a fillet round a 90 deg corner come out the right
length), resampling at a constant spacing, tangent estimation that stays
continuous through corners, and Douglas-Peucker simplification to collapse the
many tiny collinear segments a STEP tessellation leaves behind.

All lengths in millimetres.
"""

from __future__ import annotations

import numpy as np

from .transform import EPS, unit

__all__ = [
    "as_points",
    "segment_lengths",
    "cumulative_length",
    "total_length",
    "is_closed",
    "dedupe",
    "resample_by_count",
    "resample_by_spacing",
    "point_at_length",
    "tangents",
    "simplify",
    "centroid",
    "principal_axis",
    "chord_deviation",
    "reverse",
    "distance_to_point",
    "distance_to_points",
    "closest_point",
]


def as_points(points: np.ndarray | list) -> np.ndarray:
    """Coerce to a contiguous ``(N, 3)`` float array."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1 and arr.size == 3:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) point array, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def segment_lengths(points: np.ndarray) -> np.ndarray:
    """Length of each segment; ``(N-1,)`` for N points."""
    pts = as_points(points)
    if len(pts) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def cumulative_length(points: np.ndarray) -> np.ndarray:
    """Arc length at each vertex, starting at 0; ``(N,)``."""
    seg = segment_lengths(points)
    return np.concatenate([[0.0], np.cumsum(seg)])


def total_length(points: np.ndarray) -> float:
    """True arc length along the polyline.

    Not ``norm(end - start)``: for an arc or any curved seam the chord
    underestimates the weld length, and weld length drives both cycle time and
    consumable cost.
    """
    return float(segment_lengths(points).sum())


def is_closed(points: np.ndarray, tol: float = 1e-6) -> bool:
    pts = as_points(points)
    if len(pts) < 3:
        return False
    return bool(np.linalg.norm(pts[-1] - pts[0]) <= tol)


def dedupe(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Drop consecutive duplicate vertices (keeps a closing vertex intact)."""
    pts = as_points(points)
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > tol:
            keep.append(i)
    out = pts[keep]
    # Preserve an explicit closing vertex that dedupe would otherwise keep as
    # a near-duplicate of the first.
    if len(out) >= 3 and is_closed(pts, tol) and not is_closed(out, tol):
        out = np.vstack([out, out[0]])
    return out


def reverse(points: np.ndarray) -> np.ndarray:
    """Reverse travel direction."""
    return as_points(points)[::-1].copy()


def point_at_length(points: np.ndarray, s: float) -> np.ndarray:
    """Interpolated point at arc length *s* (clamped to the ends)."""
    pts = as_points(points)
    if len(pts) == 1:
        return pts[0].copy()
    cum = cumulative_length(pts)
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(pts) - 2))
    span = cum[i + 1] - cum[i]
    t = 0.0 if span < EPS else (s - cum[i]) / span
    return pts[i] + t * (pts[i + 1] - pts[i])


def resample_by_count(points: np.ndarray, count: int) -> np.ndarray:
    """Resample to exactly *count* points, evenly spaced in arc length."""
    if count < 2:
        raise ValueError("count must be >= 2")
    pts = as_points(points)
    if len(pts) < 2:
        return np.repeat(pts, count, axis=0)
    cum = cumulative_length(pts)
    targets = np.linspace(0.0, cum[-1], count)
    return np.array([point_at_length(pts, s) for s in targets])


def resample_by_spacing(
    points: np.ndarray,
    spacing: float,
    keep_ends: bool = True,
) -> np.ndarray:
    """Resample at a nominal *spacing* in mm.

    The actual spacing is adjusted down so the samples divide the polyline
    evenly -- a torch path with a short final step produces a visible speed
    glitch on most controllers.
    """
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")
    pts = as_points(points)
    length = total_length(pts)
    if length < EPS:
        return pts[:1].copy()
    n_seg = max(1, int(round(length / spacing)))
    out = resample_by_count(pts, n_seg + 1)
    if keep_ends:
        out[0] = pts[0]
        out[-1] = pts[-1]
    return out


def tangents(points: np.ndarray, closed: bool | None = None) -> np.ndarray:
    """Unit tangent at each vertex; ``(N, 3)``.

    Interior vertices use the central difference of the two adjacent chords,
    which keeps the tangent continuous where segment lengths change abruptly.
    End vertices use the one-sided chord, or wrap around when the polyline is
    closed. Degenerate vertices inherit the previous valid tangent so the
    result never contains a zero vector.
    """
    pts = dedupe(points)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.array([[1.0, 0.0, 0.0]])

    if closed is None:
        closed = is_closed(pts)

    work = pts[:-1] if (closed and is_closed(pts)) else pts
    m = len(work)
    out = np.zeros((m, 3))

    for i in range(m):
        if closed:
            prev_p = work[(i - 1) % m]
            next_p = work[(i + 1) % m]
            v = next_p - prev_p
        elif i == 0:
            v = work[1] - work[0]
        elif i == m - 1:
            v = work[-1] - work[-2]
        else:
            v = work[i + 1] - work[i - 1]
        u = unit(v)
        out[i] = u if u is not None else (out[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0]))

    if closed and is_closed(pts):
        out = np.vstack([out, out[0]])
    return out


def simplify(points: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
    """Douglas-Peucker simplification, *tolerance* in mm.

    STEP tessellation and edge chaining both produce runs of near-collinear
    vertices; feeding those straight to a robot bloats the program and can
    make the controller stutter. 0.05 mm is well below weld-pool resolution.
    """
    pts = dedupe(points)
    if len(pts) < 3:
        return pts

    closed = is_closed(pts)
    work = pts[:-1] if closed else pts
    if len(work) < 3:
        return pts

    keep = np.zeros(len(work), dtype=bool)
    keep[0] = keep[-1] = True
    stack: list[tuple[int, int]] = [(0, len(work) - 1)]

    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        a, b = work[lo], work[hi]
        ab = b - a
        norm_ab = float(np.linalg.norm(ab))
        seg = work[lo + 1 : hi] - a
        if norm_ab < EPS:
            dist = np.linalg.norm(seg, axis=1)
        else:
            dist = np.linalg.norm(np.cross(seg, ab / norm_ab), axis=1)
        idx = int(np.argmax(dist))
        if dist[idx] > tolerance:
            split = lo + 1 + idx
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))

    out = work[keep]
    if closed:
        out = np.vstack([out, out[0]])
    return out


def centroid(points: np.ndarray) -> np.ndarray:
    """Arc-length-weighted centroid (not the vertex mean).

    Vertex means are biased towards whichever end was tessellated finely.
    """
    pts = as_points(points)
    if len(pts) == 1:
        return pts[0].copy()
    seg = segment_lengths(pts)
    if seg.sum() < EPS:
        return pts.mean(axis=0)
    mids = (pts[:-1] + pts[1:]) / 2.0
    return (mids * seg[:, None]).sum(axis=0) / seg.sum()


def principal_axis(points: np.ndarray) -> np.ndarray:
    """Dominant direction of the polyline (first PCA component).

    Used to tell a straight seam from a curved one and to pick a travel
    direction hint.
    """
    pts = as_points(points)
    if len(pts) < 2:
        return np.array([1.0, 0.0, 0.0])
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = unit(vt[0])
    return axis if axis is not None else np.array([1.0, 0.0, 0.0])


def chord_deviation(points: np.ndarray) -> float:
    """Max distance from the straight chord, in mm; 0 for a straight seam."""
    pts = as_points(points)
    if len(pts) < 3:
        return 0.0
    a, b = pts[0], pts[-1]
    ab = b - a
    n = float(np.linalg.norm(ab))
    rel = pts - a
    if n < EPS:
        return float(np.linalg.norm(rel, axis=1).max())
    return float(np.linalg.norm(np.cross(rel, ab / n), axis=1).max())


# ---------------------------------------------------------------------------
# proximity
# ---------------------------------------------------------------------------


def closest_point(points: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Nearest point **on the polyline** to *query*.

    Returns ``(point, distance_mm, segment_index)``.

    Distance to the polyline, not to its vertices -- and the difference is not
    academic. A straight seam is stored as two points, so a clamp planted
    exactly halfway along a 200 mm weld is 100 mm from the nearer *vertex*
    while sitting right on the weld. Measuring to vertices reported that clamp
    as clear.
    """
    pts = as_points(points)
    q = np.asarray(query, dtype=float).reshape(3)
    if len(pts) == 1:
        return pts[0].copy(), float(np.linalg.norm(q - pts[0])), 0

    starts = pts[:-1]
    vectors = pts[1:] - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    lengths_sq = np.where(lengths_sq < 1e-18, 1.0, lengths_sq)
    t = np.clip(np.einsum("ij,ij->i", q - starts, vectors) / lengths_sq, 0.0, 1.0)
    projections = starts + t[:, None] * vectors
    distances = np.linalg.norm(projections - q, axis=1)
    index = int(np.argmin(distances))
    return projections[index], float(distances[index]), index


def distance_to_point(points: np.ndarray, query: np.ndarray) -> float:
    """Shortest distance from *query* to the polyline, in mm."""
    return closest_point(points, query)[1]


def distance_to_points(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Shortest distance from each of *queries* to the polyline; ``(M,)``."""
    qs = as_points(queries)
    return np.array([distance_to_point(points, q) for q in qs])
