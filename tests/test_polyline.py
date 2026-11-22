"""Tests for seam polyline maths."""

from __future__ import annotations

import numpy as np
import pytest

from weldforge.geom import polyline as pl


def quarter_circle(radius: float = 100.0, n: int = 721) -> np.ndarray:
    """A finely sampled 90 deg arc in the XY plane."""
    t = np.linspace(0.0, np.pi / 2, n)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)])


LINE = np.array([[0.0, 0, 0], [100.0, 0, 0], [200.0, 0, 0]])
SQUARE = np.array([[0.0, 0, 0], [100.0, 0, 0], [100.0, 100.0, 0], [0, 100.0, 0], [0.0, 0, 0]])


# --- shape handling ---------------------------------------------------------


def test_as_points_accepts_single_point():
    assert pl.as_points([1.0, 2.0, 3.0]).shape == (1, 3)


def test_as_points_rejects_wrong_width():
    with pytest.raises(ValueError):
        pl.as_points([[1.0, 2.0]])


# --- length -----------------------------------------------------------------


def test_total_length_of_a_line():
    assert np.isclose(pl.total_length(LINE), 200.0)


def test_total_length_of_an_arc_is_not_the_chord():
    """The whole point of arc length: an R100 quarter circle is 157.08 mm long,
    while its chord is only 141.42 mm. Getting this wrong under-quotes every
    curved seam."""
    arc = quarter_circle()
    chord = float(np.linalg.norm(arc[-1] - arc[0]))
    assert np.isclose(chord, 100.0 * np.sqrt(2.0), atol=1e-6)
    assert np.isclose(pl.total_length(arc), 100.0 * np.pi / 2.0, rtol=1e-5)
    assert pl.total_length(arc) > chord


def test_cumulative_length_monotonic():
    cum = pl.cumulative_length(SQUARE)
    assert cum[0] == 0.0
    assert np.all(np.diff(cum) >= 0.0)
    assert np.isclose(cum[-1], 400.0)


def test_segment_lengths_of_single_point():
    assert pl.segment_lengths([[0.0, 0, 0]]).size == 0


# --- closure / dedupe -------------------------------------------------------


def test_is_closed():
    assert pl.is_closed(SQUARE)
    assert not pl.is_closed(LINE)
    assert not pl.is_closed(np.array([[0.0, 0, 0], [1.0, 0, 0]]))


def test_dedupe_removes_duplicates():
    pts = np.array([[0.0, 0, 0], [0.0, 0, 0], [10.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    assert len(pl.dedupe(pts)) == 3


def test_dedupe_keeps_closure():
    out = pl.dedupe(SQUARE)
    assert pl.is_closed(out)


def test_reverse():
    assert np.allclose(pl.reverse(LINE)[0], LINE[-1])


# --- sampling ---------------------------------------------------------------


def test_point_at_length_endpoints_and_middle():
    assert np.allclose(pl.point_at_length(LINE, 0.0), [0, 0, 0])
    assert np.allclose(pl.point_at_length(LINE, 200.0), [200, 0, 0])
    assert np.allclose(pl.point_at_length(LINE, 150.0), [150, 0, 0])


def test_point_at_length_clamps():
    assert np.allclose(pl.point_at_length(LINE, -50.0), [0, 0, 0])
    assert np.allclose(pl.point_at_length(LINE, 1e6), [200, 0, 0])


def test_resample_by_count_is_evenly_spaced():
    out = pl.resample_by_count(SQUARE, 41)
    assert len(out) == 41
    seg = pl.segment_lengths(out)
    assert np.allclose(seg, seg[0], atol=1e-6)


def test_resample_by_count_rejects_one():
    with pytest.raises(ValueError):
        pl.resample_by_count(LINE, 1)


def test_resample_by_spacing_divides_evenly():
    """No short final step: controllers show it as a speed glitch."""
    out = pl.resample_by_spacing(LINE, 30.0)
    seg = pl.segment_lengths(out)
    assert np.allclose(seg, seg[0], atol=1e-6)
    assert np.isclose(pl.total_length(out), 200.0)
    assert 25.0 <= seg[0] <= 35.0


def test_resample_by_spacing_keeps_ends_exact():
    out = pl.resample_by_spacing(SQUARE, 7.0)
    assert np.allclose(out[0], SQUARE[0])
    assert np.allclose(out[-1], SQUARE[-1])


def test_resample_by_spacing_rejects_zero():
    with pytest.raises(ValueError):
        pl.resample_by_spacing(LINE, 0.0)


def test_resample_preserves_arc_length_of_a_circle():
    arc = quarter_circle()
    out = pl.resample_by_spacing(arc, 2.0)
    assert np.isclose(pl.total_length(out), pl.total_length(arc), rtol=1e-3)


# --- tangents ---------------------------------------------------------------


def test_tangents_are_unit():
    for pts in (LINE, SQUARE, quarter_circle(n=37)):
        t = pl.tangents(pts)
        assert np.allclose(np.linalg.norm(t, axis=1), 1.0)


def test_tangents_of_a_line_are_constant():
    t = pl.tangents(LINE)
    assert np.allclose(t, [1.0, 0.0, 0.0])


def test_tangents_of_an_arc_are_perpendicular_to_the_radius():
    arc = quarter_circle(n=361)
    t = pl.tangents(arc)
    radial = arc / np.linalg.norm(arc, axis=1)[:, None]
    # Skip the one-sided ends, which are only first-order accurate.
    dots = np.abs(np.einsum("ij,ij->i", t[1:-1], radial[1:-1]))
    assert dots.max() < 1e-3


def test_tangents_wrap_on_a_closed_loop():
    t = pl.tangents(SQUARE, closed=True)
    assert len(t) == len(SQUARE)
    assert np.allclose(t[0], t[-1])
    # At the start vertex of a closed square the central difference bisects
    # the corner, so it points at 45 deg -- not along either edge.
    assert np.isclose(abs(np.dot(t[0], [1.0, 0.0, 0.0])), np.sqrt(0.5), atol=1e-6)


def test_tangents_of_degenerate_input():
    assert pl.tangents(np.zeros((0, 3))).shape == (0, 3)
    assert pl.tangents([[1.0, 2.0, 3.0]]).shape == (1, 3)


# --- simplify ---------------------------------------------------------------


def test_simplify_collapses_collinear_runs():
    dense = np.column_stack([np.linspace(0, 200, 500), np.zeros(500), np.zeros(500)])
    out = pl.simplify(dense, tolerance=0.05)
    assert len(out) == 2
    assert np.allclose(out[0], [0, 0, 0])
    assert np.allclose(out[-1], [200, 0, 0])


def test_simplify_keeps_corners():
    out = pl.simplify(SQUARE, tolerance=0.05)
    assert len(out) == 5
    assert pl.is_closed(out)


def test_simplify_respects_tolerance_on_an_arc():
    arc = quarter_circle(n=721)
    coarse = pl.simplify(arc, tolerance=1.0)
    fine = pl.simplify(arc, tolerance=0.01)
    assert len(coarse) < len(fine) < len(arc)
    assert pl.chord_deviation(arc) > 0.0


def test_simplify_short_input_is_passthrough():
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    assert len(pl.simplify(pts)) == 2


# --- derived quantities -----------------------------------------------------


def test_centroid_is_arc_length_weighted():
    """A polyline densely sampled at one end must not drag the centroid there."""
    biased = np.vstack(
        [
            np.column_stack([np.linspace(0, 10, 200), np.zeros(200), np.zeros(200)]),
            np.array([[100.0, 0.0, 0.0]]),
        ]
    )
    assert np.isclose(pl.centroid(biased)[0], 50.0, atol=1.0)
    assert biased.mean(axis=0)[0] < 15.0  # the naive vertex mean is badly off


def test_principal_axis_of_a_line():
    axis = pl.principal_axis(LINE)
    assert np.isclose(abs(np.dot(axis, [1.0, 0.0, 0.0])), 1.0)


def test_chord_deviation_of_a_line_is_zero():
    assert np.isclose(pl.chord_deviation(LINE), 0.0)


def test_chord_deviation_of_an_arc():
    arc = quarter_circle()
    # R100 quarter circle: sagitta = R * (1 - cos(45 deg)) = 29.29 mm
    assert np.isclose(pl.chord_deviation(arc), 100.0 * (1 - np.cos(np.pi / 4)), rtol=1e-3)


# ---------------------------------------------------------------------------
# proximity
# ---------------------------------------------------------------------------


def test_distance_to_a_straight_seam_is_measured_to_the_line():
    """The bug this exists to prevent: a clamp planted mid-weld.

    A straight 200 mm seam is stored as two points. Measuring to the nearer
    *vertex* puts a clamp at the seam's midpoint 100 mm away, so a clamp
    sitting on the bead is reported as clear.
    """
    seam = np.array([[0.0, 0, 0], [200.0, 0, 0]])
    query = np.array([100.0, 0.0, 10.0])
    assert np.isclose(pl.distance_to_point(seam, query), 10.0)
    nearest_vertex = min(float(np.linalg.norm(query - p)) for p in seam)
    assert nearest_vertex > 100.0  # what the naive measure would have returned


def test_closest_point_lands_on_the_segment():
    point, distance, index = pl.closest_point(SQUARE, [50.0, -20.0, 0.0])
    assert np.allclose(point, [50.0, 0.0, 0.0])
    assert np.isclose(distance, 20.0)
    assert index == 0


def test_closest_point_clamps_to_the_ends():
    point, distance, _ = pl.closest_point(LINE, [-50.0, 0.0, 0.0])
    assert np.allclose(point, [0.0, 0.0, 0.0])
    assert np.isclose(distance, 50.0)


def test_distance_to_a_single_point_polyline():
    assert np.isclose(pl.distance_to_point([[0.0, 0, 0]], [3.0, 4.0, 0.0]), 5.0)


def test_distance_to_a_degenerate_segment():
    pts = np.array([[0.0, 0, 0], [0.0, 0, 0], [10.0, 0, 0]])
    assert np.isclose(pl.distance_to_point(pts, [5.0, 3.0, 0.0]), 3.0)


def test_distance_to_points_batch():
    seam = np.array([[0.0, 0, 0], [100.0, 0, 0]])
    queries = np.array([[50.0, 5.0, 0.0], [50.0, 0.0, 20.0]])
    assert np.allclose(pl.distance_to_points(seam, queries), [5.0, 20.0])


def test_distance_to_an_arc_beats_the_vertex_measure():
    arc = pl.simplify(quarter_circle(), tolerance=2.0)
    # A point just outside the arc at its midpoint.
    mid = pl.point_at_length(arc, pl.total_length(arc) / 2.0)
    outward = mid / np.linalg.norm(mid)
    query = mid + outward * 7.0
    assert pl.distance_to_point(arc, query) <= 7.05
