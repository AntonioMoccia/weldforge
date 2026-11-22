"""Seam recognition tests against specimens whose answers are known by construction.

Each specimen is the canonical form of one joint type, so the expected seam
count, length, groove angle, joint type and member thickness are all known in
advance -- no golden files, no tolerances pulled out of the air.

Three behaviours get their own tests because they are the ones that separate a
usable detector from a noisy one:

* **contact detection** finds joints between bodies that share no topology,
  which is every weldment exported as an assembly;
* **flush rejection** suppresses the phantom seams where two members finish
  level with each other and there is no groove;
* **through-hole rejection** stops a bolt hole being planned as a plug weld.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import PLATE_THICK, WEB_THICK, requires_cad

from weldforge.cad import load_part
from weldforge.geom import polyline as pl
from weldforge.model import JointType, WeldPosition
from weldforge.seams import (
    ChainOptions,
    DetectOptions,
    chain_seams,
    detect_seams,
    detect_topological_seams,
    find_contacts,
)
from weldforge.seams.chain import can_merge, merge_pair
from weldforge.seams.classify import ContactKind
from weldforge.seams.detect import deduplicate_seams

pytestmark = pytest.mark.needs_cad


def groove_deg(seam) -> float:
    return float(np.rad2deg(seam.mean_dihedral))


# ---------------------------------------------------------------------------
# T joint as an assembly: the case topology alone cannot see
# ---------------------------------------------------------------------------


@requires_cad
def test_tee_assembly_shares_no_topology_yet_is_detected(tee_assembly_step):
    """The headline capability: two bodies touching, no shared edge, 2 seams."""
    part = load_part(tee_assembly_step)
    assert detect_topological_seams(part) == []  # nothing to find topologically

    seams, _ = detect_seams(part)
    assert len(seams) == 2
    for seam in seams:
        assert seam.joint_type is JointType.TEE
        assert np.isclose(seam.length_mm, 200.0, atol=0.01)
        assert np.isclose(groove_deg(seam), 90.0, atol=0.5)
        assert seam.spans_two_solids


@requires_cad
def test_tee_assembly_recovers_member_thicknesses(tee_assembly_step):
    seams, _ = detect_seams(load_part(tee_assembly_step))
    for seam in seams:
        got = sorted([seam.thickness_a_mm, seam.thickness_b_mm])
        assert np.allclose(got, [WEB_THICK, PLATE_THICK], atol=0.05)
        assert np.isclose(seam.min_thickness_mm, WEB_THICK, atol=0.05)


@requires_cad
def test_tee_assembly_seams_run_along_the_footprint_edges(tee_assembly_step):
    """Both seams sit on the contact plane, 6 mm apart -- the web's thickness."""
    seams, _ = detect_seams(load_part(tee_assembly_step))
    heights = [seam.points[:, 2] for seam in seams]
    for z in heights:
        assert np.allclose(z, PLATE_THICK, atol=0.01)
    offsets = sorted(float(np.mean(seam.points[:, 1])) for seam in seams)
    assert np.isclose(offsets[1] - offsets[0], WEB_THICK, atol=0.05)


@requires_cad
def test_tee_assembly_wire_points_into_the_joint(tee_assembly_step):
    """Torch direction must have a downward component and lean toward the web."""
    seams, _ = detect_seams(load_part(tee_assembly_step))
    for seam in seams:
        for b in seam.bisector:
            assert b[2] < -0.5, "wire must point downwards into the corner"
            assert np.isclose(np.linalg.norm(b), 1.0, atol=1e-9)
        # Each seam leans towards the web, i.e. opposite ways for the two sides.
        assert abs(float(np.mean(seam.bisector[:, 1]))) > 0.5


@requires_cad
def test_flush_ends_are_not_reported_as_seams(tee_assembly_step):
    """The web runs the full length of the base, so its ends have no groove.

    Without the flank check this specimen yields two extra 6 mm "seams" where
    the two members simply finish level with each other.
    """
    part = load_part(tee_assembly_step)
    strict, _ = detect_seams(part)
    assert len(strict) == 2

    permissive_opts = DetectOptions()
    permissive_opts.contact.validate_flanks = False
    permissive, _ = detect_seams(load_part(tee_assembly_step), permissive_opts)
    assert len(permissive) > len(strict)
    assert any(s.length_mm < 10.0 for s in permissive)


# ---------------------------------------------------------------------------
# the same joint fused: topological detection
# ---------------------------------------------------------------------------


@requires_cad
def test_tee_fused_is_found_topologically(tee_fused_step):
    part = load_part(tee_fused_step)
    seams = detect_topological_seams(part)
    assert len(seams) == 2
    for seam in seams:
        assert seam.joint_type is JointType.FILLET  # one body: an inside corner
        assert np.isclose(groove_deg(seam), 90.0, atol=0.5)
        assert not seam.spans_two_solids


@requires_cad
def test_fused_body_gets_local_gauge_not_body_thickness(tee_fused_step):
    """A fused weldment has no single thickness; each flank is measured at the seam."""
    part = load_part(tee_fused_step)
    assert len(part.solids) == 1
    seams, _ = detect_seams(part)
    for seam in seams:
        got = sorted([seam.thickness_a_mm, seam.thickness_b_mm])
        assert np.allclose(got, [WEB_THICK, PLATE_THICK], atol=0.05)


@requires_cad
def test_convex_edges_of_one_body_are_rejected(tee_fused_step):
    """The fused tee has 26 edges; only the 2 concave ones are welds."""
    part = load_part(tee_fused_step)
    assert len(part.edges) == 26
    assert len(detect_topological_seams(part)) == 2


# ---------------------------------------------------------------------------
# other joint types
# ---------------------------------------------------------------------------


@requires_cad
def test_lap_joint_is_classified_as_lap_not_tee(lap_assembly_step):
    """Only the plate-axis test separates these: both measure a 90 deg groove."""
    seams, _ = detect_seams(load_part(lap_assembly_step))
    assert len(seams) == 2
    for seam in seams:
        assert seam.joint_type is JointType.LAP
        assert np.isclose(groove_deg(seam), 90.0, atol=0.5)
        assert np.allclose(
            sorted([seam.thickness_a_mm, seam.thickness_b_mm]),
            [PLATE_THICK, PLATE_THICK],
            atol=0.05,
        )


@requires_cad
def test_butt_joint_with_a_root_gap(butt_assembly_step):
    """Top, bottom and both ends: four butt seams across a 2 mm gap."""
    seams, _ = detect_seams(load_part(butt_assembly_step))
    assert len(seams) == 4
    for seam in seams:
        assert seam.joint_type is JointType.BUTT
        assert np.isclose(groove_deg(seam), 180.0, atol=1.0)
        assert 1.5 <= seam.gap_mm <= 2.5
    long_seams = [s for s in seams if s.length_mm > 100.0]
    assert len(long_seams) == 2
    # One on the top face, one underneath.
    assert {round(float(np.mean(s.points[:, 2])), 1) for s in long_seams} == {0.0, 8.0}


@requires_cad
def test_butt_seam_runs_up_the_middle_of_the_gap(butt_assembly_step):
    seams, _ = detect_seams(load_part(butt_assembly_step))
    top = max((s for s in seams if s.length_mm > 100.0), key=lambda s: np.mean(s.points[:, 2]))
    # Plates end at y=100 and start at y=102, so the midline is y=101.
    assert np.allclose(top.points[:, 1], 101.0, atol=0.1)


@requires_cad
def test_butt_positions_differ_top_and_bottom(butt_assembly_step):
    """The underside of a butt is an overhead weld and must be reported as one."""
    seams, _ = detect_seams(load_part(butt_assembly_step))
    long_seams = sorted(
        (s for s in seams if s.length_mm > 100.0), key=lambda s: np.mean(s.points[:, 2])
    )
    assert long_seams[0].dominant_position is WeldPosition.PE  # underside
    assert long_seams[0].dominant_position.is_overhead
    assert long_seams[-1].dominant_position is WeldPosition.PA  # top, in the gutter
    assert long_seams[-1].dominant_position.is_downhand


@requires_cad
def test_l_joint_is_a_fillet_not_a_tee(corner_assembly_step):
    """A member on the lip of a plate is an L; one in the middle is a T.

    The discriminator is a *parallel* flush boundary -- the receiving plate
    ending alongside the member. The member's own flush ends say nothing, which
    is why the T specimen above still comes out as a T.
    """
    seams, _ = detect_seams(load_part(corner_assembly_step))
    assert len(seams) == 1
    assert seams[0].joint_type is JointType.FILLET
    assert np.isclose(seams[0].length_mm, 200.0, atol=0.01)


@requires_cad
def test_plate_with_a_hole_has_nothing_to_weld(plate_with_hole_step):
    seams, _ = detect_seams(load_part(plate_with_hole_step))
    assert seams == []


# ---------------------------------------------------------------------------
# curved seams
# ---------------------------------------------------------------------------


@requires_cad
def test_curved_seam_arc_length_is_exact(curved_tee_step):
    """A web standing on a base along a 90 deg arc.

    Inner and outer contact lines are at R97 and R103, so the seams are
    152.4 mm and 161.8 mm. A detector using chord length would report 137 mm
    and 145 mm -- a 10% under-quote on every curved weld.
    """
    seams, _ = detect_seams(load_part(curved_tee_step))
    arcs = sorted((s for s in seams if s.length_mm > 50.0), key=lambda s: s.length_mm)
    assert len(arcs) == 2
    assert np.isclose(arcs[0].length_mm, np.pi / 2 * 97.0, rtol=2e-3)
    assert np.isclose(arcs[1].length_mm, np.pi / 2 * 103.0, rtol=2e-3)
    for seam in arcs:
        assert not seam.is_straight
        chord = float(np.linalg.norm(seam.end - seam.start))
        assert seam.length_mm > chord * 1.05


@requires_cad
def test_curved_seam_tangents_and_groove_stay_consistent(curved_tee_step):
    seams, _ = detect_seams(load_part(curved_tee_step))
    for seam in (s for s in seams if s.length_mm > 50.0):
        assert np.allclose(np.linalg.norm(seam.tangents, axis=1), 1.0, atol=1e-6)
        # The groove is 90 deg the whole way round; a wandering groove would
        # mean the flank normals are being evaluated wrongly on the cylinder.
        assert np.allclose(np.rad2deg(seam.dihedral), 90.0, atol=1.5)
        assert np.allclose(np.linalg.norm(seam.bisector, axis=1), 1.0, atol=1e-6)


@requires_cad
def test_curved_seam_wire_follows_the_curve(curved_tee_step):
    """On a curved seam the wire direction must rotate with the seam, not stay fixed."""
    seams, _ = detect_seams(load_part(curved_tee_step))
    arc = max(seams, key=lambda s: s.length_mm)
    spread = max(
        float(np.linalg.norm(arc.bisector[i] - arc.bisector[0])) for i in range(arc.n_points)
    )
    assert spread > 0.5


# ---------------------------------------------------------------------------
# contacts
# ---------------------------------------------------------------------------


@requires_cad
def test_find_contacts_characterises_a_tee(tee_assembly_step):
    contacts = find_contacts(load_part(tee_assembly_step))
    kinds = {c.kind for c in contacts}
    assert ContactKind.FACE_CONTACT in kinds
    landing = next(c for c in contacts if c.kind == ContactKind.FACE_CONTACT)
    assert landing.gap_mm < 0.01
    assert landing.overlap_fraction > 0.5


@requires_cad
def test_find_contacts_distinguishes_a_lap(lap_assembly_step):
    contacts = find_contacts(load_part(lap_assembly_step))
    assert any(c.kind == ContactKind.PARALLEL_OVERLAP for c in contacts)


@requires_cad
def test_find_contacts_sees_the_butt_gap(butt_assembly_step):
    contacts = find_contacts(load_part(butt_assembly_step))
    assert any(c.kind == ContactKind.COPLANAR_GAP for c in contacts)


@requires_cad
def test_contact_detection_can_be_switched_off(tee_assembly_step):
    part = load_part(tee_assembly_step)
    seams, _ = detect_seams(part, DetectOptions.for_single_body())
    assert seams == []


@requires_cad
def test_overlap_requirement_rejects_grazing_bodies(specimen_dir):
    """Two plates meeting only at a corner are not joined."""
    from conftest import _box, _compound, _write_step

    from weldforge.cad.backend import get_backend

    b = get_backend()
    first = _box(b, 100.0, 100.0, 8.0)
    # Diagonally offset so the bounding boxes touch at one corner only.
    second = _box(b, 100.0, 100.0, 8.0, origin=(100.0, 100.0, 8.0))
    path = _write_step(b, _compound(b, [first, second]), specimen_dir / "grazing.step")
    seams, _ = detect_seams(load_part(path))
    assert seams == []


# ---------------------------------------------------------------------------
# chaining
# ---------------------------------------------------------------------------


@requires_cad
def test_chaining_merges_a_split_seam(tee_assembly_step):
    """Split one seam in half and confirm chaining puts it back together."""
    part = load_part(tee_assembly_step)
    seams, _ = detect_seams(part)
    original = seams[0]
    half = original.n_points // 2

    import copy

    left = copy.deepcopy(original)
    right = copy.deepcopy(original)
    for attr in ("points", "tangents", "normal_a", "normal_b", "bisector"):
        setattr(left, attr, getattr(original, attr)[: half + 1])
        setattr(right, attr, getattr(original, attr)[half:])
    left.dihedral = original.dihedral[: half + 1]
    right.dihedral = original.dihedral[half:]

    merged = chain_seams([left, right])
    assert len(merged) == 1
    assert np.isclose(merged[0].length_mm, original.length_mm, rtol=1e-6)


@requires_cad
def test_chaining_keeps_a_real_corner_apart(tee_assembly_step):
    """A 90 deg direction change is a corner the torch must stop at."""
    part = load_part(tee_assembly_step)
    seams, _ = detect_seams(part)
    a, b = seams[0], seams[1]
    assert can_merge(a, b, ChainOptions()) is None


def test_merge_pair_averages_confidence_by_length():
    from weldforge.model import Seam

    long_seam = Seam(
        points=np.array([[0.0, 0, 0], [100.0, 0, 0]]),
        tangents=np.tile([1.0, 0, 0], (2, 1)),
        dihedral=np.full(2, np.pi / 2),
        confidence=0.9,
        face_a_id="f1",
        face_b_id="f2",
    )
    short_seam = Seam(
        points=np.array([[100.0, 0, 0], [105.0, 0, 0]]),
        tangents=np.tile([1.0, 0, 0], (2, 1)),
        dihedral=np.full(2, np.pi / 2),
        confidence=0.3,
        face_a_id="f1",
        face_b_id="f2",
    )
    merged = merge_pair(long_seam, short_seam, False, False)
    assert np.isclose(merged.length_mm, 105.0)
    # Length-weighted: the 5 mm doubtful fragment barely moves the result.
    assert merged.confidence > 0.85


def test_chain_drops_fragments_below_one_weld_pool():
    from weldforge.model import Seam

    tiny = Seam(
        points=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        tangents=np.tile([1.0, 0, 0], (2, 1)),
        confidence=0.9,
    )
    assert chain_seams([tiny], ChainOptions(min_length_mm=5.0)) == []


def test_chain_resampling_preserves_length_and_unit_vectors():
    from weldforge.model import Seam

    points = np.column_stack([np.linspace(0, 100, 7), np.zeros(7), np.zeros(7)])
    seam = Seam(
        points=points,
        tangents=np.tile([1.0, 0, 0], (7, 1)),
        bisector=np.tile([0.0, 0, -1.0], (7, 1)),
        normal_a=np.tile([0.0, 0, 1.0], (7, 1)),
        normal_b=np.tile([0.0, 1.0, 0], (7, 1)),
        dihedral=np.full(7, np.pi / 2),
        confidence=0.9,
    )
    out = chain_seams([seam], ChainOptions(resample_spacing_mm=5.0))
    assert len(out) == 1
    assert np.isclose(out[0].length_mm, 100.0, rtol=1e-9)
    assert out[0].n_points == 21
    assert np.allclose(np.linalg.norm(out[0].bisector, axis=1), 1.0)


# ---------------------------------------------------------------------------
# de-duplication
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_the_more_confident_twin():
    from weldforge.model import Seam

    points = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.zeros(11)])
    weak = Seam(points=points.copy(), confidence=0.4)
    strong = Seam(points=points.copy(), confidence=0.9)
    kept, removed = deduplicate_seams([weak, strong])
    assert removed == 1
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_deduplicate_keeps_genuinely_separate_seams():
    from weldforge.model import Seam

    first = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.zeros(11)])
    second = first + np.array([0.0, 50.0, 0.0])
    kept, removed = deduplicate_seams(
        [Seam(points=first, confidence=0.8), Seam(points=second, confidence=0.8)]
    )
    assert removed == 0
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# labels and reporting
# ---------------------------------------------------------------------------


@requires_cad
def test_seams_are_labelled_and_sorted_longest_first(butt_assembly_step):
    seams, report = detect_seams(load_part(butt_assembly_step))
    lengths = [s.length_mm for s in seams]
    assert lengths == sorted(lengths, reverse=True)
    assert all(s.label for s in seams)
    assert seams[0].label.startswith("01 -")
    assert report.summary()["seams_after_chaining"] == len(seams)


@requires_cad
def test_seam_summary_is_serialisable(tee_assembly_step):
    import json

    seams, _ = detect_seams(load_part(tee_assembly_step))
    payload = json.dumps([s.summary() for s in seams])
    assert "tee" in payload


# ---------------------------------------------------------------------------
# the real weldment
# ---------------------------------------------------------------------------


@requires_cad
@pytest.mark.slow
def test_company_weldment_detection_is_plausible(company_step):
    """Sanity bounds on the real 58-body part, if it is available.

    Not an exact expectation -- nobody has counted its welds by hand -- but the
    numbers that would indicate a broken detector are all checked: no seams at
    all, implausible member thicknesses, or a hole planned as a plug weld.
    """
    if company_step is None:
        pytest.skip("customer STEP not available")

    part = load_part(company_step)
    seams, report = detect_seams(part)

    assert len(seams) > 50, "a 58-body weldment must have many joints"
    assert 1000.0 < sum(s.length_mm for s in seams) < 20000.0

    thicknesses = [t for s in seams for t in (s.thickness_a_mm, s.thickness_b_mm) if t > 0]
    assert min(thicknesses) >= 2.0, "no weldable member is thinner than 2 mm"
    assert all(min(s.thickness_a_mm, s.thickness_b_mm) > 0 for s in seams)

    # Bolt holes must not come back as plug welds.
    assert not any(s.joint_type is JointType.PLUG for s in seams)

    # Every seam must carry usable torch geometry.
    for seam in seams:
        assert seam.n_points >= 2
        assert np.allclose(np.linalg.norm(seam.bisector, axis=1), 1.0, atol=1e-6)
        assert np.allclose(np.linalg.norm(seam.tangents, axis=1), 1.0, atol=1e-6)
        assert seam.length_mm > 0.0
        assert np.isclose(seam.length_mm, pl.total_length(seam.points))

    assert report.n_duplicates >= 0
    assert report.seconds < 120.0


# ---------------------------------------------------------------------------
# welding position
# ---------------------------------------------------------------------------


@requires_cad
def test_tee_fillet_is_a_horizontal_vertical_fillet(tee_assembly_step):
    """Plate flat, web upright: position PB, the most common fillet there is."""
    seams, _ = detect_seams(load_part(tee_assembly_step))
    for seam in seams:
        assert seam.dominant_position is WeldPosition.PB
        assert seam.dominant_position.is_downhand


@requires_cad
def test_turning_the_part_over_changes_the_position(tee_assembly_step):
    """A T joint welded upside down is an overhead fillet (PD), not PB.

    This is the whole reason the positioner exists, and the reason position is
    derived from the seam's orientation in the cell rather than stored once.
    """
    from weldforge.geom import transform as tf

    seams, _ = detect_seams(load_part(tee_assembly_step))
    upright = seams[0]
    assert upright.dominant_position is WeldPosition.PB

    flipped = upright.transformed(tf.pose(tf.rot_x(np.pi), [0.0, 0.0, 0.0]))
    assert flipped.dominant_position is WeldPosition.PD
    assert flipped.dominant_position.is_overhead


@requires_cad
def test_standing_the_part_on_end_gives_a_vertical_seam(tee_assembly_step):
    """Rotate so the seam runs up the plate: position PF."""
    from weldforge.geom import transform as tf

    seams, _ = detect_seams(load_part(tee_assembly_step))
    rotated = seams[0].transformed(tf.pose(tf.rot_y(np.pi / 2), [0.0, 0.0, 0.0]))
    assert rotated.dominant_position is WeldPosition.PF
    assert rotated.dominant_position.is_vertical
