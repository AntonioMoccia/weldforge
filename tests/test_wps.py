"""Welding procedure and machine setpoint tests.

The procedure engine is checked against the physics it claims to use -- mass
conservation between wire feed and deposit -- plus the practical rules a
welding engineer would apply: pass counts that respect the bead a position can
hold, heat input inside the grade's ceiling, preheat by thickness, no weaving
on a root pass, and procedures that fail loudly rather than inventing numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_seam

from weldforge.model import JointType, Process, TransferMode, WeaveShape, WeldPosition
from weldforge.units import heat_input_kj_mm
from weldforge.wps import (
    ProcedureOptions,
    available_machines,
    available_procedures,
    fillet_area_mm2,
    fillet_leg_for_thickness,
    generate_procedure,
    generate_procedures,
    groove_area_mm2,
    load_machine,
    load_materials,
    load_procedure,
    setpoints_for_pass,
    setpoints_for_procedure,
)
from weldforge.wps.machines import JobRule, SignalScale
from weldforge.wps.procedure import CONVEXITY_FACTOR, _distribute_area

# ---------------------------------------------------------------------------
# data tables
# ---------------------------------------------------------------------------


def test_procedures_and_machines_are_discoverable():
    assert "gmaw_carbon_steel" in available_procedures()
    assert "gmaw_stainless" in available_procedures()
    assert {"generic_analog", "fronius_tps", "lincoln_powerwave"} <= set(available_machines())


def test_unknown_procedure_names_the_alternatives():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_procedure("gmaw_unobtainium")


def test_synergic_line_interpolates_monotonically():
    wire = load_procedure("gmaw_carbon_steel").wire(1.2)
    currents = np.linspace(wire.min_current_a, wire.max_current_a, 25)
    voltages = [wire.voltage_for_current(c) for c in currents]
    feeds = [wire.wire_speed_for_current(c) for c in currents]
    assert np.all(np.diff(voltages) >= 0.0)
    assert np.all(np.diff(feeds) >= 0.0)


def test_synergic_line_inverts_consistently():
    """Wire feed is the quantity the engine solves for, so the inverse must hold."""
    wire = load_procedure("gmaw_carbon_steel").wire(1.2)
    for current in (140.0, 200.0, 280.0):
        feed = wire.wire_speed_for_current(current)
        assert np.isclose(wire.current_for_wire_speed(feed), current, rtol=1e-6)


def test_wire_current_clamping_reports_itself():
    wire = load_procedure("gmaw_carbon_steel").wire(1.2)
    value, clamped = wire.clamp_current(900.0)
    assert value == wire.max_current_a and clamped
    value, clamped = wire.clamp_current(200.0)
    assert value == 200.0 and not clamped


def test_wire_choice_follows_thickness():
    data = load_procedure("gmaw_carbon_steel")
    assert data.default_wire(1.5).diameter_mm == 0.8
    assert data.default_wire(4.0).diameter_mm == 1.0
    assert data.default_wire(10.0).diameter_mm == 1.2
    assert data.default_wire(30.0).diameter_mm == 1.6


def test_missing_wire_size_falls_back_to_the_nearest():
    assert load_procedure("gmaw_carbon_steel").wire(1.1).diameter_mm in (1.0, 1.2)


def test_preheat_rises_with_thickness_and_grade():
    materials = load_materials()
    s355, s460 = materials["S355"], materials["S460"]
    assert s355.preheat_for(10.0) == 20.0
    assert s355.preheat_for(25.0) > s355.preheat_for(10.0)
    assert s355.preheat_for(60.0) > s355.preheat_for(25.0)
    # The higher-strength grade needs preheat sooner.
    assert s460.preheat_for(10.0) > s355.preheat_for(10.0)


def test_austenitic_stainless_is_never_preheated():
    stainless = load_materials()["316L"]
    assert stainless.preheat_for(5.0) == 20.0
    assert stainless.preheat_for(60.0) == 20.0
    # ...but has a hard interpass ceiling against sensitisation.
    assert stainless.interpass_max_c <= 150.0


# ---------------------------------------------------------------------------
# deposit geometry
# ---------------------------------------------------------------------------


def test_fillet_leg_design_rule():
    assert fillet_leg_for_thickness(10.0, ratio=0.7) == 7.0
    assert fillet_leg_for_thickness(10.0, ratio=1.0) == 10.0
    # Rounded to the half millimetre a weld gauge reads.
    assert fillet_leg_for_thickness(6.0, ratio=0.7) == 4.0
    # Bounded at both ends.
    assert fillet_leg_for_thickness(1.0) == 3.0
    assert fillet_leg_for_thickness(100.0) == 12.0
    assert fillet_leg_for_thickness(0.0) == 3.0


def test_fillet_area_is_convex_not_triangular():
    """A real bead sits proud of the nominal triangle."""
    assert fillet_area_mm2(6.0) > 0.5 * 36.0
    assert np.isclose(fillet_area_mm2(6.0), 0.5 * 36.0 * CONVEXITY_FACTOR)


def test_square_butt_only_fills_the_gap():
    area, needs_prep = groove_area_mm2(3.0, 2.0, 60.0)
    assert not needs_prep
    assert area < 10.0


def test_thick_butt_needs_preparation_and_much_more_filler():
    thin, _ = groove_area_mm2(3.0, 2.0, 60.0)
    thick, needs_prep = groove_area_mm2(20.0, 2.0, 60.0)
    assert needs_prep
    assert thick > 10.0 * thin


def test_area_distribution_runs_a_small_root():
    areas = _distribute_area(100.0, 5)
    assert np.isclose(sum(areas), 100.0)
    assert areas[0] < areas[1]
    assert len(set(np.round(areas[1:], 9))) == 1


def test_area_distribution_of_a_single_pass():
    assert _distribute_area(42.0, 1) == [42.0]


# ---------------------------------------------------------------------------
# the engine: conservation of deposited volume
# ---------------------------------------------------------------------------


def test_wire_feed_and_travel_speed_conserve_volume():
    """The identity the whole engine is built on, checked on every pass.

    ``wire_area * feed * efficiency == deposit_area * travel_speed``

    If these two were taken from independent tables they would disagree, and
    the weld would come out over- or under-filled with no warning anywhere.
    """
    data = load_procedure("gmaw_carbon_steel")
    seam = make_seam(JointType.TEE, 12.0, 12.0)
    procedure = generate_procedure(seam)

    from weldforge.wps.procedure import _distribute_area as distribute

    total_area = fillet_area_mm2(procedure.target_leg_mm)
    expected = distribute(total_area, procedure.n_passes)

    for weld_pass, area in zip(procedure.passes, expected, strict=True):
        wire = data.wire(weld_pass.wire_diameter_mm)
        supplied = (
            wire.cross_section_mm2
            * (weld_pass.wire_speed_m_min * 1000.0 / 60.0)
            * data.deposition_efficiency
        )
        deposited = area * weld_pass.travel_speed_mm_s
        assert np.isclose(supplied, deposited, rtol=0.02), (
            f"pass {weld_pass.index}: supplies {supplied:.1f} mm3/s but "
            f"deposits {deposited:.1f} mm3/s"
        )


def test_total_deposit_matches_the_required_fillet():
    seam = make_seam(JointType.TEE, 12.0, 12.0)
    procedure = generate_procedure(seam)
    deposited = sum(fillet_area_mm2(p.leg_mm) for p in procedure.passes)
    assert np.isclose(deposited, fillet_area_mm2(procedure.target_leg_mm), rtol=0.02)


@pytest.mark.parametrize("thickness", [3.0, 6.0, 10.0, 16.0, 25.0, 40.0])
def test_no_pass_exceeds_the_bead_its_position_can_hold(thickness):
    """Pass count must be derived from area, not from leg.

    Splitting a leg in half leaves each pass carrying 71% of it, not 50% --
    area grows as the square. On a 12 mm fillet that puts an 8.5 mm bead in a
    position rated for 7 mm.
    """
    data = load_procedure("gmaw_carbon_steel")
    seam = make_seam(JointType.TEE, thickness, thickness)
    procedure = generate_procedure(seam)
    limit = data.max_leg_per_pass[procedure.position]
    assert max(p.leg_mm for p in procedure.passes) <= limit + 0.01


def test_thicker_members_need_more_passes():
    counts = [
        generate_procedure(make_seam(JointType.TEE, t, t)).n_passes for t in (4.0, 10.0, 20.0, 35.0)
    ]
    assert counts == sorted(counts)
    assert counts[0] == 1
    assert counts[-1] > counts[0]


def test_heat_input_stays_inside_the_grade_limit():
    """S355 is bought for its toughness; too much heat coarsens the HAZ."""
    material = load_materials()["S355"]
    for thickness in (6.0, 12.0, 20.0):
        procedure = generate_procedure(
            make_seam(JointType.TEE, thickness, thickness),
            ProcedureOptions(base_material="S355"),
        )
        for weld_pass in procedure.passes:
            assert weld_pass.heat_input_kj_mm() <= material.heat_input_max_kj_mm * 1.01


def test_heat_input_matches_the_standard_formula():
    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0))
    weld_pass = procedure.passes[0]
    expected = heat_input_kj_mm(
        weld_pass.current_a, weld_pass.voltage_v, weld_pass.travel_speed_mm_s, 0.8
    )
    assert np.isclose(weld_pass.heat_input_kj_mm(), expected)


# ---------------------------------------------------------------------------
# position sensitivity
# ---------------------------------------------------------------------------


def test_overhead_is_welded_slower_than_flat():
    flat = generate_procedure(make_seam(JointType.TEE, 8.0, 8.0))
    overhead = generate_procedure(make_seam(JointType.TEE, 8.0, 8.0, bisector=(0.0, 0.707, 0.707)))
    assert flat.position is WeldPosition.PB
    assert overhead.position.is_overhead
    assert overhead.passes[0].travel_speed_mm_s < flat.passes[0].travel_speed_mm_s


def test_out_of_position_avoids_spray_transfer():
    """A spray pool is too fluid to hold overhead or running up a plate."""
    for bisector in ((0.0, 0.707, 0.707), (0.0, 1.0, 0.0)):
        procedure = generate_procedure(make_seam(JointType.TEE, 16.0, 16.0, bisector=bisector))
        if procedure.position.is_overhead or procedure.position.is_vertical:
            for weld_pass in procedure.passes:
                assert weld_pass.transfer_mode is not TransferMode.SPRAY


def test_weaving_is_added_out_of_position_but_never_on_the_root():
    procedure = generate_procedure(
        make_seam(JointType.TEE, 16.0, 16.0, bisector=(0.0, 0.707, 0.707))
    )
    assert procedure.n_passes > 1
    # The root must stay narrow to fuse the joint line.
    assert procedure.passes[0].weave.shape is WeaveShape.NONE
    assert any(p.weave.active for p in procedure.passes[1:])


def test_flat_welding_does_not_weave():
    procedure = generate_procedure(make_seam(JointType.TEE, 16.0, 16.0))
    assert not any(p.weave.active for p in procedure.passes)


def test_weaving_can_be_disabled():
    procedure = generate_procedure(
        make_seam(JointType.TEE, 16.0, 16.0, bisector=(0.0, 0.707, 0.707)),
        ProcedureOptions(enable_weaving=False),
    )
    assert not any(p.weave.active for p in procedure.passes)


# ---------------------------------------------------------------------------
# joint-type sensitivity
# ---------------------------------------------------------------------------


def test_butt_passes_do_not_report_a_fillet_leg():
    procedure = generate_procedure(
        make_seam(JointType.BUTT, 12.0, 12.0, gap=2.0, bisector=(0.0, 0.0, -1.0))
    )
    assert all(p.leg_mm == 0.0 and p.throat_mm == 0.0 for p in procedure.passes)


def test_butt_above_the_square_limit_warns_about_preparation():
    procedure = generate_procedure(
        make_seam(JointType.BUTT, 12.0, 12.0, gap=2.0, bisector=(0.0, 0.0, -1.0))
    )
    assert any("edge preparation" in w for w in procedure.warnings)


def test_thin_square_butt_needs_no_preparation_warning():
    procedure = generate_procedure(
        make_seam(JointType.BUTT, 2.0, 2.0, gap=1.0, bisector=(0.0, 0.0, -1.0))
    )
    assert not any("edge preparation" in w for w in procedure.warnings)


def test_lap_biases_the_torch_towards_the_lower_plate():
    lap = generate_procedure(make_seam(JointType.LAP, 8.0, 8.0))
    fillet = generate_procedure(make_seam(JointType.TEE, 8.0, 8.0))
    assert lap.passes[0].work_angle_deg > fillet.passes[0].work_angle_deg


def test_root_gap_slows_travel_and_adds_filler():
    tight = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0, gap=0.0))
    gappy = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0, gap=2.0))
    assert gappy.passes[0].travel_speed_mm_s < tight.passes[0].travel_speed_mm_s
    assert any("gap" in w for w in gappy.warnings)


def test_excessive_gap_is_called_out_as_a_fit_up_problem():
    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0, gap=6.0))
    assert any("backing" in w or "re-fit" in w for w in procedure.warnings)


def test_unknown_thickness_is_assumed_and_flagged():
    procedure = generate_procedure(make_seam(JointType.TEE, 0.0, 0.0))
    assert any("thickness unknown" in w for w in procedure.warnings)
    assert procedure.n_passes >= 1


# ---------------------------------------------------------------------------
# overrides and failure modes
# ---------------------------------------------------------------------------


def test_leg_ratio_override_changes_the_weld_size():
    weak = generate_procedure(
        make_seam(JointType.TEE, 10.0, 10.0), ProcedureOptions(leg_thickness_ratio=0.7)
    )
    full = generate_procedure(
        make_seam(JointType.TEE, 10.0, 10.0), ProcedureOptions(leg_thickness_ratio=1.0)
    )
    assert full.target_leg_mm > weak.target_leg_mm


def test_single_pass_can_be_forced_and_is_honest_about_it():
    procedure = generate_procedure(
        make_seam(JointType.TEE, 25.0, 25.0), ProcedureOptions(force_single_pass=True)
    )
    assert procedure.n_passes == 1
    data = load_procedure("gmaw_carbon_steel")
    # Forced into one pass the bead exceeds what the position holds, and the
    # current or speed will have been clamped -- which must be reported.
    assert procedure.passes[0].leg_mm > data.max_leg_per_pass[procedure.position]


def test_pass_cap_is_reported():
    procedure = generate_procedure(
        make_seam(JointType.BUTT, 60.0, 60.0, gap=3.0, bisector=(0.0, 0.0, -1.0)),
        ProcedureOptions(max_passes=3),
    )
    assert procedure.n_passes == 3
    assert any("capped" in w for w in procedure.warnings)


def test_unknown_material_raises():
    with pytest.raises(ValueError, match="unknown base material"):
        generate_procedure(
            make_seam(JointType.TEE, 8.0, 8.0), ProcedureOptions(base_material="unobtanium")
        )


def test_stainless_uses_its_own_procedure_and_gas():
    procedure = generate_procedure(
        make_seam(JointType.TEE, 8.0, 8.0), ProcedureOptions(base_material="316L")
    )
    assert procedure.process is Process.GMAW
    assert "19 9" in procedure.filler
    assert procedure.passes[0].gas.startswith("M12")
    assert procedure.interpass_max_c <= 150.0


def test_generate_procedures_for_many_seams():
    seams = [make_seam(JointType.TEE, t, t) for t in (6.0, 10.0, 16.0)]
    procedures = generate_procedures(seams)
    assert len(procedures) == 3
    assert set(procedures) == {s.id for s in seams}


# ---------------------------------------------------------------------------
# consumption and timing
# ---------------------------------------------------------------------------


def test_arc_time_and_consumption_scale_with_length():
    seam = make_seam(JointType.TEE, 10.0, 10.0, length=200.0)
    procedure = generate_procedure(seam)
    short_time = procedure.total_arc_time_s(100.0)
    long_time = procedure.total_arc_time_s(200.0)
    assert long_time > short_time
    assert procedure.total_wire_kg(200.0) > procedure.total_wire_kg(100.0)
    # A 10 mm fillet over 200 mm is tens of grams, not kilos.
    assert 0.005 < procedure.total_wire_kg(200.0) < 0.5


def test_procedure_summary_is_serialisable():
    import json

    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0))
    payload = json.loads(json.dumps(procedure.summary()))
    assert payload["iso_4063"] == 135
    assert payload["n_passes"] == procedure.n_passes


# ---------------------------------------------------------------------------
# machine profiles
# ---------------------------------------------------------------------------


def test_signal_scale_round_trip():
    scale = SignalScale("AO1", 0.0, 500.0, 0, 32767)
    raw, clamped = scale.to_raw(250.0)
    assert not clamped
    assert np.isclose(scale.from_raw(raw), 250.0, atol=0.1)


def test_signal_scale_clamps_and_says_so():
    scale = SignalScale("AO1", 0.0, 500.0, 0, 32767)
    raw, clamped = scale.to_raw(900.0)
    assert clamped and raw == 32767
    raw, clamped = scale.to_raw(-50.0)
    assert clamped and raw == 0


def test_signal_scale_with_zero_span():
    scale = SignalScale("AO1", 5.0, 5.0, 10, 20)
    assert scale.to_raw(5.0) == (10, False)
    assert scale.from_raw(15) == 5.0


def test_job_rule_matching():
    rule = JobRule(
        job=7, wire_diameter_mm=1.2, gas="M21-ArC-18", group="carbon_steel", mode=TransferMode.PULSE
    )
    assert rule.matches(1.2, "M21-ArC-18", "carbon_steel", TransferMode.PULSE)
    assert not rule.matches(1.0, "M21-ArC-18", "carbon_steel", TransferMode.PULSE)
    assert not rule.matches(1.2, "C1", "carbon_steel", TransferMode.PULSE)
    assert not rule.matches(1.2, "M21-ArC-18", "carbon_steel", TransferMode.SPRAY)
    # An unconstrained rule matches anything.
    assert JobRule(job=1).matches(0.8, "anything", "any", TransferMode.CMT)


def test_analog_machine_commands_current_and_voltage_directly():
    profile = load_machine("generic_analog")
    assert not profile.is_synergic
    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0))
    setpoints = setpoints_for_procedure(procedure, profile)
    assert len(setpoints) == procedure.n_passes
    for sp in setpoints:
        assert sp.job_number is None
        assert set(sp.analog_words) == {"AO1", "AO2"}
        assert all(0 <= v <= 32767 for v in sp.analog_words.values())


def test_synergic_machine_selects_a_job_per_transfer_mode():
    """A root in short arc and a cap in spray are different characteristics."""
    profile = load_machine("fronius_tps")
    assert profile.is_synergic
    procedure = generate_procedure(make_seam(JointType.TEE, 12.0, 12.0))
    setpoints = setpoints_for_procedure(procedure, profile)
    jobs = {sp.pass_index: sp.job_number for sp in setpoints}
    assert all(j is not None for j in jobs.values())
    modes = {p.index: p.transfer_mode for p in procedure.passes}
    if len(set(modes.values())) > 1:
        assert len(set(jobs.values())) > 1


def test_synergic_machine_sends_wire_feed_not_current():
    profile = load_machine("fronius_tps")
    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0))
    sp = setpoints_for_pass(procedure.passes[0], profile)
    assert "AO_WIRE" in sp.analog_words
    # The current monitor word is read-only and must not be commanded.
    assert "AO_CURRENT_MONITOR" not in sp.analog_words
    assert any("synergic" in n for n in sp.notes)


def test_lincoln_trim_uses_its_multiplier_convention():
    """Lincoln trim is 1.00 nominal over 0.50-1.50, not a signed offset."""
    profile = load_machine("lincoln_powerwave")
    procedure = generate_procedure(make_seam(JointType.TEE, 10.0, 10.0))
    nominal = setpoints_for_pass(procedure.passes[0], profile)
    assert nominal.analog_words["TRIM"] == 1000
    hotter = setpoints_for_pass(procedure.passes[0], profile, arc_length_correction=5.0)
    assert hotter.analog_words["TRIM"] > 1000
    colder = setpoints_for_pass(procedure.passes[0], profile, arc_length_correction=-5.0)
    assert colder.analog_words["TRIM"] < 1000


def test_arc_on_is_not_asserted_by_the_planner():
    """Striking the arc belongs to the execution layer, at the seam start."""
    for key in available_machines():
        profile = load_machine(key)
        procedure = generate_procedure(make_seam(JointType.TEE, 8.0, 8.0))
        sp = setpoints_for_pass(procedure.passes[0], profile)
        arc_signal = profile.digital.get("arc_on")
        if arc_signal:
            assert sp.digital_bits[arc_signal] is False


def test_machine_limits_are_reported_not_silently_applied():
    profile = load_machine("generic_analog")
    procedure = generate_procedure(make_seam(JointType.TEE, 8.0, 8.0))
    weld_pass = procedure.passes[0]
    weld_pass.current_a = 900.0  # beyond the profile's 500 A ceiling
    sp = setpoints_for_pass(weld_pass, profile)
    assert any("exceeds" in n for n in sp.notes)
    assert any("clamped" in n for n in sp.notes)


def test_missing_job_falls_back_with_a_warning():
    profile = load_machine("fronius_tps")
    procedure = generate_procedure(make_seam(JointType.TEE, 8.0, 8.0))
    weld_pass = procedure.passes[0]
    weld_pass.gas = "C1-CO2-100"  # a gas the job table does not cover
    sp = setpoints_for_pass(weld_pass, profile)
    assert sp.job_number == profile.fallback_job
    assert any("fell back" in n for n in sp.notes)


def test_unknown_machine_names_the_alternatives():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_machine("acme_sparkomatic")
