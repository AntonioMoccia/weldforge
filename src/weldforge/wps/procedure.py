"""Welding procedure generation: from a recognised seam to numbers on a machine.

The chain of reasoning, in the order the physics actually runs:

1. **Governing thickness** -- the thinner member sets everything. It is read
   from the seam, where :mod:`weldforge.seams` measured it locally.
2. **Required deposit** -- a fillet of leg *z* has a cross-section of
   ``z^2 / 2``, plus convexity. A butt needs the groove volume plus the root
   gap and the cap reinforcement.
3. **Pass count** -- divided by the largest bead the welding position can
   hold. Flat carries 8 mm of leg in one go; vertical-up carries 5.
4. **Travel speed and wire feed together** -- and this is the step most
   parameter tables get wrong by listing them independently. Conservation of
   volume ties them:

   .. code-block:: text

       wire cross-section x wire feed x efficiency  =  deposit area x travel speed

   Pick the travel speed the position allows, and the wire feed follows. The
   wire feed then fixes the **current** by inverting the synergic line, and the
   current fixes the voltage. Nothing is chosen twice, so the numbers cannot
   contradict each other.
5. **Clamp and re-solve** -- if the required current falls outside the wire's
   range, the current is clamped and the travel speed recomputed from the
   feed that is actually achievable. The deposit stays right; the cycle time
   changes, and the operator is told.
6. **Check** -- heat input against the grade's ceiling, preheat from
   thickness, interpass limit, and a warning for anything the geometry says
   needs edge preparation.

Every number the engine cannot justify becomes a warning on the procedure
rather than a silent default.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ..model import (
    JointType,
    Seam,
    TransferMode,
    Weave,
    WeaveShape,
    WeldPass,
    WeldPosition,
    WeldProcedure,
)
from ..units import wire_speed_m_min_to_mm_s
from .tables import MaterialData, ProcedureData, load_materials, load_procedure

logger = logging.getLogger(__name__)

__all__ = [
    "ProcedureOptions",
    "generate_procedure",
    "generate_procedures",
    "fillet_leg_for_thickness",
    "fillet_area_mm2",
    "groove_area_mm2",
]

# Weld beads are convex, not flat triangles: real deposits run 10-20% over the
# nominal section. Under-estimating it leaves an undersized fillet.
CONVEXITY_FACTOR = 1.15
# Cap reinforcement on a butt weld, as a fraction of the groove area.
BUTT_REINFORCEMENT_FACTOR = 0.12


@dataclass
class ProcedureOptions:
    """What the engineer may override; everything else is derived."""

    base_material: str = "S355"
    procedure_name: str | None = None  # default: the material's own procedure
    wire_diameter_mm: float | None = None  # default: chosen from thickness
    target_leg_mm: float | None = None  # default: design rule from thickness
    # Fraction of the governing thickness used as the fillet leg. 0.7 is the
    # usual structural default for a non-full-strength fillet; use 1.0 when the
    # weld must match the parent metal.
    leg_thickness_ratio: float = 0.7
    min_leg_mm: float = 3.0
    max_leg_mm: float = 12.0
    # Included angle assumed for a butt that has no prepared bevel in the CAD.
    butt_included_angle_deg: float = 60.0
    # Butts thinner than this are welded square with no preparation.
    square_butt_max_thickness_mm: float = 3.0
    position_override: WeldPosition | None = None
    transfer_mode_override: TransferMode | None = None
    force_single_pass: bool = False
    max_passes: int = 12
    enable_weaving: bool = True


# ---------------------------------------------------------------------------
# deposit geometry
# ---------------------------------------------------------------------------


def fillet_leg_for_thickness(
    thickness_mm: float,
    ratio: float = 0.7,
    min_leg_mm: float = 3.0,
    max_leg_mm: float = 12.0,
) -> float:
    """Design fillet leg for a member thickness, rounded to 0.5 mm.

    ``z = ratio * t``, bounded. The default ratio of 0.7 is the common
    structural-steel default; a full-strength fillet needs ``ratio = 1.0``.
    Rounded to half a millimetre because that is the resolution a weld gauge
    reads and a procedure sheet states.

    This is a **design decision**, not a measurement: it is surfaced in
    :class:`ProcedureOptions` so an engineer can set it per job, and the chosen
    value is recorded on the procedure.
    """
    if thickness_mm <= 0.0:
        return min_leg_mm
    leg = ratio * thickness_mm
    leg = round(leg * 2.0) / 2.0
    return float(np.clip(leg, min_leg_mm, max_leg_mm))


def fillet_area_mm2(leg_mm: float, convexity: float = CONVEXITY_FACTOR) -> float:
    """Deposited cross-section of an equal-leg fillet, mm^2."""
    return 0.5 * leg_mm * leg_mm * convexity


def groove_area_mm2(
    thickness_mm: float,
    gap_mm: float,
    included_angle_deg: float,
    square_max_thickness_mm: float = 3.0,
) -> tuple[float, bool]:
    """``(area_mm2, needs_preparation)`` for a butt weld.

    A square butt -- no bevel -- only fills the root gap through the thickness,
    and is only valid on thin material. Above that a single-V is assumed, whose
    section is ``g*t + t^2 * tan(alpha/2)``, plus cap reinforcement.
    """
    thickness = max(thickness_mm, 0.0)
    gap = max(gap_mm, 0.0)

    if thickness <= square_max_thickness_mm:
        area = max(gap, 1.0) * thickness
        return area * (1.0 + BUTT_REINFORCEMENT_FACTOR), False

    half_angle = math.radians(included_angle_deg / 2.0)
    groove = gap * thickness + thickness * thickness * math.tan(half_angle)
    return groove * (1.0 + BUTT_REINFORCEMENT_FACTOR), True


# ---------------------------------------------------------------------------
# pass layout
# ---------------------------------------------------------------------------


def _pass_offsets(n_passes: int, leg_mm: float, joint_type: JointType) -> list[tuple[float, float]]:
    """``(across, normal)`` offsets per pass, mm, relative to the joint root.

    A multi-pass fillet is built root first, then in layers that widen as they
    rise -- the standard 1-2-3 / 1-2-3-4-5 sequence a welder uses. Each layer
    sits one bead height up and its passes are spread across the face so the
    toes tie in. A butt is filled in flat layers with no lateral spread until
    the cap.
    """
    if n_passes <= 1:
        return [(0.0, 0.0)]

    offsets: list[tuple[float, float]] = []
    if joint_type is JointType.BUTT:
        # Stack layers through the thickness.
        step = leg_mm / max(1, n_passes)
        for i in range(n_passes):
            offsets.append((0.0, i * step))
        return offsets

    # Triangular build-up: layer k holds k+1 passes until the count runs out.
    bead = leg_mm / max(1.0, math.sqrt(n_passes))
    placed = 0
    layer = 0
    while placed < n_passes:
        in_layer = min(layer + 1, n_passes - placed)
        spread = bead * layer / 2.0
        for j in range(in_layer):
            across = -spread + (2.0 * spread * j / max(1, in_layer - 1)) if in_layer > 1 else 0.0
            offsets.append((float(across), float(layer * bead * 0.8)))
            placed += 1
        layer += 1
    return offsets[:n_passes]


# A root pass deposits less than a fill pass: it is run small and fast to fuse
# the joint line without burning through, and without a big pool that could
# sag out of the groove.
ROOT_PASS_AREA_FRACTION = 0.6


def _distribute_area(total_area_mm2: float, n_passes: int) -> list[float]:
    """Split the deposit between passes, root first.

    The root gets ``ROOT_PASS_AREA_FRACTION`` of an average pass and the
    remainder is shared equally by the fills and the cap, so the totals still
    add up to the required section.
    """
    if n_passes <= 1:
        return [total_area_mm2]
    average = total_area_mm2 / n_passes
    root = average * ROOT_PASS_AREA_FRACTION
    rest = (total_area_mm2 - root) / (n_passes - 1)
    return [root] + [rest] * (n_passes - 1)


def _pass_name(index: int, n_passes: int) -> str:
    if n_passes == 1:
        return "single"
    if index == 0:
        return "root"
    if index == n_passes - 1:
        return "cap"
    return f"fill {index}"


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


def generate_procedure(
    seam: Seam,
    options: ProcedureOptions | None = None,
    procedure_data: ProcedureData | None = None,
    material_data: MaterialData | None = None,
) -> WeldProcedure:
    """Build a welding procedure for one seam."""
    opts = options or ProcedureOptions()
    materials = load_materials()
    material = material_data or materials.get(opts.base_material)
    if material is None:
        raise ValueError(
            f"unknown base material {opts.base_material!r}; known: {sorted(materials)}"
        )
    data = procedure_data or load_procedure(opts.procedure_name or material.procedure)

    warnings: list[str] = []
    position = opts.position_override or seam.dominant_position
    thickness = seam.min_thickness_mm
    if thickness <= 0.0:
        thickness = 6.0
        warnings.append("member thickness unknown: assumed 6 mm")

    # --- deposit geometry ---------------------------------------------------

    if seam.joint_type is JointType.BUTT:
        target_leg = opts.target_leg_mm or thickness
        total_area, needs_prep = groove_area_mm2(
            thickness,
            seam.gap_mm,
            opts.butt_included_angle_deg,
            opts.square_butt_max_thickness_mm,
        )
        if needs_prep:
            warnings.append(
                f"butt on {thickness:.1f} mm needs edge preparation: assumed single-V "
                f"{opts.butt_included_angle_deg:.0f} deg included angle"
            )
    else:
        target_leg = opts.target_leg_mm or fillet_leg_for_thickness(
            thickness, opts.leg_thickness_ratio, opts.min_leg_mm, opts.max_leg_mm
        )
        total_area = fillet_area_mm2(target_leg)
        if seam.gap_mm > 0.5:
            # A gap under a fillet has to be filled before the leg is built, and
            # the leg must grow to keep the throat (EN 1993-1-8).
            extra = seam.gap_mm * thickness
            total_area += extra
            warnings.append(f"root gap {seam.gap_mm:.1f} mm: leg increased to maintain throat")

    if seam.gap_mm > data.max_gap_mm:
        warnings.append(
            f"root gap {seam.gap_mm:.1f} mm exceeds {data.max_gap_mm:.1f} mm: "
            "backing or a re-fit is needed before welding"
        )

    # --- pass count ---------------------------------------------------------

    max_leg = data.max_leg_per_pass[position]
    # Pass count follows from **area**, for every joint type. Dividing the leg
    # instead is dimensionally wrong: area grows with the square of the leg, so
    # a 12 mm fillet split in two still asks each pass to carry an 8.5 mm bead
    # when the position can only hold 7 mm. Area division gives three passes,
    # each inside the limit.
    max_area_per_pass = fillet_area_mm2(max_leg)
    if opts.force_single_pass:
        n_passes = 1
    else:
        # The average pass area is not the binding constraint: the root is run
        # deliberately small, so the fills and the cap each carry more than the
        # average. Add passes until the *largest* one fits the bead the
        # position can hold.
        n_passes = max(1, math.ceil(total_area / max(max_area_per_pass, 1e-6)))
        while (
            n_passes < opts.max_passes
            and max(_distribute_area(total_area, n_passes)) > max_area_per_pass
        ):
            n_passes += 1
    if n_passes > opts.max_passes:
        warnings.append(
            f"{n_passes} passes required but capped at {opts.max_passes}: "
            "the joint may be under-filled"
        )
        n_passes = opts.max_passes

    # --- consumable ---------------------------------------------------------

    wire = (
        data.wire(opts.wire_diameter_mm) if opts.wire_diameter_mm else data.default_wire(thickness)
    )

    target_speed = data.target_travel_speed[position]
    if seam.gap_mm > 0.0:
        target_speed *= max(
            0.4, 1.0 + data.gap_travel_factor_per_mm * min(seam.gap_mm, data.max_gap_mm)
        )

    offsets = _pass_offsets(n_passes, target_leg, seam.joint_type)
    work_angle, travel_angle = data.torch_angle(seam.joint_type.value)
    pass_areas = _distribute_area(total_area, n_passes)

    passes: list[WeldPass] = []
    for index in range(n_passes):
        across, normal = offsets[index]
        area = pass_areas[index]
        weld_pass = _solve_pass(
            index=index,
            n_passes=n_passes,
            area_mm2=area,
            target_speed_mm_s=target_speed,
            data=data,
            wire=wire,
            position=position,
            joint_type=seam.joint_type,
            # Leg reported per pass is the one its own area actually makes, so
            # the sheet stays internally consistent.
            leg_mm=math.sqrt(2.0 * area / CONVEXITY_FACTOR),
            across_mm=across,
            normal_mm=normal,
            work_angle_deg=work_angle,
            travel_angle_deg=travel_angle,
            options=opts,
            warnings=warnings,
        )
        passes.append(weld_pass)

    # --- metallurgical checks ----------------------------------------------

    combined_thickness = seam.thickness_a_mm + seam.thickness_b_mm
    preheat = material.preheat_for(combined_thickness if combined_thickness > 0 else thickness)

    for weld_pass in passes:
        heat = weld_pass.heat_input_kj_mm(data.thermal_efficiency)
        if heat > material.heat_input_max_kj_mm:
            warnings.append(
                f"pass {weld_pass.index + 1}: heat input {heat:.2f} kJ/mm exceeds the "
                f"{material.heat_input_max_kj_mm:.2f} kJ/mm limit for {material.name}; "
                "raise travel speed or split the pass"
            )

    procedure = WeldProcedure(
        seam_id=seam.id,
        process=data.process,
        base_material=material.name,
        filler=data.filler,
        position=position,
        joint_type=seam.joint_type,
        thickness_mm=thickness,
        target_leg_mm=target_leg,
        passes=passes,
        preheat_c=preheat,
        interpass_max_c=material.interpass_max_c,
        heat_input_max_kj_mm=material.heat_input_max_kj_mm,
        warnings=warnings,
    )

    logger.debug(
        "procedure for %s: %s %.1f mm, %d pass(es), leg %.1f mm, %s",
        seam.id,
        seam.joint_type.value,
        thickness,
        n_passes,
        target_leg,
        position.value,
    )
    return procedure


def _solve_pass(
    *,
    index: int,
    n_passes: int,
    area_mm2: float,
    target_speed_mm_s: float,
    data: ProcedureData,
    wire,
    position: WeldPosition,
    joint_type: JointType,
    leg_mm: float,
    across_mm: float,
    normal_mm: float,
    work_angle_deg: float,
    travel_angle_deg: float,
    options: ProcedureOptions,
    warnings: list[str],
) -> WeldPass:
    """Solve one pass by conservation of deposited volume.

    ``wire_area * wire_feed * efficiency = deposit_area * travel_speed``

    Travel speed is proposed by the position, the wire feed follows, the
    current follows from the synergic line, and the voltage from the current.
    If the current lands outside the wire's range it is clamped and the travel
    speed re-solved from the feed that is actually available -- the deposit
    must stay correct, so it is the cycle time that gives.
    """
    speed_lo, speed_hi = data.travel_speed_limits
    speed = float(np.clip(target_speed_mm_s, speed_lo, speed_hi))

    # Volume the deposit needs, mm^3/s.
    deposit_rate = area_mm2 * speed
    # Wire feed to supply it.
    wire_feed_mm_s = deposit_rate / max(wire.cross_section_mm2 * data.deposition_efficiency, 1e-9)
    wire_feed_m_min = wire_feed_mm_s * 60.0 / 1000.0

    current = wire.current_for_wire_speed(wire_feed_m_min)
    current, clamped = wire.clamp_current(current)

    if clamped:
        # Re-solve: the achievable feed sets the speed.
        achievable_feed = wire.wire_speed_for_current(current)
        achievable_rate = (
            wire_speed_m_min_to_mm_s(achievable_feed)
            * wire.cross_section_mm2
            * data.deposition_efficiency
        )
        speed = float(np.clip(achievable_rate / max(area_mm2, 1e-9), speed_lo, speed_hi))
        wire_feed_m_min = achievable_feed
        warnings.append(
            f"pass {index + 1}: current clamped to {current:.0f} A for the "
            f"{wire.diameter_mm:.1f} mm wire; travel speed re-solved to {speed:.1f} mm/s"
        )

    voltage = wire.voltage_for_current(current)

    # --- transfer mode ------------------------------------------------------
    if options.transfer_mode_override is not None:
        mode = options.transfer_mode_override
    elif position.is_vertical or position.is_overhead:
        # The pool must be stiff enough to hold: spray will not.
        candidates = [m for m in data.out_of_position_modes if wire.supports(m)]
        mode = candidates[0] if candidates else TransferMode.SHORT_ARC
    elif current <= data.short_arc_max_current_a:
        mode = TransferMode.SHORT_ARC
    else:
        mode = TransferMode.SPRAY if wire.supports(TransferMode.SPRAY) else TransferMode.PULSE

    stickout = data.stickout.get(mode, 15.0)

    # --- weaving ------------------------------------------------------------
    weave = Weave()
    weave_spec = data.weave_for(position)
    # Never weave the root: it must be kept narrow to fuse the joint line.
    is_root = index == 0 and n_passes > 1
    if options.enable_weaving and weave_spec and not is_root:
        try:
            weave = Weave(
                shape=WeaveShape(weave_spec.get("shape", "none")),
                amplitude_mm=float(weave_spec.get("amplitude_mm", 0.0)),
                frequency_hz=float(weave_spec.get("frequency_hz", 0.0)),
                dwell_ms=float(weave_spec.get("dwell_ms", 0.0)),
            )
        except ValueError:
            logger.warning("unknown weave shape %r", weave_spec.get("shape"))

    # --- work angle per pass ------------------------------------------------
    # In a multi-pass fillet the torch tilts towards whichever leg the pass is
    # building, so the bead ties into the sidewall rather than sagging.
    work = work_angle_deg
    if n_passes > 1 and joint_type is not JointType.BUTT and abs(across_mm) > 1e-6:
        work += math.copysign(min(20.0, abs(across_mm) * 4.0), across_mm)

    # Leg and throat describe a fillet. On a butt the deposit is a groove
    # layer, not a triangle, so they are left at zero rather than reporting a
    # number that means nothing on the procedure sheet.
    is_fillet_like = joint_type is not JointType.BUTT

    return WeldPass(
        index=index,
        name=_pass_name(index, n_passes),
        leg_mm=round(leg_mm, 2) if is_fillet_like else 0.0,
        throat_mm=round(leg_mm * 0.707, 2) if is_fillet_like else 0.0,
        offset_across_mm=round(across_mm, 2),
        offset_normal_mm=round(normal_mm, 2),
        process=data.process,
        transfer_mode=mode,
        current_a=round(current, 1),
        voltage_v=round(voltage, 1),
        wire_speed_m_min=round(wire_feed_m_min, 2),
        wire_diameter_mm=wire.diameter_mm,
        gas=data.gas,
        gas_flow_l_min=data.gas_flow_l_min,
        travel_speed_mm_s=round(speed, 2),
        work_angle_deg=round(work, 1),
        travel_angle_deg=travel_angle_deg,
        stickout_mm=stickout,
        weave=weave,
    )


def generate_procedures(
    seams: list[Seam],
    options: ProcedureOptions | None = None,
) -> dict[str, WeldProcedure]:
    """Procedures for a whole job, keyed by seam id.

    The procedure data and material tables are loaded once and shared, which
    matters on a weldment with a couple of hundred seams.
    """
    opts = options or ProcedureOptions()
    materials = load_materials()
    material = materials.get(opts.base_material)
    if material is None:
        raise ValueError(
            f"unknown base material {opts.base_material!r}; known: {sorted(materials)}"
        )
    data = load_procedure(opts.procedure_name or material.procedure)

    out: dict[str, WeldProcedure] = {}
    for seam in seams:
        out[seam.id] = generate_procedure(seam, opts, data, material)
    logger.info("generated %d welding procedures", len(out))
    return out
