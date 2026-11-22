"""Unit conventions for the whole of WeldForge.

**One rule, enforced everywhere:**

===========================  =================================================
Quantity                     Internal unit
===========================  =================================================
length / position            millimetre (mm)
angle                        radian (rad) in code, degree only in I/O & YAML
time                         second (s)
linear speed                 millimetre per second (mm/s)
wire feed speed              metre per minute (m/min)  -- shop-floor convention
current                      ampere (A)
voltage                      volt (V)
gas flow                     litre per minute (l/min)
heat input                   kilojoule per millimetre (kJ/mm)
mass                         kilogram (kg)
===========================  =================================================

Millimetres are the native unit of every CAD system we read, and of every
welding procedure sheet we write, so the core keeps millimetres and converts
*only at boundaries*:

* ROS 2 / URDF / MoveIt expect **metres** -> convert in ``weldforge.post.ros2``
  and in the ROS nodes, never inside the core.
* Robot postprocessors (RAPID, KRL, URScript, Fanuc TP) each declare their own
  unit in their module.

The previous generation of this software mixed metres and millimetres across
module boundaries; that bug class is why this module exists and why the
converters below are the only sanctioned way to cross a unit boundary.
"""

from __future__ import annotations

import math

# --- length -----------------------------------------------------------------

MM_PER_M = 1000.0


def mm_to_m(value: float) -> float:
    """Millimetres -> metres (ROS/URDF boundary)."""
    return value / MM_PER_M


def m_to_mm(value: float) -> float:
    """Metres -> millimetres (ROS/URDF boundary)."""
    return value * MM_PER_M


def mm_to_inch(value: float) -> float:
    return value / 25.4


def inch_to_mm(value: float) -> float:
    return value * 25.4


# --- angle ------------------------------------------------------------------


def deg(radians: float) -> float:
    """Radians -> degrees (for display, YAML and procedure sheets)."""
    return math.degrees(radians)


def rad(degrees: float) -> float:
    """Degrees -> radians (for everything internal)."""
    return math.radians(degrees)


# --- welding-specific -------------------------------------------------------


def wire_speed_m_min_to_mm_s(value: float) -> float:
    """Wire feed speed m/min -> mm/s."""
    return value * MM_PER_M / 60.0


def wire_speed_mm_s_to_m_min(value: float) -> float:
    return value * 60.0 / MM_PER_M


def heat_input_kj_mm(
    current_a: float,
    voltage_v: float,
    travel_speed_mm_s: float,
    efficiency: float = 0.8,
) -> float:
    """Arc heat input in kJ/mm.

    ``Q = eta * U * I / v``, with *v* in mm/s and the result scaled to kJ.
    Default *efficiency* 0.8 is the usual figure for GMAW (EN 1011-1);
    use 0.6 for GTAW and 1.0 for SAW.
    """
    if travel_speed_mm_s <= 0.0:
        raise ValueError("travel speed must be > 0")
    return efficiency * voltage_v * current_a / travel_speed_mm_s / 1000.0


def deposition_rate_kg_h(
    wire_diameter_mm: float,
    wire_speed_m_min: float,
    density_kg_dm3: float = 7.85,
    efficiency: float = 0.95,
) -> float:
    """Filler deposition rate in kg/h for a solid wire."""
    area_mm2 = math.pi * (wire_diameter_mm / 2.0) ** 2
    volume_mm3_per_min = area_mm2 * wire_speed_m_min * MM_PER_M
    # mm^3 -> dm^3 is 1e-6
    return volume_mm3_per_min * 60.0 * 1e-6 * density_kg_dm3 * efficiency
